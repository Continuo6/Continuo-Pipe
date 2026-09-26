#!/usr/bin/env python3
"""Cut a shard into annotated short segments and write a manifest.

A shard pairs each audio file with a JSON sidecar. Three kinds of container
appear — ``short`` (one already-isolated utterance), ``long`` (~30 s), and ``dialogue``
(a full multi-speaker recording) — and **all three carry the same ``short`` list**:
the utterance-level annotation, each entry with a transcript, a language, a speaker,
a DNSMOS score, and exact sample offsets into its container.

That list is the unit the annotation pipeline wants, so this script decodes each
container once and slices every short entry out of it by sample index. Containers with
an empty ``short`` list have nothing annotated in them and are skipped.

Segment ids are unique across the whole shard, so a segment cut from a ``dialogue``
container is never also cut from the ``long`` container overlapping it — no dedup pass
is needed, but the script asserts it rather than assuming it.

Output: one 16-bit FLAC per segment at the container's native sample rate, plus a
manifest ready for `continuo-annotate`. 16-bit is the speech-corpus norm and sits far above
the effective precision of the lossy AAC it was decoded from, so the quantisation is
not where any information is lost. Because every segment ships its transcript and
language, the annotate pass skips ASR entirely.

    python tools/prepare_corpus.py --shard-dir corpus/shard-0001 --out-dir runs/short

``--shard-dir`` may hold the extracted ``.m4a`` + ``.json`` pairs, or be split into
``containers/`` and ``meta/`` subdirectories.

Resumable: a segment whose FLAC already exists is not re-cut.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_EXTS = (".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus")


def decode(path: Path, sample_rate: int) -> np.ndarray:
    """Decode a container to mono float32 at its native rate via ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
           "-ac", "1", "-ar", str(sample_rate), "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path.name}: "
                           f"{proc.stderr.decode('utf-8', 'replace').strip()[:200]}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def find_audio(meta_path: Path, container_dir: Path) -> Path | None:
    for ext in AUDIO_EXTS:
        candidate = container_dir / (meta_path.stem + ext)
        if candidate.is_file():
            return candidate
    return None


def cut_container(args) -> dict:
    meta_path, container_dir, out_dir, min_seconds = args
    meta_path, container_dir, out_dir = Path(meta_path), Path(container_dir), Path(out_dir)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    shorts = meta.get("short") or []
    result = {"container": meta_path.stem, "rows": [], "skipped": [], "error": None}
    if not shorts:
        return result

    sample_rate = int(meta.get("sample_rate") or 44100)
    audio_path = find_audio(meta_path, container_dir)
    if audio_path is None:
        result["error"] = f"{meta_path.stem}: no audio file beside the sidecar"
        return result

    # Decode once, then slice all segments from the container.
    wav = None
    for entry in shorts:
        seg_id = entry["id"]
        dest = out_dir / meta.get("recording_id", "unknown") / f"{seg_id}.flac"
        row = {
            "id": seg_id,
            "wav_path": str(dest),
            "txt": entry.get("text") or "",
            "lang": entry.get("language"),
            "speaker": entry.get("speaker"),
            "dnsmos": entry.get("dnsmos"),
            "duration": entry.get("duration"),
            "source_id": meta.get("id"),
            "source_type": meta.get("type"),
            "source_file": audio_path.name,
            "rel_start": entry.get("rel_start"),
            "rel_end": entry.get("rel_end"),
        }
        if dest.is_file():                       # resume: already cut
            result["rows"].append(row)
            continue

        if wav is None:
            try:
                wav = decode(audio_path, sample_rate)
            except Exception as e:
                result["error"] = str(e)
                return result

        start = int(entry.get("rel_start_samples") or round((entry.get("rel_start") or 0) * sample_rate))
        end = int(entry.get("rel_end_samples") or round((entry.get("rel_end") or 0) * sample_rate))
        # the sidecar's frame count can exceed what the decoder yields by a frame or
        # two at the tail; clamp rather than truncate the whole segment
        start = max(0, min(start, len(wav)))
        end = max(start, min(end, len(wav)))
        chunk = wav[start:end]
        if len(chunk) < min_seconds * sample_rate:
            result["skipped"].append({"id": seg_id, "samples": int(len(chunk)),
                                      "reason": "shorter than --min-seconds after clamping"})
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".flac.part")      # never leave a half-written segment
        sf.write(tmp, chunk, sample_rate, format="FLAC", subtype="PCM_16")
        tmp.replace(dest)
        result["rows"].append(row)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="prepare_corpus",
        description="Cut annotated short segments out of a shard into a manifest.")
    p.add_argument("--shard-dir", required=True,
                   help="directory of .m4a/.json pairs, or one holding containers/ and meta/")
    p.add_argument("--out-dir", required=True, help="destination for segments/ and the manifest")
    p.add_argument("--manifest", default="", help="manifest path (default: <out-dir>/manifest.jsonl)")
    p.add_argument("--min-seconds", type=float, default=0.5,
                   help="drop segments shorter than this; below ~0.5 s neither loudness "
                        "nor speaking rate is measurable (default 0.5)")
    p.add_argument("--workers", type=int, default=8, help="containers decoded in parallel")
    p.add_argument("--limit", type=int, default=0, help="cap containers, for a smoke test")
    args = p.parse_args(argv)

    shard = Path(args.shard_dir).expanduser().resolve()
    meta_dir = shard / "meta" if (shard / "meta").is_dir() else shard
    container_dir = shard / "containers" if (shard / "containers").is_dir() else shard
    out_dir = Path(args.out_dir).expanduser().resolve()
    segments_dir = out_dir / "segments"
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.jsonl"
    segments_dir.mkdir(parents=True, exist_ok=True)

    metas = sorted(meta_dir.glob("*.json"))
    if args.limit:
        metas = metas[:args.limit]
    if not metas:
        print(f"no *.json sidecars under {meta_dir}", file=sys.stderr)
        return 2
    print(f"{len(metas)} containers in {meta_dir}", flush=True)

    jobs = [(str(m), str(container_dir), str(segments_dir), args.min_seconds) for m in metas]
    rows, skipped, errors, empty = [], [], [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(cut_container, j): j[0] for j in jobs}
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["error"]:
                errors.append(result["error"])
            if not result["rows"] and not result["skipped"] and not result["error"]:
                empty += 1
            rows.extend(result["rows"])
            skipped.extend(result["skipped"])
            if n % 100 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} containers, {len(rows)} segments", flush=True)

    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        print(f"[warn] {len(ids) - len(set(ids))} duplicate segment id(s); manifest "
              "would double-count them", file=sys.stderr)
    rows.sort(key=lambda r: r["id"])

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_hours = sum(r["duration"] or 0 for r in rows) / 3600
    print(f"\n{len(rows)} segments ({total_hours:.2f} h) -> {manifest_path}")
    print(f"segments -> {segments_dir}")
    if empty:
        print(f"{empty} container(s) carried no short annotation and were skipped")
    if skipped:
        print(f"{len(skipped)} segment(s) dropped below --min-seconds {args.min_seconds}")
    for err in errors[:10]:
        print(f"[error] {err}", file=sys.stderr)
    if len(errors) > 10:
        print(f"[error] ... and {len(errors) - 10} more", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
