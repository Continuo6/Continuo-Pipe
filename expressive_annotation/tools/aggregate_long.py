#!/usr/bin/env python3
"""Fold a long recording's segment annotations back into one record per file.

``continuo-annotate`` measured every segment on its own; this rebuilds the container from
them. The interesting half is in :mod:`continuo_expressive.aggregate` — speaker-
intrinsic attributes collapse to one value, context-varying ones become a timeline of
spans. This script is the plumbing: group by ``parent_id``, attach the raw emotion
posteriors so each span can report what its clips decided, and write the result.

    python tools/aggregate_long.py \
        --annot runs/long/seg_annot.jsonl \
        --emotion-jsonl runs/long/emotion_raw.jsonl \
        --out runs/long/file_annot.jsonl \
        --manifest runs/long/manifest.jsonl \
        --caption-manifest runs/long/caption_manifest.jsonl

**The caption window.** ``continuo-caption`` takes at most 30 s of audio, so a four-minute
container needs an excerpt. Picking the first 30 s is what the unfixed pipeline
effectively did and it is not representative of anything. Instead the window is chosen
inside the *dominant emotion span* — the stretch whose label the flat ``emotion`` field
reports — so the audio the captioner hears matches the tags it is told to describe.
Within that span the longest run of segments fitting in 30 s wins, and ties go to the
run whose loudness sits closest to the file's own.

The excerpt is rebuilt from the cut segment files with the original inter-segment
pauses restored as silence, so its rhythm matches the source. Those pauses are digital
silence rather than the recording's own room tone; nothing downstream measures them
(the tags are already decided), but it is a splice and worth knowing about.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from continuo_expressive.aggregate import (DEFAULT_MAX_GAP, DEFAULT_MIN_SPAN_SECONDS,
                                            DEFAULT_SPEED_DELTA, aggregate, combine_lufs,
                                            seconds)
from continuo_expressive.audio import load_wav
from continuo_expressive.config import TARGET_SR
from continuo_expressive.jsonl import (JsonlWriter, ManifestError, index_by_id,
                                        read_jsonl, safe_name)

#: the captioner's own limit; see continuo_expressive/cli/caption.py
CAPTION_SECONDS = 30.0


def pick_window(rows: list[dict], span: dict | None,
                target_lufs: float | None) -> list[dict]:
    """Longest run of segments inside ``span`` that fits in ``CAPTION_SECONDS``.

    Ties — and they are common, because every run of the same few segments has the
    same total — go to the window whose combined loudness is nearest the file's, so
    the excerpt is not systematically the loudest or quietest part of the recording.
    """
    if span and span.get("start") is not None:
        lo, hi = float(span["start"]), float(span["end"])
        pool = [r for r in rows
                if lo - 1e-6 <= (r.get("rel_start") or 0.0) and (r.get("rel_end") or 0.0) <= hi + 1e-6]
    else:
        pool = list(rows)
    pool = sorted(pool, key=lambda r: r.get("rel_start") or 0.0)
    if not pool:
        return []

    best, best_key = [], None
    for i in range(len(pool)):
        j = i
        while j + 1 < len(pool):
            width = (pool[j + 1].get("rel_end") or 0.0) - (pool[i].get("rel_start") or 0.0)
            if width > CAPTION_SECONDS:
                break
            j += 1
        window = pool[i:j + 1]
        speech = sum(seconds(r) for r in window)
        lufs = combine_lufs(window)
        distance = abs(lufs - target_lufs) if (lufs is not None and target_lufs is not None) else 0.0
        key = (speech, -distance)
        if best_key is None or key > best_key:
            best, best_key = window, key
    return best


def write_window(window: list[dict], wav_paths: dict[str, str], dest: Path) -> float | None:
    """Splice a window's segments back together, pauses included. -> seconds written."""
    pieces: list[np.ndarray] = []
    previous_end = None
    for row in window:
        path = wav_paths.get(row["id"])
        if not path or not Path(path).is_file():
            continue
        try:
            wav = load_wav(path)
        except Exception as e:
            print(f"[warn] {row['id']}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        start = row.get("rel_start")
        if previous_end is not None and isinstance(start, (int, float)):
            gap = float(start) - previous_end
            if gap > 0:
                pieces.append(np.zeros(int(gap * TARGET_SR), dtype=np.float32))
        pieces.append(wav)
        end = row.get("rel_end")
        previous_end = float(end) if isinstance(end, (int, float)) else None
    if not pieces:
        return None
    audio = np.concatenate(pieces)[: int(CAPTION_SECONDS * TARGET_SR)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".flac.part")
    sf.write(tmp, audio, TARGET_SR, format="FLAC", subtype="PCM_16")
    tmp.replace(dest)
    return len(audio) / TARGET_SR


def dominant_span(record: dict) -> dict | None:
    """The emotion span whose label the flat ``emotion`` field reports."""
    span_list = record.get("emotion_spans") or []
    label = record.get("emotion")
    candidates = [s for s in span_list if s.get("emotion") == label] if label else span_list
    if not candidates:
        candidates = span_list
    return max(candidates, key=lambda s: s.get("seconds") or 0.0) if candidates else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aggregate_long",
        description="Fold segment annotations of long recordings into one record per file.")
    p.add_argument("--annot", required=True,
                   help="segment-level annotation JSONL from continuo-annotate, run with "
                        "--carry parent_id,parent_duration,duration,lang,rel_start,rel_end")
    p.add_argument("--emotion-jsonl", default="",
                   help="optional external emotion predictions. Supplied here rather than to "
                        "continuo-annotate so each span can report which label its clips "
                        "earned and over how much of it")
    p.add_argument("--out", required=True, help="file-level annotation JSONL")
    p.add_argument("--manifest", default="",
                   help="segment manifest, joined by id for wav_path. Required only "
                        "for --caption-manifest")
    p.add_argument("--caption-manifest", default="",
                   help="also write a manifest of representative <=30 s excerpts, "
                        "ready for continuo-caption")
    p.add_argument("--caption-dir", default="",
                   help="where the excerpts are written (default: <out dir>/caption_windows)")
    p.add_argument("--audio-dir", default="",
                   help="a directory of FULL recordings, one <parent_id>.flac each, as "
                        "written by prepare_long.py --container-dir. The caption "
                        "manifest then points at whole recordings instead of 30 s "
                        "excerpts — which is what you want once captioning runs on "
                        "text (continuo-caption --no-audio) and the audio is the deliverable "
                        "rather than the model's input.")
    p.add_argument("--emotion-tau", type=float,
                   help="required with --emotion-jsonl; confidence gate in [0, 1]")
    p.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP,
                   help=f"silence wider than this ends a span (default {DEFAULT_MAX_GAP})")
    p.add_argument("--min-span-seconds", type=float, default=DEFAULT_MIN_SPAN_SECONDS,
                   help="runs shorter than this are treated as label noise and absorbed "
                        f"into a neighbour (default {DEFAULT_MIN_SPAN_SECONDS})")
    p.add_argument("--speed-delta", type=float, default=DEFAULT_SPEED_DELTA,
                   help="relative change in characters-per-second that counts as a "
                        "change of pace. Speed spans are cut on how far the rate moves, "
                        "not on which bucket it lands in, so two stretches both called "
                        f"'measured' can still be two spans (default {DEFAULT_SPEED_DELTA})")
    p.add_argument("--min-segments", type=int, default=1,
                   help="skip containers with fewer segments than this")
    p.add_argument("--group-by", default="parent_id",
                   help="comma-separated row fields that define one record. The default "
                        "folds a container into one record; `parent_id,speaker` folds a "
                        "dialogue into one record per speaker, which is the only honest "
                        "unit for gender, age and accent when a file holds several voices. "
                        "A composite record's id is the fields joined with '_', and it "
                        "carries each field by name")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.emotion_jsonl and (args.emotion_tau is None or not 0 <= args.emotion_tau <= 1):
        build_parser().error("--emotion-jsonl requires --emotion-tau in [0, 1]")

    keys = [k.strip() for k in args.group_by.split(",") if k.strip()]
    if not keys or "parent_id" not in keys:
        build_parser().error("--group-by must include parent_id")

    def group_key(row: dict) -> tuple | None:
        vals = tuple(row.get(k) for k in keys)
        return None if any(v is None or v == "" for v in vals) else vals

    groups: dict[tuple, list[dict]] = defaultdict(list)
    orphans = 0
    # A segment must appear once: every count below (n_segments, speech_seconds, the
    # span timeline) is a sum over these rows, so a repeat inflates them silently. The
    # way it happens is a resumed run with a different worker count — `--shard i/n`
    # claims rows by `index % n`, each worker resumes from its OWN output file, so
    # going 4 cards -> 5 re-annotates rows the old split had already done into a
    # different file, and `cat work/annot*.jsonl` then holds both.
    seen: set = set()
    repeats = 0
    for row in read_jsonl(args.annot, required=("id",)):
        if row["id"] in seen:
            repeats += 1
            continue
        seen.add(row["id"])
        key = group_key(row)
        if key is None:
            orphans += 1
            continue
        groups[key].append(row)
    if repeats:
        print(f"[warn] {repeats} repeated segment id(s) in {args.annot} were dropped "
              "(first kept). This is what a resumed run with a changed worker count "
              "leaves behind — delete the run's work/annot*.jsonl and re-annotate if "
              "you want the shards clean.", file=sys.stderr)
    if orphans:
        print(f"[warn] {orphans} row(s) missing one of {keys} were skipped; re-run "
              f"continuo-annotate with --carry {','.join(keys)},...", file=sys.stderr)
    if not groups:
        print("nothing to aggregate", file=sys.stderr)
        return 2
    if not any(isinstance(r.get("parent_duration"), (int, float))
               for rows in groups.values() for r in rows):
        print("[warn] no parent_duration on any row: file_seconds and speech_ratio will "
              "be null. Add parent_duration to continuo-annotate's --carry list.",
              file=sys.stderr)

    if args.emotion_jsonl:
        raw = index_by_id(args.emotion_jsonl)
        attached = 0
        for rows in groups.values():
            for row in rows:
                entry = raw.get(row["id"])
                if entry and entry.get("scores"):
                    row["emotion_scores"] = entry["scores"]
                    attached += 1
        print(f"  emotion <- {args.emotion_jsonl} ({attached} segment(s), "
              f"tau={args.emotion_tau} per clip, survivors vote within each span)",
              file=sys.stderr)

    wav_paths: dict[str, str] = {}
    transcripts: dict[tuple, list[tuple[float, str, str | None]]] = defaultdict(list)
    if args.caption_manifest and not args.manifest and not args.audio_dir:
        build_parser().error("--caption-manifest needs --manifest for wav_path, "
                             "or --audio-dir for whole recordings")
    if args.manifest:
        # the transcript is what captions a long recording: continuo-caption --no-audio has
        # nothing else to work from, and an excerpt would have the model describe
        # minutes it never heard
        for row in read_jsonl(args.manifest, required=("id",)):
            if row.get("wav_path"):
                wav_paths[row["id"]] = row["wav_path"]
            text = (row.get("txt") or "").strip()
            key = group_key(row)
            if text and key is not None:
                transcripts[key].append((row.get("rel_start") or 0.0, text, row.get("turn")))

    out_path = Path(args.out)
    caption_dir = Path(args.caption_dir) if args.caption_dir else out_path.parent / "caption_windows"

    audio_dir = Path(args.audio_dir).expanduser().resolve() if args.audio_dir else None
    written = skipped = untagged = 0
    caption_rows: list[dict] = []
    missing_audio: list[str] = []

    def _caption(parent: str, rows: list[dict], record: dict) -> None:
        if not args.caption_manifest:
            return
        try:
            safe_name(parent, "parent_id")     # it becomes a filename below
        except ManifestError as e:
            print(f"[warn] {e}; no audio written", file=sys.stderr)
            return
        if audio_dir is not None:
            full = audio_dir / f"{parent}.flac"
            if full.is_file():
                caption_rows.append({
                    "id": parent, "wav_path": str(full),
                    "audio_seconds": record.get("file_seconds"),
                    "audio": "full recording",
                })
            else:
                missing_audio.append(parent)
        else:
            window = pick_window(rows, dominant_span(record), record.get("volume_lufs"))
            dest = caption_dir / f"{parent}.flac"
            length = write_window(window, wav_paths, dest) if window else None
            if length:
                caption_rows.append({
                    "id": parent, "wav_path": str(dest),
                    "window_seconds": round(length, 2),
                    "window_start": window[0].get("rel_start"),
                    "window_end": window[-1].get("rel_end"),
                    "window_n_seg": len(window),
                })

    nested = keys != ["parent_id"]
    # A composite key means several voices in one file. Each is folded on its own —
    # gender, age and accent are properties of a voice — but the *record* is still the
    # dialogue: one row, its speakers nested under their turn tag (S1, S2, ...), and the
    # transcript re-marked with those tags in time order, which is the shape the corpus
    # sidecar itself uses for a dialogue's text.
    per_parent: dict[str, dict] = defaultdict(lambda: {"speakers": {}, "rows": [],
                                                       "words": []})
    with JsonlWriter(args.out) as sink:
        for key, rows in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
            if len(rows) < args.min_segments:
                skipped += 1
                continue
            record = aggregate(rows, tau=args.emotion_tau, max_gap=args.max_gap,
                               min_span_seconds=args.min_span_seconds,
                               speed_delta=args.speed_delta)
            if not record:
                skipped += 1
                continue
            words = transcripts.get(key) or []
            if words:
                record["txt"] = " ".join(t for _, t, _ in sorted(words))
            parent = rows[0]["parent_id"]
            if not nested:
                record = {"id": parent, **record}
                sink.write(record)
                written += 1
                _caption(parent, rows, record)
                continue
            tag = next((r.get("turn") for r in rows if r.get("turn")), None)
            if tag is None:
                # a voice with no turn tag has no name to nest under; it is a stray (a
                # sidecar short entry outside the turns, one segment, not a speaker) and
                # went out as `('<id>_spk0',)` once. Leave it out and say so.
                untagged += 1
                continue
            spk = per_parent[parent]
            spk["speakers"][tag] = {**dict(zip(keys[1:], key[1:])), **record}
            spk["rows"].extend(rows)
            spk["words"].extend(words)

        for parent, acc in sorted(per_parent.items()):
            rows = acc["rows"]
            speakers = dict(sorted(acc["speakers"].items()))
            speech = sum(v.get("speech_seconds") or 0.0 for v in speakers.values())
            file_seconds = next((float(r["parent_duration"]) for r in rows
                                 if isinstance(r.get("parent_duration"), (int, float))), None)
            # the dialogue transcript: every turn in time order, tagged where the
            # speaker changes — the same [S1]/[S2] convention the corpus sidecar uses
            parts, last = [], None
            for _, text, tag in sorted(acc["words"], key=lambda w: w[0]):
                if tag != last:
                    parts.append(f"[{tag}]" if tag else "")
                    last = tag
                parts.append(text)
            langs = Counter(v.get("lang") for v in speakers.values() if v.get("lang"))
            # where the audio came from. A flat record inherits these from its segments;
            # the nested one is built field by field, so they were being dropped — and
            # without them a later pass (run_nv_long_pipeline.sh scoping to this file)
            # has no way to find the container in the corpus, or to tell one recording's
            # segments from another's.
            src = next((r for r in rows if r.get("source_tar")), {})
            record = {
                "id": parent,
                "source_tar": src.get("source_tar"),
                "source_member": src.get("source_member"),
                "n_speakers": len(speakers),
                "n_segments": len(rows),
                "speech_seconds": round(speech, 2),
                "file_seconds": round(file_seconds, 2) if file_seconds else None,
                "speech_ratio": round(speech / file_seconds, 3) if file_seconds else None,
                "lang": langs.most_common(1)[0][0] if langs else None,
                "txt": " ".join(x for x in parts if x),
                "speakers": speakers,
            }
            sink.write(record)
            written += 1
            _caption(parent, rows, record)

    print(f"{written} file-level record(s) -> {args.out}")
    if skipped:
        print(f"{skipped} container(s) skipped (below --min-segments or unusable)")
    if untagged:
        print(f"{untagged} tagless voice group(s) left out of `speakers` (sidecar strays)")
    if args.caption_manifest:
        Path(args.caption_manifest).parent.mkdir(parents=True, exist_ok=True)
        with open(args.caption_manifest, "w", encoding="utf-8") as f:
            for row in caption_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        key = "audio_seconds" if audio_dir is not None else "window_seconds"
        total = sum(r.get(key) or 0.0 for r in caption_rows)
        kind = "full recording(s)" if audio_dir is not None else "caption window(s)"
        print(f"{len(caption_rows)} {kind} ({total / 60:.1f} min) -> {args.caption_manifest}")
        print(f"audio -> {audio_dir if audio_dir is not None else caption_dir}")
        if missing_audio:
            print(f"[warn] {len(missing_audio)} record(s) have no audio in {audio_dir}; "
                  "re-run prepare_long.py with --container-dir")
        elif len(caption_rows) < written:
            print(f"[warn] {written - len(caption_rows)} record(s) got no excerpt; "
                  "their segment audio was missing from --manifest")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ManifestError as e:
        print(f"manifest error: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)
