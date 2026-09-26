"""``continuo-nv`` — the nonverbal-vocalization pass: NVV tags per clip, and nothing else."""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

from .. import config
from ..audio import batched
from ..jsonl import JsonlWriter, done_ids, dumps, load_manifest, resolve_audio_path

SAMPLE_RATE = 16000

# The model writes events as `[Laughter]`, `[Surprise-oh]`, `[Question-en]`. Letters and
# hyphens only, so this cannot swallow a bracket that belongs to the transcript itself.
TAG_RE = re.compile(r"\[([A-Za-z][A-Za-z-]*)\]")

# Punctuation differs between the corpus transcript and this model's output, so it is
# stripped from both before their lengths are compared. See ``asr_ratio``.
_PUNCT_RE = re.compile(r"[\uff0c\u3002\uff01\uff1f\u3001,.!?;:\"'()\[\]\u2014\u2026\s]")


def add_nvbench_repo() -> Path:
    """Put the NV-Bench checkout on ``sys.path`` (idempotent) and return its root."""
    root = config.nvbench_repo().resolve()
    if not (root / "model.py").is_file():
        raise FileNotFoundError(
            f"NV-Bench checkout not found at {root} (no model.py). Set CONTINUO_EXPRESSIVE_NVBENCH_REPO "
            "to your clone of https://github.com/nvbench/NV-Bench.")
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return root


def split_tags(text: str) -> tuple[str, list[str]]:

    tags = TAG_RE.findall(text)
    clean = TAG_RE.sub("", text)
    return " ".join(clean.split()), tags


def asr_ratio(asr_text: str, manifest_text: str | None) -> float | None:
    """How much of the corpus transcript this model also heard, as a length ratio.

    A low ratio can indicate that the model emitted an event instead of
    transcribing speech, particularly on singing or sustained vowels.

    So a low ratio means the tags on that row are describing a failure to transcribe,
    not a vocalization. Nothing is dropped here — the gate belongs downstream, where
    retuning it costs a file read rather than another pass over the corpus — but without
    this number the caller cannot tell the two apart at all.

    ``None`` when the manifest carried no transcript to compare against.
    """
    if not manifest_text:
        return None
    ref = _PUNCT_RE.sub("", manifest_text)
    if not ref:
        return None
    return round(len(_PUNCT_RE.sub("", asr_text)) / len(ref), 2)


def _audio_with_prefetch(rows: Sequence[dict], loader, size: int, workers: int):
    """Yield ``(rows, wavs)`` per chunk, decoding the next one on a thread pool.

    Decoding is CPU work and the forward pass is not, so doing them in turn leaves the
    GPU idle for whichever is slower. Same pattern as the tags pass's prefetch.
    """
    blocks = list(batched(list(rows), size))
    if not blocks:
        return
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        def prepare(block):
            kept, wavs = [], []
            for row, wav in zip(block, pool.map(loader, block)):
                if wav is None:
                    continue
                kept.append(row)
                wavs.append(wav)
            return kept, wavs

        ahead = ThreadPoolExecutor(max_workers=1)
        try:
            pending = ahead.submit(prepare, blocks[0])
            for nxt in blocks[1:] + [None]:
                current = pending.result()
                pending = ahead.submit(prepare, nxt) if nxt is not None else None
                yield current
                if pending is None:
                    break
        finally:
            ahead.shutdown(wait=False)


def load_model(model_dir: Path, device: str):
    add_nvbench_repo()
    from model import SenseVoiceSmall            # noqa: E402  (needs the repo on sys.path)

    model, kwargs = SenseVoiceSmall.from_pretrained(model=str(model_dir), device=device)
    model.eval()
    return model, kwargs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="continuo-nv",
        description="Tag nonverbal vocalizations (laughter, coughs, sighs, hesitations) "
                    "with Multilingual-NVASR.")
    src = p.add_argument_group("input")
    src.add_argument("--manifest", help="JSONL; each row needs wav_path or source_tar/source_member")
    src.add_argument("--audio", help="annotate a single audio file")
    src.add_argument("--audio-root", default="", help="root for relative wav_path, enforced")
    src.add_argument("--tar-dir", default="",
                     help="corpus directory for source_tar rows (also CONTINUO_EXPRESSIVE_TAR_DIR)")
    src.add_argument("--limit", type=int, default=0, help="stop after N clips")
    src.add_argument("--shard", default="",
                     help="`i/n` — take rows where index %% n == i, by line number")

    out = p.add_argument_group("output")
    out.add_argument("--out", default="", help="write JSONL here")
    out.add_argument("--carry", default="",
                     help="comma-separated manifest fields to copy into every row")
    out.add_argument("--resume", action="store_true",
                     help="append to --out, skipping ids already in it")
    out.add_argument("--done-from", nargs="*", default=[], metavar="GLOB",
                     help="extra result files whose ids also count as done under "
                          "--resume; pass the sibling shards so changing the worker "
                          "count does not redo finished work")
    out.add_argument("--stdout", action="store_true", help="also echo records to stdout")

    run = p.add_argument_group("runtime")
    run.add_argument("--model-dir", default="",
                     help="NVASR model directory (overrides CONTINUO_EXPRESSIVE_NVASR_DIR)")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--batch-size", type=int, default=8,
                     help="clips per forward pass; 8 measured fastest on a 24 GB card")
    run.add_argument("--workers", type=int, default=8, help="audio-decoding threads")
    run.add_argument("--language", default="auto",
                     choices=["auto", "zh", "en", "yue", "ja", "ko"],
                     help="`auto` detects per clip; naming it is faster and safer when known")
    run.add_argument("--suspect-below", type=float, default=0.6,
                     help="flag a row `suspect` when its asr_ratio falls under this. "
                          "0.6 separates the collapse cases from the healthy tags "
                          "(measured: Crying 0.17, everything else 0.93-1.00); it only "
                          "labels rows, it never drops them. 0 turns the flag off.")
    run.add_argument("--timestamps", action="store_true",
                     help="also emit CTC-aligned event times. Forces --batch-size 1: the "
                          "aligner raises a CUDA assert on a batch, so this costs ~6x.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config.apply_hf_endpoint()
    if args.tar_dir:
        os.environ["CONTINUO_EXPRESSIVE_TAR_DIR"] = args.tar_dir

    if args.audio:
        path = resolve_audio_path(args.audio, args.audio_root or None)
        rows = [{"id": path.name, "_path": path}]
    elif args.manifest:
        shard = None
        if args.shard:
            try:
                i, n = (int(x) for x in args.shard.split("/", 1))
            except ValueError:
                build_parser().error(f"--shard {args.shard!r} is not `i/n`")
            if not 0 <= i < n:
                build_parser().error(f"--shard {args.shard!r}: need 0 <= i < n")
            shard = (i, n)
        rows = load_manifest(args.manifest, limit=args.limit,
                             audio_root=args.audio_root or None, shard=shard)
        if shard is not None:
            print(f"  shard {shard[0]}/{shard[1]}: {len(rows)} clip(s) of this manifest",
                  file=sys.stderr)
    else:
        build_parser().error("give --manifest or --audio")

    if args.resume and not args.out:
        build_parser().error("--resume needs --out")
    if args.resume:
        # A result row is keyed by id and says nothing about which shard produced it, so
        # every sibling shard file is valid resume evidence. Reading only --out means a
        # run resumed with a different worker count re-does most of its finished work,
        # and the card count on a shared box is not something a restart gets to assume.
        already = done_ids(args.out)
        out_path = os.path.abspath(args.out)
        for path in sorted(set(sum((glob.glob(p) for p in args.done_from), []))):
            if os.path.abspath(path) != out_path:
                already |= done_ids(path)
        if already:
            rows = [r for r in rows if r["id"] not in already]
            print(f"  resuming: {len(already)} done, {len(rows)} left", file=sys.stderr)
    if not rows:
        print("  nothing to do", file=sys.stderr)
        return 0

    batch_size = args.batch_size
    if args.timestamps and batch_size != 1:
        print("  --timestamps: forcing --batch-size 1 (the CTC aligner cannot batch)",
              file=sys.stderr)
        batch_size = 1

    model_dir = Path(args.model_dir).expanduser() if args.model_dir else config.nvasr_dir()
    model, kwargs = load_model(model_dir.resolve(), args.device)
    print(f"  loaded NVASR <- {model_dir}", file=sys.stderr)

    carry = [f.strip() for f in args.carry.split(",") if f.strip()]
    echo = args.stdout or not args.out

    def decode_one(row):
        try:
            from ..audio import load_source
            return load_source(row, SAMPLE_RATE)
        except Exception as e:
            print(f"[warn] skip {row['id']}: {type(e).__name__}: {e}", file=sys.stderr)
            return None

    written = tagged = suspect = 0
    started = time.time()
    with JsonlWriter(args.out, append=args.resume) as sink:
        for kept, wavs in _audio_with_prefetch(rows, decode_one, batch_size, args.workers):
            if not wavs:
                continue
            try:
                res = model.inference(
                    data_in=wavs if len(wavs) > 1 else wavs[0],
                    language=args.language, use_itn=False, ban_emo_unk=False,
                    output_timestamp=args.timestamps, fs=SAMPLE_RATE, **kwargs)
            except Exception as e:
                # One bad batch should not cost the run; those ids simply stay absent
                # from --out, so a later --resume picks them up.
                print(f"[warn] batch of {len(wavs)} failed ({type(e).__name__}: {e}); "
                      "its clips stay pending", file=sys.stderr)
                continue

            from utils.postprocess import rich_transcription_postprocess  # noqa: E402

            for row, item in zip(kept, res[0]):
                nv_text = rich_transcription_postprocess(item["text"])
                clean, tags = split_tags(nv_text)
                rec = {"id": row["id"], "nv_text": nv_text, "text": clean,
                       "nv_tags": tags, "nv_counts": dict(Counter(tags)), "n_nv": len(tags),
                       "asr_ratio": asr_ratio(clean, row.get("txt"))}
                if args.timestamps:
                    rec["timestamp"] = item.get("timestamp", [])
                if rec["asr_ratio"] is not None and rec["asr_ratio"] < args.suspect_below:
                    rec["suspect"] = True
                    suspect += 1
                for f in carry:
                    if f in row:
                        rec[f] = row[f]
                sink.write(rec)
                written += 1
                tagged += bool(tags)
                if echo:
                    print(dumps(rec), flush=True)

            if written and written % 1000 < len(wavs):
                rate = written / max(time.time() - started, 1e-9)
                print(f"  {written} clip(s), {tagged} with an NVV, {rate:.1f} clips/s",
                      file=sys.stderr)

    elapsed = time.time() - started
    note = f"; {suspect} flagged suspect (asr_ratio < {args.suspect_below})" if suspect else ""
    print(f"  done: {written} clip(s) in {elapsed:.1f}s ({written / max(elapsed, 1e-9):.1f} "
          f"clips/s); {tagged} carried at least one NVV{note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
