"""``continuo-firered`` — optional FireRedLID pass, the first stage of the Chinese cascade.

Emits ``{id, pred}`` per clip. Feed it to ``continuo-annotate --firered-jsonl`` and Chinese
accent switches from Voxlect alone to the FireRedLID -> Voxlect cascade; see
:mod:`continuo_expressive.ensemble.dialect`.

Its own pass because FireRedLID's dependencies (kaldi_native_fbank, kaldiio) do not
coexist with the dialect-head environment.

Neither the FireRedASR2S source nor the LID weights ship here; point
``CONTINUO_EXPRESSIVE_FIRERED_REPO`` and ``CONTINUO_EXPRESSIVE_FIRERED_MODEL_DIR`` at your checkout and weights.

``process()`` returns nothing for very large batches, so clips are fed in fixed
chunks — that is a property of the upstream API, not a tuning knob.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .. import config
from ..jsonl import JsonlWriter, ManifestError, done_ids, dumps, load_manifest

CHUNK = 16


def _load_model(repo: Path, model_dir: Path, use_gpu: bool):
    if not (repo / "fireredasr2s").is_dir():
        raise FileNotFoundError(
            f"FireRedASR2S checkout not found at {repo}. Set CONTINUO_EXPRESSIVE_FIRERED_REPO "
            "(or --repo) to your clone.")
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"FireRedLID weights not found at {model_dir}. Set CONTINUO_EXPRESSIVE_FIRERED_MODEL_DIR "
            "(or --model-dir).")
    sys.path.insert(0, str(repo))
    from fireredasr2s.fireredlid.lid import FireRedLid, FireRedLidConfig
    return FireRedLid.from_pretrained(
        str(model_dir), FireRedLidConfig(use_gpu=use_gpu, use_half=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="continuo-firered",
        description="FireRedLID language/dialect pass -> {id, pred} for the zh accent cascade.")
    p.add_argument("--manifest", required=True, help="JSONL with wav_path (+ optional id)")
    p.add_argument("--audio-root", default="", help="root for relative wav_path, enforced")
    p.add_argument("--out", default="", help="write JSONL here")
    p.add_argument("--resume", action="store_true", help="append to --out, skipping done ids")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--repo", default="", help="FireRedASR2S checkout (overrides CONTINUO_EXPRESSIVE_FIRERED_REPO)")
    p.add_argument("--model-dir", default="", help="LID weights (overrides CONTINUO_EXPRESSIVE_FIRERED_MODEL_DIR)")
    p.add_argument("--cpu", action="store_true", help="run on CPU")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_manifest(args.manifest, limit=args.limit,
                         audio_root=args.audio_root or None)

    if args.resume and not args.out:
        build_parser().error("--resume needs --out")
    if args.resume:
        already = done_ids(args.out)
        if already:
            rows = [r for r in rows if r["id"] not in already]
            print(f"  resuming: {len(already)} done, {len(rows)} left", file=sys.stderr)
    if not rows:
        print("  nothing to do", file=sys.stderr)
        return 0

    repo = Path(args.repo).expanduser() if args.repo else config.firered_repo()
    model_dir = Path(args.model_dir).expanduser() if args.model_dir else config.firered_model_dir()
    model = _load_model(repo.resolve(), model_dir.resolve(), use_gpu=not args.cpu)
    print(f"  loaded FireRedLID <- {model_dir}", file=sys.stderr)

    echo = not args.out
    written = 0
    with JsonlWriter(args.out, append=args.resume) as sink:
        for start in range(0, len(rows), CHUNK):
            chunk = rows[start:start + CHUNK]
            try:
                results = model.process([r["id"] for r in chunk],
                                        [str(r["_path"]) for r in chunk])
            except Exception as e:
                print(f"[warn] chunk at {start} failed: {type(e).__name__}: {e}",
                      file=sys.stderr)
                continue
            by_id = {r["uttid"]: (r.get("lang") or "") for r in results}
            for row in chunk:
                rec = {"id": row["id"], "pred": by_id.get(row["id"], "")}
                sink.write(rec)
                if echo:
                    print(dumps(rec))
                written += 1

    print(f"  done: {written} clip(s)", file=sys.stderr)
    if args.out:
        print(f"  wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ManifestError as e:
        print(f"manifest error: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)
