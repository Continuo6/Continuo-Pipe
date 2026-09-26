#!/usr/bin/env python3
"""Fold external emotion predictions into annotation files written without them.

``continuo-annotate --emotion-jsonl`` merges emotion as it writes each record, which needs
the external prediction pass to have finished first. On a corpus that runs both passes
side by side that ordering does not hold, and re-running the annotation just to attach
a label already computed would cost GPU-days.

The gate lives at merge time by design — "retuning tau costs a file read rather than
another pass over the corpus" — so this applies it offline, exactly as the annotate
pass would:

    python tools/merge_emotion.py --annot runs/x/work/shard0.jsonl \
        --emotion runs/x/work/emotion_a.jsonl runs/x/work/emotion_b.jsonl \
        --out runs/x/work/shard0.emo.jsonl --tau <your-threshold>

Writing is atomic, so the annotation file survives a crash mid-merge; ``--in-place``
replaces the input once the new file is complete.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from continuo_expressive.ensemble.emotion_gate import FIELDS, gate_emotion
from continuo_expressive.jsonl import ManifestError, index_by_id, read_jsonl, write_jsonl_atomic


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="merge_emotion",
        description="Attach gated emotion to annotation records, by clip id.")
    p.add_argument("--annot", required=True, help="annotation JSONL to enrich")
    p.add_argument("--emotion", nargs="+", required=True,
                   help="external emotion prediction files; several may be given")
    p.add_argument("--out", default="", help="destination (default: alongside --annot)")
    p.add_argument("--in-place", action="store_true",
                   help="replace --annot once the merged file is complete")
    p.add_argument("--tau", type=float, required=True,
                   help="confidence gate in [0, 1]; top-1 below tau -> emotion null")
    args = p.parse_args(argv)
    if not 0 <= args.tau <= 1:
        p.error("--tau must be in [0, 1]")
    if args.in_place and args.out:
        # --in-place renames the result over the input, so --out would name a file that
        # does not exist when the run finishes. Silently, until whatever reads it next
        # dies with FileNotFoundError.
        p.error("--in-place replaces --annot; do not also pass --out")

    raw: dict[str, dict] = {}
    for path in args.emotion:
        found = index_by_id(path)
        print(f"  {path}: {len(found)} clip(s)", flush=True)
        raw.update(found)

    out_path = args.out or (args.annot[:-6] if args.annot.endswith(".jsonl")
                            else args.annot) + ".emo.jsonl"
    matched = kept = total = 0

    def rows():
        nonlocal matched, kept, total
        for row in read_jsonl(args.annot, required=("id",)):
            total += 1
            entry = raw.get(row["id"])
            if entry is not None:
                matched += 1
                gated = gate_emotion(entry, args.tau)
                row.update(gated)
                kept += bool(gated["emotion"])
            else:
                # a clip the external pass has not reached: leave the fields absent rather
                # than stamp a null, so a later merge can still fill it in
                for field in FIELDS:
                    row.pop(field, None)
            yield row

    written = write_jsonl_atomic(out_path, rows())
    print(f"\n{written} row(s) -> {out_path}")
    print(f"  {matched}/{total} had a SER result; {kept} survived tau={args.tau} "
          f"({100 * kept / matched:.1f}% of those measured)" if matched else
          f"  no clip in {args.annot} appears in the SER output")
    if args.in_place:
        Path(out_path).replace(args.annot)
        print(f"  replaced {args.annot}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ManifestError as e:
        print(f"manifest error: {e}", file=sys.stderr)
        sys.exit(2)
