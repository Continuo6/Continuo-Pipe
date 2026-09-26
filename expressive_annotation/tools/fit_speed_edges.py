#!/usr/bin/env python3
"""Derive speaking-rate edges for languages that have none, by shape transfer.

Only English has physical-speed gold, so only English edges are fitted. Every other
language gets its edges the way Chinese did: keep the SHAPE of the English fit — its
edges sit at 0.843x and 1.335x of the English median — and re-anchor the SCALE on the
target language's own median character rate.

Why not tertile the corpus: that would declare exactly a third of every language fast,
and human speed labels are not uniform. The English gold itself is 24/68/7%. The shape
transfer reproduces that skew; a tertile erases it.

What the method cannot do is tell you whether a language's speed distribution really is
shaped like English. Nothing here verifies that, and no gold exists to verify it with.
Read the output as a within-language ranking.

    python tools/fit_speed_edges.py runs/continuo/annotations.jsonl

Prints a dict ready to paste into features/buckets.py. Constants belong in source, not
in a per-run computation: an edge re-derived from whatever corpus is in front of you
makes `speed` mean "fast for this batch", and two corpora stop being comparable.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from continuo_expressive.features.buckets import (  # noqa: E402
    SPEED_CPS_EDGES, SPEED_FIT_MIN_CLIPS, SPEED_LABELS, SPEED_SHAPE)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fit_speed_edges", description=__doc__.splitlines()[0])
    p.add_argument("annotations", nargs="+", help="annotation JSONL with speed_cps + lang")
    p.add_argument("--min-clips", type=int, default=SPEED_FIT_MIN_CLIPS,
                   help=f"skip languages with fewer clips (default {SPEED_FIT_MIN_CLIPS})")
    p.add_argument("--refit-existing", action="store_true",
                   help="also print languages that already have edges, for comparison")
    args = p.parse_args(argv)

    rates: dict[str, list[float]] = defaultdict(list)
    for path in args.annotations:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("speed_cps") is not None and row.get("lang"):
                    rates[row["lang"]].append(row["speed_cps"])

    lo_mult, hi_mult = SPEED_SHAPE
    print(f"# en gold edges = {lo_mult:.3f}x / {hi_mult:.3f}x of the en median\n")
    print("SPEED_CPS_EDGES = {")
    skipped = []
    for lang, values in sorted(rates.items(), key=lambda kv: -len(kv[1])):
        known = lang in SPEED_CPS_EDGES
        if len(values) < args.min_clips:
            skipped.append((lang, len(values)))
            continue
        median = statistics.median(values)
        lo, hi = round(median * lo_mult, 1), round(median * hi_mult, 1)
        if known and not args.refit_existing:
            cur_lo, cur_hi = SPEED_CPS_EDGES[lang]
            shift = ""
            if abs(median * lo_mult - cur_lo) > 0.15 * cur_lo:
                # a shipped anchor that disagrees with this corpus is worth seeing:
                # either this corpus really is faster, or the anchor does not transfer
                shift = f"  <-- corpus median would give ({lo}, {hi})"
            print(f'    "{lang}": {SPEED_CPS_EDGES[lang]},'.ljust(28)
                  + f"# shipped, n={len(values)} here{shift}")
            continue
        slow = sum(v < lo for v in values)
        fast = sum(v > hi for v in values)
        mid = len(values) - slow - fast
        print(f'    "{lang}": ({lo}, {hi}),'.ljust(28)
              + f"# n={len(values)}, median {median:.2f} CPS"
              + f" -> {100*slow/len(values):.0f}/{100*mid/len(values):.0f}/{100*fast/len(values):.0f}%")
    print("}")
    print(f"\n# reference: the en gold's own human labels are 24/68/7% "
          f"({'/'.join(SPEED_LABELS)})")
    if skipped:
        listing = ", ".join(f"{lang} n={n}" for lang, n in sorted(skipped, key=lambda kv: -kv[1]))
        print(f"# below --min-clips {args.min_clips}, left without edges: {listing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
