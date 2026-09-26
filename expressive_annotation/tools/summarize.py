#!/usr/bin/env python3
"""Summarise an annotated corpus: coverage per field, distributions, and what is null.

Coverage is the first thing to look at and the easiest to skip. Every field in this
pipeline can legitimately come back null — the clip was too short, the language has no
bucket, the confidence gate rejected the label — and a distribution table that silently
drops nulls will make a field look far more complete than it is. So this reports the
null rate for every field, per language, before it reports anything else.

    python tools/summarize.py runs/continuo/annotations.jsonl
    python tools/summarize.py runs/continuo/annotations.jsonl --markdown > runs/continuo/REPORT.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

CATEGORICAL = ["gender", "age_band", "accent", "volume", "pitch", "speed", "emotion",
               "caption_form", "caption_lang"]
CONTINUOUS = ["age_years", "age_vox_years", "volume_lufs", "pitch_hz", "speed_cps",
              "emotion_confidence"]


def load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(n: int, total: int) -> str:
    return f"{100 * n / total:5.1f}%" if total else "    -"


def quantiles(values: list[float]) -> str:
    if not values:
        return "-"
    values = sorted(values)
    q = lambda f: values[min(len(values) - 1, int(len(values) * f))]  # noqa: E731
    return f"p10 {q(.1):.1f} | p50 {q(.5):.1f} | p90 {q(.9):.1f}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="summarize", description=__doc__.splitlines()[0])
    p.add_argument("annotations", help="annotations JSONL")
    p.add_argument("--markdown", action="store_true", help="emit a Markdown report")
    p.add_argument("--top", type=int, default=8, help="values shown per categorical field")
    args = p.parse_args(argv)

    rows = load(args.annotations)
    if not rows:
        print("no rows", file=sys.stderr)
        return 1
    n = len(rows)
    out = print
    h1, h2 = ("# ", "## ") if args.markdown else ("== ", "-- ")

    out(f"{h1}Annotation summary")
    out(f"\n{n} clips from `{args.annotations}`\n")

    langs = Counter(r.get("lang") for r in rows)
    out(f"{h2}Languages\n")
    if args.markdown:
        out("| language | clips | share |\n|---|---:|---:|")
    for lang, count in langs.most_common():
        row = f"{lang or '(none)':>8} {count:6d} {pct(count, n)}"
        out(f"| {lang or '(none)'} | {count} | {pct(count, n)} |" if args.markdown else "  " + row)

    out(f"\n{h2}Field coverage\n")
    if args.markdown:
        out("| field | present | coverage |\n|---|---:|---:|")
    for field in CATEGORICAL + CONTINUOUS:
        if not any(field in r for r in rows):
            continue
        present = sum(1 for r in rows if r.get(field) is not None)
        line = f"{field:20s} {present:6d}/{n} {pct(present, n)}"
        out(f"| `{field}` | {present}/{n} | {pct(present, n)} |" if args.markdown else "  " + line)

    out(f"\n{h2}Distributions\n")
    for field in CATEGORICAL:
        values = [r.get(field) for r in rows if r.get(field) is not None]
        if not values:
            continue
        counts = Counter(values)
        shown = ", ".join(f"{v} {c} ({pct(c, len(values)).strip()})"
                          for v, c in counts.most_common(args.top))
        extra = f", +{len(counts) - args.top} more" if len(counts) > args.top else ""
        out(f"- **{field}** ({len(values)} labelled): {shown}{extra}" if args.markdown
            else f"  {field:14s} {shown}{extra}")

    out(f"\n{h2}Measurements\n")
    for field in CONTINUOUS:
        values = [r[field] for r in rows if isinstance(r.get(field), (int, float))]
        if values:
            out(f"- **{field}**: {quantiles(values)}" if args.markdown
                else f"  {field:20s} {quantiles(values)}")

    # Accent and speed are language-routed: a null there is by design for a language
    # with no head or no fitted edges, and should not read as a failure.
    out(f"\n{h2}Language-routed fields\n")
    by_lang = defaultdict(lambda: {"n": 0, "accent": 0, "speed": 0, "emotion": 0})
    for r in rows:
        slot = by_lang[r.get("lang")]
        slot["n"] += 1
        for field in ("accent", "speed", "emotion"):
            if r.get(field) is not None:
                slot[field] += 1
    if args.markdown:
        out("| language | clips | accent | speed | emotion |\n|---|---:|---:|---:|---:|")
    for lang, slot in sorted(by_lang.items(), key=lambda kv: -kv[1]["n"]):
        cells = [pct(slot[f], slot["n"]) for f in ("accent", "speed", "emotion")]
        if args.markdown:
            out(f"| {lang or '(none)'} | {slot['n']} | " + " | ".join(cells) + " |")
        else:
            out(f"  {lang or '(none)':>8} {slot['n']:6d}  accent {cells[0]}  "
                f"speed {cells[1]}  emotion {cells[2]}")

    if any("caption" in r for r in rows):
        captioned = sum(1 for r in rows if r.get("caption"))
        lengths = [len(r["caption"]) for r in rows if r.get("caption")]
        out(f"\n{h2}Instruction text\n")
        out(f"- captioned: {captioned}/{n} ({pct(captioned, n).strip()})" if args.markdown
            else f"  captioned {captioned}/{n} {pct(captioned, n)}")
        if lengths:
            out(f"- caption length (chars): {quantiles([float(x) for x in lengths])}"
                if args.markdown else f"  length (chars) {quantiles([float(x) for x in lengths])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
