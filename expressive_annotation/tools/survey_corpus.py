#!/usr/bin/env python3
"""Count what is actually in a set of Continuo tars, by language and by view.

Before committing GPU-weeks to a corpus you want to know what it holds. The pack
index lists every entity but carries neither language nor duration, and those live one
per JSON sidecar — so this reads the sidecars, which is the expensive part, and caches
the result per tar so it is paid once.

    python tools/survey_corpus.py --tars 'corpus/*.tar' \
        --out runs/survey.jsonl

Resumable and incremental: each tar's counts are appended as it finishes and skipped on
a re-run, so a scan can be stopped, and a partial scan still summarises. Re-running with
``--summary-only`` reports on whatever has been collected so far.

Durations are the sidecars' own, in seconds of audio — not of speech. For a ``short``
container the two are the same thing; for ``long`` and ``dialogue`` they are not, and
the speech figure is what a segmented run would actually annotate.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from continuo_expressive.jsonl import read_jsonl

VIEWS = ("short", "long", "dialogue")


def load_idx(idx_path: Path) -> list[tuple[str, int, int]]:
    out = []
    with open(idx_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3 and parts[0].endswith(".json"):
                out.append((parts[0], int(parts[1]), int(parts[2])))
    return out


def survey_tar(tar_path: str) -> dict:
    """One tar -> ``{language: {view: [count, seconds, speech_seconds]}}``."""
    tar = Path(tar_path)
    result: dict = {"tar": tar.name, "error": None, "by_lang": {}}
    idx_path = tar.with_suffix(tar.suffix + ".idx")
    if not idx_path.is_file():
        result["error"] = f"{idx_path.name} missing"
        return result

    tally: dict = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0]))
    try:
        entries = load_idx(idx_path)
        with open(tar, "rb") as f:
            for name, offset, size in entries:
                f.seek(offset)
                try:
                    meta = json.loads(f.read(size))
                except json.JSONDecodeError:
                    continue
                view = meta.get("type")
                if view not in VIEWS:
                    continue
                shorts = meta.get("short") or []
                # a container's language is the one its utterances carry; the top-level
                # list can hold several and says nothing about which dominates
                langs = [s.get("language") for s in shorts if s.get("language")]
                lang = max(set(langs), key=langs.count) if langs else None
                if lang is None:
                    declared = [x for x in (meta.get("languages") or []) if x]
                    lang = declared[0] if len(declared) == 1 else "unknown"
                speech = sum(s.get("duration") or 0.0 for s in shorts)
                cell = tally[lang][view]
                cell[0] += 1
                cell[1] += float(meta.get("duration") or 0.0)
                cell[2] += speech
    except Exception as e:                       # one unreadable tar must not stop a scan
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    result["by_lang"] = {lang: {view: cell for view, cell in views.items()}
                         for lang, views in tally.items()}
    return result


def summarise(rows: list[dict]) -> dict:
    total: dict = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0]))
    for row in rows:
        for lang, views in (row.get("by_lang") or {}).items():
            for view, cell in views.items():
                acc = total[lang][view]
                acc[0] += cell[0]
                acc[1] += cell[1]
                acc[2] += cell[2]
    return total


def report(total: dict, scanned: int, of: int, target_hours: float = 0.0) -> None:
    """Hours per language, ranked by the thing that decides a run: annotatable speech."""
    def speech(lang):
        return sum(v[2] for v in total[lang].values()) / 3600

    langs = sorted(total, key=lambda l: -speech(l))
    scale = (of / scanned) if scanned and of > scanned else 1.0
    print(f"\n{scanned}/{of} tar(s) scanned"
          + (f" — the projected column extrapolates to all {of}" if scale > 1 else ""))
    print("\nspeech hours per language, by where the utterances live:\n")
    head = (f"{'lang':<8}{'standalone':>12}{'in long':>10}{'in dialogue':>13}"
            f"{'total h':>10}" + (f"{'projected':>11}" if scale > 1 else ""))
    print(head)
    print("-" * len(head))
    shown = 0.0
    for lang in langs:
        hours = {v: total[lang].get(v, [0, 0.0, 0.0])[2] / 3600 for v in VIEWS}
        if speech(lang) < 0.05:
            continue
        shown += speech(lang)
        line = (f"{lang:<8}{hours['short']:>12.1f}{hours['long']:>10.1f}"
                f"{hours['dialogue']:>13.1f}{speech(lang):>10.1f}")
        if scale > 1:
            line += f"{speech(lang) * scale:>11.0f}"
        print(line)
    total_h = sum(speech(l) for l in langs)
    print("-" * len(head))
    line = f"{'all':<8}{'':>12}{'':>10}{'':>13}{total_h:>10.1f}"
    if scale > 1:
        line += f"{total_h * scale:>11.0f}"
    print(line)
    if total_h - shown > 0.05:
        print(f"({total_h - shown:.1f} h in languages under 0.05 h each, omitted)")

    if target_hours:
        print(f"\nagainst a {target_hours:g} h target:")
        for lang in langs[:8]:
            have = speech(lang) * scale
            if have < 1:
                continue
            mark = "reaches it" if have >= target_hours else f"{have / target_hours:.0%} of it"
            print(f"  {lang:<5} {have:>8.0f} h projected — {mark}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="survey_corpus",
        description="Count Continuo tars by language and view, with durations.")
    p.add_argument("--tars", required=True, help="tar path or glob")
    p.add_argument("--out", required=True, help="per-tar counts, appended and resumed")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="cap tars, for a quick estimate")
    p.add_argument("--target-hours", type=float, default=0.0,
                   help="report each language against an hour budget, so a run can be "
                        "planned before it is started")
    p.add_argument("--summary-only", action="store_true",
                   help="report on what --out already holds and scan nothing")
    args = p.parse_args(argv)

    out_path = Path(args.out)
    done: dict[str, dict] = {}
    if out_path.exists():
        for row in read_jsonl(out_path):
            done[row["tar"]] = row

    tars = [Path(t) for t in sorted(glob.glob(args.tars)) if t.endswith(".tar")]
    if args.limit:
        tars = tars[:args.limit]
    todo = [t for t in tars if t.name not in done]

    if args.summary_only or not todo:
        rows = list(done.values())
        if not rows:
            print("nothing scanned yet", file=sys.stderr)
            return 2
        report(summarise(rows), len(rows), len(tars) or len(rows), args.target_hours)
        return 0

    print(f"{len(tars)} tar(s), {len(done)} already scanned, {len(todo)} to go",
          flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    with open(out_path, "a", encoding="utf-8") as sink, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, result in enumerate(pool.map(survey_tar, [str(t) for t in todo]), 1):
            if result["error"]:
                errors += 1
                print(f"[warn] {result['tar']}: {result['error']}", file=sys.stderr)
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            sink.flush()
            done[result["tar"]] = result
            if n % 20 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)

    report(summarise(list(done.values())), len(done), len(tars), args.target_hours)
    if errors:
        print(f"\n{errors} tar(s) could not be read", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
