#!/usr/bin/env python3
"""Append a finished run's annotations to the corpus, or refuse and change nothing.

``--resume`` keys on clip id, and so does everything downstream. An id that lands in
``final.jsonl`` twice is invisible from that moment: no pass will complain, and the
duplicate only shows up as a row count that no longer matches the ledger. So the check
happens here, before a byte is written, and covers both directions — duplicates inside
the incoming files, and ids the corpus already has.

Appending is not atomic, so the sizes of both corpus files are recorded first and the
files are truncated back to them if the row counts afterwards are not exactly what was
expected. A crash mid-append therefore costs a re-run of this step, not the corpus.

``tars.txt`` is rebuilt from the merged ``final.jsonl`` rather than appended to, because
it is the ledger the next increment diffs against: derived from the data it describes,
it cannot drift away from it.

A tar that was read and yielded nothing still has to be in it. Some shards
hold only ``_dlg_``/``_long_`` containers, which the short pipeline filters out. Left out of the ledger
they look unprocessed forever and every later increment re-indexes them for nothing, so
``--attempted`` names the tars that were tried and they are written with a count of 0.

    python tools/merge_corpus.py --work runs/short-new/work --corpus-dir runs/short-all
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ID = re.compile(rb'"id"\s*:\s*"([^"]*)"')
TAR = re.compile(rb'"source_tar"\s*:\s*"([^"]*)"')


def ids_of(path: Path):
    with open(path, "rb") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            m = ID.search(line)
            if m is None:
                raise SystemExit(f"{path}:{n}: row has no id")
            yield m.group(1)


def append(dest: Path, sources: list[Path]) -> int:
    added = 0
    with open(dest, "ab") as out:
        for s in sources:
            with open(s, "rb") as f:
                for line in f:
                    if line.strip():
                        if not line.endswith(b"\n"):
                            line += b"\n"
                        out.write(line)
                        added += 1
    return added


def count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in open(path, "rb") if line.strip())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="merge_corpus")
    p.add_argument("--work", required=True, help="the run's work/ directory")
    p.add_argument("--corpus-dir", required=True, help="destination corpus directory")
    p.add_argument("--annot-glob", default="annot*.jsonl")
    p.add_argument("--emotion-glob", default="emotion*.jsonl")
    p.add_argument("--emotion-scope", default="row", choices=("row", "finer"),
                   help="'row' (default): one emotion score per annotation, and a count "
                        "mismatch is a bug worth saying so. 'finer': the scores are at a "
                        "smaller unit than the annotations — the long-audio path scores "
                        "every segment while a record is a whole recording — so they are "
                        "not expected to line up and the count is reported, not warned.")
    p.add_argument("--no-ledger", action="store_true",
                   help="skip rebuilding tars.txt. The long-audio path needs this: its "
                        "records are one per recording and carry no source_tar, so there "
                        "is nothing to rebuild a per-tar ledger from and the caller "
                        "writes it instead.")
    p.add_argument("--attempted", default="",
                   help="file of tar paths that were read this run; any that produced no "
                        "rows are recorded in the ledger with a count of 0 so the next "
                        "increment does not read them again")
    args = p.parse_args(argv)

    work = Path(args.work)
    corpus = Path(args.corpus_dir)
    corpus.mkdir(parents=True, exist_ok=True)
    final = corpus / "final.jsonl"
    scores = corpus / "emotion_scores.jsonl"

    annot = sorted(work.glob(args.annot_glob))
    emotion = sorted(work.glob(args.emotion_glob))
    if not annot:
        print(f"no {args.annot_glob} in {work}", file=sys.stderr)
        return 2

    # --- the check, before anything is written -----------------------------------
    incoming: dict[bytes, Path] = {}
    dupes: list[tuple[bytes, Path, Path]] = []
    for f in annot:
        for i in ids_of(f):
            if i in incoming:
                dupes.append((i, incoming[i], f))
            else:
                incoming[i] = f
    if dupes:
        print(f"REFUSED: {len(dupes)} duplicate id(s) among the incoming files", file=sys.stderr)
        for i, a, b in dupes[:5]:
            print(f"  {i.decode()}  in {a.name} and {b.name}", file=sys.stderr)
        return 1

    have = set(ids_of(final)) if final.exists() else set()
    clash = sorted(incoming.keys() & have)
    if clash:
        print(f"REFUSED: {len(clash)} id(s) are already in {final}", file=sys.stderr)
        for i in clash[:5]:
            print(f"  {i.decode()}", file=sys.stderr)
        return 1

    n_emotion = sum(count(f) for f in emotion)
    if emotion and n_emotion != len(incoming):
        if args.emotion_scope == "finer":
            print(f"{n_emotion} emotion score(s) for {len(incoming)} record(s) "
                  f"({n_emotion / len(incoming):.1f} per record, as expected)")
        else:
            print(f"[warn] {n_emotion} emotion row(s) against {len(incoming)} "
                  "annotation(s) — the raw scores will not line up 1:1", file=sys.stderr)

    # --- append, with the sizes to fall back to ----------------------------------
    before = {final: final.stat().st_size if final.exists() else 0,
              scores: scores.stat().st_size if scores.exists() else 0}
    want_final = len(have) + len(incoming)
    want_scores = count(scores) + n_emotion

    try:
        added = append(final, annot)
        if emotion:
            append(scores, emotion)
        got_final, got_scores = count(final), count(scores)
        if got_final != want_final or (emotion and got_scores != want_scores):
            raise RuntimeError(
                f"row counts after append are wrong: final {got_final} (wanted {want_final}), "
                f"emotion_scores {got_scores} (wanted {want_scores})")
    except BaseException as e:                      # KeyboardInterrupt included, on purpose
        for path, size in before.items():
            if path.exists():
                with open(path, "r+b") as f:
                    f.truncate(size)
        print(f"REFUSED: {e}\n  both corpus files truncated back to their previous size",
              file=sys.stderr)
        return 1

    # --- ledger, rebuilt from the data it describes ------------------------------
    if args.no_ledger:
        print(f"merged {added} row(s) -> {final} ({got_final} total)")
        if emotion:
            print(f"merged {n_emotion} emotion score row(s) -> {scores} ({got_scores} total)")
        return 0

    tally: collections.Counter[str] = collections.Counter()
    with open(final, "rb") as f:
        for line in f:
            m = TAR.search(line)
            if m:
                tally[m.group(1).decode()] += 1
    empty = 0
    if args.attempted:
        for line in open(args.attempted, encoding="utf-8"):
            name = Path(line.strip()).name
            if name and name not in tally:
                tally[name] = 0
                empty += 1
    with open(corpus / "tars.txt", "w", encoding="utf-8") as f:
        for tar, n in sorted(tally.items()):
            f.write(f"{tar}\t{n}\n")

    print(f"merged {added} row(s) -> {final} ({got_final} total)")
    if emotion:
        print(f"merged {n_emotion} emotion score row(s) -> {scores} ({got_scores} total)")
    print(f"ledger rebuilt: {len(tally)} tar(s) -> {corpus / 'tars.txt'}"
          + (f" ({empty} of them read but empty)" if empty else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
