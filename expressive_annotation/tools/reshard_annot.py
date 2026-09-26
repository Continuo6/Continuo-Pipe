#!/usr/bin/env python3
"""Re-split a half-finished sharded pass so it can resume on a different worker count.

``continuo-annotate --shard i/n`` claims rows by ``crc32(parent_id) % n``,
and each worker resumes from **its own** output file. So a run that stops on 5 cards and
comes back on 6 re-annotates every row whose owner changed, into a different file — the
work is redone and ``cat work/annot*.jsonl`` then holds the row twice, which inflates
every count the fold sums (``n_segments``, ``speech_seconds``, the span timeline).

This moves each finished row into the file the *new* split says owns it. Nothing is
recomputed and nothing is duplicated: after it runs, worker i's resume file holds exactly
the rows worker i owns and has already done.

    python tools/reshard_annot.py --work runs/x/chunks/0007/work \\
        --manifest runs/x/chunks/0007/manifest.jsonl --shards 6

``--prefix emotion`` does the same for external prediction shards. Files above the new count are
removed once their rows have been redistributed, so the concatenation stays exact.
Writes through temporary files and renames, so an interrupted run leaves the old split
intact rather than half of each.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from pathlib import Path


def shard_of(parent_id, index: int, n: int) -> int:
    """The owner of a row, exactly as :func:`continuo_expressive.jsonl.load_manifest`."""
    key = zlib.crc32(str(parent_id).encode()) if parent_id else index
    return key % n


def manifest_keys(path: Path) -> dict:
    """``id -> (parent_id, line index)`` for every manifest row."""
    keys = {}
    with open(path) as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = row.get("id")
            if rid is not None:
                keys[rid] = (row.get("parent_id"), index)
    return keys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="reshard_annot")
    p.add_argument("--work", required=True, help="directory holding <prefix>NN.jsonl")
    p.add_argument("--manifest", required=True, help="the manifest those rows came from")
    p.add_argument("--shards", type=int, required=True, help="the NEW worker count")
    p.add_argument("--prefix", default="annot", help="annot (default) or emotion")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    work = Path(args.work)
    if args.shards < 1:
        p.error("--shards must be >= 1")
    files = sorted(work.glob(f"{args.prefix}[0-9][0-9].jsonl"))
    if not files:
        print(f"no {args.prefix}NN.jsonl in {work}; nothing to reshard")
        return 0

    keys = manifest_keys(Path(args.manifest))
    buckets: dict[int, list[str]] = {i: [] for i in range(args.shards)}
    seen: set[str] = set()
    total = repeats = unknown = 0
    for f in files:
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                total += 1
                rid = json.loads(line).get("id")
                if rid is None or rid in seen:
                    repeats += 1
                    continue
                seen.add(rid)
                parent, index = keys.get(rid, (None, None))
                if index is None:
                    # not in this manifest: keep it where the old split had it rather
                    # than guess, so no finished work is thrown away
                    unknown += 1
                    index = zlib.crc32(rid.encode())
                buckets[shard_of(parent, index, args.shards)].append(
                    line if line.endswith("\n") else line + "\n")

    moved = sum(len(v) for v in buckets.values())
    print(f"{total} row(s) in {len(files)} file(s) -> {moved} unique across {args.shards} "
          f"shard(s); {repeats} repeat(s) dropped, {unknown} not in the manifest, "
          f"{len(keys) - moved} of the manifest still to do")
    for i in sorted(buckets):
        print(f"  {args.prefix}{i:02d}.jsonl: {len(buckets[i])}")
    if args.dry_run:
        return 0

    for i, rows in buckets.items():
        dest = work / f"{args.prefix}{i:02d}.jsonl"
        tmp = dest.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as fh:
            fh.writelines(rows)
        os.replace(tmp, dest)
    for f in files:
        if int(f.stem[len(args.prefix):]) >= args.shards:
            f.unlink()
            print(f"  removed {f.name} (above the new shard count)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
