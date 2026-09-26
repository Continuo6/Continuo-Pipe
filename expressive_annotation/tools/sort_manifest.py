#!/usr/bin/env python3
"""Order a manifest by clip duration, so a batch is not mostly padding.

Every head runs on a padded batch, and the pad is to the batch's longest clip.
Sorting by duration reduces wasted padding when clip lengths vary.

Order matters twice over, so this has to run **before** sharding: `continuo-annotate --shard
i/n` claims rows by line number, and every n-th row of a sorted file is itself sorted, so
each worker's own batches are homogeneous too.

Nothing else about the rows changes, and `--resume` keys on id, so a manifest sorted
after a partial run still resumes correctly.

    python tools/sort_manifest.py --in runs/x/manifest.jsonl --out runs/x/manifest.jsonl

Reordering does change what shares a batch, which matters for exactly one head: the
voxprofile age head is batch-size dependent when `--batch-age-head` is enabled.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sort_manifest")
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", dest="dst", default="", help="default: in place")
    p.add_argument("--key", default="duration", help="numeric field to sort on")
    p.add_argument("--descending", action="store_true")
    args = p.parse_args(argv)
    dst = args.dst or args.src

    rows, missing = [], 0
    for n, line in enumerate(open(args.src, "rb"), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"{args.src}:{n}: invalid JSON ({e.msg})", file=sys.stderr)
            return 1
        if not isinstance(d.get(args.key), (int, float)):
            missing += 1
        rows.append((d.get(args.key) if isinstance(d.get(args.key), (int, float))
                     else float("inf"), line))
    if missing:
        print(f"[warn] {missing} row(s) have no numeric {args.key!r}; they sort last",
              file=sys.stderr)

    rows.sort(key=lambda t: t[0], reverse=args.descending)

    # atomic: a reader following this file never sees a half-written one
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(dst)) or ".",
                               suffix=".sorting")
    try:
        with os.fdopen(fd, "wb") as out:
            for _, line in rows:
                out.write(line if line.endswith(b"\n") else line + b"\n")
        os.replace(tmp, dst)
    except BaseException:
        os.unlink(tmp)
        raise

    lo = rows[0][0] if rows else 0
    hi = rows[-1][0] if rows else 0
    print(f"{len(rows)} row(s) sorted by {args.key} "
          f"({lo:.2f} -> {hi:.2f}) -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
