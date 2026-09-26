#!/usr/bin/env python3
"""Which tars a cutting run still has to do, for a supervisor that restarts it.

A crashed ``prepare_long.py`` has to resume from somewhere, and the manifest is the
wrong place to look: a tar holding nothing in ``--languages`` writes no rows, so
"tars absent from the manifest" never shrinks past those and a restart loop retries
them forever. ``prepare_long --tar-log`` records each tar as it finishes, whether or
not it produced anything; this subtracts that from the full list.

    python tools/pending_tars.py --tars-from all.txt --tar-log done.txt --out todo.txt

Prints the count and exits 0 when work remains, 1 when the list is empty — so a shell
supervisor can branch on it without parsing anything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def read_list(path: str) -> list[str]:
    if not path or not Path(path).is_file():
        return []
    return [ln.strip() for ln in Path(path).read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pending_tars",
        description="Tars in --tars-from that --tar-log does not record as finished.")
    p.add_argument("--tars-from", required=True, help="the full list for this run")
    p.add_argument("--tar-log", default="", help="what prepare_long has finished")
    p.add_argument("--out", default="", help="write the pending list here")
    args = p.parse_args(argv)

    # compare by basename: the same tar reached through a different mount or a relative
    # path is the same tar, and a supervisor that thinks otherwise re-cuts a whole shard
    done = {Path(x).name for x in read_list(args.tar_log)}
    todo = [x for x in read_list(args.tars_from) if Path(x).name not in done]

    if args.out:
        Path(args.out).write_text("".join(x + "\n" for x in todo))
    print(f"{len(todo)} tar(s) pending, {len(done)} finished", file=sys.stderr)
    return 0 if todo else 1


if __name__ == "__main__":
    sys.exit(main())
