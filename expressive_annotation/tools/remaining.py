#!/usr/bin/env python3
"""Build a manifest of the clips an annotation run has not reached yet.

Splitting a corpus across machines part-way through a run needs the work that is
actually left, not the whole manifest: handing every worker the full list means the
ones that start late redo what is already done. ``continuo-annotate --resume`` skips ids
present in *its own* output, which is the right behaviour for restarting one worker and
the wrong one for adding a second machine.

    python tools/remaining.py --manifest runs/x/manifest.jsonl \
        --done runs/x/work/annot00.jsonl runs/x/work/annot01.jsonl \
        --out runs/x/remaining.jsonl --relative-to runs/x

``--relative-to`` rewrites each ``wav_path`` relative to that directory, so the manifest
can be carried to a machine that mounts the audio somewhere else and run there with
``--audio-root``. Without it the paths stay as they are.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from continuo_expressive.jsonl import done_ids, read_jsonl


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="remaining",
        description="Manifest rows not present in any of the given annotation outputs.")
    p.add_argument("--manifest", required=True, help="the full manifest")
    p.add_argument("--done", nargs="*", default=[],
                   help="annotation JSONLs already produced; their ids are excluded")
    p.add_argument("--out", required=True, help="where to write the remaining rows")
    p.add_argument("--relative-to", default="",
                   help="rewrite wav_path relative to this directory, for a manifest "
                        "that has to run on another machine")
    args = p.parse_args(argv)

    seen: set[str] = set()
    for path in args.done:
        found = done_ids(path)
        print(f"  {path}: {len(found)} done", flush=True)
        seen |= found

    root = Path(args.relative_to).expanduser().resolve() if args.relative_to else None
    total = kept = rebased = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as sink:
        for row in read_jsonl(args.manifest, required=("id",)):
            total += 1
            if row["id"] in seen:
                continue
            if root is not None and row.get("wav_path", "").startswith("/"):
                try:
                    row["wav_path"] = str(Path(row["wav_path"]).resolve().relative_to(root))
                    rebased += 1
                except ValueError:
                    # outside the root: leave it absolute rather than invent a path
                    pass
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
        sink.flush()

    print(f"\n{kept} of {total} rows remain -> {args.out}")
    if root is not None:
        print(f"{rebased} wav_path(s) rewritten relative to {root}; "
              "run the workers with --audio-root pointing at their own copy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
