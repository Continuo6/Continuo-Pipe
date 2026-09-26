#!/usr/bin/env python3
"""Recompute pitch for annotations written while penn's checkpoint was unreachable.

``features/pitch.measure`` catches every exception and leaves ``pitch_hz``/``pitch``
null, because one unreadable clip should not stop a corpus run. A missing checkpoint
can also produce nulls for every clip, so this repair tool can recompute pitch later.

Re-running the whole annotate pass would recompute attributes that are already
present. This reads the rows back, decodes their audio from the
tars again, and fills in only the two pitch fields. Everything else is copied through
byte for byte.

Pitch depends on ``gender`` (the band edges differ), and gender is already in the row,
so the patched value is exactly what the pass would have written.

    CONTINUO_EXPRESSIVE_TAR_DIR=corpus python tools/patch_pitch.py --annot runs/x/work/annot00.jsonl --gpu 0

Resumable: rows are appended to ``<annot>.pitched`` as they are done and the file only
replaces the input once its ids match the input's, in order. Killing this mid-run costs
the current batch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from continuo_expressive.features import pitch as pitch_feat      # noqa: E402
from continuo_expressive.tarsource import load_row                # noqa: E402


def batched(rows, n):
    buf = []
    for r in rows:
        buf.append(r)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="patch_pitch")
    p.add_argument("--annot", required=True, help="annotation JSONL to patch in place")
    p.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8, help="audio-decoding threads")
    p.add_argument("--all", action="store_true",
                   help="recompute every row, not only the ones with a null pitch")
    args = p.parse_args(argv)

    src = Path(args.annot)
    dst = Path(str(src) + ".pitched")
    rows = [json.loads(line) for line in open(src, "rb") if line.strip()]
    done = 0
    if dst.exists():
        done = sum(1 for line in open(dst, "rb") if line.strip())
        if done > len(rows):
            print(f"{dst} is longer than {src}; refusing", file=sys.stderr)
            return 1
    print(f"{src.name}: {len(rows)} row(s), resuming at {done}", flush=True)

    gpu = None if args.gpu < 0 else args.gpu
    pool = ThreadPoolExecutor(max_workers=args.workers)
    filled = skipped = failed = 0
    t0 = time.time()

    with open(dst, "ab") as out:
        for chunk in batched(rows[done:], args.batch_size):
            need = [r for r in chunk
                    if args.all or r.get("pitch_hz") is None]
            if need:
                wavs = list(pool.map(_safe_load, need))
                res = pitch_feat.measure_batch(wavs, [r.get("gender") for r in need],
                                               gpu=gpu)
                for r, got in zip(need, res):
                    r["pitch_hz"] = got["pitch_hz"]
                    r["pitch"] = got["pitch"]
                    if got["pitch_hz"] is None:
                        failed += 1
                    else:
                        filled += 1
            skipped += len(chunk) - len(need)
            for r in chunk:
                clean = {k: v for k, v in r.items() if not k.startswith("_")}
                out.write(json.dumps(clean, ensure_ascii=False).encode() + b"\n")
            out.flush()
            n = done + filled + failed + skipped
            if n % (args.batch_size * 50) < args.batch_size:
                rate = (n - done) / max(time.time() - t0, 1e-9)
                print(f"  {n}/{len(rows)} ({rate:.1f}/s)", flush=True)

    # only replace once the patched file says the same clips in the same order
    got = [json.loads(line)["id"] for line in open(dst, "rb") if line.strip()]
    want = [r["id"] for r in rows]
    if got != want:
        print(f"REFUSED: {dst} does not match {src} row for row", file=sys.stderr)
        return 1
    os.replace(dst, src)
    print(f"{src.name}: filled {filled}, still null {failed}, untouched {skipped} "
          f"in {time.time() - t0:.0f}s", flush=True)
    return 0


def _safe_load(row):
    """Decode a row's audio, leaving the row as it was found.

    ``tarsource.load_row`` caches the resolved tar on the row it is given (``_tar``, a
    Path), which is fine for a pass that writes selected fields and fatal for one that
    writes the row back: json cannot serialise it. Hand it a shallow copy.
    """
    try:
        return load_row(dict(row))
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
