#!/usr/bin/env python3
"""Run independent NV verification shards and merge only complete results."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import zlib
from run_emotion_parallel import read_scores, collect, write_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nv', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--python', required=True)
    args, verify_args = parser.parse_known_args()
    cards = args.gpus.split(',')
    if not cards or any(not c for c in cards) or len(set(cards)) != len(cards):
        parser.error('GPU list must be nonempty and unique')
    rows = read_scores(args.nv)
    work = args.out.parent / 'verify-work' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    work.mkdir(parents=True)
    shards = [[] for _ in cards]
    for row in rows.values():
        index = zlib.crc32(str(row.get('parent_id') or row['id']).encode()) % len(cards)
        shards[index].append(row)
    jobs, outputs = [], []
    try:
        for i, gpu in enumerate(cards):
            source = work / f'input{i:02d}.jsonl'
            output = work / f'verified{i:02d}.jsonl'
            write_atomic(source, shards[i])
            command = [args.python, str(Path(__file__).with_name('nv_verify.py').resolve()),
                       '--nv', str(source.resolve()), '--out', str(output.resolve()), *verify_args]
            with (work / f'verify{i:02d}.log').open('wb') as log:
                job = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                                       stdout=log, stderr=subprocess.STDOUT)
            jobs.append(job)
            outputs.append(output)
            print(f'NV verify GPU {gpu}: PID {job.pid}, {len(shards[i])} rows, log {log.name}', flush=True)
        failed = []
        for job in jobs:
            rc = job.wait()
            if rc:
                failed.append((job.pid, rc))
        if failed:
            raise RuntimeError(f'NV verification workers failed: {failed}')
    finally:
        for job in jobs:
            if job.poll() is None:
                job.terminate()
        for job in jobs:
            if job.poll() is None:
                job.wait()
    verified = collect(outputs, rows)
    if verified.keys() != rows.keys():
        raise ValueError(f'NV verification coverage incomplete: {len(verified)}/{len(rows)}')
    for row in verified.values():
        if 'nv_verified' not in row or 'nv_rejected' not in row:
            raise ValueError('Worker output is not verified NV data')
    write_atomic(args.out, (verified[sid] for sid in rows))
    print(f'NV verification complete: {len(verified)}/{len(rows)} unique IDs, original order preserved', flush=True)


if __name__ == '__main__':
    main()
