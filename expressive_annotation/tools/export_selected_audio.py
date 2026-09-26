#!/usr/bin/env python3
"""Restore a selected simple-JSON audio set at its recorded export paths.

The simple export stores absolute ``audio_path`` values whose seven-digit sequence and
``batch_XXXX`` directory came from a larger export.  Re-exporting only a subset with the
bulk exporter renumbers it, so this tool instead preserves every requested path exactly
and copies the original encoded member bytes from the corpus tar files.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path


EXPORT_NAME = re.compile(r"^[0-9]{7}_(.+)(\.[A-Za-z0-9]+)$")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clip_id(row: dict) -> str:
    if row.get("id"):
        return str(row["id"])
    raw = row.get("audio_path")
    match = EXPORT_NAME.fullmatch(Path(raw or "").name)
    if not match:
        raise ValueError(f"cannot recover clip id from audio_path {raw!r}")
    return match.group(1)


def load_selection(path: Path, output_root: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a JSON array")
    rows, seen_ids, seen_paths = [], set(), set()
    root = output_root.resolve()
    for position, row in enumerate(value, 1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: item {position} is not an object")
        rid = clip_id(row)
        raw = row.get("audio_path")
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"{path}: item {position} has no audio_path")
        destination = Path(raw).expanduser().resolve(strict=False)
        if root != destination and root not in destination.parents:
            raise ValueError(
                f"{path}: item {position} points outside --output-root: {destination}")
        if rid in seen_ids or destination in seen_paths:
            raise ValueError(f"{path}: duplicate id or audio_path at item {position}")
        seen_ids.add(rid)
        seen_paths.add(destination)
        rows.append({"id": rid, "destination": destination})
    return rows


def load_sources(path: Path, wanted: set[str]) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rid = str(row.get("id", ""))
            if rid not in wanted:
                continue
            if rid in out:
                raise ValueError(f"{path}:{lineno}: duplicate id {rid!r}")
            missing = [key for key in ("source_tar", "source_member", "tar_offset", "tar_size")
                       if row.get(key) is None]
            if missing:
                raise ValueError(f"{path}:{lineno}: {rid}: missing {missing}")
            out[rid] = row
    missing = wanted - out.keys()
    if missing:
        raise ValueError(f"{path}: {len(missing)} selected id(s) are missing: "
                         f"{sorted(missing)[:5]}")
    return out


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".audio-export-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def extract_tar(tar_name: str, jobs: list[dict], tar_root: Path) -> dict:
    tar_path = tar_root / tar_name
    if not tar_path.is_file():
        raise ValueError(f"missing source tar: {tar_path}")
    created = reused = total_bytes = 0
    with tar_path.open("rb") as source:
        for job in sorted(jobs, key=lambda item: int(item["tar_offset"])):
            offset, size = int(job["tar_offset"]), int(job["tar_size"])
            if offset < 0 or size <= 0:
                raise ValueError(f"{tar_name}:{job['id']}: invalid offset/size")
            source.seek(offset)
            data = source.read(size)
            if len(data) != size:
                raise ValueError(f"{tar_name}:{job['id']}: short read {len(data)}/{size}")
            suffix = Path(job["source_member"]).suffix.lower()
            destination: Path = job["destination"]
            if destination.suffix.lower() != suffix:
                raise ValueError(f"{job['id']}: source/destination suffix differs")
            if suffix == ".m4a" and data[4:8] != b"ftyp":
                raise ValueError(f"{tar_name}:{job['id']}: invalid M4A header")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                current = destination.read_bytes()
                if current != data:
                    raise ValueError(f"existing audio differs from source: {destination}")
                reused += 1
            else:
                temporary = destination.with_name(
                    destination.name + f".partial-{os.getpid()}")
                with temporary.open("xb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temporary, destination)
                created += 1
            total_bytes += size
    return {"created": created, "reused": reused, "bytes": total_bytes,
            "archives": 1, "files": len(jobs)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--selection", type=Path, required=True,
                    help="simple JSON array containing the desired audio_path values")
    ap.add_argument("--sources", type=Path, required=True,
                    help="JSONL with source_tar/member and tar_offset/tar_size")
    ap.add_argument("--output-root", type=Path, required=True,
                    help="every recorded audio_path must be under this directory")
    ap.add_argument("--tar-root", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args(argv)
    if args.workers < 1:
        ap.error("--workers must be positive")

    selected = load_selection(args.selection, args.output_root)
    sources = load_sources(args.sources, {row["id"] for row in selected})
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        source = sources[row["id"]]
        grouped[str(source["source_tar"])].append({**source, **row})

    totals = {"created": 0, "reused": 0, "bytes": 0, "archives": 0, "files": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(extract_tar, tar, jobs, args.tar_root.resolve())
                   for tar, jobs in grouped.items()]
        for future in concurrent.futures.as_completed(pending):
            result = future.result()
            for key in totals:
                totals[key] += result[key]
    report = {
        "status": "complete",
        "selection": str(args.selection.resolve()),
        "sources": str(args.sources.resolve()),
        "output_root": str(args.output_root.resolve()),
        "tar_root": str(args.tar_root.resolve()),
        **totals,
    }
    if totals["files"] != len(selected) or totals["created"] + totals["reused"] != len(selected):
        raise RuntimeError(f"export count mismatch: {report}")
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
