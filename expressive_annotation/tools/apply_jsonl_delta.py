#!/usr/bin/env python3
"""Apply a verified replace/remove JSONL delta and optionally commit atomically."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--patch-sha256", required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    if sha256(args.patch) != args.patch_sha256:
        raise SystemExit("patch SHA256 mismatch")
    with gzip.open(args.patch, "rt", encoding="utf-8") as src:
        delta = json.load(src)
    if delta.get("format") != "jsonl-id-replace-remove-v1":
        raise SystemExit("unsupported patch format")
    if sha256(args.base) != delta["base_sha256"]:
        raise SystemExit("base SHA256 mismatch")

    replacements = delta["replacements"]
    removed = set(delta["removed_ids"])
    if removed & set(replacements):
        raise SystemExit("an id cannot be both replaced and removed")

    fd, tmp_name = tempfile.mkstemp(
        dir=args.base.parent, prefix=f".{args.base.name}.delta-", suffix=".tmp")
    tmp = Path(tmp_name)
    seen: set[str] = set()
    input_rows = output_rows = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as dst, args.base.open(
                encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                input_rows += 1
                row_id = str(json.loads(line)["id"])
                if row_id in seen:
                    raise RuntimeError(f"duplicate base id: {row_id}")
                seen.add(row_id)
                if row_id in removed:
                    continue
                replacement = replacements.get(row_id)
                if replacement is not None:
                    line = json.dumps(replacement, ensure_ascii=False) + "\n"
                dst.write(line if line.endswith("\n") else line + "\n")
                output_rows += 1
            dst.flush()
            os.fsync(dst.fileno())

        missing = (set(replacements) | removed) - seen
        if missing:
            raise RuntimeError(f"{len(missing)} patch id(s) absent from base")
        if input_rows != delta["base_rows"]:
            raise RuntimeError(f"base rows {input_rows} != {delta['base_rows']}")
        if output_rows != delta["output_rows"]:
            raise RuntimeError(f"output rows {output_rows} != {delta['output_rows']}")
        output_sha = sha256(tmp)
        if output_sha != delta["output_sha256"]:
            raise RuntimeError(
                f"output SHA256 {output_sha} != {delta['output_sha256']}")

        if args.replace:
            os.replace(tmp, args.base)
            dir_fd = os.open(args.base.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        print(json.dumps({
            "status": "committed" if args.replace else "verified",
            "input_rows": input_rows,
            "output_rows": output_rows,
            "replacements": len(replacements),
            "removed": len(removed),
            "output_sha256": output_sha,
        }, indent=2))
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
