#!/usr/bin/env python3
"""Join expressive stage outputs by id without losing input provenance."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from pathlib import Path


def rows(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict) or not row.get("id"):
                    raise ValueError(f"invalid row at {path.name}:{line_no}")
                yield row


def index_stage(db: sqlite3.Connection, name: str, path: Path) -> int:
    db.execute(f"CREATE TABLE {name} (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    count = 0
    try:
        with db:
            for row in rows(path):
                db.execute(f"INSERT INTO {name} VALUES (?, ?)",
                           (row["id"], json.dumps(row, ensure_ascii=False)))
                count += 1
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"duplicate id in {path.name}") from exc
    return count


def merge(manifest: Path, out: Path, annotations: Path | None = None,
          nv: Path | None = None, allow_partial: bool = False) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    db_fd, db_temp = tempfile.mkstemp(prefix=".merge-", suffix=".sqlite", dir=out.parent)
    os.close(db_fd)
    fd, temp = tempfile.mkstemp(prefix=".final-", suffix=".jsonl", dir=out.parent)
    count = 0
    try:
        with sqlite3.connect(db_temp) as db, os.fdopen(fd, "w", encoding="utf-8") as sink:
            db.execute("CREATE TABLE seen (id TEXT PRIMARY KEY)")
            stage_counts = {}
            matched = {"annot": 0, "nonverbal": 0}
            if annotations:
                stage_counts["annot"] = index_stage(db, "annot", annotations)
            if nv:
                stage_counts["nonverbal"] = index_stage(db, "nonverbal", nv)
            for source in rows(manifest):
                ident = source["id"]
                try:
                    db.execute("INSERT INTO seen VALUES (?)", (ident,))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"duplicate id in manifest: {ident}") from exc
                item = dict(source)
                if annotations:
                    found = db.execute("SELECT payload FROM annot WHERE id=?", (ident,)).fetchone()
                    result = json.loads(found[0]) if found else None
                    if result is None and not allow_partial:
                        raise ValueError(f"missing tag result: {ident}")
                    if result:
                        matched["annot"] += 1
                        item.update({k: v for k, v in result.items()
                                     if k not in {"id", "wav_path", "_path", "_tar"}})
                if nv:
                    found = db.execute("SELECT payload FROM nonverbal WHERE id=?", (ident,)).fetchone()
                    result = json.loads(found[0]) if found else None
                    if result is None and not allow_partial:
                        raise ValueError(f"missing NV result: {ident}")
                    if result:
                        matched["nonverbal"] += 1
                        item.update({k: v for k, v in result.items()
                                     if k.startswith("nv_") or k in {"n_nv", "asr_ratio", "suspect"}})
                        if "timestamp" in result:
                            item["nv_timestamp"] = result["timestamp"]
                sink.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
            for stage, total in stage_counts.items():
                if total != matched[stage]:
                    raise ValueError(f"{stage} output contains {total - matched[stage]} unknown id(s)")
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temp, out)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
    finally:
        Path(db_temp).unlink(missing_ok=True)
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--annotations", type=Path)
    ap.add_argument("--nv", type=Path)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    print(f"{merge(args.manifest, args.out, args.annotations, args.nv, args.allow_partial)} rows -> {args.out}")


if __name__ == "__main__":
    main()
