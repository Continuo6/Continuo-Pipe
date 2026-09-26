#!/usr/bin/env python3
"""Convert completed Continuo records to a portable expressive JSONL manifest."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Iterator


SUFFIX = {"short": ".json", "long": ".long.json", "dialogue": ".dialogue.json"}


def _record_files(root: Path, tracks: list[str]) -> Iterator[tuple[str, Path, Path]]:
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("_"))
        parent = Path(directory)
        for track in tracks:
            name = parent.name + SUFFIX[track]
            if name in files:
                yield track, parent, parent / name


def _audio(row: dict, parent: Path) -> tuple[Path, int | None, int | None, int | None]:
    carrier = row.get("carrier_path")
    raw = carrier or row.get("audio_path")
    if not raw:
        raise ValueError("completed row has no carrier_path or audio_path")
    path = Path(raw)
    if not path.is_absolute():
        path = parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"audio file does not exist: {path}")
    if not carrier:
        return path, None, None, row.get("sample_rate")
    sr = row.get("sample_rate")
    if not isinstance(sr, int) or sr <= 0:
        raise ValueError("carrier row needs a positive integer sample_rate")
    start, end = row.get("carrier_start_samples"), row.get("carrier_end_samples")
    if not isinstance(start, int) or not isinstance(end, int) or end <= start:
        raise ValueError("carrier row needs valid integer sample bounds")
    return path, start, end, sr


def _units(track: str, row: dict) -> list[dict]:
    if track == "short":
        return [row]
    if track == "long":
        return row.get("members") or [row]
    return row.get("turns") or row.get("segments") or []


def convert_row(track: str, sid: str, parent: Path, row: dict) -> Iterator[dict]:
    index = row.get("index")
    if index is None:
        raise ValueError("completed row has no index")
    path, carrier_start, carrier_end, sr = _audio(row, parent)
    units = _units(track, row)
    for n, unit in enumerate(units):
        start = float(unit.get("start", row.get("start", 0)))
        end = float(unit.get("end", row.get("end", 0)))
        if end <= start:
            raise ValueError(f"invalid segment interval in {sid}_{index}")
        if track == "short":
            ident = f"{sid}_{index}"
            a, b = carrier_start, carrier_end
        else:
            ident = f"{sid}_{index}_{'m' if track == 'long' else 't'}{n:04d}"
            if carrier_start is None or carrier_end is None:
                raise ValueError(f"{track} subsegments require a shared carrier: {sid}_{index}")
            a, b = round(start * sr), round(end * sr)
            a = max(carrier_start, a)
            b = min(carrier_end, b)
            if b <= a:
                raise ValueError(f"subsegment outside carrier window: {ident}")
        item = {
            "id": ident,
            "wav_path": str(path),
            "txt": unit.get("text") or unit.get("phase2_text") or "",
            "lang": unit.get("language") or row.get("language"),
            "speaker": unit.get("speaker", row.get("speaker")),
            "duration": round(end - start, 6),
            "source_track": track,
            "source_recording_id": sid,
            "source_index": index,
            "start": start,
            "end": end,
        }
        if a is not None and b is not None:
            item.update(carrier_start_samples=int(a), carrier_end_samples=int(b),
                        sample_rate=sr)
        yield item


def export(root: Path, out: Path, tracks: list[str]) -> int:
    if not root.is_dir():
        raise ValueError(f"processed root does not exist: {root}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".manifest-", suffix=".jsonl", dir=out.parent)
    count = 0
    seen: set[str] = set()
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as sink:
            for track, parent, path in _record_files(root, tracks):
                rows = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(rows, list):
                    raise ValueError(f"expected a JSON array: {path}")
                for row in rows:
                    for item in convert_row(track, parent.name, parent, row):
                        if item["id"] in seen:
                            raise ValueError(f"duplicate expressive id: {item['id']}")
                        seen.add(item["id"])
                        sink.write(json.dumps(item, ensure_ascii=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temp, out)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
    if not seen:
        raise ValueError(f"no completed rows found under {root}")
    return len(seen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tracks", default="short", help="comma-separated: short,long,dialogue")
    args = ap.parse_args()
    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    if not tracks or len(tracks) != len(set(tracks)) or any(t not in SUFFIX for t in tracks):
        ap.error("--tracks must list distinct values from short,long,dialogue")
    count = export(args.processed_root, args.out, tracks)
    print(f"{count} expressive units -> {args.out}")


if __name__ == "__main__":
    main()
