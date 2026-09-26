#!/usr/bin/env python3
"""Build run_long_pipeline-compatible inputs from the verified Continuo test set."""
from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def read_cases(root: Path, max_seconds: float) -> list[dict]:
    rows: list[dict] = []
    for lang in ("zh", "en"):
        path = root / f"verified_test_cases_{lang}.jsonl"
        with path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if float(row["duration"]) <= max_seconds:
                    rows.append(row)
    return sorted(rows, key=lambda row: (row["source_track"], row["language"], row["sid"]))


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    archive.addfile(info, io.BytesIO(data))


def convert_audio(source: Path, destination: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-map_metadata", "-1",
         "-c:a", "alac", str(destination)],
        check=True,
    )


def relative_turns(row: dict) -> list[dict]:
    duration = float(row["duration"])
    turns = []
    for turn in row["turns"]:
        start = max(0.0, float(turn["start_rel"]))
        end = min(duration, float(turn["end_rel"]))
        if end <= start:
            continue
        turns.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "text": turn.get("text") or "",
            "tag": turn.get("tag"),
        })
    return turns


def write_view(rows: list[dict], out_root: Path, view: str) -> None:
    marker = "long" if view == "long" else "dlg"
    input_dir = out_root / "input" / view
    meta_dir = out_root / "input" / "metainfo" / view
    input_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    tar_path = input_dir / f"continuo-testset-{view}.tar"
    meta_path = meta_dir / tar_path.name

    with tempfile.TemporaryDirectory(prefix=f"continuo-testset-{view}-") as tmp:
        tmp_dir = Path(tmp)
        with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as audio_tar, \
                tarfile.open(meta_path, "w", format=tarfile.PAX_FORMAT) as meta_tar:
            for row in rows:
                case_id = f"{row['language']}_{row['sid']}"
                stem = f"{case_id}_{marker}_00000"
                audio_name = f"{stem}.m4a"
                json_name = f"{stem}.json"
                encoded = tmp_dir / audio_name
                convert_audio(Path(row["gt_audio_path"]), encoded)
                info = audio_tar.gettarinfo(str(encoded), arcname=audio_name)
                info.mtime = 0
                with encoded.open("rb") as source:
                    audio_tar.addfile(info, source)

                turns = relative_turns(row)
                short = [{
                    "rel_start": turn["start"],
                    "rel_end": turn["end"],
                    "text": turn["text"],
                    "language": row["language"],
                    **({"speaker": turn["tag"]} if view == "dialogue" else {}),
                } for turn in turns]
                local = {
                    "id": case_id,
                    "duration": float(row["duration"]),
                    "languages": [row["language"]],
                    "short": short,
                }
                members = [{
                    "start": turn["start"],
                    "end": turn["end"],
                    "text": turn["text"],
                    "lang_hint": row["language"],
                    **({"speaker": turn["tag"], "tag": turn["tag"]}
                       if view == "dialogue" else {}),
                } for turn in turns]
                meta = {
                    "is_valid": True,
                    "duration": float(row["duration"]),
                    "languages": [row["language"]],
                    "members": members,
                }
                add_bytes(audio_tar, json_name,
                          json.dumps(local, ensure_ascii=False).encode("utf-8"))
                add_bytes(meta_tar, json_name,
                          json.dumps(meta, ensure_ascii=False).encode("utf-8"))

    with tarfile.open(tar_path) as archive, Path(str(tar_path) + ".idx").open(
            "w", encoding="utf-8") as index:
        for member in archive.getmembers():
            if member.isfile():
                index.write(f"{member.name}\t{member.offset_data}\t{member.size}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out_dir.exists():
        if not args.force:
            raise SystemExit(f"{args.out_dir} already exists; pass --force to replace it")
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)

    rows = read_cases(args.testset, args.max_seconds)
    with (args.out_dir / "selection.jsonl").open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    for view in ("long", "dialogue"):
        write_view([row for row in rows if row["source_track"] == view], args.out_dir, view)

    counts = {view: sum(row["source_track"] == view for row in rows)
              for view in ("long", "dialogue")}
    print(f"wrote {len(rows)} case(s): {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
