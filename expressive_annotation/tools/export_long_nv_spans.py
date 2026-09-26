#!/usr/bin/env python3
"""Export one listenable audio clip per tagged span in a long-audio NV JSONL.

The long corpus stores whole recordings in tar members and places NV events on their
timeline with ``nv_spans[].start/end``.  This tool reads each source member once,
cuts all matching spans in one ffmpeg invocation, and writes a simple JSON manifest,
a rich JSONL manifest, a TSV, and an M3U playlist.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path


SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".long-nv-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def build_plan(source: Path, output_root: Path, event: str, batch_size: int,
               limit: int = 0) -> list[dict]:
    plan: list[dict] = []
    with source.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            parent = str(row.get("id", ""))
            tar = str(row.get("source_tar", ""))
            member = str(row.get("source_member", ""))
            if not SAFE.fullmatch(parent) or Path(tar).name != tar or Path(member).name != member:
                raise ValueError(f"{source}:{line_no}: unsafe id/tar/member")
            hits = [span for span in (row.get("nv_spans") or [])
                    if event in (span.get("tags") or [])]
            for local_index, span in enumerate(hits, 1):
                start, end = float(span["start"]), float(span["end"])
                if not 0 <= start < end:
                    raise ValueError(f"{source}:{line_no}: invalid span {start}..{end}")
                sequence = len(plan) + 1
                batch = f"batch_{(sequence - 1) // batch_size + 1:04d}"
                clip_id = f"{parent}_{event.lower()}{local_index:02d}"
                filename = f"{sequence:07d}_{clip_id}.m4a"
                destination = (output_root / batch / filename).resolve()
                plan.append({
                    "sequence": sequence,
                    "id": clip_id,
                    "parent_id": parent,
                    "source_tar": tar,
                    "source_member": member,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "lang": row.get("lang"),
                    "tags": span.get("tags") or [],
                    "transcript": span.get("text") or "",
                    "audio_path": str(destination),
                    "relative_path": str(Path(batch) / filename),
                })
                if limit and len(plan) >= limit:
                    return plan
    return plan


def load_index(tar_path: Path, members: set[str]) -> dict[str, tuple[int, int]]:
    out = {}
    with Path(str(tar_path) + ".idx").open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3 and parts[0] in members:
                out[parts[0]] = int(parts[1]), int(parts[2])
    missing = members - out.keys()
    if missing:
        raise ValueError(f"{tar_path}: {len(missing)} member(s) absent from index: "
                         f"{sorted(missing)[:3]}")
    return out


def encode_member(data: bytes, suffix: str, jobs: list[dict]) -> tuple[int, int]:
    pending = [job for job in jobs if not Path(job["audio_path"]).exists()]
    if not pending:
        return 0, len(jobs)
    source_name = ""
    temporary_outputs: list[tuple[Path, Path]] = []
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as source:
            source.write(data)
            source_name = source.name
        split = "".join(f"[s{i}]" for i in range(len(pending)))
        if len(pending) == 1:
            filters = [f"[0:a]atrim=start={pending[0]['start']:.6f}:"
                       f"end={pending[0]['end']:.6f},asetpts=PTS-STARTPTS[o0]"]
        else:
            filters = [f"[0:a]asplit={len(pending)}{split}"]
            filters.extend(
                f"[s{i}]atrim=start={job['start']:.6f}:end={job['end']:.6f},"
                f"asetpts=PTS-STARTPTS[o{i}]"
                for i, job in enumerate(pending))
        command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", source_name,
                   "-filter_complex", ";".join(filters)]
        for i, job in enumerate(pending):
            destination = Path(job["audio_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(
                f".{destination.stem}.{os.getpid()}.{threading.get_ident()}.partial.m4a")
            temporary_outputs.append((temporary, destination))
            command.extend(["-map", f"[o{i}]", "-vn", "-map_metadata", "-1",
                            "-c:a", "aac", "-b:a", "64k", "-ar", "16000", "-ac", "1",
                            "-threads", "1", "-movflags", "+faststart", str(temporary)])
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                text=True)
        if result.returncode:
            raise RuntimeError(f"ffmpeg exited {result.returncode}: {result.stderr[:1000]}")
        for temporary, destination in temporary_outputs:
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise RuntimeError(f"ffmpeg did not create {temporary}")
            os.replace(temporary, destination)
        return len(pending), len(jobs) - len(pending)
    finally:
        if source_name and os.path.exists(source_name):
            os.unlink(source_name)
        for temporary, _ in temporary_outputs:
            if temporary.exists():
                temporary.unlink()


def export_tar(tar_name: str, members: dict[str, list[dict]], tar_root: Path) -> dict:
    tar_path = tar_root / tar_name
    index = load_index(tar_path, set(members))
    created = reused = source_bytes = 0
    with tar_path.open("rb") as source:
        for member in sorted(members, key=lambda name: index[name][0]):
            offset, size = index[member]
            source.seek(offset)
            data = source.read(size)
            if len(data) != size:
                raise ValueError(f"{tar_path}:{member}: short read {len(data)}/{size}")
            new, old = encode_member(data, Path(member).suffix, members[member])
            created += new
            reused += old
            source_bytes += size
    return {"created": created, "reused": reused, "source_bytes": source_bytes,
            "source_members": len(members), "source_archives": 1}


def write_manifests(plan: list[dict], simple_out: Path, manifest_out: Path,
                    output_root: Path) -> None:
    simple = [{"audio_path": row["audio_path"], "transcript": row["transcript"]}
              for row in plan]
    atomic_text(simple_out, json.dumps(simple, ensure_ascii=False, indent=2) + "\n")
    atomic_text(manifest_out, "".join(json.dumps(row, ensure_ascii=False) + "\n"
                                      for row in plan))
    playlist = ["#EXTM3U"]
    for row in plan:
        playlist.extend([f"#EXTINF:{row['duration']:.3f},{row['id']}", row["relative_path"]])
    atomic_text(output_root / "playlist.m3u8", "\n".join(playlist) + "\n")

    table_path = output_root / "tracks.tsv"
    fd, temporary = tempfile.mkstemp(dir=output_root, prefix=".long-nv-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["sequence", "file", "id", "parent_id", "start", "end",
                             "duration", "lang", "tags", "transcript"])
            for row in plan:
                writer.writerow([row["sequence"], row["relative_path"], row["id"],
                                 row["parent_id"], row["start"], row["end"],
                                 round(row["duration"], 3), row["lang"] or "",
                                 ",".join(row["tags"]), row["transcript"]])
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, table_path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--simple-out", type=Path, required=True)
    ap.add_argument("--manifest-out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--tar-root", type=Path, required=True)
    ap.add_argument("--event", default="Cough")
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="probe only the first N spans")
    args = ap.parse_args(argv)
    if args.batch_size < 1 or args.workers < 1 or args.limit < 0:
        ap.error("batch size and workers must be positive; limit must be non-negative")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    plan = build_plan(args.input, output_root, args.event, args.batch_size, args.limit)
    if not plan:
        raise SystemExit(f"no {args.event} spans in {args.input}")
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in plan:
        grouped[row["source_tar"]][row["source_member"]].append(row)

    totals = {"created": 0, "reused": 0, "source_bytes": 0,
              "source_members": 0, "source_archives": 0}
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(export_tar, tar, members, args.tar_root.resolve())
                   for tar, members in grouped.items()]
        for future in concurrent.futures.as_completed(pending):
            result = future.result()
            for key in totals:
                totals[key] += result[key]
            completed += result["created"] + result["reused"]
            if completed % 500 < result["created"] + result["reused"]:
                print(f"exported {completed}/{len(plan)} span(s)", flush=True)

    if totals["created"] + totals["reused"] != len(plan):
        raise RuntimeError(f"output count mismatch: {totals} for {len(plan)} spans")
    write_manifests(plan, args.simple_out, args.manifest_out, output_root)
    audio_bytes = sum(Path(row["audio_path"]).stat().st_size for row in plan)
    report = {
        "status": "complete",
        "event": args.event,
        "input": str(args.input.resolve()),
        "output_root": str(output_root),
        "simple_manifest": str(args.simple_out.resolve()),
        "rich_manifest": str(args.manifest_out.resolve()),
        "spans": len(plan),
        "event_tags": sum(row["tags"].count(args.event) for row in plan),
        "duration_hours": sum(row["duration"] for row in plan) / 3600,
        "audio_bytes": audio_bytes,
        "codec": "AAC-LC, mono, 16 kHz, 64 kb/s",
        "elapsed_seconds": round(time.monotonic() - started, 1),
        **totals,
    }
    atomic_text(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
