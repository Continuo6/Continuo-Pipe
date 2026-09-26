#!/usr/bin/env python3
"""Run data annotation, then expressive annotation, with separate entry points."""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, help="audio folder for data annotation")
    ap.add_argument("--data-config", type=Path, default=ROOT / "data_annotation/examples/qwen3_asr.json")
    ap.add_argument("--data-stages", choices=("both", "phase1", "asr", "none"), default="both")
    ap.add_argument("--processed-root", type=Path, help="existing data annotation output tree")
    ap.add_argument("--expressive-manifest", type=Path,
                    help="use an independent manifest instead of converting data annotation output")
    ap.add_argument("--tracks", default="short", help="tracks to convert when no manifest is supplied")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--components", default="tags,caption,nv")
    ap.add_argument("--nv-mode", choices=("direct", "verified"), default="verified")
    ap.add_argument("--caption-backend", choices=("vllm", "transformers"), default="vllm")
    ap.add_argument("--phase1-python", type=Path)
    ap.add_argument("--asr-python", type=Path)
    ap.add_argument("--tags-python", type=Path)
    ap.add_argument("--caption-python", type=Path)
    ap.add_argument("--nv-python", type=Path)
    ap.add_argument("--phase1-gpus", default="")
    ap.add_argument("--asr-gpus", default="")
    ap.add_argument("--audio-root", type=Path)
    ap.add_argument("--tar-dir", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.data_stages != "none" and not args.input:
        ap.error("--input is required when data annotation runs")
    if not args.expressive_manifest and not (args.processed_root or args.input):
        ap.error("provide --processed-root or --input to build an expressive manifest")
    out_dir = args.out_dir.expanduser().resolve()
    input_dir = args.input.expanduser().resolve() if args.input else None
    processed = (args.processed_root.expanduser().resolve() if args.processed_root else
                 Path(str(input_dir).rstrip(os.sep) + "_processed") if input_dir else None)
    if args.data_stages != "none":
        command = [sys.executable, str(ROOT / "scripts/run_data_annotation.py"),
                   "--input", str(input_dir), "--config", str(args.data_config.expanduser().resolve()),
                   "--processed-root", str(processed), "--stages", args.data_stages]
        for flag in ("phase1-python", "asr-python"):
            value = getattr(args, flag.replace("-", "_"))
            if value:
                command += [f"--{flag}", str(value.expanduser().resolve())]
        for flag in ("phase1-gpus", "asr-gpus"):
            value = getattr(args, flag.replace("-", "_"))
            if value:
                command += [f"--{flag}", value]
        print(f"[data] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    if args.expressive_manifest:
        manifest = args.expressive_manifest.expanduser().resolve()
    else:
        manifest = out_dir / "data_manifest.jsonl"
        command = [sys.executable, str(ROOT / "scripts/export_data_manifest.py"),
                   "--processed-root", str(processed), "--out", str(manifest),
                   "--tracks", args.tracks]
        print(f"[bridge] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    command = [sys.executable, str(ROOT / "scripts/run_expressive_annotation.py"),
               "--manifest", str(manifest), "--out-dir", str(out_dir),
               "--components", args.components, "--nv-mode", args.nv_mode,
               "--caption-backend", args.caption_backend]
    for flag in ("tags-python", "caption-python", "nv-python", "audio-root", "tar-dir"):
        value = getattr(args, flag.replace("-", "_"))
        if value:
            command += [f"--{flag}", str(value.expanduser().resolve())]
    print(f"[expressive] {shlex.join(command)}", flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
