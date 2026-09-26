#!/usr/bin/env python3
"""Run Continuo data annotation with separate Phase 1 and ASR interpreters."""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_annotation"


def commands(input_dir: Path, config: Path, processed_root: Path,
             phase1_python: Path, asr_python: Path, stages: str) -> list[tuple[str, list[str]]]:
    result = []
    if stages in ("both", "phase1"):
        expected = Path(str(input_dir).rstrip(os.sep) + "_processed")
        if processed_root != expected:
            raise ValueError("main.py writes beside the input folder; --processed-root must match that location")
        result.append(("phase1", [str(phase1_python), "main.py", "--config_path", str(config),
                                   "--input_folder_path", str(input_dir)]))
    if stages in ("both", "asr"):
        result.append(("asr", [str(asr_python), "transcribe.py", "--config_path", str(config),
                                "--processed-root", str(processed_root)]))
    return result


def phase1_library_path(python: str, current: str) -> str:
    site = subprocess.check_output(
        [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        text=True).strip()
    libraries = [Path(site) / "nvidia" / name / "lib" for name in
                 ("cudnn", "cublas", "cuda_runtime", "cuda_nvrtc", "cufft",
                  "curand", "cusparse", "cusolver")]
    present = [str(path) for path in libraries if path.is_dir()]
    return os.pathsep.join(present + ([current] if current else []))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="directory of source audio files")
    ap.add_argument("--config", type=Path, default=DATA / "examples/qwen3_asr.json")
    ap.add_argument("--processed-root", type=Path, help="existing Phase 1 output for --stages asr")
    ap.add_argument("--phase1-python", type=Path, default=DATA / ".venv-phase1/bin/python")
    ap.add_argument("--asr-python", type=Path, default=ROOT / ".venv-data-asr/bin/python")
    ap.add_argument("--stages", choices=("both", "phase1", "asr"), default="both")
    ap.add_argument("--phase1-gpus", default="", help="CUDA_VISIBLE_DEVICES for Phase 1")
    ap.add_argument("--asr-gpus", default="", help="CUDA_VISIBLE_DEVICES for ASR")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    input_dir = args.input.expanduser().resolve()
    config = args.config.expanduser().resolve()
    processed_root = (args.processed_root.expanduser().resolve() if args.processed_root else
                      Path(str(input_dir).rstrip(os.sep) + "_processed"))
    if not args.dry_run and not input_dir.is_dir():
        ap.error(f"input directory does not exist: {input_dir}")
    if not config.is_file():
        ap.error(f"config does not exist: {config}")
    try:
        jobs = commands(input_dir, config, processed_root, args.phase1_python.expanduser().resolve(),
                        args.asr_python.expanduser().resolve(), args.stages)
    except ValueError as exc:
        ap.error(str(exc))
    if not args.dry_run:
        for stage, cmd in jobs:
            if not Path(cmd[0]).is_file():
                ap.error(f"{stage} interpreter does not exist: {cmd[0]}")
    for stage, cmd in jobs:
        env = os.environ.copy()
        gpus = args.phase1_gpus if stage == "phase1" else args.asr_gpus
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = gpus
        print(f"[{stage}] {shlex.join(cmd)}", flush=True)
        if not args.dry_run:
            if stage == "phase1":
                env["LD_LIBRARY_PATH"] = phase1_library_path(cmd[0], env.get("LD_LIBRARY_PATH", ""))
            else:
                env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            subprocess.run(cmd, cwd=DATA, env=env, check=True)
    print(f"processed root: {processed_root}")


if __name__ == "__main__":
    main()
