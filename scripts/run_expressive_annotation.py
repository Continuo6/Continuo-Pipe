#!/usr/bin/env python3
"""Run instruction tags, captions and NV as independent expressive passes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPRESSIVE = ROOT / "expressive_annotation"


def normalize_manifest(source: Path, target: Path, audio_root: Path | None) -> str:
    base = audio_root or source.parent
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".input-", suffix=".jsonl", dir=target.parent)
    digest = hashlib.sha256()
    try:
        with source.open(encoding="utf-8") as incoming, os.fdopen(fd, "w", encoding="utf-8") as out:
            for line_no, line in enumerate(incoming, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not row.get("id"):
                    raise ValueError(f"invalid manifest row at line {line_no}")
                if row.get("wav_path"):
                    path = Path(row["wav_path"]).expanduser()
                    if not path.is_absolute():
                        path = base / path
                    path = path.resolve()
                    if audio_root and path != audio_root and audio_root not in path.parents:
                        raise ValueError(f"audio path escapes --audio-root at line {line_no}")
                    if not path.is_file():
                        raise ValueError(f"audio file missing at line {line_no}: {path}")
                    row["wav_path"] = str(path)
                encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode()
                out.write(encoded.decode())
                digest.update(encoded)
            out.flush()
            os.fsync(out.fileno())
        value = digest.hexdigest()
        marker = target.parent / "input_manifest.sha256"
        if marker.is_file() and marker.read_text().strip() != value:
            raise ValueError("input manifest changed; use a new output directory for a new run")
        os.replace(temporary, target)
        marker.write_text(value + "\n", encoding="ascii")
        return value
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--audio-root", type=Path, help="base for relative wav_path values")
    ap.add_argument("--tar-dir", type=Path, help="base for relative source_tar values")
    ap.add_argument("--components", default="tags,caption,nv", help="comma-separated: tags,caption,nv")
    ap.add_argument("--tags-python", type=Path, default=ROOT / ".venv-expressive-tags/bin/python")
    ap.add_argument("--caption-python", type=Path, default=ROOT / ".venv-expressive-caption/bin/python")
    ap.add_argument("--nv-python", type=Path, default=ROOT / ".venv-expressive-nv/bin/python")
    ap.add_argument("--nv-mode", choices=("direct", "verified"), default="verified")
    ap.add_argument("--caption-backend", choices=("vllm", "transformers"), default="vllm")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--emotion-jsonl", type=Path, help="optional external emotion results")
    ap.add_argument("--emotion-tau", type=float, help="required when --emotion-jsonl is given")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    components = [part.strip() for part in args.components.split(",") if part.strip()]
    if not components or len(components) != len(set(components)) or any(
        part not in {"tags", "caption", "nv"} for part in components
    ):
        ap.error("--components must list distinct values from tags,caption,nv")
    if args.batch_size < 1:
        ap.error("--batch-size must be positive")
    if args.emotion_jsonl and (args.emotion_tau is None or not 0 <= args.emotion_tau <= 1):
        ap.error("--emotion-jsonl requires --emotion-tau in [0, 1]")
    if args.emotion_jsonl and "tags" not in components:
        ap.error("--emotion-jsonl requires tags in --components")
    source = args.manifest.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    audio_root = args.audio_root.expanduser().resolve() if args.audio_root else None
    tar_dir = args.tar_dir.expanduser().resolve() if args.tar_dir else None
    if not source.is_file():
        ap.error(f"manifest does not exist: {source}")
    if not args.dry_run:
        interpreters = {"tags": args.tags_python, "caption": args.caption_python,
                        "nv": args.nv_python}
        for stage in components:
            interpreter = interpreters[stage].expanduser().resolve()
            if not interpreter.is_file():
                ap.error(f"{stage} interpreter does not exist: {interpreter}")
    prepared = out_dir / "input_manifest.jsonl"
    if not args.dry_run:
        normalize_manifest(source, prepared, audio_root)
    annotations = out_dir / "annotations.jsonl"
    nv_output = (out_dir / "nv_verified/final.jsonl" if args.nv_mode == "verified"
                 else out_dir / "nv_direct.jsonl")
    jobs: list[tuple[str, list[str], dict[str, str]]] = []
    env = os.environ.copy()
    if tar_dir:
        env["CONTINUO_EXPRESSIVE_TAR_DIR"] = str(tar_dir)
        env["CORPUS"] = str(tar_dir)
    if "tags" in components:
        cmd = [str(args.tags_python.expanduser().resolve()), "-m", "continuo_expressive.cli.annotate",
               "--manifest", str(prepared), "--out", str(annotations), "--resume",
               "--batch-size", str(args.batch_size)]
        if args.emotion_jsonl:
            cmd += ["--emotion-jsonl", str(args.emotion_jsonl.expanduser().resolve()),
                    "--emotion-tau", str(args.emotion_tau)]
        jobs.append(("tags", cmd, env))
    if "caption" in components:
        if "tags" not in components and not annotations.is_file() and not args.dry_run:
            ap.error("caption needs existing annotations.jsonl or tags in --components")
        cmd = [str(args.caption_python.expanduser().resolve()), "-m", "continuo_expressive.cli.caption",
               "--annot", str(annotations), "--manifest", str(prepared),
               "--backend", args.caption_backend]
        jobs.append(("caption", cmd, env))
    if "nv" in components:
        nv_python = str(args.nv_python.expanduser().resolve())
        if args.nv_mode == "verified":
            nv_env = env.copy()
            nv_env["PY_NV"] = nv_python
            cmd = ["bash", str(EXPRESSIVE / "scripts/run_nv_pipeline.sh"),
                   str(prepared), str(out_dir / "nv_verified")]
            jobs.append(("nv", cmd, nv_env))
        else:
            cmd = [nv_python, "-m", "continuo_expressive.cli.nv", "--manifest", str(prepared),
                   "--out", str(nv_output), "--resume"]
            jobs.append(("nv", cmd, env))
    for stage, command, stage_env in jobs:
        print(f"[{stage}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=EXPRESSIVE, env=stage_env, check=True)
    final = out_dir / "final.jsonl"
    merge_cmd = [sys.executable, str(ROOT / "scripts/merge_results.py"),
                 "--manifest", str(prepared), "--out", str(final)]
    if "tags" in components or "caption" in components:
        merge_cmd += ["--annotations", str(annotations)]
    if "nv" in components:
        merge_cmd += ["--nv", str(nv_output)]
    print(f"[merge] {shlex.join(merge_cmd)}", flush=True)
    if not args.dry_run:
        subprocess.run(merge_cmd, check=True)
    print(f"final: {final}")


if __name__ == "__main__":
    main()
