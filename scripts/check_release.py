#!/usr/bin/env python3
"""Reject machine paths, private model hooks, secrets and binary assets.

Inspect the Git index as well as the working tree: an ignored runtime file can
still be force-added, and a staged secret can differ from the current file.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
             "third_party", "checkpoints", "runs", "audio", "corpus", "outputs"}
BINARY_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".m4a",
                   ".mp3", ".flac", ".wav", ".tar", ".pem", ".key"}
MACHINE_PATH = re.compile(r"/(?:120\d+|mnt|home|root|tmp|data|srv|workspace|scratch|gpfs|nfs|nas|path|export|shared|Users|private)(?:/|$)")
MACHINE_HOST = re.compile(r"\b(?:\d+-)?node\d+\b", re.I)
NON_ENGLISH_SOURCE = re.compile(r"[\u3400-\u9fff\u3000-\u303f\uff00-\uffef]")
SECRET = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bAKIA[0-9A-Z]{16}\b|\bhf_[A-Za-z0-9]{24,}\b")


def _git_paths(*flags: str) -> list[Path] | None:
    try:
        result = subprocess.run(["git", "ls-files", *flags, "-z"], cwd=ROOT,
                                capture_output=True, check=False)
    except FileNotFoundError:
        return None
    if result.returncode:
        return None
    return [Path(os.fsdecode(name)) for name in result.stdout.split(b"\0") if name]


def _scan(rel: Path, data: bytes, label: str) -> list[str]:
    if rel.suffix.lower() in BINARY_SUFFIXES or b"\0" in data:
        return [f"binary asset: {rel} ({label})"]
    if rel.name == ".env" or rel.name.startswith(".env."):
        return [f"private config: {rel} ({label})"]
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        return [f"non-text file: {rel} ({label})"]
    return [f"review {rel}:{line_no} ({label})"
            for line_no, line in enumerate(content.splitlines(), 1)
            if (MACHINE_PATH.search(line) or MACHINE_HOST.search(line)
                or NON_ENGLISH_SOURCE.search(line)
                or SECRET.search(line))]


def check() -> list[str]:
    problems: list[str] = []
    tracked = _git_paths("--cached")
    if tracked is None:
        candidates = []
        for directory, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".venv")]
            candidates.extend((Path(directory) / name).relative_to(ROOT) for name in files)
    else:
        candidates = sorted(set(tracked + (_git_paths("--others", "--exclude-standard") or [])))
        for rel in tracked:
            blob = subprocess.run(["git", "show", f":{rel.as_posix()}"], cwd=ROOT,
                                  capture_output=True, check=False)
            if blob.returncode:
                problems.append(f"cannot read staged file: {rel}")
            else:
                problems.extend(_scan(rel, blob.stdout, "index"))
    for rel in candidates:
        path = ROOT / rel
        if path.is_symlink():
            problems.append(f"symlink asset: {rel}")
        elif path.is_file():
            problems.extend(_scan(rel, path.read_bytes(), "worktree"))
    return problems


if __name__ == "__main__":
    issues = check()
    if issues:
        print("\n".join(issues))
        raise SystemExit(1)
    print("release scan passed")
