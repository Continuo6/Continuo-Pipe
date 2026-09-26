"""Runtime configuration — every path, endpoint and privilege is injectable.

Nothing here is hardcoded to one machine: each setting is read from a ``CONTINUO_EXPRESSIVE_*``
environment variable (or a CLI flag that overrides it), so the same checkout runs
on a laptop, a cluster node, or CI without editing source.

``CONTINUO_EXPRESSIVE_TRUST_REMOTE_CODE``
    ``trust_remote_code=True`` executes Python shipped inside a model repo with
    the full privileges of this process. Required by Qwen3-Omni; never assumed.

Paths to model repos / checkpoints default to ``third_party/`` and ``checkpoints/``
under the repo root, both of which are gitignored — see README §Install.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo root = parent of the package directory. Used only to resolve DEFAULT
# locations; every one of them can be overridden by an env var or CLI flag.
ROOT = Path(__file__).resolve().parent.parent

TARGET_SR = 16_000

_TRUE = {"1", "true", "yes", "on"}


def flag(name: str, default: bool = False) -> bool:
    """Read a boolean ``CONTINUO_EXPRESSIVE_<name>`` env var."""
    raw = os.environ.get(f"CONTINUO_EXPRESSIVE_{name}")
    return default if raw is None else raw.strip().lower() in _TRUE


def path(name: str, default: str | os.PathLike) -> Path:
    """Read a path from ``CONTINUO_EXPRESSIVE_<name>``, falling back to a repo-relative default."""
    raw = os.environ.get(f"CONTINUO_EXPRESSIVE_{name}")
    return Path(raw).expanduser() if raw else (ROOT / default)


def text(name: str, default: str) -> str:
    return os.environ.get(f"CONTINUO_EXPRESSIVE_{name}") or default


# --- third-party model source (vendored git repos; see scripts/setup_models.sh) ---
def voxprofile_repo() -> Path:
    return path("VOXPROFILE_REPO", "third_party/vox-profile-release")


def voxlect_repo() -> Path:
    return path("VOXLECT_REPO", "third_party/voxlect")


# --- external model sources and checkpoints ---
def firered_repo() -> Path:
    return path("FIRERED_REPO", "third_party/FireRedASR2S")


def nvbench_repo() -> Path:
    """NV-Bench checkout — source of the ``SenseVoiceSmall`` NVASR model class.

    Consumed as a checkout rather than a package because that is how it ships; the
    ``continuo-nv`` pass puts it on ``sys.path`` the way the dialect heads' repos are.
    """
    return path("NVBENCH_REPO", "third_party/NV-Bench")


def nvasr_dir() -> Path:
    """Multilingual-NVASR model directory (nonverbal-vocalization pass).

    Holds ``model.pt``, ``config.yaml``, ``am.mvn`` and the paralinguistic tokenizer.
    Not redistributed here — point CONTINUO_EXPRESSIVE_NVASR_DIR at wherever you unpacked it.
    """
    return path("NVASR_DIR", "checkpoints/Multilingual-NVASR")


def firered_model_dir() -> Path:
    return path("FIRERED_MODEL_DIR", "checkpoints/FireRedLID")


# --- Hugging Face model ids (override to pin a local snapshot) ---
def captioner_model() -> str:
    return text("CAPTIONER_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Captioner")


def asr_model() -> str:
    return text("ASR_MODEL", "openai/whisper-small")


def trust_remote_code() -> bool:
    """Whether models may execute their own bundled Python in this process."""
    return flag("TRUST_REMOTE_CODE", False)


def apply_hf_endpoint() -> str | None:
    """Respect an endpoint explicitly selected by the operator."""
    return os.environ.get("HF_ENDPOINT")


def describe() -> str:
    """One-line provenance string for run logs."""
    return (f"hf_endpoint={os.environ.get('HF_ENDPOINT', 'huggingface.co')} "
            f"trust_remote_code={trust_remote_code()}")
