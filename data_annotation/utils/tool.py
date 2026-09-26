"""Small helpers shared by data annotation stages."""

from __future__ import annotations

import json
import os
from pathlib import Path
import unicodedata

import numpy as np

from utils.logger import Logger


_AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".aac"}


def get_audio_files(folder_path):
    """Find input audio whose per-recording result is absent or invalid."""
    found = []
    for directory, subdirs, filenames in os.walk(folder_path):
        subdirs[:] = [name for name in subdirs if not name.endswith("_processed")]
        source_dir = Path(directory)
        for filename in filenames:
            source = source_dir / filename
            if source.suffix.lower() not in _AUDIO_SUFFIXES or ".temp" in filename:
                continue
            item_id = source.stem if source.suffix.lower() == ".mp3" else filename
            result = Path(f"{directory}_processed") / item_id / f"{item_id}.json"
            try:
                with result.open(encoding="utf-8") as handle:
                    json.load(handle)
            except (OSError, ValueError):
                found.append(str(source))
    return found


def detect_gpu() -> bool:
    """Report whether the required PyTorch CUDA and cuDNN backends are usable."""
    import onnxruntime as ort
    import torch

    log = Logger.get_logger()
    if not torch.cuda.is_available():
        log.warning("CUDA is unavailable")
        return False
    if not torch.backends.cudnn.is_available():
        log.warning("cuDNN is unavailable")
        return False
    log.info("CUDA devices: %d", torch.cuda.device_count())
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        log.warning("ONNX Runtime CUDA provider is unavailable")
    return True


def resolve_dtype(name: str) -> "torch.dtype":
    """Resolve a configured precision and reject unsupported bf16 hardware."""
    import torch

    choices = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    key = (name or "fp32").lower()
    if key not in choices:
        raise ValueError(f"Unsupported dtype {name!r}; choose fp32, fp16 or bf16")
    if key == "bf16" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("bf16 requires a CUDA device with bf16 support")
    return choices[key]


def check_env(logger):
    """Report relevant configuration without printing local paths or credentials."""
    for name in ("CUDA_VISIBLE_DEVICES", "HF_ENDPOINT", "http_proxy", "https_proxy"):
        logger.debug("%s: %s", name, "set" if os.getenv(name) else "unset")


def get_char_count(text):
    """Count non-whitespace characters outside Unicode punctuation categories."""
    return sum(
        not character.isspace()
        and not unicodedata.category(character).startswith(("P", "Z"))
        for character in (text or "")
    )


def calculate_audio_stats(
    data, min_duration=2, max_duration=30, min_dnsmos=2.8, min_char_count=2
):
    """Return accepted and total ``(row_index, duration)`` pairs."""
    rows = [(index, entry["end"] - entry["start"], get_char_count(entry["text"]),
             entry["dnsmos"]) for index, entry in enumerate(data)]
    ratios = [duration / count for _, duration, count, _ in rows if count]
    if ratios:
        low_quartile, high_quartile = np.percentile(ratios, [25, 75])
        spread = high_quartile - low_quartile
        bounds = (low_quartile - 1.5 * spread, high_quartile + 1.5 * spread)
    else:
        bounds = (0, float("inf"))
    accepted = [
        (index, duration) for index, duration, count, score in rows
        if min_duration <= duration <= max_duration
        and score >= min_dnsmos
        and count >= min_char_count
        and bounds[0] <= (duration / count if count else 0) <= bounds[1]
    ]
    return accepted, [(index, duration) for index, duration, _, _ in rows]
