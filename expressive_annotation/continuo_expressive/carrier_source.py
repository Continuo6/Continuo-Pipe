"""Read a sample range from one shared recording carrier.

Continuo's final rows point to a single encoded carrier and give sample bounds at
the carrier's native rate. Decode once per carrier, slice at that rate, then
resample the slice for the expressive models.
"""
from __future__ import annotations

import os
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .jsonl import ManifestError

_CACHE: OrderedDict[str, tuple[np.ndarray, int]] = OrderedDict()
_LOCK = threading.Lock()
_LOADING: dict[str, threading.Lock] = {}
_MAX = max(0, int(os.environ.get("CONTINUO_EXPRESSIVE_CARRIER_CACHE", "2")))


def _decode(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return np.asarray(wav, dtype=np.float32), int(sr)
    except Exception:
        pass
    try:
        import torchaudio
        wav, sr = torchaudio.load(str(path))
        wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav[0]
        return np.asarray(wav.numpy(), dtype=np.float32), int(sr)
    except Exception:
        pass
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "default=nokey=1:noprint_wrappers=1",
         str(path)], capture_output=True, text=True, check=False)
    if probe.returncode or not probe.stdout.strip().isdigit():
        raise ManifestError(f"cannot read carrier sample rate: {path}")
    sr = int(probe.stdout.strip())
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
         "-ac", "1", "-ar", str(sr), "-"], capture_output=True, check=False)
    if proc.returncode:
        raise ManifestError(f"cannot decode carrier: {path}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy(), sr


def _cached(path: Path) -> tuple[np.ndarray, int]:
    key = str(path)
    if _MAX == 0:
        return _decode(path)
    while True:
        with _LOCK:
            found = _CACHE.get(key)
            if found is not None:
                _CACHE.move_to_end(key)
                return found
            loading = _LOADING.get(key)
            mine = loading is None
            if mine:
                loading = _LOADING[key] = threading.Lock()
                loading.acquire()
        if not mine:
            with loading:
                pass
            continue
        try:
            found = _decode(path)
            with _LOCK:
                _CACHE[key] = found
                while len(_CACHE) > _MAX:
                    _CACHE.popitem(last=False)
            return found
        finally:
            with _LOCK:
                _LOADING.pop(key, None)
            loading.release()


def load_carrier_slice(row: dict, target_sr: int) -> np.ndarray:
    path = Path(row.get("_path") or row["wav_path"])
    wav, sr = _cached(path)
    declared = row.get("sample_rate")
    if declared is not None and int(declared) != sr:
        raise ManifestError(f"carrier sample rate mismatch for {path.name}: {declared} != {sr}")
    try:
        start = int(row["carrier_start_samples"])
        end = int(row["carrier_end_samples"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError("carrier row needs integer sample bounds") from exc
    if start < 0 or end <= start or end > len(wav) + 2:
        raise ManifestError(f"carrier bounds outside audio: {path.name} [{start}, {end})")
    clip = wav[start:min(end, len(wav))]
    if sr != target_sr:
        from .audio import _resample
        clip = _resample(clip, sr, target_sr)
    return np.ascontiguousarray(clip, dtype=np.float32)
