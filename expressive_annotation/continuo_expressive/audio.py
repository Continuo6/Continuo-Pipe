"""Audio decoding + a prefetching batch loader.

Every model in the pipeline consumes mono float32 at 16 kHz, so decoding happens
once here and the array is reused by all of them.

:class:`BatchLoader` exists for throughput. Decoding is I/O- and CPU-bound while
inference is GPU-bound, so running them in lockstep leaves one of the two idle for
most of a long run. The loader decodes each batch across a small thread pool
(soundfile and librosa both release the GIL in their C paths) and keeps one batch
decoded ahead of the consumer, so disk reads overlap the previous batch's forward
pass. Any per-clip CPU-only feature can be computed inside the same pool by passing
``per_clip``, which pulls it off the critical path entirely.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from .config import TARGET_SR


def load_source(row: dict, target_sr: int = TARGET_SR) -> np.ndarray:
    """A manifest row -> audio, whether it names a file or a tar member.

    Imported lazily so :mod:`continuo_expressive.tarsource` — and the torchaudio it
    decodes with — is not pulled in by callers that only ever read files.
    """
    from .tarsource import load_row
    return load_row(row, target_sr)


def load_wav(path: str | os.PathLike, target_sr: int = TARGET_SR) -> np.ndarray:
    """Mono float32 at ``target_sr``. soundfile first, librosa as the fallback."""
    try:
        import soundfile as sf
        wav, sr = sf.read(os.fspath(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            wav = _resample(wav, sr, target_sr)
        return np.ascontiguousarray(wav, dtype=np.float32)
    except Exception:
        import librosa
        wav, _ = librosa.load(os.fspath(path), sr=target_sr, mono=True)
        return np.ascontiguousarray(wav, dtype=np.float32)


def _resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    try:
        import torch
        import torchaudio.functional as AF
        return AF.resample(torch.from_numpy(wav), sr, target_sr).numpy()
    except Exception:
        import librosa
        return librosa.resample(wav, orig_sr=sr, target_sr=target_sr)


def truncate(wav: np.ndarray, seconds: float, sr: int = TARGET_SR) -> np.ndarray:
    n = int(seconds * sr)
    return wav[:n] if wav.shape[-1] > n else wav


def batched(items: Sequence[Any], size: int) -> Iterator[list]:
    size = max(1, size)
    for i in range(0, len(items), size):
        yield list(items[i:i + size])


class BatchLoader:
    """Iterate ``(rows, wavs, extras)`` batches with decoding overlapped on threads.

    ``per_clip(row, wav) -> dict`` runs in the worker thread; use it for CPU-only
    features (loudness) so they cost no wall-clock next to GPU work. A clip that
    fails to decode is dropped from its batch with a warning — one unreadable file
    must not stop the rest of the run.
    """

    def __init__(self, rows: Sequence[dict], batch_size: int = 8, workers: int = 4,
                 per_clip: Callable[[dict, np.ndarray], dict] | None = None,
                 on_error: Callable[[dict, Exception], None] | None = None):
        self.rows = list(rows)
        self.batch_size = max(1, batch_size)
        self.workers = max(1, workers)
        self.per_clip = per_clip
        self.on_error = on_error

    def _decode(self, row: dict):
        try:
            wav = load_source(row)
        except Exception as e:                      # unreadable / corrupt clip
            if self.on_error:
                self.on_error(row, e)
            return None
        extra = self.per_clip(row, wav) if self.per_clip else {}
        return row, wav, extra

    def __iter__(self) -> Iterator[tuple[list[dict], list[np.ndarray], list[dict]]]:
        chunks = list(batched(self.rows, self.batch_size))
        if not chunks:
            return
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            pending = pool.map(self._decode, chunks[0])
            for nxt in chunks[1:] + [None]:
                decoded = [d for d in pending if d is not None]
                # start the next batch's I/O before yielding, so it decodes while
                # the consumer runs its forward pass on this one
                pending = pool.map(self._decode, nxt) if nxt is not None else iter(())
                if decoded:
                    rows, wavs, extras = zip(*decoded)
                    yield list(rows), list(wavs), list(extras)

    def __len__(self) -> int:
        return (len(self.rows) + self.batch_size - 1) // self.batch_size
