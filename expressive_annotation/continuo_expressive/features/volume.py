"""Volume = integrated loudness (ITU-R BS.1770 LUFS), pure DSP.

LUFS is the right physical quantity once ``volume`` is defined as *playback
loudness* rather than vocal effort. Adding a second feature was tried three times
(high-frequency ratio, spectral-tilt effort, F0) and none of them moved the percept
once level was matched; plain unweighted RMS even ties LUFS on the listening
calibration. So the measurement stays plain loudness — only the bucket edges were
tuned, and those live in :mod:`.buckets`.

Cheap and CPU-only, which is why the annotate CLI runs it inside the audio-decoding
thread pool: it costs no wall-clock beside the GPU heads.

Caveat worth repeating wherever this field is consumed: on level-normalised audio,
LUFS measures the normaliser, not the speaker.
"""
from __future__ import annotations

import sys

import numpy as np

from ..config import TARGET_SR
from .buckets import VOLUME_EDGES, VOLUME_LABELS, bucket

MIN_SECONDS = 0.5              # pyloudnorm's block size floor

_meters: dict[int, object] = {}
_warned = False


def _meter(sr: int):
    """One Meter per sample rate — it precomputes filter coefficients."""
    if sr not in _meters:
        import pyloudnorm as pyln
        _meters[sr] = pyln.Meter(sr)
    return _meters[sr]


def measure(wav: np.ndarray | None, sr: int = TARGET_SR) -> dict:
    """-> ``{"volume_lufs": float|None, "volume": "soft"|"normal"|"loud"|None}``."""
    global _warned
    out = {"volume_lufs": None, "volume": None}
    if wav is None or len(wav) < sr * MIN_SECONDS:
        return out
    try:
        lufs = float(_meter(sr).integrated_loudness(wav))
    except Exception as e:
        if not _warned:
            print(f"[volume] loudness failed ({type(e).__name__}: {e}); volume left "
                  "null. Install into this env: pip install pyloudnorm", file=sys.stderr)
            _warned = True
        return out
    if not np.isfinite(lufs):      # silence -> -inf; no meaningful bucket
        return out
    out["volume_lufs"] = round(lufs, 1)
    out["volume"] = bucket(lufs, VOLUME_EDGES, VOLUME_LABELS)
    return out
