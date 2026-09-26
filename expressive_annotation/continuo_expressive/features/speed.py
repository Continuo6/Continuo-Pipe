"""Speed = characters per second, from whichever transcript is available.

Two entry points because the transcript can come from two places, and the denominator
differs between them:

:func:`from_text`
    The manifest already carries ``txt``. Denominator is the **full** clip duration; skipping ASR
    removes the pipeline's largest per-clip cost.

:func:`from_asr`
    No transcript, so whisper produces one. Whisper only sees the first 30 s, so the
    denominator is capped at 30 s too — numerator and denominator must cover the same
    window or the rate is nonsense for long clips.

:func:`from_span`
    The clip is a *window* of a longer utterance, and the transcript belongs to the
    utterance rather than the window. Neither of the above works then: the window that
    holds the text has only part of the audio (rate too high, measured median +32%),
    and the windows that do not hold it have no text at all. Both numerator and
    denominator come from the utterance instead, so every window of one utterance
    reports that utterance's rate — which is what speaking rate means for a stretch of
    one person talking.

Character count is raw ``len(text)``, spaces and punctuation included, matching the
PSC ``transcription`` field the English edges were fit on.

Short clips are noisy because any leading or trailing silence has a large
effect on their measured duration. Treat values below about 2 s as indicative.
"""
from __future__ import annotations

import numpy as np

from ..config import TARGET_SR
from .buckets import SPEED_CPS_EDGES, SPEED_LABELS, bucket

MIN_SECONDS = 0.5
ASR_WINDOW_SECONDS = 30.0


def _rate(chars: int, duration: float, lang: str | None) -> dict:
    out = {"speed_cps": None, "speed": None}
    if not chars or duration <= 0:
        return out
    cps = chars / duration
    out["speed_cps"] = round(cps, 1)
    edges = SPEED_CPS_EDGES.get(lang or "")     # unlisted language -> rate, no bucket
    if edges:
        out["speed"] = bucket(cps, edges, SPEED_LABELS)
    return out


def from_text(text: str | None, lang: str | None, wav: np.ndarray,
              sr: int = TARGET_SR) -> dict:
    """CPS from a transcript supplied by the manifest, over the whole clip."""
    duration = len(wav) / sr
    if duration < MIN_SECONDS:
        return {"speed_cps": None, "speed": None}
    return _rate(len(text or ""), duration, lang)


def from_span(chars: int | None, lang: str | None, seconds: float | None) -> dict:
    """CPS for a window of an utterance, from the whole utterance's own numbers.

    ``chars``/``seconds`` describe the utterance the window was cut from, not the
    window. See tools/prepare_long.split_windows, which carries them.
    """
    if not chars or not seconds or seconds < MIN_SECONDS:
        return {"speed_cps": None, "speed": None}
    return _rate(int(chars), float(seconds), lang)


def from_asr(text: str | None, lang: str | None, wav: np.ndarray,
             sr: int = TARGET_SR) -> dict:
    """CPS from an ASR transcript, over the same 30 s window the ASR saw."""
    duration = len(wav) / sr
    if duration < MIN_SECONDS:
        return {"speed_cps": None, "speed": None}
    return _rate(len(text or ""), min(duration, ASR_WINDOW_SECONDS), lang)
