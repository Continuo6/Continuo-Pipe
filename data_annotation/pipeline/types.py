"""Typed data contracts shared between pipeline stages.

These dataclasses replace the loose ``dict`` payloads used by the original
script, so each stage's input/output is statically describable and
IDE-discoverable. Fields are added as the bundle flows through the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AudioBundle:
    """A standardized audio buffer passed between stages.

    Carries an internal resample cache so multiple downstream stages
    that need the same target sample rate (e.g. 16 k for DNSMOS, VAD,
    LID, deepfake, ASR) share one resample instead of repeating it.
    Hit the cache via :meth:`get_at_sr` — the first call for a given
    target SR runs ``librosa.resample`` and stores the result; every
    subsequent call returns the cached buffer.
    """

    waveform: np.ndarray
    sample_rate: int
    name: str
    # Don't dump the cache in __repr__ / equality. Lives on the
    # instance; lazy-populated by get_at_sr.
    _resample_cache: dict[int, np.ndarray] = field(
        default_factory=dict, repr=False, compare=False,
    )

    def get_at_sr(self, target_sr: int) -> np.ndarray:
        """Return the waveform at ``target_sr``, resampling if needed.

        Identity-resamples (target_sr == self.sample_rate) return the
        original buffer with no copy. Otherwise the resampled buffer is
        cached and reused across calls.
        """
        if int(target_sr) == int(self.sample_rate):
            return self.waveform
        key = int(target_sr)
        cached = self._resample_cache.get(key)
        if cached is not None:
            return cached
        # Lazy imports — AudioBundle stays importable in envs that
        # don't have librosa loaded. This method only fires inside an
        # adapter that already needs librosa anyway.
        import librosa
        from utils.logger import time_span
        with time_span(
            f"resample_bundle_{self.sample_rate}_to_{key}_{self.name}"
        ):
            out = librosa.resample(
                self.waveform, orig_sr=self.sample_rate, target_sr=key,
            ).astype(np.float32, copy=False)
        self._resample_cache[key] = out
        return out

    def to_legacy_dict(self) -> dict[str, Any]:
        """Adapter helper: shape expected by the original model classes."""
        return {
            "waveform": self.waveform,
            "sample_rate": self.sample_rate,
            "name": self.name,
        }


@dataclass
class DiarizationFrame:
    """A single speaker-labeled time range produced by the diarizer."""

    start: float
    end: float
    speaker: str


@dataclass
class Segment:
    """A speech segment that gradually accretes information through the pipeline.

    Stages typically populate:

    * VAD stage: ``index, start, end, speaker``
    * ASR stage: ``text, language``
    * Scoring stage: ``quality`` (alias for the DNSMOS OVRL score in the
      legacy pipeline; named generically so other scorers can fill it)
    """

    start: float
    end: float
    speaker: str | None = None
    index: str | None = None
    text: str | None = None
    language: str | None = None
    quality: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_legacy_dict(self) -> dict[str, Any]:
        """Serialize in the same shape the original JSON output used.

        The legacy field name was ``dnsmos`` for the quality score; we keep
        that on disk for byte-identical compatibility with downstream tools
        that already consume the output JSONs.
        """
        out: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
        }
        if self.index is not None:
            out["index"] = self.index
        if self.speaker is not None:
            out["speaker"] = self.speaker
        if self.text is not None:
            out["text"] = self.text
        if self.language is not None:
            out["language"] = self.language
        if self.quality is not None:
            out["dnsmos"] = self.quality
        out.update(self.extra)
        return out

    @classmethod
    def from_legacy_dict(cls, data: dict[str, Any]) -> "Segment":
        known = {"start", "end", "speaker", "index", "text", "language", "dnsmos"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            start=data["start"],
            end=data["end"],
            speaker=data.get("speaker"),
            index=data.get("index"),
            text=data.get("text"),
            language=data.get("language"),
            quality=data.get("dnsmos"),
            extra=extra,
        )
