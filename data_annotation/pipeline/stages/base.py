"""Abstract base classes that define the pipeline's contracts.

Every concrete model (UVR-MDX-NET, pyannote, Silero, WhisperX, DNSMOS, ...)
is plugged into the pipeline through one of these interfaces via a thin
adapter (see ``pipeline/adapters/``). Adding a new model means writing a
new adapter that satisfies the same interface; no other code changes.

The :class:`Stage` base also gives every component a uniform lifecycle
(``warmup`` / ``release``) so the orchestrator can manage memory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from pipeline.context import PipelineContext
    from pipeline.types import AudioBundle, DiarizationFrame, Segment


class Stage(ABC):
    """A unit of pipeline work. Subclasses mutate the ``ctx`` in place."""

    name: str = ""

    @abstractmethod
    def run(self, ctx: "PipelineContext") -> None:
        ...

    def warmup(self) -> None:  # pragma: no cover - optional hook
        """Eagerly load any heavy resources. Called once before the run loop."""

    def release(self) -> None:  # pragma: no cover - optional hook
        """Free any heavy resources. Called when the orchestrator shuts down."""


# ----- Model interfaces (each has exactly one concrete adapter today) -----


class _Warmable(ABC):
    def warmup(self) -> None: ...
    def release(self) -> None: ...


class SourceSeparator(_Warmable):
    """Separate vocals from a music/noise mixture.

    Returned bundle contains only the vocals waveform (mono, same sample
    rate as the input). The adapter is responsible for whatever internal
    sample-rate dance the underlying model needs.
    """

    @abstractmethod
    def separate(self, audio: "AudioBundle") -> "AudioBundle":
        ...


class SpeakerDiarizer(_Warmable):
    """Produce speaker-labeled time ranges for an audio bundle."""

    @abstractmethod
    def diarize(self, audio: "AudioBundle") -> "list[DiarizationFrame]":
        ...


class VoiceActivityDetector(_Warmable):
    """Refine diarization frames into tight speech segments.

    Semantics mirror the legacy Silero stage: the VAD runs *within* each
    diarization frame, splitting long monologues into shorter chunks while
    keeping speaker labels attached.
    """

    @abstractmethod
    def detect(
        self,
        audio: "AudioBundle",
        diarization: "list[DiarizationFrame]",
    ) -> "list[Segment]":
        ...


class LanguageIdentifier(_Warmable):
    """Classify the spoken language of audio clips.

    Used to label per-segment language *before* ASR, so the downstream
    transcriber can dispatch to the right model / pick the right tokenizer
    instead of doing its own per-clip language sniffing. The clip semantics
    are deliberately abstract: the caller picks the sample-rate-correct
    waveform slice; the adapter handles whatever feature pipeline the
    underlying model needs.
    """

    @abstractmethod
    def identify_batch(
        self, clips: "list[tuple[int, np.ndarray]]"
    ) -> "list[tuple[str, float]]":
        """Classify each ``(sample_rate, waveform)`` clip.

        Returns a list of ``(language_code, confidence)`` aligned with the
        input. Empty input → empty output.
        """


class ASRModel(_Warmable):
    """Transcribe audio inside a set of pre-detected speech segments."""

    @abstractmethod
    def detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        """Return ``(language_code, probability)`` for a chunk of audio."""

    @abstractmethod
    def transcribe(
        self,
        audio: "AudioBundle",
        segments: "list[Segment]",
        language: str | None,
        batch_size: int,
    ) -> "list[Segment]":
        """Fill ``text`` and ``language`` on each segment and return them."""

    def accepts(self, language: str | None) -> bool:
        """Whether Phase 2 should keep an utterance whose detected language
        is ``language``.

        Default: keep anything except an empty / ``"unknown"`` label —
        suitable for ASR backends that try to handle everything they're
        given. Backends with a narrower trusted-language set (e.g.
        Qwen3-ASR which gates out commonly-misclassified ``ms``/``id``)
        override to express that policy.
        """
        return bool(language) and language != "unknown"


class QualityScorer(_Warmable):
    """Annotate each segment with a scalar quality score.

    Legacy DNSMOS fills the ``quality`` field with the OVRL score in [1, 5];
    other scorers (NISQA, UTMOS, ...) can do the same and the rest of the
    pipeline remains unchanged.
    """

    @abstractmethod
    def score(
        self,
        audio: "AudioBundle",
        segments: "list[Segment]",
    ) -> tuple[float, "list[Segment]"]:
        """Return ``(average_quality, segments_with_quality)``."""
