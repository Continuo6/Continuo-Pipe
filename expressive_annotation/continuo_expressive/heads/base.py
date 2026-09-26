"""The uniform interface every neural head implements.

A head takes a batch of decoded 16 kHz mono waveforms and returns one
:class:`Prediction` per input, in order. Heavy imports (torch, the vendored
model repos) live inside :meth:`Head.load`, so importing this package stays cheap
for callers that only want the label maps or the prompt builders.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..config import TARGET_SR


@dataclass
class Prediction:
    """One head's output for one clip."""
    attribute: str
    pred_label: str | list[str] | None = None
    probs: dict[str, float] = field(default_factory=dict)
    continuous: dict[str, float] = field(default_factory=dict)

    def top(self, k: int = 3) -> dict[str, float]:
        ranked = sorted(self.probs.items(), key=lambda kv: -kv[1])[:k]
        return {label: round(p, 3) for label, p in ranked}


class Head(ABC):
    #: attribute produced, e.g. "gender"
    attribute: str = "unknown"
    #: encoder input cap; Vox-Profile / Voxlect hard-cap at 15 s
    max_seconds: float = 15.0

    def __init__(self, model_id: str, device: str = "cuda", **kw):
        self.model_id = model_id
        self.device = device
        self.kw = kw

    @abstractmethod
    def load(self) -> None:
        """Load weights onto ``self.device``. Called once before :meth:`predict`."""

    @abstractmethod
    def predict(self, waveforms: Sequence[np.ndarray]) -> list[Prediction]:
        """Run one batch of float32 mono 16 kHz waveforms."""

    def truncate(self, wav: np.ndarray) -> np.ndarray:
        n = int(self.max_seconds * TARGET_SR)
        return wav[:n] if wav.shape[-1] > n else wav

    @staticmethod
    def argmax_label(probs: dict[str, float]) -> str | None:
        return max(probs, key=probs.get) if probs else None


def pad_batch(waveforms: Sequence[np.ndarray], torch):
    """Right-pad a list of 1-D waveforms into ``(B, T)`` + their true lengths.

    Both vendored repos take ``length`` alongside the padded tensor and build their
    own encoder masks from it, so padding does not leak into their pooled output.
    """
    lengths = [len(w) for w in waveforms]
    x = torch.zeros(len(waveforms), max(lengths), dtype=torch.float32)
    for i, w in enumerate(waveforms):
        x[i, :len(w)] = torch.from_numpy(w)
    return x, torch.tensor(lengths)
