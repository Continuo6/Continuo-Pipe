"""Adapter: DNSMOS ONNX quality scorer.

Wraps ``models.dnsmos.ComputeScore`` (unchanged) to expose
:class:`QualityScorer`. The score written to each :class:`Segment` is the
DNSMOS OVRL value, matching the legacy pipeline's ``"dnsmos"`` JSON key.
"""

from __future__ import annotations

import numpy as np
import tqdm

from models import dnsmos
from pipeline.config import DNSMOSScorerParams
from pipeline.stages.base import QualityScorer
from pipeline.types import AudioBundle, Segment


class DNSMOSScorer(QualityScorer):
    TARGET_SR = 16000

    def __init__(self, params: DNSMOSScorerParams, device: str):
        self._params = params
        self._device = device
        self._scorer: dnsmos.ComputeScore | None = None

    def warmup(self) -> None:
        if self._scorer is None:
            self._scorer = dnsmos.ComputeScore(self._params.model_path, self._device)

    def score(
        self,
        audio: AudioBundle,
        segments: list[Segment],
    ) -> tuple[float, list[Segment]]:
        if self._scorer is None:
            raise RuntimeError("call warmup() before score()")

        waveform = audio.get_at_sr(self.TARGET_SR)
        for seg in tqdm.tqdm(segments, desc="DNSMOS"):
            start = int(seg.start * self.TARGET_SR)
            end = int(seg.end * self.TARGET_SR)
            chunk = waveform[start:end]
            seg.quality = float(self._scorer(chunk, self.TARGET_SR, False)["OVRL"])

        if not segments:
            return 0.0, segments
        avg = float(np.mean([s.quality for s in segments]))
        return avg, segments

    def release(self) -> None:
        self._scorer = None
