"""whisper-small transcription — used only when the manifest has no ``txt``.

Two outputs matter downstream: the transcript (character count -> speed) and the
detected language (which routes the accent head). The transcript itself is emitted
as ``asr_text`` so a surprising speed value can be traced to the words behind it.

Batched, unlike the original per-clip loop. Whisper always pads its input to a fixed
30 s / 3000-frame window, so a batch is bit-identical to the same clips run one at a
time — the padding is whisper's own, not an artefact of batching.

This is by far the most expensive stage per clip, and it is entirely skippable: a
corpus that ships transcripts never loads this model.
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from .. import config
from ..config import TARGET_SR

_LANG_TAG = re.compile(r"<\|([a-z]{2})\|>")
WINDOW_SAMPLES = 30 * TARGET_SR
MAX_NEW_TOKENS = 200


class Transcriber:
    """whisper-small wrapper: ``transcribe(wavs) -> [(text, lang), ...]``."""

    def __init__(self, device: str = "cuda", model_id: str | None = None):
        self.device = device
        self.model_id = model_id or config.asr_model()

    def load(self) -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self._torch = torch
        # fp16 is a GPU-only win; on CPU it is slower and often unsupported
        self.dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=self.dtype).to(self.device).eval()

    def transcribe(self, waveforms: Sequence[np.ndarray]) -> list[tuple[str, str | None]]:
        torch = self._torch
        clipped = [w[:WINDOW_SAMPLES] for w in waveforms]
        features = self.processor(clipped, sampling_rate=TARGET_SR,
                                  return_tensors="pt").input_features
        features = features.to(self.device, self.dtype)
        with torch.no_grad():
            ids = self.model.generate(features, task="transcribe",
                                      max_new_tokens=MAX_NEW_TOKENS)
        with_tags = self.processor.batch_decode(ids, skip_special_tokens=False)
        texts = self.processor.batch_decode(ids, skip_special_tokens=True)

        out = []
        for tagged, text in zip(with_tags, texts):
            match = _LANG_TAG.search(tagged)
            out.append((text.strip(), match.group(1) if match else None))
        return out
