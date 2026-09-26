"""``audeering/wav2vec2-large-robust-24-ft-age-gender`` — gender (3-way) + age (years).

The HF repo ships weights but no modeling code, so the architecture is rebuilt here
from the model card: wav2vec2 trunk -> pooled hidden state -> an age head (1 unit,
sigmoid-scaled to 0..1, x100 = years) and a gender head (3-way softmax over
female/male/child). One forward yields both.

**Padding correctness.** The model card's snippet mean-pools *every* frame, which is
right for a batch of one and wrong for a padded batch: the zeros tacked onto short
clips drag the mean toward zero, so a clip's prediction would depend on which other
clips happened to share its batch. Here the trunk receives the processor's
``attention_mask`` and pooling averages only real frames, which makes a batched
result identical to running the clip alone — see ``scripts/verify_batching.py``.
If the processor declines to return a mask, the head falls back to one clip at a
time rather than silently producing padding-polluted numbers.
"""
from __future__ import annotations

import sys
from typing import Sequence

import numpy as np

from ..config import TARGET_SR
from .base import Head, Prediction

GENDER_LABELS = ["female", "male", "child"]


class AgeGenderHead(Head):
    attribute = "gender"
    max_seconds = 15.0

    def load(self) -> None:
        import torch
        import torch.nn as nn
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model, Wav2Vec2PreTrainedModel)

        class ModelHead(nn.Module):
            def __init__(self, config, num_labels):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, num_labels)

            def forward(self, x):
                x = self.dropout(x)
                x = torch.tanh(self.dense(x))
                return self.out_proj(self.dropout(x))

        class AgeGenderModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.config = config
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = ModelHead(config, 1)
                self.gender = ModelHead(config, 3)
                self.init_weights()

            def forward(self, input_values, attention_mask=None):
                hidden = self.wav2vec2(input_values, attention_mask=attention_mask)[0]
                if attention_mask is None:
                    pooled = hidden.mean(dim=1)
                else:
                    # map the sample-level mask onto encoder frames, then average
                    # over real frames only (== the unbatched result)
                    frames = self.wav2vec2._get_feature_vector_attention_mask(
                        hidden.shape[1], attention_mask)
                    frames = frames.to(hidden.dtype).unsqueeze(-1)
                    pooled = (hidden * frames).sum(1) / frames.sum(1).clamp(min=1.0)
                return pooled, self.age(pooled), torch.softmax(self.gender(pooled), dim=1)

        self._torch = torch
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self.model = AgeGenderModel.from_pretrained(self.model_id).to(self.device).eval()
        self._warned_no_mask = False

    def predict(self, waveforms: Sequence[np.ndarray]) -> list[Prediction]:
        torch = self._torch
        # enforce the encoder cap here rather than trusting every caller to. Idempotent
        # for callers that already truncate (continuo-annotate does).
        waveforms = [self.truncate(w) for w in waveforms]
        inputs = self.processor(list(waveforms), sampling_rate=TARGET_SR,
                                return_tensors="pt", padding=True,
                                return_attention_mask=True)
        mask = inputs.get("attention_mask")
        if mask is None and len(waveforms) > 1:
            # no mask available -> padding would pollute pooling; degrade to bs=1
            if not self._warned_no_mask:
                print("[audeering] processor returned no attention_mask; running one "
                      "clip at a time to keep results padding-independent",
                      file=sys.stderr)
                self._warned_no_mask = True
            out: list[Prediction] = []
            for w in waveforms:
                out.extend(self.predict([w]))
            return out

        x = inputs["input_values"].to(self.device)
        am = mask.to(self.device) if mask is not None else None
        with torch.no_grad():
            _, age, gender = self.model(x, attention_mask=am)
        age = age.squeeze(-1).float().cpu().numpy()
        gender = gender.float().cpu().numpy()

        preds = []
        for years, g in zip(age, gender):
            probs = {label: float(p) for label, p in zip(GENDER_LABELS, g)}
            preds.append(Prediction(
                attribute=self.attribute, pred_label=self.argmax_label(probs),
                probs=probs, continuous={"age_years": float(years) * 100.0},
            ))
        return preds
