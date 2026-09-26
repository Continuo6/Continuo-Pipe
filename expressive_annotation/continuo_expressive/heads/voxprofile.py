"""Vox-Profile (USC SAIL) age/sex head — the pipeline's second age estimate.

Only the ``age_sex`` head is wired up: it is the one the age cascade consumes
(``tiantiaf/wavlm-large-age-sex``). Vox-Profile also publishes emotion, accent and
voice-quality heads; those are not part of this pipeline (emotion is supplied
separately, accent comes from Voxlect, and voice quality is not used), so
they are not carried here. Adding one back means a new ``_MODPATH`` entry plus the
matching forward-arity branch.

Needs the vendored repo on ``sys.path`` — :mod:`continuo_expressive.heads.loader`
puts it there. The repo's own forward hardcodes ``.cuda()``, so ``device`` must be a
CUDA device.

**This head does not batch cleanly, and that is not fixable from here.** Its forward
runs ``Wav2Vec2FeatureExtractor`` over each *already-padded* row
(``wavlm_demographics.py``: ``self.processor(x[idx], ..., padding=True)``), so the
zero-mean/unit-variance statistics are computed over the real samples *plus* whatever
padding the batch's longest clip introduced. The encoder mask and the pooling are both
correct; the input scaling is not. A clip's ``age_years`` therefore drifts slightly
depending on which clips share its batch.

Batch-dependent output is undesirable for a corpus, so this head defaults to one
clip at a time. Pass ``allow_batching=True``
(``continuo-annotate --batch-age-head``) to trade that exactness for throughput.
"""
from __future__ import annotations

import importlib
from typing import Sequence

import numpy as np

from .base import Head, Prediction, pad_batch

SEX_LABELS = ["Female", "Male"]

_MODPATH = {
    "wavlm": ("src.model.age_sex.wavlm_demographics", "WavLMWrapper"),
    "whisper": ("src.model.age_sex.whisper_demographics", "WhisperWrapper"),
}


class AgeSexHead(Head):
    attribute = "age"
    max_seconds = 15.0

    def __init__(self, model_id: str, device: str = "cuda", backbone_cls: str = "wavlm",
                 allow_batching: bool = False, **kw):
        super().__init__(model_id, device, **kw)
        self.backbone_cls = backbone_cls
        self.allow_batching = allow_batching

    def load(self) -> None:
        import torch
        import torch.nn.functional as F
        self._torch, self._F = torch, F
        module, cls_name = _MODPATH[self.backbone_cls]
        wrapper_cls = getattr(importlib.import_module(module), cls_name)
        self.model = wrapper_cls.from_pretrained(self.model_id, **self.kw).to(self.device).eval()

    def predict(self, waveforms: Sequence[np.ndarray]) -> list[Prediction]:
        # enforce the encoder cap here rather than trusting every caller to. Idempotent
        # for callers that already truncate (continuo-annotate does).
        waveforms = [self.truncate(w) for w in waveforms]
        if len(waveforms) > 1 and not self.allow_batching:
            # padding shifts this head's input normalisation — see the module docstring
            return [p for w in waveforms for p in self._forward([w])]
        return self._forward(waveforms)

    def _forward(self, waveforms: Sequence[np.ndarray]) -> list[Prediction]:
        torch, F = self._torch, self._F
        x, length = pad_batch(waveforms, torch)
        with torch.no_grad():
            out = self.model(x.to(self.device), length=length, return_feature=True)
        age_logits, sex_logits = out[0], out[1]            # (age, sex, feature)
        sex_p = F.softmax(sex_logits, dim=1).float().cpu().numpy()
        years = age_logits.squeeze(-1).float().cpu().numpy()

        preds = []
        for row, yr in zip(sex_p, years):
            probs = {label: float(p) for label, p in zip(SEX_LABELS, row)}
            preds.append(Prediction(
                attribute=self.attribute, pred_label=self.argmax_label(probs),
                probs=probs, continuous={"age_years": float(yr) * 100.0},
            ))
        return preds
