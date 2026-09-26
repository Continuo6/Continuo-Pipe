"""Voxlect (USC SAIL) dialect heads — accent as a *language-internal* taxonomy.

``accent`` here is not "foreign accent in English"; it is the dialect / L1-family
grouping inside one language, so the head is chosen by the clip's language (see
:mod:`continuo_expressive.cli.annotate`). Label ordering is per-checkpoint and is
supplied from the model card via ``label_list`` — without it the logits carry no
names and predictions come back as ``class_<i>``.

Needs the vendored voxlect repo on ``sys.path``; the loader handles that.
"""
from __future__ import annotations

import importlib
from typing import Sequence

import numpy as np

from .base import Head, Prediction, pad_batch

_MODPATH = {
    "whisper": ("src.model.dialect.whisper_dialect", "WhisperWrapper"),
    "mms": ("src.model.dialect.mms_dialect", "MMSWrapper"),
}


class DialectHead(Head):
    attribute = "accent"
    max_seconds = 15.0

    def __init__(self, model_id: str, device: str = "cuda", backbone_cls: str = "whisper",
                 label_list: Sequence[str] | None = None, **kw):
        super().__init__(model_id, device, **kw)
        self.backbone_cls = backbone_cls
        self.label_list = list(label_list) if label_list else None

    def load(self) -> None:
        import torch
        import torch.nn.functional as F
        self._torch, self._F = torch, F
        module, cls_name = _MODPATH[self.backbone_cls]
        wrapper_cls = getattr(importlib.import_module(module), cls_name)
        self.model = wrapper_cls.from_pretrained(self.model_id).to(self.device).eval()

    def predict(self, waveforms: Sequence[np.ndarray]) -> list[Prediction]:
        torch, F = self._torch, self._F
        # enforce the encoder cap here rather than trusting every caller to. Idempotent
        # for callers that already truncate (continuo-annotate does).
        waveforms = [self.truncate(w) for w in waveforms]
        x, length = pad_batch(waveforms, torch)
        with torch.no_grad():
            out = self.model(x.to(self.device), length=length, return_feature=True)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        probs = F.softmax(logits, dim=1).float().cpu().numpy()
        labels = self.label_list or [f"class_{i}" for i in range(probs.shape[1])]

        preds = []
        for row in probs:
            pm = {label: float(p) for label, p in zip(labels, row)}
            preds.append(Prediction(attribute=self.attribute,
                                    pred_label=self.argmax_label(pm), probs=pm))
        return preds
