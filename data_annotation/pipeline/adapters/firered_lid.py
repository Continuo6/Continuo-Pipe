"""Adapter: FireRedLID (FireRedTeam/FireRedLID) language identification.

The upstream package isn't pip-installable in our env (its pyproject pins
torch 2.10 / transformers 5.1 / numpy 2.4, none of which match our
runtime), so the module is loaded by adding the cloned source tree to
``sys.path`` at warmup time.

FireRedLID's ``FeatExtractor`` already accepts ``(sample_rate, np_array)``
tuples — when the first element of the input list isn't a string, the
loader skips the kaldiio file read entirely. So this adapter passes
numpy clips straight through; no temp WAVs.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from pipeline.config import FireRedLIDParams
from pipeline.stages.base import LanguageIdentifier


_INT16_SCALE = 32768.0


def _scale_to_int16_magnitude(wav: np.ndarray) -> np.ndarray:
    # FireRedLID's upstream kaldi-style fbank expects int16 magnitude.
    scaled = np.ascontiguousarray(wav, dtype=np.float32)
    # Avoid mutating the caller's array when no copy was needed.
    if scaled is wav or np.shares_memory(scaled, wav):
        scaled = scaled.copy()
    scaled *= _INT16_SCALE
    return scaled


class FireRedLIDAdapter(LanguageIdentifier):
    def __init__(self, params: FireRedLIDParams, device: str):
        self._params = params
        self._device = device
        self._model: object | None = None

    def warmup(self) -> None:
        if self._model is not None:
            return
        # The fireredasr2s package isn't on PyPI in a version compatible
        # with our env; load it from a cloned source tree pinned by config.
        src = self._params.source_dir
        if src and src not in sys.path:
            sys.path.insert(0, src)

        # Imported lazily so envs without FireRedLID don't pay the import cost.
        from fireredasr2s.fireredlid.lid import FireRedLid, FireRedLidConfig

        model_dir = self._params.model_dir
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"FireRedLID model_dir not found: {model_dir}. "
                f"Run huggingface_hub snapshot_download(\"FireRedTeam/FireRedLID\") "
                f"and point model_dir at the resulting snapshot path."
            )

        # Explicit ``dtype`` wins over legacy ``use_half``; default fp32.
        import torch
        from utils.tool import resolve_dtype
        if self._params.dtype is not None:
            target_dtype = resolve_dtype(self._params.dtype)
        else:
            target_dtype = torch.float16 if self._params.use_half else torch.float32

        # Construct upstream as fp32 and cast ourselves: upstream only knows
        # fp32 / fp16, so a unified path here handles bf16 too.
        cfg = FireRedLidConfig(
            use_gpu=(self._device.startswith("cuda")),
            use_half=False,
        )
        self._model = FireRedLid.from_pretrained(model_dir, cfg)

        if target_dtype is not torch.float32:
            # Cast the encoder/decoder + wrap inner ``process`` so the fp32
            # fbank features from kaldi-native-fbank get cast on the way in.
            self._model.model = self._model.model.to(target_dtype)
            inner_process = self._model.model.process
            self._model.model.process = (
                lambda feats, lengths, *a, **kw:
                inner_process(feats.to(target_dtype), lengths, *a, **kw)
            )

    def identify_batch(
        self, clips: list[tuple[int, np.ndarray]]
    ) -> list[tuple[str, float]]:
        if self._model is None:
            raise RuntimeError("call warmup() before identify_batch()")
        if not clips:
            return []

        # FireRedLID expects either a list of wav-path strings or a list of
        # (sample_rate, 1D ndarray) tuples. We feed the latter.
        #
        # IMPORTANT: kaldi-style fbank (which the upstream feature pipeline
        # uses via ``kaldi_native_fbank.OnlineFbank``) expects samples in
        # int16-magnitude (i.e. floats in ``[-32768, 32767]``). The
        # reference path loads via ``kaldiio.load_mat`` which returns int16.
        # Our pipeline waveforms are float32 in ``[-1, 1]`` post-loudness-
        # normalization — fed unscaled, the fbank features are essentially
        # silence and the model returns a generic low-confidence guess.
        # Scale up before handing off.
        #
        # Mini-batch the forward pass: the conformer encoder's attention is
        # O(L^2 x batch), so a long-form file's 100+ segments fed in one
        # ``process()`` call OOMs a 24 GB GPU. Upstream wraps the forward in
        # a bare ``except`` that returns ``lang=""`` for the whole batch, so
        # an OOM surfaces not as a crash but as every segment silently going
        # 'unknown'. Bounding the batch keeps memory in check; uttids are
        # chunk-local so by-id alignment stays correct within each call.
        max_bs = max(1, int(self._params.max_batch_size))
        out: list[tuple[str, float]] = []
        for base in range(0, len(clips), max_bs):
            chunk = clips[base:base + max_bs]
            uttids = [f"seg_{i}" for i in range(len(chunk))]
            wav_inputs = [
                (sr, _scale_to_int16_magnitude(wav)) for sr, wav in chunk
            ]
            rows = self._model.process(uttids, wav_inputs)
            # FireRedLid.process drops degenerate clips and returns rows with
            # ``lang=""`` for them, *preserving the original uttid order*.
            # Walk by uttid to align back to clip order; missing → ("", 0.0).
            by_id = {r["uttid"]: r for r in rows}
            for u in uttids:
                r = by_id.get(u)
                out.append(
                    ("", 0.0)
                    if (r is None or not r.get("lang"))
                    else (r["lang"], float(r.get("confidence") or 0.0))
                )
        return out

    def release(self) -> None:
        self._model = None
