"""Adapter: AI4Bharat IndicConformer-600M-Multilingual (CTC / RNNT).

Wraps the ``ai4bharat/indic-conformer-600m-multilingual`` model published
on the Hugging Face Hub. The model is a hybrid:

* preprocessor.ts — TorchScript mel feature extractor (runs on GPU when
  available),
* encoder.onnx + ctc_decoder.onnx + rnnt_decoder.onnx + per-language
  joint_post_net_*.onnx — onnxruntime inference sessions.

The HF custom-code wrapper exposes a single callable with the signature
``model(wav, language_code, decoder_type)`` where ``wav`` is a torch
tensor at 16 kHz mono and ``decoder_type ∈ {"ctc", "rnnt"}``. There's
no internal batching for CTC ("currently no batching" — see the model's
own _ctc_decode), so we iterate segment-by-segment from Python and let
the upstream batch_size knob be a hint, not a hard requirement.

Language handling: IndicConformer cannot detect language itself — the
caller must pass a 2-letter code. Phase 2 normally takes that from the
LID stage's per-segment hint; if a segment arrives with no usable hint
(``None`` / ``"unknown"``), the adapter falls back to
:attr:`IndicConformerASRParams.default_language` (default ``"bn"``,
Bengali — the primary use case this adapter was added for).

``accepts`` returns True only for the 22 Indic codes the model
supports, so a Phase-2 pass configured with this adapter will skip
(``drop_lang``) anything outside that set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pipeline.config import IndicConformerASRParams
from pipeline.stages.base import ASRModel
from pipeline.types import AudioBundle, Segment

if TYPE_CHECKING:
    pass


# The model card lists these 22 codes; the per-language ONNX heads
# (joint_post_net_<code>.onnx) and vocab.json keys are the source of
# truth, but we list them here to fail fast in ``accepts``.
_SUPPORTED_INDIC_LANGUAGES: frozenset[str] = frozenset({
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks",
    "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat",
    "sd", "ta", "te", "ur",
})


class IndicConformerASR(ASRModel):
    SR = 16000

    SUPPORTED_LANGUAGES = _SUPPORTED_INDIC_LANGUAGES

    def __init__(self, params: IndicConformerASRParams, device: str):
        self._params = params
        self._device = device
        self._model = None  # set in warmup()
        self._torch = None  # cached torch module ref (avoid re-import per call)

    # ----- ASRModel API ------------------------------------------------

    def accepts(self, language: str | None) -> bool:
        return bool(
            language
            and language != "unknown"
            and language in self.SUPPORTED_LANGUAGES
            and language not in self._params.skip_languages
        )

    def warmup(self) -> None:
        if self._model is not None:
            return
        # Lazy imports so envs that don't use this adapter don't pay the
        # transformers / onnxruntime import cost.
        import torch
        from transformers import AutoModel

        self._torch = torch
        kwargs = {"trust_remote_code": True}
        if self._params.revision:
            # Custom model code is executable; pin it to an audited commit.
            kwargs["revision"] = self._params.revision
        model = AutoModel.from_pretrained(self._params.model_path, **kwargs)
        if self._device.startswith("cuda") and torch.cuda.is_available():
            # The custom-code model already picks GPU for the preprocessor
            # via torch.cuda.is_available() in its __init__; calling
            # .cuda() on the wrapper is a defensive no-op for the ONNX
            # sessions (they were created with CUDAExecutionProvider).
            model = model.cuda()
        model.eval()
        self._model = model

    def detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        """Not supported. Phase 2 should call this only when there is no
        per-segment hint, and the result is fed back as the ``language``
        argument to :meth:`transcribe` — which will then fall back to
        ``default_language``. Returning ``("unknown", 0.0)`` is the
        signal that no detection happened.
        """
        return ("unknown", 0.0)

    def transcribe(
        self,
        audio: AudioBundle,
        segments: list[Segment],
        language: str | None,
        batch_size: int,  # noqa: ARG002 - CTC has no batching; kept for ABC compat
    ) -> list[Segment]:
        """Transcribe each segment by slicing the parent waveform.

        ``language`` is the Phase-2 bucket hint (one code per call). The
        adapter falls back to ``default_language`` if it's empty.
        """
        if self._model is None:
            raise RuntimeError("call warmup() before transcribe()")
        if not segments:
            return []
        torch = self._torch
        assert torch is not None  # warmup sets it

        # Resample once via the bundle's shared cache (also amortizes
        # across DNSMOS / VAD / LID / deepfake / Qwen3-ASR which all
        # land at 16 k).
        waveform_f32 = audio.get_at_sr(self.SR).astype(np.float32, copy=False)

        # Resolve the language hint into something the model accepts.
        # The forward() signature requires a known code (it indexes a
        # per-language vocab/mask); falling back to default_language
        # lets us still process segments whose LID came back empty.
        lang_code = (language or "").strip().lower()
        if lang_code in ("", "unknown") or lang_code not in self.SUPPORTED_LANGUAGES:
            lang_code = self._params.default_language

        decoder = self._params.decoder_type
        out: list[Segment] = []
        with torch.no_grad():
            for seg in segments:
                start = max(0, int(seg.start * self.SR))
                end = max(start, int(seg.end * self.SR))
                clip = waveform_f32[start:end]
                # The preprocessor expects a 2-D tensor (1, T).
                wav_tensor = torch.from_numpy(clip).unsqueeze(0)
                try:
                    text = self._model(wav_tensor, lang_code, decoder)
                except Exception as e:  # noqa: BLE001 - keep the batch alive
                    # One bad clip shouldn't kill the rest. Emit empty
                    # text so Phase 2's drop_text filter handles it.
                    text = ""
                    if self._params.raise_on_segment_error:
                        raise
                    _log_warn(f"IndicConformer segment failed ({e}); "
                              f"emitting empty text for index={seg.index}")
                out.append(Segment(
                    start=seg.start,
                    end=seg.end,
                    speaker=seg.speaker,
                    index=seg.index,
                    text=(text or "").strip(),
                    language=lang_code,
                    extra=dict(seg.extra),
                ))
        return out

    def release(self) -> None:
        self._model = None


def _log_warn(msg: str) -> None:
    # utils.logger may not be imported in every embedding context.
    try:
        from utils.logger import Logger
        Logger.get_logger().warning(msg)
    except Exception:  # noqa: BLE001
        import sys
        print(f"[indic_conformer] WARN: {msg}", file=sys.stderr)
