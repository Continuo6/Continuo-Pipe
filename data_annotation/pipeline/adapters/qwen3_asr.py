"""Adapter: Qwen3-ASR (vLLM backend).

Wraps ``qwen_asr.Qwen3ASRModel.LLM`` (the vLLM-backed inference path) to
satisfy :class:`pipeline.stages.base.ASRModel`. Used in Phase 2
(``transcribe.py``) as the default ASR; the Phase-2 driver groups
utterances by their LID-assigned language, calls :meth:`transcribe`
once per language, and post-filters by :meth:`accepts`.

Notes on the mapping from Qwen3-ASR semantics to the ASRModel ABC:

* Qwen3-ASR transcribes whole audio clips, one entry per input. We slice
  the parent ``AudioBundle`` per segment and submit them as a batch;
  vLLM handles the real GPU batching via ``max_inference_batch_size``.
* Qwen3-ASR returns full language names (``"Chinese"``, ``"English"``)
  while the rest of the pipeline speaks ISO codes.
  :data:`_NAME_TO_CODE` and :data:`_CODE_TO_NAME` keep the two in sync.
* ``detect_language`` is implemented by running a 1-sample transcribe with
  ``language=None`` and reading the detected language back. Probability
  isn't exposed by the model; we return 1.0 (callers that need a
  hard threshold should use :meth:`accepts` instead).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pipeline.config import Qwen3ASRParams
from pipeline.stages.base import ASRModel
from pipeline.types import AudioBundle, Segment

if TYPE_CHECKING:
    from qwen_asr import Qwen3ASRModel as _Qwen3ASRModel


# Qwen3-ASR's public name <-> ISO code. Extend as needed.
_CODE_TO_NAME: dict[str, str] = {
    "zh": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ro": "Romanian",
}
_NAME_TO_CODE: dict[str, str] = {v.lower(): k for k, v in _CODE_TO_NAME.items()}

# Qwen3-ASR also emits Chinese regional dialects / accents in the language
# slot. ``parse_asr_output`` only normalizes case — it does NOT validate
# against SUPPORTED_LANGUAGES — so these strings pass through verbatim in
# ``result.language``. Map each to its base ISO code so accepts() / routing
# treat them as Mandarin (zh) or Cantonese (yue); the verbatim label is kept
# separately in ``asr_lang_raw`` for dialect-level downstream use.
_DIALECT_TO_BASE: dict[str, str] = {
    # Mandarin regional accents
    "anhui": "zh", "dongbei": "zh", "fujian": "zh", "gansu": "zh",
    "guizhou": "zh", "hebei": "zh", "henan": "zh", "hubei": "zh",
    "hunan": "zh", "jiangxi": "zh", "ningxia": "zh", "shandong": "zh",
    "shaanxi": "zh", "shanxi": "zh", "sichuan": "zh", "tianjin": "zh",
    "yunnan": "zh", "zhejiang": "zh",
    # Sinitic languages the model tags at dialect granularity
    "wu language": "zh", "minnan language": "zh",
    # Cantonese regional accents (Cantonese itself is ISO yue)
    "cantonese (hong kong accent)": "yue",
    "cantonese (guangdong accent)": "yue",
}


def _normalize_lang_name(value: str | None) -> str:
    return (value or "").strip().lower()


def _lang_code_from_name(value: str | None, *, fallback: str | None = None) -> str:
    """Resolve Qwen3-ASR's language string to a base ISO code.

    Handles the three shapes the model emits in the language slot:
      * a canonical name ("Chinese" -> zh, "Cantonese" -> yue)
      * a code-switch list ("Chinese,English") -> route by the first lang
      * a regional dialect / accent ("Sichuan", "Cantonese (Hong Kong
        accent)") -> base code via ``_DIALECT_TO_BASE``
    The verbatim string is preserved by the caller in ``asr_lang_raw``;
    this only produces the routing code used by :meth:`accepts`.
    """
    name = _normalize_lang_name(value)
    if not name:
        return fallback or "unk"
    head = name.split(",", 1)[0].strip()  # code-switch: route by first lang
    if head in _NAME_TO_CODE:
        return _NAME_TO_CODE[head]
    if name in _DIALECT_TO_BASE:
        return _DIALECT_TO_BASE[name]
    # Enumerated Cantonese variants still route to yue.
    if "cantonese" in name:
        return "yue"
    # Unrecognized label. The old code assumed Qwen3-ASR emits non-canonical
    # labels ONLY for unenumerated Sinitic dialects and mapped anything else to
    # zh — but in practice the model returns plenty of out-of-set names for
    # non-Chinese audio ("Tagalog", "Swahili", "Hebrew", "Bengali",
    # "Brazilian_portuguese", even hallucinations like "Python"). Mapping those
    # to zh leaked foreign speech into the zh set (and, via the dialect floor,
    # under the 3.0 DNSMOS gate). Policy: anything whose detected language is
    # outside the predefined supported set is dropped. Return the "unknown"
    # sentinel so ``ASRModel.accepts()`` rejects it. The verbatim string is
    # still preserved by the caller in ``asr_lang_raw`` for diagnostics.
    return "unknown"


def _is_dialect_label(value: str | None) -> bool:
    """True if Qwen3-ASR's verbatim language label is a Sinitic dialect /
    regional accent rather than a canonical language name.

    Canonical names ("Chinese", "English", "Cantonese", or a code-switch list
    led by one) → False. Regional accents ("Sichuan"), dialect-grained Sinitic
    ("Wu language", "Minnan language"), accented Cantonese, or any other
    non-canonical label (which ``_lang_code_from_name`` collapses to zh/yue)
    → True. Phase 2 uses this to give scarce dialect data a looser DNSMOS
    floor than mainstream zh/en.
    """
    head = _normalize_lang_name(value).split(",", 1)[0].strip()
    if not head:
        return False
    return head not in _NAME_TO_CODE


class Qwen3ASR(ASRModel):
    SR = 16000  # Qwen3-ASR's expected audio sample rate.

    #: Languages the model knows about. Anything outside this set is a
    #: hallucinated label and should be dropped by Phase 2.
    SUPPORTED_LANGUAGES = frozenset(_CODE_TO_NAME.keys())

    def __init__(self, params: Qwen3ASRParams, device: str):
        self._params = params
        self._device = device  # vLLM picks GPUs via CUDA_VISIBLE_DEVICES
        self._model: "_Qwen3ASRModel | None" = None

    def accepts(self, language: str | None) -> bool:
        """Phase 2 keeps an utterance only when Qwen3-ASR knows the
        language and it is not in the explicit skip list.

        ``ms`` (Malay) and ``id`` (Indonesian) are commonly confused with
        each other and with several South-East-Asian languages on this
        model; the user policy is to skip them outright (a future
        language-specific ASR adapter would simply not have them in its
        own ``skip_languages``).
        """
        return bool(
            language
            and language != "unknown"
            and language not in self._params.skip_languages
            and language in self.SUPPORTED_LANGUAGES
        )

    def warmup(self) -> None:
        if self._model is not None:
            return
        # Imported lazily so projects that don't use Qwen3-ASR don't pay
        # the vLLM import cost.
        import torch
        from qwen_asr import Qwen3ASRModel

        kwargs = dict(
            model=self._params.model_path,
            gpu_memory_utilization=self._params.gpu_memory_utilization,
            max_inference_batch_size=self._params.max_inference_batch_size,
            max_new_tokens=self._params.max_new_tokens,
            enforce_eager=self._params.enforce_eager,
        )
        if self._params.max_model_len is not None:
            kwargs["max_model_len"] = self._params.max_model_len
        if self._params.max_num_batched_tokens is not None:
            # vLLM forwards extra kwargs to LLMEngineArgs; chunked
            # prefill picks this up there.
            kwargs["max_num_batched_tokens"] = self._params.max_num_batched_tokens
        if self._params.forced_aligner_path:
            kwargs["forced_aligner"] = self._params.forced_aligner_path
            kwargs["forced_aligner_kwargs"] = dict(
                dtype=getattr(torch, self._params.forced_aligner_dtype),
                device_map=self._device if ":" in self._device else f"{self._device}:0",
            )
        self._model = Qwen3ASRModel.LLM(**kwargs)

    # ----- ASRModel API ------------------------------------------------

    def detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        """Run a one-shot transcribe with auto-detect and return ``(code, 1.0)``.

        Qwen3-ASR does not expose a separate confidence score, so we return
        1.0; Phase 2's :meth:`accepts` (or its caller's language whitelist)
        is the real gate.
        """
        if self._model is None:
            raise RuntimeError("call warmup() before detect_language()")
        results = self._model.transcribe(
            audio=[(audio.astype(np.float32, copy=False), self.SR)],
            language=[None],
            return_time_stamps=False,
        )
        return _lang_code_from_name(results[0].language), 1.0

    def transcribe(
        self,
        audio: AudioBundle,
        segments: list[Segment],
        language: str | None,
        batch_size: int,  # noqa: ARG002 - ABC contract; vLLM handles real batching
    ) -> list[Segment]:
        """Transcribe each segment by slicing the parent waveform.

        ``batch_size`` is accepted for ABC compatibility; the real
        batching knob lives inside the vLLM engine and is configured via
        ``max_inference_batch_size`` in :class:`Qwen3ASRParams`.
        """
        if self._model is None:
            raise RuntimeError("call warmup() before transcribe()")
        if not segments:
            return []

        # Resample to 16k via the bundle's shared cache.
        waveform_f32 = audio.get_at_sr(self.SR).astype(np.float32, copy=False)

        clips: list[tuple[np.ndarray, int]] = []
        for s in segments:
            start = int(s.start * self.SR)
            end = int(s.end * self.SR)
            clips.append((waveform_f32[start:end], self.SR))

        # S4: always auto-detect — never force the LID language as a decode
        # hint. A wrong LID (e.g. a small language misread as zh) would force a
        # garbage transcription that then passes the filters; trusting
        # Qwen3-ASR's own detection avoids that. The incoming ``language``
        # (the LID routing bucket) is used ONLY as a resolution fallback for an
        # unrecognized detected label (see ``_lang_code_from_name``).
        results = self._model.transcribe(
            audio=clips,
            language=[None] * len(clips),
            return_time_stamps=False,
        )
        if len(results) != len(segments):
            raise RuntimeError(
                "Qwen3-ASR result count mismatch: "
                f"expected {len(segments)}, got {len(results)}"
            )

        out: list[Segment] = []
        for seg, r in zip(segments, results):
            detected_code = _lang_code_from_name(r.language, fallback=language)
            seg_extra = dict(seg.extra)
            # Preserve Qwen3-ASR's verbatim language string (dialect /
            # accent / code-switch list) so downstream can filter at
            # dialect granularity even though ``language`` is collapsed to
            # a base ISO code for routing / accepts.
            if r.language:
                seg_extra["asr_lang_raw"] = r.language
                # Flag Sinitic dialects/accents so Phase 2 can apply the
                # looser dialect DNSMOS floor (scarce data) instead of the
                # strict zh/en floor.
                if _is_dialect_label(r.language):
                    seg_extra["is_dialect"] = True
            out.append(
                Segment(
                    start=seg.start,
                    end=seg.end,
                    speaker=seg.speaker,
                    index=seg.index,
                    text=r.text,
                    language=detected_code,
                    extra=seg_extra,
                )
            )
        return out

    def release(self) -> None:
        self._model = None
