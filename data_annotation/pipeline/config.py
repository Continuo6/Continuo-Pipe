"""Pydantic v2 configuration schema for the pipeline.

This is a clean new schema (no backward compat with the original
``example_config.json``). Each stage's config is a tagged union of the form
``{"name": "<adapter-key>", "params": {...}}`` so adding a new model is one
new params class + one line in the registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Per-model param classes. One class per adapter; ``model_config = strict``
# means typos raise instead of silently being ignored.
# ---------------------------------------------------------------------------


# Shared dtype enum for adapters that expose precision as a knob; resolved by
# ``utils.tool.resolve_dtype``.
DtypeLiteral = Literal["fp32", "fp16", "bf16"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MelBandRoformerKimFT3Params(_StrictModel):
    """MelBand Roformer Kim FT3 via the ``audio_separator`` library.

    ``model_filename`` is the canonical Kim FT3 checkpoint; ``model_file_dir``
    is where the library caches downloaded weights when left empty.
    """

    model_filename: str = "mel_band_roformer_kim_ft3_unwa.ckpt"
    model_file_dir: str = ""
    normalization_threshold: float = 0.9
    use_autocast: bool = False


class DiariZenParams(_StrictModel):
    """DiariZen (BUT-FIT WavLM-large s80) speaker diarization.

    ``cache_dir`` pins the HF hub cache directory. When set, the pipeline
    runs in ``local_files_only=True`` mode, so all model files must
    already be present under that dir (both the diarizen checkpoint and
    the ``pyannote/wespeaker-voxceleb-resnet34-LM`` embedding model). Leave
    empty to rely on the default HF resolution (``HF_HOME``) and allow
    online fetches.

    ``embedding_batch_size`` / ``segmentation_batch_size`` override the
    default (32 from the checkpoint config) on the pyannote pipeline. The
    embedding stage is GPU-bound and benefits from a larger batch on
    long-form audio that contributes many (chunk, speaker) slots — the
    upstream BUTSpeechFIT optimization repo recommends bumping it to 128.
    """

    model: str = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
    cache_dir: str = ""
    embedding_batch_size: int = 128
    segmentation_batch_size: int = 32


class FireRedVADParams(_StrictModel):
    """FireRedVAD (FireRedTeam/FireRedVAD) DFSMN-based VAD.

    All ``*_frame`` values are 10 ms-frames (kaldi default).

    * ``max_speech_frame=3000`` → 30 s cap per speech segment
    * ``min_silence_frame=20`` (200 ms) → fast hysteresis at the sentence end
    * ``extend_speech_frame=0`` → clean boundaries; no leak into the next
      utterance's leading consonant"""

    source_dir: str = "third_party/FireRedASR2S"
    model_dir: str
    smooth_window_size: int = 5
    speech_threshold: float = 0.5
    min_speech_frame: int = 20
    max_speech_frame: int = 3000     # 30 s cap → sentence-grain output
    min_silence_frame: int = 20

    merge_silence_frame: int = 300
    extend_speech_frame: int = 0
    chunk_max_frame: int = 30000     # 300 s VAD window stride


class FireRedLIDParams(_StrictModel):
    """FireRedLID (FireRedTeam/FireRedLID) per-segment language identifier.

    ``source_dir`` is the path to a cloned ``FireRedASR2S`` source tree; the
    upstream package is not pip-installable against our env (its
    pyproject pins torch 2.10 / transformers 5.1) so we add it to
    ``sys.path`` at warmup time.

    ``model_dir`` is the HF snapshot directory containing ``model.pth.tar``,
    ``cmvn.ark``, and ``dict.txt`` — typically a path under
    ``<HF_HOME>/hub/models--FireRedTeam--FireRedLID/snapshots/<rev>/``.
    """

    source_dir: str = "third_party/FireRedASR2S"
    model_dir: str
    use_half: bool = Field(
        False,
        description="Legacy fp16 toggle; prefer ``dtype`` (supports bf16). ``dtype`` wins.",
    )
    dtype: DtypeLiteral | None = Field(
        None,
        description=(
            "Cast the LID conformer encoder + AED decoder + per-clip "
            "fbank features to the given dtype. ``bf16`` requires "
            "Ampere (sm_80+) or newer. When None, falls back to ``use_half``."
        ),
    )
    max_batch_size: int = Field(
        16,
        description=(
            "Max segments per LID forward pass. The conformer encoder's "
            "attention is O(L^2 x batch); a long-form file can yield 100+ "
            "segments which OOMs a 24 GB GPU when fed in one call — and "
            "upstream's bare ``except`` swallows the OOM into empty labels "
            "(every segment silently tagged 'unknown'). Mini-batching keeps "
            "memory bounded. Lower if very long clips still OOM."
        ),
    )


class Qwen3ASRParams(_StrictModel):
    """Qwen3-ASR served via the vLLM backend.

    ``model_path`` and ``forced_aligner_path`` accept HF repo IDs or local
    directories. Leave ``forced_aligner_path`` empty when timestamps are
    not needed; that saves the ~600M-param companion model.
    """

    model_path: str = "Qwen/Qwen3-ASR-1.7B"
    gpu_memory_utilization: float = 0.7
    max_inference_batch_size: int = 128
    max_new_tokens: int = 4096
    max_model_len: int | None = Field(
        32768,
        description=(
            "Token budget per request. The model's native max is 65536 but that "
            "needs ~7 GiB of KV cache, which doesn't fit on a 24 GiB card at the "
            "default gpu_memory_utilization. 32768 tokens still covers very long "
            "audio segments (Qwen3-ASR consumes ~1 token per ~20 ms of audio), "
            "while halving the KV-cache requirement. Set to None to use the model default."
        ),
    )
    enforce_eager: bool = Field(
        False,
        description=(
            "If True, skip vLLM's CUDA-graph capture. ~30% slower per-call but "
            "cuts startup from minutes to seconds. Useful for short jobs or smoke "
            "tests; leave False for long-running production batches."
        ),
    )
    max_num_batched_tokens: int | None = Field(
        None,
        description=(
            "vLLM chunked-prefill chunk size. Default (None) uses vLLM's own "
            "default of 8192 — at ~50 tokens/s of audio, that caps a single "
            "request's prefill chunk at ~2.7 minutes of audio. Bigger values "
            "let longer single-call audio fit in one scheduling round; we "
            "recommend 32768 (~10 min/chunk) when running whole-chunk "
            "fallback on long-track audio. Set proportional to your KV cache "
            "budget — too large and the engine spends warmup compiling huge "
            "graphs."
        ),
    )
    forced_aligner_path: str = ""
    forced_aligner_dtype: str = "bfloat16"
    skip_languages: list[str] = Field(
        default_factory=lambda: ["ms", "id"],
        description=(
            "ISO codes that this adapter recognizes but Phase 2 should still drop. "
            "Default skips Malay and Indonesian because they're commonly confused with "
            "neighbouring South-East-Asian languages on Qwen3-ASR; set to [] if you "
            "want to keep them."
        ),
    )


class IndicConformerASRParams(_StrictModel):
    """AI4Bharat IndicConformer-600M-Multilingual (CTC / RNNT)."""

    model_path: str = "ai4bharat/indic-conformer-600m-multilingual"
    revision: str = ""
    decoder_type: Literal["ctc", "rnnt"] = Field(
        "ctc",
        description=(
            "Which decoder head to use. CTC is faster and the default the "
            "pipeline integrates for; RNNT is slightly more accurate "
            "but iterates token-by-token (slower wall time)."
        ),
    )
    default_language: str = Field(
        "bn",
        description=(
            "ISO-639-1/3 code used when a segment arrives with no LID hint "
            "(or with ``unknown``). Must be one of the 22 supported Indic "
            "codes: as, bn, brx, doi, gu, hi, kn, kok, ks, mai, ml, mni, "
            "mr, ne, or, pa, sa, sat, sd, ta, te, ur."
        ),
    )
    skip_languages: list[str] = Field(
        default_factory=list,
        description=(
            "Indic codes recognised by the model but to drop in Phase 2 "
            "anyway. Default empty — keep everything the model supports."
        ),
    )
    raise_on_segment_error: bool = Field(
        False,
        description=(
            "If True, an exception in a single-segment ``model(wav, lang, "
            "decoder)`` call aborts the entire transcribe() batch. Default "
            "False: log a warning and emit empty text for that segment, "
            "which downstream Phase 2's drop_text filter handles."
        ),
    )


class DNSMOSScorerParams(_StrictModel):
    model_path: str


class RatioBounds(_StrictModel):
    """Allowed range for ``duration_sec / char_count`` of an utterance.

    Set very wide on purpose — this filter catches *obvious* misalignment
    cases (ASR hallucinations producing 80-char strings from 1 s of
    audio; or 30 s of audio yielding a single character because most of
    it was non-speech the diarizer let through). It is not a quality
    filter; pair it with DNSMOS / VAD / min_char_count for that.
    """

    min: float = Field(
        0.015,
        description=(
            "Lower bound in seconds per character (i.e. duration / char_count). "
            "Below this means the text is suspiciously long for the audio — "
            "the usual culprit is the ASR repeating a token. 0.015 s/char on "
            "Latin text corresponds to ~67 chars/sec which only happens in "
            "garbled output."
        ),
    )
    max: float = Field(
        1.0,
        description=(
            "Upper bound in seconds per character. Above this means the text "
            "is suspiciously short for the audio — usually a long silent / "
            "non-speech tail. 1.0 s/char ≈ 60 chars/min."
        ),
    )


# ---------------------------------------------------------------------------
# Discriminated stage configs. ``name`` selects which adapter to build; the
# registry then casts ``params`` to the right pydantic model.
# ---------------------------------------------------------------------------


class StageRef(_StrictModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class StagesConfig(_StrictModel):
    separator: StageRef
    diarizer: StageRef
    vad: StageRef
    scorer: StageRef
    lid: StageRef | None = Field(
        None,
        description=(
            "Phase 1: optional language-ID stage placed after the quality-filter. "
            "When set, each surviving segment gets segment.language populated up "
            "front and Phase 2 uses it as the transcription language hint."
        ),
    )
    deepfake: None = Field(
        None,
        description="Optional synthetic-speech detector stage; unset by default.",
    )

    asr: StageRef | None = Field(
        None,
        description=(
            "Phase 2 only: which ASR adapter transcribe.py runs. main.py (Phase 1) "
            "ignores this field. Leave unset if you only need feature extraction "
            "without transcription."
        ),
    )


# ---------------------------------------------------------------------------
# Cross-cutting blocks.
# ---------------------------------------------------------------------------


class LanguageConfig(_StrictModel):
    multilingual: bool = True
    supported: list[str] = Field(default_factory=list)
    detect_threshold_multilingual: float = 0.5
    detect_threshold_single: float = 0.8
    lid_min_confidence: float = Field(
        0.8,
        description=(
            "Minimum LID confidence to trust the upstream label. Segments below "
            "this get language='unknown' (raw label + confidence preserved under "
            "extra['lid_*']); the ASR stage then falls back to its own per-segment "
            "language detection on those."
        ),
    )


class IOConfig(_StrictModel):
    input_folder_path: str
    sample_rate: int = 24000
    # One shared carrier per recording; manifests hold sample windows.
    export_format: Literal["wav", "flac", "m4a"] = "wav"
    m4a_bitrate: str = "128k"
    #: Keep two channels through Standardization → Separator. The Kim FT3
    #: separator natively wants ``(2, n)`` input; by default we collapse to
    #: mono in Standardization and the separator then re-duplicates
    #: mono→dual-mono — pointless when the source is already stereo. With
    #: ``preserve_stereo=True`` a stereo source stays stereo (a mono source
    #: is still duplicated) and the separator runs on real stereo, then
    #: downmixes its vocals to mono on output — so every downstream stage
    #: is unchanged. Default False = legacy mono-then-duplicate behavior,
    #: so a live run is unaffected unless you opt in.
    preserve_stereo: bool = False


class OverlapConfig(_StrictModel):


    enabled: bool = True

    max_overlap_s: float = 2.0

    max_overlap_share: float = 0.40


class SegmentationConfig(_StrictModel):
    """Post-VAD cut/merge knobs (legacy: cut_by_speaker_label).

    Operates on the VAD's per-frame output: merges **all** adjacent same-speaker
    segments separated by < ``merge_gap`` (up to ``max_segment_length``), drops
    anything still shorter than ``min_segment_length``, force-splits anything
    longer than ``max_segment_length``."""


    merge_gap: float = 3.0
    min_segment_length: float = 0.5
    max_segment_length: float = 30.0


    overlap: OverlapConfig = Field(default_factory=OverlapConfig)

    def overlap_policy(self):

        from overlap_policy import OverlapPolicy

        return OverlapPolicy(
            max_overlap_s=self.overlap.max_overlap_s,
            max_overlap_share=self.overlap.max_overlap_share,
        )


class LongChunkConfig(_StrictModel):
    """Greedy merger that combines adjacent same-speaker short segments
    into longer chunks for long-form downstream use (long-form ASR /
    diarized recording archives).

    Single-speaker is enforced — a speaker change always breaks the
    chunk. Within a speaker, segments are merged while the time gap to
    the next segment is ≤ ``max_gap_s``. The previous ``max_duration_s``
    upper bound was removed (effectively infinity): long chunks can be
    any length the speaker holds the floor for. Phase 2's long-track
    ASR pipeline handles arbitrarily-long chunks via a per-member
    walk, with concat-and-skip patching + split-then-refilter on
    member-level failures.

    Emission filters (applied after the merge):

    * ``min_duration_s`` — drop chunks shorter than this. A "long chunk"
      that's only 3 s isn't a long chunk; it's just a single short
      bracketed by speaker changes. 30 s default keeps the long track
      meaningfully distinct from the short track.
    * ``min_mean_dnsmos`` — drop chunks whose source shorts have a low
      average DNSMOS. LongChunkStage sees all shorts, including those
      marked for exclusion from the short track, and checks their
      aggregate quality here.

    Set ``enabled = false`` to skip emitting long chunks entirely.
    """

    enabled: bool = True
    #: Hard cap on a single long chunk's duration. Default ``inf`` —
    #: no upper bound (a 2-hour monologue becomes one 2-hour long
    #: chunk). Phase 2 deals with that. Set to a finite value if your
    #: ASR backend can't handle very long single-call audio and you
    #: want to force splits at the merge stage instead.
    max_duration_s: float = float("inf")
    max_gap_s: float = 2.0             # max silence to bridge while merging
    min_duration_s: float = 30.0       # drop merged chunks shorter than this
    min_mean_dnsmos: float = 2.4       # drop chunks whose member shorts avg below this
    max_fake_coverage: float | None = None


class FilterConfig(_StrictModel):
    """Final segment filter knobs.

    ``min_duration`` / ``max_duration`` / per-language ``min_quality`` apply
    in Phase 1 (``QualityFilterStage``, which now *marks* instead of removes
    so LongChunkStage and the exporter both see the full short list);
    ``min_char_count`` and ``ratio_filter`` apply in Phase 2
    (``transcribe.py``) after ASR text is available.

    DNSMOS thresholds are split by language because the requirement on a
    Chinese / English short is stricter (zh+en is what most downstream
    TTS use, and the cleaner the better) than on the long-tail languages
    where data is scarce and a 2.4 floor is already aggressive.
    """

    min_duration: float = 0.5
    max_duration: float = 30.0
    min_quality: float = Field(
        2.4,
        description=(
            "Default DNSMOS floor used for every language that doesn't have "
            "an explicit override in ``min_quality_overrides``. 2.4 is the "
            "long-tail default; zh/en is overridden up to 3.0 below."
        ),
    )
    min_quality_overrides: dict[str, float] = Field(
        default_factory=lambda: {"zh": 3.0, "en": 3.0},
        description=(
            "Per-language DNSMOS overrides. A short's effective floor is "
            "``min_quality_overrides.get(seg.language, min_quality)``. "
            "Default tightens zh/en to 3.0 since they're the most-used "
            "languages downstream."
        ),
    )
    drop_fake_zh_en: bool = False
    dialect_dnsmos_floor: float = Field(
        2.4,
        description=(
            "DNSMOS floor for the Phase-2 post-ASR re-filter when the ASR "
            "resolved a short to a Chinese *dialect* / regional accent "
            "(Sichuanese, Wu, Minnan, accented Cantonese, ...) rather than "
            "canonical Mandarin. Dialect data is scarce, so it gets the "
            "looser long-tail floor (2.4) instead of the strict zh/en 3.0. "
            "Only applies when ASR flips a non-zh/en LID short to zh; see "
            "transcribe.py's post-ASR gate."
        ),
    )
    min_char_count: int = 2

    ratio_filter: dict[str, RatioBounds] = Field(
        default_factory=lambda: {
            # Latin / Cyrillic / Arabic / Indic / European default:
            # each char is sub-syllable so seconds-per-char is small.
            "default": RatioBounds(min=0.015, max=1.0),
            # CJK + Korean + Thai: one "char" is roughly a syllable, so
            # seconds-per-char is naturally bigger. Bounds are wider on
            # both sides to stay lenient.
            "zh":  RatioBounds(min=0.04, max=2.0),
            "yue": RatioBounds(min=0.04, max=2.0),
            "ja":  RatioBounds(min=0.04, max=2.0),
            "ko":  RatioBounds(min=0.04, max=2.0),
            "th":  RatioBounds(min=0.04, max=2.0),
        },
        description=(
            "Per-language bounds for the duration/char_count ratio applied "
            "in Phase 2. Key 'default' is the fallback for languages with "
            "no explicit entry. Tune wider, not tighter — this filter is "
            "for catching obviously broken ASR outputs, not for trimming "
            "borderline-quality utterances."
        ),
    )


class RuntimeConfig(_StrictModel):
    batch_size: int = 16
    global_size: int = 1
    local_index: int = 0
    max_audio_hours: float = 5.0


class DialogueWindowConfig(_StrictModel):


    enabled: bool = False


    max_gap_s: float = 6.0
    min_duration_s: float = 30.0
    min_turns: int = 2  # floor; evaluate() also requires >= actual speaker count
    max_avg_turn_s: float = 60.0
    max_speakers: int = 6  # reduced from the original 10
    min_mean_dnsmos: float = 2.4
    max_fake_coverage: float | None = None
    max_overlap_coverage: float = 0.05
    min_secondary_share: float = 0.15
    min_reuse_share: float = 0.05

    def gates(self):

        from dialogue.select import Gates
        return Gates(
            max_gap_s=self.max_gap_s,
            min_duration_s=self.min_duration_s,
            min_turns=self.min_turns,
            max_avg_turn_s=self.max_avg_turn_s,
            max_speakers=self.max_speakers,
            min_mean_dnsmos=self.min_mean_dnsmos,
            max_fake_coverage=self.max_fake_coverage,
            max_overlap_coverage=self.max_overlap_coverage,
            min_secondary_share=self.min_secondary_share,
            min_reuse_share=self.min_reuse_share,
        )


class PipelineConfig(_StrictModel):
    io: IOConfig
    language: LanguageConfig = Field(default_factory=LanguageConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    long_chunk: LongChunkConfig = Field(default_factory=LongChunkConfig)
    dialogue_window: DialogueWindowConfig = Field(default_factory=DialogueWindowConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    stages: StagesConfig

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Copy example_config.json to {p} and fill it in."
            )
        with p.open() as f:
            raw = json.load(f)
        return cls.model_validate(raw)


# Public mapping used by the registry to coerce ``params`` dicts into
# typed pydantic models. Keys MUST match the corresponding key in
# ``pipeline.registry``.
PARAM_MODELS: dict[str, type[_StrictModel]] = {
    "mel_band_roformer_kim_ft3": MelBandRoformerKimFT3Params,
    "diarizen": DiariZenParams,
    "firered_vad": FireRedVADParams,
    "firered_lid": FireRedLIDParams,
    "qwen3_asr": Qwen3ASRParams,
    "indic_conformer": IndicConformerASRParams,
    "dnsmos": DNSMOSScorerParams,
}
