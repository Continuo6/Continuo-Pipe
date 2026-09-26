"""Shared Phase-1 pipeline composition.

One ``build_pipeline`` function used by both ``main.py`` (production entry)
and any benchmarking harness, so the stage order can't drift between them.
ASR is intentionally absent — that runs in Phase 2 via ``transcribe.py``.
"""

from __future__ import annotations

from pipeline.config import PipelineConfig
from pipeline.orchestrator import Pipeline
from pipeline.registry import build_adapter
from pipeline.stages.dialogue_window_stage import DialogueWindowStage
from pipeline.stages.diarizer_stage import DiarizerStage
from pipeline.stages.exporter_stage import ExporterStage
from pipeline.stages.lid_stage import LanguageIDStage
from pipeline.stages.long_chunk_stage import LongChunkStage
from pipeline.stages.quality_filter_stage import QualityFilterStage
from pipeline.stages.scoring_stage import ScoringStage
from pipeline.stages.separator_stage import SeparatorStage
from pipeline.stages.standardization import StandardizationStage
from pipeline.stages.vad_stage import VADStage


def build_pipeline(cfg: PipelineConfig, device: str) -> Pipeline:
    """Compose the Phase-1 feature-extraction pipeline.

    Order (LID is optional, gated on its config slot):

        Standardize → Separator → Diarizer → VAD
        → Scoring (DNSMOS)
        → [optional] LID
        → QualityFilter (mark-only: drops never leave ctx.segments)
        → LongChunk (sees the full marked short list; min_duration_s /
            min_mean_dnsmos gates)
        → Exporter (writes 4 manifests + one shared M4A carrier)

    Note the QualityFilter + LongChunk swap relative to the previous
    pipeline shape: the filter now MARKS instead of removes, so the
    long-track merge can run after it and still see every short. The
    exporter handles the actual kept-vs-dropped split at write time.
    """
    separator = build_adapter("separator", cfg.stages.separator, device)
    diarizer = build_adapter("diarizer", cfg.stages.diarizer, device)
    vad = build_adapter("vad", cfg.stages.vad, device)
    scorer = build_adapter("scorer", cfg.stages.scorer, device)

    stages: list = [
        StandardizationStage(
            sample_rate=cfg.io.sample_rate,
            max_audio_hours=cfg.runtime.max_audio_hours,
            mono=not cfg.io.preserve_stereo,
        ),
        SeparatorStage(separator),
        DiarizerStage(diarizer),
        VADStage(vad, segmentation=cfg.segmentation),
        ScoringStage(scorer),
    ]
    # LID feeds language tags downstream, and
    # QualityFilter uses per-language DNSMOS thresholds. Skipping LID is
    # still allowed; the filter then treats every short as the
    # ``default`` (non-zh/en) bucket.
    if cfg.stages.lid is not None:
        lid = build_adapter("lid", cfg.stages.lid, device)
        stages.append(LanguageIDStage(
            lid, min_confidence=cfg.language.lid_min_confidence,
        ))
    stages.append(QualityFilterStage(criteria=cfg.filter))
    stages.append(LongChunkStage(cfg.long_chunk))


    if cfg.dialogue_window.enabled:


        stages.append(DialogueWindowStage(
            gates=cfg.dialogue_window.gates(),
            carrier_fmt=cfg.io.export_format,
        ))
    stages.append(ExporterStage(
        fmt=cfg.io.export_format,
        bitrate=cfg.io.m4a_bitrate,
    ))
    return Pipeline(stages=stages)
