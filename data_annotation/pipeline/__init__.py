"""Continuo-Pipe abstraction layer: stages, adapters, orchestrator."""

from pipeline.context import PipelineContext
from pipeline.orchestrator import Pipeline
from pipeline.types import AudioBundle, DiarizationFrame, Segment

__all__ = [
    "Pipeline",
    "PipelineContext",
    "AudioBundle",
    "DiarizationFrame",
    "Segment",
]
