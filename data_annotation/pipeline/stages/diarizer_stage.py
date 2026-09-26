"""Speaker-diarization stage: thin orchestration layer over a SpeakerDiarizer."""

from __future__ import annotations

from pipeline.context import PipelineContext
from pipeline.stages.base import SpeakerDiarizer, Stage
from utils.logger import time_logger


class DiarizerStage(Stage):
    name = "diarizer"

    def __init__(self, diarizer: SpeakerDiarizer):
        self._diarizer = diarizer

    def warmup(self) -> None:
        self._diarizer.warmup()

    def release(self) -> None:
        self._diarizer.release()

    @time_logger
    def run(self, ctx: PipelineContext) -> None:
        assert ctx.audio is not None
        ctx.diarization = self._diarizer.diarize(ctx.audio)
