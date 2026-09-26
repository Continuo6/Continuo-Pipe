"""Source-separation stage: thin orchestration layer over a SourceSeparator."""

from __future__ import annotations

from pipeline.context import PipelineContext
from pipeline.stages.base import SourceSeparator, Stage
from utils.logger import time_logger


class SeparatorStage(Stage):
    name = "separator"

    def __init__(self, separator: SourceSeparator):
        self._separator = separator

    def warmup(self) -> None:
        self._separator.warmup()

    def release(self) -> None:
        self._separator.release()

    @time_logger
    def run(self, ctx: PipelineContext) -> None:
        assert ctx.audio is not None
        ctx.audio = self._separator.separate(ctx.audio)
