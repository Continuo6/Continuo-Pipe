"""Quality-scoring stage: thin orchestration layer over a QualityScorer."""

from __future__ import annotations

from pipeline.context import PipelineContext
from pipeline.stages.base import QualityScorer, Stage
from utils.logger import Logger, time_logger


class ScoringStage(Stage):
    name = "scoring"

    def __init__(self, scorer: QualityScorer):
        self._scorer = scorer
        self._logger = Logger.get_logger()

    def warmup(self) -> None:
        self._scorer.warmup()

    def release(self) -> None:
        self._scorer.release()

    @time_logger
    def run(self, ctx: PipelineContext) -> None:
        assert ctx.audio is not None
        avg, segments = self._scorer.score(ctx.audio, ctx.segments)
        ctx.avg_quality = avg
        ctx.segments = segments
        self._logger.debug(f"avg quality for whole audio: {avg}")
