"""Phase-1 quality filter: mark short segments for drop without removing them.

Lives between scoring/LID and LongChunkStage. Unlike the legacy filter,
this one **does not** prune ``ctx.segments``: it only writes
``extra["kept"]`` and ``extra["dropped_reason"]`` on each short. The
short list flows on to LongChunkStage so the merger can build a
long-chunk that spans low-DNSMOS material when the aggregate
quality is acceptable. The exporter is what actually splits the list
into ``partial.json`` (kept-only) and ``all_shorts.json`` (everything).

Per-language thresholds:

* ``zh`` / ``en``: ``filter.min_quality_overrides`` (default 3.0).
  The optional external synthetic-speech flag is ignored by default.
* every other language (including ``unknown``): ``filter.min_quality``
  (default 2.4).

Drop reasons (in priority order, so each short carries at most one):

  ``duration_too_short`` | ``duration_too_long`` | ``missing_dnsmos`` |
  ``low_dnsmos`` | ``fake_zh_en``

Text-length / character-rate filtering moves to Phase 2 where ``text``
is available.
"""

from __future__ import annotations

from pipeline.config import FilterConfig
from pipeline.context import PipelineContext
from pipeline.stages.base import Stage
from utils.logger import Logger


class QualityFilterStage(Stage):
    name = "quality_filter"

    def __init__(self, criteria: FilterConfig):
        self._criteria = criteria
        self._logger = Logger.get_logger()

    def _threshold_for(self, language: str | None) -> float:
        lang = (language or "").strip().lower()
        return float(
            self._criteria.min_quality_overrides.get(lang, self._criteria.min_quality)
        )

    def run(self, ctx: PipelineContext) -> None:
        if not ctx.segments:
            return

        min_d = self._criteria.min_duration
        max_d = self._criteria.max_duration
        drop_fake = self._criteria.drop_fake_zh_en

        reasons: dict[str, int] = {}
        for seg in ctx.segments:
            seg.extra.setdefault("is_fake", None)
            seg.extra.setdefault("deepfake_score", None)
            dur = seg.end - seg.start
            language = (seg.language or "").strip().lower()
            min_q = self._threshold_for(language)
            reason: str | None = None
            if dur < min_d:
                reason = "duration_too_short"
            elif dur > max_d:
                reason = "duration_too_long"
            elif seg.quality is None:
                # Scoring should have populated this; if not, the
                # pipeline is misconfigured. Drop conservatively.
                reason = "missing_dnsmos"
            elif seg.quality < min_q:
                reason = "low_dnsmos"
            elif (
                drop_fake
                and language in ("zh", "en")
                and bool(seg.extra.get("is_fake"))
            ):
                reason = "fake_zh_en"

            seg.extra["kept"] = (reason is None)
            if reason is not None:
                seg.extra["dropped_reason"] = reason
                reasons[reason] = reasons.get(reason, 0) + 1

        # Crucially, ``segments_all`` and ``segments`` are now the same
        # list — there's no separate filtered subset at this stage. The
        # exporter splits kept-vs-dropped at write time.
        ctx.segments_all = ctx.segments

        n_dropped = sum(reasons.values())
        if ctx.segments:
            self._logger.debug(
                f"quality_filter > {n_dropped}/{len(ctx.segments)} "
                f"({n_dropped / len(ctx.segments):.0%}) marked drop "
                f"reasons={reasons}"
            )
