"""Greedy merger: short single-speaker segments → long single-speaker chunks.

Operates on the short list after VAD, scoring and quality-filter marking.
Aggregate quality is also enforced by the ``min_mean_dnsmos`` floor below.

Walks the segments in time order and extends the current chunk while:

* the speaker hasn't changed, AND
* the time gap to the next short segment is ≤ ``max_gap_s``, AND
* the resulting chunk duration would stay ≤ ``max_duration_s``.

A speaker change always forces a new chunk, so every long chunk is
guaranteed single-speaker.

After greedy merging, two emission filters apply:

* duration ≥ ``min_duration_s`` — a "long chunk" should actually be long.
  Without this, a single short bracketed by speaker changes becomes a
  same-length "long" chunk that adds nothing.
* mean DNSMOS of source shorts ≥ ``min_mean_dnsmos`` — aggregate-quality
  check. Per-short DNSMOS is available because Scoring already ran;
  stretches whose member shorts average below the floor are dropped.

Each surviving long chunk has ``source_indices`` into ``ctx.segments``
at the point this stage runs and ``mean_dnsmos`` in the ``extra`` dict.

The stage doesn't touch ``ctx.segments`` — short and long live side by
side in the context. Set ``cfg.long_chunk.enabled = false`` to skip it.
"""

from __future__ import annotations


import overlap_policy
from pipeline.config import LongChunkConfig
from pipeline.context import PipelineContext
from pipeline.stages.base import Stage
from pipeline.types import Segment
from utils.logger import Logger


class LongChunkStage(Stage):
    name = "long_chunk"

    def __init__(self, cfg: LongChunkConfig):
        self._cfg = cfg
        self._logger = Logger.get_logger()

    def run(self, ctx: PipelineContext) -> None:
        if not self._cfg.enabled:
            return
        if not ctx.segments:
            return

        # Sort by start so the greedy walk doesn't depend on upstream order.
        ordered = sorted(
            enumerate(ctx.segments), key=lambda ix_seg: ix_seg[1].start
        )

        all_chunks: list[Segment] = []      # pre-filter, annotated
        kept_chunks: list[Segment] = []     # post-filter, retains current
                                            # ``L0000`` index numbering for
                                            # backward compat with consumers
                                            # of long_chunks.json
        cur_start = ordered[0][1].start
        cur_end = ordered[0][1].end
        cur_speaker = ordered[0][1].speaker
        cur_sources: list[int] = [ordered[0][0]]

        def _flush() -> None:
            # Aggregate-quality gate uses the per-short DNSMOS values that
            # Scoring assigned. A source short with no dnsmos (shouldn't
            # happen post-Scoring, defensive) contributes nothing.
            mos_weighted = [
                (float(ctx.segments[i].quality), float(ctx.segments[i].extra.get(
                    "speech_s", ctx.segments[i].end - ctx.segments[i].start)))
                for i in cur_sources if ctx.segments[i].quality is not None
            ]
            mos_secs = sum(max(0.0, w) for _, w in mos_weighted)
            mean_mos = (sum(v * max(0.0, w) for v, w in mos_weighted) / mos_secs
                        if mos_secs else 0.0)
            duration = cur_end - cur_start

            # ``members`` is a Phase 2 affordance: each member carries
            # absolute timestamps + per-short DNSMOS + the original VAD
            # ``index`` so the long-track ASR can address members
            # directly (no per-long WAV; member WAVs are what's on disk).
            # The synthetic-speech fields stay null unless a detector stage
            # supplies them.
            members = []
            for i in cur_sources:
                s = ctx.segments[i]
                members.append({
                    "index": s.index,
                    "start": s.start,
                    "end": s.end,
                    "speech_s": s.extra.get("speech_s", s.end - s.start),
                    "dnsmos": s.quality,
                    "speaker": s.speaker,
                    "language": s.language,
                    "is_fake": s.extra.get("is_fake"),
                    "deepfake_score": s.extra.get("deepfake_score"),
                })

            # Preserve an unevaluated state when no external classifier ran.
            evaluated = any(m["is_fake"] is not None for m in members)
            fake_secs = sum(
                m["end"] - m["start"] for m in members if m["is_fake"] is True
            )
            fake_ratio = ((fake_secs / duration) if duration > 0 else 0.0) if evaluated else None

            reason: str | None = None
            if duration < self._cfg.min_duration_s:
                reason = "duration_too_short"
            elif mean_mos < self._cfg.min_mean_dnsmos:
                reason = "low_mean_dnsmos"
            elif (fake_ratio is not None and self._cfg.max_fake_coverage is not None
                  and fake_ratio > self._cfg.max_fake_coverage):
                reason = "fake_coverage_too_high"

            # All-chunks view uses ``A0000`` index to distinguish from
            # the kept-only manifest. Kept-chunks keep the legacy
            # ``L0000`` numbering for downstream compat.
            all_chunks.append(Segment(
                start=cur_start,
                end=cur_end,
                speaker=cur_speaker,
                index=f"A{len(all_chunks):04d}",
                extra={
                    "source_indices": list(cur_sources),
                    "mean_dnsmos": round(mean_mos, 4),
                    "fake_coverage": round(fake_ratio, 4) if fake_ratio is not None else None,
                    "members": members,
                    "kept": reason is None,
                    **({"dropped_reason": reason} if reason else {}),
                },
            ))
            if reason is not None:
                return

            kept_chunks.append(Segment(
                start=cur_start,
                end=cur_end,
                speaker=cur_speaker,
                index=f"L{len(kept_chunks):04d}",
                extra={
                    "source_indices": list(cur_sources),
                    "mean_dnsmos": round(mean_mos, 4),
                    "fake_coverage": round(fake_ratio, 4) if fake_ratio is not None else None,
                    "members": members,
                },
            ))

        for orig_idx, seg in ordered[1:]:
            gap = seg.start - cur_end
            new_dur = seg.end - cur_start
            if (
                seg.speaker != cur_speaker
                or gap > self._cfg.max_gap_s
                or new_dur > self._cfg.max_duration_s


                or overlap_policy.spans_break(cur_end, seg.start,
                                              ctx.overlap_breaks)
            ):
                _flush()
                cur_start = seg.start
                cur_end = seg.end
                cur_speaker = seg.speaker
                cur_sources = [orig_idx]
            else:
                cur_end = seg.end
                cur_sources.append(orig_idx)
        _flush()

        ctx.long_chunks = kept_chunks
        ctx.long_chunks_all = all_chunks
        self._logger.debug(
            f"long_chunk > {len(ctx.segments)} shorts → "
            f"{len(all_chunks)} merged → {len(kept_chunks)} kept "
            f"(min_dur={self._cfg.min_duration_s}s, "
            f"min_mean_dnsmos={self._cfg.min_mean_dnsmos}, "
            f"max_dur={self._cfg.max_duration_s}s, max_gap={self._cfg.max_gap_s}s)"
        )
