"""VAD stage: runs the VAD adapter, then applies the cut/merge post-process.

The cut/merge logic is the same as the legacy ``cut_by_speaker_label``; it
is in the stage (not the adapter) because it is independent of which VAD
model is in use and applies after every diarization-aware VAD.
"""

from __future__ import annotations

from copy import copy

import overlap_policy
from pipeline.config import SegmentationConfig
from pipeline.context import PipelineContext
from pipeline.stages.base import Stage, VoiceActivityDetector
from pipeline.types import Segment
from utils.logger import Logger, time_logger


class VADStage(Stage):
    name = "vad"

    def __init__(self, vad: VoiceActivityDetector, segmentation: SegmentationConfig):
        self._vad = vad
        self._cfg = segmentation
        self._logger = Logger.get_logger()

    def warmup(self) -> None:
        self._vad.warmup()

    def release(self) -> None:
        self._vad.release()

    @time_logger
    def run(self, ctx: PipelineContext) -> None:
        assert ctx.audio is not None and ctx.diarization is not None
        raw = self._vad.detect(ctx.audio, ctx.diarization)
        raw = self._apply_overlap_policy(ctx, raw)
        ctx.segments = self._cut_by_speaker_label(raw)

    def _apply_overlap_policy(self, ctx: PipelineContext,
                              raw: list[Segment]) -> list[Segment]:

        cfg = self._cfg.overlap
        if not cfg.enabled or not ctx.diarization:
            return raw
        drop, breaks = overlap_policy.evaluate(
            ctx.diarization, self._cfg.overlap_policy())
        ctx.overlap_breaks = breaks
        if not drop:
            return raw
        kept = [s for s in raw if s.extra.get("diar_frame") not in drop]
        self._logger.debug(
            f"overlap_policy > dropped {len(raw) - len(kept)}/{len(raw)} segments "
            f"from {len(drop)} conflicting diarization frames; "
            f"{len(breaks)} break(s) for long/dialogue tracks"
        )
        return kept

    def _cut_by_speaker_label(self, vad_list: list[Segment]) -> list[Segment]:
        merge_gap = self._cfg.merge_gap
        min_len = self._cfg.min_segment_length
        max_len = self._cfg.max_segment_length

        updated_list: list[Segment] = []

        for vad in vad_list:
            last_start = updated_list[-1].start if updated_list else None
            last_end = updated_list[-1].end if updated_list else None
            last_speaker = updated_list[-1].speaker if updated_list else None


            vad.extra.setdefault("speech_s", round(vad.end - vad.start, 3))

            if vad.end - vad.start > max_len:
                current_start = vad.start
                segment_end = vad.end
                original_span = max(1e-9, vad.end - vad.start)
                total_speech = float(vad.extra["speech_s"])
                self._logger.warning(
                    "cut_by_speaker_label > segment longer than max, force trimming to smaller segments"
                )
                while segment_end - current_start > max_len:
                    piece = copy(vad)
                    piece.extra = dict(vad.extra)
                    piece.start = current_start
                    piece.end = current_start + max_len
                    piece.extra["speech_s"] = round(
                        total_speech * (piece.end - piece.start) / original_span, 3)
                    updated_list.append(piece)
                    current_start += max_len
                tail = copy(vad)
                tail.extra = dict(vad.extra)
                tail.start = current_start
                tail.end = segment_end
                tail.extra["speech_s"] = round(
                    total_speech * (tail.end - tail.start) / original_span, 3)
                updated_list.append(tail)
                continue


            if (
                last_speaker is not None
                and last_speaker == vad.speaker
                and vad.start - last_end < merge_gap
                and vad.end - last_start <= max_len
            ):
                prev = updated_list[-1]
                prev_end = prev.end
                overlap = max(0.0, min(prev_end, vad.end) - max(prev.start, vad.start))
                prev.extra["speech_s"] = round(
                    prev.extra.get("speech_s", prev_end - prev.start)
                    + vad.extra.get("speech_s", vad.end - vad.start) - overlap, 3)
                prev.end = max(prev_end, vad.end)  # never shrink on contained overlap
                continue

            updated_list.append(vad)

        self._logger.debug(
            f"cut_by_speaker_label > merged {len(vad_list) - len(updated_list)} segments"
        )

        filtered = [s for s in updated_list if s.end - s.start >= min_len]
        self._logger.debug(
            f"cut_by_speaker_label > removed: {len(updated_list) - len(filtered)} segments by length"
        )
        return filtered
