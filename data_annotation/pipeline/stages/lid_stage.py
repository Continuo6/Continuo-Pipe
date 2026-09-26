"""Language-ID stage: per-segment language classification before ASR.

Sits between VAD and ASR. Reads ``ctx.segments`` (start/end/speaker/index
already populated by the VAD stage), slices the post-separator waveform per
segment, runs the LID model on the whole batch in one call, and writes the
result back onto each segment:

  * ``segment.language``                — ISO code (e.g. ``"zh"``, ``"en"``)
  * ``segment.extra["lid_confidence"]`` — model confidence in ``[0, 1]``
  * ``segment.extra["lid_raw"]``        — the raw label string (kept so we
    don't lose any sub-language info, e.g. ``"zh mandarin"`` from the
    Chinese-dialect variant of the model)

All other segment fields — speaker, index, time bounds — are preserved
untouched. The downstream ASR stage looks at ``segment.language`` first and
only falls back to its own ``detect_language`` when LID hasn't been run.
"""

from __future__ import annotations

import numpy as np

from pipeline.context import PipelineContext
from pipeline.stages.base import LanguageIdentifier, Stage
from utils.logger import Logger, time_logger


class LanguageIDStage(Stage):
    name = "lid"
    TARGET_SR = 16000  # FireRedLID + most LID models are 16 k mono trained.
    UNKNOWN_LABEL = "unknown"

    def __init__(self, identifier: LanguageIdentifier, min_confidence: float = 0.8):
        self._lid = identifier
        self._min_confidence = min_confidence
        self._logger = Logger.get_logger()

    def warmup(self) -> None:
        self._lid.warmup()

    def release(self) -> None:
        self._lid.release()

    @time_logger
    def run(self, ctx: PipelineContext) -> None:
        assert ctx.audio is not None
        if not ctx.segments:
            return

        waveform = ctx.audio.get_at_sr(self.TARGET_SR)

        clips: list[tuple[int, np.ndarray]] = []
        for seg in ctx.segments:
            start = int(seg.start * self.TARGET_SR)
            end = int(seg.end * self.TARGET_SR)
            clips.append((self.TARGET_SR, waveform[start:end]))

        results = self._lid.identify_batch(clips)
        if len(results) != len(ctx.segments):
            raise RuntimeError(
                f"LID returned {len(results)} results for "
                f"{len(ctx.segments)} segments — adapter contract violated"
            )

        n_unknown = 0
        for seg, (lang_raw, conf) in zip(ctx.segments, results):
            # Keep the raw token (may include dialect like "zh mandarin")
            # for debugging / future routing, and the model confidence.
            lang_raw = (lang_raw or "").strip()
            conf_f = float(conf) if conf is not None else 0.0
            seg.extra["lid_raw"] = lang_raw
            seg.extra["lid_confidence"] = conf_f

            # Below the trust threshold (or empty label from a degenerate
            # clip) → don't propagate a guess. Tag as "unknown" so the ASR
            # stage knows to re-detect inline; this keeps low-confidence
            # cases honest instead of silently locking in the wrong
            # tokenizer.
            if not lang_raw or conf_f < self._min_confidence:
                seg.language = self.UNKNOWN_LABEL
                n_unknown += 1
            else:
                # Write the first space-delimited token as the ISO code
                # (e.g. "zh mandarin" → "zh"); downstream gating uses ISO.
                seg.language = lang_raw.split(" ", 1)[0]

        self._logger.debug(
            f"LID labels (min_conf={self._min_confidence}): "
            f"{n_unknown}/{len(ctx.segments)} unknown; "
            f"sample={[(s.language, round(s.extra.get('lid_confidence') or 0, 2)) for s in ctx.segments[:8]]}"
            f"{' ...' if len(ctx.segments) > 8 else ''}"
        )
