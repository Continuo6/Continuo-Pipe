"""Pipeline: deterministic stage chain, one PipelineContext per audio.

Stage order is intentionally hardcoded (see :func:`build_pipeline` in
``main.py``). Config picks which adapter fills each slot, not the
ordering itself.
"""

from __future__ import annotations

import os
from typing import Iterable

from pipeline.io import atomic_dump
from pipeline.context import PipelineContext
from pipeline.errors import PipelineAbort
from pipeline.stages.base import Stage
from utils.logger import Logger


class Pipeline:
    def __init__(self, stages: Iterable[Stage]):
        self._stages: list[Stage] = list(stages)
        self._logger = Logger.get_logger()
        # Split point for the prefetch driver (run_pipeline_multi): the front
        # = up to & including ``standardization`` (decode + resample, CPU/IO,
        # no GPU) is prefetchable on a worker thread; everything after is the
        # GPU + export back half. Found by name so stage reordering can't
        # silently mis-split. Defaults to 1 (just the first stage).
        self._prep_n = 1
        for i, s in enumerate(self._stages):
            if s.name == "standardization":
                self._prep_n = i + 1
                break
        # Tail split: everything from ``quality_filter`` onward is

        # GPU. The driver runs it on a separate thread so it overlaps the NEXT
        # sample's GPU stages. Found by name,

        self._tail_n = len(self._stages)
        for i, s in enumerate(self._stages):
            if s.name == "quality_filter":
                self._tail_n = i
                break
        assert self._tail_n >= self._prep_n

    def warmup(self) -> None:
        for stage in self._stages:
            stage.warmup()

    def release(self) -> None:
        for stage in self._stages:
            stage.release()

    PARTIAL_SUFFIX = ".partial.json"

    def process(self, audio_path: str, save_path: str | None = None,
                audio_name: str | None = None) -> PipelineContext | None:
        """Run the full Phase-1 chain on a single audio file.

        The resume marker is ``<save_path>/<audio_name>.partial.json``
        (what Phase-1's ExporterStage writes). Phase 2 (``transcribe.py``)
        consumes that file and writes the final ``<audio_name>.json``; it
        owns its own resume gating against the final filename.

        Returns ``None`` if the file was already processed; otherwise the
        final :class:`PipelineContext`.
        """
        audio_name = audio_name or os.path.splitext(os.path.basename(audio_path))[0]
        save_path = save_path or os.path.join(
            os.path.dirname(audio_path) + "_processed", audio_name
        )
        final_path = os.path.join(save_path, audio_name + self.PARTIAL_SUFFIX)
        if os.path.exists(final_path):
            print(final_path, "exists")
            return None
        os.makedirs(save_path, exist_ok=True)

        ctx = PipelineContext(
            audio_path=audio_path,
            save_path=save_path,
            audio_name=audio_name,
            final_json_path=final_path,
        )
        self._logger.debug(
            f"Processing audio: {audio_name}, from {audio_path}, save to: {save_path}"
        )

        try:
            for stage in self._stages:
                self._logger.info(f"-> {stage.name}")
                stage.run(ctx)
        except PipelineAbort as e:
            self._logger.info(f"skip {audio_path}: {e.reason}")
            # The empty [] is itself the resume marker → must be atomic.
            atomic_dump([], final_path)
            return ctx

        return ctx

    # ----- split front/back for the prefetch driver (run_pipeline_multi) -----
    # ``prep`` + ``finish`` together == ``process`` (same stages, same abort
    # handling), just split at the standardization boundary so the CPU/IO front
    # can run one sample ahead on a worker thread while the GPU back half of the
    # previous sample runs on the main thread.

    def prep(self, audio_path: str, save_path: str | None = None,
             audio_name: str | None = None):
        """Front half (prefetchable, CPU/IO, no GPU). Returns one of:

          * ``None``           — already processed (skip)
          * ``(ctx, "abort")`` — a front stage aborted (5h/PCM); the empty
            ``partial.json`` resume marker is already written; do NOT finish()
          * ``(ctx, "ok")``    — standardized; pass ctx to :meth:`finish`
        """
        audio_name = audio_name or os.path.splitext(os.path.basename(audio_path))[0]
        save_path = save_path or os.path.join(
            os.path.dirname(audio_path) + "_processed", audio_name
        )
        final_path = os.path.join(save_path, audio_name + self.PARTIAL_SUFFIX)
        if os.path.exists(final_path):
            return None
        os.makedirs(save_path, exist_ok=True)
        ctx = PipelineContext(
            audio_path=audio_path,
            save_path=save_path,
            audio_name=audio_name,
            final_json_path=final_path,
        )
        try:
            for stage in self._stages[: self._prep_n]:
                self._logger.info(f"-> {stage.name}")
                stage.run(ctx)
        except PipelineAbort as e:
            self._logger.info(f"skip {audio_path}: {e.reason}")
            atomic_dump([], final_path)
            return ctx, "abort"
        return ctx, "ok"

    def finish(self, ctx: PipelineContext) -> PipelineContext:
        """Back half (main thread): the GPU stages + exporter (everything after
        standardization). Mirrors :meth:`process`'s abort handling."""
        if self.finish_gpu(ctx) == "ok":
            self.tail(ctx)
        return ctx

    def finish_gpu(self, ctx: PipelineContext) -> str:

        try:
            for stage in self._stages[self._prep_n: self._tail_n]:
                self._logger.info(f"-> {stage.name}")
                stage.run(ctx)
        except PipelineAbort as e:
            self._logger.info(f"skip {ctx.audio_path}: {e.reason}")
            atomic_dump([], ctx.final_json_path)
            return "abort"
        return "ok"

    def tail(self, ctx: PipelineContext) -> PipelineContext:

        try:
            for stage in self._stages[self._tail_n:]:
                self._logger.info(f"-> {stage.name}")
                stage.run(ctx)
        except PipelineAbort as e:
            self._logger.info(f"skip {ctx.audio_path}: {e.reason}")
            atomic_dump([], ctx.final_json_path)
        return ctx
