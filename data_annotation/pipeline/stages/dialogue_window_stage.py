"""Select dialogue windows before exporting Phase-1 results.

Windows are sample ranges in the shared audio carrier, so no separate audio
files or second source-separation pass are needed. This stage writes
``<sid>.dialogue_chunk.json``; Phase 2 writes the final
``<sid>.dialogue.json`` after transcription and content checks.
"""
from __future__ import annotations

import os

from dialogue.select import Gates, evaluate, speaker_map, windows
from pipeline.context import PipelineContext
from pipeline.io import atomic_dump
from pipeline.stages.base import Stage
from pipeline.stages.exporter_stage import carrier_bounds, carrier_name
from utils.logger import Logger, time_logger


class DialogueWindowStage(Stage):


    name = "dialogue_window"

    def __init__(self, gates: Gates, carrier_fmt: str = "m4a") -> None:
        self._gates = gates
        self._carrier_fmt = carrier_fmt
        self._logger = Logger.get_logger()

    @time_logger
    def run(self, ctx: PipelineContext) -> None:
        assert ctx.audio is not None
        sr = int(ctx.audio.sample_rate)
        n_samples = int(ctx.audio.waveform.shape[0])


        # start/end/index/speaker/language/dnsmos + extra(is_fake/kept).


        segs = ctx.segments_all or ctx.segments
        rows = [s.to_legacy_dict() for s in segs]
        out_path = self._sibling_path(ctx, "dialogue_chunk.json")
        if not rows:
            atomic_dump([], out_path)
            return

        name = carrier_name(ctx.audio_name, self._carrier_fmt)
        kept: list[dict] = []
        for win in windows(rows, self._gates, ctx.overlap_breaks):
            reason, metrics = evaluate(win, self._gates)
            if reason is not None:
                continue
            a, b = carrier_bounds(metrics["start"], metrics["end"], sr, n_samples)
            if b <= a:
                self._logger.warning(
                    f"dialogue window {ctx.audio_name}[{len(kept)}] "
                    f"({metrics['start']}–{metrics['end']}s) is empty — skipping")
                continue
            kept.append(self._window_row(
                win, metrics, f"D{len(kept):04d}", ctx.audio_name, name, a, b, sr))

        atomic_dump(kept, out_path)
        self._logger.info(
            f"phase1 dialogue windows: {len(kept)} kept "
            f"({sum(r['duration'] for r in kept) / 3600:.3f} h) → {out_path}"
        )

    # ---------------------------------------------------------------- helpers
    def _window_row(self, win: list[dict], metrics: dict, d_index: str, sid: str,
                    carrier: str, a: int, b: int, sr: int) -> dict:

        smap = speaker_map(win)
        segments = [
            dict(index=s["index"], speaker=s["speaker"], tag=smap[s["speaker"]],
                 start=s["start"], end=s["end"], dnsmos=s.get("dnsmos"),
                 language=s.get("language"), is_fake=s.get("is_fake"),


                 seg_source=None, phase2_text=None)
            for s in win
        ]
        return dict(
            index=d_index, sid=sid,
            start=metrics["start"], end=metrics["end"], duration=metrics["duration"],
            num_speakers=metrics["num_speakers"], num_turns=metrics["num_turns"],
            num_speaker_turns=metrics["num_turns"],
            required_turns=metrics["required_turns"],
            mean_dnsmos=metrics["mean_dnsmos"], fake_coverage=metrics["fake_coverage"],
            secondary_share=metrics["secondary_share"],
            speaker_map=smap, segments=segments,
            transcript=None, languages=None,


            sample_rate=sr, carrier_path=carrier,
            carrier_start_samples=a, carrier_end_samples=b,


            materialized_by="phase1",
        )

    @staticmethod
    def _sibling_path(ctx: PipelineContext, suffix: str) -> str:

        candidate = ctx.final_json_path.replace(".partial.json", "." + suffix)
        if candidate == ctx.final_json_path:
            candidate = os.path.join(ctx.save_path, f"{ctx.audio_name}.{suffix}")
        return candidate
