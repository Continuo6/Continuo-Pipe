"""Adapter: FireRedVAD (FireRedTeam/FireRedVAD) voice-activity detection.

The upstream ``fireredasr2s.fireredvad.vad.FireRedVad`` model takes a wav
path or an ``(numpy, sample_rate)`` tuple and returns
``{"dur", "timestamps": [(start_s, end_s), ...]}``. It runs at 16 k
mono, kaldi-style fbank features — same input convention as FireRedLID
(int16-magnitude floats).

The :class:`VoiceActivityDetector` ABC takes the full audio plus the
diarizer's frames; to keep every output segment single-speaker we slice
the resampled waveform per diarization frame, run FireRedVAD on each
slice, and shift the returned timestamps back into the file's absolute
timeline. With the "short" config (``max_speech_frame=3000`` = 30 s,
``merge_silence_frame=300`` = 3 s) the result is sentence-grain segments
suitable for TTS data cleanup."""

from __future__ import annotations

import os
import sys

import numpy as np

from pipeline.config import FireRedVADParams
from utils.logger import Logger
from pipeline.stages.base import VoiceActivityDetector
from pipeline.types import AudioBundle, DiarizationFrame, Segment


class FireRedVADAdapter(VoiceActivityDetector):
    TARGET_SR = 16000

    # Minimum diarization-frame length (in samples at 16 k) we'll feed to
    # the model. Below this, the kaldi-fbank → smoothing → lookback_filter
    # chain produces an empty time-axis tensor and the Conv1d crashes
    # with "Only zero batch or zero channel inputs are supported, but got
    # input shape: [1, 128, 1, 0]". 0.5 s gives plenty of headroom over
    # the 25 ms frame_length + 5-frame smoothing window (= ~75 ms hard
    # floor). Frames shorter than this contribute no segments.
    MIN_CLIP_SAMPLES = 16000 // 2

    def __init__(self, params: FireRedVADParams, device: str):
        self._params = params
        self._device = device
        self._vad = None

    def warmup(self) -> None:
        if self._vad is not None:
            return
        # Upstream isn't pip-installable against our env; add the cloned
        # source tree to sys.path on first use, same trick as FireRedLID.
        if self._params.source_dir:
            src = self._params.source_dir
            if src not in sys.path:
                sys.path.insert(0, src)

        from fireredasr2s.fireredvad.vad import FireRedVad, FireRedVadConfig

        if not os.path.isdir(self._params.model_dir):
            raise FileNotFoundError(
                f"FireRedVAD model_dir not found: {self._params.model_dir}. "
                f"Run huggingface-cli download FireRedTeam/FireRedVAD "
                f"--local-dir <that path>."
            )

        cfg = FireRedVadConfig(
            use_gpu=self._device.startswith("cuda"),
            smooth_window_size=self._params.smooth_window_size,
            speech_threshold=self._params.speech_threshold,
            min_speech_frame=self._params.min_speech_frame,
            max_speech_frame=self._params.max_speech_frame,
            min_silence_frame=self._params.min_silence_frame,
            merge_silence_frame=self._params.merge_silence_frame,
            extend_speech_frame=self._params.extend_speech_frame,
            chunk_max_frame=self._params.chunk_max_frame,
        )
        self._vad = FireRedVad.from_pretrained(self._params.model_dir, cfg)

    def detect(
        self,
        audio: AudioBundle,
        diarization: list[DiarizationFrame],
    ) -> list[Segment]:
        if self._vad is None:
            raise RuntimeError("call warmup() before detect()")

        # 1. Resample to 16 k mono once; scale to int16-magnitude floats
        # so kaldi-style fbank produces the right energy. Same convention
        # used by the FireRedLID adapter.
        wav16 = audio.get_at_sr(self.TARGET_SR)
        wav16_int = np.ascontiguousarray(
            wav16.astype(np.float32) * 32768.0, dtype=np.float32
        )

        out: list[Segment] = []
        # 2. Run VAD inside each diarization frame so segments stay
        # single-speaker. The diarizer already split the audio by
        # speaker; we just need to detect speech *within* each frame and
        # carry the speaker label through.
        for frame_idx, frame in enumerate(diarization):
            start_i = int(frame.start * self.TARGET_SR)
            end_i = int(frame.end * self.TARGET_SR)
            if end_i - start_i < self.MIN_CLIP_SAMPLES:
                # FireRed fbank crashes below 0.5 s, but diarization frames in

                # often define a speaker switch. Preserve the diarizer's exact
                # boundary as a conservative speech fallback. VADStage still
                # applies segmentation.min_segment_length, and QualityFilter's
                # 0.5 s short-track floor keeps these out of ordinary TTS rows;
                # dialogue/all-shorts can retain and transcribe them.
                if end_i > start_i:
                    out.append(Segment(
                        start=frame.start, end=frame.end,
                        speaker=frame.speaker, index=f"{frame_idx:03d}_000",
                        extra={"vad_fallback": "short_diarization_frame",


                               "diar_frame": frame_idx},
                    ))
                continue
            clip = wav16_int[start_i:end_i]
            # FireRedVad.detect accepts (wav_np, sample_rate) tuples; the
            # ``audio_feat.extract`` branch unpacks it (wav first).


            # (fireredvad/core/vad_postprocessor.py:103).


            try:
                result, _probs = self._vad.detect((clip, self.TARGET_SR))
            except Exception as ex:
                Logger.get_logger().warning(
                    "VAD skipped frame=%d [%.2f, %.2f]s (%s: %s)",
                    frame_idx, frame.start, frame.end,
                    type(ex).__name__, ex)
                continue
            timestamps = result.get("timestamps", []) if result else []
            for j, (s, e) in enumerate(timestamps):
                out.append(Segment(
                    start=frame.start + float(s),
                    end=frame.start + float(e),
                    speaker=frame.speaker,
                    index=f"{frame_idx:03d}_{j:03d}",
                    extra={"diar_frame": frame_idx},
                ))

        return out

    def release(self) -> None:
        self._vad = None
