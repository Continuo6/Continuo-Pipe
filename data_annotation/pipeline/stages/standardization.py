"""Standardization stage: load + resample + loudness-normalize audio."""

from __future__ import annotations

import os

import numpy as np
from pydub import AudioSegment
from pydub.utils import mediainfo

from pipeline.context import PipelineContext
from pipeline.errors import PipelineAbort
from pipeline.stages.base import Stage
from pipeline.types import AudioBundle
from utils.logger import Logger, time_logger


class StandardizationStage(Stage):
    """Loud-normalize a file into a fixed-rate mono float32 waveform.

    Behavior:

    * Reject inputs whose *native* sample rate is below ``min_input_sample_rate``.
      Low-SR audio (8 k phone calls etc.) can't carry the high-frequency
      content downstream stages (Kim FT3 separator at 44.1 k, DiariZen
      WavLM) need, and upsampling them just feeds the pipeline silence-band
      energy. Surfaces as PipelineAbort so the file is recorded skipped.
    * Output float32 at ``sample_rate`` (defaults to 44.1 k so the
      separator no longer round-trips 24 k → 44.1 k → 24 k). Channels:
      ``mono=True`` (default) → 1-D mono ``(n,)``; ``mono=False`` →
      2-channel ``(2, n)`` so a stereo source feeds the separator as real
      stereo instead of being collapsed to mono and re-duplicated there.
    * Loudness target: dBFS = -20 with a +/-3 dB clamp, then peak-normalized.
    """

    name = "standardization"

    def __init__(
        self,
        sample_rate: int,
        max_audio_hours: float = 5.0,
        min_input_sample_rate: int = 24000,
        mono: bool = True,
    ):
        self._sample_rate = sample_rate
        self._max_audio_hours = max_audio_hours
        self._min_input_sample_rate = min_input_sample_rate
        self._mono = mono
        self._audio_count = 0
        self._logger = Logger.get_logger()

    @time_logger
    def run(self, ctx: PipelineContext) -> None:
        audio = self._standardize(ctx.audio_path)
        if audio is None:
            raise PipelineAbort("audio missing or longer than the configured cap")
        ctx.audio = audio

    def _standardize(self, audio):  # noqa: ANN001 - mirrors legacy signature
        from utils.logger import time_span
        name = "audio"

        # 1. Native-SR / duration gate. soundfile reads libsndfile-native
        # formats (WAV/FLAC/OGG/AIFF) from the header in ~µs; mediainfo
        # shells out to ffprobe (~110 ms) for everything else (MP3/M4A).
        if isinstance(audio, str):
            channels = 0
            with time_span("std_probe_header"):
                try:
                    import soundfile as sf
                    sf_info = sf.info(audio)
                    src_sr = int(sf_info.samplerate)
                    duration = float(sf_info.duration)
                    channels = int(sf_info.channels)
                except Exception:  # noqa: BLE001 - libsndfile-incompatible format
                    try:
                        info = mediainfo(audio)
                        src_sr = int(info.get("sample_rate") or 0)
                        duration = float(info.get("duration") or 0.0)
                        channels = int(info.get("channels") or 0)
                    except (TypeError, ValueError):
                        src_sr, duration, channels = 0, 0.0, 0
            if src_sr == 0 or duration == 0.0:
                self._logger.warning(
                    f"could not probe {audio}: sr={src_sr} dur={duration}"
                )
                return None
            if src_sr < self._min_input_sample_rate:
                raise PipelineAbort(
                    f"native sample rate {src_sr} Hz < "
                    f"{self._min_input_sample_rate} Hz cutoff"
                )
            if (duration / 60 / 60) >= self._max_audio_hours:
                return None
            # pydub ``from_file`` decodes to a 16-bit WAV; the RIFF size field
            # is 32-bit, so decoded PCM > 4 GB overflows and the decode fails.
            # A fixed 5 h cap only protects 44.1k/48k; a 96k stereo clip hits
            # ~6.9 GB at 5 h. Gate on the actual decoded size instead (16-bit
            # = sr*ch*2 B/s). channels unknown → assume stereo (worst case).
            ch = channels or 2
            pcm_bytes = duration * src_sr * ch * 2
            if pcm_bytes > 3.5e9:  # 0.5 GB headroom under the 4 GB ceiling
                raise PipelineAbort(
                    f"decoded PCM {pcm_bytes / 1e9:.1f} GB exceeds pydub's "
                    f"~4 GB WAV limit ({src_sr} Hz x {ch} ch x "
                    f"{duration / 3600:.1f} h)"
                )

        if isinstance(audio, str):
            name = os.path.basename(audio)
            with time_span("std_decode_pydub_from_file"):
                audio = AudioSegment.from_file(audio)
        elif isinstance(audio, AudioSegment):
            name = f"audio_{self._audio_count}"
            self._audio_count += 1
        else:
            raise ValueError("Invalid audio type")

        # 2. Format conversion. pydub set_frame_rate / set_sample_width /
        # set_channels each do pure-Python audioop transforms; we time the
        # block to spot expensive resamples.
        with time_span("std_pydub_setup"):
            audio = audio.set_frame_rate(self._sample_rate)
            audio = audio.set_sample_width(2)
            audio = audio.set_channels(1 if self._mono else 2)

        # 3. Loudness gain. dBFS computation walks every sample once;
        # apply_gain does another full pass.
        with time_span("std_gain"):
            target_dBFS = -20
            gain = target_dBFS - audio.dBFS
            normalized_audio = audio.apply_gain(min(max(gain, -3), 3))

        # 4. PCM int16 → float32 ndarray + peak-normalize.
        with time_span("std_to_np_and_peak_norm"):
            waveform = np.array(
                normalized_audio.get_array_of_samples(), dtype=np.float32,
            )
            # Stereo: pydub returns interleaved [L0,R0,L1,R1,...]; deinterleave
            # to (2, n) for the separator. One global peak scale across both
            # channels preserves the inter-channel balance.
            if not self._mono:
                waveform = waveform.reshape(-1, 2).T
            max_amplitude = np.max(np.abs(waveform))
            if max_amplitude > 0:
                waveform /= max_amplitude

        self._logger.debug(f"waveform shape: {waveform.shape}")
        self._logger.debug("waveform in np ndarray, dtype=" + str(waveform.dtype))

        return AudioBundle(
            waveform=waveform,
            sample_rate=self._sample_rate,
            name=name,
        )
