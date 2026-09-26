"""Adapter: MelBand Roformer Kim FT3 source separator.

Wraps the ``audio_separator`` library. The library's public API is file-based
(path in, files out), but its underlying MDXC model exposes a numpy
in→numpy out path via ``prepare_mix`` + ``demix``. This adapter drives that
internal path so the AudioBundle flows straight through without round-tripping
through a temp WAV.

The Stage above does not know about any of this — the SourceSeparator ABC is
unchanged. Library-internal bindings stay quarantined inside the adapter.
"""

from __future__ import annotations

import contextlib
import gc

import librosa
import numpy as np
import torch
from torch.amp.autocast_mode import autocast, is_autocast_available

from audio_separator.separator import Separator

from pipeline.config import MelBandRoformerKimFT3Params
from pipeline.stages.base import SourceSeparator
from pipeline.types import AudioBundle


def _peak_normalize(wave, max_peak=1.0, min_peak=None):


    maxv = max(abs(wave.max()), abs(wave.min()))
    if maxv > max_peak:
        wave *= max_peak / maxv
    elif min_peak is not None and maxv < min_peak:
        wave *= min_peak / maxv
    return wave


class MelBandRoformerKimFT3Separator(SourceSeparator):
    INTERNAL_SR = 44100  # Kim FT3 expects 44.1 kHz stereo internally.
    _STEREO_CHANNELS = 2

    def __init__(self, params: MelBandRoformerKimFT3Params, device: str):
        self._params = params
        self._device = device
        self._sep: Separator | None = None
        self._autocast_device: str | None = None
        self._chunk = 0  # MDXC chunk size; resolved once in warmup()

    def warmup(self) -> None:
        if self._sep is not None:
            return

        kwargs: dict = dict(
            output_format="WAV",
            output_single_stem="Vocals",
            sample_rate=self.INTERNAL_SR,
            normalization_threshold=self._params.normalization_threshold,
            use_autocast=self._params.use_autocast,
        )
        if self._params.model_file_dir:
            kwargs["model_file_dir"] = self._params.model_file_dir

        self._sep = Separator(**kwargs)
        self._sep.load_model(model_filename=self._params.model_filename)

        # Resolve autocast device once. audio-separator only wraps autocast
        # around its file-based ``separate()`` entry (separator.py:1024);
        # we drive ``inst.demix`` directly to skip the temp-WAV round-trip,
        # so we re-apply the same contract ourselves.
        if self._params.use_autocast:
            dev = self._sep.torch_device.type
            if is_autocast_available(dev):
                self._autocast_device = dev

        # Resolve the chunk size once (constant per loaded model) so separate()
        # doesn't recompute it per file.
        self._chunk = self._chunk_size(self._sep.model_instance)

    def _chunk_size(self, inst) -> int:
        """The MDXC temporal chunk size, mirrored from the library's demix()
        (``stft_hop_length * (segment_size - 1)`` — see audio_separator
        ``mdxc_separator.py`` demix() ~L277-298; re-diff if that lib is
        upgraded). Resolved once in :meth:`warmup` and cached as ``self._chunk``;
        used to pad sub-chunk audio in :meth:`separate` around a library bug
        that crashes on input shorter than one chunk. Safe fallback on drift.
        """
        try:
            cfg = inst.model_data_cfgdict
            seg = inst.segment_size if inst.override_model_segment_size else cfg.inference.dim_t
            hop = getattr(cfg.model, "stft_hop_length", None) or cfg.audio.hop_length
            n = int(hop) * (int(seg) - 1)
            return n if n > 0 else 16 * self.INTERNAL_SR
        except Exception:  # noqa: BLE001 - config-shape drift → safe fallback
            return 16 * self.INTERNAL_SR

    def separate(self, audio: AudioBundle) -> AudioBundle:
        if self._sep is None:
            raise RuntimeError("call warmup() before separate()")

        # demix() expects stereo float32 @ 44.1kHz with shape (2, n_samples).
        # Standardization hands us either 1-D mono (n,) — duplicated to
        # dual-mono here — or real 2-D stereo (2, n) (io.preserve_stereo),
        # which we feed straight in without the pointless mono→dual re-expand.
        rate = audio.sample_rate
        waveform = audio.waveform.astype(np.float32, copy=False)
        stereo_in = waveform.ndim == 2 and waveform.shape[0] == 2
        if rate != self.INTERNAL_SR:
            from utils.logger import time_span
            with time_span(f"resample_separator_in_{rate}_to_{self.INTERNAL_SR}"):
                # librosa.resample works along the last axis, so both (n,) and
                # (2, n) resample correctly.
                wav44 = librosa.resample(
                    waveform, orig_sr=rate, target_sr=self.INTERNAL_SR
                )
        else:
            wav44 = waveform


        mix = wav44.copy() if stereo_in else np.stack([wav44, wav44], axis=0)

        # Run the library's in-memory demix path (no temp WAVs).
        inst = self._sep.model_instance
        inst.primary_source = None
        inst.secondary_source = None


        inst.is_primary_stem_main_target = False
        # Pad audio shorter than one model chunk. MDXC demix's last-chunk
        # branch computes start_idx = result_len - chunk_size; for sub-chunk
        # input that goes negative and torch slices the target to empty →
        # "size of tensor a (0)" RuntimeError, which otherwise drops every
        # source clip shorter than chunk_size (~8 s). Pad to one full chunk;
        # the vocals are trimmed back to the real length after demix.
        orig_len = mix.shape[-1]
        padded = orig_len < self._chunk
        if padded:
            mix = np.pad(mix, ((0, 0), (0, self._chunk - orig_len)))
        mix = _peak_normalize(
            mix,
            max_peak=inst.normalization_threshold,
            min_peak=inst.amplification_threshold,
        )

        ctx = (
            autocast(self._autocast_device)
            if self._autocast_device is not None
            else contextlib.nullcontext()
        )
        with ctx:
            # Use the upstream demix implementation for predictable output.
            source = inst.demix(mix=mix)
            # MDXC may return a dict keyed by stem name or a bare array

            if isinstance(source, dict):
                vocals = source[inst.primary_stem_name]
            else:
                vocals = source
        vocals = _peak_normalize(
            vocals,
            max_peak=inst.normalization_threshold,
            min_peak=inst.amplification_threshold,
        )
        # vocals shape is (channels, samples) at this point.

        # 4. Downmix the separated vocals to mono, drop back to the pipeline SR.
        #    dual-mono input → channel 0 (legacy, bit-for-bit unchanged default);
        #    real stereo input → average both separated channels.
        if stereo_in:


            mono_vocals = vocals[0] + vocals[1]
            mono_vocals /= 2
        else:
            mono_vocals = np.ascontiguousarray(vocals[0], dtype=np.float32)
        # Drop the chunk-padding we may have added for sub-chunk audio.
        if padded:
            mono_vocals = mono_vocals[:orig_len]
        if rate != self.INTERNAL_SR:
            from utils.logger import time_span
            with time_span(f"resample_separator_out_{self.INTERNAL_SR}_to_{rate}"):
                mono_vocals = librosa.resample(
                    mono_vocals, orig_sr=self.INTERNAL_SR, target_sr=rate
                )

        # Defensive: librosa.resample returns float32, but the model output
        # path could be float64 → keep AudioBundle invariants explicit.
        return AudioBundle(
            waveform=mono_vocals.astype(np.float32, copy=False),
            sample_rate=rate,
            name=audio.name,
        )

    def release(self) -> None:
        self._sep = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
