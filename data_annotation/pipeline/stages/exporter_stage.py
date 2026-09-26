"""Phase-1 exporter: manifests plus one shared M4A carrier per recording.

Every short and long is a view into ``<sid>.m4a``. The manifests store
exact carrier sample boundaries; no per-segment audio files are created. This
lets Phase 2 open/decode the carrier once instead of issuing hundreds or
thousands of small NFS reads.

``partial.json`` remains the atomic resume marker and is written last. A crash
before that point is retried, including atomic replacement of the carrier.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time

import numpy as np

from pipeline.context import PipelineContext
from pipeline.io import atomic_dump
from pipeline.stages.base import Stage
from utils.logger import Logger


_ENCODE_BLOCK_SAMPLES = max(
    65_536, int(os.environ.get("CONTINUO_DATA_P1_ENCODE_BLOCK_SAMPLES", "1048576"))
)

#: Minimum forward step for :func:`bump_prefix_mtime`. 1 ms clears NFSv3's
#: microsecond attribute granularity (a 1 ns step would truncate to a no-op).
_MTIME_BUMP_NS = 1_000_000


def carrier_name(audio_name: str, fmt: str = "m4a") -> str:

    return f"{audio_name}.{fmt}"


def bump_prefix_mtime(sid_dir: str) -> None:
    """Touch ``<prefix>/`` after landing manifests inside ``<prefix>/<sid>/``.

    Phase 2 skips a prefix whose drained marker still matches the prefix dir's
    (entry count, mtime) fingerprint. A worker creates the sid dir when it
    claims the recording, minutes before ``partial.json`` lands inside it, so
    the arrival changes neither half of that fingerprint — a ``--loop`` phase 2
    that scanned the prefix during the gap would skip it forever.

    The new mtime is forced STRICTLY GREATER than the old one: filesystem
    "now" timestamps are coarse (10 ms on ext4 here), so a plain touch is a
    no-op whenever the prefix was last modified inside the same tick — the
    exact same-tick hole that put the entry count in the fingerprint.
    """
    prefix = os.path.dirname(sid_dir)
    try:
        st = os.stat(prefix)
        os.utime(prefix, ns=(
            st.st_atime_ns,
            max(time.time_ns(), st.st_mtime_ns + _MTIME_BUMP_NS),
        ))
    except OSError:
        pass


def carrier_bounds(start_s: float, end_s: float, sr: int, n_samples: int) -> tuple[int, int]:

    a = max(0, min(n_samples, int(float(start_s) * sr)))
    b = max(a, min(n_samples, int(float(end_s) * sr)))
    return a, b


def _mono_waveform(waveform: np.ndarray) -> np.ndarray:
    """Return time-major mono float32 without copying the whole array twice."""
    wav = np.asarray(waveform)
    if wav.ndim == 1:
        return wav.astype(np.float32, copy=False)
    if wav.ndim != 2:
        raise ValueError(f"expected 1-D/2-D waveform, got shape={wav.shape}")
    if wav.shape[0] <= 8:
        wav = wav.mean(axis=0)
    elif wav.shape[1] <= 8:
        wav = wav.mean(axis=1)
    else:
        raise ValueError(f"cannot infer channel axis from shape={wav.shape}")
    return wav.astype(np.float32, copy=False)


def _encode_m4a_atomic(
    dst: str, waveform: np.ndarray, sample_rate: int, bitrate: str,
) -> None:
    """Stream PCM to ffmpeg in bounded blocks and atomically publish M4A."""
    wav = _mono_waveform(waveform)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{os.path.basename(dst)}.tmp.", suffix=".m4a",
        dir=os.path.dirname(dst),
    )
    os.close(fd)
    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "f32le", "-ar", str(sample_rate), "-ac", "1",
                "-i", "pipe:0", "-c:a", "aac", "-b:a", bitrate,
                "-movflags", "+faststart", tmp,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None
        for start in range(0, wav.shape[0], _ENCODE_BLOCK_SAMPLES):
            block = np.ascontiguousarray(
                wav[start:start + _ENCODE_BLOCK_SAMPLES], dtype=np.float32,
            )
            proc.stdin.write(memoryview(block).cast("B"))
        proc.stdin.close()
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(
                f"ffmpeg M4A encode failed for {dst}: "
                f"{stderr.decode(errors='replace').strip()[:600]}"
            )
        if os.path.getsize(tmp) <= 0:
            raise RuntimeError(f"ffmpeg produced an empty M4A: {dst}")
        # mkstemp creates 0600 regardless of the cooperative launcher umask.
        # Publish the inode with group-readable/writable mode atomically.
        os.chmod(tmp, 0o664)
        os.replace(tmp, dst)
    except BaseException:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


class ExporterStage(Stage):
    name = "exporter"

    def __init__(self, fmt: str = "m4a", bitrate: str = "128k"):
        if fmt != "m4a":
            raise ValueError(
                f"single-carrier exporter requires m4a, got {fmt!r}"
            )
        self._fmt = fmt
        self._bitrate = bitrate
        self._logger = Logger.get_logger()

    def run(self, ctx: PipelineContext) -> None:
        assert ctx.audio is not None
        sr = int(ctx.audio.sample_rate)
        waveform = _mono_waveform(ctx.audio.waveform)
        n_samples = int(waveform.shape[0])

        all_shorts = ctx.segments_all or ctx.segments
        kept_shorts = [s for s in all_shorts if s.extra.get("kept", True)]
        kept_longs = ctx.long_chunks
        all_longs = ctx.long_chunks_all or []

        name = carrier_name(ctx.audio_name, self._fmt)
        carrier_path = os.path.join(ctx.save_path, name)
        _encode_m4a_atomic(
            carrier_path, waveform, sr, bitrate=self._bitrate,
        )

        def carrier_row(seg) -> dict:
            row = seg.to_legacy_dict()
            start, end = carrier_bounds(seg.start, seg.end, sr, n_samples)
            row.update({
                "sample_rate": sr,
                "carrier_path": name,
                "carrier_start_samples": start,
                "carrier_end_samples": end,
            })
            return row

        # Debug/long manifests first; partial.json is the LAST atomic write.
        all_short_rows = [carrier_row(seg) for seg in all_shorts]
        all_shorts_path = self._sibling_path(ctx, "all_shorts.json")
        atomic_dump(all_short_rows, all_shorts_path)
        n_dropped = sum(
            1 for s in all_shorts if not s.extra.get("kept", True)
        )
        self._logger.info(
            f"phase1 all-shorts done, {len(all_short_rows)} "
            f"({n_dropped} dropped) → {all_shorts_path}"
        )

        # Always publish siblings, including [], so a retry cannot leave stale
        # long manifests from an older config/model decision.
        long_path = self._sibling_path(ctx, "long_chunks.json")
        atomic_dump([carrier_row(seg) for seg in kept_longs], long_path)
        self._logger.info(
            f"phase1 long-chunks done, {len(kept_longs)} → {long_path}"
        )

        all_longs_path = self._sibling_path(ctx, "all_longs.json")
        atomic_dump([carrier_row(seg) for seg in all_longs], all_longs_path)
        n_long_dropped = sum(
            1 for c in all_longs if not c.extra.get("kept", True)
        )
        self._logger.info(
            f"phase1 all-longs done, {len(all_longs)} "
            f"({n_long_dropped} dropped) → {all_longs_path}"
        )

        kept_rows = [carrier_row(seg) for seg in kept_shorts]
        atomic_dump(kept_rows, ctx.final_json_path)
        bump_prefix_mtime(ctx.save_path)
        self._logger.info(
            f"phase1 done, {len(kept_rows)}/{len(all_shorts)} shorts kept; "
            f"one carrier ({n_samples / sr / 3600:.3f} h) → {carrier_path}"
        )

    @staticmethod
    def _sibling_path(ctx: PipelineContext, suffix: str) -> str:
        candidate = ctx.final_json_path.replace(".partial.json", "." + suffix)
        if candidate == ctx.final_json_path:
            candidate = os.path.join(ctx.save_path, f"{ctx.audio_name}.{suffix}")
        return candidate
