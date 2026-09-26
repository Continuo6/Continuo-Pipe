"""Phase 2 — batched ASR over the partial manifests Phase 1 produced.

Phase 1 (``main.py``) walks each audio file through standardize → separator
→ diarizer → VAD → DNSMOS → quality-filter → LID and lands each surviving
recording as one M4A carrier plus a ``<sid>.partial.json`` manifest under
``<input>_processed/<sid>/``.

This script:

1. Scans ``<input>_processed/*/<sid>.partial.json`` for files not yet
   transcribed (skip when ``<sid>.json`` already exists).
2. Decodes each manifest's shared carrier once and slices one big batch indexed by
   ``(sid, seg_idx)``.
3. Runs the configured ASR adapter (default ``qwen3_asr``) on the batch.
   The ``language=`` hint per segment comes from the partial — if Phase 1
   tagged it ``"unknown"`` (LID confidence below threshold) the adapter
   sees ``None`` and auto-detects.
4. Applies the adapter's ``accepts(language)`` rule against ASR's
   *output* language. Utterances whose resolved language is unknown,
   unsupported, or explicitly skip-listed (e.g. ``ms``/``id`` for
   Qwen3-ASR) are dropped here.
5. Applies the text-length filter (``filter.min_char_count``) — the part
   of the legacy ``calculate_audio_stats`` that needs text.
6. Writes ``<sid>.json`` per file with the final segment list, augmented
   with ``text`` / ``language`` / ASR provenance. Per-segment WAVs are
   retained (re-runnable) unless ``--prune-segments`` is passed.

CLI:

    python transcribe.py --config_path config.json
    python transcribe.py --config_path config.json --processed-root processed
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import os
import random
import socket
import subprocess
import time
import warnings
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
import torch
import tqdm

from pipeline.config import PipelineConfig
from pipeline.io import atomic_dump, nodes, work_queue
from pipeline.registry import build_adapter

# Max duration (s) for the long-track WHOLE-CHUNK fallback one-shot ASR call.
# Long chunks can be hours; a multi-minute single clip hangs Qwen3-ASR/vLLM,
# so only attempt the whole-chunk fallback on short longs (per-member handles
# the rest). Tunable via env if needed.
_WHOLE_CHUNK_MAX_S = float(os.environ.get("CONTINUO_DATA_WHOLE_CHUNK_MAX_S", "60"))
# Min duration (s) for a SPLIT run to be kept as its own long. main() overrides
# from ``cfg.long_chunk.min_duration_s`` (env wins if explicitly set); the
# literal here only covers direct callers that never go through main() (tests).
_LONG_MIN_DUR_S = float(os.environ.get("CONTINUO_DATA_LONG_MIN_DUR_S", "30"))


#:


_LONG_MAX_GAP_S = float(os.environ.get("CONTINUO_DATA_LONG_MAX_GAP_S", "2"))
from pipeline.stages.base import ASRModel
from pipeline.types import AudioBundle, Segment

import asr_texts
from dialogue import phase2 as dialogue_phase2
from dialogue.select import Gates
from utils.logger import Logger
from utils.tool import detect_gpu, get_char_count

warnings.filterwarnings("ignore")


# Process-global, bounded pool for per-segment audio reads. One <sid> can
# carry 50–2000 segment files; reading them serially over NFS (one blocking
# ``sf.read`` each, latency-bound) starves the GPU whenever a big sid is the
# file being loaded — the other prefetch slots sit idle behind that one
# serial loop. We fan a file's reads across this shared pool so a big sid
# issues many NFS RPCs at once, and bound it PROCESS-GLOBALLY so total
# in-flight reads stay near the NFS sweet spot no matter how many files
# prefetch concurrently. Reassembly is by original order, so the concatenated
# buffer + per-segment offsets preserve the serial path's order. Tune via
# ``CONTINUO_DATA_P2_SEG_READERS`` (raise if
# the NFS mount has read headroom; lower if many instances contend).
_SEG_READERS = max(1, int(os.environ.get("CONTINUO_DATA_P2_SEG_READERS", "8")))


_CARRIER_TAIL_SLACK = 1024

_SEG_READ_POOL = ThreadPoolExecutor(
    max_workers=_SEG_READERS, thread_name_prefix="p2segread"
)


# Sinitic dialect subtags the Phase-1 LID model (FireRedLID) emits as
# ``"zh <subtag>"``. Standard Mandarin (``"zh mandarin"`` / bare ``"zh"``) is
# NOT a dialect and does not qualify for the looser dialect DNSMOS floor.
_LID_DIALECT_SUBTAGS = frozenset(
    {"yue", "wu", "min", "xinan", "xiang", "hakka", "gan", "jin",
     "kejia", "hui", "pinghua", "cantonese"}
)


def _lid_is_dialect(lid_raw: str | None) -> bool:
    """True when Phase-1 LID's raw label names a Sinitic *dialect* (e.g.
    ``"zh xinan"``, ``"zh wu"``), not standard Mandarin. Used together with the
    ASR-side ``is_dialect`` flag: the looser dialect DNSMOS floor applies only
    when BOTH the ASR and the LID independently point at a dialect."""
    parts = (lid_raw or "").strip().lower().split()
    return len(parts) >= 2 and parts[0] == "zh" and parts[1] in _LID_DIALECT_SUBTAGS


def _own_audio(r: dict) -> str | None:

    ap = r.get("audio_path")
    if not ap:
        return None
    if r.get("carrier_path"):
        return None
    return ap


def _resolve_path(path: str, base_dir: str) -> str:
    """Resolve a manifest path (absolute or relative to ``base_dir``)."""
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def _load_segment_audio(seg_meta: dict, base_dir: str) -> tuple[np.ndarray, int]:
    """Read a per-segment WAV referenced by a partial-manifest row."""
    full = _resolve_path(seg_meta["audio_path"], base_dir)
    waveform, sr = sf.read(full, dtype="float32", always_2d=False)
    if waveform.ndim == 2:  # defensive — phase 1 writes mono
        waveform = waveform.mean(axis=1).astype(np.float32)
    return waveform, int(sr)


def _read_carrier_windows(
    rows: list[dict], sid_dir: str, target_sr: int | None = None,
) -> list[tuple[np.ndarray, int]]:
    """Decode one carrier sequentially and retain only requested windows.

    PCM is streamed from ffmpeg in bounded chunks, so a five-hour recording
    never becomes a multi-GB temporary numpy array. Overlapping windows are
    supported and result order matches ``rows``.
    """
    if not rows:
        return []
    carrier_names = {r.get("carrier_path") for r in rows}
    source_rates = {int(r.get("sample_rate") or 0) for r in rows}
    if None in carrier_names or len(carrier_names) != 1:
        raise RuntimeError(f"mixed/missing carrier_path in one sid: {carrier_names}")
    if 0 in source_rates or len(source_rates) != 1:
        raise RuntimeError(f"mixed/missing carrier sample rates: {source_rates}")
    source_sr = next(iter(source_rates))
    decode_sr = int(target_sr or source_sr)
    if decode_sr <= 0:
        raise ValueError(f"invalid carrier decode rate: {decode_sr}")

    carrier = _resolve_path(next(iter(carrier_names)), sid_dir)
    windows: list[tuple[int, int, int]] = []
    outputs: list[np.ndarray] = []
    for i, row in enumerate(rows):
        try:
            source_a = int(row["carrier_start_samples"])
            source_b = int(row["carrier_end_samples"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid carrier window in row {i}: {row}") from exc
        source_a = max(0, source_a)
        source_b = max(source_a, source_b)
        # Integer floor matches AudioBundle's int(seconds * sample_rate) cut.
        a = source_a * decode_sr // source_sr
        b = source_b * decode_sr // source_sr
        max_window_s = float(os.environ.get("CONTINUO_DATA_MAX_CARRIER_WINDOW_S", "21600"))
        if b - a > int(max_window_s * decode_sr):
            raise RuntimeError(
                f"carrier window {i} is {(b-a)/decode_sr:.1f}s, "
                f"above safety cap {max_window_s:.1f}s"
            )
        windows.append((a, b, i))
        outputs.append(np.empty(max(0, b - a), dtype=np.float32))

    ordered = sorted(windows, key=lambda x: (x[0], x[1], x[2]))
    filled = [0] * len(rows)
    max_requested_end = max(b for _, b, _ in ordered)
    proc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-i", carrier,
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ar", str(decode_sr), "-ac", "1", "pipe:1",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    active: list[tuple[int, int, int]] = []
    next_window = 0
    frame_pos = 0
    remainder = b""
    completed_early = False
    try:
        while True:
            raw = proc.stdout.read(4 * 262_144)
            if not raw:
                break
            raw = remainder + raw
            usable = len(raw) - (len(raw) % 4)
            remainder = raw[usable:]
            if usable == 0:
                continue
            chunk = np.frombuffer(raw[:usable], dtype=np.float32)
            chunk_start = frame_pos
            chunk_end = frame_pos + int(chunk.shape[0])
            while next_window < len(ordered) and ordered[next_window][0] < chunk_end:
                active.append(ordered[next_window])
                next_window += 1
            still_active = []
            for a, b, idx in active:
                lo = max(a, chunk_start)
                hi = min(b, chunk_end)
                if hi > lo:
                    outputs[idx][lo - a:hi - a] = chunk[lo - chunk_start:hi - chunk_start]
                    filled[idx] += hi - lo
                if b > chunk_end:
                    still_active.append((a, b, idx))
            active = still_active
            frame_pos = chunk_end
            if frame_pos >= max_requested_end and next_window == len(ordered) and not active:
                completed_early = True
                if proc.poll() is None:
                    proc.terminate()  # no need to decode the carrier tail
                # Closing our read end is what actually stops it. SIGTERM alone
                # deadlocks whenever the undecoded tail exceeds the 64 KiB pipe
                # buffer: ffmpeg blocks writing into the pipe we just stopped
                # draining while we block on stderr's EOF, and the GPU idles
                # until someone notices. The close makes that write fail.
                proc.stdout.close()
                break
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        returncode = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    if returncode != 0 and not completed_early:
        raise RuntimeError(
            f"ffmpeg carrier decode failed for {carrier}: "
            f"{stderr.decode(errors='replace').strip()[:600]}"
        )
    for i, ((a, b, _), got) in enumerate(zip(windows, filled)):
        if got == b - a:
            continue


        available = max(0, min(b, frame_pos) - a)
        short = (b - a) - got
        if got > 0 and got == available and 0 < short <= _CARRIER_TAIL_SLACK:
            outputs[i][got:] = 0.0
            Logger.get_logger().warning(
                "carrier window %d is short by %d samples (%.2f ms); padded: %s",
                i, short, short / decode_sr * 1000.0, carrier,
            )
            continue
        raise RuntimeError(
            f"carrier window {i} [{a}, {b}) exceeds decoded "
            f"length {frame_pos} for {carrier} (got {got}/{b - a})"
        )
    return [(wav, decode_sr) for wav in outputs]


def _read_segments_parallel(
    rows: list[dict], sid_dir: str, target_sr: int | None = None,
) -> list[tuple[np.ndarray, int]]:
    """Read legacy loose clips, or decode one carrier for all new rows."""
    if rows and any(r.get("carrier_path") for r in rows):
        if not all(r.get("carrier_path") for r in rows):
            raise RuntimeError("mixed loose-audio and carrier rows in one sid")
        return _read_carrier_windows(rows, sid_dir, target_sr=target_sr)
    if len(rows) > 1:
        return list(_SEG_READ_POOL.map(lambda r: _load_segment_audio(r, sid_dir), rows))
    return [_load_segment_audio(rows[0], sid_dir)]


def _segment_from_meta(meta: dict) -> Segment:
    """Rebuild a Segment object from a partial-manifest row.

    ``to_legacy_dict`` flattens speaker/language/quality/extra into the
    JSON; ``from_legacy_dict`` reverses that. Anything Phase 2 wrote on
    its own (``audio_path``, ``sample_rate``) goes into ``extra``.
    """
    return Segment.from_legacy_dict(meta)


def _shard_dirs(root: str, shard: int, num_shards: int) -> list[str]:
    """Top-level data dirs assigned to this shard.

    The big run uses a 2-hex fan-out (``<root>/<sid[:2]>/<sid>/``). We shard
    by the top dir name's hex value (``int(name, 16) % num_shards``) so each
    concurrent Phase-2 instance lists only its ~1/num_shards of the 256
    fan-out dirs, instead of every instance globbing the whole tree each round
    (an NFS metadata storm at 9+ instances). A sid's owner is therefore
    deterministic and stable across instances: ``int(sid[:2], 16) %
    num_shards``. The flat single-folder layout (``<root>/<sid>/``) shards its
    sid dirs the same way. ``num_shards <= 1`` returns every data dir;
    ``_``-prefixed dirs (_claims/_manifests/_state/markers) are skipped.
    """
    out: list[str] = []
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for name in names:
        if name.startswith("_"):
            continue
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        try:
            key = int(name, 16)
        except ValueError:
            key = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16)
        if num_shards <= 1 or key % num_shards == shard:
            out.append(full)
    return out


def _glob_processed(
    root: str, suffix: str, shard: int = 0, num_shards: int = 1
) -> list[str]:
    """Match both layouts, scanning only this shard's top-level dirs:"""
    out: list[str] = []
    for d in _shard_dirs(root, shard, num_shards):
        out.extend(glob.glob(os.path.join(d, f"*{suffix}")))        # flat
        out.extend(glob.glob(os.path.join(d, "*", f"*{suffix}")))   # fan-out
    return out


def _scan_partials(
    processed_root: str, shard: int = 0, num_shards: int = 1
) -> list[tuple[str, str]]:
    """Find ``(partial_path, final_path)`` pairs that still need Phase 2."""
    out: list[tuple[str, str]] = []
    for partial in _glob_processed(processed_root, ".partial.json", shard, num_shards):
        sid_dir = os.path.dirname(partial)
        sid = os.path.basename(partial)[: -len(".partial.json")]
        final = os.path.join(sid_dir, sid + ".json")
        if os.path.exists(final):
            continue  # already transcribed
        out.append((partial, final))
    return out


def _scan_long_chunks(
    processed_root: str, shard: int = 0, num_shards: int = 1
) -> list[tuple[str, str]]:
    """Find ``(long_chunks_path, long_final_path)`` pairs needing Phase 2.

    Mirrors ``_scan_partials`` but for the long track. Resume is
    independent — a file may have its shorts transcribed but not yet
    its longs, or vice versa.
    """
    out: list[tuple[str, str]] = []
    for src in _glob_processed(processed_root, ".long_chunks.json", shard, num_shards):
        sid_dir = os.path.dirname(src)
        sid = os.path.basename(src)[: -len(".long_chunks.json")]
        dst = os.path.join(sid_dir, sid + ".long.json")
        if os.path.exists(dst):
            continue
        out.append((src, dst))
    return out


def _prefix_dirs(root: str) -> list[str]:
    """Existing top-level data dirs (fan-out prefixes or flat SIDs)."""
    out: list[str] = []
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for name in names:
        if name.startswith("_"):
            continue
        if os.path.isdir(os.path.join(root, name)):
            out.append(name)
    return out


def _claim_lock(lock_path: str, lease_s: float) -> bool:
    """Atomically claim ``lock_path`` (O_EXCL); steal it if its mtime is older
    than ``lease_s`` (previous owner died mid-prefix). World-writable for
    cross-uid multi-node. Returns True iff this process now owns it."""
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        os.write(fd, str(time.time()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(lock_path) > lease_s:
                os.utime(lock_path, None)  # steal a stale claim
                return True
        except OSError:
            pass
        return False
    except OSError:
        return False


# ---- drained-prefix markers -------------------------------------------------
# The claim lock only says "someone is working this prefix RIGHT NOW"; it is
# removed the moment they finish. So every instance still walks the full 256-
# prefix list and pays a real ``_scan_prefix`` (a ~3900-sid-dir glob) on prefixes
# a peer already drained, just to learn there is nothing left. With N instances
# that multiplies redundant scans across the 256 prefixes.
# A drained marker makes that "nothing left" answer cost 1 read + 1 stat instead
# of the glob. Two deliberate constraints keep it honest:
#   * Written ONLY when a scan found NOTHING — never after draining work. A
#     prefix whose drain hit a per-file failure (logged to _failed_phase2.jsonl,
#     final not written) therefore stays discoverable and is still retried by the
#     next instance exactly as before. This costs one extra scan per prefix (the
#     one that observes it empty and marks it) but changes no resume/retry
#     semantics: scans drop ~16x -> ~2x per prefix, not to 1x.
#   * Invalidated by a fingerprint of the prefix dir, captured BEFORE the scan
#     so anything phase 1 adds mid-scan invalidates the marker instead of being
#     silently skipped. This is what makes the marker safe under ``--loop``
#     (phase 1 still streaming), not just for a one-shot drain.


def _prefix_fingerprint(pdir: str) -> "str | None":
    """Cheap fingerprint of a prefix dir: (entry count, mtime). None if unstattable.

    Phase 1 lands each recording in a NEW ``<prefix>/<sid>/`` dir and nothing
    ever removes one (``--prune-segments`` deletes segment FILES inside a sid
    dir, never the dir), so the entry count grows strictly monotonically as
    fresh work arrives.

    The count — not the mtime — is what makes this correct. mtime alone is
    unusable here for three independent reasons, all of which silently skip real
    work: (1) a sid created in the same filesystem timestamp tick as the marker
    leaves mtime byte-identical; (2) NFS
    attribute caching can serve a stale dir mtime for up to acdirmax (~60s);
    (3) mtime comes from the server's clock while any freshness comparison would
    use the client's, so clock skew corrupts it. An entry count has none of those
    failure modes. mtime rides along only as a cheap second signal.

    Costs 1 readdir + 1 stat, versus _scan_prefix's ~3900 per-sid readdirs.
    """
    try:
        n = len(os.listdir(pdir))
        mt = os.stat(pdir).st_mtime
    except OSError:
        return None
    return f"{n}:{mt!r}"


def _prefix_drained(marker: str, pdir: str) -> bool:
    """True iff a peer proved this prefix empty for our track AND the prefix dir
    is unchanged since (no new phase-1 output landed)."""
    try:
        with open(marker) as f:
            recorded = f.read().strip()
    except OSError:
        return False  # never marked (or unreadable) -> scan it
    fp = _prefix_fingerprint(pdir)
    return fp is not None and recorded == fp


def _mark_drained(marker: str, fp_before: "str | None") -> None:
    """Record that this prefix scanned empty at fingerprint ``fp_before``
    (captured BEFORE the scan). Best-effort and atomic; world-writable for
    cross-uid multi-node. A lost marker only costs a redundant scan, never
    correctness."""
    if fp_before is None:
        return
    tmp = f"{marker}.{os.getpid()}.tmp"
    try:
        fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o666)
        try:
            os.write(fd, fp_before.encode())
        finally:
            os.close(fd)
        os.replace(tmp, marker)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _scan_prefix(
    root: str, prefix: str, secondary: bool, tracks: str = "both",
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """``(short_work, long_work)`` for a single fan-out prefix dir.

    Primary mode skips sids whose ``<sid>.json`` already exists; secondary
    mode returns every partial (the fill-in logic skips covered indices).
    Long track is primary-only.

    ``tracks`` ("both"/"short"/"long") gates which half is *scanned*. The
    caller could scan both and discard the unwanted half, causing unnecessary
    metadata requests. Scanning only the requested track preserves the same
    (tracks x secondary) behavior."""
    pdir = os.path.join(root, prefix)

    def _file_names(d: str) -> list[str]:

        try:
            with os.scandir(d) as it:
                return [e.name for e in it]
        except OSError:
            return []

    def _entries(d: str) -> list[tuple[str, bool]]:

        try:
            with os.scandir(d) as it:
                return [(e.name, e.is_dir()) for e in it]
        except OSError:
            return []

    want_short = tracks in ("both", "short")
    want_long = (not secondary) and tracks in ("both", "long")

    def _collect(dirpath: str, names: list[str], work: list, long_work: list) -> None:

        fset = set(names)
        if want_short:
            for n in names:
                if not n.endswith(".partial.json"):
                    continue
                sid = n[: -len(".partial.json")]
                if secondary or (sid + ".json") not in fset:
                    work.append((os.path.join(dirpath, n),
                                 os.path.join(dirpath, sid + ".json")))
        if want_long:
            for n in names:
                if not n.endswith(".long_chunks.json"):
                    continue
                sid = n[: -len(".long_chunks.json")]
                if (sid + ".long.json") not in fset:
                    long_work.append((os.path.join(dirpath, n),
                                      os.path.join(dirpath, sid + ".long.json")))

    work: list[tuple[str, str]] = []
    long_work: list[tuple[str, str]] = []
    top = _entries(pdir)

    _collect(pdir, [n for n, _ in top], work, long_work)

    for name, is_dir in top:
        if not is_dir:
            continue
        sub = os.path.join(pdir, name)
        _collect(sub, _file_names(sub), work, long_work)
    return work, long_work


def _queue_prefix(
    root: str, prefix: str, tracks: str,
    pend_short: list[str], pend_long: list[str],
) -> tuple[list, list, list, list, int]:

    want_short = tracks in ("both", "short")
    want_long = tracks in ("both", "long")

    _names: dict[str, list[str]] = {}

    def names(sid: str) -> list[str]:

        if sid not in _names:
            try:
                with os.scandir(os.path.join(root, prefix, sid)) as it:
                    _names[sid] = [e.name for e in it]
            except OSError:
                _names[sid] = []
        return _names[sid]

    work: list[tuple[str, str]] = []
    long_work: list[tuple[str, str]] = []
    ack_short: list[tuple[str, str]] = []
    ack_long: list[tuple[str, str]] = []
    stale = 0

    def collect(sids, src_suffix, dst_suffix, out, acks, track):
        nonlocal stale
        for sid in sids:
            fset = set(names(sid))
            hit = False
            for n in sorted(fset):
                if not n.endswith(src_suffix):
                    continue
                base = n[: -len(src_suffix)]
                dst = base + dst_suffix
                sid_dir = os.path.join(root, prefix, sid)
                if dst in fset:
                    continue
                hit = True
                out.append((os.path.join(sid_dir, n), os.path.join(sid_dir, dst)))
                acks.append((sid, os.path.join(sid_dir, dst)))
            if not hit:
                work_queue.ack(root, sid, track)
                stale += 1

    if want_short:
        collect(pend_short, ".partial.json", ".json", work, ack_short, "short")
    if want_long:
        collect(pend_long, ".long_chunks.json", ".long.json",
                long_work, ack_long, "long")
    return work, long_work, ack_short, ack_long, stale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dialogue-rescue", action="store_true",
        help="enable permissive dialogue rescue for existing data; accept any "
             "nonempty transcription. Disabled by default",
    )
    ap.add_argument(
        "--config_path", type=str, default="config.json",
        help="Same config Phase 1 used. Phase 2 reads ``stages.asr`` "
             "(must be set), ``filter.min_char_count``, and ``runtime`` knobs.",
    )
    ap.add_argument(
        "--processed-root", type=str, default="",
        help="Override the directory holding ``<sid>/<sid>.partial.json`` "
             "files. Defaults to ``<input_folder_path>_processed`` derived "
             "from the config.",
    )
    ap.add_argument(
        "--prune-segments", action="store_true",
        help="Delete per-segment WAVs of files that finish Phase 2 "
             "successfully. Off by default so failures stay re-runnable.",
    )
    ap.add_argument(
        "--secondary", action="store_true",
        help=(
            "Fill-in pass: don't skip files where ``<sid>.json`` already "
            "exists. Instead, for each ``partial.json`` row whose index "
            "isn't already in ``<sid>.json``, run the configured ASR if "
            "its ``accepts()`` returns True for that row's language, and "
            "merge the resulting rows into ``<sid>.json`` (re-sorted by "
            "start time). Intended workflow: run primary pass with one "
            "ASR (e.g. Qwen3-ASR), then run secondary pass with another "
            "(e.g. IndicConformer for Bengali) so each ASR handles only "
            "the languages it knows."
        ),
    )
    ap.add_argument(
        "--loop", action="store_true",
        help=(
            "Synchronous-pipeline mode: load vLLM ONCE, then repeatedly scan "
            "for new partials and transcribe them, consuming Phase-1's output "
            "as it streams in. Idles when there's no new work; exits when "
            "``_PHASE1_DONE`` exists AND the backlog is drained. Without this "
            "flag, runs a single pass over existing partials and exits "
            "(use that after Phase-1 has fully finished)."
        ),
    )
    ap.add_argument(
        "--phase1-nodes", type=int, default=1,
        help=(
            "minimum number of phase-one nodes to wait for in --loop mode. "
            "RUN.json provides the fixed topology when present; this value "
            "supports older output trees without that file."
        ),
    )
    ap.add_argument(
        "--num-shards", type=int, default=1,
        help=(
            "Total number of concurrent Phase-2 instances across ALL GPUs/nodes. "
            "Default 1 = no sharding (this instance takes everything). Set to the "
            "total instance count and give each instance a distinct --shard so "
            "they split the backlog with zero overlap (no duplicate transcription)."
        ),
    )
    ap.add_argument(
        "--shard", type=int, default=0,
        help=(
            "This instance's shard id in [0, num-shards). Each instance only "
            "processes sids where md5(sid) mod num-shards == shard."
        ),
    )
    ap.add_argument(
        "--prefetch", type=int, default=4,
        help=(
            "Stage 1: how many files' I/O (read partial + segment WAVs, build "
            "the bundle) to prefetch on a background thread pool ahead of the "
            "GPU ASR call, so NAS reads overlap inference. 0/1 = one loader "
            "(still one-ahead); higher hides more NAS latency at more memory. "
            "Default 4."
        ),
    )
    ap.add_argument(
        "--batch-files", type=int, default=1,
        help=(
            "Stage 2: how many files' segments to feed the ASR in ONE batched "
            "call (so the engine stays fed across file boundaries instead of a "
            "small per-file batch). 1 (default) = one file per call (= Stage 1 "
            "only). Output is bit-identical regardless; this only trades a "
            "larger in-flight memory footprint for higher GPU utilization. Try "
            "4-16; set proportional to your KV-cache / memory budget."
        ),
    )
    ap.add_argument(
        "--batch-base", type=str, default="",
        help=(
            "batch-mode parent directory containing independent phase-one output "
            "trees. Phase two scans unfinished batches and writes _PHASE2_DONE "
            "when a sealed batch has no pending work. Mutually exclusive with "
            "--processed-root; phase one uses --batch-recordings"
        ),
    )
    ap.add_argument(
        "--sealed-only", action="store_true",
        help=(
            "in batch mode, process only batches marked _SEALED. A single-pass "
            "run exits after all sealed batches are complete; newly sealed batches "
            "are picked up during the next scan"
        ),
    )
    ap.add_argument(
        "--use-queue", action="store_true",
        help=(
            "discover phase-two work from the optional phase-one queue. Disabled "
            "by default; falls back to scanning when the queue is not enabled"
        ),
    )
    ap.add_argument(
        "--no-prescan", action="store_true",
        help=(
            "Skip the single-shot pre-warmup full-tree scan (``_scan_partials`` "
            "/ ``_scan_long_chunks``) that decides whether to bother loading the "
            "ASR. That scan globs every prefix + stats every partial across the "
            "whole processed tree — slow on NAS and redundant when N instances "
            "each do it. The per-prefix claim loop already discovers work "
            "incrementally (1× total, distributed), so for a single-pass cycle "
            "over a huge tree, skip the pre-scan and go straight to warmup + one "
            "claim round. Cost: warmup runs even if there turns out to be no "
            "work. No effect with --loop or --secondary (they skip it already)."
        ),
    )
    ap.add_argument(
        "--tracks", choices=("both", "short", "long"), default="both",
        help=(
            "Which track(s) to clean. ``both`` (default) transcribes shorts "
            "(<sid>.json) and longs (<sid>.long.json). ``short`` only does the "
            "short track; ``long`` only the long track. Resume is independent "
            "per track, so you can run the two separately (even on different "
            "GPUs/instances). Applies identically to the --secondary (indic) "
            "pass."
        ),
    )
    args = ap.parse_args()
    if args.num_shards < 1 or not (0 <= args.shard < args.num_shards):
        raise SystemExit(
            f"bad sharding: --shard {args.shard} --num-shards {args.num_shards} "
            "(need num_shards >= 1 and 0 <= shard < num_shards)"
        )


    if args.batch_base and args.secondary:
        raise SystemExit(
            "--batch-base cannot be used with --secondary: _PHASE2_DONE does not "
            "track separate passes. Run secondary against one output tree with "
            "--processed-root."
        )
    if args.batch_base and args.tracks != "both":
        raise SystemExit(
            "--batch-base requires --tracks both: _PHASE2_DONE does not track "
            "short and long tracks separately."
        )
    if args.sealed_only and not args.batch_base:
        raise SystemExit("--sealed-only requires --batch-base")

    cfg = PipelineConfig.load(args.config_path)
    if cfg.stages.asr is None:
        raise SystemExit(
            "config.stages.asr is None — Phase 2 needs an ASR adapter. "
            "Set ``stages.asr`` in the config and re-run."
        )

    logger = Logger.get_logger()
    device = "cuda" if detect_gpu() else "cpu"
    logger.info(f"phase-2 device: {device}")
    if args.num_shards > 1:


        logger.info(
            f"phase-2 shard {args.shard}/{args.num_shards} "
            f"(int(dir,16) % num_shards partition)"
        )


    if args.batch_base:
        processed_root = args.batch_base
        if not os.path.isdir(processed_root):
            raise SystemExit(f"batch base not found: {processed_root}")
    else:
        processed_root = (
            args.processed_root
            or cfg.io.input_folder_path.rstrip("/") + "_processed"
        )
        if not os.path.isdir(processed_root):
            raise SystemExit(f"processed root not found: {processed_root}")

    # Single-shot mode: skip the vLLM warmup if there's genuinely nothing to
    # do. Worth it for the PRIMARY pass (the scan is cheap next to the qwen3
    # warmup it guards). NOT worth it for the SECONDARY (indic) pass: a full
    # pre-scan would walk every partial.json in the dataset, so N concurrent
    # instances each pay an N× redundant full-tree scan (tens of
    # minutes) just to answer "is there any work?" — which the claim loop below
    # already discovers per-prefix (distributed across instances, 1× total). So
    # for secondary we skip the pre-check and go straight to warmup + the loop;
    # it finds and drains the work, then exits when no prefix yields any.


    _queue_avail = args.use_queue and (
        bool(args.batch_base) or work_queue.is_enabled(processed_root))
    if not args.loop and not args.secondary and not args.no_prescan:
        if args.batch_base:
            logger.info("batch mode: skip full-tree prescan; batches discover work")
        elif _queue_avail:


            logger.info("queue mode: skip full-tree prescan; the main loop discovers work")
        else:
            _init = (
                _scan_partials(processed_root, args.shard, args.num_shards)
                if args.tracks in ("both", "short") else []
            )
            _init_long = (
                _scan_long_chunks(processed_root, args.shard, args.num_shards)
                if args.tracks in ("both", "long") else []
            )
            if not _init and not _init_long:
                logger.info(f"nothing to do under {processed_root}")
                return

    asr: ASRModel = build_adapter("asr", cfg.stages.asr, device)
    t0 = time.time()
    asr.warmup()  # vLLM loaded ONCE; the loop below reuses it, never reloads.
    logger.info(f"asr warmup {time.time() - t0:.1f}s")

    batch_size = cfg.runtime.batch_size
    dialogue_phase2.set_lid_unsupported(lid_unsupported)
    global _DIALOGUE_RESCUE, _DIALOGUE_GATES, _LONG_MAX_GAP_S, _LONG_MIN_DUR_S
    _DIALOGUE_RESCUE = bool(getattr(args, 'dialogue_rescue', False))

    _DIALOGUE_GATES = cfg.dialogue_window.gates()


    if "CONTINUO_DATA_LONG_MAX_GAP_S" not in os.environ:
        _LONG_MAX_GAP_S = float(cfg.long_chunk.max_gap_s)
    if "CONTINUO_DATA_LONG_MIN_DUR_S" not in os.environ:
        _LONG_MIN_DUR_S = float(cfg.long_chunk.min_duration_s)
    min_char = cfg.filter.min_char_count
    ratio_filter = cfg.filter.ratio_filter


    def _phase1_done() -> bool:


        return nodes.phase1_done(processed_root, min_nodes=args.phase1_nodes)
    LOOP_SLEEP = 30    # seconds to idle when no new partials yet
    LEASE_S = 1800.0   # prefix-claim lease; a lock older than this is stolen
    # Dynamic prefix claiming (self-balancing, elastic): each instance claims
    # one fan-out prefix dir at a time via an O_EXCL lease lock, drains its
    # backlog, releases, and moves on. Fast instances naturally claim more
    # prefixes; a slow/remote node claims fewer; a killed instance's lock is
    # stolen after the lease. Scans stay cheap (one ~1/256 prefix per claim,
    # not the whole tree) and instances never duplicate each other. Add/stop/
    # move instances freely — no fixed partition to reconfigure. Primary and
    # secondary (indic) use separate claim dirs so they don't block each other.


    batch_base = args.batch_base or ""
    if batch_base:
        from pipeline.io import batches

    def _active_roots() -> list:
        if not batch_base:
            return [(None, processed_root)]
        roots = list(batches.active_roots(batch_base))
        if args.sealed_only:


            roots = [(i, r) for i, r in roots if batches.is_sealed(batch_base, i)]
        return roots

    _rs: dict = {}

    def _root_state(root: str) -> dict:
        st = _rs.get(root)
        if st is None:
            cd = os.path.join(
                root, "_phase2_claims" + ("_sec" if args.secondary else ""))
            try:
                os.makedirs(cd, exist_ok=True)
            except OSError:
                pass


            qm = (args.use_queue and not args.secondary
                  and work_queue.is_enabled(root))
            st = _rs[root] = {
                "claims_dir": cd,
                "failed": os.path.join(root, "_failed_phase2.jsonl"),
                "queue_mode": qm,


                "reconciled": not qm,
                "scan_override": False,
            }
            logger.info(f"phase-2 root {root}: "
                        f"discovery={'QUEUE' if qm else 'SCAN'}")
        return st

    def _drain_root(root: str, st: dict, round_i: int) -> bool:

        claims_dir = st["claims_dir"]
        st["contested"] = False
        failed_log_path = st["failed"]
        use_queue = st["queue_mode"] and not st["scan_override"]
        _want_tracks = [t for t in ("short", "long")
                        if args.tracks in ("both", t)]
        prefixes = (
            work_queue.prefixes(root, _want_tracks) if use_queue
            else _prefix_dirs(root)
        )
        random.shuffle(prefixes)
        did_any = False
        for prefix in prefixes:
            pdir = os.path.join(root, prefix)
            pend_short = pend_long = []
            if use_queue:


                if args.tracks in ("both", "short"):
                    pend_short = work_queue.pending(root, prefix, "short")
                if args.tracks in ("both", "long"):
                    pend_long = work_queue.pending(root, prefix, "long")
                if not (pend_short or pend_long):
                    continue
            else:
                # Per-track marker: short and long instances share this claims dir,
                # so a long-drained prefix must not make a short instance skip.
                marker = os.path.join(claims_dir, f"{prefix}.done_{args.tracks}")
                if _prefix_drained(marker, pdir):
                    continue  # peer proved it empty + nothing new landed (1 read+1 stat)
            lock = os.path.join(claims_dir, prefix + ".lock")
            if not _claim_lock(lock, LEASE_S):


                st["contested"] = True
                continue  # another instance owns this prefix
            ack_short: list = []
            ack_long: list = []
            try:
                if use_queue:
                    work, long_work, ack_short, ack_long, stale = _queue_prefix(
                        root, prefix, args.tracks, pend_short, pend_long
                    )
                    if stale:
                        logger.info(
                            f"[round {round_i}] prefix {prefix}: cleared {stale} stale "
                            f"entries (final output exists or source is missing)"
                        )
                else:
                    # Capture BEFORE the scan so phase-1 output landing mid-scan
                    # invalidates the marker instead of being skipped.
                    fp_before = _prefix_fingerprint(pdir)
                    work, long_work = _scan_prefix(
                        root, prefix, args.secondary, args.tracks
                    )


                n_poison = len(work) + len(long_work)
                work = [t for t in work if not _poisoned(t[0])]
                long_work = [t for t in long_work if not _poisoned(t[0])]
                n_poison -= len(work) + len(long_work)
                if not (work or long_work):


                    if not use_queue and not n_poison:
                        _mark_drained(marker, fp_before)
                    continue
                did_any = True
                logger.info(
                    f"[round {round_i}] prefix {prefix}: {len(work)} short"
                    f"{' SECONDARY' if args.secondary else ''}, {len(long_work)} long"
                )
                _run_short_track(
                    work, asr, batch_size=batch_size, min_char=min_char,
                    ratio_filter=ratio_filter, filter_cfg=cfg.filter,
                    prune_segments=args.prune_segments, secondary=args.secondary,
                    logger=logger, failed_log_path=failed_log_path,
                    heartbeat=lambda lk=lock: _touch(lk), desc=f"tx {prefix}",
                    prefetch=args.prefetch, batch_files=args.batch_files,
                )
                for src, dst in long_work:
                    sid_dir = os.path.dirname(src)
                    try:
                        _transcribe_long_chunks(
                            src_path=src, dst_path=dst, sid_dir=sid_dir,
                            asr=asr, logger=logger, ratio_filter=ratio_filter,
                        )
                    except Exception as e:  # noqa: BLE001 - keep loop alive
                        logger.exception(f"long-track failed on {src}: {e}")
                        _record_phase2_fail(failed_log_path, src, e, key="long_chunks")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _touch(lock)
            finally:


                if ack_short:
                    work_queue.ack_finished(root, "short", ack_short)
                if ack_long:
                    work_queue.ack_finished(root, "long", ack_long)
                try:
                    os.remove(lock)  # release (re-claimable for new partials)
                except OSError:
                    pass
        return did_any

    def _finish_batch(bi: int, root: str, st: dict, worked: bool) -> bool:

        if worked or not batches.is_sealed(batch_base, bi):
            return False

        def _blocked() -> bool:
            if st.get("contested"):
                logger.info(
                    f"batch {batches.batch_name(bi)}: a prefix is locked by another "
                    f"worker; its status is unknown, so sealing is deferred"
                )
                return True
            bad = _root_poison(root)
            if bad:
                logger.warning(
                    f"batch {batches.batch_name(bi)}: {len(bad)} files failed "
                    f"{_P2_MAX_FAILS} times; _PHASE2_DONE is withheld pending "
                    f"restart or review. Examples: {bad[:3]}"
                )
                return True
            return False

        if st["scan_override"]:
            if _blocked():
                return False
            batches.mark_phase2_done(batch_base, bi)
            logger.info(f"batch {batches.batch_name(bi)} complete: _PHASE2_DONE")
            _rs.pop(root, None)
            return True
        if not st["reconciled"]:
            st["scan_override"] = True
            st["reconciled"] = True
            logger.info(f"batch {batches.batch_name(bi)} queue empty; running fallback scan")
            return True
        if _blocked():
            return False
        batches.mark_phase2_done(batch_base, bi)
        logger.info(f"batch {batches.batch_name(bi)} complete: _PHASE2_DONE")
        _rs.pop(root, None)
        return True


    reconciled = True
    scan_override = False
    if not batch_base:
        st0 = _root_state(processed_root)
        reconciled = st0["reconciled"]

    round_i = 0
    while True:
        round_i += 1
        did_any = False
        try:
            round_roots = _active_roots()
        except OSError as ex:


            logger.warning(f"cannot read batch metadata; will not exit this round: {ex}")
            if args.loop:
                time.sleep(LOOP_SLEEP)
                continue
            raise
        for bi, root in round_roots:
            st = _root_state(root)
            if not batch_base:
                st["scan_override"] = scan_override
            worked = _drain_root(root, st, round_i)
            did_any = did_any or worked
            if batch_base and _finish_batch(bi, root, st, worked):
                did_any = True

        # ---- loop control ----


        def _needs_reconcile() -> bool:
            nonlocal reconciled, scan_override
            if reconciled:
                return False
            logger.info("queue drained; running a fallback scan for missed work")
            reconciled = True
            scan_override = True
            return True

        if batch_base:


            if not args.loop:
                if did_any:
                    continue
                break
            phase1_done = _phase1_done()
            remaining_roots = None
            if phase1_done and not did_any:
                try:
                    remaining_roots = _active_roots()
                except OSError as ex:
                    logger.warning(f"cannot recheck batch metadata; will not exit: {ex}")
                    time.sleep(LOOP_SLEEP)
                    continue
            if phase1_done and not did_any and remaining_roots == []:
                logger.info("phase1 and all batches complete; exiting --loop")
                break
            if phase1_done and not did_any and _poisoned_all():


                logger.warning(
                    "phase1 complete but repeatedly failed files remain; exiting --loop "
                    "with their batches unsealed for retry or review"
                )
                break
            if not did_any:
                time.sleep(LOOP_SLEEP)
            continue

        if scan_override:
            scan_override = False
            if did_any:


                reconciled = False
                continue
        if not args.loop:
            if _needs_reconcile():
                continue
            break  # single-shot: one pass over the backlog, then exit
        if _phase1_done() and not did_any:
            if _needs_reconcile():
                continue
            logger.info("phase1 done + no claimable work → exiting --loop")
            break  # synchronous pipeline: phase1 finished and nothing pending
        if not did_any:
            time.sleep(LOOP_SLEEP)  # idle: wait for phase1 to produce more

    bad = _poisoned_all()
    if bad:
        logger.warning(
            f"this process skipped {len(bad)} files after {_P2_MAX_FAILS} failures "
            f"(not finalized or acknowledged; restart to retry): "
            + ", ".join(bad[:5]) + (" ..." if len(bad) > 5 else "")
        )


def _ratio_ok(seg, text: str, ratio_filter) -> bool:
    """Whether ``duration / char_count`` falls inside the per-language band.

    See :class:`pipeline.config.FilterConfig.ratio_filter` for the bound
    table. Per-language bounds wins over ``default``. Segments with no
    text or zero chars after stripping are rejected here too (no division
    by zero downstream)."""
    chars = get_char_count(text)
    if chars == 0:
        return False
    duration = float(
        (getattr(seg, "extra", None) or {}).get("speech_s")
        or (seg.end - seg.start))
    if duration <= 0:
        return False
    ratio = duration / chars
    bounds = ratio_filter.get(seg.language) or ratio_filter.get("default")
    if bounds is None:
        return True  # no rule configured for this language
    return bounds.min <= ratio <= bounds.max


_LEDGER_DIRNAME = "_shorts_ledger"


def _ledger_dir(sid_dir: str) -> str:
    coord = os.environ.get("CONTINUO_DATA_COORD_DIR")
    if coord:
        return os.path.join(coord, _LEDGER_DIRNAME)
    root = os.path.dirname(os.path.dirname(os.path.abspath(sid_dir)))
    return os.path.join(root, _LEDGER_DIRNAME)


def _append_shorts_ledger(sid_dir: str, sid: str, rows: list, secondary: bool) -> None:

    if not rows:
        return
    secs: dict[str, float] = {}
    for r in rows:
        try:
            d = float(r.get("end", 0)) - float(r.get("start", 0))
        except (TypeError, ValueError):
            continue
        if d <= 0:
            continue
        lang = (r.get("language") or "unknown")
        secs[lang] = round(secs.get(lang, 0.0) + d, 3)
    if not secs:
        return
    try:
        led = _ledger_dir(sid_dir)
        os.makedirs(led, exist_ok=True)
        path = os.path.join(led, f"{socket.gethostname()}_{os.getpid()}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"sid": sid, "sec": secs, "ts": time.time(),
                 "pass": "secondary" if secondary else "primary"},
                ensure_ascii=False) + "\n")
    except OSError:
        pass


# ====================================================================
# Short-track load / infer / finalize seam
# ====================================================================
# ``_transcribe_one_file`` below is the per-file path. The throughput driver
# (``_run_short_track``)
# is built from the same three steps factored apart so I/O can be prefetched
# off the inference thread (Stage 1) and several files' segments can share one
# ASR call (Stage 2):
#   _load_file_for_asr  — CPU/IO only, prefetchable, no GPU. Reads the
#                         partial, applies LID routing, loads the per-segment
#                         WAVs, builds the concat bundle + relative segments.
#                         Returns None to skip, or a _LoadedFile (which may be
#                         a ``write_empty`` marker needing no inference).
#   _infer_one          — the GPU step for ONE file (bucket by language hint,
#                         one asr.transcribe per bucket). Stage 2 replaces this
#                         with a cross-file batched variant.
#   _finalize_file      — stitch + the 4 post-ASR filters + secondary merge +
#                         atomic write + optional prune. CPU/IO only.


@dataclass
class _LoadedFile:
    """One file's CPU/IO-prepared state, ready for the GPU inference step.

    ``write_empty`` marks the terminal "write [] resume-marker" case (empty
    partial, or all rows routed away) — the driver writes the marker and skips
    inference. Otherwise ``bundle`` + ``rel_segments`` feed the ASR and
    ``transcribed`` (position i -> Segment) is filled in before finalize.
    """

    partial: str
    final: str
    sid_dir: str
    secondary: bool
    prune: bool = False  # driver policy, set on the returned object by the caller
    write_empty: bool = False


    dialogue_only: bool = False
    segments: list = field(default_factory=list)       # orig Segments (orig bounds)
    rows: list = field(default_factory=list)           # orig partial rows, aligned
    rel_segments: list = field(default_factory=list)   # relative to bundle, aligned
    bundle: object = None                              # AudioBundle | None
    sample_rate: int = 0
    existing_rows: list = field(default_factory=list)  # secondary pre-existing final
    transcribed: dict = field(default_factory=dict)    # position i -> Segment


def _load_all_shorts_rows(partial: str) -> list[dict]:

    p = partial.replace(".partial.json", ".all_shorts.json")
    if p != partial:
        try:
            with open(p) as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    with open(partial) as f:
        return json.load(f)


def _load_file_for_asr(
    partial: str, final: str, asr: ASRModel, secondary: bool,
) -> "_LoadedFile | None":
    """CPU/IO half of Phase-2 for one file (prefetchable; touches no GPU).

    Mirrors the load+route+build-bundle section of ``_transcribe_one_file``.
    Returns ``None`` to skip the file entirely, or a :class:`_LoadedFile`
    (``write_empty=True`` for the empty-final marker case, else carrying the
    bundle + relative segments to infer). ``prune`` is the driver's policy, set
    on the returned object by the caller.
    """
    sid_dir = os.path.dirname(partial)


    rows = _load_all_shorts_rows(partial)


    rows = [r for r in rows if not lid_unsupported(r.get("language"))]

    def _mk(**kw) -> "_LoadedFile":
        return _LoadedFile(
            partial=partial, final=final, sid_dir=sid_dir,
            secondary=secondary, **kw,
        )

    if not rows:
        # Phase 1 produced no surviving segments. Write the empty final so
        # resume marks it done — unless secondary already has a final.


        if secondary and os.path.exists(final):
            return None
        return _mk(write_empty=True)

    # ---- LID-based routing (S4), identical to _transcribe_one_file ----
    existing_rows: list[dict] = []
    if secondary:
        if os.path.exists(final):
            with open(final) as f:
                existing_rows = json.load(f)
            covered = {r.get("index") for r in existing_rows if r.get("index")}
            rows = [r for r in rows if r.get("index") not in covered]
        rows = [r for r in rows if asr.accepts((r.get("language") or "").strip().lower())]
        if not rows:
            return None
    else:
        rows = [r for r in rows
                if (lid := (r.get("language") or "").strip().lower()) in ("", "unknown")
                or asr.accepts(lid)]
        if not rows:
            # Nothing for the primary ASR here (all indic-only LID). Write an
            # empty final so resume marks the primary short-track pass done;
            # the secondary pass merges its rows in afterwards.
            if os.path.exists(final):
                return _mk(dialogue_only=True) if _has_phase1_windows(sid_dir) else None
            return _mk(write_empty=True)

    segments = [_segment_from_meta(r) for r in rows]

    # Concatenate the per-segment WAVs into one buffer; rel_segments carry
    # start/end within that buffer (the ASR slices it per segment).
    clips: list[np.ndarray] = []
    sample_rate = 0
    cursor = 0.0
    rel_segments: list[Segment] = []
    loaded = _read_segments_parallel(
        rows, sid_dir, target_sr=int(getattr(asr, "SR", 0) or 0) or None,
    )
    for seg, (wav, sr) in zip(segments, loaded):
        if sample_rate == 0:
            sample_rate = sr
        elif sr != sample_rate:
            raise RuntimeError(
                f"mixed sample rates in {partial}: {sr} vs {sample_rate}"
            )
        dur = len(wav) / sr
        clips.append(wav)
        rel_segments.append(Segment(
            start=cursor, end=cursor + dur,
            speaker=seg.speaker, index=seg.index,
            language=seg.language if seg.language and seg.language != "unknown" else None,
            extra=dict(seg.extra),
        ))
        cursor += dur

    big_buffer = np.concatenate(clips).astype(np.float32, copy=False)
    bundle = AudioBundle(waveform=big_buffer, sample_rate=sample_rate, name="phase2")
    return _mk(
        segments=segments, rows=rows, rel_segments=rel_segments,
        bundle=bundle, sample_rate=sample_rate, existing_rows=existing_rows,
    )


def _checked_transcribe(
    asr: ASRModel, bundle: AudioBundle, segments: list[Segment], *,
    language: str | None, batch_size: int, context: str,
) -> list[Segment]:
    """Run ASR and prove one ordered result exists for every input segment.

    Positional ``zip`` silently truncates when a model drops a result, which
    can attach a short transcript to a longer audio interval. Cardinality and
    stable segment identity are therefore resume-critical invariants.
    """
    out = list(asr.transcribe(
        bundle, segments, language=language, batch_size=batch_size,
    ))
    if len(out) != len(segments):
        raise RuntimeError(
            f"ASR result count mismatch in {context}: "
            f"expected {len(segments)}, got {len(out)}"
        )
    for pos, (expected, actual) in enumerate(zip(segments, out)):
        if actual is None:
            raise RuntimeError(f"ASR returned None at {context}[{pos}]")
        if expected.index is not None and actual.index != expected.index:
            raise RuntimeError(
                f"ASR result order/id mismatch in {context}[{pos}]: "
                f"expected {expected.index!r}, got {actual.index!r}"
            )
    return out


def _infer_one(asr: ASRModel, lf: "_LoadedFile", batch_size: int, logger) -> None:
    """Single-file GPU step: bucket by language hint, one transcribe per bucket.

    Fills ``lf.transcribed`` (position i -> Segment). Identical grouping to
    ``_transcribe_one_file``. Stage 2's cross-file variant fills the same map.
    """
    buckets: dict[str | None, list[int]] = defaultdict(list)
    for i, rs in enumerate(lf.rel_segments):
        buckets[rs.language].append(i)
    for lang, idxs in buckets.items():
        subs = [lf.rel_segments[i] for i in idxs]
        out = _checked_transcribe(
            asr, lf.bundle, subs, language=lang, batch_size=batch_size,
            context=f"{lf.partial} lang={lang!r}",
        )
        for i, t in zip(idxs, out):
            lf.transcribed[i] = t


def _infer_batch(
    asr: ASRModel, lfs: "list[_LoadedFile]", batch_size: int, logger,
) -> None:
    """Stage 2: one ASR pass over SEVERAL files' segments at once.

    Concatenates each file's audio — already resampled to the adapter's working
    SR (``asr.SR``) — into one bundle, then offsets every file's segments into
    that shared timeline and buckets the whole lot by language for a single
    ``transcribe`` call per language. This keeps the vLLM engine fed across file
    boundaries instead of one small per-file batch at a time.

    **Output is bit-identical to ``_infer_one``.** Each file is resampled exactly
    as the per-file path would (same source buffer, same target SR, same cache),
    and the ``+0.5``-sample offset cancels float truncation, so for every segment
    the adapter slices precisely the same ``[ps:pe]`` window of that file's
    resampled buffer. The only thing that changes is how many segments share one
    ``transcribe`` call. (Relies on the adapter slicing ``int(seg.start*asr.SR)``,
    which qwen3 / indic both do.)

    Falls back to per-file when the adapter doesn't expose ``SR`` (can't pin the
    bundle SR) or the batch is a single file. Fills each ``lf.transcribed``.
    """
    target_sr = int(getattr(asr, "SR", 0) or 0)
    if target_sr <= 0 or len(lfs) == 1:
        for lf in lfs:
            _infer_one(asr, lf, batch_size, logger)
        return
    # ``get_at_sr(target_sr)`` resamples each file's buffer to target_sr from its
    # own source SR, so files with different source SRs can share one mega-bundle
    # without grouping — they're all at target_sr once concatenated.
    buffers: list[np.ndarray] = []
    mega_segs: list[Segment] = []
    routing: list[tuple] = []  # parallel to mega_segs: (lf, local_i)
    cursor = 0                 # running sample offset at target_sr
    for lf in lfs:
        buf = lf.bundle.get_at_sr(target_sr).astype(np.float32, copy=False)
        for li, rs in enumerate(lf.rel_segments):
            ps = int(rs.start * target_sr)
            pe = int(rs.end * target_sr)
            mega_segs.append(Segment(
                start=(cursor + ps + 0.5) / target_sr,
                end=(cursor + pe + 0.5) / target_sr,
                speaker=rs.speaker, index=rs.index,
                language=rs.language, extra=dict(rs.extra),
            ))
            routing.append((lf, li))
        buffers.append(buf)
        cursor += len(buf)
    mega = AudioBundle(
        waveform=np.concatenate(buffers).astype(np.float32, copy=False),
        sample_rate=target_sr, name="phase2_batch",
    )
    buckets: dict[str | None, list[int]] = defaultdict(list)
    for gi, ms in enumerate(mega_segs):
        buckets[ms.language].append(gi)
    for lang, gidxs in buckets.items():
        subs = [mega_segs[gi] for gi in gidxs]
        out = _checked_transcribe(
            asr, mega, subs, language=lang, batch_size=batch_size,
            context=f"phase2 batch lang={lang!r} files={len(lfs)}",
        )
        for gi, t in zip(gidxs, out):
            lf, li = routing[gi]
            lf.transcribed[li] = t


DIALOGUE_MIN_CHAR = 1


_DIALOGUE_RESCUE = False


_DIALOGUE_GATES = Gates()


def _dialogue_paths(lf: "_LoadedFile") -> tuple[str, str]:

    def _sib(suffix: str) -> str:
        p = lf.partial.replace(".partial.json", suffix)
        return p if p != lf.partial else os.path.join(
            lf.sid_dir, os.path.basename(lf.sid_dir) + suffix)
    return _sib(".dialogue_chunk.json"), _sib(".dialogue.json")


_SUPPORTED_LIDS: frozenset[str] | None = None


def supported_lids() -> frozenset[str]:

    global _SUPPORTED_LIDS
    if _SUPPORTED_LIDS is not None:
        return _SUPPORTED_LIDS
    langs: set[str] = set()
    for mod_name in ("pipeline.adapters.qwen3_asr",
                     "pipeline.adapters.indic_conformer_asr"):
        try:
            mod = __import__(mod_name, fromlist=["x"])
        except Exception:
            continue
        for n in dir(mod):
            obj = getattr(mod, n)
            tbl = getattr(obj, "SUPPORTED_LANGUAGES", None)
            if isinstance(tbl, (set, frozenset)):
                langs |= set(tbl)
    _SUPPORTED_LIDS = frozenset(langs)
    return _SUPPORTED_LIDS


def lid_unsupported(lid: str | None) -> bool:

    lid = (lid or "").strip().lower()
    if lid in ("", "unknown"):
        return False
    tbl = supported_lids()
    if not tbl:
        return False
    return lid not in tbl


def _asr_model_name(asr: ASRModel) -> str:

    return getattr(asr, "NAME", None) or type(asr).__name__


def _dump_asr_texts(asr: ASRModel, lf: "_LoadedFile", segments: list, rows: list) -> None:

    model = _asr_model_name(asr)
    table = {}
    for i, row in enumerate(rows):
        idx = row.get("index")
        if idx is None:
            continue
        table[idx] = asr_texts.build_row(lf.transcribed.get(i), asr, model)
    if not table:
        return
    sid = os.path.basename(lf.partial)[: -len(".partial.json")]
    path = asr_texts.path_for(lf.sid_dir, sid)
    merged = asr_texts.merge(asr_texts.load(path), table)
    atomic_dump(asr_texts.dump(merged), path)


def _has_phase1_windows(sid_dir: str) -> bool:

    sid = os.path.basename(os.path.normpath(sid_dir))
    try:
        with open(os.path.join(sid_dir, f"{sid}.dialogue_chunk.json")) as f:
            wins = json.load(f)
    except (OSError, ValueError):
        return False
    return any(dialogue_phase2.is_phase1_materialized(w) for w in wins)


def _finalize_dialogue_only(asr: ASRModel, lf: "_LoadedFile", ratio_filter, logger) -> None:

    try:
        with open(lf.final) as f:
            final_rows = json.load(f)
    except (OSError, ValueError):
        final_rows = []
    _finalize_dialogue_windows(asr, lf, final_rows, ratio_filter, logger,
                               rescue=_DIALOGUE_RESCUE)


def _finalize_dialogue_windows(
    asr: ASRModel, lf: "_LoadedFile", final_rows: list[dict], ratio_filter, logger,
    rescue: bool = False,
) -> None:

    if lf.secondary:
        return
    src, dst = _dialogue_paths(lf)


    try:
        with open(src) as f:
            wins = json.load(f)
    except (OSError, ValueError):
        return
    if not wins:
        atomic_dump([], dst)
        return
    if not any(dialogue_phase2.is_phase1_materialized(w) for w in wins):
        return

    text_map = {r["index"]: r["text"] for r in final_rows
                if r.get("index") and r.get("text")}
    part_idx = {r.get("index") for r in lf.rows}
    gates = _DIALOGUE_GATES

    sid = os.path.basename(lf.partial)[: -len(".partial.json")]
    table = asr_texts.load(asr_texts.path_for(lf.sid_dir, sid))

    kept: list[dict] = []
    n_from_table = 0
    todo: list[tuple[dict, list[dict]]] = []     # (window, units this pass transcribes)
    for w in wins:
        if not dialogue_phase2.is_phase1_materialized(w):
            kept.append(w)
            continue


        dialogue_phase2.prepare(w, text_map, part_idx, gates, final_pass=False)
        if table:
            n_from_table += dialogue_phase2.fill_from_table(
                w, table, min_char=DIALOGUE_MIN_CHAR, ratio_filter=ratio_filter)

        units = dialogue_phase2.pending_units(w)
        if units:
            todo.append((w, units))
        kept.append(w)

    if todo:
        _transcribe_dialogue_units(asr, lf, todo, ratio_filter, logger)

    if rescue:


        if table:
            for w in kept:
                if dialogue_phase2.is_phase1_materialized(w):
                    n_from_table += dialogue_phase2.rescue_from_table(w, table)
        todo_rescue = [(w, t) for w in kept
                       if dialogue_phase2.is_phase1_materialized(w)
                       for t in [dialogue_phase2.rescue_targets(w)] if t]
        if todo_rescue:
            _rescue_dialogue_units(asr, lf, todo_rescue, logger)

    out: list[dict] = []
    for w in kept:
        if not dialogue_phase2.is_phase1_materialized(w):
            out.append(w)
            continue


        for c in dialogue_phase2.split_window(w, gates):
            dialogue_phase2.finalize(c)
            out.append(c)
    if n_from_table:
        logger.info(f"dialogue: {n_from_table} units from asr_texts table (no ASR)")
    atomic_dump(out, dst)


def _transcribe_dialogue_units(
    asr: ASRModel, lf: "_LoadedFile", todo: list, ratio_filter, logger,
) -> None:

    target_sr = int(getattr(asr, "SR", 0) or 0) or None
    win_rows = [{"sample_rate": w["sample_rate"], "carrier_path": w["carrier_path"],
                 "carrier_start_samples": w["carrier_start_samples"],
                 "carrier_end_samples": w["carrier_end_samples"]} for w, _ in todo]
    loaded = _read_carrier_windows(win_rows, lf.sid_dir, target_sr=target_sr)

    n_ok = n_unit = 0
    for (w, units), (wav, sr) in zip(todo, loaded):
        bundle = AudioBundle(
            waveform=wav.astype(np.float32, copy=False), sample_rate=sr,
            name=w["index"])
        w0 = w["start"]


        rel = [Segment(start=u["start"] - w0, end=u["end"] - w0,
                       speaker=u["speaker"], index="|".join(u["seg_indices"]),
                       language=None, extra={}) for u in units]
        checked = _checked_transcribe(
            asr, bundle, rel, language=None, batch_size=len(rel),
            context=f"dialogue {w['index']}",
        )
        res = {r.index: r for r in checked}
        for u in units:
            n_unit += 1
            if dialogue_phase2.apply_result(
                    u, res.get("|".join(u["seg_indices"])), asr,
                    DIALOGUE_MIN_CHAR, ratio_filter):
                n_ok += 1
    if n_unit:
        logger.info(f"dialogue units: {n_ok}/{n_unit} ok "
                    f"({len(todo)} windows, 1 carrier decode)")


def _rescue_dialogue_units(
    asr: ASRModel, lf: "_LoadedFile", rescue: list, logger,
) -> None:

    target_sr = int(getattr(asr, "SR", 0) or 0) or None
    win_rows = [{"sample_rate": w["sample_rate"], "carrier_path": w["carrier_path"],
                 "carrier_start_samples": w["carrier_start_samples"],
                 "carrier_end_samples": w["carrier_end_samples"]} for w, _ in rescue]
    loaded = _read_carrier_windows(win_rows, lf.sid_dir, target_sr=target_sr)

    n_ok = n_try = 0
    for (w, units), (wav, sr) in zip(rescue, loaded):
        bundle = AudioBundle(
            waveform=wav.astype(np.float32, copy=False), sample_rate=sr,
            name=w["index"])
        w0 = w["start"]
        rel = [Segment(start=max(0.0, u["start"] - w0), end=u["end"] - w0,
                       speaker=u.get("speaker"),
                       index="|".join(u.get("seg_indices") or [str(i)]),
                       language=None, extra={})
               for i, u in enumerate(units)]
        checked = _checked_transcribe(
            asr, bundle, rel, language=None, batch_size=len(rel),
            context=f"dialogue rescue {w['index']}",
        )
        res = {r.index: r for r in checked}
        for u, s in zip(units, rel):
            n_try += 1
            if dialogue_phase2.apply_rescue(u, res.get(s.index)):
                n_ok += 1
    if n_try:
        logger.info(f"dialogue rescue: recovered {n_ok}/{n_try} "
                    f"({len(rescue)} windows)")


def _finalize_file(
    asr: ASRModel, lf: "_LoadedFile", min_char: int, ratio_filter, filter_cfg, logger,
) -> None:
    """CPU/IO half: stitch ``lf.transcribed`` back, apply the 4 post-ASR
    filters, merge (secondary), atomic-write ``<sid>.json``, optional prune."""
    segments, rows = lf.segments, lf.rows
    zh_en_dnsmos_floor = float(
        filter_cfg.min_quality_overrides.get(
            "zh", filter_cfg.min_quality_overrides.get(
                "en", filter_cfg.min_quality,
            ),
        )
    )
    drop_fake_zh_en = bool(filter_cfg.drop_fake_zh_en)
    dialect_dnsmos_floor = float(filter_cfg.dialect_dnsmos_floor)

    final_rows: list[dict] = []
    kept = drop_by_lang = drop_by_text = drop_by_ratio = 0
    drop_by_post_asr_zh_en = drop_by_content = 0
    for i, (orig_seg, orig_row) in enumerate(zip(segments, rows)):


        if not orig_row.get("kept", True):
            continue
        t = lf.transcribed.get(i)
        if t is None:
            drop_by_text += 1
            continue


        detected = t.language or orig_seg.language
        text = (t.text or "").strip()
        st = asr_texts.SegText(
            text=text, language=detected,
            accepted=bool(asr.accepts(detected)),
            asr_model=_asr_model_name(asr), extra=dict(t.extra or {}),
        )
        seg_dnsmos = orig_seg.quality
        if seg_dnsmos is None:
            seg_dnsmos = orig_row.get("dnsmos")
        reason = asr_texts.short_verdict(
            st, seg_dnsmos,
            orig_seg.language,
            orig_seg.extra.get("lid_raw") or orig_row.get("lid_raw"),
            bool(orig_seg.extra.get("is_fake")) or bool(orig_row.get("is_fake")),
            float(orig_seg.extra.get("speech_s") or (orig_seg.end - orig_seg.start)),
            min_char=min_char, ratio_filter=ratio_filter,
            zh_en_floor=zh_en_dnsmos_floor, dialect_floor=dialect_dnsmos_floor,
            drop_fake_zh_en=drop_fake_zh_en, char_count=get_char_count,
            lid_is_dialect=_lid_is_dialect,
        )
        if reason is not None:
            if reason == "lang":
                drop_by_lang += 1
            elif reason in ("text", "no_result"):
                drop_by_text += 1
            elif reason == "post_asr_zh_en":
                drop_by_post_asr_zh_en += 1
            elif reason == "ratio":
                drop_by_ratio += 1
            else:
                drop_by_content += 1
            continue

        orig_seg.text = text
        orig_seg.language = detected
        _asr_raw = t.extra.get("asr_lang_raw")
        if _asr_raw:
            orig_seg.extra["asr_lang_raw"] = _asr_raw

        out_row = orig_seg.to_legacy_dict()
        for k, v in orig_row.items():
            if k not in out_row:
                out_row[k] = v
        final_rows.append(out_row)
        kept += 1

    logger.info(
        f"{os.path.basename(lf.final)}{' [secondary]' if lf.secondary else ''}: "
        f"kept={kept} drop_lang={drop_by_lang} drop_text={drop_by_text} "
        f"drop_ratio={drop_by_ratio} drop_post_asr_zh_en={drop_by_post_asr_zh_en} "
        f"drop_content={drop_by_content} in={len(segments)}"
    )


    own_rows = final_rows
    if lf.secondary:
        merged = lf.existing_rows + final_rows
        merged.sort(key=lambda r: r.get("start", 0))
        final_rows = merged


    _dump_asr_texts(asr, lf, segments, rows)


    _finalize_dialogue_windows(asr, lf, final_rows, ratio_filter, logger,
                               rescue=_DIALOGUE_RESCUE)

    atomic_dump(final_rows, lf.final)


    _append_shorts_ledger(lf.sid_dir, os.path.basename(lf.final)[:-len(".json")],
                          own_rows, lf.secondary)

    # Delete loose audio of segments THIS pass DROPPED (not in final_rows) so
    # ASR-empty segments never accumulate as orphans.
    # SAFE: phase2 primary/secondary process DISJOINT phase1-LID partitions
    # (primary = ""/unknown/qwen3-accepted; secondary = indic-accepted), so a
    # segment one pass drops is never wanted by the other. Kept segments (in
    # final_rows — reuse anchors + packing sources) are preserved. Phase3 never
    # needs a dropped seg's loose audio: 3a reads all_shorts metadata, 3b anchors
    # ONLY on reuse (=phase2-kept) segments, 3c uses the window flac. Source tars
    # are read-only → reversible via re-run. Match by index (unambiguous) and
    # refuse to unlink anything outside the sid dir.
    kept_idx = {r.get("index") for r in final_rows}
    sid_dir_abs = os.path.abspath(lf.sid_dir)
    for r in rows:
        if r.get("index") in kept_idx:
            continue
        ap = _own_audio(r)
        if not ap:
            continue
        p = os.path.abspath(_resolve_path(ap, lf.sid_dir))
        if os.path.dirname(p) != sid_dir_abs:   # never escape the sid dir
            continue
        try:
            os.remove(p)
        except OSError:
            pass

    if lf.prune:
        for r in rows:
            ap = _own_audio(r)
            if not ap:
                continue
            p = _resolve_path(ap, lf.sid_dir)
            try:
                os.remove(p)
            except OSError:
                pass


def _touch(path: str) -> None:
    """Best-effort mtime bump (lease heartbeat). Silent on a vanished lock."""
    try:
        os.utime(path, None)
    except OSError:
        pass


_P2_MAX_FAILS = 3
_P2_FAILS: dict[str, int] = {}


def _poisoned(src: str) -> bool:
    return _P2_FAILS.get(src, 0) >= _P2_MAX_FAILS


def _poisoned_all() -> list[str]:
    return sorted(p for p, n in _P2_FAILS.items() if n >= _P2_MAX_FAILS)


def _root_poison(root: str) -> list[str]:
    pref = os.path.abspath(root).rstrip(os.sep) + os.sep
    return [p for p in _poisoned_all() if os.path.abspath(p).startswith(pref)]


def _record_phase2_fail(
    failed_log_path: str, src: str, e: Exception, key: str = "partial",
) -> None:
    """Append one failure record. ``key`` is the field name for ``src``
    (``partial`` for the short track, ``long_chunks`` for the long track)."""
    with open(failed_log_path, "a") as f:
        f.write(json.dumps({
            "ts": time.time(), key: src,
            "error_type": type(e).__name__, "error_msg": str(e),
        }, ensure_ascii=False) + "\n")
    n = _P2_FAILS.get(src, 0) + 1
    _P2_FAILS[src] = n
    if n == _P2_MAX_FAILS:
        Logger.get_logger().warning(
            f"{src}: failed {n} times; no more retries in this process "
            f"(not finalized; queue entry remains for restart). Details: "
            f"{os.path.basename(failed_log_path)}"
        )


def _iter_prefetched(work, load_fn, depth):
    """Yield ``(item, future)`` in completion order, ≤ ``depth`` loads in flight.

    A small thread pool runs ``load_fn(item)`` (the CPU/IO ``_load_file_for_asr``)
    ahead of the main thread's GPU inference, overlapping NAS reads with the ASR
    call. ``depth`` bounds both the pool size and the number of in-flight loaded
    buffers, so memory stays bounded regardless of the work-list length.
    """
    work_it = iter(work)
    with ThreadPoolExecutor(max_workers=depth, thread_name_prefix="p2load") as ex:
        inflight: dict = {}

        def _submit_next() -> bool:
            try:
                item = next(work_it)
            except StopIteration:
                return False
            inflight[ex.submit(load_fn, item)] = item
            return True

        for _ in range(depth):
            if not _submit_next():
                break
        while inflight:
            done, _ = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                item = inflight.pop(fut)
                _submit_next()  # keep the pipeline full
                yield item, fut


# Internal pacing for _run_short_track (not caller knobs): files between
# lease-lock heartbeats, and between gc.collect()+cuda.empty_cache() sweeps.
_HEARTBEAT_EVERY = 20
_CLEANUP_EVERY = 32


def _run_short_track(
    work, asr: ASRModel, *, batch_size: int, min_char: int, ratio_filter,
    filter_cfg, prune_segments: bool, secondary: bool, logger,
    failed_log_path: str, heartbeat=None, desc: str = "tx",
    prefetch: int = 4, batch_files: int = 1,
) -> None:
    """Drive the short track over a prefix's ``work`` list.

    Stage 1 (``prefetch``): file I/O (``_load_file_for_asr``) runs on a small
    thread pool ahead of the GPU ASR call so NAS reads of the next file overlap
    inference. Stage 2 (``batch_files`` > 1): up to ``batch_files`` loaded files
    share ONE batched ASR call (``_infer_batch``) so the engine stays fed across
    file boundaries; ``batch_files == 1`` infers one file at a time
    (``_infer_one``). Both produce per-file output bit-identical to the
    sequential path.

    Resume stays per-file: every file is atomic-written on its own, so a crash
    mid-batch just redoes that batch's files next run. Failure isolation: a load
    error, a whole-batch infer error (falls back to per-file infer), or a
    finalize error are all logged to ``_failed_phase2.jsonl`` without killing the
    loop. CUDA cache is cleared every ``_CLEANUP_EVERY`` files, not per-file.
    """
    work = list(work)
    if not work:
        return
    batch_files = max(1, batch_files)
    depth = max(1, prefetch, batch_files)  # keep enough in flight to fill a batch

    def load_fn(item):
        return _load_file_for_asr(item[0], item[1], asr, secondary)

    pending: list = []  # loaded files awaiting inference (flushed as a batch)

    def _flush() -> None:
        if not pending:
            return
        successful: list = []
        if len(pending) == 1:
            lf = pending[0]
            try:
                _infer_one(asr, lf, batch_size, logger)
                successful.append(lf)
            except Exception as e:  # noqa: BLE001 - leave final absent for retry
                logger.exception(f"per-file infer failed {lf.partial}: {e}")
                _record_phase2_fail(failed_log_path, lf.partial, e)
        else:
            try:
                _infer_batch(asr, pending, batch_size, logger)
                successful.extend(pending)
            except Exception as e:  # noqa: BLE001 - batch -> isolated retries
                logger.exception(
                    f"batched infer failed ({len(pending)} files); per-file retry: {e}"
                )
                for lf in pending:
                    lf.transcribed.clear()
                    try:
                        _infer_one(asr, lf, batch_size, logger)
                        successful.append(lf)
                    except Exception as e2:  # noqa: BLE001
                        logger.exception(f"per-file infer failed {lf.partial}: {e2}")
                        _record_phase2_fail(failed_log_path, lf.partial, e2)
        for lf in successful:
            try:
                _finalize_file(asr, lf, min_char, ratio_filter, filter_cfg, logger)
            except Exception as e:  # noqa: BLE001 - keep the rest of the batch
                logger.exception(f"finalize failed on {lf.partial}: {e}")
                _record_phase2_fail(failed_log_path, lf.partial, e)
        pending.clear()

    n = 0
    pbar = tqdm.tqdm(total=len(work), desc=desc)
    try:
        for (partial, _final), fut in _iter_prefetched(work, load_fn, depth):
            try:
                lf = fut.result()
                if lf is None:
                    pass
                elif lf.dialogue_only:

                    try:
                        _finalize_dialogue_only(asr, lf, ratio_filter, logger)
                    except Exception as e:  # noqa: BLE001
                        logger.exception(f"dialogue-only finalize failed {lf.partial}: {e}")
                        _record_phase2_fail(failed_log_path, lf.partial, e)
                elif lf.write_empty:
                    lf.prune = prune_segments

                    if _has_phase1_windows(lf.sid_dir):
                        _finalize_dialogue_windows(
                            asr, lf, [], ratio_filter, logger,
                        )
                    # Resume marker is last; dialogue failure remains retryable.
                    atomic_dump([], lf.final)
                else:
                    lf.prune = prune_segments
                    pending.append(lf)
                    if len(pending) >= batch_files:
                        _flush()
            except Exception as e:  # noqa: BLE001 - load failure for this file
                logger.exception(f"transcribe failed on {partial}: {e}")
                _record_phase2_fail(failed_log_path, partial, e)
            n += 1
            pbar.update(1)
            if n % _CLEANUP_EVERY == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if heartbeat and n % _HEARTBEAT_EVERY == 0:
                heartbeat()
        _flush()  # tail batch (< batch_files)
    finally:
        pbar.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _transcribe_one_file(
    partial: str,
    final: str,
    asr: ASRModel,
    batch_size: int,
    min_char: int,
    ratio_filter,
    filter_cfg,
    prune_segments: bool,
    secondary: bool,
    logger,
) -> None:
    """Phase-2 work for a single ``<sid>.partial.json``.

    In primary mode (``secondary=False``) every row in partial.json is
    fed to the ASR. In secondary mode rows whose index already appears
    in an existing ``<sid>.json`` are skipped, rows the ASR doesn't
    ``accept`` are skipped, and the newly-transcribed rows are merged
    into the existing ``<sid>.json`` (re-sorted by start time).
    """
    sid_dir = os.path.dirname(partial)
    with open(partial) as f:
        rows = json.load(f)

    if not rows:
        # Phase 1 produced no surviving segments — write the empty final
        # JSON so resume marks it done.
        if not (secondary and os.path.exists(final)):
            atomic_dump([], final)
        return

    # ---- LID-based routing (S4) ----
    # Route each row by its Phase-1 LID, so the two ASRs split the work
    # instead of qwen3 force-transcribing everything:
    #   * secondary (indic): rows whose LID the indic adapter accepts and that
    #     aren't already in <sid>.json. (indic's skip_languages excludes the
    #     qwen3∩indic overlap, e.g. hi → handled by primary.)
    #   * primary (qwen3): rows whose LID qwen3 handles, OR unknown/empty
    #     (auto-detect gets a shot). indic-only-LID rows are left for the
    #     secondary pass, which merges into this same <sid>.json.
    existing_rows: list[dict] = []
    if secondary:
        if os.path.exists(final):
            with open(final) as f:
                existing_rows = json.load(f)
            covered = {r.get("index") for r in existing_rows if r.get("index")}
            rows = [r for r in rows if r.get("index") not in covered]
        rows = [r for r in rows if asr.accepts((r.get("language") or "").strip().lower())]
        if not rows:
            return
    else:
        rows = [r for r in rows
                if (lid := (r.get("language") or "").strip().lower()) in ("", "unknown")
                or asr.accepts(lid)]
        if not rows:
            # Nothing for qwen3 here (all indic-only LID). Write an empty final
            # so resume marks the primary short-track pass done; the secondary
            # (indic) pass will merge its rows into this file afterwards.
            if not os.path.exists(final):
                atomic_dump([], final)
            return

    segments = [_segment_from_meta(r) for r in rows]

    # Load per-segment audio. All clips share the same sample rate, so we
    # build one ``AudioBundle`` covering the concatenated buffer and pass
    # segments with start/end in seconds within that buffer. Each segment
    # gets its own slice; the ASR adapter just sees a normal batched call.
    clips: list[np.ndarray] = []
    sample_rate = 0
    cursor = 0.0
    rel_segments: list[Segment] = []
    loaded = _read_segments_parallel(
        rows, sid_dir, target_sr=int(getattr(asr, "SR", 0) or 0) or None,
    )
    for seg, (wav, sr) in zip(segments, loaded):
        if sample_rate == 0:
            sample_rate = sr
        elif sr != sample_rate:
            raise RuntimeError(
                f"mixed sample rates in {partial}: {sr} vs {sample_rate}"
            )
        dur = len(wav) / sr
        clips.append(wav)
        # Build a Segment relative to the concatenated buffer Phase 2
        # constructs; keep the *original* time bounds (and everything
        # else) on ``seg`` so the final JSON preserves them.
        rel = Segment(
            start=cursor, end=cursor + dur,
            speaker=seg.speaker, index=seg.index,
            language=seg.language if seg.language and seg.language != "unknown" else None,
            extra=dict(seg.extra),
        )
        rel_segments.append(rel)
        cursor += dur

    big_buffer = np.concatenate(clips).astype(np.float32, copy=False)
    bundle = AudioBundle(
        waveform=big_buffer, sample_rate=sample_rate, name="phase2",
    )

    # Multilingual fan-out: group rel_segments by the language hint Phase 1
    # / LID provided. Segments with no hint (LID was "unknown") go to a
    # special bucket the adapter auto-detects.
    buckets: dict[str | None, list[int]] = defaultdict(list)
    for i, rs in enumerate(rel_segments):
        buckets[rs.language].append(i)

    transcribed_by_index: dict[int, Segment] = {}
    for lang, idxs in buckets.items():
        subs = [rel_segments[i] for i in idxs]
        try:
            out = asr.transcribe(
                bundle, subs, language=lang, batch_size=batch_size,
            )
        except Exception as e:
            logger.exception(
                f"ASR transcribe failed for lang={lang!r} ({len(subs)} segs) "
                f"on {partial}: {e}"
            )
            continue
        for i, t in zip(idxs, out):
            transcribed_by_index[i] = t

    # Stitch results back, applying Phase-2 filter rules.
    # ``zh_en_dnsmos_floor`` enforces the user-spec rule: a short whose
    # LID was not zh/en but whose ASR comes back zh/en must clear the
    # stricter zh/en DNSMOS floor (and not be flagged FAKE by the
    # Phase-1 deepfake stage). Phase 1 already enforced this for shorts
    # LID was confident about; here we catch the ones that flipped
    # language via ASR.
    zh_en_dnsmos_floor = float(
        filter_cfg.min_quality_overrides.get(
            "zh", filter_cfg.min_quality_overrides.get(
                "en", filter_cfg.min_quality,
            ),
        )
    )
    drop_fake_zh_en = bool(filter_cfg.drop_fake_zh_en)
    # Dialects/accents the ASR flips to zh (Sichuanese, Wu, Min, ...) are
    # scarce data — give them a looser floor than mainstream zh/en.
    dialect_dnsmos_floor = float(filter_cfg.dialect_dnsmos_floor)

    final_rows: list[dict] = []
    kept = 0
    drop_by_lang = 0
    drop_by_text = 0
    drop_by_ratio = 0
    drop_by_post_asr_zh_en = 0
    drop_by_content = 0
    for i, (orig_seg, orig_row) in enumerate(zip(segments, rows)):
        t = transcribed_by_index.get(i)
        if t is None:
            # ASR didn't return anything (failure or filter inside adapter).
            drop_by_text += 1
            continue
        detected = t.language or orig_seg.language
        if not asr.accepts(detected):
            drop_by_lang += 1
            continue
        text = (t.text or "").strip()
        if get_char_count(text) < min_char:
            drop_by_text += 1
            continue

        content_reason = asr_texts.content_verdict(
            asr_texts.SegText(
                text=text, language=detected, accepted=True,
                asr_model=_asr_model_name(asr), extra=dict(t.extra or {}),
            ),
            float(orig_seg.extra.get("speech_s") or (orig_seg.end - orig_seg.start)),
        )
        if content_reason is not None:
            drop_by_content += 1
            continue

        # Post-ASR re-filter: ASR may flip a "unknown" / non-zh-en LID
        # short to zh/en. Phase 1's QualityFilter only applied the
        # stricter zh/en gates (DNSMOS ≥ 3.0 + not is_fake) when LID was
        # already zh/en. Re-check here so flipped shorts don't sneak
        # through with a 2.4 floor.
        lid_language = (orig_seg.language or "").strip().lower()
        if detected in ("zh", "en") and lid_language not in ("zh", "en"):
            seg_dnsmos = orig_seg.quality
            if seg_dnsmos is None:
                seg_dnsmos = orig_row.get("dnsmos")
            is_fake_phase1 = bool(orig_seg.extra.get("is_fake")) or bool(
                orig_row.get("is_fake")
            )
            # Scarce dialect data (ASR flips it to zh: Sichuanese, Wu, ...)
            # gets the looser dialect floor instead of the strict zh/en one.
            # The looser dialect floor applies ONLY when the ASR flagged a
            # Sinitic dialect AND Phase-1 LID independently pointed at a
            # dialect ("zh <subtag>", excluding standard Mandarin). Either
            # signal alone — e.g. an ASR dialect label on a clip whose LID was
            # "unknown"/foreign — keeps the strict zh/en floor.
            lid_raw = orig_seg.extra.get("lid_raw") or orig_row.get("lid_raw")
            floor = (
                dialect_dnsmos_floor
                if (t.extra.get("is_dialect") and _lid_is_dialect(lid_raw))
                else zh_en_dnsmos_floor
            )
            if (
                seg_dnsmos is None
                or float(seg_dnsmos) < floor
                or (drop_fake_zh_en and is_fake_phase1)
            ):
                drop_by_post_asr_zh_en += 1
                continue

        # Set the resolved language up front so the per-language ratio
        # lookup uses the ASR's final call, not the LID hint.
        orig_seg.text = text
        orig_seg.language = detected
        # Carry the ASR's verbatim language label (dialect / accent /
        # code-switch string) onto the original segment so it survives into
        # the final JSON next to the collapsed ``language`` code.
        _asr_raw = t.extra.get("asr_lang_raw")
        if _asr_raw:
            orig_seg.extra["asr_lang_raw"] = _asr_raw
        if not _ratio_ok(orig_seg, text, ratio_filter):
            drop_by_ratio += 1
            continue

        # Write back onto the *original* segment so we keep original time
        # bounds, lid_raw, lid_confidence, audio_path, etc.
        out_row = orig_seg.to_legacy_dict()
        # Preserve everything we wrote into the partial that isn't part of
        # the canonical legacy dict (e.g. ``audio_path``, ``sample_rate``).
        for k, v in orig_row.items():
            if k not in out_row:
                out_row[k] = v
        final_rows.append(out_row)
        kept += 1

    logger.info(
        f"{os.path.basename(final)}{' [secondary]' if secondary else ''}: "
        f"kept={kept} drop_lang={drop_by_lang} drop_text={drop_by_text} "
        f"drop_ratio={drop_by_ratio} drop_post_asr_zh_en={drop_by_post_asr_zh_en} "
        f"drop_content={drop_by_content} in={len(segments)}"
    )

    # Secondary mode: merge the newly-transcribed rows into the
    # existing ``<sid>.json`` and re-sort. Primary mode overwrites.
    if secondary:
        merged = existing_rows + final_rows
        merged.sort(key=lambda r: r.get("start", 0))
        final_rows = merged

    atomic_dump(final_rows, final)

    if prune_segments:
        for r in rows:
            ap = _own_audio(r)
            if not ap:
                continue
            p = _resolve_path(ap, sid_dir)
            try:
                os.remove(p)
            except OSError:
                pass


# ====================================================================
# Long-track transcription
# ====================================================================
# Reads ``<sid>.long_chunks.json`` (Phase 1 kept-longs, with member
# detail) and produces ``<sid>.long.json`` containing only longs whose
# text was successfully recovered. Per :func:`_process_one_long` the
# strategy is, in order:
#   1. per-member  — slice each member from the long's WAV, ASR each
#   2. whole-chunk — one ASR call on the whole long (short chunks only).
#      Needs vLLM's ``max_num_batched_tokens`` bumped to at least 32 768
#      so a ~20-min single call fits in one scheduling round
#   3. split — cut at the failed member(s), keep >=30s runs as _p<k>
#   4. drop — if everything failed, the long isn't emitted
# We no longer split longs into smaller longs on irrecoverable failures.
# The configured Phase-1 long-chunk duration and quality gates define what a
# long is; a partial reconstruction
# wouldn't re-pass them cleanly.


def _long_audio_bundle(
    long_meta: dict, sid_dir: str, target_sr: int | None = None,
) -> "AudioBundle":
    """Load one legacy long clip or one window from a shared carrier."""
    if long_meta.get("carrier_path"):
        wav, sr = _read_carrier_windows(
            [long_meta], sid_dir, target_sr=target_sr,
        )[0]
        return AudioBundle(waveform=wav, sample_rate=sr, name="long")
    path = long_meta.get("audio_path")
    if not path:
        raise RuntimeError(
            f"long-chunk {long_meta.get('index')} has neither audio_path "
            f"nor carrier_path; Phase 1 export is inconsistent"
        )
    wav, sr = sf.read(
        _resolve_path(path, sid_dir),
        dtype="float32", always_2d=False,
    )
    if wav.ndim == 2:
        wav = wav.mean(axis=1).astype(np.float32)
    return AudioBundle(waveform=wav, sample_rate=int(sr), name="long")


def _rel_member_segments(
    long_start: float, members: list[dict],
) -> list[Segment]:
    """Build Segments relative to the long-chunk WAV (start=0 = long.start)."""
    out = []
    for m in members:
        out.append(Segment(
            start=m["start"] - long_start,
            end=m["end"] - long_start,
            speaker=m.get("speaker"),
            index=m.get("index"),
            quality=m.get("dnsmos"),
            language=m.get("language"),
            extra={"is_fake": m.get("is_fake")},
        ))
    return out


def _safe_transcribe(
    asr: ASRModel, bundle: "AudioBundle",
    rel_segs: list[Segment], logger, ratio_filter=None,
) -> list[tuple[str, str] | None]:
    """Per-segment ``(text, detected_language)``; None for any segment whose
    ASR call returned empty / no language / wasn't accepted by the adapter's
    language gate.

    ``detected_language`` is the ASR's OWN call (base ISO code), not the
    phase1 LID. The long track keys language off this so a chunk reflects
    what was actually transcribed — LID may say 'unknown' while the ASR
    knows it's en/zh. Batch-level exception → all None.
    """
    if not rel_segs:
        return []
    results = _checked_transcribe(
        asr, bundle, rel_segs, language=None, batch_size=len(rel_segs),
        context="long-track",
    )
    out: list[tuple[str, str] | None] = []
    for seg, r in zip(rel_segs, results):
        text = (r.text or "").strip() if r is not None else ""
        st = asr_texts.SegText(
            text=text, language=(r.language if r is not None else None),
            accepted=bool(r is not None and asr.accepts(r.language)),
            asr_model=_asr_model_name(asr),
            extra=dict((r.extra if r is not None else None) or {}),
        )
        reason = asr_texts.chunk_verdict(
            st, float(seg.extra.get("speech_s") or (seg.end - seg.start)),
            min_char=DIALOGUE_MIN_CHAR, ratio_filter=ratio_filter,
            char_count=get_char_count,
        )
        if reason is not None:
            out.append(None)
            continue
        out.append((text, r.language))
    return out


def _emit_long_row(
    long_meta: dict, sub_start: float, sub_end: float,
    members_with_tl: list,  # list[(member_meta, (text, detected_lang) | None)]
    audio_path: str | None,
    source: str,
    top_text: str | None,
    top_lang: str | None = None,
) -> dict:
    """Shape a long-track JSON row.

    ``members_with_tl`` is the per-member breakdown: one
    ``(member_meta, (text, detected_lang) | None)`` per member (None =
    failed / covered by an adjacent concat-patch). ``top_text`` is the
    canonical chunk text; ``top_lang`` is the detected language for the
    whole-chunk fallback (which has no per-member breakdown).

    Language uses the ASR's OWN detection, not the phase1 LID (LID may be
    'unknown' while the ASR knows the language). And the chunk can carry
    MULTIPLE languages — long audio often code-switches (zh+en). So:
      * ``language``  = comma-joined unique detected langs ("zh" or "zh,en")
      * ``languages`` = the list form, for clean downstream filtering
    Member ``language`` is the ASR detection for that member (falls back to
    the phase1 LID only when the ASR produced nothing for it).
    """
    row: dict = {
        "start": sub_start,
        "end": sub_end,
        "speaker": long_meta.get("speaker"),
        "index": long_meta.get("index"),
        "mean_dnsmos": long_meta.get("mean_dnsmos"),
        "text": top_text,
        "source": source,
    }
    detected: list[str] = []
    if members_with_tl:
        row["members"] = []
        for m, tl in members_with_tl:
            text = tl[0] if tl else None
            # ASR-detected lang for this member; fall back to phase1 LID
            # only when the ASR gave nothing (failed / concat-covered).
            lang = tl[1] if tl else (m.get("language") or None)
            if lang and lang != "unknown":
                detected.append(lang)
            row["members"].append({
                "index": m.get("index"),
                "start": m["start"],
                "end": m["end"],
                "speaker": m.get("speaker"),
                "language": lang,
                "dnsmos": m.get("dnsmos"),
                "is_fake": m.get("is_fake"),
                "deepfake_score": m.get("deepfake_score"),
                "text": text,
            })
    elif top_lang and top_lang != "unknown":
        detected.append(top_lang)
    # Chunk language(s): unique ASR-detected langs, order preserved. Single
    # language → "zh"; code-switch → "zh,en". `languages` is the list form.
    uniq = list(dict.fromkeys(detected))
    if uniq:
        row["language"] = ",".join(uniq)
        row["languages"] = uniq
    if audio_path is not None:
        row["audio_path"] = audio_path
        row["sample_rate"] = long_meta.get("sample_rate")
    elif long_meta.get("carrier_path"):
        sr = int(long_meta["sample_rate"])
        parent_a = int(long_meta["carrier_start_samples"])
        parent_b = int(long_meta["carrier_end_samples"])
        if sub_start == long_meta.get("start") and sub_end == long_meta.get("end"):
            a, b = parent_a, parent_b
        else:
            a = max(parent_a, min(parent_b, int(float(sub_start) * sr)))
            b = max(a, min(parent_b, int(float(sub_end) * sr)))
        row.update({
            "sample_rate": sr,
            "carrier_path": long_meta["carrier_path"],
            "carrier_start_samples": a,
            "carrier_end_samples": b,
        })
    return row


def _split_long_into_runs(
    long_meta: dict, sid_dir: str, members: list, rel_segs: list,
    patched: list, bundle: "AudioBundle", logger,
    roles: list[str] | None = None, max_gap_s: float = 2.0,
) -> list[dict]:

    if roles is None:
        roles = [asr_texts.BREAK if t is None else asr_texts.OK for t in patched]

    runs = asr_texts.assemble_runs(
        members, roles, max_gap_s=max_gap_s,
        min_duration_s=_LONG_MIN_DUR_S, min_speakers=1)

    wav = bundle.waveform if bundle is not None else None
    sr = bundle.sample_rate if bundle is not None else (
        long_meta.get("sample_rate") or 44100)
    base, ext = os.path.splitext(
        os.path.basename(long_meta.get("audio_path") or "long.flac"))
    rows: list[dict] = []
    for k, run in enumerate(runs):
        i, j = run[0], run[-1]
        run_start, run_end = members[i]["start"], members[j]["end"]
        sub = None
        if not long_meta.get("carrier_path"):
            if wav is None:
                continue
            a = max(0, int(rel_segs[i].start * sr))
            b = int(rel_segs[j].end * sr)
            clip = wav[a:b]
            if not clip.size:
                continue
            sub = f"{base}_p{k}{ext}"
            sf.write(os.path.join(sid_dir, sub), clip, sr, subtype="PCM_16")


        text = " ".join(patched[x][0] for x in run
                        if roles[x] == asr_texts.OK and patched[x] and patched[x][0])
        rows.append(_emit_long_row(
            long_meta, run_start, run_end,
            members_with_tl=[(members[x],
                              patched[x] if roles[x] == asr_texts.OK else None)
                             for x in run],
            audio_path=sub, source="per_member_split", top_text=text,
        ))
    n_dropped = sum(1 for r in roles if r != asr_texts.OK)
    if rows or n_dropped:
        logger.debug(f"long-track split: {long_meta.get('index')} → {len(rows)} "
                     f"run(s), {n_dropped} non-ok member(s)")
    return rows


def _member_speech_s(m: dict) -> float:

    return float(m.get("speech_s") or (m.get("end", 0) - m.get("start", 0)))


def _member_roles_from_table(members: list[dict], table: dict,
                             ratio_filter=None) -> list[str] | None:

    out = []
    for m in members:


        if lid_unsupported(m.get("language")):
            out.append(asr_texts.BREAK)
            continue
        st = table.get(m.get("index"))
        if st is None:
            return None
        out.append(asr_texts.role_of(asr_texts.chunk_verdict(
            st, _member_speech_s(m), min_char=DIALOGUE_MIN_CHAR,
            ratio_filter=ratio_filter)))
    return out


def _member_texts_from_table(members: list[dict], table: dict,
                             ratio_filter=None) -> list | None:

    out = []
    for m in members:
        if lid_unsupported(m.get("language")):
            out.append(None)
            continue
        st = table.get(m.get("index"))
        if st is None:
            return None
        bad = asr_texts.chunk_verdict(
            st, _member_speech_s(m), min_char=DIALOGUE_MIN_CHAR,
            ratio_filter=ratio_filter)
        out.append(None if bad is not None else (st.text, st.language))
    return out


def _process_one_long(
    long_meta: dict, sid_dir: str, asr: ASRModel, logger,
    bundle: "AudioBundle | None" = None,
    table: dict | None = None,
    ratio_filter=None,
) -> list[dict]:
    """Process one Phase-1 long chunk → 0 or 1 kept long-track row.

    Strategy (per the latest spec):

      1. Per-member ASR walk. All members succeed → emit one row with
         ``source=per_member`` and per-member text.

      2. Some members fail (empty text / rejected language e.g. ms/id) →
         whole-chunk ASR as a fallback, short chunks only
         (``_WHOLE_CHUNK_MAX_S``; vLLM hangs on multi-minute one-shots).
         Success → ``source=whole_chunk_fallback``, single top-level
         text, no member breakdown.

      3. Otherwise split at the failed member(s) into runs of
         consecutive transcribed members (``source=per_member_split``,
         physical ``_p<k>`` sub-clips; failed members' audio excluded).

    """
    members = long_meta.get("members") or []
    if not members:
        return []


    long_start = long_meta["start"]
    long_end = long_meta["end"]


    rel_segs = _rel_member_segments(long_start, members)


    def _need_bundle():
        nonlocal bundle
        if bundle is None:
            bundle = _long_audio_bundle(
                long_meta, sid_dir, target_sr=int(getattr(asr, "SR", 0) or 0) or None,
            )
        return bundle


    roles = None
    results = _member_texts_from_table(members, table, ratio_filter) if table else None
    if results is not None:
        roles = _member_roles_from_table(members, table, ratio_filter)
    else:
        results = _safe_transcribe(
            asr, _need_bundle(), rel_segs, logger, ratio_filter=ratio_filter,
        )

    # ---- 1. per-member happy path ----
    if all(r is not None for r in results):
        top = " ".join(r[0] for r in results)  # type: ignore[index]
        return [_emit_long_row(
            long_meta, long_start, long_end,
            members_with_tl=list(zip(members, results)),
            audio_path=long_meta.get("audio_path"),
            source="per_member",
            top_text=top,
        )]

    patched: list[tuple[str, str] | None] = list(results)

    # ---- 3. whole-chunk fallback (SHORT longs only) ----
    # Qwen3-ASR/vLLM HANGS on a multi-minute single clip (long chunks can be
    # up to hours — a 90-min whole-chunk call wedged the engine for >1h). The
    # per-member path above is safe (it sends short member clips); only this
    # whole-chunk one-shot is dangerous, so cap it. Longer chunks whose
    # per-member already failed are dropped rather than risk hanging phase-2.
    whole_dur = long_end - long_start
    if whole_dur <= _WHOLE_CHUNK_MAX_S:
        whole_seg = Segment(start=0.0, end=whole_dur)
        whole = _safe_transcribe(
            asr, _need_bundle(), [whole_seg], logger, ratio_filter=ratio_filter,
        )
        if whole and whole[0]:
            wt, wl = whole[0]
            return [_emit_long_row(
                long_meta, long_start, long_end,
                members_with_tl=[],
                audio_path=long_meta.get("audio_path"),
                source="whole_chunk_fallback",
                top_text=wt, top_lang=wl,
            )]

    # ---- 4. split at the failed member(s), keep transcribed runs ----
    # (per-member couldn't be fully covered and whole-chunk wasn't applicable;
    # salvage the good stretches instead of dropping the whole long.)

    _b = None if long_meta.get("carrier_path") else _need_bundle()
    return _split_long_into_runs(
        long_meta, sid_dir, members, rel_segs, patched, _b, logger,
        roles=roles, max_gap_s=_LONG_MAX_GAP_S)


def _transcribe_long_chunks(
    src_path: str, dst_path: str, sid_dir: str, asr: ASRModel, logger,
    ratio_filter=None,
) -> None:
    """Phase-2 long-track work for a single ``<sid>.long_chunks.json``."""
    with open(src_path) as f:
        longs = json.load(f)
    if not longs:
        atomic_dump([], dst_path)
        return

    out: list[dict] = []

    _sid = os.path.basename(src_path)[: -len(".long_chunks.json")]
    table = asr_texts.load(asr_texts.path_for(sid_dir, _sid))
    need_audio = not table or not all(
        all(table.get(m.get("index")) is not None for m in (lm.get("members") or []))
        for lm in longs)
    if table and not need_audio:
        logger.info(f"long track: {len(longs)} chunks from asr_texts table "
                    f"(no ASR, no decode)")

    carrier_bundles = None
    if need_audio and all(lm.get("carrier_path") for lm in longs):
        loaded = _read_carrier_windows(
            longs, sid_dir,
            target_sr=int(getattr(asr, "SR", 0) or 0) or None,
        )
        carrier_bundles = [
            AudioBundle(waveform=wav, sample_rate=sr, name="long")
            for wav, sr in loaded
        ]
    for i, lm in enumerate(longs):
        bundle = carrier_bundles[i] if carrier_bundles is not None else None
        rows = _process_one_long(
            lm, sid_dir, asr, logger=logger, bundle=bundle, table=table,
            ratio_filter=ratio_filter,
        )
        out.extend(rows)

    atomic_dump(out, dst_path)

    n_pm = sum(1 for r in out if r.get("source") == "per_member")
    n_split = sum(1 for r in out if r.get("source") == "per_member_split")
    n_whole = sum(1 for r in out if r.get("source") == "whole_chunk_fallback")
    logger.info(
        f"{os.path.basename(dst_path)}: "
        f"kept={len(out)} (per_member={n_pm} "
        f"split={n_split} whole={n_whole}) in={len(longs)}"
    )


if __name__ == "__main__":
    main()
