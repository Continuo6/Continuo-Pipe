"""Read a clip straight out of a WebDataset-style tar, instead of a file on disk.

Cutting a corpus into one file per clip costs three things: hours of ffmpeg before any
labelling can start, a file per clip forever after, and a second lossy generation if
those files are MP3. For standalone utterances none of it buys anything — a
segment *is* a tar member, start to end — so the pipeline can read the member's bytes at
its offset and decode them in memory.

A manifest row therefore names its source instead of a path::

    {"id": "...", "source_tar": "continuo-00214.tar", "source_member": "x_spk1.m4a",
     "tar_offset": 123456, "tar_size": 10656, "rel_start": 0.0, "rel_end": 3.82}

``source_tar`` may be a bare name, resolved against ``--tar-dir`` / ``CONTINUO_EXPRESSIVE_TAR_DIR``, so
one manifest works on any machine that has the corpus. ``tar_offset``/``tar_size`` are
optional: without them the ``.tar.idx`` beside the tar is read once and cached — worth
baking into the manifest for a corpus-scale run, since that cache is per process and
per tar and never shrinks.

Open tar handles are cached per thread and capped at ``CONTINUO_EXPRESSIVE_TAR_HANDLES`` (8). Grouping a
manifest by tar makes almost every clip a cache hit.

**Rows that are slices of a container** (``rel_start``/``rel_end``, which is how long
recordings are annotated without cutting them to disk) would otherwise decode that whole
container once per segment. Decoded containers are cached too, capped at
``CONTINUO_EXPRESSIVE_DECODED_CONTAINERS`` (8), which is why a slice manifest should be grouped by tar and
ordered inside it.

**Decoding.** m4a needs an edit-list-aware decoder or every offset shifts by ~2112
samples. torchaudio's is one and runs in-process, which matters at this scale — a
subprocess per clip is a subprocess per clip per pass. ffmpeg remains the fallback.

Rows that carry ``wav_path`` still work and take the old path, so a cut corpus and an
uncut one can be mixed in the same run.
"""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from .config import TARGET_SR
from .jsonl import ManifestError

#: `member -> (offset, size)` per tar, filled from the .tar.idx on first use
_INDEX: dict[str, dict[str, tuple[int, int]]] = {}
_INDEX_LOCK = threading.Lock()
#: open tar handles, per thread — the loader runs on a pool and a file object is not
#: safe to seek from two threads at once. A handle retains buffered data, so the
#: cache must be bounded. A manifest grouped by tar reuses the
#: most recent handle almost every clip, so a small cache gives up nothing; a scattered
#: one pays an open() per miss, which is nothing beside the decode.
_HANDLES = threading.local()
_MAX_HANDLES = max(1, int(os.environ.get("CONTINUO_EXPRESSIVE_TAR_HANDLES", "8")))
#: decoded containers, for manifests whose rows are *slices* of one. A long recording
#: may hold several segments, and without this each one decodes the whole
#: container again. Rows are
#: grouped by tar and ordered inside it, so a handful of entries catches nearly every
#: repeat. Shared, not per-thread: the pool hands one container's segments to different
#: threads at the same time, so a thread-local copy would miss most of them.
#:
#: Only slices are cached — a whole-member row is used once and would just evict.
_DECODED: "OrderedDict[tuple[str, str, int], np.ndarray]" = OrderedDict()
_DECODED_LOCK = threading.Lock()
_MAX_DECODED = max(0, int(os.environ.get("CONTINUO_EXPRESSIVE_DECODED_CONTAINERS", "8")))
#: one lock per container being decoded. Without it the cache is nearly useless on a
#: pool: the loader is handed a chunk of rows at once, so every thread holding a segment
#: of the same container misses together, decodes the same minutes of audio, and stores
#: the same array. The first thread to miss decodes; the rest wait on it
#: and then hit.
_DECODED_LOADING: "dict[tuple[str, str, int], threading.Lock]" = {}


def tar_dir(explicit: str | os.PathLike | None = None) -> Path | None:
    raw = explicit or os.environ.get("CONTINUO_EXPRESSIVE_TAR_DIR")
    return Path(raw).expanduser().resolve() if raw else None


def resolve_tar(name: str, root: Path | None) -> Path:
    """A manifest's ``source_tar`` -> a real path, absolute names passed through."""
    p = Path(name).expanduser()
    if not p.is_absolute():
        if root is None:
            raise ManifestError(
                f"source_tar {name!r} is relative and no tar directory is set; "
                "pass --tar-dir or set CONTINUO_EXPRESSIVE_TAR_DIR")
        p = root / p.name
    if not p.is_file():
        raise ManifestError(f"tar not found: {p}")
    return p


def load_index(tar: Path) -> dict[str, tuple[int, int]]:
    """``member -> (offset, size)`` from ``<tar>.idx``, read once per process."""
    key = str(tar)
    with _INDEX_LOCK:
        cached = _INDEX.get(key)
    if cached is not None:
        return cached
    idx_path = Path(str(tar) + ".idx")
    if not idx_path.is_file():
        raise ManifestError(f"no index beside {tar}; expected {idx_path}")
    found: dict[str, tuple[int, int]] = {}
    with open(idx_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                found[parts[0]] = (int(parts[1]), int(parts[2]))
    with _INDEX_LOCK:
        _INDEX[key] = found
    return found


def _handle(tar: Path):
    """One open file per tar per thread, the ``_MAX_HANDLES`` most recent kept."""
    cache = getattr(_HANDLES, "files", None)
    if cache is None:
        cache = _HANDLES.files = OrderedDict()
    key = str(tar)
    f = cache.get(key)
    if f is None:
        f = cache[key] = open(tar, "rb")
        while len(cache) > _MAX_HANDLES:
            cache.popitem(last=False)[1].close()
    else:
        cache.move_to_end(key)
    return f


def read_member(tar: Path, offset: int, size: int) -> bytes:
    f = _handle(tar)
    f.seek(offset)
    blob = f.read(size)
    if len(blob) != size:
        raise ManifestError(f"{tar}:{offset}+{size}: read {len(blob)} bytes")
    return blob


def decode_bytes(blob: bytes, target_sr: int = TARGET_SR,
                 fmt: str = "mp4") -> np.ndarray:
    """Encoded audio in memory -> mono float32 at ``target_sr``."""
    try:
        import torchaudio
        import torchaudio.functional as AF
        wav, sr = torchaudio.load(io.BytesIO(blob), format=fmt)
        wav = wav.mean(0) if wav.shape[0] > 1 else wav[0]
        if sr != target_sr:
            wav = AF.resample(wav, sr, target_sr)
        return np.ascontiguousarray(wav.numpy(), dtype=np.float32)
    except Exception:
        # ffmpeg reads a temp file rather than stdin: an MP4's moov atom is not
        # guaranteed to precede the media data, and a non-seekable input then fails
        with tempfile.NamedTemporaryFile(suffix="." + fmt) as tmp:
            tmp.write(blob)
            tmp.flush()
            proc = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", tmp.name, "-f", "f32le",
                 "-ac", "1", "-ar", str(target_sr), "-"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            raise ManifestError(
                f"decode failed: {proc.stderr.decode('utf-8', 'replace').strip()[:200]}")
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def is_tar_row(row: dict[str, Any]) -> bool:
    """Read from the tar only when there is no local copy to read instead.

    A cut manifest carries ``source_tar``/``source_member`` as provenance alongside its
    ``wav_path``, so treating their presence as "read from the tar" would quietly change
    where a finished corpus is read from.
    """
    if row.get("_path") or row.get("wav_path"):
        return False
    return bool(row.get("source_tar") and row.get("source_member"))


def load_row(row: dict[str, Any], target_sr: int = TARGET_SR,
             root: Path | None = None) -> np.ndarray:
    """A manifest row -> its audio, from a tar member or from ``_path``."""
    if row.get("carrier_start_samples") is not None or row.get("carrier_end_samples") is not None:
        from .carrier_source import load_carrier_slice
        return load_carrier_slice(row, target_sr)
    if not is_tar_row(row):
        from .audio import load_wav
        return load_wav(row["_path"], target_sr)

    tar = row.get("_tar")
    if tar is None:
        # the decode runs on a worker thread with no argument threaded through to it,
        # so the corpus location comes from the environment when the caller did not pass
        # one; --tar-dir sets it
        tar = row["_tar"] = resolve_tar(row["source_tar"], root or tar_dir())
    member = row["source_member"]

    offset, size = row.get("tar_offset"), row.get("tar_size")
    if offset is None or size is None:
        entry = load_index(tar).get(member)
        if entry is None:
            raise ManifestError(f"{member} not in {tar}.idx")
        offset, size = entry

    start, end = row.get("rel_start"), row.get("rel_end")
    is_slice = start is not None or end is not None
    key = (str(tar), member, target_sr)

    def _decode() -> np.ndarray:
        suffix = Path(member).suffix.lstrip(".").lower() or "mp4"
        return decode_bytes(read_member(tar, offset, size), target_sr,
                            fmt="mp4" if suffix in ("m4a", "mp4") else suffix)

    wav = None
    from_cache = False
    if not (is_slice and _MAX_DECODED):
        wav = _decode()
    else:
        while wav is None:
            with _DECODED_LOCK:
                wav = _DECODED.get(key)
                if wav is not None:
                    _DECODED.move_to_end(key)
                    from_cache = True
                    break
                loader = _DECODED_LOADING.get(key)
                mine = loader is None
                if mine:
                    loader = _DECODED_LOADING[key] = threading.Lock()
                    loader.acquire()
            if not mine:
                # Someone else is decoding this container: wait for them and look again.
                # If it was evicted in between, the next turn of the loop makes us the
                # decoder — slower, never wrong.
                with loader:
                    pass
                continue
            try:
                wav = _decode()
                with _DECODED_LOCK:
                    _DECODED[key] = wav
                    while len(_DECODED) > _MAX_DECODED:
                        _DECODED.popitem(last=False)
            finally:
                with _DECODED_LOCK:
                    _DECODED_LOADING.pop(key, None)
                loader.release()

    # A standalone utterance is the whole member, so the common case slices nothing.
    if start or (end is not None and end < len(wav) / target_sr - 1e-6):
        a = max(0, min(int(round((start or 0.0) * target_sr)), len(wav)))
        b = len(wav) if end is None else max(a, min(int(round(end * target_sr)), len(wav)))
        if b <= a:
            # A window that begins at or after the end of the decoded audio. Returning
            # the empty array let it travel: PANNs padded a batch to its longest clip,
            # got a (n, 0) tensor and died inside the model, and the supervisor restarted
            # the worker onto the same row forever. Say so here, where every caller
            # already treats a raised error as "skip this clip and warn".
            raise ManifestError(
                f"{row.get('id', member)}: window {start}-{end}s lies outside the "
                f"{len(wav) / target_sr:.2f}s decoded from {member}")
        return np.ascontiguousarray(wav[a:b])
    # A segment that turns out to span its whole container would otherwise hand the
    # caller the cached array itself, and one in-place write would corrupt every later
    # read of it. Slices above are already copies.
    return wav.copy() if from_cache else wav
