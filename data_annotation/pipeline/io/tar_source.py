"""Tar metadata data source for the multi-node Phase-1 launcher.

Pure stdlib — imports NOTHING from ``pipeline`` / stages, so tar/IO bugs are
debuggable in isolation without standing up the model stack. The launcher
(``run_pipeline_multi.py``) is the only seam that wires this to the pipeline.

Design (backed by single-GPU bench on 4 real tars):
  * Stream members from NAS via ``extractfile`` — NO whole-tar copy.
    14-way ``extractfile`` vs ``copy``+extract was 0.96x (both bound by the
    NAS ~110 MB/s aggregate), and I/O is <1% of compute, so the simple
    no-prefetch path wins on complexity.
  * tar-affinity sharding: each tar goes to exactly ONE worker (avoids
    14 workers randomly seeking the same tar), ordered by first-seen line
    so workers advance roughly in crawl/time order.
  * extracted member is written to a local temp file with a NEUTRAL suffix
    (``.audio``) so downstream probes format from magic bytes, never from
    the (possibly wrong) metadata ext.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import tarfile
import time
from pathlib import Path


def resolve_snapshot(symlink_or_path: str) -> str:
    """Lock the metadata symlink to its concrete dated target at startup.

    ``master_metadata.jsonl`` is a symlink that rolls forward when a new
    snapshot is generated. Resolving once at launch keeps line numbers /
    tar-sharding stable for the whole run; ``--loop`` re-resolves on restart.
    """
    return os.path.realpath(symlink_or_path)


def iter_metadata(snapshot: str):
    """Yield ``(line_index, entry_dict)`` for each valid JSONL line."""
    with open(snapshot) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield idx, json.loads(line)
            except json.JSONDecodeError:
                continue


def weighted_schedule(weights: list[float]) -> list[int]:
    """Round-robin worker-id schedule, each id repeated ∝ its weight.

    Interleaved by *round* (round r emits every worker whose weight > r) so a
    worker's tars stay spread across the time-ordered tar list rather than
    clustered. Equal weights reduce to ``[0, 1, ..., n-1]`` — i.e. plain
    ``i % n`` modulo, the uniform default.

    Example: weights ``[1]*7 + [3]*7`` → length-28 schedule where each fast
    worker appears 3× and each slow worker 1× → fast workers pull 3× the tars.
    """
    w = [max(1, int(round(x))) for x in weights]
    sched: list[int] = []
    for r in range(max(w)):
        for wid, wt in enumerate(w):
            if r < wt:
                sched.append(wid)
    return sched


def shard_ordered(ordered, global_worker_id: int, num_workers: int,
                  weights: list[float] | None = None):
    """Static weighted sharding over an already-ordered work-unit list."""
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1 (got {num_workers})")
    if not (0 <= global_worker_id < num_workers):
        raise ValueError(
            f"global_worker_id must be in [0, {num_workers}) (got {global_worker_id})"
        )
    if weights is None:
        weights = [1.0] * num_workers
    if len(weights) != num_workers:
        raise ValueError(
            f"weights length {len(weights)} != num_workers {num_workers}")
    sched = weighted_schedule(weights)
    return [ordered[i] for i in range(len(ordered))
            if sched[i % len(sched)] == global_worker_id]


def shard_tars(snapshot: str, global_worker_id: int, num_workers: int,
               weights: list[float] | None = None):
    """Return this worker's tars as ``[(tar_path, [entries...]), ...]``.

    Groups entries by ``tar_path``, orders tars by first-seen line index
    (= crawl/time order, since snapshots are append-only), then assigns each
    tar to a worker via :func:`weighted_schedule`. tars are uniform (~77
    samples each), so a worker's tar count ≈ its share of total work.

    ``weights`` (one per global worker, default all-1) lets a faster machine
    pull proportionally more: e.g. a node 3× as fast gets weight 3 on each of
    its workers, so it processes ~3× the tars and both nodes finish together.
    Equal weights == plain ``i % num_workers`` modulo (backward compatible).
    """
    return shard_ordered(all_tars_ordered(snapshot),
                         global_worker_id, num_workers, weights)


# Compact tar-index cache -----------------------------------------------------
# The metadata JSONL can be large, but Phase-1 only ever
# reads THREE fields per sample: sample_id, member, tar_path. Every worker used
# to re-parse the whole file at startup. We cache a compact
# JSON sidecar keyed by the snapshot's
# (size, mtime); a rolled snapshot invalidates it automatically. Best-effort:
# any cache error falls straight back to a full parse, so outputs never change.
# Do not use pickle here: shared replaceable pickle caches can execute code.
_TARINDEX_VERSION = 2


def _tarindex_cache_path(snapshot: str) -> str:
    return f"{snapshot}.tarindex.v{_TARINDEX_VERSION}.json"


def _snapshot_sig(snapshot: str) -> tuple[int, int]:
    st = os.stat(snapshot)
    return (st.st_size, int(st.st_mtime))


def _build_tar_index(snapshot: str) -> list[tuple[str, list[tuple]]]:
    """Parse the JSONL once → ``[(tar_path, [(sample_id, member), ...]), ...]``.

    Keeps only the fields Phase-1 uses. Same grouping + first-seen ordering as
    the legacy :func:`all_tars_ordered` so downstream behaviour is identical.
    """
    first_seen: dict[str, int] = {}
    by_tar: dict[str, list[tuple]] = {}
    for idx, e in iter_metadata(snapshot):
        t = e.get("tar_path", "")
        if not t:
            continue
        if t not in first_seen:
            first_seen[t] = idx
            by_tar[t] = []
        by_tar[t].append((e.get("sample_id"), e.get("member")))
    tars = sorted(first_seen, key=lambda t: first_seen[t])
    return [(t, by_tar[t]) for t in tars]


def _cleanup_stale_caches(snapshot: str) -> None:
    """Best-effort housekeeping, run only right after publishing a fresh cache
    (i.e. rarely — a new snapshot or version bump). Drops THIS snapshot's
    superseded cache versions and any abandoned ``.tmp`` from killed builds, so
    the sidecars don't accumulate 129 MB/snapshot forever. Never removes the
    live cache; scoped to this snapshot's own files so it can't race another
    snapshot's build. A losing concurrent same-snapshot build just fails its
    ``os.replace`` harmlessly (caught below)."""
    import glob
    keep = _tarindex_cache_path(snapshot)
    for pat in (f"{keep}.*.tmp",  # abandoned tmp from an interrupted build
                f"{snapshot}.tarindex.v*.pkl",
                f"{snapshot}.tarindex.v*.json"):  # superseded versions
        for p in glob.glob(pat):
            if p == keep:
                continue
            try:
                os.remove(p)
            except OSError:
                pass


def _load_or_build_tar_index(snapshot: str) -> list[tuple[str, list[tuple]]]:
    cache = _tarindex_cache_path(snapshot)
    try:
        sig = _snapshot_sig(snapshot)
        with open(cache, encoding="utf-8") as f:
            blob = json.load(f)
        if tuple(blob.get("sig", ())) == sig:
            return blob["tars"]
    except Exception:  # noqa: BLE001 - missing/stale/corrupt cache → rebuild
        pass
    tars = _build_tar_index(snapshot)
    try:  # publish atomically; best-effort (parse already succeeded)
        sig = _snapshot_sig(snapshot)
        tmp = f"{cache}.{socket.gethostname()}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"sig": sig, "tars": tars}, f, ensure_ascii=False)
        os.replace(tmp, cache)
        _cleanup_stale_caches(snapshot)
    except Exception:  # noqa: BLE001 - read-only dir / race; not fatal
        pass
    return tars


def all_tars_ordered(snapshot: str) -> list[tuple[str, list]]:
    """Full time-ordered ``[(tar_path, [entries...]), ...]`` (no sharding).

    Groups entries by ``tar_path`` and orders tars by first-seen line index
    (= crawl/time order, snapshots being append-only). This is the basis for
    BOTH static sharding (:func:`shard_tars` filters it) and the pull-based
    claiming loop (every worker iterates the same list front-to-back).

    Backed by a compact per-snapshot cache (see :func:`_load_or_build_tar_index`)
    so the multi-GB JSONL is parsed at most once per snapshot rather than once
    per worker. Entries are re-expanded to the ``{sample_id, member, tar_path}``
    dicts the launcher expects, so callers are unchanged.
    """
    compact = _load_or_build_tar_index(snapshot)
    return [
        (t, [{"sample_id": sid, "member": member, "tar_path": t}
             for sid, member in pairs])
        for t, pairs in compact
    ]


class ClaimDir:
    """Filesystem work-queue on the shared NAS — turns static sharding into
    pull-based work-stealing across machines, with NO preset speed ratio.

    A free worker scans the time-ordered tar list and grabs the earliest tar
    that is neither ``.done`` nor live-``.lock``ed, so a faster machine simply
    pulls more. Two markers per tar live under ``<out>/_claims/``:

      * ``<tar>.lock`` — a heartbeated lease ``{owner, ts}``. Claimed via an
        atomic ``O_CREAT|O_EXCL`` create (atomic on NFSv3+/v4). A lock whose
        ``ts`` is older than ``lease_s`` is *stale* (owner crashed) and gets
        stolen — this is the free crash-recovery.
      * ``<tar>.done`` — terminal; written once all the tar's samples are
        processed. Workers skip ``.done`` tars on every scan.

    Correctness rests on the atomic create: two workers can never both win a
    fresh lock. A stale-steal race (rare; only on crash) can at worst make two
    workers redo one tar — harmless, since per-sample ``partial.json`` dedups
    the actual inference.

    Access is stat-by-exact-path only (never ``listdir``), so the flat
    ~2 files/tar layout is fine even at ~6k tars.
    """

    def __init__(self, out_root: str, lease_s: float = 1800.0,
                 claims_dir: "str | None" = None):

        self.dir = Path(claims_dir) if claims_dir else Path(out_root) / "_claims"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lease_s = lease_s
        self._hb_interval = lease_s / 4  # throttle heartbeat writes to this
        self._hb_last: dict[str, float] = {}  # tar-name → last lease write ts

    def _lock(self, tar_path: str) -> Path:
        return self.dir / (Path(tar_path).name + ".lock")

    def _done(self, tar_path: str) -> Path:
        return self.dir / (Path(tar_path).name + ".done")

    def is_done(self, tar_path: str) -> bool:
        return self._done(tar_path).exists()

    def done_names(self) -> set[str]:
        """One bulk ``scandir`` of the claims dir → the set of tar *basenames*
        that already carry a ``.done`` marker (e.g. ``bilibili_00232.tar``).

        Seeds the dynamic loop's ``seen_done`` in a handful of readdir RPCs
        instead of one ``stat`` per tar. At ~57k tars a fresh worker otherwise
        stats every tar's ``.done`` on its first pass — a metadata storm paid
        again on every relaunch. readdir batches ~hundreds of names per RPC, so
        even 114k entries (2/tar) is a few hundred RPCs, not 57k stats. This is
        pure memoisation: the per-tar :meth:`is_done` / :meth:`acquire` checks
        in the claim loop still gate correctness, so a stale snapshot is safe.
        """
        out: set[str] = set()
        try:
            with os.scandir(self.dir) as it:
                for de in it:
                    if de.name.endswith(".done"):
                        out.add(de.name[:-len(".done")])
        except FileNotFoundError:
            pass
        return out

    def mark_done(self, tar_path: str) -> None:
        # write-then-rename = atomic publish (no half-written .done observed)
        tmp = self._done(tar_path).with_suffix(
            f".done.{socket.gethostname()}.{os.getpid()}.tmp")
        tmp.write_text(str(time.time()))
        os.chmod(tmp, 0o664)
        os.replace(tmp, self._done(tar_path))

    def release(self, tar_path: str, owner: str) -> None:
        """Release a failed tar for immediate retry, but only if still owner."""
        lock = self._lock(tar_path)
        try:
            payload = json.loads(lock.read_text())
            if payload.get("owner") != owner:
                return
            lock.unlink()
            self._hb_last.pop(lock.name, None)
        except (FileNotFoundError, OSError, ValueError, TypeError):
            return

    def acquire(self, tar_path: str, owner: str) -> bool:
        """Try to claim ``tar_path``. Returns True if this worker now owns it.

        Wins iff the lock is free (atomic create) or stale (owner past lease).
        """
        lock = self._lock(tar_path)
        payload = json.dumps({"owner": owner, "ts": time.time()}).encode()
        try:
            # Group members may recover stale locks without making them world-writable.
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
        except FileExistsError:
            # Someone holds (or is mid-create of) the lock. Read its ts; if the
            # payload isn't readable yet (the O_EXCL winner hasn't written it),
            # fall back to the file's mtime — a just-created lock then reads as
            # FRESH, so the create-loser backs off instead of racing in via the
            # steal path. Only a genuinely old lock (crashed owner) is stolen.
            ts = 0.0
            try:
                ts = float(json.loads(lock.read_text()).get("ts", 0))
            except Exception:  # noqa: BLE001 - empty/partial/corrupt payload
                pass
            if ts <= 0:
                try:
                    ts = lock.stat().st_mtime
                except FileNotFoundError:
                    return False  # released mid-check; next scan retries
            if time.time() - ts < self.lease_s:
                return False  # a live owner holds it
            try:  # stale (crashed owner) → steal, best-effort reopen-truncate
                fd = os.open(lock, os.O_WRONLY | os.O_TRUNC)
            except (FileNotFoundError, PermissionError, OSError):
                # released mid-check, OR an old non-world-writable lock from a
                # different uid we can't steal — skip it (don't crash the
                # worker); the owner/lease or a chmod will sort it out.
                return False
        os.write(fd, payload)
        os.close(fd)
        self._hb_last[lock.name] = time.time()  # fresh ts just written
        return True

    def heartbeat(self, tar_path: str, owner: str) -> None:
        """Refresh the lease ``ts`` so a slow-but-alive worker isn't stolen.

        Throttled to ~lease/4: callers fire this per sample (~480k times over a
        run), but the lease is 1800s, so one NAS write every few hundred
        seconds keeps it fresh without a per-sample write storm.
        """
        lock = self._lock(tar_path)
        now = time.time()
        if now - self._hb_last.get(lock.name, 0.0) < self._hb_interval:
            return
        try:
            lock.write_text(json.dumps({"owner": owner, "ts": now}))
            self._hb_last[lock.name] = now
        except Exception:  # noqa: BLE001 - heartbeat is best-effort
            pass

    @staticmethod
    def owner_id(worker_id: int) -> str:
        return f"{socket.gethostname()}:w{worker_id}:{os.getpid()}"


class TarReader:
    """Hold one NAS tar handle; extract a member to a local temp file.

    The handle is reused across all members of the tar (members of one tar
    are processed back-to-back by the same worker, thanks to tar-affinity).
    """

    def __init__(self, tar_path: str, tmp_dir: str):
        self.tar_path = tar_path
        self._tf = tarfile.open(tar_path, "r")  # raises if tar unreadable
        self._tmp = Path(tmp_dir)
        self._tmp.mkdir(parents=True, exist_ok=True)

    def extract(self, member: str, sid: str) -> str:
        """``extractfile`` a member from NAS → local temp file; return path.

        Neutral ``.audio`` suffix: the pipeline probes format from content,
        so a wrong metadata ext can never mislead the decoder.
        """
        f = self._tf.extractfile(member)
        if f is None:
            raise FileNotFoundError(f"member {member!r} not in {self.tar_path}")
        p = self._tmp / f"{sid}.audio"
        # Stream instead of materialising a potentially multi-GB member twice.
        with f, p.open("wb") as dst:
            shutil.copyfileobj(f, dst, length=8 * 1024 * 1024)
        return str(p)

    def close(self) -> None:
        try:
            self._tf.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
