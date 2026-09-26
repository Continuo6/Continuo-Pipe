"""Assign recordings to sealed batches for incremental phase-two processing.

Every tar stays in one batch. Phase one seals a batch only after all its members
finish; phase two marks it done after its backlog has been drained.
"""
from __future__ import annotations

import hashlib
import fcntl
import os
import socket
import time
import uuid
from contextlib import contextmanager

BATCHES_DIRNAME = "_batches"


SEAL_GRACE_S = 180.0


SEALER_LEASE_S = 900.0
CURRENT_FILE = "CURRENT"
BY_TAR_DIRNAME = "by_tar"
ASSIGN_PENDING_DIRNAME = "assign_pending"
SEALED_MARKER = "_SEALED"

SEAL_PENDING_MARKER = "_SEAL_PENDING"

SEALER_LOCK_FILE = "_SEALER.lock"
ASSIGN_LOCK_FILE = "_ASSIGN.lock"
PHASE2_DONE_MARKER = "_PHASE2_DONE"


DEFAULT_BATCH_RECORDINGS = 10_000


def batch_name(i: int) -> str:
    return f"b{i:06d}"


def batch_root(base: str, i: int) -> str:
    return os.path.join(base, batch_name(i))


def _bdir(base: str) -> str:
    return os.path.join(base, BATCHES_DIRNAME)


def _members_dir(base: str, i: int) -> str:
    return os.path.join(_bdir(base), batch_name(i))


def _by_tar(base: str, tar_name: str) -> str:
    return os.path.join(_bdir(base), BY_TAR_DIRNAME, tar_name)


def _read_int(path: str, default: int = 0) -> int:
    try:
        with open(path) as f:
            return int(f.read().strip() or default)
    except (OSError, ValueError):
        return default


def _read_int_strict(path: str, *, missing: "int | None" = None) -> "int | None":

    try:
        with open(path) as f:
            raw = f.read().strip()
    except FileNotFoundError:
        return missing
    if not raw:
        raise RuntimeError(f"empty integer metadata: {path}")
    try:
        return int(raw)
    except ValueError as ex:
        raise RuntimeError(f"invalid integer metadata: {path}: {raw!r}") from ex


def _write_atomic(path: str, text: str) -> None:

    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o666)
        try:
            payload = text.encode()
            view = memoryview(payload)
            while view:
                n = os.write(fd, view)
                if n <= 0:
                    raise OSError(f"short write while writing {tmp}")
                view = view[n:]
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_owner(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def _require_owner(path: str, token: str) -> None:
    if _read_owner(path) != token:
        raise RuntimeError(f"lost filesystem lease while updating batch metadata: {path}")


@contextmanager
def _guard_file(path: str, *, wait_s: float = 30.0):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    deadline = time.monotonic() + wait_s
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for filesystem guard: {path}")
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@contextmanager
def _exclusive_file(path: str, *, stale_s: "float | None" = 300.0,
                    wait_s: float = 30.0):

    token = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
    deadline = time.monotonic() + wait_s
    os.makedirs(os.path.dirname(path), exist_ok=True)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
            try:
                payload = token.encode()
                if os.write(fd, payload) != len(payload):
                    raise OSError(f"short write while acquiring batch lock: {path}")
            except BaseException:
                try:
                    os.unlink(path)
                except OSError:
                    pass
                raise
            finally:
                os.close(fd)
            break
        except FileExistsError:
            stale = False
            if stale_s is not None:
                try:
                    stale = time.time() - os.stat(path).st_mtime > stale_s
                except OSError:
                    stale = False
            if stale_s is not None and stale:


                guard = path + ".takeover"
                with _guard_file(guard, wait_s=wait_s):
                    try:
                        if time.time() - os.stat(path).st_mtime <= stale_s:
                            continue
                    except FileNotFoundError:
                        continue
                    stale_path = f"{path}.stale.{uuid.uuid4().hex}"
                    try:
                        os.replace(path, stale_path)
                    except OSError:
                        continue
                    try:
                        os.unlink(stale_path)
                    except OSError:
                        pass
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for batch lock: {path}")
            time.sleep(0.05)
    try:
        yield token
    finally:
        if stale_s is None:
            if _read_owner(path) == token:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        else:

            with _guard_file(path + ".takeover", wait_s=wait_s):
                if _read_owner(path) == token:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass


def init(base: str) -> None:

    os.makedirs(os.path.join(_bdir(base), BY_TAR_DIRNAME), exist_ok=True)
    os.makedirs(os.path.join(_bdir(base), ASSIGN_PENDING_DIRNAME), exist_ok=True)
    cur = os.path.join(_bdir(base), CURRENT_FILE)
    lock = os.path.join(_bdir(base), ASSIGN_LOCK_FILE)
    with _exclusive_file(lock) as token:
        existing = _read_int_strict(cur, missing=None)
        if existing is None:
            _require_owner(lock, token)
            _write_atomic(cur, "0")
        elif existing < 0:
            raise RuntimeError(f"negative CURRENT metadata: {cur}: {existing}")
    os.makedirs(_members_dir(base, current(base)), exist_ok=True)


def current(base: str) -> int:
    path = os.path.join(_bdir(base), CURRENT_FILE)
    value = _read_int_strict(path, missing=None)
    if value is None:
        raise RuntimeError(f"batch metadata is not initialized: {path}")
    return value


def _pending_path(base: str, tar_name: str) -> str:
    return os.path.join(_bdir(base), ASSIGN_PENDING_DIRNAME, tar_name)


def _read_pending(base: str, tar_name: str) -> "tuple[int, int] | None":
    path = _pending_path(base, tar_name)
    try:
        with open(path) as f:
            parts = f.read().strip().split()
    except FileNotFoundError:
        return None
    if len(parts) != 2:
        raise RuntimeError(f"invalid pending batch assignment: {path}")
    try:
        batch_i, recordings = int(parts[0]), int(parts[1])
    except ValueError as ex:
        raise RuntimeError(f"invalid pending batch assignment: {path}") from ex
    if batch_i < 0 or recordings < 0:
        raise RuntimeError(f"negative pending batch assignment: {path}")
    return batch_i, recordings


def _is_atomic_tmp_name(name: str) -> bool:

    parts = name.rsplit(".", 3)
    return len(parts) == 4 and parts[1].isdigit() and parts[3] == "tmp" \
        and len(parts[2]) == 32 and all(c in "0123456789abcdef" for c in parts[2])


def _pending_batches(base: str) -> set[int]:
    out: set[int] = set()
    d = os.path.join(_bdir(base), ASSIGN_PENDING_DIRNAME)
    try:
        with os.scandir(d) as it:
            for e in it:


                if _is_atomic_tmp_name(e.name):
                    continue
                try:
                    with open(e.path) as f:
                        out.add(int(f.read().strip().split()[0]))
                except (OSError, ValueError, IndexError):

                    out.add(-1)
    except FileNotFoundError:
        pass
    return out


def _pending_blocks_seal(base: str, i: int) -> bool:
    pending = _pending_batches(base)
    if -1 in pending:
        raise RuntimeError("corrupt pending batch assignment blocks sealing")
    return i in pending


def _parse_member(name: str) -> tuple[str, int]:

    tar, sep, n = name.rpartition("#")
    if not sep:
        return name, 1
    try:
        return tar, int(n)
    except ValueError:
        return name, 1


def member_recordings(base: str, i: int) -> int:

    try:
        with os.scandir(_members_dir(base, i)) as it:
            return sum(_parse_member(e.name)[1] for e in it)
    except FileNotFoundError:
        return 0


def members(base: str, i: int) -> list[str]:

    try:
        with os.scandir(_members_dir(base, i)) as it:
            return [_parse_member(e.name)[0] for e in it]
    except FileNotFoundError:
        return []


def _create_member(base: str, i: int, tar_name: str, n_recordings: int) -> None:
    d = _members_dir(base, i)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{tar_name}#{int(n_recordings)}")
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        os.close(fd)
    except FileExistsError:
        pass


def _ensure_member(base: str, i: int, tar_name: str, n_recordings: int) -> None:

    d = _members_dir(base, i)
    os.makedirs(d, exist_ok=True)
    try:
        with os.scandir(d) as it:
            for e in it:
                existing_tar, existing_n = _parse_member(e.name)
                if existing_tar == tar_name:
                    if existing_n != int(n_recordings):
                        raise RuntimeError(
                            f"recording count changed for {tar_name}: "
                            f"{existing_n} -> {n_recordings}")
                    return
    except OSError:
        raise
    _create_member(base, i, tar_name, n_recordings)


def assign(base: str, tar_path: str, n_recordings: int = 1,
           batch_recordings: int = DEFAULT_BATCH_RECORDINGS) -> int:

    tar_name = os.path.basename(tar_path)
    bt = _by_tar(base, tar_name)

    lock = os.path.join(_bdir(base), ASSIGN_LOCK_FILE)
    with _exclusive_file(lock) as token:
        existing = _read_int_strict(bt, missing=None)
        pending = _read_pending(base, tar_name)
        if existing is not None:
            i = existing
            if i < 0:
                raise RuntimeError(f"negative batch mapping: {bt}: {i}")
            if pending is not None and pending[0] != i:
                raise RuntimeError(f"mapping/pending disagree for {tar_name}: {i} vs {pending}")


            if is_sealed(base, i) or is_phase2_done(base, i):
                raise RuntimeError(
                    f"refusing to repair/write assignment in sealed batch: {tar_name} -> {batch_name(i)}")
        elif pending is not None:

            i, recorded_n = pending
            if recorded_n != int(n_recordings):
                raise RuntimeError(
                    f"recording count changed while repairing {tar_name}: {recorded_n} -> {n_recordings}")
        else:
            i = current(base)
            if member_recordings(base, i) >= batch_recordings:
                i += 1
                os.makedirs(_members_dir(base, i), exist_ok=True)
                _require_owner(lock, token)
                _write_atomic(os.path.join(_bdir(base), CURRENT_FILE), str(i))

        pend = _pending_path(base, tar_name)
        if pending is None:
            _require_owner(lock, token)
            _write_atomic(pend, f"{i} {int(n_recordings)}")
        if existing is None:
            os.makedirs(os.path.dirname(bt), exist_ok=True)
            _require_owner(lock, token)
            _write_atomic(bt, str(i))
        _require_owner(lock, token)
        _ensure_member(base, i, tar_name, n_recordings)
        _require_owner(lock, token)
        os.unlink(pend)
        return i


def close_current(base: str) -> int:

    lock = os.path.join(_bdir(base), ASSIGN_LOCK_FILE)
    with _exclusive_file(lock) as token:
        i = current(base)


        _require_owner(lock, token)
        _write_atomic(os.path.join(_bdir(base), CURRENT_FILE), str(i + 1))
        _require_owner(lock, token)
        return i


def lookup(base: str, tar_path: str) -> "int | None":
    v = _read_int_strict(_by_tar(base, os.path.basename(tar_path)), missing=None)
    if v is not None and v < 0:
        raise RuntimeError(f"negative batch mapping for {tar_path}: {v}")
    return v


def all_batches(base: str) -> list[int]:

    out = []
    try:
        with os.scandir(_bdir(base)) as it:
            for e in it:
                if e.name.startswith("b") and e.name[1:].isdigit():
                    out.append(int(e.name[1:]))
    except FileNotFoundError:
        pass
    return sorted(out)


def is_sealed(base: str, i: int) -> bool:
    return os.path.exists(os.path.join(batch_root(base, i), SEALED_MARKER))


def is_phase2_done(base: str, i: int) -> bool:
    return os.path.exists(os.path.join(batch_root(base, i), PHASE2_DONE_MARKER))


def _member_fingerprint(mem: "list[str]") -> str:

    return f"{len(mem)}:" + hashlib.sha1(
        "\0".join(sorted(mem)).encode()).hexdigest()[:16]


def try_seal(base: str, i: int, is_terminal, grace_s: float = SEAL_GRACE_S) -> bool:

    if is_sealed(base, i):
        return True
    lock = os.path.join(_bdir(base), ASSIGN_LOCK_FILE)


    with _exclusive_file(lock):
        if current(base) <= i:
            return False
        if _pending_blocks_seal(base, i):
            return False
        mem = members(base, i)
    if not mem or not all(is_terminal(t) for t in mem):
        return False

    root = batch_root(base, i)
    if grace_s <= 0:


        try:
            with _exclusive_file(lock):
                if current(base) <= i or _pending_blocks_seal(base, i) \
                        or _member_fingerprint(members(base, i)) != _member_fingerprint(mem):
                    return False
                os.makedirs(root, exist_ok=True)
                _write_atomic(os.path.join(root, SEALED_MARKER), str(len(mem)))
        except OSError:
            return False
        return True

    fp = _member_fingerprint(mem)
    pend = os.path.join(root, SEAL_PENDING_MARKER)
    now = time.time()
    try:
        os.makedirs(root, exist_ok=True)
        prev = ""
        try:
            with open(pend) as f:
                prev = f.read().strip()
        except OSError:
            prev = ""
        prev_fp, _, prev_ts = prev.partition(" ")
        if prev_fp != fp:
            _write_atomic(pend, f"{fp} {now:.3f}")
            return False
        try:
            waited = now - float(prev_ts)
        except ValueError:
            _write_atomic(pend, f"{fp} {now:.3f}")
            return False
        if waited < grace_s:
            return False
        with _exclusive_file(lock):
            if current(base) <= i or _pending_blocks_seal(base, i) \
                    or _member_fingerprint(members(base, i)) != fp:
                return False
            _write_atomic(os.path.join(root, SEALED_MARKER), str(len(mem)))
    except OSError:
        return False
    return True


def sealer_lease(base: str, lease_s: float = SEALER_LEASE_S,
                 owner: "str | None" = None) -> bool:

    owner = owner or f"{socket.gethostname()}:{os.getpid()}"
    path = os.path.join(_bdir(base), SEALER_LOCK_FILE)
    try:
        os.makedirs(_bdir(base), exist_ok=True)
        takeover = path + ".takeover"


        with _guard_file(takeover):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
                try:
                    payload = owner.encode()
                    if os.write(fd, payload) != len(payload):
                        raise OSError(f"short write while acquiring sealer lease: {path}")
                finally:
                    os.close(fd)
                return True
            except FileExistsError:
                pass
            cur_owner = _read_owner(path)
            if cur_owner == owner:
                os.utime(path, None)
                return True
            try:
                age = time.time() - os.stat(path).st_mtime
            except OSError:
                return False
            if age <= lease_s:
                return False
            stale_path = f"{path}.stale.{uuid.uuid4().hex}"
            try:
                os.replace(path, stale_path)
            except OSError:
                return False
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
            except FileExistsError:
                return False
            try:
                payload = owner.encode()
                if os.write(fd, payload) != len(payload):
                    raise OSError(f"short write while taking sealer lease: {path}")
            finally:
                os.close(fd)
            try:
                os.unlink(stale_path)
            except OSError:
                pass
            return True
    except OSError:
        pass
    return False


def release_sealer_lease(base: str, owner: "str | None" = None) -> bool:

    owner = owner or f"{socket.gethostname()}:{os.getpid()}"
    path = os.path.join(_bdir(base), SEALER_LOCK_FILE)

    try:
        with _guard_file(path + ".takeover"):
            if _read_owner(path) != owner:
                return False
            try:
                os.unlink(path)
                return True
            except OSError:
                return False
    except (OSError, TimeoutError):
        return False


def mark_phase2_done(base: str, i: int, note: str = "") -> None:

    _write_atomic(
        os.path.join(batch_root(base, i), PHASE2_DONE_MARKER), note or "ok")


def is_vacant(base: str, i: int) -> bool:

    if member_recordings(base, i) > 0:
        return False
    try:
        with os.scandir(batch_root(base, i)) as it:
            return not any(e.name[:1] != "_" and e.is_dir() for e in it)
    except FileNotFoundError:
        return True


def active_roots(base: str) -> list[tuple[int, str]]:

    return [(i, batch_root(base, i)) for i in all_batches(base)
            if not is_phase2_done(base, i) and not is_vacant(base, i)]
