"""Coordinate node ownership and completion markers."""
from __future__ import annotations

import glob
import fcntl
import json
import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager

NODES_DIRNAME = "_nodes"
HEARTBEAT_S = 60.0
LEASE_S = 1800.0
RUN_FILE = "RUN.json"
REGISTRY_LOCK_FILE = ".registry.lock"


def nodes_dir(root: str) -> str:
    return os.path.join(root, NODES_DIRNAME)


def _owner_file(root: str, node_id: int) -> str:
    return os.path.join(nodes_dir(root), f"n{node_id}.owner")


def _run_file(root: str) -> str:
    return os.path.join(nodes_dir(root), RUN_FILE)


def _self_owner() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _owner_host_pid(owner: str) -> tuple[str, "int | None"]:
    parts = owner.split(":", 2)
    try:
        pid = int(parts[1]) if len(parts) > 1 else None
    except ValueError:
        pid = None
    return (parts[0] if parts else ""), pid


def _pid_alive(pid: "int | None") -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_owner(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def _write_atomic(path: str, text: str) -> None:
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _create_owner(path: str, owner: str) -> bool:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    except FileExistsError:
        return False
    try:
        payload = owner.encode()
        if os.write(fd, payload) != len(payload):
            raise OSError(f"short write while registering node owner: {path}")
    except BaseException:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    finally:
        os.close(fd)
    return True


@contextmanager
def _registry_lock(root: str, wait_s: float = 30.0):

    d = nodes_dir(root)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, REGISTRY_LOCK_FILE)
    deadline = time.monotonic() + wait_s
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for node registry lock: {path}")
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _read_run(root: str) -> "dict | None":
    try:
        with open(_run_file(root)) as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError) as ex:
        raise RuntimeError(f"invalid node run metadata: {_run_file(root)}") from ex
    expected = data.get("expected_ids")
    generation = data.get("generation")
    if not isinstance(expected, list) or not all(isinstance(i, int) for i in expected) \
            or not isinstance(generation, str) or not generation:
        raise RuntimeError(f"invalid node run metadata: {_run_file(root)}")
    return data


def _marker_generation(root: str, node_id: int) -> str:
    try:
        with open(os.path.join(root, f"_PHASE1_DONE_n{node_id}")) as f:
            return f.read().strip().split()[0]
    except (OSError, IndexError):
        return ""


def _prepare_run_locked(root: str, expected_nodes: int,
                        lease_s: float = LEASE_S) -> dict:
    if expected_nodes < 1:
        raise ValueError(f"--nodes must be >= 1 (got {expected_nodes})")
    expected = list(range(expected_nodes))
    state = _read_run(root)


    live = live_ids(root, lease_s)
    rotate = state is None
    if state is not None and state["expected_ids"] != expected:
        if live:
            raise SystemExit(
                f"[nodes] topology is fixed at {state['expected_ids']} while nodes {live} "
                f"are active; cannot change it to {expected}")
        rotate = True
    elif state is not None and not live:
        generation = state["generation"]


        rotate = all(_marker_generation(root, i) == generation
                     for i in state["expected_ids"])
    if rotate:
        state = {
            "generation": uuid.uuid4().hex,
            "expected_ids": expected,
            "created": time.time(),
        }
        _write_atomic(_run_file(root), json.dumps(state, sort_keys=True))
    return state


def register(root: str, node_id: int, lease_s: float = LEASE_S,
             start_heartbeat: bool = True,
             expected_nodes: "int | None" = None) -> str:

    if node_id < 0:
        raise ValueError(f"node_id must be >= 0 (got {node_id})")
    me = f"{_self_owner()}:{uuid.uuid4().hex}"
    with _registry_lock(root):
        state = (_prepare_run_locked(root, expected_nodes, lease_s)
                 if expected_nodes is not None else None)
        if state is not None and node_id not in state["expected_ids"]:
            raise ValueError(
                f"--node-id {node_id} outside expected ids {state['expected_ids']}")
        path = _owner_file(root, node_id)
        if not _create_owner(path, me):
            prev = _read_owner(path)
            prev_host, prev_pid = _owner_host_pid(prev)
            try:
                age = time.time() - os.stat(path).st_mtime
            except OSError:
                age = float("inf")
            if prev_host == socket.gethostname() and prev_pid == os.getpid():
                me = prev
            else:
                same_host_alive = (prev_host == socket.gethostname()
                                   and _pid_alive(prev_pid))
                if age < lease_s and (prev_host != socket.gethostname()
                                      or same_host_alive):
                    raise SystemExit(
                        f"[nodes] node-id {node_id} is owned by {prev} "
                        f"(last heartbeat {age:.0f}s ago). Two launchers sharing a "
                        "node-id would overwrite logs, shards, and completion markers.")
                stale_path = f"{path}.stale.{uuid.uuid4().hex}"
                try:
                    os.replace(path, stale_path)
                    if not _create_owner(path, me):
                        raise RuntimeError(f"failed to publish new node owner: {path}")
                finally:
                    try:
                        os.unlink(stale_path)
                    except OSError:
                        pass

        try:
            os.unlink(os.path.join(root, f"_PHASE1_DONE_n{node_id}"))
        except FileNotFoundError:
            pass
        if node_id == 0:
            try:
                os.unlink(os.path.join(root, "_PHASE1_DONE"))
            except FileNotFoundError:
                pass
        os.utime(path, None)

    if start_heartbeat:
        interval = min(HEARTBEAT_S, max(1.0, lease_s / 3.0))

        def _beat() -> None:
            while True:
                time.sleep(interval)
                if not heartbeat(root, node_id, me):
                    return
        threading.Thread(target=_beat, daemon=True, name="node-heartbeat").start()
    return me


def heartbeat(root: str, node_id: int, owner: "str | None" = None) -> bool:

    with _registry_lock(root):
        path = _owner_file(root, node_id)
        current = _read_owner(path)
        if owner is None:
            host, pid = _owner_host_pid(current)
            if host != socket.gethostname() or pid != os.getpid():
                return False
            owner = current
        if not current or current != owner:
            return False
        try:
            os.utime(path, None)
            return True
        except OSError:
            return False


def unregister(root: str, node_id: int, owner: "str | None" = None) -> bool:

    with _registry_lock(root):
        path = _owner_file(root, node_id)
        current = _read_owner(path)
        if owner is None:
            host, pid = _owner_host_pid(current)
            if host != socket.gethostname() or pid != os.getpid():
                return False
            owner = current
        if not current or current != owner:
            return False
        try:
            os.unlink(path)
            return True
        except OSError:
            return False


def mark_done(root: str, node_id: int, owner: str) -> None:

    with _registry_lock(root):
        if _read_owner(_owner_file(root, node_id)) != owner:
            raise RuntimeError(f"lost node ownership before marking done: node {node_id}")
        state = _read_run(root)
        generation = state["generation"] if state is not None else "legacy"
        _write_atomic(
            os.path.join(root, f"_PHASE1_DONE_n{node_id}"),
            f"{generation} {time.time()}")
        if state is not None and state["expected_ids"] == [0] and node_id == 0:
            _write_atomic(os.path.join(root, "_PHASE1_DONE"),
                          f"{generation} {time.time()}")


def registered_ids(root: str) -> list[int]:
    out = []
    for p in glob.glob(os.path.join(nodes_dir(root), "n*.owner")):
        stem = os.path.basename(p)[1:-len(".owner")]
        if stem.isdigit():
            out.append(int(stem))
    return sorted(out)


def live_ids(root: str, lease_s: float = LEASE_S) -> list[int]:
    now = time.time()
    out = []
    for i in registered_ids(root):
        try:
            if now - os.stat(_owner_file(root, i)).st_mtime < lease_s:
                out.append(i)
        except OSError:
            pass
    return out


def phase1_done(root: str, min_nodes: int = 1, lease_s: float = LEASE_S) -> bool:

    state = _read_run(root)
    if state is not None:
        expected = state["expected_ids"]
        if len(expected) < max(1, min_nodes):
            return False
        generation = state["generation"]
        return all(_marker_generation(root, i) == generation for i in expected)


    live = live_ids(root, lease_s)
    if os.path.exists(os.path.join(root, "_PHASE1_DONE")) and not live:
        return True
    done = {
        int(os.path.basename(p)[len("_PHASE1_DONE_n"):])
        for p in glob.glob(os.path.join(root, "_PHASE1_DONE_n*"))
        if os.path.basename(p)[len("_PHASE1_DONE_n"):].isdigit()
    }
    if len(done) < max(1, min_nodes):
        return False
    return all(i in done for i in live)
