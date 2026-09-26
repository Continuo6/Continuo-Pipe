"""Publish and consume optional phase-two work items.

Items are created as phase one completes recordings. Prefix directories are
created lazily so unused batches do not accumulate empty directories.
"""
from __future__ import annotations

import os

QUEUE_DIRNAME = "_p2queue"
ENABLED_MARKER = "ENABLED"
TRACKS = ("short", "long")


_HEX = "0123456789abcdef"
PREFIXES = tuple(a + b for a in _HEX for b in _HEX)


def queue_root(processed_root: str) -> str:
    return os.path.join(processed_root, QUEUE_DIRNAME)


def _track_dir(processed_root: str, track: str, prefix: str) -> str:
    return os.path.join(processed_root, QUEUE_DIRNAME, track, prefix)


def enable(processed_root: str) -> None:

    for track in TRACKS:
        os.makedirs(os.path.join(processed_root, QUEUE_DIRNAME, track),
                    exist_ok=True)
    marker = os.path.join(queue_root(processed_root), ENABLED_MARKER)
    try:
        fd = os.open(marker, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o666)
        os.close(fd)
    except OSError:
        pass


def is_enabled(processed_root: str) -> bool:

    return os.path.exists(
        os.path.join(queue_root(processed_root), ENABLED_MARKER)
    )


def publish(processed_root: str, sid: str, *, short: bool = True,
            long: bool = False) -> None:

    prefix = sid[:2]
    for track, want in (("short", short), ("long", long)):
        if not want:
            continue
        path = os.path.join(_track_dir(processed_root, track, prefix), sid)
        try:
            fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o666)
            os.close(fd)
        except FileNotFoundError:

            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o666)
                os.close(fd)
            except OSError:
                pass
        except OSError:
            pass


def prefixes(processed_root: str, tracks) -> list[str]:

    out: set[str] = set()
    for track in tracks:
        try:
            with os.scandir(os.path.join(processed_root, QUEUE_DIRNAME, track)) as it:
                out.update(e.name for e in it)
        except OSError:
            pass
    return sorted(out)


def pending(processed_root: str, prefix: str, track: str) -> list[str]:

    try:
        with os.scandir(_track_dir(processed_root, track, prefix)) as it:
            return [e.name for e in it]
    except OSError:
        return []


def ack(processed_root: str, sid: str, track: str) -> None:

    try:
        os.unlink(os.path.join(_track_dir(processed_root, track, sid[:2]), sid))
    except OSError:
        pass


def ack_finished(processed_root: str, track: str, items) -> int:

    n = 0
    for sid, final_path in items:
        if os.path.exists(final_path):
            ack(processed_root, sid, track)
            n += 1
    return n
