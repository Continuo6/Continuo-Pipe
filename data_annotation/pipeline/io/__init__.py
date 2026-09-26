"""I/O helpers for large-scale runs (tar metadata source, sharding)."""
from __future__ import annotations

import json
import os
import tempfile


def atomic_dump(obj, path: str) -> None:
    """Write ``obj`` as JSON to ``path`` atomically (tmp + ``os.replace``).

    The single writer for every resume-critical manifest. A crash / OOM-kill
    mid-write must never leave a truncated file: the launcher treats a resume
    marker's mere existence as "done", so a torn file would be skipped forever
    AND fail the reader's ``json.load``. ``os.replace`` is atomic on POSIX
    (incl. same-dir renames on NFS), so a reader sees either the old file or
    the whole new one — never a partial. Use this for partial.json, the empty
    ``[]`` markers, and the Phase-2 finals alike.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=parent,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        os.chmod(tmp, 0o664)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
