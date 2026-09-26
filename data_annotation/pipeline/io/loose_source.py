"""Read loose audio files as an alternative phase-one source.

An identity-list JSONL maps source filenames to stable sample IDs. The index
is cached next to that list and invalidated when the list changes.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
from pathlib import Path


UNIT_SEP = "__"


AUDIO_EXTS = frozenset(
    {".m4a", ".mp3", ".wav", ".flac", ".aac", ".opus", ".ogg", ".webm"})

_LOOSEINDEX_VERSION = 1


def _index_cache_path(list_path: str) -> str:
    return f"{list_path}.looseindex.v{_LOOSEINDEX_VERSION}.json"


def _list_sig(list_path: str) -> tuple[int, int]:
    st = os.stat(list_path)
    return (st.st_size, int(st.st_mtime))


def _iter_identity(list_path: str):
    """Yield ``(video_id, sample_id)`` from the identity JSONL (bad lines skipped)."""
    with open(list_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = e.get("video_id")
            sid = e.get("sample_id")
            if vid and sid:
                yield vid, sid


def _build_index(list_path: str, raw_root: str):

    vid2sid: dict[str, str] = {}
    for vid, sid in _iter_identity(list_path):
        vid2sid.setdefault(vid, sid)


    units: list[list] = []
    seen_vid: set[str] = set()
    n_files = n_dup = 0
    root = Path(raw_root)
    for category in sorted(os.listdir(root)):
        cat_dir = root / category
        if not cat_dir.is_dir():
            continue
        for batch in sorted(os.listdir(cat_dir)):
            audio_dir = cat_dir / batch / "audio"
            if not audio_dir.is_dir():
                continue
            pairs: list[list[str]] = []
            with os.scandir(audio_dir) as it:
                names = sorted(de.name for de in it if de.is_file())
            for name in names:
                stem, ext = os.path.splitext(name)
                if ext.lower() not in AUDIO_EXTS:
                    continue
                n_files += 1
                sid = vid2sid.get(stem)
                if sid is None:
                    continue
                if stem in seen_vid:
                    n_dup += 1
                    continue
                seen_vid.add(stem)
                pairs.append([sid, f"{category}/{batch}/audio/{name}"])
            if pairs:
                units.append([f"{category}{UNIT_SEP}{batch}", pairs])

    stats = {
        "list_rows": len(vid2sid),
        "tree_audio_files": n_files,
        "matched": sum(len(p) for _, p in units),
        "dup_files": n_dup,
        "missing": len(vid2sid) - len(seen_vid),
        "units": len(units),
    }
    return units, stats


def load_or_build_index(list_path: str, raw_root: str):

    cache = _index_cache_path(list_path)
    try:
        sig = _list_sig(list_path)
        with open(cache, encoding="utf-8") as f:
            blob = json.load(f)
        if (tuple(blob.get("sig", ())) == sig
                and blob.get("raw_root") == raw_root):
            return blob["units"], blob.get("stats", {})
    except Exception:  # noqa: BLE001 - missing/stale/corrupt cache → rebuild
        pass
    units, stats = _build_index(list_path, raw_root)
    try:
        sig = _list_sig(list_path)
        tmp = f"{cache}.{socket.gethostname()}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"sig": sig, "raw_root": raw_root,
                       "units": units, "stats": stats}, f, ensure_ascii=False)
        os.replace(tmp, cache)
    except Exception:  # noqa: BLE001 - read-only dir / race; not fatal
        pass
    return units, stats


def all_units_ordered(list_path: str, raw_root: str):

    units, _stats = load_or_build_index(list_path, raw_root)
    return [
        (unit, [{"sample_id": sid, "member": rel, "tar_path": unit}
                for sid, rel in pairs])
        for unit, pairs in units
    ]


class LooseReader:


    def __init__(self, raw_root: str, tmp_dir: str):
        self.raw_root = raw_root
        self._tmp = Path(tmp_dir)
        self._tmp.mkdir(parents=True, exist_ok=True)

    def extract(self, member: str, sid: str) -> str:
        src = os.path.join(self.raw_root, member)
        p = self._tmp / f"{sid}.audio"
        with open(src, "rb") as f, p.open("wb") as dst:
            shutil.copyfileobj(f, dst, length=8 * 1024 * 1024)
        return str(p)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
