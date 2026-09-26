"""Read audio members and indexes from tar archives."""
from __future__ import annotations

import io
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf


class TarMemberFile(io.RawIOBase):


    def __init__(self, tar_path: str, offset: int, size: int):
        self._f = open(tar_path, "rb", buffering=0)
        self._off, self._size, self._pos = offset, size, 0


    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, pos, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, min(self._size, base + pos))
        return self._pos

    def readinto(self, b):
        n = min(len(b), self._size - self._pos)
        if n <= 0:
            return 0
        self._f.seek(self._off + self._pos)
        data = self._f.read(n)
        b[:len(data)] = data
        self._pos += len(data)
        return len(data)

    def read(self, n=-1):
        if n is None or n < 0:
            n = self._size - self._pos
        b = bytearray(n)
        got = self.readinto(b)
        return bytes(b[:got])

    def close(self):
        try:
            self._f.close()
        finally:
            super().close()


def load_pack_index(index_dir: str, kind: str = "dialogue") -> dict:

    idx = {}
    for fn in sorted(os.listdir(index_dir)):
        if not (fn.endswith(".tsv") and fn.startswith("pack.")):
            continue
        with open(os.path.join(index_dir, fn)) as f:
            for line in f:
                c = line.rstrip("\n").split("\t")
                if len(c) >= 4 and c[0] == kind:
                    idx[c[1]] = (c[2], c[3])
    return idx


def idx_offsets(tar_abs: str) -> dict:
    """<tar>.idx → {member: (offset, size)}"""
    out = {}
    try:
        with open(tar_abs + ".idx") as f:
            for line in f:
                c = line.rstrip("\n").split("\t")
                if len(c) == 3:
                    out[c[0]] = (int(c[1]), int(c[2]))
    except OSError:
        pass
    return out


def open_member(tar_abs: str, member: str, offs: dict | None = None):

    offs = offs if offs is not None else idx_offsets(tar_abs)
    o = offs.get(member)
    if o is None:
        raise KeyError(f"{member} not in {os.path.basename(tar_abs)}.idx")
    fh = TarMemberFile(tar_abs, o[0], o[1])
    if member.endswith(".m4a"):
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False,
                                         dir=os.environ.get("CONTINUO_DATA_TMP_DIR") or tempfile.gettempdir()) as tf:
            tf.write(fh.read())
            tmp = tf.name
        fh.close()
        wav_path = tmp + ".wav"
        p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, wav_path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        os.unlink(tmp)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg: {p.stderr[:200]!r}")
        return sf.SoundFile(wav_path), wav_path
    return sf.SoundFile(fh), fh


def read_span(tar_abs: str, member: str, start_frame: int, n_frames: int,
              offs: dict | None = None, dtype="float32") -> tuple:

    snd, holder = open_member(tar_abs, member, offs)
    try:
        snd.seek(max(0, start_frame))
        data = snd.read(n_frames, dtype=dtype, always_2d=False)
        sr = snd.samplerate
    finally:
        snd.close()
        if isinstance(holder, str):
            try:
                os.unlink(holder)
            except OSError:
                pass
        else:
            holder.close()
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32), sr
