"""Reading a clip out of a tar, and — more importantly — when not to.

The decode itself needs a real corpus, so what is tested here is the routing: which
source a row is read from, and the failures that would otherwise be silent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from continuo_expressive.jsonl import ManifestError
from continuo_expressive import tarsource as ts


def test_wav_path_wins_over_tar_provenance():
    """A cut manifest keeps source_tar as provenance; it must still read the file.

    Getting this backwards changes where a finished corpus is read from without saying
    so — and the two are not equivalent audio.
    """
    row = {"id": "x", "wav_path": "audio/x.mp3",
           "source_tar": "a.tar", "source_member": "x.m4a"}
    assert ts.is_tar_row(row) is False
    assert ts.is_tar_row({k: v for k, v in row.items() if k != "wav_path"}) is True


def test_row_with_neither_source_is_not_a_tar_row():
    assert ts.is_tar_row({"id": "x"}) is False


def test_relative_tar_without_a_root_is_an_error_not_a_guess():
    with pytest.raises(ManifestError, match="no tar directory"):
        ts.resolve_tar("continuo-00000.tar", None)


def test_missing_tar_says_which_one(tmp_path):
    with pytest.raises(ManifestError, match="tar not found"):
        ts.resolve_tar("nope.tar", tmp_path)


def test_absolute_tar_ignores_the_root(tmp_path):
    real = tmp_path / "here.tar"
    real.write_bytes(b"")
    assert ts.resolve_tar(str(real), tmp_path / "elsewhere") == real


def test_index_is_parsed_and_cached(tmp_path):
    tar = tmp_path / "a.tar"
    tar.write_bytes(b"")
    (tmp_path / "a.tar.idx").write_text("m1.m4a\t0\t10\nm2.m4a\t10\t20\nbroken\n")
    idx = ts.load_index(tar)
    assert idx == {"m1.m4a": (0, 10), "m2.m4a": (10, 20)}
    assert ts.load_index(tar) is idx, "the index should be read once per process"


def test_missing_index_names_the_file_it_wanted(tmp_path):
    tar = tmp_path / "b.tar"
    tar.write_bytes(b"")
    with pytest.raises(ManifestError, match=r"b\.tar\.idx"):
        ts.load_index(tar)


def test_open_handles_are_capped_and_the_evicted_one_is_closed(tmp_path, monkeypatch):
    """Bound tar handles because each open handle retains buffered data."""
    monkeypatch.setattr(ts, "_MAX_HANDLES", 2)
    monkeypatch.setattr(ts._HANDLES, "files", None, raising=False)
    tars = []
    for i in range(4):
        t = tmp_path / f"h{i}.tar"
        t.write_bytes(b"0123456789")
        tars.append(t)
        ts.read_member(t, 0, 4)

    cache = ts._HANDLES.files
    assert len(cache) == 2, "the cache must not grow with the number of tars touched"
    assert [Path(k).name for k in cache] == ["h2.tar", "h3.tar"]
    assert all(not f.closed for f in cache.values())


def test_a_reused_tar_keeps_its_handle_rather_than_reopening(tmp_path, monkeypatch):
    """The manifest is grouped by tar, so the common case is a hit — capping handles
    must not turn a sequential read into an open() per clip."""
    monkeypatch.setattr(ts, "_MAX_HANDLES", 2)
    monkeypatch.setattr(ts._HANDLES, "files", None, raising=False)
    tar = tmp_path / "same.tar"
    tar.write_bytes(b"0123456789")
    ts.read_member(tar, 0, 4)
    first = ts._HANDLES.files[str(tar)]
    for _ in range(5):
        ts.read_member(tar, 2, 4)
    assert ts._HANDLES.files[str(tar)] is first


def test_a_short_read_is_an_error(tmp_path):
    """A truncated tar must not silently yield a truncated clip."""
    tar = tmp_path / "c.tar"
    tar.write_bytes(b"0123456789")
    with pytest.raises(ManifestError, match="read 5 bytes"):
        ts.read_member(tar, 5, 20)


def test_a_window_past_the_end_raises_rather_than_returning_nothing(monkeypatch, tmp_path):
    """A turn whose sidecar timestamp begins after the audio ends. The empty array used
    to reach PANNs, which padded a batch to a longest length of 0 and died in the model;
    the worker was then restarted onto the same row forever."""
    import numpy as np

    from continuo_expressive import tarsource
    from continuo_expressive.jsonl import ManifestError

    wav = np.zeros(16000, dtype="float32")          # exactly 1.0 s
    monkeypatch.setattr(tarsource, "read_member", lambda *a, **k: b"x")
    monkeypatch.setattr(tarsource, "decode_bytes", lambda *a, **k: wav)
    monkeypatch.setattr(tarsource, "resolve_tar", lambda *a, **k: tmp_path / "t.tar")
    row = {"id": "s1", "source_tar": "t.tar", "source_member": "m.m4a",
           "tar_offset": 0, "tar_size": 1, "rel_start": 2.0, "rel_end": 3.0}
    with pytest.raises(ManifestError, match="outside"):
        tarsource.load_row(dict(row))
    # and a window that merely overruns still returns the part that exists
    row["rel_start"], row["rel_end"] = 0.5, 9.0
    assert len(tarsource.load_row(dict(row))) == 8000
