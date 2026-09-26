"""Manifest parsing and path handling — the parts that read untrusted input."""
from __future__ import annotations

import json
import os

import pytest

from continuo_expressive.jsonl import (JsonlWriter, ManifestError, done_ids,
                                        index_by_id, load_manifest, read_jsonl,
                                        resolve_audio_path, write_jsonl_atomic)


def write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_bad_json_names_the_line(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"id": "a"}\n{"id": broken}\n', encoding="utf-8")
    with pytest.raises(ManifestError) as e:
        list(read_jsonl(p))
    assert "m.jsonl:2" in str(e.value)


def test_missing_required_field_names_the_line(tmp_path):
    p = write(tmp_path / "m.jsonl", [{"wav_path": "a.wav"}, {"id": "b"}])
    with pytest.raises(ManifestError) as e:
        list(read_jsonl(p, required=("wav_path",)))
    assert "m.jsonl:2" in str(e.value)
    assert "wav_path" in str(e.value)


def test_blank_lines_are_skipped(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"id": "a"}\n\n\n{"id": "b"}\n', encoding="utf-8")
    assert [r["id"] for r in read_jsonl(p)] == ["a", "b"]


# --------------------------------------------------------------- path safety
def test_relative_path_resolves_against_root(tmp_path):
    root = tmp_path / "corpus"
    (root / "sub").mkdir(parents=True)
    clip = root / "sub" / "a.wav"
    clip.write_bytes(b"RIFF")
    assert resolve_audio_path("sub/a.wav", root) == clip.resolve()


def test_traversal_out_of_root_is_rejected(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    outside = tmp_path / "secret.wav"
    outside.write_bytes(b"RIFF")
    with pytest.raises(ManifestError, match="outside"):
        resolve_audio_path("../secret.wav", root)


def test_symlink_escaping_root_is_rejected(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    outside = tmp_path / "secret.wav"
    outside.write_bytes(b"RIFF")
    (root / "link.wav").symlink_to(outside)
    with pytest.raises(ManifestError, match="outside"):
        resolve_audio_path("link.wav", root)


def test_absolute_path_outside_root_is_rejected(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    outside = tmp_path / "secret.wav"
    outside.write_bytes(b"RIFF")
    with pytest.raises(ManifestError, match="outside"):
        resolve_audio_path(str(outside), root)


def test_no_root_allows_any_existing_file(tmp_path):
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"RIFF")
    assert resolve_audio_path(str(clip)) == clip.resolve()


def test_directory_is_not_a_clip(tmp_path):
    with pytest.raises(ManifestError, match="regular file"):
        resolve_audio_path(str(tmp_path))


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(ManifestError, match="not found"):
        resolve_audio_path(str(tmp_path / "nope.wav"))


@pytest.mark.parametrize("bad", [None, 42, "", "   ", "a\x00b"])
def test_non_string_and_nul_rejected(bad):
    with pytest.raises(ManifestError):
        resolve_audio_path(bad)


# --------------------------------------------------------------- manifests
def test_load_manifest_defaults_id_to_basename(tmp_path):
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"RIFF")
    p = write(tmp_path / "m.jsonl", [{"wav_path": str(clip)}])
    rows = load_manifest(p)
    assert rows[0]["id"] == "a.wav"
    assert rows[0]["_path"] == clip.resolve()


def test_load_manifest_skips_unusable_rows_by_default(tmp_path):
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"RIFF")
    p = write(tmp_path / "m.jsonl",
              [{"wav_path": str(clip)}, {"wav_path": str(tmp_path / "gone.wav")}])
    assert len(load_manifest(p)) == 1
    with pytest.raises(ManifestError):
        load_manifest(p, strict=True)


def test_load_manifest_limit_counts_kept_rows(tmp_path):
    for name in "abc":
        (tmp_path / f"{name}.wav").write_bytes(b"RIFF")
    p = write(tmp_path / "m.jsonl",
              [{"wav_path": str(tmp_path / f"{n}.wav")} for n in "abc"])
    assert len(load_manifest(p, limit=2)) == 2


def test_index_by_id_keeps_the_last_duplicate(tmp_path, capsys):
    p = write(tmp_path / "s.jsonl",
              [{"id": "x", "v": 1}, {"id": "x", "v": 2}, {"id": "y", "v": 3}])
    idx = index_by_id(p)
    assert idx["x"]["v"] == 2 and len(idx) == 2
    assert "duplicate" in capsys.readouterr().err


# --------------------------------------------------------------- writers
def test_writer_flushes_each_record(tmp_path):
    out = tmp_path / "out.jsonl"
    with JsonlWriter(out) as w:
        w.write({"id": "a"})
        # readable before the writer closes — this is what makes --resume work
        assert json.loads(out.read_text().strip())["id"] == "a"
        w.write({"id": "b"})
    assert len(out.read_text().strip().splitlines()) == 2


def test_writer_with_no_path_is_a_sink():
    with JsonlWriter("") as w:
        w.write({"id": "a"})          # must not raise


def test_writer_append_mode_preserves_existing(tmp_path):
    out = write(tmp_path / "out.jsonl", [{"id": "a"}])
    with JsonlWriter(out, append=True) as w:
        w.write({"id": "b"})
    assert [json.loads(l)["id"] for l in out.read_text().splitlines()] == ["a", "b"]


def test_done_ids_tolerates_a_truncated_last_line(tmp_path):
    out = tmp_path / "out.jsonl"
    out.write_text('{"id": "a"}\n{"id": "b"}\n{"id": "c", "part', encoding="utf-8")
    assert done_ids(out) == {"a", "b"}


def test_done_ids_on_missing_file_is_empty(tmp_path):
    assert done_ids(tmp_path / "nope.jsonl") == set()


def test_atomic_write_leaves_no_temp_files(tmp_path):
    out = tmp_path / "out.jsonl"
    assert write_jsonl_atomic(out, [{"id": "a"}, {"id": "b"}]) == 2
    assert [p.name for p in tmp_path.iterdir()] == ["out.jsonl"]


def test_atomic_write_keeps_the_old_file_when_the_source_raises(tmp_path):
    out = write(tmp_path / "out.jsonl", [{"id": "original"}])

    def exploding():
        yield {"id": "new"}
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        write_jsonl_atomic(out, exploding())
    assert json.loads(out.read_text().strip())["id"] == "original"
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".continuo-")]


def test_atomic_write_creates_missing_parents(tmp_path):
    out = tmp_path / "deep" / "nested" / "out.jsonl"
    write_jsonl_atomic(out, [{"id": "a"}])
    assert os.path.exists(out)


# ------------------------------------------------------------------ manifest shards
def _manifest(tmp_path, n):
    import json
    path = tmp_path / "m.jsonl"
    audio = tmp_path / "a.wav"
    import numpy as np
    import soundfile as sf
    sf.write(audio, np.zeros(1600, dtype="float32"), 16000)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"c{i}", "wav_path": str(audio)}) + "\n")
    return path


@pytest.mark.parametrize("count", [1, 2, 3, 7])
def test_shards_partition_the_manifest_exactly_once(tmp_path, count):
    from continuo_expressive.cli.annotate import build_parser
    from continuo_expressive.jsonl import load_manifest

    path = _manifest(tmp_path, 50)
    rows = load_manifest(path)
    seen = []
    for i in range(count):
        args = build_parser().parse_args(["--manifest", str(path), "--shard", f"{i}/{count}"])
        index, total = (int(x) for x in args.shard.split("/"))
        seen.extend(r["id"] for r in rows[index::total])
    assert sorted(seen) == sorted(r["id"] for r in rows)
    assert len(seen) == len(set(seen)), "a clip landed in two shards"
