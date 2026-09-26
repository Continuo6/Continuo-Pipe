from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "export_selected_audio", Path(__file__).resolve().parents[1] / "tools" /
    "export_selected_audio.py")
export_selected_audio = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_selected_audio)


def test_preserves_recorded_sparse_path_and_reuses_identical_audio(tmp_path):
    root = tmp_path / "audio"
    tar_root = tmp_path / "tars"
    tar_root.mkdir()
    payload = b"\x00\x00\x00\x18ftypM4A test payload"
    prefix = b"unused-prefix"
    (tar_root / "x.tar").write_bytes(prefix + payload)
    selection = tmp_path / "selection.json"
    destination = root / "batch_0096" / "0476123_clip_a.m4a"
    selection.write_text(json.dumps([
        {"audio_path": str(destination), "transcript": "[Cough]"}]))
    sources = tmp_path / "sources.jsonl"
    sources.write_text(json.dumps({
        "id": "clip_a", "source_tar": "x.tar", "source_member": "original.m4a",
        "tar_offset": len(prefix), "tar_size": len(payload)}) + "\n")
    report = tmp_path / "report.json"
    argv = ["--selection", str(selection), "--sources", str(sources),
            "--output-root", str(root), "--tar-root", str(tar_root),
            "--report", str(report), "--workers", "1"]

    assert export_selected_audio.main(argv) == 0
    assert destination.read_bytes() == payload
    assert json.loads(report.read_text())["created"] == 1
    assert export_selected_audio.main(argv) == 0
    assert json.loads(report.read_text())["reused"] == 1


def test_rejects_recorded_path_outside_output_root(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps([
        {"audio_path": str(tmp_path / "elsewhere" / "0000001_x.m4a")}]))
    with pytest.raises(ValueError, match="outside --output-root"):
        export_selected_audio.load_selection(selection, tmp_path / "audio")
