from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "filter_cough_panns", Path(__file__).resolve().parents[1] / "tools" /
    "filter_cough_panns.py")
filter_cough_panns = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(filter_cough_panns)


def audio(serial: int, cid: str) -> str:
    return f"export/{serial:07d}_{cid}.m4a"


def score(cid: str, cough: float, throat: float) -> dict:
    top = sorted([["Cough", cough], ["Throat clearing", throat], ["Speech", 0.8]],
                 key=lambda item: item[1], reverse=True)
    return {"id": cid, "top": top}


def test_low_cough_is_removed_but_other_nv_survives(tmp_path):
    src = tmp_path / "in.json"
    scores = tmp_path / "scores.jsonl"
    out = tmp_path / "out.json"
    report = tmp_path / "report.json"
    evidence = tmp_path / "evidence.jsonl"
    rows = [
        {"audio_path": audio(1, "low_only"), "transcript": "hi [Cough] there"},
        {"audio_path": audio(2, "low_mixed"),
         "transcript": "[Laughter] hi [Cough] there"},
        {"audio_path": audio(3, "high"), "transcript": "hi[Cough]there"},
        {"audio_path": audio(4, "other"), "transcript": "hi [Sigh]"},
    ]
    src.write_text(json.dumps(rows))
    scores.write_text("".join(json.dumps(r) + "\n" for r in [
        score("low_only", 0.001, 0.002), score("low_mixed", 0.01, 0.02),
        score("high", 0.01, 0.04)]))

    rc = filter_cough_panns.main([
        "--input", str(src), "--scores", str(scores), "--out", str(out),
        "--report", str(report), "--evidence", str(evidence), "--threshold", "0.03"])

    assert rc == 0
    got = json.loads(out.read_text())
    assert [r["audio_path"] for r in got] == [audio(2, "low_mixed"), audio(3, "high"),
                                                audio(4, "other")]
    assert got[0]["transcript"] == "[Laughter] hi there"
    assert got[1]["transcript"] == "hi[Cough]there"
    rep = json.loads(report.read_text())
    assert rep["cough_records_kept"] == 1
    assert rep["cough_records_rejected"] == 2
    assert rep["records_dropped"] == 1
    assert rep["records_retained_without_cough"] == 1
    assert len(evidence.read_text().splitlines()) == 3


def test_missing_score_fails_closed_without_writing_output(tmp_path):
    src = tmp_path / "in.json"
    scores = tmp_path / "scores.jsonl"
    out = tmp_path / "out.json"
    report = tmp_path / "report.json"
    src.write_text(json.dumps([
        {"audio_path": audio(1, "missing"), "transcript": "[Cough] hi"}]))
    scores.write_text("")

    with pytest.raises(SystemExit, match="no PANNs score"):
        filter_cough_panns.main([
            "--input", str(src), "--scores", str(scores), "--out", str(out),
            "--report", str(report)])
    assert not out.exists() and not report.exists()


def test_top3_scores_are_rejected_because_absence_is_not_zero(tmp_path):
    scores = tmp_path / "scores.jsonl"
    scores.write_text(json.dumps({"id": "x", "top": [["Speech", 0.9],
                                                        ["Cough", 0.1]]}) + "\n")
    with pytest.raises(ValueError, match="topk 527"):
        filter_cough_panns.load_scores(scores)
