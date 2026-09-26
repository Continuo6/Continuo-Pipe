"""Folding NV back onto a dialogue: the file's totals, and each voice's own.

A dialogue record nests its speakers, and a nonverbal vocalization belongs to whoever
made it — a laugh among six voices attributed to the file says almost nothing. So the
fold runs twice, and the two must agree: every speaker's tags sum to the file's.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "merge_nv", Path(__file__).resolve().parents[1] / "tools" / "merge_nv.py")
merge_nv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge_nv)


def seg(parent, start, end, tags, turn=None, text=""):
    r = {"id": f"{parent}_{int(start):04d}", "parent_id": parent,
         "rel_start": start, "rel_end": end, "nv_tags": tags, "nv_verified": tags,
         "nv_text": text or (f"[{tags[0]}]hi" if tags else "hi")}
    if turn is not None:
        r["turn"] = turn
    return r


def dialogue_record(rid="d1", tags=("S1", "S2")):
    return {"id": rid, "n_speakers": len(tags),
            "speakers": {t: {"speaker": str(i), "gender": "female"}
                         for i, t in enumerate(tags)}}


def run(tmp_path, records, segments, extra=(), skeleton=None):
    rec_f = tmp_path / "records.jsonl"
    nv_f = tmp_path / "nv.jsonl"
    out_f = tmp_path / "out.jsonl"
    rec_f.write_text("".join(json.dumps(r) + "\n" for r in records))
    nv_f.write_text("".join(json.dumps(r) + "\n" for r in segments))
    if skeleton is not None:
        seg_f = tmp_path / "manifest.jsonl"
        seg_f.write_text("".join(json.dumps(r) + "\n" for r in skeleton))
        extra = (*extra, "--segments", str(seg_f))
    rc = merge_nv.main(["--records", str(rec_f), "--nv", str(nv_f),
                        "--out", str(out_f), "--by", "parent_id", *extra])
    assert rc in (0, None)
    return [json.loads(l) for l in out_f.read_text().splitlines() if l.strip()]


def test_tags_land_on_the_speaker_who_made_them(tmp_path):
    segs = [seg("d1", 0.0, 2.0, ["Laughter"], turn="S1"),
            seg("d1", 3.0, 5.0, [], turn="S2"),
            seg("d1", 6.0, 8.0, ["Cough"], turn="S2"),
            seg("d1", 9.0, 11.0, ["Laughter"], turn="S2")]
    out = run(tmp_path, [dialogue_record()], segs)[0]
    assert out["n_nv"] == 3 and out["nv_counts"] == {"Laughter": 2, "Cough": 1}
    s1, s2 = out["speakers"]["S1"], out["speakers"]["S2"]
    assert s1["nv_counts"] == {"Laughter": 1} and s1["n_nv"] == 1
    assert s2["nv_counts"] == {"Cough": 1, "Laughter": 1} and s2["n_nv"] == 2
    # the speaker's own attributes survive the fold
    assert s1["gender"] == "female"
    # segment counts are per voice, and the file's is the whole conversation
    assert s1["nv_segments"] == 1 and s2["nv_segments"] == 3 and out["nv_segments"] == 4


def test_the_two_folds_agree(tmp_path):
    segs = [seg("d1", float(i), i + 1.0, ["Laughter"] if i % 2 else ["Sigh"],
                turn=f"S{i % 3 + 1}") for i in range(12)]
    out = run(tmp_path, [dialogue_record(tags=("S1", "S2", "S3"))], segs)[0]
    total = {}
    for spk in out["speakers"].values():
        for k, v in spk["nv_counts"].items():
            total[k] = total.get(k, 0) + v
    assert total == out["nv_counts"]
    assert sum(s["n_nv"] for s in out["speakers"].values()) == out["n_nv"]


def test_a_file_level_span_names_its_speaker(tmp_path):
    out = run(tmp_path, [dialogue_record()],
              [seg("d1", 1.0, 2.0, ["Cough"], turn="S2")])[0]
    assert out["nv_spans"][0]["speaker"] == "S2"
    assert out["speakers"]["S2"]["nv_spans"][0]["start"] == 1.0
    assert out["speakers"]["S1"]["nv_spans"] == []


def test_a_long_recording_is_untouched(tmp_path):
    """No `speakers` on the record: exactly the old behaviour, no per-speaker keys."""
    out = run(tmp_path, [{"id": "r1", "file_seconds": 60.0}],
              [seg("r1", 1.0, 2.0, ["Laughter"])])[0]
    assert out["n_nv"] == 1 and "speakers" not in out
    assert "speaker" not in out["nv_spans"][0]


def test_rows_without_a_turn_are_reported_not_silently_empty(tmp_path, capsys):
    """The failure mode this guards against: annotating without --carry ...,turn and
    getting a corpus of dialogues whose speakers all read `n_nv: 0`."""
    out = run(tmp_path, [dialogue_record()], [seg("d1", 1.0, 2.0, ["Cough"])])[0]
    assert out["n_nv"] == 1, "the file still gets its tags"
    assert "n_nv" not in out["speakers"]["S1"], "but no empty per-speaker fields"
    assert "carry" in capsys.readouterr().err


def test_an_unknown_turn_tag_is_reported(tmp_path, capsys):
    out = run(tmp_path, [dialogue_record()],
              [seg("d1", 1.0, 2.0, ["Cough"], turn="S9")])[0]
    assert out["n_nv"] == 1
    assert out["speakers"]["S1"]["n_nv"] == 0 and out["speakers"]["S2"]["n_nv"] == 0
    assert "S9" in capsys.readouterr().err


def test_excluded_ids_clear_the_speaker_fields_too(tmp_path):
    ex = tmp_path / "ex.txt"
    ex.write_text("d1\n")
    out = run(tmp_path, [dialogue_record()],
              [seg("d1", 1.0, 2.0, ["Cough"], turn="S1")],
              extra=["--exclude-ids", str(ex)])[0]
    assert out["n_nv"] == 0 and out["speakers"]["S1"]["n_nv"] == 0


def test_dropped_tags_are_dropped_on_both_levels(tmp_path):
    segs = [seg("d1", 1.0, 2.0, ["Cough"], turn="S1"),
            seg("d1", 3.0, 4.0, ["Laughter"], turn="S1")]
    out = run(tmp_path, [dialogue_record()], segs, extra=["--drop-tags", "Cough"])[0]
    assert out["nv_counts"] == {"Laughter": 1}
    assert out["speakers"]["S1"]["nv_counts"] == {"Laughter": 1}


def man(sid, parent, start, turn, txt):
    return {"id": sid, "parent_id": parent, "rel_start": start, "turn": turn, "txt": txt}


def test_the_tags_are_woven_back_into_the_dialogue_text(tmp_path):
    """A span list says a laugh happened at 3.0 s; the training set wants the line."""
    segs = [seg("d1", 0.0, 2.0, [], turn="S1"),
            seg("d1", 3.0, 5.0, ["Laughter"], turn="S2", text="yeah [Laughter] right"),
            seg("d1", 6.0, 8.0, [], turn="S1")]
    skeleton = [man("d1_0000", "d1", 0.0, "S1", "hello there"),
                man("d1_0003", "d1", 3.0, "S2", "Yeah, right."),
                man("d1_0006", "d1", 6.0, "S1", "ok then")]
    # the ids the nv rows carry must match the manifest's
    for r, m in zip(segs, skeleton):
        r["id"] = m["id"]
    out = run(tmp_path, [dialogue_record()], segs, skeleton=skeleton)[0]
    assert out["nv_txt"] == "[S1] hello there [S2] yeah [Laughter] right [S1] ok then"
    # the corpus transcript is left exactly as it was
    assert out.get("txt") == dialogue_record().get("txt")


def test_a_recording_with_no_tag_gets_no_nv_txt(tmp_path):
    segs = [seg("d1", 0.0, 2.0, [], turn="S1")]
    skeleton = [man("d1_0000", "d1", 0.0, "S1", "hello there")]
    segs[0]["id"] = skeleton[0]["id"]
    out = run(tmp_path, [dialogue_record()], segs, skeleton=skeleton)[0]
    assert "nv_txt" not in out, "a copy of txt under another name helps nobody"


def test_a_long_recording_gets_the_text_without_speaker_marks(tmp_path):
    segs = [seg("r1", 0.0, 2.0, ["Cough"], text="ahem [Cough] so")]
    skeleton = [man("r1_0000", "r1", 0.0, None, "Ahem, so.")]
    segs[0]["id"] = skeleton[0]["id"]
    out = run(tmp_path, [{"id": "r1"}], segs, skeleton=skeleton)[0]
    assert out["nv_txt"] == "ahem [Cough] so"


def test_a_dropped_tag_does_not_leave_its_text_behind(tmp_path):
    segs = [seg("d1", 0.0, 2.0, ["Cough"], turn="S1", text="hi [Cough] there")]
    skeleton = [man("d1_0000", "d1", 0.0, "S1", "Hi there.")]
    segs[0]["id"] = skeleton[0]["id"]
    out = run(tmp_path, [dialogue_record()], segs, extra=["--drop-tags", "Cough"],
              skeleton=skeleton)[0]
    assert "nv_txt" not in out and out["n_nv"] == 0
