"""tools/aggregate_long.py's grouping — the fold from segments to records.

Only the grouping is tested here; what a record contains is continuo_expressive.aggregate's
business and is tested there.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "aggregate_long", Path(__file__).resolve().parent.parent / "tools" / "aggregate_long.py")
aggregate_long = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(aggregate_long)


def seg(sid, parent, start, end, **extra):  # noqa: D401
    return {"id": sid, "parent_id": parent, "parent_duration": 60.0,
            "rel_start": start, "rel_end": end, "duration": end - start,
            "lang": "en", "gender": "female", **extra}


def run(tmp_path, rows, group_by):
    # the same rows serve as the manifest, which is where the transcript comes from —
    # exactly what run_long_pipeline.sh passes
    annot = tmp_path / "annot.jsonl"
    annot.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out.jsonl"
    rc = aggregate_long.main(["--annot", str(annot), "--manifest", str(annot),
                              "--out", str(out), "--group-by", group_by])
    assert rc in (0, None)
    return [json.loads(l) for l in out.read_text().splitlines() if l.strip()]


def test_default_grouping_is_one_record_per_container(tmp_path):
    rows = [seg("a1", "A", 0, 5), seg("a2", "A", 5, 10), seg("b1", "B", 0, 4)]
    recs = run(tmp_path, rows, "parent_id")
    assert sorted(r["id"] for r in recs) == ["A", "B"]
    assert "speaker" not in recs[0]


def test_a_dialogue_is_one_record_with_its_speakers_nested(tmp_path):
    """Three voices in one file must not average into one gender and one age — but the
    record is still the dialogue, one row, speakers nested under their turn tag."""
    rows = [seg("a1", "A", 0, 5, speaker="14", turn="S1", txt="hello there"),
            seg("a2", "A", 5, 10, speaker="15", turn="S2", txt="hi", gender="male"),
            seg("a3", "A", 10, 14, speaker="14", turn="S1", txt="how are you")]
    recs = run(tmp_path, rows, "parent_id,speaker")
    assert [r["id"] for r in recs] == ["A"]
    rec = recs[0]
    assert rec["n_speakers"] == 2 and rec["n_segments"] == 3
    assert set(rec["speakers"]) == {"S1", "S2"}
    s1, s2 = rec["speakers"]["S1"], rec["speakers"]["S2"]
    assert s1["speaker"] == "14" and s1["n_segments"] == 2 and s1["gender"] == "female"
    assert s2["speaker"] == "15" and s2["n_segments"] == 1 and s2["gender"] == "male"
    assert s1["txt"] == "hello there how are you"
    # the dialogue transcript is re-marked where the speaker changes, in time order
    assert rec["txt"] == "[S1] hello there [S2] hi [S1] how are you"
    assert rec["speech_seconds"] == 14.0


def test_a_row_missing_the_speaker_is_an_orphan_not_a_crash(tmp_path):
    rows = [seg("a1", "A", 0, 5, speaker="14", turn="S1"), seg("a2", "A", 5, 10)]   # second has none
    recs = run(tmp_path, rows, "parent_id,speaker")
    assert [r["id"] for r in recs] == ["A"] and list(recs[0]["speakers"]) == ["S1"]


def test_a_tagless_voice_is_left_out_not_named_after_a_tuple(tmp_path):
    rows = [seg("a_0", "a", 0.0, 2.0, speaker="0", turn="S1"),
            seg("a_1", "a", 2.0, 4.0, speaker="1", turn="S2"),
            seg("a_2", "a", 4.0, 4.9, speaker="a_spk0", turn=None)]
    out = run(tmp_path, rows, "parent_id,speaker")
    assert len(out) == 1
    assert sorted(out[0]["speakers"]) == ["S1", "S2"]
    assert out[0]["n_speakers"] == 2


def test_a_repeated_segment_is_not_counted_twice(tmp_path):
    """A resumed run with a changed worker count re-annotates rows into another shard
    file; the fold must not add the same segment's seconds twice."""
    rows = [seg("a_0", "a", 0.0, 2.0, speaker="0", turn="S1"),
            seg("a_1", "a", 2.0, 5.0, speaker="0", turn="S1")]
    a, b = tmp_path / "once", tmp_path / "twice"
    a.mkdir(); b.mkdir()
    once = run(a, rows, "parent_id,speaker")
    twice = run(b, rows + [rows[1]], "parent_id,speaker")
    assert once[0]["speakers"]["S1"]["n_segments"] == 2
    assert twice[0]["speakers"]["S1"] == once[0]["speakers"]["S1"]


def test_a_dialogue_record_keeps_where_its_audio_came_from(tmp_path):
    """Without source_tar/source_member a later pass cannot find the container again —
    run_nv_long_pipeline.sh scopes on exactly these two fields."""
    rows = [seg("a_0", "a", 0.0, 2.0, speaker="0", turn="S1",
                source_tar="continuo-00007.tar", source_member="a_dialogue_00000.m4a"),
            seg("a_1", "a", 2.0, 4.0, speaker="1", turn="S2",
                source_tar="continuo-00007.tar", source_member="a_dialogue_00000.m4a")]
    out = run(tmp_path, rows, "parent_id,speaker")[0]
    assert out["source_tar"] == "continuo-00007.tar"
    assert out["source_member"] == "a_dialogue_00000.m4a"
