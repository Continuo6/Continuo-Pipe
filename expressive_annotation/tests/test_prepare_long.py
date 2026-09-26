"""Segment construction for long containers: source fusion, language, windowing.

``tools/prepare_long.py`` decides which boundaries a long recording is cut on and what
metadata each piece inherits. Getting that wrong is invisible downstream — the manifest
still looks like a manifest — so the fusion rules are pinned here. The audio path
(ffmpeg, FLAC writing) is left to the end-to-end run; everything below is pure.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "prepare_long", Path(__file__).resolve().parent.parent / "tools" / "prepare_long.py")
prepare_long = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare_long)


def short_entry(start, end, text="hi", language="de", dnsmos=3.0):
    return {"rel_start": start, "rel_end": end, "text": text,
            "language": language, "dnsmos": dnsmos}


def member(start, end, text="hi"):
    return {"start": start, "end": end, "text": text}


# ------------------------------------------------------------------- overlap math
@pytest.mark.parametrize("a,b,expected", [
    ((0.0, 10.0), (0.0, 10.0), 1.0),
    ((0.0, 10.0), (10.0, 20.0), 0.0),
    ((0.0, 10.0), (20.0, 30.0), 0.0),
    ((0.0, 10.0), (0.0, 20.0), 0.5),
])
def test_iou(a, b, expected):
    assert prepare_long.iou(a, b) == pytest.approx(expected)


def test_overlap_frac_is_asymmetric():
    assert prepare_long.overlap_frac((0.0, 10.0), (0.0, 5.0)) == pytest.approx(0.5)
    assert prepare_long.overlap_frac((0.0, 5.0), (0.0, 10.0)) == pytest.approx(1.0)


# ---------------------------------------------------------------- source fusion
def test_short_mode_uses_only_the_containers_own_view():
    local = {"short": [short_entry(0, 2), short_entry(5, 7)]}
    meta = {"members": [member(0, 2), member(5, 7), member(9, 12)]}
    out = prepare_long.build_segments(local, meta, "short")
    assert len(out) == 2
    assert all(s["seg_source"] == "short" for s in out)


def test_metainfo_mode_takes_the_superset_of_boundaries():
    local = {"short": [short_entry(0, 2)]}
    meta = {"members": [member(0, 2), member(5, 7), member(9, 12)]}
    out = prepare_long.build_segments(local, meta, "metainfo")
    assert len(out) == 3
    assert all(s["seg_source"] == "metainfo" for s in out)


def test_an_overlapping_short_entry_donates_its_metadata_to_the_member():
    # the same utterance, boundaries a hair apart: metainfo has no language of its own
    # and must inherit the one the short view recorded
    local = {"short": [short_entry(0.0, 2.0, language="zh", dnsmos=3.7)]}
    meta = {"members": [member(0.05, 2.05)]}
    out = prepare_long.build_segments(local, meta, "union")
    assert out[0]["lang"] == "zh"
    assert out[0]["dnsmos"] == pytest.approx(3.7)


def test_a_barely_overlapping_short_entry_donates_nothing():
    local = {"short": [short_entry(0.0, 2.0, language="zh")]}
    meta = {"members": [member(1.9, 8.0)]}          # IoU well under 0.5
    out = prepare_long.build_segments(local, meta, "union")
    member_row = next(s for s in out if s["seg_source"] == "metainfo")
    assert member_row["lang"] is None


def test_union_re_adds_short_entries_metainfo_does_not_cover():
    local = {"short": [short_entry(0, 2), short_entry(50, 52, language="en")]}
    meta = {"members": [member(0, 2)]}
    out = prepare_long.build_segments(local, meta, "union")
    assert len(out) == 2
    orphan = next(s for s in out if s["start"] == 50)
    assert orphan["seg_source"] == "short" and orphan["lang"] == "en"


def test_union_does_not_re_add_a_short_entry_that_metainfo_already_covers():
    # re-adding it would overlap a member and count the same audio twice
    local = {"short": [short_entry(1.0, 3.0)]}
    meta = {"members": [member(0.0, 8.0)]}
    out = prepare_long.build_segments(local, meta, "union")
    assert len(out) == 1 and out[0]["seg_source"] == "metainfo"


def test_missing_metainfo_falls_back_to_the_short_view():
    local = {"short": [short_entry(0, 2)]}
    assert len(prepare_long.build_segments(local, None, "union")) == 1


def test_segments_come_back_in_time_order():
    local = {"short": []}
    meta = {"members": [member(9, 12), member(0, 2), member(5, 7)]}
    out = prepare_long.build_segments(local, meta, "metainfo")
    assert [s["start"] for s in out] == [0, 5, 9]


# -------------------------------------------------------------------- languages
def test_a_matched_segment_keeps_its_own_language():
    segments = [{"start": 0, "end": 2, "lang": "de"}]
    prepare_long.assign_languages(segments, ["de"])
    assert (segments[0]["lang"], segments[0]["lang_source"]) == ("de", "segment")


def test_an_unmatched_segment_borrows_from_the_nearest_neighbour_in_time():
    segments = [{"start": 0, "end": 2, "lang": "zh"},
                {"start": 3, "end": 5, "lang": None},
                {"start": 200, "end": 202, "lang": "en"}]
    prepare_long.assign_languages(segments, ["zh", "en"])
    assert segments[1]["lang"] == "zh"
    assert segments[1]["lang_source"] == "neighbour"


def test_the_file_language_is_used_only_when_it_is_unambiguous():
    segments = [{"start": 0, "end": 2, "lang": None}]
    prepare_long.assign_languages(segments, ["de"])
    assert (segments[0]["lang"], segments[0]["lang_source"]) == ("de", "file")


def test_a_multilingual_file_never_stamps_one_language_on_an_unmatched_segment():
    # 37% of long containers list more than one language; that list is a file-level
    # fact and using it per segment would route the wrong accent head and pick the
    # wrong speed edges. None is the honest answer.
    segments = [{"start": 0, "end": 2, "lang": None}]
    prepare_long.assign_languages(segments, ["zh", "en"])
    assert (segments[0]["lang"], segments[0]["lang_source"]) == (None, "none")


def test_no_language_anywhere_leaves_it_null():
    segments = [{"start": 0, "end": 2, "lang": None}]
    prepare_long.assign_languages(segments, [])
    assert segments[0]["lang"] is None


# ---------------------------------------------------------------------- windows
def test_a_segment_within_the_cap_is_left_alone():
    out = prepare_long.split_windows([{"start": 0, "end": 10, "text": "x"}], 15.0, 15.0, 0.5)
    assert len(out) == 1 and out[0]["sub"] is None


def test_an_over_long_segment_is_cut_into_windows():
    out = prepare_long.split_windows([{"start": 0, "end": 40, "text": "x"}], 15.0, 15.0, 0.5)
    assert [(s["start"], s["end"]) for s in out] == [(0, 15), (15, 30), (30, 40)]
    assert [s["sub"] for s in out] == [0, 1, 2]


def test_the_transcript_stays_on_the_first_window_only():
    # duplicating it would multiply every window's characters-per-second by the
    # number of windows
    out = prepare_long.split_windows([{"start": 0, "end": 40, "text": "hello"}],
                                     15.0, 15.0, 0.5)
    assert [s["text"] for s in out] == ["hello", "", ""]


def test_a_sub_minimum_tail_window_is_dropped():
    out = prepare_long.split_windows([{"start": 0, "end": 30.2, "text": "x"}],
                                     15.0, 15.0, 0.5)
    assert [(s["start"], s["end"]) for s in out] == [(0, 15), (15, 30)]


def test_window_ids_stay_unique_within_a_container():
    segments = [{"start": 0, "end": 40, "text": "a"}, {"start": 40, "end": 45, "text": "b"}]
    windows = prepare_long.split_windows(segments, 15.0, 15.0, 0.5)
    ids = [prepare_long.segment_id("P", w["index"], w["sub"]) for w in windows]
    assert ids == ["P_0000.0", "P_0000.1", "P_0000.2", "P_0001"]
    assert len(set(ids)) == len(ids)


# --------------------------------------------------------------------- idx files
def test_load_idx_parses_member_offset_size(tmp_path):
    path = tmp_path / "x.tar.idx"
    path.write_text("a.m4a\t512\t100\nb.json\t1024\t50\n\n", encoding="utf-8")
    assert prepare_long.load_idx(path) == {"a.m4a": (512, 100), "b.json": (1024, 50)}


# ---------------------------------------------------------------- full recordings
def test_the_container_dir_is_opt_in():
    parser = prepare_long.build_parser()
    args = parser.parse_args(["--tars", "x.tar", "--out-dir", "o"])
    assert args.container_dir == "", "doubling the disk should never be the default"
    args = parser.parse_args(["--tars", "x.tar", "--out-dir", "o", "--container-dir", "a"])
    assert args.container_dir == "a"


def test_the_five_minute_cap_is_the_default():
    args = prepare_long.build_parser().parse_args(["--tars", "x.tar", "--out-dir", "o"])
    assert args.max_seconds == 300.0
    assert prepare_long.build_parser().parse_args(
        ["--tars", "x.tar", "--out-dir", "o", "--max-seconds", "0"]).max_seconds == 0.0


def _job(duration, max_seconds, **over):
    job = {"local": {"id": "C", "duration": duration, "short": [short_entry(0, 2)]},
           "meta": None, "out_dir": "o", "segment_sr": 16000, "segments": "short",
           "min_seconds": 0.5, "window_seconds": 15.0, "window_hop": 15.0,
           "container_dir": "", "max_seconds": max_seconds,
           "format": "flac", "quality": 0.7,
           "tar_name": "continuo-00000.tar", "audio_member": "C.m4a",
           "relative_to": "",
           "audio_tar": "x.tar", "audio_offset": 0, "audio_size": 0, "member": "C.json"}
    job.update(over)
    return job


def test_a_container_over_the_cap_is_skipped_and_counted():
    result = prepare_long.cut_container(_job(600.0, 300.0))
    assert result["rows"] == []
    assert result["too_long"] == 600.0, "the dropped duration is reported, not silent"


def test_a_container_under_the_cap_is_kept():
    # it gets as far as decoding (which fails here — there is no real tar behind the
    # stub), rather than being turned away on length. The audio path is exercised by
    # the end-to-end run.
    result = prepare_long.cut_container(_job(120.0, 300.0))
    assert result["too_long"] == 0.0
    assert result["error"] is not None, "it reached the audio stage, not the cap"


def test_the_cap_can_be_disabled():
    assert prepare_long.cut_container(_job(7261.0, 0.0))["too_long"] == 0.0


def test_a_raised_cap_is_warned_about_when_workers_could_exhaust_memory(capsys):
    # the container is decoded whole; a four-hour one is ~920 MB per worker
    prepare_long.warn_memory(max_seconds=0.0, workers=8)
    warned = capsys.readouterr().err
    assert "--workers" in warned and "memory" in warned.lower()


def test_the_default_cap_warns_about_nothing():
    import io
    import contextlib
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        prepare_long.warn_memory(max_seconds=300.0, workers=8)
    assert err.getvalue() == ""


@pytest.mark.parametrize("member,view", [
    ("rec_long_00000.json", "long"),
    ("rec_dlg_00000.json", "dialogue"),
    ("rec_001_000_spk1.json", "short"),
    ("rec_212_003_spk16.json", "short"),
])
def test_a_member_name_says_which_view_it_holds(member, view):
    assert prepare_long.member_view(member) == view


def test_long_stays_the_default_so_the_long_path_is_unchanged():
    args = prepare_long.build_parser().parse_args(["--tars", "x.tar", "--out-dir", "o"])
    assert args.container_types == "long"


def test_splitting_can_be_switched_off_for_standalone_utterances():
    # a standalone container is one utterance; splitting it yields two clips that are
    # each not one, the first with a whole transcript over fifteen seconds
    long_one = [{"start": 0, "end": 40, "text": "the whole thing"}]
    out = prepare_long.split_windows(long_one, 0.0, 0.0, 0.5)
    assert len(out) == 1
    assert out[0]["sub"] is None and out[0]["text"] == "the whole thing"


def test_an_hour_budget_can_be_set():
    args = prepare_long.build_parser().parse_args(
        ["--tars", "x.tar", "--out-dir", "o", "--max-hours", "10000"])
    assert args.max_hours == 10000.0
    assert prepare_long.build_parser().parse_args(
        ["--tars", "x.tar", "--out-dir", "o"]).max_hours == 0.0


# ------------------------------------------------------------------ audio format
@pytest.mark.parametrize("fmt,suffix", [("flac", ".flac"), ("mp3", ".mp3")])
def test_a_segment_round_trips_through_either_format(tmp_path, fmt, suffix):
    import numpy as np
    from continuo_expressive.audio import load_wav

    sr = 16000
    tone = (0.2 * np.sin(2 * np.pi * 220 * np.arange(sr * 2) / sr)).astype(np.float32)
    dest = tmp_path / f"clip{suffix}"
    prepare_long.write_audio(tone, sr, dest, fmt, quality=0.7)

    assert dest.is_file()
    assert not list(tmp_path.glob("*.part")), "a temp file was left behind"
    back = load_wav(dest)
    # mp3 pads and is lossy; the pipeline only needs it to come back as the same sound
    assert abs(len(back) - len(tone)) < sr * 0.1
    assert 0.1 < float(np.sqrt(np.mean(back[:len(tone)] ** 2))) < 0.3


def test_mp3_is_the_smaller_of_the_two(tmp_path):
    import numpy as np

    sr = 16000
    rng = np.random.default_rng(0)
    speechlike = (0.1 * np.sin(2 * np.pi * 180 * np.arange(sr * 3) / sr)
                  + 0.02 * rng.standard_normal(sr * 3)).astype(np.float32)
    prepare_long.write_audio(speechlike, sr, tmp_path / "a.flac", "flac", 0.7)
    prepare_long.write_audio(speechlike, sr, tmp_path / "a.mp3", "mp3", 0.7)
    assert (tmp_path / "a.mp3").stat().st_size < (tmp_path / "a.flac").stat().st_size


def test_flac_stays_the_default_so_nothing_silently_becomes_lossy():
    args = prepare_long.build_parser().parse_args(["--tars", "x.tar", "--out-dir", "o"])
    assert args.segment_format == "flac"


def test_a_row_records_where_its_audio_came_from(tmp_path, monkeypatch):
    # cut audio is only safely deletable if it can be cut again, and the corpus's own
    # pack index cannot do it: its member names predate the m4a migration
    import numpy as np

    monkeypatch.setattr(prepare_long, "decode",
                        lambda data, sr: np.zeros(sr * 3, dtype=np.float32))
    monkeypatch.setattr(prepare_long, "read_at", lambda *a: b"")
    job = _job(3.0, 300.0, out_dir=str(tmp_path),
               tar_name="continuo-00042.tar", audio_member="C_001_000_spk1.m4a")
    result = prepare_long.cut_container(job)
    assert result["error"] is None, result["error"]
    assert result["rows"], "the container should have yielded a segment"
    row = result["rows"][0]
    assert row["source_tar"] == "continuo-00042.tar"
    assert row["source_member"] == "C_001_000_spk1.m4a"


def test_a_tar_without_an_index_is_skipped_not_fatal(tmp_path, capsys):
    # one unusable tar ended a twelve-hour extraction on its 3510th tar of 3512
    (tmp_path / "x.tar").write_bytes(b"")
    args = prepare_long.build_parser().parse_args(
        ["--tars", str(tmp_path / "*.tar"), "--out-dir", str(tmp_path / "out")])
    assert prepare_long.build_jobs(tmp_path / "x.tar", None, tmp_path, args) == []
    assert "no .idx sidecar" in capsys.readouterr().err


def test_relative_paths_make_a_manifest_portable(tmp_path, monkeypatch):
    # an absolute wav_path only works on the machine that cut it; --audio-root exists
    # precisely so another machine can resolve a relative one against its own copy
    import numpy as np

    monkeypatch.setattr(prepare_long, "decode",
                        lambda data, sr: np.zeros(sr * 3, dtype=np.float32))
    monkeypatch.setattr(prepare_long, "read_at", lambda *a: b"")
    out = tmp_path / "run" / "segments"
    job = _job(3.0, 300.0, out_dir=str(out), relative_to=str(out.parent))
    row = prepare_long.cut_container(job)["rows"][0]
    assert not row["wav_path"].startswith("/")
    assert row["wav_path"].startswith("segments/")


# ------------------------------------------------------------------- dialogue turns
def test_a_metainfo_member_keeps_who_said_it():
    """A dialogue's metainfo members are turns. The speaker is what lets a multi-speaker
    file be folded per voice instead of per file, so it must survive segmentation."""
    meta = {"members": [{**member(0, 4), "speaker": "14", "tag": "S1", "lang_hint": "en",
                         "dnsmos": 2.8},
                        {**member(4, 9), "speaker": "15", "tag": "S2"}]}
    out = prepare_long.build_segments({"short": []}, meta, "union")
    assert [(s["speaker"], s["turn"]) for s in out] == [("14", "S1"), ("15", "S2")]
    assert out[0]["lang"] == "en" and out[0]["dnsmos"] == pytest.approx(2.8)
    assert out[1]["lang"] is None                      # no hint: assign_languages' job


def test_long_members_are_unchanged_by_the_turn_fields():
    """Long metainfo has no speaker or tag; the fields must come out None, not KeyError,
    and the row builder must then leave them off entirely."""
    out = prepare_long.build_segments({"short": []}, {"members": [member(0, 4)]}, "union")
    assert out[0]["speaker"] is None and out[0]["turn"] is None


def test_windowing_keeps_the_speaker_on_every_piece():
    segs = [{"start": 0.0, "end": 40.0, "text": "t", "lang": "en", "dnsmos": None,
             "seg_source": "metainfo", "speaker": "14", "turn": "S1"}]
    out = prepare_long.split_windows(segs, window=15.0, hop=15.0, min_seconds=0.5)
    assert len(out) > 1 and all(s["speaker"] == "14" and s["turn"] == "S1" for s in out)


def test_union_adds_no_short_strays_to_a_dialogue():
    """Dialogue members carry turn tags; the sidecar's short entries name speakers in
    another id space and have none, so union must not append them."""
    local = {"short": [short_entry(0.0, 1.0), short_entry(50.0, 51.0)]}
    meta = {"members": [{**member(0.0, 1.0), "tag": "S1", "speaker": "0"},
                        {**member(2.0, 3.0), "tag": "S2", "speaker": "1"}]}
    out = prepare_long.build_segments(local, meta, "union")
    assert [s["turn"] for s in out] == ["S1", "S2"]
    assert all(s["seg_source"] == "metainfo" for s in out)
    # a long container (no tags) still gets the uncovered short entry back
    meta_long = {"members": [member(0.0, 1.0), member(2.0, 3.0)]}
    assert len(prepare_long.build_segments(local, meta_long, "union")) == 3


def test_a_segment_past_the_container_is_not_named():
    """Do not name a segment past the recording's declared duration."""
    local = {"id": "d1", "duration": 10.0, "short": [], "languages": ["en"]}
    meta = {"duration": 10.0, "languages": ["en"], "members": [
        {"start": 0.0, "end": 4.0, "text": "a", "tag": "S1", "speaker": "0"},
        {"start": 8.0, "end": 12.0, "text": "b", "tag": "S2", "speaker": "1"},   # overruns
        {"start": 11.0, "end": 13.0, "text": "c", "tag": "S1", "speaker": "0"},  # past the end
    ]}
    job = {"segments": "metainfo", "window_seconds": 15.0, "window_hop": None,
           "min_seconds": 0.5, "max_seconds": 0, "manifest_only": True,
           "container_dir": "", "out_dir": "nonexistent", "segment_sr": 16000,
           "local": local, "meta": meta, "member": "d1_dlg_00000.m4a",
           "tar_name": "continuo-00000.tar", "audio_offset": 512, "audio_size": 1024,
           "audio_member": "d1_dlg_00000.m4a"}
    out = prepare_long.cut_container(job)
    rows = out["rows"]
    assert [r["rel_end"] for r in rows] == [4.0, 10.0], "the overrun is clamped to the container"
    assert all(r["rel_start"] < 10.0 for r in rows)
    assert any("starts past" in x.get("reason", "") for x in out["skipped"])
