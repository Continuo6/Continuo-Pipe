"""The context-varying tier: run-length encoding, smoothing, and re-measurement.

These are the parts of long-audio aggregation that are easy to get subtly wrong and
impossible to notice afterwards — a span list always *looks* plausible. Two of the
three recomputations are not averages (loudness is logarithmic, speaking rate is a
ratio), and the smoothing pass is a fixed-point loop that has to converge.
"""
from __future__ import annotations

import pytest

from continuo_expressive.aggregate import (DEFAULT_MAX_GAP, combine_lufs, mean_scores,
                                            seconds, spans, _emotion_label,
                                            _emotion_of_span, _speed_of_span,
                                            _volume_of_span)


def seg(start, end, **fields):
    row = {"rel_start": start, "rel_end": end, "dur_s": end - start}
    row.update(fields)
    return row


def labels(span_list, key):
    return [s[key] for s in span_list]


# ------------------------------------------------------- loudness is logarithmic
def test_lufs_combine_in_the_energy_domain_not_by_averaging_decibels():
    rows = [seg(0, 10, volume_lufs=-30.0), seg(10, 20, volume_lufs=-20.0)]
    combined = combine_lufs(rows)
    assert combined is not None
    # the arithmetic mean would be -25; in the energy domain it is -22.6, because the
    # louder half dominates what is actually heard
    assert combined != pytest.approx(-25.0, abs=0.5)
    assert combined == pytest.approx(-22.6, abs=0.05)


def test_lufs_combine_is_identity_on_a_single_segment():
    assert combine_lufs([seg(0, 5, volume_lufs=-24.3)]) == pytest.approx(-24.3, abs=1e-9)


def test_lufs_combine_weights_by_duration():
    short_loud = [seg(0, 1, volume_lufs=-10.0), seg(1, 101, volume_lufs=-40.0)]
    # 100 s of quiet must not be outvoted by 1 s of loud on a per-segment basis
    assert combine_lufs(short_loud) < -18.0


def test_lufs_combine_ignores_missing_and_infinite_values():
    rows = [seg(0, 5, volume_lufs=-24.0), seg(5, 10), seg(10, 15, volume_lufs=float("-inf"))]
    assert combine_lufs(rows) == pytest.approx(-24.0, abs=1e-9)


# -------------------------------------------------- speaking rate is a total ratio
def test_span_speed_is_total_characters_over_total_seconds():
    # 100 chars in 10 s (10 cps) then 30 chars in 1 s (30 cps): the honest rate is
    # 130/11 = 11.8, not the plain mean of the two rates (20.0)
    run = [seg(0, 10, speed_cps=10.0, lang="en"), seg(10, 11, speed_cps=30.0, lang="en")]
    out = _speed_of_span(run)
    assert out["speed_cps"] == pytest.approx(11.8, abs=0.05)
    assert out["speed_cps"] != pytest.approx(20.0, abs=1.0)


def test_span_speed_buckets_with_the_dominant_language_edges():
    # zh edges are (3.4, 5.4) and en edges (12.5, 19.8): the same 4.5 cps is "measured"
    # under zh and "slow" under en, so picking the language matters
    zh = _speed_of_span([seg(0, 10, speed_cps=4.5, lang="zh")])
    en = _speed_of_span([seg(0, 10, speed_cps=4.5, lang="en")])
    assert zh["speed"] == "measured"
    assert en["speed"] == "slow"


def test_span_speed_has_no_bucket_for_a_language_with_no_edges():
    out = _speed_of_span([seg(0, 10, speed_cps=9.0, lang="es")])
    assert out["speed"] is None
    assert out["speed_cps"] == pytest.approx(9.0)     # the rate is still reported


# ------------------------------------------------------------- run-length encoding
def test_adjacent_same_label_segments_become_one_span():
    rows = [seg(i * 3.0, i * 3.0 + 2.5, volume="normal", volume_lufs=-24.0)
            for i in range(6)]
    out = spans(rows, lambda r: r.get("volume"), _volume_of_span, "volume")
    assert len(out) == 1
    assert out[0]["n_seg"] == 6
    assert out[0]["start"] == 0.0 and out[0]["end"] == 17.5


def test_a_label_change_starts_a_new_span():
    rows = ([seg(i * 3.0, i * 3.0 + 2.5, volume="soft", volume_lufs=-30.0) for i in range(4)]
            + [seg(12 + i * 3.0, 12 + i * 3.0 + 2.5, volume="loud", volume_lufs=-15.0)
               for i in range(4)])
    out = spans(rows, lambda r: r.get("volume"), _volume_of_span, "volume")
    assert labels(out, "volume") == ["soft", "loud"]


def test_silence_wider_than_max_gap_breaks_a_span():
    rows = [seg(0, 5, volume="normal", volume_lufs=-24.0),
            seg(60, 65, volume="normal", volume_lufs=-24.0)]      # 55 s of nothing
    out = spans(rows, lambda r: r.get("volume"), _volume_of_span, "volume", max_gap=DEFAULT_MAX_GAP)
    assert len(out) == 2, "a minute of silence is not one continuous stretch"


def test_a_narrow_gap_does_not_break_a_span():
    rows = [seg(0, 5, volume="normal", volume_lufs=-24.0),
            seg(6, 11, volume="normal", volume_lufs=-24.0)]       # 1 s pause
    out = spans(rows, lambda r: r.get("volume"), _volume_of_span, "volume", max_gap=2.0)
    assert len(out) == 1


def test_spans_partition_the_segments_and_do_not_overlap():
    rows = [seg(i * 3.0, i * 3.0 + 2.5,
                volume="soft" if i % 3 else "loud",
                volume_lufs=-30.0 if i % 3 else -15.0) for i in range(12)]
    out = spans(rows, lambda r: r.get("volume"), _volume_of_span, "volume", min_span_seconds=0.0)
    assert sum(s["n_seg"] for s in out) == len(rows)
    for left, right in zip(out, out[1:]):
        assert left["end"] <= right["start"]


# -------------------------------------------------------------------- smoothing
def test_thrashing_labels_collapse_to_a_single_span():
    # A-B-A-B-A on 2.5 s segments is what per-clip noise looks like, not variation
    rows = [seg(i * 3.0, i * 3.0 + 2.5,
                volume="soft" if i % 2 else "normal",
                volume_lufs=-28.0 if i % 2 else -24.0) for i in range(5)]
    out = spans(rows, lambda r: r.get("volume"), _volume_of_span, "volume", min_span_seconds=4.0)
    assert len(out) == 1, f"expected the flicker to be smoothed away, got {out}"
    assert out[0]["n_seg"] == 5


def test_a_real_change_survives_smoothing():
    rows = ([seg(i * 3.0, i * 3.0 + 2.5, volume="soft", volume_lufs=-30.0) for i in range(6)]
            + [seg(18 + i * 3.0, 18 + i * 3.0 + 2.5, volume="loud", volume_lufs=-15.0)
               for i in range(6)])
    out = spans(rows, lambda r: r.get("volume"), _volume_of_span, "volume", min_span_seconds=4.0)
    assert labels(out, "volume") == ["soft", "loud"]


def test_a_short_run_is_absorbed_into_the_longer_neighbour():
    rows = ([seg(i * 3.0, i * 3.0 + 2.5, volume="soft", volume_lufs=-30.0) for i in range(6)]
            + [seg(18.0, 19.0, volume="loud", volume_lufs=-15.0)]        # 1 s blip
            + [seg(21 + i * 3.0, 21 + i * 3.0 + 2.5, volume="soft", volume_lufs=-30.0)
               for i in range(2)])
    out = spans(rows, lambda r: r.get("volume"), _volume_of_span, "volume", min_span_seconds=4.0)
    assert len(out) == 1 and out[0]["n_seg"] == 9


def test_absorbing_rightwards_still_coalesces_with_the_run_on_the_left():
    # short run, blip, long run: the blip is absorbed into the *right* neighbour
    # because that one is longer. A run identified by its first member's label would
    # then read as "soft", stop matching the "normal" run on its left, and leave two
    # adjacent identical spans where there is only one stretch of unchanging volume.
    rows = ([seg(0.0, 2.5, volume="normal", volume_lufs=-23.0),
             seg(3.0, 5.5, volume="normal", volume_lufs=-23.0),
             seg(6.0, 7.0, volume="soft", volume_lufs=-30.0)]
            + [seg(8.0 + i * 3.0, 8.0 + i * 3.0 + 2.5, volume="normal", volume_lufs=-23.0)
               for i in range(8)])
    out = spans(rows, lambda r: r.get("volume"), _volume_of_span, "volume", min_span_seconds=4.0)
    assert len(out) == 1, f"one unchanging stretch, got {[s['volume'] for s in out]}"
    assert out[0]["n_seg"] == len(rows)


def test_an_isolated_short_utterance_is_kept_not_smoothed():
    # silence on both sides means this is a lone utterance, not a label flickering
    # inside a longer stretch — smoothing must not reach across the gap to erase it
    rows = ([seg(i * 3.0, i * 3.0 + 2.5, volume="soft", volume_lufs=-30.0) for i in range(4)]
            + [seg(120.0, 121.0, volume="loud", volume_lufs=-15.0)])
    out = spans(rows, lambda r: r.get("volume"), _volume_of_span, "volume", min_span_seconds=4.0)
    assert len(out) == 2
    assert out[1]["n_seg"] == 1 and out[1]["seconds"] == pytest.approx(1.0)


def test_smoothing_terminates_on_pathological_input():
    rows = [seg(i * 1.0, i * 1.0 + 0.9, volume=("a", "b", "c")[i % 3], volume_lufs=-24.0)
            for i in range(60)]
    out = spans(rows, lambda r: r.get("volume"), _volume_of_span, "volume", min_span_seconds=30.0)
    assert sum(s["n_seg"] for s in out) == 60


# ------------------------------------------------------ re-measurement wins the tie
def test_neighbours_that_agree_after_re_measurement_are_merged():
    # the middle run is loud enough per segment to be its own run and long enough to
    # survive smoothing, but once the quiet blip beside it is absorbed its combined
    # loudness lands back where its neighbours are. Two boundaries would then sit on
    # the timeline with nothing changing across them.
    rows = ([seg(i * 3.0, i * 3.0 + 2.5, volume="normal", volume_lufs=-22.0) for i in range(6)]
            + [seg(18.0, 19.0, volume="soft", volume_lufs=-40.0)]          # 1 s blip
            + [seg(20 + i * 3.0, 20 + i * 3.0 + 2.5, volume="normal", volume_lufs=-22.0)
               for i in range(6)])
    out = spans(rows, lambda r: r.get("volume"), _volume_of_span, "volume",
                min_span_seconds=4.0)
    assert [s["volume"] for s in out] == ["normal"]
    assert out[0]["n_seg"] == len(rows)


def test_a_timeline_never_has_two_adjacent_spans_with_the_same_label():
    rows = ([seg(i * 3.0, i * 3.0 + 2.5, volume="normal", volume_lufs=-22.0) for i in range(5)]
            + [seg(15 + i * 3.0, 15 + i * 3.0 + 2.5, volume="loud", volume_lufs=-12.0)
               for i in range(5)]
            + [seg(30 + i * 3.0, 30 + i * 3.0 + 2.5, volume="normal", volume_lufs=-22.0)
               for i in range(5)])
    out = spans(rows, lambda r: r.get("volume"), _volume_of_span, "volume")
    assert [s["volume"] for s in out] == ["normal", "loud", "normal"]
    for left, right in zip(out, out[1:]):
        assert left["volume"] != right["volume"]


def test_the_span_is_rebucketed_from_its_own_measurement():
    # every member reads "soft" per-segment, but the span's combined loudness is
    # "normal": the span rests on more audio, so its verdict is the published one
    rows = [seg(0, 10, volume="soft", volume_lufs=-27.5),
            seg(10, 20, volume="soft", volume_lufs=-27.5),
            seg(20, 30, volume="soft", volume_lufs=-19.5)]
    out = spans(rows, lambda r: r.get("volume"), _volume_of_span, "volume", min_span_seconds=0.0)
    assert len(out) == 1
    assert out[0]["volume"] == "normal"
    assert out[0]["volume_lufs"] == pytest.approx(-23.0, abs=0.3)


# -------------------------------------------------------------------- emotion
def test_a_span_inherits_the_verdict_of_the_clips_that_passed_the_gate():
    # one confident segment plus four that fall short: the label is earned at clip
    # level and then covers the stretch it belongs to
    run = ([seg(0, 3, emotion_scores={"happy": 0.85, "neutral": 0.15})]
           + [seg(3 + i * 3, 6 + i * 3, emotion_scores={"happy": 0.55, "neutral": 0.45})
              for i in range(4)])
    out = _emotion_of_span(run, tau=0.7)
    assert out["emotion"] == "happy"
    assert out["emotion_support"] == pytest.approx(3 / 15, abs=0.01)


def test_averaging_the_posteriors_first_would_have_gated_everything_away():
    # the design this replaced. Every member peaks below tau and their mean does too,
    # so thresholding the average yields nothing — while the clip-level gate keeps the
    # one segment that earned it.
    run = [seg(i * 3, i * 3 + 3, emotion_scores={"sad": 0.55, "neutral": 0.45})
           for i in range(4)] + [seg(12, 15, emotion_scores={"sad": 0.9, "neutral": 0.1})]
    averaged = mean_scores(run)
    assert max(averaged.values()) < 0.7
    assert _emotion_of_span(run, tau=0.7)["emotion"] == "sad"


def test_a_span_where_nothing_passed_reports_null_but_keeps_its_posterior():
    run = [seg(i * 3, i * 3 + 3, emotion_scores={"sad": 0.5, "neutral": 0.5})
           for i in range(4)]
    out = _emotion_of_span(run, tau=0.7)
    assert out["emotion"] is None
    assert out["emotion_support"] == 0.0
    assert out["emotion_top3"], "a null label still has to be explainable"


def test_survivors_vote_by_duration_not_by_count():
    run = [seg(0, 1, emotion_scores={"happy": 0.95, "neutral": 0.05}),
           seg(1, 2, emotion_scores={"happy": 0.95, "neutral": 0.05}),
           seg(2, 62, emotion_scores={"sad": 0.9, "neutral": 0.1})]
    assert _emotion_of_span(run, tau=0.7)["emotion"] == "sad"


def test_emotion_run_length_uses_the_ungated_argmax():
    # gating each 2.5 s clip would blank most of them and leave a timeline of holes;
    # the threshold belongs at span level, after averaging
    row = seg(0, 2.5, emotion=None, emotion_scores={"happy": 0.42, "neutral": 0.38})
    assert _emotion_label(row) == "happy"


def test_emotion_label_falls_back_to_the_gated_field_without_raw_scores():
    assert _emotion_label(seg(0, 2.5, emotion="sad")) == "sad"


# -------------------------------------------------------------------- duration
def test_seconds_prefers_the_measured_duration_over_the_manifest_claim():
    assert seconds({"dur_s": 2.0, "duration": 9.0, "rel_start": 0, "rel_end": 9}) == 2.0
    assert seconds({"duration": 9.0}) == 9.0
    assert seconds({"rel_start": 4.0, "rel_end": 6.5}) == pytest.approx(2.5)
    assert seconds({}) == 0.0
