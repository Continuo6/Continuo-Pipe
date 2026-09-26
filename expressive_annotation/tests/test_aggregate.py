"""The speaker-intrinsic tier, and how the two tiers assemble into one record.

Gender, accent, age and pitch belong to the voice, so they collapse to one value per
file. What is pinned here is *how* they collapse — duration weighting rather than
per-segment votes, summed probabilities rather than voted accent labels, and the age
cascade re-run on aggregated inputs rather than a mode over per-segment bands. Each of
those is a place where the cheap version gives a different, worse answer.
"""
from __future__ import annotations

import pytest

from continuo_expressive.aggregate import (aggregate, aggregate_speaker,
                                            weighted_mean, weighted_vote)


def seg(start, end, **fields):
    row = {"rel_start": start, "rel_end": end, "dur_s": end - start,
           "parent_duration": 300.0}
    row.update(fields)
    return row


# ------------------------------------------------------------ duration weighting
def test_a_vote_is_weighted_by_duration_not_by_segment_count():
    rows = [seg(0, 1, gender="female"), seg(1, 2, gender="female"),
            seg(2, 62, gender="male")]
    label, agreement = weighted_vote(rows, "gender")
    assert label == "male", "three segments, but one of them is 60 s of the 62"
    assert agreement == pytest.approx(60 / 62, abs=1e-6)


def test_agreement_is_one_when_every_labelled_second_agrees():
    rows = [seg(i * 3.0, i * 3.0 + 2.0, gender="male") for i in range(5)]
    assert weighted_vote(rows, "gender")[1] == pytest.approx(1.0)


def test_unlabelled_segments_take_no_part_in_the_vote():
    rows = [seg(0, 100), seg(100, 105, gender="female")]
    label, agreement = weighted_vote(rows, "gender")
    assert (label, agreement) == ("female", pytest.approx(1.0))


def test_a_vote_with_nothing_labelled_is_none():
    assert weighted_vote([seg(0, 5), seg(5, 10)], "gender") == (None, 0.0)


def test_weighted_mean_ignores_missing_values():
    rows = [seg(0, 10, pitch_hz=100.0), seg(10, 20), seg(20, 30, pitch_hz=200.0)]
    assert weighted_mean(rows, "pitch_hz") == pytest.approx(150.0)


# ------------------------------------------------------------------------ accent
def test_accent_is_decided_on_summed_probability_not_on_voted_labels():
    # three segments each mildly prefer "b"; one is a touch more confident about "a".
    # Voting labels would return "a" 1 - 0 on the strength of that single segment.
    rows = [seg(0, 5, accent_top3={"a": 0.40, "b": 0.35}),
            seg(5, 10, accent_top3={"b": 0.38, "a": 0.30}),
            seg(10, 15, accent_top3={"b": 0.38, "a": 0.30})]
    out = aggregate_speaker(rows)
    assert out["accent"] == "b"
    assert list(out["accent_top3"]) [0] == "b"


def test_accent_is_none_when_no_segment_carries_probabilities():
    out = aggregate_speaker([seg(0, 5, gender="male")])
    assert out["accent"] is None and out["accent_top3"] is None


# --------------------------------------------------------------------------- age
def test_the_age_cascade_is_re_run_on_aggregated_inputs():
    # per-segment bands would be young/young/middle -> a mode of "young". The cascade
    # on the aggregated years (mean 45.0) bins to "middle", which is the honest answer:
    # binning a mean is not the same as taking the mode of bins.
    rows = [seg(0, 10, gender="male", age_years=42.0, age_vox_years=42.0),
            seg(10, 20, gender="male", age_years=42.0, age_vox_years=42.0),
            seg(20, 30, gender="male", age_years=51.0, age_vox_years=51.0)]
    out = aggregate_speaker(rows)
    assert out["age_years"] == pytest.approx(45.0)
    assert out["age_band"] == "middle"


def test_the_child_branch_of_the_cascade_still_wins_after_aggregation():
    # the gender head's "child" class is the only thing that recovers children; the
    # regressions put them near 18, and no binning of that recovers them
    rows = [seg(0, 10, gender="child", age_years=18.0, age_vox_years=19.0)]
    assert aggregate_speaker(rows)["age_band"] == "child"


# ------------------------------------------------------------------------- pitch
def test_pitch_is_bucketed_against_the_aggregated_gender():
    # 130 Hz is "medium" for a male speaker and "low" for a female one, so the bucket
    # has to be taken after the gender vote, not per segment
    male = aggregate_speaker([seg(0, 10, gender="male", pitch_hz=130.0)])
    female = aggregate_speaker([seg(0, 10, gender="female", pitch_hz=130.0)])
    assert male["pitch"] == "medium"
    assert female["pitch"] == "low"


# --------------------------------------------------------------------- languages
def test_multiple_languages_are_counted_and_the_dominant_one_reported():
    rows = [seg(0, 60, lang="zh"), seg(60, 80, lang="en")]
    out = aggregate_speaker(rows)
    assert out["lang"] == "zh"
    assert out["n_languages"] == 2
    assert out["dominant_lang_ratio"] == pytest.approx(0.75)


# -------------------------------------------------------------------- assembly
def test_a_single_segment_aggregates_to_itself():
    row = seg(0, 10, gender="male", age_years=40.0, age_vox_years=40.0,
              pitch_hz=120.0, lang="en", speed="measured", speed_cps=15.0,
              volume="normal", volume_lufs=-23.0)
    out = aggregate([row])
    assert out["gender"] == "male"
    assert out["pitch_hz"] == pytest.approx(120.0)
    assert out["speed"] == "measured" and out["volume"] == "normal"
    assert out["n_segments"] == 1
    assert out["speed_variability"] == 0.0 and out["volume_variability"] == 0.0


def test_speech_ratio_comes_from_the_parent_duration():
    rows = [seg(0, 50, gender="male"), seg(100, 150, gender="male")]
    out = aggregate(rows)
    assert out["speech_seconds"] == pytest.approx(100.0)
    assert out["file_seconds"] == pytest.approx(300.0)
    assert out["speech_ratio"] == pytest.approx(0.333, abs=1e-3)


def test_variability_is_zero_when_nothing_changes_and_rises_when_it_does():
    steady = [seg(i * 3.0, i * 3.0 + 2.5, volume="normal", volume_lufs=-23.0)
              for i in range(10)]
    assert aggregate(steady)["volume_variability"] == 0.0

    changing = ([seg(i * 3.0, i * 3.0 + 2.5, volume="soft", volume_lufs=-30.0)
                 for i in range(6)]
                + [seg(18 + i * 3.0, 18 + i * 3.0 + 2.5, volume="loud", volume_lufs=-15.0)
                   for i in range(6)])
    out = aggregate(changing)
    assert out["n_volume_spans"] == 2
    assert out["volume_variability"] == pytest.approx(0.5, abs=0.05)


def test_the_flat_scalars_the_caption_prompt_reads_are_always_present():
    from continuo_expressive.prompts.tags import tag_lines_en

    rows = [seg(i * 3.0, i * 3.0 + 2.5, gender="male", age_years=40.0,
                age_vox_years=40.0, pitch_hz=120.0, lang="en", speed="measured",
                speed_cps=15.0, volume="normal", volume_lufs=-23.0,
                emotion_scores={"happy": 0.8, "neutral": 0.2}) for i in range(8)]
    out = aggregate(rows, tau=0.7)
    for key in ("gender", "age_band", "pitch", "volume", "speed", "emotion"):
        assert key in out, f"{key} missing; continuo-caption reads it directly"
    lines = tag_lines_en(out)
    assert any(line.startswith("- Emotion") for line in lines)
    assert any(line.startswith("- Speaking Rate") for line in lines)


def test_aggregating_nothing_gives_an_empty_record():
    assert aggregate([]) == {}
