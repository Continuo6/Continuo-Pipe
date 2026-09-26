"""The pure decision logic: bucket edges, cascades, and the emotion gate.

These are the functions that turn measurements into published labels, so they are
pinned here — an accidental edit to an edge or a cascade branch would change every
label in a corpus without failing anything else.
"""
from __future__ import annotations

import pytest

from continuo_expressive.ensemble.age import age_band, cascade_age
from continuo_expressive.ensemble.dialect import cascade_dialect
from continuo_expressive.registry import ACCENT_LABELS_FOR_LANG, restrict_accent
from continuo_expressive.ensemble.emotion_gate import gate_emotion
from continuo_expressive.features.buckets import (PITCH_LABELS, SPEED_CPS_EDGES,
                                                   SPEED_SHAPE,
                                                   SPEED_LABELS, VOLUME_EDGES,
                                                   VOLUME_LABELS, bucket,
                                                   pitch_edges_for)


# ------------------------------------------------------------------ buckets
@pytest.mark.parametrize("lufs,expected", [
    (-40.0, "soft"), (-27.1, "soft"),
    (-27.0, "normal"), (-23.0, "normal"), (-19.0, "normal"),
    (-18.9, "loud"), (-5.0, "loud"),
])
def test_volume_edges_are_inclusive_in_the_middle(lufs, expected):
    assert bucket(lufs, VOLUME_EDGES, VOLUME_LABELS) == expected


def test_bucket_is_none_safe():
    assert bucket(None, VOLUME_EDGES, VOLUME_LABELS) is None
    assert bucket(float("nan"), VOLUME_EDGES, VOLUME_LABELS) is None


def test_pitch_edges_split_by_gender():
    assert pitch_edges_for("male") == (115.7, 149.7)
    assert pitch_edges_for("female") == (141.6, 184.5)


@pytest.mark.parametrize("gender", ["child", None, "", "unknown"])
def test_unknown_gender_uses_the_female_pitch_range(gender):
    assert pitch_edges_for(gender) == pitch_edges_for("female")


def test_same_f0_can_bucket_differently_per_gender():
    hz = 130.0                      # mid-range for a man, low for a woman
    assert bucket(hz, pitch_edges_for("male"), PITCH_LABELS) == "medium"
    assert bucket(hz, pitch_edges_for("female"), PITCH_LABELS) == "low"


def test_speed_scale_differs_by_an_order_between_zh_and_en():
    # a Chinese character is a syllable, an English one is not
    assert bucket(4.5, SPEED_CPS_EDGES["zh"], SPEED_LABELS) == "measured"
    assert bucket(4.5, SPEED_CPS_EDGES["en"], SPEED_LABELS) == "slow"


# ------------------------------------------------------------------ age
@pytest.mark.parametrize("years,band", [
    (4, "child"), (12.9, "child"), (13, "teen"), (17.9, "teen"),
    (18, "young"), (42.9, "young"), (43, "middle"), (59.9, "middle"), (60, "senior"),
])
def test_age_bands(years, band):
    assert age_band(years) == band


def test_child_classifier_overrides_both_regressions():
    # both regressions over-estimate children badly; the classifier is the only
    # signal that recovers them, so it wins outright
    assert cascade_age("child", aud_age=22.0, vox_age=31.0) == "child"


def test_adult_path_takes_the_median_of_the_two_heads():
    assert cascade_age("male", aud_age=50.0, vox_age=40.0) == "middle"   # median 45
    assert cascade_age("male", aud_age=30.0, vox_age=40.0) == "young"    # median 35


def test_adult_path_falls_back_to_whichever_head_answered():
    assert cascade_age("female", aud_age=None, vox_age=55.0) == "middle"
    assert cascade_age("female", aud_age=25.0, vox_age=None) == "young"


def test_no_age_signal_yields_none():
    assert cascade_age("male", None, None) is None


def test_adult_source_can_be_forced():
    assert cascade_age("male", aud_age=30.0, vox_age=65.0, adult="aud") == "young"
    assert cascade_age("male", aud_age=30.0, vox_age=65.0, adult="vox") == "senior"


# ------------------------------------------------------------------ dialect
def test_coarse_firered_buckets_defer_to_voxlect():
    assert cascade_dialect("zh north", "Zhongyuan") == "Zhongyuan"
    assert cascade_dialect("other", "Jiang-Huai") == "Jiang-Huai"


def test_confident_firered_buckets_win():
    assert cascade_dialect("zh mandarin", "Zhongyuan") == "Standard Mandarin"
    assert cascade_dialect("zh xinan", "Ji-Lu") == "Southwestern"
    assert cascade_dialect("zh yue", "Standard Mandarin") == "Cantonese"


def test_varieties_voxlect_lacks_are_kept_from_firered():
    assert cascade_dialect("zh wu", "Standard Mandarin") == "wu"
    assert cascade_dialect("zh xiang", "Zhongyuan") == "xiang"


def test_unrecognised_or_missing_lid_falls_back_to_voxlect():
    assert cascade_dialect("en", "Zhongyuan") == "Zhongyuan"
    assert cascade_dialect(None, "Zhongyuan") == "Zhongyuan"
    assert cascade_dialect("", "Zhongyuan") == "Zhongyuan"


# ------------------------------------------------------------------ emotion gate
def _prediction(label="happy", confidence=0.9):
    scores = {"happy": 0.9, "neutral": 0.05, "sad": 0.03, "angry": 0.02}
    return {"label": label, "confidence": confidence, "scores": scores}


def test_confident_label_passes():
    out = gate_emotion(_prediction(confidence=0.9), tau=0.7)
    assert out["emotion"] == "happy"
    assert out["emotion_confidence"] == 0.9


def test_tau_is_inclusive():
    assert gate_emotion(_prediction(confidence=0.7), tau=0.7)["emotion"] == "happy"


def test_below_tau_abstains_but_keeps_the_confidence():
    out = gate_emotion(_prediction(confidence=0.55), tau=0.7)
    assert out["emotion"] is None
    assert out["emotion_confidence"] == 0.55      # kept, so a null can be explained
    assert out["emotion_top3"]["happy"] == 0.9


def test_unknown_never_surfaces_however_confident():
    assert gate_emotion(_prediction(label="unknown", confidence=0.99), tau=0.7)["emotion"] is None


def test_other_is_a_real_class_and_survives():
    assert gate_emotion(_prediction(label="other", confidence=0.8), tau=0.7)["emotion"] == "other"


def test_missing_ser_record_yields_all_nulls():
    out = gate_emotion(None)
    assert out == {"emotion": None, "emotion_confidence": None, "emotion_top3": None}


def test_top3_is_ranked_and_capped():
    top3 = gate_emotion(_prediction(), tau=0.7)["emotion_top3"]
    assert list(top3) == ["happy", "neutral", "sad"]


# ------------------------------------------------------------------ speed coverage
def test_only_english_edges_are_gold_fitted():
    # everything else is a shape transfer anchored on a corpus median; the distinction
    # is the whole reason non-en `speed` is a ranking rather than a physical claim
    assert SPEED_CPS_EDGES["en"] == (12.5, 19.8)


def test_shape_transfer_reproduces_the_english_gold_ratios():
    lo_mult, hi_mult = SPEED_SHAPE
    for lang, (lo, hi) in SPEED_CPS_EDGES.items():
        if lang == "en":
            continue
        median = lo / lo_mult                      # the anchor these edges came from
        assert abs(hi / median - hi_mult) < 0.02, f"{lang} edges are not a shape transfer"


def test_every_language_with_edges_orders_them():
    for lang, (lo, hi) in SPEED_CPS_EDGES.items():
        assert 0 < lo < hi, lang


def test_scale_gap_between_scripts_is_preserved():
    # a Chinese character is a syllable, a German one is not; sharing edges would put
    # every Chinese clip in `slow`
    assert SPEED_CPS_EDGES["zh"][1] < SPEED_CPS_EDGES["de"][0]


def test_language_without_edges_is_left_alone():
    assert "ko" not in SPEED_CPS_EDGES        # below the min-clips floor, stays null
    assert bucket(9.0, SPEED_CPS_EDGES.get("ko", (1e9, 1e9)), SPEED_LABELS) == "slow"


# ------------------------------------------ the declared language constrains the accent
def test_a_chinese_clip_is_never_given_a_cantonese_accent():
    # the mandarin-cantonese head covers both, but Cantonese is `yue` — a different
    # language. A corpus that has already declared a clip `zh` is stronger evidence
    # than the head's own preference.
    probs = {"Southwestern": 0.368, "Cantonese": 0.339, "Jiang-Huai": 0.154,
             "Standard Mandarin": 0.139}
    out = restrict_accent(probs, "zh")
    assert "Cantonese" not in out
    assert max(out, key=out.get) == "Southwestern"


def test_a_cantonese_clip_is_never_given_a_mandarin_subgroup():
    probs = {"Zhongyuan": 0.6, "Cantonese": 0.4}
    assert restrict_accent(probs, "yue") == {"Cantonese": 1.0}


def test_the_restricted_distribution_still_sums_to_one():
    probs = {"Southwestern": 0.4, "Cantonese": 0.4, "Zhongyuan": 0.2}
    out = restrict_accent(probs, "zh")
    assert sum(out.values()) == pytest.approx(1.0)


def test_an_unlisted_language_is_left_alone():
    # inventing a constraint is worse than declining to apply one
    probs = {"Zhongyuan": 0.6, "Cantonese": 0.4}
    assert restrict_accent(probs, "de") == probs
    assert restrict_accent(probs, None) == probs


def test_a_distribution_with_no_admissible_mass_is_left_alone():
    assert restrict_accent({"Cantonese": 1.0}, "zh") == {"Cantonese": 1.0}


def test_every_language_that_routes_to_a_head_has_an_admissible_set():
    from continuo_expressive.cli.annotate import ACCENT_HEADS
    for lang in ACCENT_HEADS:
        assert lang in ACCENT_LABELS_FOR_LANG, lang


def test_follow_needs_a_manifest():
    """--follow re-reads a file; there is nothing to re-read for --audio."""
    import pytest
    from continuo_expressive.cli import annotate as ann
    with pytest.raises(SystemExit):
        ann.main(["--audio", "x.wav", "--follow", "5"])


def test_follow_defaults_off():
    """A plain run must still be one pass: --follow is opt-in."""
    from continuo_expressive.cli import annotate as ann
    args = ann.build_parser().parse_args(["--manifest", "m.jsonl"])
    assert args.follow == 0.0
    assert args.follow_until == ""
