"""The line that tells a captioner how a long recording moves.

The risk here is not a crash, it is a plausible-looking prompt that says something
false — "measured, then measured, then measured", or a narrative of change on a
recording that never changed. Both would be laundered into a caption and become part
of the corpus, so the conditions under which the line appears at all are pinned here.
"""
from __future__ import annotations

from continuo_expressive.prompts import build_prompt
from continuo_expressive.prompts.tags import (VARIATION_MIN, tag_lines_en,
                                               variation_line_en, variation_line_zh)


def span(attr, label, start, end):
    return {"start": start, "end": end, "seconds": end - start, "n_seg": 5, attr: label}


def sspan(label, cps, start, end):
    """A speed span, which carries the measurement its boundaries were cut on."""
    return {"start": start, "end": end, "seconds": end - start, "n_seg": 5,
            "speed": label, "speed_cps": cps}


def with_speed(spans, **over):
    """Record whose speed movement is derived from the spans, as aggregate() does."""
    rates = [s["speed_cps"] for s in spans if s.get("speed_cps")]
    spread = (max(rates) - min(rates)) / max(rates) if len(rates) > 1 else 0.0
    return record(speed_spans=spans, speed_cps_spread=round(spread, 3), **over)


def record(**over):
    rec = {"id": "x", "file_seconds": 100.0, "gender": "male", "age_band": "middle",
           "pitch": "medium", "volume": "normal", "speed": "measured"}
    rec.update(over)
    return rec


def test_a_steady_recording_gets_no_variation_line():
    rec = record(volume_spans=[span("volume", "normal", 0, 100)],
                 volume_variability=0.0)
    assert variation_line_en(rec) is None


def test_barely_varying_is_treated_as_steady():
    rec = record(volume_spans=[span("volume", "normal", 0, 95), span("volume", "soft", 95, 100)],
                 volume_variability=VARIATION_MIN / 2)
    assert variation_line_en(rec) is None


def test_a_real_change_is_reported_with_positions():
    rec = record(volume_spans=[span("volume", "normal", 0, 60), span("volume", "soft", 60, 100)],
                 volume_variability=0.4)
    line = variation_line_en(rec)
    assert line is not None
    # positions are timestamps, not shares: a captioner cannot keep two kinds of
    # percentage apart, and read a phase at 46% of a 190 s file as "46 seconds in"
    assert "moderate (0:00-1:00)" in line and "soft (1:00-1:40)" in line
    assert "%" not in line.split("volume — ")[1], "the only percentages left are magnitudes"


def test_dropping_a_short_volume_span_never_leaves_a_repeated_label():
    # the middle span is under the 5% floor and gets filtered out; without collapsing,
    # the line would read "moderate ..., moderate ..." and say nothing at all
    rec = record(volume_spans=[span("volume", "normal", 0, 40),
                               span("volume", "soft", 40, 42),
                               span("volume", "normal", 42, 100)],
                 volume_variability=0.3)
    line = variation_line_en(rec)
    assert line is None, "one label across the whole file is not a change"


def test_a_surviving_volume_change_still_reads_in_order():
    rec = record(volume_spans=[span("volume", "normal", 0, 30),
                               span("volume", "soft", 30, 32),      # dropped, under 5%
                               span("volume", "normal", 32, 60),
                               span("volume", "loud", 60, 100)],
                 volume_variability=0.4)
    line = variation_line_en(rec)
    assert line is not None
    # the two "moderate" stretches either side of the dropped span read as one
    assert line.count("moderate") == 1
    assert line.index("moderate") < line.index("loud")


def test_no_volume_label_is_ever_reported_twice_in_a_row():
    labels = ["normal", "soft", "normal", "soft", "normal", "soft", "normal", "soft",
              "normal"]
    spans = [span("volume", label, i * 10, i * 10 + 10) for i, label in enumerate(labels)]
    rec = record(file_seconds=90.0, volume_spans=spans, volume_variability=0.5)
    line = variation_line_en(rec)
    assert line is not None
    reported = [chunk.split(" (")[0] for chunk in line.split(": ", 1)[1].split(" — ")[1].split(", ")]
    for left, right in zip(reported, reported[1:]):
        assert left != right


# ------------------------------------------------------- speed moves, not just buckets
def test_a_rate_swing_inside_one_bucket_is_still_reported():
    # every span is "measured", so a label-based line would say nothing — but the rate
    # moves 40%, which is the change worth describing
    rec = with_speed([sspan("measured", 20.0, 0, 40),
                      sspan("measured", 12.0, 40, 100)])
    line = variation_line_en(rec)
    assert line is not None, "a 40% swing inside one bucket is a change"
    assert "slower" in line
    assert line.count("measured") == 2, "both phases are named, not collapsed"


def test_the_reported_direction_and_size_match_the_rates():
    rec = with_speed([sspan("measured", 10.0, 0, 50), sspan("measured", 15.0, 50, 100)])
    assert "50% faster" in variation_line_en(rec)


def test_a_rate_that_barely_moves_is_not_reported():
    rec = with_speed([sspan("measured", 15.0, 0, 50), sspan("measured", 15.6, 50, 100)])
    assert variation_line_en(rec) is None


def test_several_attributes_are_reported_together():
    rec = with_speed([sspan("slow", 9.0, 0, 50), sspan("fast", 21.0, 50, 100)],
                     volume_spans=[span("volume", "normal", 0, 50),
                                   span("volume", "loud", 50, 100)],
                     volume_variability=0.5)
    line = variation_line_en(rec)
    assert "volume" in line and "speaking rate" in line


def test_chinese_output_prompt_receives_english_variation_data():
    rec = with_speed([sspan("slow", 9.0, 0, 50), sspan("fast", 21.0, 50, 100)])
    line = variation_line_zh(rec)
    assert "speaking rate" in line and "slow" in line and "fast" in line


def test_chinese_output_prompt_preserves_movement_data():
    rec = with_speed([sspan("measured", 10.0, 0, 50), sspan("measured", 15.0, 50, 100)])
    line = variation_line_zh(rec)
    assert "50% faster than before" in line


def test_a_single_clip_record_is_unaffected():
    assert variation_line_en(record()) is None
    assert all("Change over the recording" not in line for line in tag_lines_en(record()))


def test_the_flag_is_what_lets_spans_reach_the_prompt():
    rec = with_speed([sspan("slow", 9.0, 0, 50), sspan("fast", 21.0, 50, 100)])
    assert "Change over the recording" not in build_prompt(rec, form="aps", lang="en")
    assert "Change over the recording" in build_prompt(rec, form="aps", lang="en",
                                                       describe_variation=True)


def test_the_flag_reaches_every_form_and_language():
    rec = with_speed([sspan("slow", 9.0, 0, 50), sspan("fast", 21.0, 50, 100)])
    for form in ("free", "aps", "dsd", "rp"):
        assert "Change over the recording" in build_prompt(
            rec, form=form, lang="en", describe_variation=True), form
        zh_prompt = build_prompt(rec, form=form, lang="zh", describe_variation=True)
        assert "Change over the recording" in zh_prompt, form
        assert "Simplified Chinese" in zh_prompt, form
