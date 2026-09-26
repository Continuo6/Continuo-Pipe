"""Segmenting a timeline on how far a measurement moves, not on its bucket.

Speaking rate is continuous and its buckets are three fixed edges, so run-length
encoding the label makes those edges the definition of "changed" — which they are not.
These tests pin the alternative: boundaries where the rate actually moves.
"""
from __future__ import annotations

import pytest

from continuo_expressive.aggregate import (DEFAULT_SPEED_DELTA, aggregate, value_spans,
                                            weighted_mean, _speed_of_span)


def seg(start, end, cps, lang="en", **fields):
    row = {"rel_start": start, "rel_end": end, "dur_s": end - start,
           "speed_cps": cps, "lang": lang, "parent_duration": 300.0}
    row.update(fields)
    return row


def rate_spans(rows, delta=DEFAULT_SPEED_DELTA, **kw):
    return value_spans(rows, lambda run: weighted_mean(run, "speed_cps"),
                       _speed_of_span, delta, relative=True, **kw)


def test_a_bucket_edge_between_two_similar_rates_is_not_a_change():
    # German edges are (14.2, 22.4): 15.1 and 13.6 straddle one and would be two spans
    # under label encoding, though they differ by under 10%
    rows = ([seg(i * 5, i * 5 + 4, 15.1, lang="de") for i in range(6)]
            + [seg(30 + i * 5, 30 + i * 5 + 4, 13.6, lang="de") for i in range(6)])
    out = rate_spans(rows)
    assert len(out) == 1, f"a 10% difference is not a change of pace, got {out}"


def test_a_large_move_inside_one_bucket_is_a_change():
    # 14.5 and 22.0 are both "measured" in German, but the second is half again as fast
    rows = ([seg(i * 5, i * 5 + 4, 14.5, lang="de") for i in range(6)]
            + [seg(30 + i * 5, 30 + i * 5 + 4, 22.0, lang="de") for i in range(6)])
    out = rate_spans(rows)
    assert len(out) == 2
    assert [s["speed"] for s in out] == ["measured", "measured"], \
        "both sides are the same bucket; the change is in the rate, not the label"
    assert out[0]["speed_cps"] == pytest.approx(14.5, abs=0.1)
    assert out[1]["speed_cps"] == pytest.approx(22.0, abs=0.1)


def test_the_threshold_is_relative_so_it_transfers_across_languages():
    # the same 50% move, at Chinese and English scales, must segment identically
    zh = ([seg(i * 5, i * 5 + 4, 4.0, lang="zh") for i in range(6)]
          + [seg(30 + i * 5, 30 + i * 5 + 4, 6.0, lang="zh") for i in range(6)])
    en = ([seg(i * 5, i * 5 + 4, 12.0) for i in range(6)]
          + [seg(30 + i * 5, 30 + i * 5 + 4, 18.0) for i in range(6)])
    assert len(rate_spans(zh)) == len(rate_spans(en)) == 2


def test_a_steady_rate_stays_one_span():
    rows = [seg(i * 5, i * 5 + 4, 15.0 + (i % 2) * 0.3) for i in range(12)]
    assert len(rate_spans(rows)) == 1


def test_raising_the_threshold_never_produces_more_spans():
    rows = [seg(i * 5, i * 5 + 4, 10.0 + 2.5 * (i % 4)) for i in range(20)]
    counts = [len(rate_spans(rows, delta=d)) for d in (0.05, 0.10, 0.20, 0.40, 0.80)]
    assert counts == sorted(counts, reverse=True), counts


def test_silence_still_breaks_a_span_however_steady_the_rate():
    rows = [seg(0, 10, 15.0), seg(120, 130, 15.0)]
    out = rate_spans(rows, max_gap=10.0)
    assert len(out) == 2


def test_a_run_below_the_duration_floor_is_absorbed_despite_its_rate():
    rows = ([seg(i * 5, i * 5 + 4, 15.0) for i in range(6)]
            + [seg(30, 31, 40.0)]                       # 1 s spike, under the floor
            + [seg(32 + i * 5, 32 + i * 5 + 4, 15.0) for i in range(6)])
    out = rate_spans(rows, min_span_seconds=4.0)
    assert len(out) == 1
    assert sum(s["n_seg"] for s in out) == len(rows)


def test_spans_partition_the_segments_and_stay_in_order():
    rows = [seg(i * 5, i * 5 + 4, 8.0 + 4.0 * (i % 3)) for i in range(15)]
    out = rate_spans(rows)
    assert sum(s["n_seg"] for s in out) == len(rows)
    for left, right in zip(out, out[1:]):
        assert left["end"] <= right["start"]


def test_segments_without_a_rate_do_not_invent_a_boundary():
    rows = ([seg(i * 5, i * 5 + 4, 15.0) for i in range(4)]
            + [seg(20 + i * 5, 20 + i * 5 + 4, None) for i in range(4)]
            + [seg(40 + i * 5, 40 + i * 5 + 4, 15.0) for i in range(4)])
    assert len(rate_spans(rows)) == 1


# ------------------------------------------------------------- reported through aggregate
def test_aggregate_reports_the_move_between_speed_spans():
    rows = ([seg(i * 5, i * 5 + 4, 12.0) for i in range(6)]
            + [seg(30 + i * 5, 30 + i * 5 + 4, 18.0) for i in range(6)])
    out = aggregate(rows)
    assert out["n_speed_spans"] == 2
    assert out["speed_spans"][1]["speed_change"] == pytest.approx(0.5, abs=0.02)
    assert out["speed_spans"][1]["speed_cps_delta"] == pytest.approx(6.0, abs=0.2)
    assert out["speed_cps_spread"] == pytest.approx(1 - 12 / 18, abs=0.02)


def test_the_spread_sees_movement_the_label_share_cannot():
    # one bucket end to end, so label-share variability is zero, but the rate swings 40%
    rows = ([seg(i * 5, i * 5 + 4, 13.0) for i in range(6)]
            + [seg(30 + i * 5, 30 + i * 5 + 4, 19.5) for i in range(6)])
    out = aggregate(rows)
    assert out["speed"] == "measured" and out["speed_variability"] == 0.0
    assert out["speed_cps_spread"] > 0.3


def test_a_single_speed_span_reports_no_spread():
    rows = [seg(i * 5, i * 5 + 4, 15.0) for i in range(6)]
    assert aggregate(rows)["speed_cps_spread"] == 0.0
