"""Randomised checks on the span builders.

The hand-written tests pin behaviours someone thought of. These pin the properties
that have to hold for *any* input, which is where the bugs that survive review live:
a timeline that loses a segment, or overlaps itself, or fails to converge, is wrong in
a way that no single example is likely to expose.

The generator is seeded, so a failure is reproducible from the printed case.
"""
from __future__ import annotations

import random

import pytest

from continuo_expressive.aggregate import (aggregate, combine_lufs, seconds, spans,
                                            value_spans, weighted_mean, _speed_of_span,
                                            _volume_of_span)

VOLUMES = ("soft", "normal", "loud")
SPEEDS = ("slow", "measured", "fast")


def corpus(rng, n):
    """A plausible segment sequence: short utterances, gaps, occasional holes."""
    rows, t = [], 0.0
    for _ in range(n):
        t += rng.choice([0.1, 0.4, 1.2, 2.5, 12.0, 40.0])       # gaps, some very wide
        dur = rng.choice([0.6, 1.5, 2.5, 4.0, 9.0])
        row = {"rel_start": round(t, 3), "rel_end": round(t + dur, 3), "dur_s": dur,
               "parent_duration": 4000.0}
        if rng.random() > 0.1:
            row["volume"] = rng.choice(VOLUMES)
            row["volume_lufs"] = round(rng.uniform(-40.0, -8.0), 1)
        if rng.random() > 0.1:
            row["speed"] = rng.choice(SPEEDS)
            row["speed_cps"] = round(rng.uniform(1.0, 30.0), 1)
            row["lang"] = rng.choice(["en", "zh", "de", None])
        if rng.random() > 0.2:
            row["emotion_scores"] = {"happy": rng.random(), "neutral": rng.random(),
                                     "sad": rng.random()}
        rows.append(row)
        t += dur
    return rows


def check_timeline(out, rows, label):
    assert sum(s["n_seg"] for s in out) == len(rows), f"{label}: segments lost or doubled"
    for left, right in zip(out, out[1:]):
        assert left["end"] <= right["start"], f"{label}: spans overlap"
        assert left["start"] <= left["end"]
    if out:
        assert out[0]["start"] == pytest.approx(min(r["rel_start"] for r in rows))
        assert out[-1]["end"] == pytest.approx(max(r["rel_end"] for r in rows))


@pytest.mark.parametrize("seed", range(25))
def test_label_spans_partition_their_input(seed):
    rng = random.Random(seed)
    rows = corpus(rng, rng.randint(1, 60))
    out = spans(rows, lambda r: r.get("volume"), _volume_of_span, "volume",
                max_gap=rng.choice([2.0, 10.0, 1e9]),
                min_span_seconds=rng.choice([0.0, 4.0, 30.0]))
    check_timeline(out, rows, f"label seed={seed}")


@pytest.mark.parametrize("seed", range(25))
def test_value_spans_partition_their_input(seed):
    rng = random.Random(seed)
    rows = corpus(rng, rng.randint(1, 60))
    out = value_spans(rows, lambda run: weighted_mean(run, "speed_cps"), _speed_of_span,
                      min_delta=rng.choice([0.0, 0.15, 0.5, 5.0]),
                      max_gap=rng.choice([2.0, 10.0, 1e9]),
                      min_span_seconds=rng.choice([0.0, 4.0, 30.0]))
    check_timeline(out, rows, f"value seed={seed}")


@pytest.mark.parametrize("seed", range(25))
def test_every_surviving_boundary_is_a_real_move(seed):
    rng = random.Random(seed)
    rows = corpus(rng, rng.randint(2, 60))
    delta = 0.25
    out = value_spans(rows, lambda run: weighted_mean(run, "speed_cps"), _speed_of_span,
                      min_delta=delta, max_gap=1e9, min_span_seconds=0.0)
    for left, right in zip(out, out[1:]):
        a, b = left.get("speed_cps"), right.get("speed_cps")
        if a is None or b is None:
            continue
        assert abs(a - b) / max(a, b) >= delta - 1e-9, \
            f"seed={seed}: kept a boundary of {abs(a - b) / max(a, b):.3f} under {delta}"


@pytest.mark.parametrize("seed", range(15))
def test_a_merged_value_stays_within_its_members(seed):
    # a weighted mean cannot leave the range its members span; if it does, the weights
    # are wrong somewhere
    rng = random.Random(seed)
    rows = corpus(rng, rng.randint(2, 40))
    out = value_spans(rows, lambda run: weighted_mean(run, "speed_cps"), _speed_of_span,
                      min_delta=0.15, max_gap=1e9, min_span_seconds=4.0)
    rates = [r["speed_cps"] for r in rows if r.get("speed_cps") is not None]
    if not rates:
        return
    for span in out:
        if span.get("speed_cps") is not None:
            assert min(rates) - 0.05 <= span["speed_cps"] <= max(rates) + 0.05


@pytest.mark.parametrize("seed", range(15))
def test_loudness_combines_within_its_members(seed):
    rng = random.Random(seed)
    rows = corpus(rng, rng.randint(2, 40))
    levels = [r["volume_lufs"] for r in rows if r.get("volume_lufs") is not None]
    combined = combine_lufs(rows)
    if not levels:
        assert combined is None
        return
    assert min(levels) - 1e-6 <= combined <= max(levels) + 1e-6


@pytest.mark.parametrize("seed", range(20))
def test_aggregate_is_internally_consistent(seed):
    rng = random.Random(seed)
    rows = corpus(rng, rng.randint(1, 80))
    out = aggregate(rows, tau=0.7)
    assert out["n_segments"] == len(rows)
    assert out["speech_seconds"] == pytest.approx(sum(seconds(r) for r in rows), abs=0.01)
    for key in ("emotion", "speed", "volume"):
        span_list = out[f"{key}_spans"]
        assert out[f"n_{key}_spans"] == len(span_list)
        assert 0.0 <= out[f"{key}_variability"] <= 1.0
        # the flat scalar is one of the labels the timeline actually reports
        labels = {s.get(key) for s in span_list}
        assert out[key] in labels or out[key] is None
    assert 0.0 <= out["speed_cps_spread"] <= 1.0


def test_a_pathological_alternation_still_converges():
    # the merge loops carry iteration guards; hitting one would silently truncate the
    # timeline, so make sure a worst case finishes well inside them
    rows = [{"rel_start": i * 1.0, "rel_end": i * 1.0 + 0.9, "dur_s": 0.9,
             "parent_duration": 3000.0, "speed_cps": 5.0 + 20.0 * (i % 2),
             "speed": SPEEDS[i % 3], "volume": VOLUMES[i % 3],
             "volume_lufs": -30.0 + 20.0 * (i % 2), "lang": "en"}
            for i in range(600)]
    out = aggregate(rows, min_span_seconds=60.0)
    assert sum(s["n_seg"] for s in out["speed_spans"]) == 600
    assert sum(s["n_seg"] for s in out["volume_spans"]) == 600
