"""Resolve overlapping diarization frames before VAD.

Judge each frame by the overlap's duration and its share of that frame. Drop
frames that cross either limit; when both frames are dropped, record a break
for the long and dialogue tracks.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OverlapPolicy:
    """Duration and per-frame share thresholds for overlap rejection."""

    max_overlap_s: float = 2.0

    max_overlap_share: float = 0.40


def _dur(f) -> float:
    return max(0.0, float(f.end) - float(f.start))


def evaluate(frames: list, policy: OverlapPolicy
             ) -> tuple[set[int], list[tuple[float, float]]]:
    """Return dropped frame indices and merged break intervals."""
    n = len(frames)
    order = sorted(range(n), key=lambda i: (frames[i].start, -frames[i].end))
    drop: set[int] = set()
    raw_breaks: list[tuple[float, float]] = []

    def _hit(d: float, dur: float) -> bool:

        if d > policy.max_overlap_s:
            return True
        return (d / dur) > policy.max_overlap_share if dur > 0 else True

    for pos, i in enumerate(order):
        a = frames[i]
        for j in order[pos + 1:]:
            b = frames[j]
            if float(b.start) >= float(a.end):


                continue
            lo = max(float(a.start), float(b.start))
            hi = min(float(a.end), float(b.end))
            d = hi - lo
            if d <= 0:
                continue

            hit_a, hit_b = _hit(d, _dur(a)), _hit(d, _dur(b))
            if hit_a:
                drop.add(i)
            if hit_b:
                drop.add(j)
            if hit_a and hit_b:

                raw_breaks.append((lo, hi))

    return drop, merge_intervals(raw_breaks)


def merge_intervals(intervals: list[tuple[float, float]]
                    ) -> list[tuple[float, float]]:

    out: list[list[float]] = []
    for a, b in sorted((float(x), float(y)) for x, y in intervals if y > x):
        if not out or a > out[-1][1]:
            out.append([a, b])
        else:
            out[-1][1] = max(out[-1][1], b)
    return [(a, b) for a, b in out]


def spans_break(lo: float, hi: float,
                breaks: "list[tuple[float, float]] | tuple") -> bool:

    if not breaks or hi <= lo:
        return False
    return any(bs < hi and be > lo for bs, be in breaks)
