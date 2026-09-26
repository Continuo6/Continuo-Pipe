"""Select and evaluate dialogue windows."""
from __future__ import annotations

import collections
import statistics
from dataclasses import dataclass

import overlap_policy


@dataclass(frozen=True)
class Gates:
    """Selection knobs. Defaults are the values validated on real data
    (see the Phase-3 plan); every one is overridable on the CLI."""

    max_gap_s: float = 6.0          # break a window on silence longer than this
    min_duration_s: float = 30.0    # a "long" dialogue must be at least this long
    min_turns: int = 2              # absolute floor; effective minimum is
                                    # max(this, number of distinct speakers)
    max_avg_turn_s: float = 60.0    # mean turn length must stay below this
    max_speakers: int = 6           # production cap; reduced from the old 10
                                    # to limit diarization/alignment ambiguity
    min_mean_dnsmos: float = 2.4    # aggregate quality floor (reused dnsmos)
    max_fake_coverage: float | None = None
    max_overlap_coverage: float = 0.05  # ambiguous multi-speaker mixture
    min_secondary_share: float = 0.15  # 2nd speaker's share of turns (balance)
    min_reuse_share: float = 0.05   # below this, a skip-dominated window is
                                    # untranscribable (see untranscribable)


# Reasons a candidate window is rejected; surfaced only in --dry-run stats.
DROP_REASONS = (
    "too_short", "one_speaker", "too_many_speakers", "too_few_turns",
    "avg_turn_too_long", "unbalanced", "low_mean_dnsmos", "fake_coverage",
    "overlap_coverage", "untranscribable",
)


def windows(segs: list[dict], gates: Gates,
            breaks: "list[tuple[float, float]] | tuple" = ()) -> list[list[dict]]:
    """Greedy cross-speaker windows over the full timeline. A window grows
    until the gap to the next segment exceeds ``max_gap_s`` (speaker changes
    do NOT break it — that is the whole point vs the single-speaker long
    track). No duration cap."""
    segs = sorted(segs, key=lambda s: s["start"])
    out: list[list[dict]] = []
    cur = [segs[0]]


    cur_end = segs[0]["end"]
    for s in segs[1:]:
        if ((s["start"] - cur_end) > gates.max_gap_s
                or overlap_policy.spans_break(cur_end, s["start"], breaks)):
            out.append(cur)
            cur = [s]
            cur_end = s["end"]
        else:
            cur.append(s)
            if s["end"] > cur_end:
                cur_end = s["end"]
    out.append(cur)
    return out


def turns(win: list[dict]) -> list[list[dict]]:
    """Group the window's (already time-sorted) segments into turns — maximal
    runs of the same speaker."""
    runs: list[list[dict]] = []
    cur_spk = object()
    for s in win:
        if s["speaker"] != cur_spk:
            runs.append([s])
            cur_spk = s["speaker"]
        else:
            runs[-1].append(s)
    return runs


def interval_union_duration(intervals: list[tuple[float, float]]) -> float:
    """Length of the union, so overlapping annotations are not double-counted."""
    merged: list[list[float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in intervals if b > a):
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    return sum(b - a for a, b in merged)


def overlap_coverage(win: list[dict], start: float, end: float) -> float:
    """Fraction covered by at least two distinct active speakers."""
    events: dict[float, list[tuple[str, int]]] = collections.defaultdict(list)
    for s in win:
        a = max(start, float(s["start"]))
        b = min(end, float(s["end"]))
        if b <= a:
            continue
        spk = str(s.get("speaker"))
        events[a].append((spk, 1))
        events[b].append((spk, -1))
    active: collections.Counter = collections.Counter()
    overlap = 0.0
    prev = start
    for t in sorted(events):
        if sum(v > 0 for v in active.values()) >= 2:
            overlap += max(0.0, t - prev)
        for spk, delta in events[t]:
            active[spk] += delta
            if active[spk] <= 0:
                del active[spk]
        prev = t
    dur = end - start
    return overlap / dur if dur > 0 else 0.0


def evaluate(win: list[dict], gates: Gates) -> tuple[str | None, dict]:
    """Return ``(drop_reason | None, metrics)`` for a candidate window."""


    start, end = win[0]["start"], max(s["end"] for s in win)
    dur = end - start
    runs = turns(win)
    n_turns = len(runs)
    speakers = {s["speaker"] for s in win}
    required_turns = max(2, gates.min_turns, len(speakers))
    avg_turn = statistics.mean([
        max(x["end"] for x in r) - r[0]["start"] for r in runs
    ])
    turns_per_spk = collections.Counter(r[0]["speaker"] for r in runs)
    ranked = sorted(turns_per_spk.values(), reverse=True)
    sec_share = (ranked[1] / n_turns) if len(ranked) >= 2 else 0.0
    mos_weighted = [(float(s["dnsmos"]), max(0.0, s["end"] - s["start"]))
                    for s in win if s.get("dnsmos") is not None]
    mos_secs = sum(w for _, w in mos_weighted)
    mean_mos = sum(v * w for v, w in mos_weighted) / mos_secs if mos_secs else 0.0
    fake_secs = interval_union_duration([
        (s["start"], s["end"]) for s in win if s.get("is_fake")
    ])
    evaluated = any(s.get("is_fake") is not None for s in win)
    fake_cov = ((fake_secs / dur) if dur > 0 else 0.0) if evaluated else None
    overlap_cov = overlap_coverage(win, start, end)

    metrics = dict(start=start, end=end, duration=round(dur, 3), num_turns=n_turns,
                   num_speakers=len(speakers), required_turns=required_turns,
                   avg_turn=round(avg_turn, 3),
                   secondary_share=round(sec_share, 3), mean_dnsmos=round(mean_mos, 4),
                   fake_coverage=round(fake_cov, 4) if fake_cov is not None else None,
                   overlap_coverage=round(overlap_cov, 4))

    reason = None
    if dur < gates.min_duration_s:
        reason = "too_short"
    elif len(speakers) < 2:
        reason = "one_speaker"
    elif len(speakers) > gates.max_speakers:
        reason = "too_many_speakers"
    elif n_turns < required_turns:
        reason = "too_few_turns"
    elif avg_turn >= gates.max_avg_turn_s:
        reason = "avg_turn_too_long"
    elif sec_share < gates.min_secondary_share:
        reason = "unbalanced"
    elif mean_mos + 1e-9 < gates.min_mean_dnsmos:
        reason = "low_mean_dnsmos"
    elif (fake_cov is not None and gates.max_fake_coverage is not None
          and fake_cov > gates.max_fake_coverage + 1e-9):
        reason = "fake_coverage"
    elif overlap_cov > gates.max_overlap_coverage + 1e-9:
        reason = "overlap_coverage"
    return reason, metrics


def seg_status(idx: str, text_map: dict[str, str],
               part_idx: set[str]) -> tuple[str, str | None]:

    if idx in text_map:
        return "reuse", text_map[idx]
    if idx in part_idx:
        return "skip", None
    return "fill", None


def untranscribable(win: list[dict], text_map: dict[str, str],
                    part_idx: set[str], gates: Gates) -> str | None:

    dur = {"reuse": 0.0, "skip": 0.0, "fill": 0.0}
    for s in win:
        dur[seg_status(s["index"], text_map, part_idx)[0]] += s["end"] - s["start"]
    tot = sum(dur.values())
    if tot <= 0 or dur["reuse"] / tot >= gates.min_reuse_share:
        return None
    return "untranscribable" if dur["skip"] > dur["fill"] else None


def speaker_map(win: list[dict]) -> dict[str, str]:

    order: list[str] = []
    for s in win:
        if s["speaker"] not in order:
            order.append(s["speaker"])
    return {spk: f"S{i + 1}" for i, spk in enumerate(order)}
