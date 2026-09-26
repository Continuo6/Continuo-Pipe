"""Rebuild a per-file answer from a long recording's per-segment annotations.

A long container is cut into utterances, each measured on its own by the normal
pipeline, and this module puts the pieces back together. The central claim is that
**the seven attributes are not one kind of thing and must not be collapsed one way**:

*Speaker-intrinsic* — ``gender``, ``accent``, ``age``, ``pitch``.
    A person's sex, dialect, age and vocal register do not change over one recording.
    Disagreement between segments is measurement noise, so these collapse to a single
    duration-weighted value per file. The disagreement is still worth keeping, as a
    quality signal rather than an answer: ``gender_agreement`` well below 1 on a
    nominally single-speaker container means the segmentation or that assumption is
    wrong, and the file should be looked at rather than trusted.

*Context-varying* — ``emotion``, ``speed``, ``volume``.
    These change with what is being said, and averaging them over four minutes destroys
    exactly what makes a long recording interesting. They are kept as a **timeline** of
    spans, so "calm and measured at first, faster and brighter later" survives into the
    output.

    How a timeline is cut depends on what the attribute is. Emotion is categorical and
    volume's three-way split is what anyone consuming it acts on, so those merge
    adjacent segments that share a label. Speaking rate is a continuous quantity that
    happens to be bucketed, and cutting *it* on label changes would put the boundaries
    wherever the bucket edges sit rather than wherever the rate moved — so speed is
    segmented on the measurement itself. See :func:`spans` and :func:`value_spans`.

Scalar ``emotion`` / ``speed`` / ``volume`` fields are emitted alongside the spans,
carrying the dominant span's label. They are not redundant: the caption prompt builder
(:mod:`continuo_expressive.prompts.tags`) reads those flat keys, so dropping them
would break ``continuo-caption``. The spans are the new information; the scalars are the
compatibility surface.

Three things the span merge has to get right, none of them obvious:

**Silence breaks a span.** Speech covers about two thirds of a long container. Two
segments either side of a ten-second pause are not a continuous stretch of anything, so
a gap wider than ``max_gap`` ends the span.

**Short spans are smoothed away.** Segments run ~2.5 s at the median, and per-clip
labels at that length are noisy — a raw run-length encoding produces a shower of
one-segment spans that look like variation and are not. Runs shorter than
``min_span_seconds`` are absorbed into whichever neighbour they actually touch, and the
result is re-coalesced, until nothing short is left. Smoothing never reaches across a
gap: a genuinely isolated short utterance is not label thrash, and survives.

**The merged span is re-measured, not inherited.** A span's continuous value is
recomputed from its members and re-bucketed, and that verdict wins over the label its
members happened to carry, because it rests on more audio. This only ever changes
anything for a run that smoothing built out of disagreeing segments — a weighted mean
is bounded by its members, so a run whose members all agreed cannot re-bucket
elsewhere. Where it does change something, the neighbours are re-checked and merged if
they now agree, so no two adjacent label-cut spans ever publish the same label. (Speed
is exempt: two stretches both called ``measured`` that differ by 30% are two spans on
purpose.) The recomputation is per-quantity, and two of the three are not plain
averages:

*   loudness is logarithmic, so LUFS combine in the energy domain, never by averaging
    decibels;
*   speaking rate is a ratio, so the span rate is total characters over total seconds.
    That happens to equal the *duration-weighted* mean of the per-segment rates —
    ``cps_i * seconds_i`` is just ``characters_i`` — which is why no transcript has to
    be carried this far. It is emphatically not the plain mean of the rates;
*   emotion is gated per clip, where the threshold was validated, and the survivors
    then vote by duration. Averaging the posteriors first and thresholding the average
    is the tempting version and it is measurably worse — see :func:`_emotion_of_span`.

Grouping is the caller's job: :func:`aggregate` takes the rows of one group. Passing
one speaker's rows instead of one file's is all a future multi-speaker path needs.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Sequence

from .ensemble.age import cascade_age
from .ensemble.emotion_gate import gate_emotion
from .features.buckets import (PITCH_LABELS, SPEED_CPS_EDGES, SPEED_LABELS,
                               VOLUME_EDGES, VOLUME_LABELS, bucket, pitch_edges_for)

#: a silence wider than this ends a span.
#:
#: Set above ordinary inter-utterance pauses so a continuous attribute is not
#: split at each breath. Large gaps usually indicate silence or an interlude.
DEFAULT_MAX_GAP = 10.0
#: runs shorter than this are label thrash and get absorbed into a neighbour
DEFAULT_MIN_SPAN_SECONDS = 4.0

#: how far characters-per-second has to move, in relative terms, before it counts as a
#: change of pace. Relative because the quantity is not commensurable across languages:
#: Chinese runs ~4 CPS and English ~15, so one absolute threshold cannot serve both.
#:
#: Swept over five long German containers: 0.05 leaves 14 spans per file and 0.10 leaves
#: 8, both of them still tracking segment noise; 0.20 collapses two of the five to a
#: single span, because this speaker's pace never moves that far between neighbouring
#: stretches. At 0.15 each file keeps 3-5 spans and the boundaries that survive are
#: moves of 26-95% — well clear of the threshold rather than balanced on it, which is
#: what a well-chosen cut looks like.
DEFAULT_SPEED_DELTA = 0.15



# ------------------------------------------------------------------- primitives
def seconds(row: dict) -> float:
    """How much audio this row speaks for.

    ``dur_s`` is what the annotate pass measured off the decoded waveform; ``duration``
    is the manifest's claim; the offsets are the last resort. They agree in practice,
    but the measured one is the one that matches what the models actually saw.
    """
    for key in ("dur_s", "duration"):
        value = row.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    start, end = row.get("rel_start"), row.get("rel_end")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
        return float(end - start)
    return 0.0


def weighted_vote(rows: Sequence[dict], key: str) -> tuple[object, float]:
    """Longest-total-duration label, and the share of labelled duration it holds.

    The share is the agreement statistic: 1.0 means every labelled second said the same
    thing. Rows with no label take no part in either number.
    """
    totals: dict[object, float] = defaultdict(float)
    for row in rows:
        label = row.get(key)
        if label is None or label == "":
            continue
        totals[label] += seconds(row)
    if not totals:
        return None, 0.0
    total = sum(totals.values())
    winner = max(totals, key=lambda k: (totals[k], str(k)))
    return winner, (totals[winner] / total if total else 0.0)


def weighted_mean(rows: Sequence[dict], key: str) -> float | None:
    num = den = 0.0
    for row in rows:
        value = row.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        w = seconds(row)
        num += float(value) * w
        den += w
    return num / den if den else None


def combine_lufs(rows: Sequence[dict], key: str = "volume_lufs") -> float | None:
    """Duration-weighted loudness, combined in the energy domain.

    LUFS is a logarithm. Averaging two segments at -30 and -20 gives -25, which is not
    what the pair sounds like: back in the energy domain the answer is -22.6, because
    the louder half dominates what a listener hears.

    This is a plain duration-weighted power mean. BS.1770's ``-0.691`` offset cancels
    out of a single-channel weighted combination — it would only matter if the channel
    gains ``G_i`` differed, and this pipeline is mono throughout — so carrying it here
    would be a constant that does nothing.
    """
    num = den = 0.0
    for row in rows:
        value = row.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if not math.isfinite(float(value)):
            continue
        w = seconds(row)
        num += w * (10.0 ** (float(value) / 10.0))
        den += w
    if not den or num <= 0:
        return None
    return 10.0 * math.log10(num / den)


def mean_scores(rows: Sequence[dict], key: str = "emotion_scores") -> dict[str, float]:
    """Duration-weighted mean of the raw 9-class posteriors over a set of rows."""
    totals: dict[str, float] = defaultdict(float)
    den = 0.0
    for row in rows:
        scores = row.get(key)
        if not isinstance(scores, dict) or not scores:
            continue
        w = seconds(row)
        for label, p in scores.items():
            totals[label] += w * float(p)
        den += w
    if not den:
        return {}
    return {label: value / den for label, value in totals.items()}


# ----------------------------------------------------------- speaker-intrinsic tier
def aggregate_speaker(rows: Sequence[dict]) -> dict:
    """Collapse the attributes that belong to the voice, not to the moment."""
    gender, gender_agreement = weighted_vote(rows, "gender")
    accent_head, _ = weighted_vote(rows, "accent_head")
    lang, lang_share = weighted_vote(rows, "lang")

    # accent by summed probability rather than by voted label: a segment that was 0.4
    # confident should not outvote three that were 0.34 each on the same runner-up
    probs: dict[str, float] = defaultdict(float)
    accent_den = 0.0
    for row in rows:
        top3 = row.get("accent_top3")
        if not isinstance(top3, dict) or not top3:
            continue
        w = seconds(row)
        for label, p in top3.items():
            probs[label] += w * float(p)
        accent_den += w
    accent = accent_top3 = None
    accent_agreement = 0.0
    if probs and accent_den:
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        accent = ranked[0][0]
        accent_top3 = {label: round(value / accent_den, 3) for label, value in ranked[:3]}
        total = sum(probs.values())
        accent_agreement = probs[accent] / total if total else 0.0

    age_years = weighted_mean(rows, "age_years")
    age_vox_years = weighted_mean(rows, "age_vox_years")
    pitch_hz = weighted_mean(rows, "pitch_hz")

    languages = {row.get("lang") for row in rows if row.get("lang")}

    return {
        "gender": gender,
        "gender_agreement": round(gender_agreement, 3),
        "age_years": round(age_years, 1) if age_years is not None else None,
        "age_vox_years": round(age_vox_years, 1) if age_vox_years is not None else None,
        # re-run the cascade on the aggregated inputs rather than voting on per-segment
        # bands: the cascade's child branch keys off gender, and binning a mean is not
        # the same as taking the mode of bins
        "age_band": cascade_age(gender, age_years, age_vox_years),
        "accent": accent,
        "accent_top3": accent_top3,
        "accent_head": accent_head,
        "accent_agreement": round(accent_agreement, 3),
        "pitch_hz": round(pitch_hz, 1) if pitch_hz is not None else None,
        "pitch": bucket(pitch_hz, pitch_edges_for(gender), PITCH_LABELS),
        "lang": lang,
        "n_languages": len(languages),
        "dominant_lang_ratio": round(lang_share, 3),
    }


# --------------------------------------------------------------- context-varying tier
#: a run is ``(label, rows)``. The label is carried explicitly rather than re-read from
#: a member, because smoothing merges rows of a different label into a run and the run
#: must keep its own identity afterwards — deriving it from ``rows[0]`` makes a run
#: change label whenever an absorbed row lands at its front, which silently splits one
#: unchanging stretch into two adjacent identical spans.
Run = tuple


def _runs(rows: Sequence[dict], label_of: Callable[[dict], object],
          max_gap: float) -> list[Run]:
    """Run-length encode by label, breaking wherever silence exceeds ``max_gap``."""
    runs: list[Run] = []
    previous_label, previous_end = object(), None
    for row in rows:
        label = label_of(row)
        start = row.get("rel_start")
        gap = (float(start) - previous_end) if (previous_end is not None
                                                and isinstance(start, (int, float))) else 0.0
        if not runs or label != previous_label or gap > max_gap:
            runs.append((label, [row]))
        else:
            runs[-1][1].append(row)
        previous_label = label
        end = row.get("rel_end")
        previous_end = float(end) if isinstance(end, (int, float)) else None
    return runs


def _touching(left: list[dict], right: list[dict], max_gap: float) -> bool:
    """Whether two consecutive runs are separated by silence narrow enough to merge."""
    end, start = left[-1].get("rel_end"), right[0].get("rel_start")
    if not isinstance(end, (int, float)) or not isinstance(start, (int, float)):
        return True
    return (float(start) - float(end)) <= max_gap


def _coalesce(runs: list[Run], max_gap: float) -> list[Run]:
    """Merge neighbouring runs that share a label and are not split by silence."""
    merged: list[Run] = []
    for label, rows in runs:
        if merged and merged[-1][0] == label and _touching(merged[-1][1], rows, max_gap):
            merged[-1][1].extend(rows)
        else:
            merged.append((label, list(rows)))
    return merged


def _smooth(runs: list[Run], max_gap: float, min_span_seconds: float) -> list[Run]:
    """Absorb sub-``min_span_seconds`` runs into a touching neighbour, until stable.

    The absorbed rows take on the host run's label — that is the point, they were
    label noise. A run isolated by silence on both sides is left alone however short it
    is: a lone utterance is not a label flickering inside a longer stretch.
    """
    guard = 0
    while len(runs) > 1 and guard < 1000:
        guard += 1
        target = None
        for i, (_, rows) in enumerate(runs):
            if sum(seconds(r) for r in rows) >= min_span_seconds:
                continue
            left_ok = i > 0 and _touching(runs[i - 1][1], rows, max_gap)
            right_ok = i + 1 < len(runs) and _touching(rows, runs[i + 1][1], max_gap)
            if left_ok or right_ok:
                target = (i, left_ok, right_ok)
                break
        if target is None:
            break
        i, left_ok, right_ok = target
        left_len = sum(seconds(r) for r in runs[i - 1][1]) if left_ok else -1.0
        right_len = sum(seconds(r) for r in runs[i + 1][1]) if right_ok else -1.0
        into = i - 1 if left_len >= right_len else i + 1
        host_label, host_rows = runs[into]
        runs[into] = (host_label,
                      sorted(host_rows + runs[i][1],
                             key=lambda r: (r.get("rel_start") or 0.0)))
        runs.pop(i)
        # absorbing can leave two same-label runs adjacent; coalesce so the next pass
        # sees the merged length rather than two halves of it
        runs = _coalesce(runs, max_gap)
    return runs


def spans(rows: Sequence[dict], label_of: Callable[[dict], object],
          recompute: Callable[[Sequence[dict]], dict], label_key: str,
          max_gap: float = DEFAULT_MAX_GAP,
          min_span_seconds: float = DEFAULT_MIN_SPAN_SECONDS) -> list[dict]:
    """One attribute's timeline: ``[{start, end, seconds, n_seg, ...recomputed}]``.

    ``label_key`` names the published label inside ``recompute``'s output. It is needed
    because re-measurement can hand two neighbours the same verdict even though their
    members disagreed — a run of quiet segments whose combined loudness still lands in
    ``normal``, say. Leaving those side by side would draw a boundary on the timeline
    where nothing changes, so they are merged and re-measured until no two adjacent
    spans say the same thing.
    """
    ordered = _ordered(rows)
    if not ordered:
        return []
    runs = _smooth(_runs(ordered, label_of, max_gap), max_gap, min_span_seconds)

    while True:
        computed = [recompute(run) for _, run in runs]
        for i in range(len(runs) - 1):
            if (computed[i].get(label_key) == computed[i + 1].get(label_key)
                    and _touching(runs[i][1], runs[i + 1][1], max_gap)):
                runs[i] = (runs[i][0], runs[i][1] + runs[i + 1][1])
                runs.pop(i + 1)
                break
        else:
            break

    return _assemble([rows_ for _, rows_ in runs], recompute)


def _ordered(rows: Sequence[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (r.get("rel_start") if isinstance(
        r.get("rel_start"), (int, float)) else 0.0))


def _assemble(runs: Sequence[Sequence[dict]],
              recompute: Callable[[Sequence[dict]], dict]) -> list[dict]:
    out = []
    for run in runs:
        starts = [r.get("rel_start") for r in run if isinstance(r.get("rel_start"), (int, float))]
        ends = [r.get("rel_end") for r in run if isinstance(r.get("rel_end"), (int, float))]
        span = {
            "start": round(min(starts), 2) if starts else None,
            "end": round(max(ends), 2) if ends else None,
            "seconds": round(sum(seconds(r) for r in run), 2),
            "n_seg": len(run),
        }
        span.update(recompute(run))
        out.append(span)
    return out


def _distance(a: float | None, b: float | None, relative: bool) -> float:
    """How far apart two spans' measurements are. Unknown values are not a difference."""
    if a is None or b is None:
        return 0.0
    if not relative:
        return abs(a - b)
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale else 0.0


def value_spans(rows: Sequence[dict], value_of: Callable[[Sequence[dict]], float | None],
                recompute: Callable[[Sequence[dict]], dict], min_delta: float,
                relative: bool = True, max_gap: float = DEFAULT_MAX_GAP,
                min_span_seconds: float = DEFAULT_MIN_SPAN_SECONDS) -> list[dict]:
    """A timeline segmented by how much a **measurement** moves, not by its bucket.

    Run-length encoding a bucketed label makes the bucket edges the definition of
    "changed", and they are not: on German edges of (14.2, 22.4), 15.1 and 13.6 cps sit
    either side of a boundary and read as a change though they differ by 10%, while 14.5
    and 22.0 are both ``measured`` and read as no change though the second is half again
    as fast. Where the underlying quantity is continuous, the honest question is how far
    it moved.

    So spans are built bottom-up: every segment starts alone, and the adjacent pair
    whose values are closest is merged, repeatedly, until every remaining boundary is a
    move of at least ``min_delta``. What survives is exactly the set of changes big
    enough to be worth reporting, and the boundaries land where the quantity actually
    moved rather than where a threshold happens to sit.

    ``relative`` compares by ratio rather than by absolute difference, which is what a
    rate needs: characters per second runs ~4 in Chinese and ~15 in English, so one
    absolute threshold cannot serve both.

    Adjacent spans may well publish the same bucket label here, and that is the point —
    two stretches both called ``measured`` that differ by 30% are a real change of pace.
    """
    ordered = _ordered(rows)
    if not ordered:
        return []
    runs: list[list[dict]] = [[row] for row in ordered]

    guard = 0
    while len(runs) > 1 and guard < 10_000:
        guard += 1
        values = [value_of(run) for run in runs]
        touching = [_touching(runs[i], runs[i + 1], max_gap) for i in range(len(runs) - 1)]

        # Merge on value first. The duration floor is a constraint on the spans that
        # come out, not a driver of the merge order, and applying it first is actively
        # harmful: every segment starts alone, so a one-second outlier gets absorbed
        # while its neighbours are still single segments, drags one of them far enough
        # to look like a phase, and that phase then survives on its own. Merging by
        # value first gives the outlier a long neighbour to disappear into.
        pick = None
        candidates = [(_distance(values[i], values[i + 1], relative), i)
                      for i in range(len(runs) - 1) if touching[i]]
        if candidates:
            delta, i = min(candidates)
            if delta < min_delta:
                pick = (i, i + 1)

        if pick is None:
            for i, run in enumerate(runs):
                if sum(seconds(r) for r in run) >= min_span_seconds:
                    continue
                options = ([i - 1] if i > 0 and touching[i - 1] else []) + \
                          ([i + 1] if i + 1 < len(runs) and touching[i] else [])
                if options:
                    # nearest in value, and on a tie the longer host, so a stray second
                    # is diluted rather than allowed to redefine a short neighbour
                    pick = (i, min(options, key=lambda j: (
                        _distance(values[i], values[j], relative),
                        -sum(seconds(r) for r in runs[j]))))
                    break
        if pick is None:                # every boundary is a real move, every span fits
            break

        lo, hi = sorted(pick)
        runs[lo] = runs[lo] + runs[hi]
        runs.pop(hi)

    return _assemble(runs, recompute)


def _speed_of_span(run: Sequence[dict]) -> dict:
    """Span speaking rate: total characters over total seconds, then re-bucketed.

    ``cps_i * seconds_i`` is ``characters_i``, so the duration-weighted mean of the
    per-segment rates is the total ratio exactly — the plain mean of the rates is not,
    and would let a one-second segment count as much as a ten-second one.
    """
    cps = weighted_mean(run, "speed_cps")
    lang, _ = weighted_vote(run, "lang")
    edges = SPEED_CPS_EDGES.get(lang or "")
    return {"speed": bucket(cps, edges, SPEED_LABELS) if edges else None,
            "speed_cps": round(cps, 1) if cps is not None else None,
            "lang": lang}


def _volume_of_span(run: Sequence[dict]) -> dict:
    lufs = combine_lufs(run)
    return {"volume": bucket(lufs, VOLUME_EDGES, VOLUME_LABELS),
            "volume_lufs": round(lufs, 1) if lufs is not None else None}


def _emotion_of_span(run: Sequence[dict], tau: float | None) -> dict:
    """Gate each member, then let the survivors vote by duration.

    The supplied threshold applies to each clip. The span reports the longest-running
    surviving label and the share of its duration whose clips passed the gate.
    """
    scores = mean_scores(run)
    if not scores:
        return {"emotion": None, "emotion_confidence": None, "emotion_top3": None}
    top = max(scores, key=scores.get)

    survivors, confidences, passed, total = defaultdict(float), [], 0.0, 0.0
    for row in run:
        member = row.get("emotion_scores")
        w = seconds(row)
        total += w
        if not isinstance(member, dict) or not member:
            continue
        label = max(member, key=member.get)
        gated = gate_emotion({"label": label, "confidence": member[label],
                              "scores": member}, tau)
        if gated["emotion"]:
            survivors[gated["emotion"]] += w
            confidences.append((w, float(member[label])))
            passed += w

    label = max(survivors, key=lambda k: (survivors[k], k)) if survivors else None
    confidence = (sum(w * c for w, c in confidences) / sum(w for w, _ in confidences)
                  if confidences else scores[top])
    return {"emotion": label,
            "emotion_confidence": round(confidence, 4),
            "emotion_top3": {k: round(v, 3)
                             for k, v in sorted(scores.items(), key=lambda kv: -kv[1])[:3]},
            "emotion_support": round(passed / total, 3) if total else 0.0,
            "emotion_argmax": top}


def _emotion_label(row: dict) -> object:
    """Per-segment emotion for the run-length encoding: the *ungated* argmax.

    Gating segment by segment would blank most of them and the timeline would be
    mostly holes. The threshold belongs at span level, after averaging.
    """
    scores = row.get("emotion_scores")
    if isinstance(scores, dict) and scores:
        return max(scores, key=scores.get)
    return row.get("emotion")


def _dominant(span_list: Sequence[dict], key: str) -> tuple[object, float]:
    """Longest-running non-null label across spans, and how varied the rest is.

    Variability is the share of labelled time *not* spent on the dominant label: 0.0 is
    a recording that never changes, and it rises as the timeline breaks up.
    """
    totals: dict[object, float] = defaultdict(float)
    for span in span_list:
        label = span.get(key)
        if label is None:
            continue
        totals[label] += float(span.get("seconds") or 0.0)
    if not totals:
        return None, 0.0
    total = sum(totals.values())
    winner = max(totals, key=lambda k: (totals[k], str(k)))
    return winner, round(1.0 - (totals[winner] / total if total else 1.0), 3)


# ------------------------------------------------------------------------ assembly
def aggregate(rows: Sequence[dict], tau: float | None = None,
              max_gap: float = DEFAULT_MAX_GAP,
              min_span_seconds: float = DEFAULT_MIN_SPAN_SECONDS,
              speed_delta: float = DEFAULT_SPEED_DELTA) -> dict:
    """One group of segment annotations -> one file-level record.

    ``rows`` are the segment rows of a single container (or, later, a single speaker
    within one), each carrying the annotate pass's fields plus ``rel_start`` /
    ``rel_end`` and — for span-level emotion gating — the raw ``emotion_scores``.
    """
    rows = [r for r in rows if seconds(r) > 0]
    if not rows:
        return {}
    if any(r.get("emotion_scores") for r in rows) and (tau is None or not 0 <= tau <= 1):
        raise ValueError("an explicit emotion threshold in [0, 1] is required")

    speech = sum(seconds(r) for r in rows)
    file_seconds = next((float(r["parent_duration"]) for r in rows
                         if isinstance(r.get("parent_duration"), (int, float))), None)

    record: dict = {
        "n_segments": len(rows),
        "speech_seconds": round(speech, 2),
        "file_seconds": round(file_seconds, 2) if file_seconds else None,
        "speech_ratio": round(speech / file_seconds, 3) if file_seconds else None,
    }
    record.update(aggregate_speaker(rows))

    # boundaries come from the ungated argmax (gating first would leave the timeline
    # mostly holes), but neighbours are merged on the *published* label — two spans
    # that both say "neutral", or both say nothing, are one stretch however their raw
    # posteriors differed
    emotion_spans = spans(rows, _emotion_label, lambda run: _emotion_of_span(run, tau),
                          "emotion", max_gap, min_span_seconds)
    # speed is segmented on the measurement, not on the bucket — see value_spans
    speed_spans = value_spans(rows, lambda run: weighted_mean(run, "speed_cps"),
                              _speed_of_span, speed_delta, relative=True,
                              max_gap=max_gap, min_span_seconds=min_span_seconds)
    previous = None
    for span in speed_spans:
        cps = span.get("speed_cps")
        if previous is not None and cps is not None:
            span["speed_cps_delta"] = round(cps - previous, 1)
            span["speed_change"] = round((cps - previous) / previous, 3) if previous else None
        if cps is not None:
            previous = cps
    volume_spans = spans(rows, lambda r: r.get("volume"), _volume_of_span,
                         "volume", max_gap, min_span_seconds)

    record["emotion_spans"] = emotion_spans
    record["speed_spans"] = speed_spans
    record["volume_spans"] = volume_spans

    # flat scalars for the caption prompt builder, which reads these keys directly
    for key, span_list in (("emotion", emotion_spans), ("speed", speed_spans),
                           ("volume", volume_spans)):
        label, variability = _dominant(span_list, key)
        record[key] = label
        record[f"{key}_variability"] = variability
        record[f"n_{key}_spans"] = len(span_list)

    # how far the rate actually travels, which the label-share variability cannot say:
    # a recording can hold one bucket end to end and still swing 40%
    rates = [s["speed_cps"] for s in speed_spans if s.get("speed_cps") is not None]
    record["speed_cps_spread"] = (round((max(rates) - min(rates)) / max(rates), 3)
                                  if len(rates) > 1 and max(rates) else 0.0)
    return record
