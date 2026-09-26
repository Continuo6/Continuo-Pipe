"""Turning predicted tags into the phrases a prompt can hand to the captioner.

The captioner is told these values and instructed to use them verbatim, so the
wording here is load-bearing: "moderate" and "loud" are the two things the model is
allowed to say about a `normal` and a `loud` clip respectively, and nothing else.

A tag that came back null is **omitted** rather than rendered as "unknown". The
prompts all forbid describing an absent characteristic, and an explicit "unknown"
line invites the model to guess and then commit to the guess.

``Standard Mandarin`` gets special handling in both languages: it is the neutral
standard, not an accent, so it is phrased as *speaking* standard Mandarin rather than
*having* a standard Mandarin accent.

A record aggregated from a long recording (:mod:`continuo_expressive.aggregate`) also
carries ``*_spans`` timelines, and those add one more line describing how volume,
speaking rate and emotion move across the file. It is rendered as positioned data
rather than prose for the same reason every other tag is: the model is here to write,
not to decide what happened. The line appears only when a timeline actually changes —
inviting a caption to narrate a change in a recording that has none is exactly the kind
of confident invention the rest of this module exists to prevent.
"""
from __future__ import annotations

# --------------------------------------------------------------- English
AGE_PHRASE_EN = {"child": "a child", "teen": "a teenager", "young": "a young adult",
                 "middle": "middle-aged", "senior": "elderly"}
SPEED_PHRASE_EN = {"slow": "slow", "measured": "measured (moderate)", "fast": "fast"}
VOLUME_PHRASE_EN = {"loud": "loud", "normal": "moderate", "soft": "soft"}
ACCENT_PHRASE_EN = {"Standard Mandarin": "standard Mandarin",
                    "Zhongyuan": "a Zhongyuan Mandarin accent"}


# --------------------------------------------------- variation over a long recording
#: below this share of speech spent off the dominant label, a recording is steady
#: enough that pointing at "changes" would be reading noise
VARIATION_MIN = 0.15
#: a span covering less than this share of the file is not a phase of the recording
PHASE_MIN_SHARE = 0.05
#: at most this many phases per attribute, so one line does not become a transcript
PHASE_MAX = 6

_ATTRS = ("volume", "speed", "emotion")
_ATTR_EN = {"volume": "volume", "speed": "speaking rate", "emotion": "emotion"}


def _at(start: float, end: float, total: float) -> str:
    """Where in the recording a phase sits, as a timestamp.

    Positions were percentages first and a captioner cannot keep them straight: it read
    a leading "52% slower" as the position, and then, once the magnitude was moved out
    of the way, turned a phase at 46% of a 190 s recording into "around the 46 second
    mark". A clock time is not confusable with either — it cannot be read as a share of
    anything, and the only percentages left in the line are magnitudes.
    """
    del total                                   # positions are absolute, not shares
    return f"({int(start) // 60}:{int(start) % 60:02d}-{int(end) // 60}:{int(end) % 60:02d})"


def _collapse(phases: list[tuple]) -> list[tuple]:
    """Merge neighbouring entries that carry the same label.

    Dropping a short span, or truncating the list, can leave two same-labelled entries
    side by side even though the timeline itself never repeats a label — and telling a
    captioner "measured, then measured, then measured" would be worse than saying
    nothing. Merged entries span from the first start to the last end, which is honest
    about the stretch and silent about what was filtered out of the middle.
    """
    out: list[tuple] = []
    for label, start, end in phases:
        if out and out[-1][0] == label:
            out[-1] = (label, out[-1][1], end)
        else:
            out.append((label, start, end))
    return out


#: how each language says "20% faster, measured (at 3-37%)".
#:
#: A phase holds two percentages meaning different things — where in the recording it
#: sits, and how far the rate moved to get there — so both are marked. The position
#: leads and carries an "at"; the magnitude trails and says what it is measured
#: against. Leading with a bare magnitude was tried and the captioner read it as the
#: position, writing "a slower pace around the 52% mark" for a phase at 33-36%.
SPEED_MOVE_EN = ("faster", "slower", "{label} {where}, {pct} {dir} than before")


def _speed_phases(spans: list[dict], total: float, phrase: dict,
                  move: tuple = SPEED_MOVE_EN) -> list[str]:
    """Speed phases, stated as movement rather than as a sequence of bucket names.

    No share-of-file floor here, unlike the other two. Speed spans have already cleared
    two bars — a minimum duration and a minimum move — so a short one is not noise, it
    is a brief change of pace, and it is usually the short stretches that carry the
    largest moves. Filtering them by share drops exactly the phases worth mentioning
    and leaves two near-identical neighbours saying nothing.
    """
    kept = [{"label": s["speed"], "cps": s.get("speed_cps"), "start": s["start"],
             "end": s["end"], "seconds": s.get("seconds") or 0.0} for s in spans]
    if len(kept) > PHASE_MAX:      # keep the longest, but still tell it in time order
        longest = set(id(p) for p in sorted(kept, key=lambda p: -p["seconds"])[:PHASE_MAX])
        kept = [p for p in kept if id(p) in longest]

    # truncation can leave neighbours that no longer differ; fold those back together
    # so the line never names two phases between which nothing happened
    folded: list[dict] = []
    for phase in kept:
        last = folded[-1] if folded else None
        if last and last["cps"] and phase["cps"] and \
                abs(phase["cps"] - last["cps"]) / max(last["cps"], phase["cps"]) < VARIATION_MIN:
            weight = last["seconds"] + phase["seconds"]
            last["cps"] = ((last["cps"] * last["seconds"] + phase["cps"] * phase["seconds"])
                           / weight) if weight else last["cps"]
            last["label"] = (last if last["seconds"] >= phase["seconds"] else phase)["label"]
            last["end"], last["seconds"] = phase["end"], weight
        else:
            folded.append(dict(phase))
    if len(folded) < 2:
        return []

    faster, slower, template = move
    out, previous = [], None
    for phase in folded:
        where = _at(phase["start"], phase["end"], total)
        label = phrase.get(phase["label"], phase["label"])
        cps = phase["cps"]
        if previous and cps:
            change = (cps - previous) / previous
            out.append(template.format(pct=f"{abs(change):.0%}",
                                       dir=faster if change > 0 else slower,
                                       label=label, where=where))
        else:
            out.append(f"{label} {where}")
        previous = cps or previous
    return out


def _phases(rec: dict, attr: str, phrase: dict,
            speed_move: tuple = SPEED_MOVE_EN) -> list[str]:
    """``label (a-b%)`` for each span of ``attr`` worth calling a phase.

    Speed is the odd one out. Its spans are cut on how far the rate moves rather than
    on which bucket it lands in, so a recording can hold ``measured`` end to end while
    the rate swings 30% — and collapsing those neighbours by label, as volume and
    emotion want, would erase exactly the change worth describing. So speed keeps every
    phase and states the movement between them.
    """
    spans = rec.get(f"{attr}_spans") or []
    total = rec.get("file_seconds") or 0.0
    if not spans or not total:
        return []
    # speed's own measure of movement: the label share cannot see a swing inside a bucket
    moved = (rec.get("speed_cps_spread") if attr == "speed"
             else rec.get(f"{attr}_variability")) or 0.0
    if moved < VARIATION_MIN:
        return []

    if attr == "speed":
        return _speed_phases([s for s in spans if s.get(attr)], total, phrase,
                             speed_move)

    kept = [s for s in spans
            if s.get(attr) and (s.get("seconds") or 0.0) / total >= PHASE_MIN_SHARE]
    usable = _collapse([(s[attr], s["start"], s["end"]) for s in kept])
    if len(usable) > PHASE_MAX:      # keep the longest, but still tell it in time order
        keep = set(id(p) for p in sorted(usable, key=lambda p: -(p[2] - p[1]))[:PHASE_MAX])
        usable = _collapse([p for p in usable if id(p) in keep])
    if len(usable) < 2:              # one phase is not a change
        return []
    return [f"{phrase.get(label, label)} {_at(start, end, total)}"
            for label, start, end in usable]


def variation_line_en(rec: dict) -> str | None:
    phrases = {"volume": VOLUME_PHRASE_EN, "speed": SPEED_PHRASE_EN, "emotion": {}}
    parts = [f"{_ATTR_EN[a]} — " + ", ".join(p)
             for a in _ATTRS if (p := _phases(rec, a, phrases[a], SPEED_MOVE_EN))]
    if not parts:
        return None
    return ("- Change over the recording (timestamps are positions in the recording; "
            "mention this movement in the caption): " + "; ".join(parts))


def variation_line_zh(rec: dict) -> str | None:
    """Return English tag data for the Chinese-output prompt builder."""
    return variation_line_en(rec)


def tag_lines_en(rec: dict) -> list[str]:
    """The `- Field: value` block every English prompt embeds."""
    lines = []
    if rec.get("gender"):
        lines.append(f"- Gender: {rec['gender']}")
    if rec.get("age_band"):
        lines.append(f"- Age: {AGE_PHRASE_EN.get(rec['age_band'], rec['age_band'])}")
    if rec.get("accent"):
        lines.append(f"- Accent: {ACCENT_PHRASE_EN.get(rec['accent'], rec['accent'] + ' accent')}")
    if rec.get("pitch"):
        lines.append(f"- Pitch: {rec['pitch']}-pitched")
    if rec.get("volume"):
        lines.append(f"- Volume: {VOLUME_PHRASE_EN.get(rec['volume'], rec['volume'])}")
    if rec.get("speed"):
        lines.append(f"- Speaking Rate: {SPEED_PHRASE_EN.get(rec['speed'], rec['speed'])}")
    if rec.get("emotion"):      # gated: a null emotion contributes no line at all
        lines.append(f"- Emotion / Expressiveness: {rec['emotion']}")
    line = variation_line_en(rec)     # long recordings only; absent for a single clip
    if line:
        lines.append(line)
    return lines


def tag_lines_zh(rec: dict) -> list[str]:
    """Return English tag data for the Chinese-output prompt builder."""
    return tag_lines_en(rec)
