"""Finalize phase-one dialogue windows after short-track transcription.

Classify segments for text reuse, transcribe missing turns, and split windows
at turns whose audio and text cannot be aligned. This module manages metadata;
the caller owns the ASR model and audio decoding.
"""
from __future__ import annotations

from dialogue.common import (
    assemble_transcript, dominant_languages, merge_units,
    seg_source as _seg_source_of,
)
from dialogue.select import Gates, evaluate, seg_status, untranscribable


def is_phase1_materialized(win: dict) -> bool:
    """Return whether phase one materialized this window."""
    return win.get("materialized_by") == "phase1"


def classify(win: dict, text_map: dict[str, str], part_idx: set[str]) -> None:
    """Assign segment text sources and reusable transcriptions in place."""
    for s in win.get("segments", []):
        status, text = seg_status(s["index"], text_map, part_idx)
        s["seg_source"] = status
        s["phase2_text"] = text


def should_drop(win: dict, text_map: dict[str, str], part_idx: set[str],
                gates: Gates) -> str | None:
    """Apply the legacy empty-window check and return a drop reason."""
    return untranscribable(win.get("segments", []), text_map, part_idx, gates)


def ensure_units(win: dict) -> None:
    """Group turns and reuse text only when every member segment has it."""
    if not win.get("turns"):
        win["turns"] = merge_units(win.get("segments", []))
    by_idx = {s["index"]: s for s in win.get("segments", [])}
    for u in win["turns"]:
        if u.get("text_status") != "pending":
            continue
        srcs = [by_idx.get(i) for i in u.get("seg_indices", [])]
        srcs = [s for s in srcs if s is not None]
        if not srcs:
            continue


        if all(_seg_source_of(s) == "reuse" for s in srcs):
            parts = [(s.get("phase2_text") or "").strip() for s in srcs]
            parts = [p for p in parts if p]
            if parts:
                u["text"] = " ".join(parts)
                u["language"] = next(
                    (s.get("language") for s in srcs if s.get("language")), None)
                u["text_status"] = "reused"


def _unit_from_table(unit: dict, table: dict):

    idxs = unit.get("seg_indices") or []
    if not idxs:
        return None
    rows = [table.get(i) for i in idxs]
    if any(r is None for r in rows):
        return None
    return rows


def fill_from_table(win: dict, table: dict, *, min_char: int, ratio_filter) -> int:

    import asr_texts
    from utils.tool import get_char_count

    n = 0
    for u in pending_units(win):
        rows = _unit_from_table(u, table)
        if rows is None:
            continue
        text = " ".join(r.text for r in rows if r.text).strip()
        lang = next((r.language for r in rows if r.language), None)
        st = asr_texts.SegText(
            text=text, language=lang,
            accepted=all(r.accepted for r in rows),
            asr_model=rows[0].asr_model,
            extra=dict(rows[0].extra))
        dur = u.get("speech_s") or (u["end"] - u["start"])
        # A merged dialogue unit may reuse several independently transcribed
        # table rows. Test repetition per row, never on their concatenation.
        reason = next((
            verdict for r in rows
            if (verdict := asr_texts.content_verdict(
                r, float(r.extra.get("speech_s") or 0.0))) is not None
        ), None)
        if reason is None:
            reason = asr_texts.chunk_verdict(
                st, dur, min_char=min_char, ratio_filter=ratio_filter,
                char_count=get_char_count, check_repetition=False)
        if reason is None:
            u["text"], u["language"], u["text_status"] = text, lang, "ok"
            if st.extra.get("asr_lang_raw"):
                u["asr_lang_raw"] = st.extra["asr_lang_raw"]
        else:
            u["text_status"] = "empty"
            u["empty_reason"] = reason
            u["asr_detected"] = lang
            u["asr_chars"] = get_char_count(text)
            if text:
                u["asr_text_rejected"] = text
        u["from_table"] = 1
        n += 1
    return n


def rescue_from_table(win: dict, table: dict) -> int:

    import asr_texts

    n = 0
    for u in rescue_targets(win):
        rows = _unit_from_table(u, table)
        if rows is None:
            continue
        text = " ".join(r.text for r in rows if r.text).strip()
        lang = next((r.language for r in rows if r.language), None)
        # Table rows are separate ASR requests; do not test their joined text.
        repeated = next((r for r in rows if asr_texts.is_asr_chunk_repetition(
            r.text, float(r.extra.get("speech_s") or 0.0))), None)
        if repeated is not None:
            u["rescue_tried"] = 1
            u["rescue_skip"] = "asr_repetition"
            u["asr_text_rejected"] = repeated.text
            continue
        res = type("_R", (), {"text": text, "language": lang,
                              "extra": dict(rows[0].extra)})()
        if apply_rescue(u, res, source="phase2-table",
                        check_repetition=False):
            n += 1
    return n


def pending_units(win: dict) -> list[dict]:

    return [u for u in win.get("turns", []) if u.get("text_status") == "pending"]


def ratio_ok(dur: float, lang: str | None, text: str, ratio_filter) -> bool:

    from utils.tool import get_char_count

    chars = get_char_count(text)
    if chars == 0 or dur <= 0:
        return False
    bounds = ratio_filter.get(lang) or ratio_filter.get("default")
    return True if bounds is None else (bounds.min <= dur / chars <= bounds.max)


def apply_result(unit: dict, result, asr, min_char: int, ratio_filter) -> bool:

    import asr_texts
    from utils.tool import get_char_count

    text = (result.text or "").strip() if result else ""
    detected = (result.language if result else None) or unit.get("lang_hint")

    dur = unit.get("speech_s") or (unit["end"] - unit["start"])
    st = asr_texts.SegText(
        text=text, language=(detected if result is not None else None),
        accepted=bool(result is not None and asr.accepts(detected)),
        asr_model=type(asr).__name__,
        extra=dict((result.extra if result is not None else None) or {}),
    )
    reason = asr_texts.chunk_verdict(
        st, dur, min_char=min_char, ratio_filter=ratio_filter,
        char_count=get_char_count,
    )
    if reason is None:
        unit["text"], unit["language"], unit["text_status"] = text, detected, "ok"
        if result.extra.get("asr_lang_raw"):
            unit["asr_lang_raw"] = result.extra["asr_lang_raw"]
        return True

    ch = get_char_count(text)
    unit["text_status"] = "empty"
    unit["empty_reason"] = reason
    unit["asr_detected"] = detected
    unit["asr_chars"] = ch
    if text:
        unit["asr_text_rejected"] = text
    return False


def rescue_targets(win: dict) -> list[dict]:

    return [u for u in win.get("turns", [])
            if u.get("text_status") in ("pending", "unroutable", "empty")
            and not u.get("rescue_tried")]


def apply_rescue(unit: dict, result, source: str = "phase2", *,
                 check_repetition: bool = True) -> bool:

    from dialogue.common import is_degenerate

    unit["rescue_tried"] = 1
    text = (result.text or "").strip() if result else ""
    if result is not None and result.language:
        unit["asr_detected"] = result.language
    if result is not None and result.extra.get("asr_lang_raw"):
        unit["asr_lang_raw"] = result.extra["asr_lang_raw"]
    if not text:
        return False
    if is_degenerate(text, unit["end"] - unit["start"]):
        unit["rescue_skip"] = "degenerate"
        unit["asr_text_rejected"] = text
        return False
    if check_repetition:
        import asr_texts
        if asr_texts.is_asr_chunk_repetition(
                text, unit["end"] - unit["start"]):
            unit["rescue_skip"] = "asr_repetition"
            unit["asr_text_rejected"] = text
            return False

    unit["rescue_prev_status"] = unit.get("text_status")
    if unit.get("empty_reason") is not None:
        unit["rescue_prev_reason"] = unit.pop("empty_reason")
    unit["text"] = text
    unit["text_status"] = "ok"
    unit["rescued"] = source
    unit.pop("asr_text_rejected", None)
    return True


def _lid_unsupported_turn(turn: dict, by_idx: dict) -> bool:

    pred = _LID_UNSUPPORTED
    if pred is None:
        return False
    idxs = turn.get("seg_indices") or []
    if not idxs:
        return False
    langs = [(by_idx.get(i) or {}).get("language") for i in idxs]
    return bool(langs) and all(pred(l) for l in langs)


_LID_UNSUPPORTED = None


def set_lid_unsupported(pred) -> None:

    global _LID_UNSUPPORTED
    _LID_UNSUPPORTED = pred


def mark_lid_unsupported(win: dict) -> int:

    pred = _LID_UNSUPPORTED
    if pred is None:
        return 0
    by_idx = {s["index"]: s for s in (win.get("segments") or [])}
    n = 0
    for u in win.get("turns") or []:
        if u.get("text_status") == "pending" and _lid_unsupported_turn(u, by_idx):
            u["text_status"] = "lid_unsupported"
            n += 1
    return n


def _turn_role(turn: dict, by_idx: dict | None = None) -> str:

    import asr_texts

    st = turn.get("text_status")


    if st == "lid_unsupported":
        return asr_texts.BREAK
    if by_idx is not None and _lid_unsupported_turn(turn, by_idx):
        return asr_texts.BREAK
    if st in ("ok", "reused"):
        return asr_texts.OK
    if st == "unroutable":
        return asr_texts.BREAK
    if st == "empty":
        return asr_texts.role_of(turn.get("empty_reason"))
    if st == "pending":

        return asr_texts.BREAK
    return asr_texts.role_of(st)


def split_window(win: dict, gates: Gates) -> list[dict]:

    import asr_texts

    turns = win.get("turns") or []
    if not turns:
        return []
    by_idx = {s["index"]: s for s in (win.get("segments") or [])}
    roles = [_turn_role(t, by_idx) for t in turns]
    runs = asr_texts.assemble_runs(
        turns, roles, max_gap_s=gates.max_gap_s,
        min_duration_s=gates.min_duration_s, min_speakers=2)

    if len(runs) == 1 and len(runs[0]) == len(turns) and all(
            r == asr_texts.OK for r in roles):
        return [win]

    frs = [[turns[i] for i in run] for run in runs]
    sr = int(win["sample_rate"])
    lo = int(win["carrier_start_samples"])
    hi = int(win["carrier_end_samples"])
    segs_by_idx = {s["index"]: s for s in win.get("segments", [])}

    out: list[dict] = []
    for fr in frs:

        a_s, b_s = fr[0]["start"], max(t["end"] for t in fr)
        speakers = [t.get("speaker") for t in fr]

        idxs = [i for t in fr for i in (t.get("seg_indices") or [])]
        segs = [dict(segs_by_idx[i]) for i in idxs if i in segs_by_idx]
        segs.sort(key=lambda s: s.get("start", 0))


        order: list = []
        for spk in speakers:
            if spk not in order:
                order.append(spk)
        smap = {spk: f"S{i + 1}" for i, spk in enumerate(order)}

        turns = [{**t, "tag": smap.get(t.get("speaker"), t.get("tag"))} for t in fr]
        for s in segs:
            s["tag"] = smap.get(s.get("speaker"), s.get("tag"))

        # A fragment is a new candidate: re-apply every quality/alignment gate.
        gate_rows = segs or [
            {
                "start": t["start"], "end": t["end"],
                "speaker": t.get("speaker"),
                # Legacy metadata may not carry segments; preserve its known
                # parent quality rather than silently bypassing structure gates.
                "dnsmos": win.get("mean_dnsmos", gates.min_mean_dnsmos),
                "is_fake": None,
            }
            for t in fr
        ]
        reason, metrics = evaluate(gate_rows, gates)
        if reason is not None:
            continue
        a_s, b_s = metrics["start"], metrics["end"]

        child = dict(win)
        child["turns"] = turns
        child["segments"] = segs
        child["speaker_map"] = smap
        child.update(metrics)
        child["carrier_start_samples"] = max(lo, min(hi, int(a_s * sr)))
        child["carrier_end_samples"] = max(child["carrier_start_samples"],
                                           min(hi, int(b_s * sr)))

        child["split_from"] = win["index"]
        out.append(child)

    for i, child in enumerate(out):
        child["index"] = f"{win['index']}_p{i}"
    return out


def finalize(win: dict) -> None:

    units = sorted(win.get("turns", []), key=lambda t: t.get("start", 0))
    win["transcript"] = assemble_transcript(units)
    win["languages"] = dominant_languages(units)
    win["num_asr_units"] = len(units)
    speaker_turns = 0
    previous = object()
    for unit in units:
        speaker = unit.get("speaker")
        if speaker != previous:
            speaker_turns += 1
            previous = speaker
    win["num_speaker_turns"] = speaker_turns


def prepare(win: dict, text_map: dict[str, str], part_idx: set[str],
            gates: Gates, final_pass: bool = True) -> str | None:

    classify(win, text_map, part_idx)
    if final_pass:
        reason = should_drop(win, text_map, part_idx, gates)
        if reason is not None:
            return reason
    ensure_units(win)

    mark_lid_unsupported(win)
    return None
