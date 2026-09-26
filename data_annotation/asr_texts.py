"""Share raw ASR results across the short, long, and dialogue tracks.

Each track applies its own acceptance rules when assembling output. The table
stores transcription facts, not a track-specific keep-or-drop decision.
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field


@dataclass
class SegText:
    """One segment's raw ASR output before track-specific filtering."""

    text: str = ""
    language: str | None = None

    accepted: bool = False


    asr_model: str = ""

    extra: dict = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.text


def build_row(result, asr, asr_model: str) -> SegText:
    """Convert one adapter result into a reusable segment-text record."""
    if result is None:
        return SegText(asr_model=asr_model)
    text = (result.text or "").strip()
    lang = result.language
    return SegText(text=text, language=lang, accepted=bool(asr.accepts(lang)),
                   asr_model=asr_model, extra=dict(result.extra or {}))


REPEAT_MIN_COUNT = 5


def _repeated_ngram_coverage(tokens: list[str], ngram: int) -> float:
    """Coverage of tokens participating in an n-gram seen ≥ REPEAT_MIN_COUNT times."""
    if len(tokens) < ngram * 3:
        return 0.0
    grams = [tuple(tokens[i:i + ngram])
             for i in range(len(tokens) - ngram + 1)]
    repeated = {gram for gram, count in collections.Counter(grams).items()
                if count >= REPEAT_MIN_COUNT}
    if not repeated:
        return 0.0
    covered: set[int] = set()
    for i, gram in enumerate(grams):
        if gram in repeated:
            covered.update(range(i, i + ngram))
    return len(covered) / len(tokens)


def is_asr_chunk_repetition(text: str, speech_s: float) -> bool:
    """High-confidence decoding repetition inside ONE ASR request.

    This is deliberately not a song/chorus classifier. Callers must pass one
    model output and must never concatenate members or dialogue turns first.
    Requiring a ``REPEAT_MIN_COUNT``x (=5) repeated n-gram to cover at least
    80% avoids ordinary local word reuse while catching the common LLM-ASR
    loop failure. Word tokens handle whitespace languages; a punctuation/
    space-free character view covers CJK and other unsegmented scripts.
    """
    _ = speech_s  # callers naturally have chunk duration; repetition is textual
    text = (text or "").strip().casefold()
    if not text:
        return False
    words = re.findall(r"[^\W_]+(?:'[^\W_]+)?", text, flags=re.UNICODE)
    if len(words) >= 18 and _repeated_ngram_coverage(words, 4) >= 0.80:
        return True
    # Character n-grams are for text that word tokenisation cannot represent
    # well (notably CJK without spaces), not a second noisy pass over English.
    if len(words) < 18:
        chars = [ch for ch in text if ch.isalnum()]
        if len(chars) >= 30 and _repeated_ngram_coverage(chars, 8) >= 0.80:
            return True
    return False


def content_verdict(st: SegText, speech_s: float) -> str | None:
    """Track-independent content-form gates after ASR language acceptance.

    The ASR language label is metadata, not a TTS text-quality target. In
    particular, never reject a sample merely because its Unicode script does
    not agree with that label; audio/text alignment is what matters here.
    """
    if is_asr_chunk_repetition(st.text, speech_s):
        return "asr_repetition"
    return None


def short_verdict(st: SegText, seg_dnsmos, lid_language: str | None,
                  lid_raw: str | None, is_fake: bool, dur_s: float,
                  *, min_char: int, ratio_filter, zh_en_floor: float,
                  dialect_floor: float, drop_fake_zh_en: bool,
                  char_count, lid_is_dialect) -> str | None:

    if st.empty and not st.language:
        return "no_result"
    if not st.accepted:
        return "lang"
    content_reason = content_verdict(st, dur_s)
    if content_reason is not None:
        return content_reason
    if char_count(st.text) < min_char:
        return "text"

    detected = st.language
    lid = (lid_language or "").strip().lower()
    if detected in ("zh", "en") and lid not in ("zh", "en"):
        floor = (dialect_floor
                 if (st.extra.get("is_dialect") and lid_is_dialect(lid_raw))
                 else zh_en_floor)
        if (seg_dnsmos is None or float(seg_dnsmos) < floor
                or (drop_fake_zh_en and is_fake)):
            return "post_asr_zh_en"


    chars = char_count(st.text)
    if chars == 0 or dur_s <= 0:
        return "ratio"
    bounds = ratio_filter.get(detected) or ratio_filter.get("default")
    if bounds is not None and not (bounds.min <= dur_s / chars <= bounds.max):
        return "ratio"
    return None


def chunk_verdict(st: SegText, speech_s: float, *, min_char: int = 1,
                  ratio_filter=None, char_count=None,
                  check_repetition: bool = True) -> str | None:

    if char_count is None:
        from utils.tool import get_char_count as char_count
    if st.empty and not st.language:
        return "no_result"
    if not st.accepted:
        return "lang_rejected"
    if check_repetition:
        content_reason = content_verdict(st, speech_s)
        if content_reason is not None:
            return content_reason
    chars = char_count(st.text)
    if chars == 0:
        return "blank"
    if chars < min_char:
        return "too_short"
    if speech_s <= 0:
        return "ratio"
    if ratio_filter:
        bounds = ratio_filter.get(st.language) or ratio_filter.get("default")
        if bounds is not None and not (bounds.min <= speech_s / chars <= bounds.max):
            return "ratio"
    return None


long_verdict = chunk_verdict
dialogue_verdict = chunk_verdict


OK = "ok"
SILENCE = "silence"
BREAK = "break"


_ROLE_BY_REASON = {
    None: OK,
    "blank": SILENCE,
    "too_short": SILENCE,
    "lang_rejected": BREAK,
    "ratio": BREAK,
    "asr_repetition": BREAK,
    "no_result": BREAK,
    "lang": BREAK,
}


def role_of(reason: str | None) -> str:

    return _ROLE_BY_REASON.get(reason, BREAK)


def assemble_runs(items: list, roles: list[str], *, max_gap_s: float,
                  min_duration_s: float, min_speakers: int = 1,
                  speaker_of=None) -> list[list[int]]:

    if not items:
        return []
    n = len(items)


    pieces: list[list[int]] = []
    cur: list[int] = []
    for i in range(n):
        if roles[i] == BREAK:
            if cur:
                pieces.append(cur)
                cur = []
            continue
        cur.append(i)
    if cur:
        pieces.append(cur)

    out: list[list[int]] = []
    for piece in pieces:

        while piece and roles[piece[0]] != OK:
            piece = piece[1:]
        while piece and roles[piece[-1]] != OK:
            piece = piece[:-1]
        if not piece:
            continue


        runs: list[list[int]] = []
        cur = [piece[0]]
        last_ok = piece[0]
        for i in piece[1:]:
            if roles[i] == OK:
                if items[i]["start"] - items[last_ok]["end"] > max_gap_s:
                    runs.append(cur)
                    cur = [i]
                else:
                    cur.append(i)
                last_ok = i
            else:
                cur.append(i)
        runs.append(cur)


        for r in runs:
            while r and roles[r[-1]] != OK:
                r.pop()
            if not r:
                continue

            a, b = items[r[0]]["start"], items[r[-1]]["end"]
            if b - a < min_duration_s:
                continue
            if min_speakers > 1:
                spk = {(speaker_of(items[i]) if speaker_of else items[i].get("speaker"))
                       for i in r if roles[i] == OK}
                if len(spk) < min_speakers:
                    continue
            out.append(r)
    return out


SUFFIX = ".asr_texts.json"


def path_for(sid_dir: str, sid: str) -> str:
    import os
    return os.path.join(sid_dir, sid + SUFFIX)


def dump(table: dict) -> list[dict]:

    return [{"index": k, "text": v.text, "language": v.language,
             "accepted": v.accepted, "asr_model": v.asr_model,
             "extra": v.extra}
            for k, v in sorted(table.items(), key=lambda kv: str(kv[0]))]


def load(path: str) -> dict:

    import json
    try:
        with open(path) as f:
            rows = json.load(f)
    except (OSError, ValueError):
        return {}
    return {r["index"]: SegText(
        text=r.get("text") or "", language=r.get("language"),
        accepted=bool(r.get("accepted")), asr_model=r.get("asr_model") or "",
        extra=r.get("extra") or {}) for r in rows if r.get("index") is not None}


def merge(base: dict, new: dict) -> dict:

    out = dict(base)
    out.update(new)
    return out
