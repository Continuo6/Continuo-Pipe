"""English prompt builders for the four instruction forms.

The guard rails repeat across forms because they defend against different failure
modes, all observed:

*Content leakage* — the captioner listens to speech, understands it, and retells it.
An instruction that describes what was said is useless for TTS, so every form is told
to judge the voice as if the language were unintelligible.

*Tag override* — the model substitutes its own impression for a supplied tag, most
often calling an energetic delivery "loud" when the meter says moderate. Hence the
explicit volume clause in nearly every rule block.

``rp`` is the deliberate exception: it may invent a scene and persona, and treats the
tags as a guide rather than a checklist. It still may not contradict a tag, and it
still may not take its imagery from the words.
"""
from __future__ import annotations

from .tags import tag_lines_en

# Appended to every form: the standard is not an accent.
ACCENT_NOTE = (" If the accent is standard Mandarin, refer to it simply as speaking standard Mandarin; "
               "do NOT call it a standard Mandarin accent.")

NOLEAK = ("Never quote, paraphrase, or retell what is being said — not the words, events, actions, names, or plot; "
          "and never use quotation marks. Judge the voice by its SOUND (pitch, pace, volume, energy, timbre), NOT "
          "by the meaning of the words — as if you did not understand the language.")
#: Same prohibition, minus the one clause that stops making sense once the words are on
#: the page. "As if you did not understand the language" is the right instruction when
#: the model is listening; handed a transcript it is simply not followable, and a rule
#: that cannot be followed gets ignored wholesale rather than in part.
NOLEAK_TEXT = ("Never quote, paraphrase, or retell what is being said — not the words, events, actions, names, or "
               "plot; and never use quotation marks. The transcript is there to place the speaker, not to be "
               "described: it may inform the general register and setting, and nothing else. Every acoustic "
               "statement comes from the tags, never from what the words suggest the delivery must have been.")
VOICE_ONLY = ("Describe only the human voice (not background sounds or audio quality). State the volume exactly "
              "as the tag says (loud / moderate / soft) — an energetic or fast delivery is not the same as loud volume.")
RP_GUARD = ("Build the role, mood, and imagery ONLY from how the VOICE SOUNDS — its pitch, pace, volume, energy, and "
            "timbre — as if the words were in a language you do not understand. NEVER let the topic, plot, events, "
            "characters, or meaning of what is said shape the role or scene; never quote or retell it; no quotation "
            "marks. Lean on evocative imagery, not a list of acoustic parameters, mentioning concrete attributes only "
            "sparingly — but any you DO mention must match its tag and never contradict it (never call a `loud` voice "
            "soft, or a `high` pitch low). Invent your own imagery — do not reuse the example's.")
#: Unused for now: only `free` is shown the transcript, and `rp` is precisely the form
#: that must not take its imagery from the words. Kept because the wording is the
#: starting point if `rp` is ever given the transcript — but see the note in
#: `_free_rules` first: the clause this drops is the one doing the work.
RP_GUARD_TEXT = ("Build the role, mood, and imagery from how the VOICE SOUNDS — its pitch, pace, volume, energy, and "
                 "timbre, as given in the tags. The transcript may place the setting in the broadest terms; it must "
                 "NEVER supply the plot, the events, the characters, or any specific thing that is said, and must "
                 "never be quoted or retold; no quotation marks. Lean on evocative imagery, not a list of acoustic "
                 "parameters, mentioning concrete attributes only sparingly — but any you DO mention must match its "
                 "tag and never contradict it (never call a `loud` voice soft, or a `high` pitch low). Invent your "
                 "own imagery — do not reuse the example's.")


FREE_RULES = """CRITICAL RULES
1. NEVER describe, quote, paraphrase, or retell the content of the speech — not the words spoken, and not the specific events, actions, names, or plot being talked about. NEVER contain quotation marks (""). What is being said may ONLY loosely inform the GENERAL mood or register; it must never be reproduced or referred to specifically.
2. Describe the HUMAN VOICE strictly from the tags above, stating each tag's value verbatim — do NOT override an acoustic tag with your own impression of the audio. State the Volume EXACTLY as labeled (loud / moderate / soft): never soften a `loud` to moderate, and never upgrade a `moderate` to loud — an energetic, fast, or expressive delivery is NOT the same as loud volume. Do NOT describe literal background sounds or audio-recording quality.
3. NEVER mention the absence of a characteristic (describe only what is present).
4. The speaker's identity and scene are a PLAUSIBLE, GENERAL inference from the voice — a broad persona and setting (e.g. a formal announcement, an intimate conversation, or a dramatic narration), NOT a retelling of what is happening in the utterance. Keep them grounded, not wild, overly specific, or a paraphrase of the content.
5. Failure to follow these rules will result in an invalid output."""

FREE_EXAMPLE = ("A young male with a clear, medium-high pitched voice and an American accent speaks in a "
                "casual, conversational style. He begins at a fast, rushed pace with a highly energetic "
                "and emphatic intonation, using a high pitch to express strong emphasis, and maintains a "
                "loud volume with an expressive, fluctuating tone throughout the fluent delivery. He comes "
                "across as a tech reviewer or vlogger, enthusiastically walking an online audience through "
                "a product in an informal, self-recorded session.")
APS_EXAMPLE = "Male, 40–55 years old; low-pitched, deep timbre; measured pace, moderate volume; standard Mandarin."
DSD_EXAMPLE = ("A deep, resonant voice at a medium pitch, delivered at a steady, measured pace and a moderate "
               "volume, with a calm and composed emotional undertone that conveys a mature, steady, and "
               "responsible character.")

#: characters of transcript a prompt may carry. Dialogue transcripts run to 80k
#: characters (a 70-minute container), and the captioner's context is 8,192 tokens with
#: ~1k reserved for the answer; one over-long prompt failed a whole chunk of 64 in vLLM.
#: 3,000 characters is ~750 English or ~2,000 Chinese tokens, and the opening minutes
#: are what identity and scene are read from anyway.
TRANSCRIPT_MAX_CHARS = 3000


def clip_transcript(text: str, limit: int = TRANSCRIPT_MAX_CHARS) -> str:
    """The transcript cut to ``limit`` characters at a word boundary, marked as cut."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut[limit // 2:]:
        cut = cut[:cut.rfind(" ")]
    return cut + " […] (transcript cut here; the recording continues)"


def transcript_block(rec: dict) -> str:
    """The words, when the captioner is working from text instead of audio.

    Only ``free`` gets this. It is the one form whose task includes inferring an
    identity and a scene, so the words can tell it something the tags cannot; ``aps``
    and ``dsd`` are acoustic descriptions with nothing to gain, and ``rp`` is the form
    that must build its imagery from sound alone — handed the transcript it quoted the
    topics straight out of it, quotation marks and all, which is the failure the
    original guard existed to prevent.
    """
    text = clip_transcript((rec.get("txt") or "").strip())
    if not text:
        return ""
    return ("\n\nTranscript of the recording (context only — never quote or retell it):\n"
            + text + "\n")


def _free_rules(rec: dict) -> str:
    """Rule 1 loses its "as if you did not understand the language" clause on text.

    That clause is unfollowable once the words are on the page, and an unfollowable
    rule gets dropped whole rather than in part — which is how the transcript ends up
    quoted. What replaces it keeps every prohibition and drops only the pretence.
    """
    if not (rec.get("txt") or "").strip():
        return FREE_RULES
    return FREE_RULES.replace(
        "What is being said may ONLY loosely inform the GENERAL mood or register; it "
        "must never be reproduced or referred to specifically.",
        "The transcript is given only to place the speaker: it may inform the GENERAL "
        "register and setting and nothing else, and must never be reproduced, quoted, "
        "or referred to specifically — not a topic, not a phrase, not a word of it.")


# Structurally different openings (scene-first / role-first / sensory / directive /
# arc), rotated per clip so the corpus does not end up cloning one shape.
RP_EXAMPLES = [
    "Imagine a late-night radio host easing a restless city toward calm — let your voice stay low and unhurried, each phrase settling like the last light left on in a quiet room.",
    "Rain streaks the window of an all-night diner; speak as the weary waitress who has seen every kind of trouble walk in — worn, kind, her words drifting out slow and soft.",
    "A street magician works the summer crowd, bright and quick, one flourish tumbling into the next, every word daring you not to look away.",
    "Picture a lighthouse keeper narrating to an empty sea: deliberate and deep, each sentence rolling in like a wave that has traveled a long way to arrive.",
    "Step under the big top as the ringmaster — grand and theatrical, flinging every word to the back row with a showman's relish.",
    "An old friend leans in across a candlelit table, confiding and close, a smile slipping into the quiet spaces between the words.",
]


def build_free(rec: dict, include_example: bool = False) -> str:
    return (
        "Your task is to generate ONE caption that (a) describes the characteristics of the speaker's "
        "voice and (b) infers a plausible identity and scene for the speaker.\n\n"
        "Use the following tags EXACTLY as given — describe each with the labeled value, do NOT replace "
        "it with your own impression from the audio:\n" + "\n".join(tag_lines_en(rec))
        + transcript_block(rec) + "\n\n"
        "Then, in the same caption, briefly infer WHO the speaker is likely to be (a plausible role or "
        "persona) and WHAT scene or situation they seem to be in, consistent with the voice.\n\n"
        + _free_rules(rec) + ACCENT_NOTE
        + (("\n\nGood Example\n" + FREE_EXAMPLE) if include_example else "") + "\n\nYOUR CAPTION:"
    )


def build_aps(rec: dict) -> str:
    return ("Task: produce an APS (acoustic-parameter spec) — a terse, semicolon-separated list of voice "
            "attributes only. No sentences, no scene, no identity, no speech content.\n\n"
            "Use these tags exactly (do not alter them):\n" + "\n".join(tag_lines_en(rec)) + "\n\n"
            "Rules: " + NOLEAK + " " + VOICE_ONLY + ACCENT_NOTE
            + " Use only the given tags; do not guess timbre/personality/scene.\n\n"
            "Example (format only, do not copy content):\n" + APS_EXAMPLE + "\n\nYOUR APS:")


def build_dsd(rec: dict) -> str:
    return ("Task: produce a DSD (descriptive-style directive) — flowing prose describing pitch, pace, volume, "
            "emotional undertone, and general character/temperament.\n\n"
            "Use these tags exactly:\n" + "\n".join(tag_lines_en(rec)) + "\n\n"
            "Rules: " + NOLEAK + " " + VOICE_ONLY + ACCENT_NOTE
            + " Emotion and character are only a general impression; do not name topic details.\n\n"
            "Example (format only, do not copy content):\n" + DSD_EXAMPLE + "\n\nYOUR DSD:")


def build_rp(rec: dict, example: str) -> str:
    return ("Task: produce an RP (role-play) voice direction — an evocative prompt that casts the speaker as an abstract "
            "ROLE within a MOOD or ATMOSPHERE, inferred from how the VOICE SOUNDS (pitch, pace, volume, energy, timbre) "
            "and NOT from what is being said. Lean on imagery rather than a list of acoustic parameters. VARY your "
            "structure: open on the scene, the character, a sensory image, or a direct instruction — do NOT fall into a "
            "fixed formula or a stock opening. A dynamic arc that shifts tone is welcome.\n\n"
            "Tags (a light guide — weave them into the imagery, don't just list them; anything you state must match):\n"
            + "\n".join(tag_lines_en(rec)) + "\n\n"
            "Rules: " + RP_GUARD + ACCENT_NOTE + "\n\n"
            "Example (just ONE of many possible shapes — invent a different role, imagery, and opening):\n"
            + example + "\n\nYOUR RP:")


# ------------------------------------------------------------------ dialogue
# A conversation is several voices, so a dialogue caption is the drawn form applied to
# each speaker in turn: one line per speaker, S1 then S2, each written from that
# speaker's own tags in that form's style. The form is drawn once per dialogue, as for
# any other record, so the corpus keeps its four styles on dialogues too. The transcript
# reaches only `free`, for the same reason transcript_block gives.
_DIALOGUE_TASK = {
    "aps": ("For EACH speaker produce an APS (acoustic-parameter spec) — a terse, semicolon-separated "
            "list of that voice's attributes only. No sentences, no scene, no identity, no speech content."),
    "dsd": ("For EACH speaker produce a DSD (descriptive-style directive) — flowing prose describing that "
            "voice's pitch, pace, volume, emotional undertone, and general character/temperament."),
    "rp":  ("For EACH speaker produce an RP (role-play) voice direction — an evocative prompt casting that "
            "speaker as an abstract ROLE within a MOOD or ATMOSPHERE, inferred from how THEIR voice sounds "
            "and not from what is said. Vary the structure across speakers; no stock opening."),
    "free": ("For EACH speaker write ONE caption that (a) describes that voice's characteristics and "
             "(b) infers a plausible identity and scene for that speaker, consistent with the voice."),
}


def _speaker_blocks(rec: dict) -> str:
    blocks = []
    for tag, spk in (rec.get("speakers") or {}).items():
        lines = tag_lines_en(spk) or ["- (no tags available)"]
        blocks.append(f"{tag}:\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def _speaker_example(rec: dict, one: "str | list[str]") -> str:
    # rp hands a list so S1 and S2 see two different castings; asking for varied
    # structure while showing the same line twice would pull the other way
    ones = [one] if isinstance(one, str) else list(one)
    tags = list((rec.get("speakers") or {}).keys()) or ["S1", "S2"]
    return ("\n".join(f"{t}: {ones[i % len(ones)]}" for i, t in enumerate(tags[:2]))
            + ("\n..." if len(tags) > 2 else ""))


def _rp_pair(rec: dict, first: "str | None") -> list:
    from . import pick_example
    rid = rec.get("id", "")
    first = first or pick_example(RP_EXAMPLES, rid)
    second = pick_example(RP_EXAMPLES, rid + "#2")
    if second == first:
        second = RP_EXAMPLES[(RP_EXAMPLES.index(first) + 1) % len(RP_EXAMPLES)]
    return [first, second]


def build_dialogue(rec: dict, form: str = "dsd", example: str | None = None) -> str:
    if form not in _DIALOGUE_TASK:
        form = "dsd"
    speakers = list((rec.get("speakers") or {}).keys())
    order = ", ".join(speakers) if speakers else "S1, S2, ..."
    head = ("Task: this recording is a conversation between several speakers. "
            + _DIALOGUE_TASK[form]
            + f" Write ONE line per speaker, in the form `S1: ...` then `S2: ...`, in this order: {order}. "
            "Each line stands on its own — do not compare the speakers and do not describe the conversation.\n\n"
            "Speakers and their tags (use each speaker's tags exactly, and only for that speaker):\n\n"
            + _speaker_blocks(rec) + "\n")
    marks = (" The transcript's [S1]/[S2] marks show only who is speaking; they are not to be described or "
             "quoted.")
    if form == "aps":
        rules = "Rules: " + NOLEAK + " " + VOICE_ONLY + ACCENT_NOTE + " Use only the given tags; do not guess timbre/personality/scene."
        ex = _speaker_example(rec, APS_EXAMPLE)
        tail = "YOUR APS (one line per speaker):"
    elif form == "rp":
        rules = "Rules: " + RP_GUARD + ACCENT_NOTE
        ex = _speaker_example(rec, _rp_pair(rec, example))
        tail = "YOUR RP (one line per speaker):"
    elif form == "free":
        rules = _free_rules(rec) + ACCENT_NOTE + marks
        ex = _speaker_example(rec, FREE_EXAMPLE)
        tail = "YOUR CAPTIONS (one per speaker):"
        head += transcript_block(rec)
    else:
        rules = ("Rules: " + NOLEAK + " " + VOICE_ONLY + ACCENT_NOTE
                 + " Emotion and character are only a general impression; do not name topic details.")
        ex = _speaker_example(rec, DSD_EXAMPLE)
        tail = "YOUR DSD (one line per speaker):"
    return head + "\n" + rules + "\n\nExample (format only, do not copy content):\n" + ex + "\n\n" + tail
