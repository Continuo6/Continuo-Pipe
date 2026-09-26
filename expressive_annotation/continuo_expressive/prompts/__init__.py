"""Instruction prompt assembly: four forms x two languages, chosen per clip.

Forms follow the InstructTTSEval taxonomy:

``free``  free-form voice description plus a plausible identity and scene
``aps``   Acoustic-Parameter Spec — a terse, semicolon-separated attribute list
``dsd``   Descriptive-Style Directive — prose over pitch/pace/volume/affect
``rp``    Role-Play — an evocative voice direction built from how the voice sounds

Form and language are independent axes, both drawn per clip so a corpus ends up with
a mix rather than 200k copies of one template. Every draw is a hash of the clip id,
which makes the whole assignment reproducible and, more usefully, **resumable**: a run
that dies halfway and restarts gives clip X the same form, the same language, the same
rotated example and the same sampling seed it would have had.

MD5 is the hash purely because it is fast, stable across Python versions, and already
what existing annotations were built with — changing it would silently reshuffle the assignment
of every clip in existing corpora. It is a dispatcher, not a security primitive,
hence ``usedforsecurity=False``.
"""
from __future__ import annotations

import hashlib

from . import en, zh

FORMS = ("free", "aps", "dsd", "rp")
LANGS = ("en", "zh")


def _digest(key: str) -> int:
    return int(hashlib.md5(key.encode("utf-8"), usedforsecurity=False).hexdigest(), 16)


def pick_form(clip_id: str, seed: int = 0) -> str:
    """Deterministic per-clip form."""
    return FORMS[_digest(f"{seed}:{clip_id}") % len(FORMS)]


def pick_lang(clip_id: str, seed: int = 0) -> str:
    """Deterministic per-clip caption language, independent of the audio's language."""
    return LANGS[_digest(f"lang:{seed}:{clip_id}") % len(LANGS)]


def pick_example(pool: list[str], clip_id: str) -> str:
    """Rotate one style example per clip so outputs do not all clone one opening."""
    return pool[_digest(f"rpex:{clip_id}") % len(pool)]


def generation_seed(clip_id: str) -> int:
    """Per-clip torch seed: sampled generation stays reproducible across reruns."""
    return _digest(f"gen:{clip_id}") % (2 ** 31)


def build_prompt(rec: dict, form: str = "free", lang: str = "en",
                 include_example: bool = False,
                 describe_variation: bool = False,
                 use_transcript: bool = False) -> str:
    """Assemble the prompt for one clip's tags.

    ``include_example`` only affects ``free`` in English; the other forms always
    carry their own format example.

    ``describe_variation`` lets a record aggregated from a long recording contribute a
    line about how volume, rate and emotion move across the file. Off by default
    because it changes what captions talk about, and that is a corpus-wide decision to
    make deliberately rather than inherit. A record with no ``*_spans`` — every single
    clip — is unaffected either way.

    ``use_transcript`` puts the recording's words in the prompt, for captioning without
    audio. It also swaps the content guards for their text-mode wording: "as if you did
    not understand the language" is unfollowable once the words are on the page, and an
    unfollowable rule gets dropped whole rather than in part. The voice description
    still comes from the tags alone; the transcript may only place the register and
    setting. Off by default — both flags are corpus-wide decisions.
    """
    if lang not in LANGS:
        raise ValueError(f"unknown caption language {lang!r}; expected one of {LANGS}")
    if form not in FORMS:
        raise ValueError(f"unknown caption form {form!r}; expected one of {FORMS}")

    if not describe_variation:
        rec = {k: v for k, v in rec.items() if not k.endswith("_spans")}
        if rec.get("speakers"):
            # the spans live on each speaker of a dialogue record, not at the top
            rec = {**rec, "speakers": {tag: {k: v for k, v in spk.items() if not k.endswith("_spans")}
                                       for tag, spk in rec["speakers"].items()}}
    if not use_transcript:
        rec = {k: v for k, v in rec.items() if k != "txt"}

    module = zh if lang == "zh" else en
    if rec.get("speakers"):
        # a dialogue record: several voices, each with its own tags. The drawn form is
        # applied to each speaker in turn — one line per speaker, S1 then S2 — so a
        # dialogue is captioned in the same four styles as everything else.
        ex = pick_example(module.RP_EXAMPLES, rec.get("id", "")) if form == "rp" else None
        return module.build_dialogue(rec, form=form, example=ex)
    if form == "aps":
        return module.build_aps(rec)
    if form == "dsd":
        return module.build_dsd(rec)
    if form == "rp":
        return module.build_rp(rec, pick_example(module.RP_EXAMPLES, rec.get("id", "")))
    if lang == "zh":
        return module.build_free(rec)
    return module.build_free(rec, include_example=include_example)
