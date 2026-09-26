"""English-written prompt templates that request Simplified Chinese output.

The task and acoustic guards are shared with the English templates. Keeping
the source instructions in English avoids maintaining two divergent rule sets.
"""
from __future__ import annotations

from . import en

RP_EXAMPLES = en.RP_EXAMPLES

LANGUAGE_RULE = (
    "Write the final answer entirely in Simplified Chinese. The task instructions "
    "and acoustic tags below are in English; translate their meaning accurately. "
    "When a rule says to use a tag exactly, preserve its value and intensity in "
    "Chinese rather than copying the English word. Keep speaker identifiers such "
    "as S1 and S2 unchanged.\n\n"
)


def build_free(rec: dict) -> str:
    return LANGUAGE_RULE + en.build_free(rec)


def build_aps(rec: dict) -> str:
    return LANGUAGE_RULE + en.build_aps(rec)


def build_dsd(rec: dict) -> str:
    return LANGUAGE_RULE + en.build_dsd(rec)


def build_rp(rec: dict, example: str) -> str:
    return LANGUAGE_RULE + en.build_rp(rec, example)


def build_dialogue(rec: dict, form: str = "dsd", example: str | None = None) -> str:
    return LANGUAGE_RULE + en.build_dialogue(rec, form=form, example=example)
