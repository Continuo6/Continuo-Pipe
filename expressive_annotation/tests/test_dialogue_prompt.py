"""A dialogue record is several voices: the drawn form, applied to each in turn."""
from __future__ import annotations

import pytest

from continuo_expressive.prompts import build_prompt


def dialogue():
    return {"id": "d1", "n_speakers": 2,
            "txt": "[S1] hello there [S2] hi how are you",
            "speakers": {"S1": {"speaker": "14", "gender": "female", "age_band": "young",
                                "accent": "North America", "pitch": "high", "speed": "fast",
                                "volume": "normal"},
                         "S2": {"speaker": "15", "gender": "male", "age_band": "middle",
                                "pitch": "low", "speed": "measured", "volume": "loud"}}}


MARK = {"en": {"aps": "APS", "dsd": "DSD", "rp": "RP", "free": "identity"},
        "zh": {"aps": "APS", "dsd": "DSD", "rp": "RP", "free": "identity"}}


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("form", ["aps", "dsd", "rp", "free"])
def test_each_form_is_applied_per_speaker(lang, form):
    p = build_prompt(dialogue(), form=form, lang=lang, use_transcript=True)
    assert MARK[lang][form] in p, "the drawn form's style survives on a dialogue"
    assert p.index("S1") < p.index("S2")
    assert "female" in p and "male" in p
    # the example is shown per speaker, so the model sees the S1/S2 shape it must emit
    example = p.split("xample")[-1]
    assert "S1" in example and "S2" in example


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_only_free_sees_the_transcript(lang):
    for form in ("aps", "dsd", "rp"):
        assert "hello there" not in build_prompt(dialogue(), form=form, lang=lang, use_transcript=True)
    assert "hello there" in build_prompt(dialogue(), form="free", lang=lang, use_transcript=True)
    assert "hello there" not in build_prompt(dialogue(), form="free", lang=lang, use_transcript=False)


def test_a_single_voice_record_is_untouched():
    rec = {"id": "c1", "gender": "female", "pitch": "high", "txt": "x"}
    assert "S1" not in build_prompt(rec, form="dsd", lang="en")


def test_dialogue_is_not_a_form_name():
    with pytest.raises(ValueError):
        build_prompt(dialogue(), form="dialogue", lang="en")


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_rp_shows_a_different_casting_per_speaker(lang):
    p = build_prompt(dialogue(), form="rp", lang=lang)
    example = p.split("xample")[-1]
    s1 = next(ln for ln in example.splitlines() if ln.startswith("S1"))
    s2 = next(ln for ln in example.splitlines() if ln.startswith("S2"))
    assert s1[2:].lstrip(": ") != s2[2:].lstrip(": ")


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_a_long_transcript_is_cut_in_the_prompt(lang):
    from continuo_expressive.prompts.en import TRANSCRIPT_MAX_CHARS
    rec = dialogue()
    rec["txt"] = ("[S1] " + "word " * 20000) if lang == "en" else ("[S1] " + "\u5b57" * 60000)
    p = build_prompt(rec, form="free", lang=lang, use_transcript=True)
    # the free prompt's own scaffolding (tags, rules, example) is ~3.5k chars
    assert len(p) < TRANSCRIPT_MAX_CHARS + 5000 and len(p) < len(rec["txt"]) // 8
    assert "transcript cut here" in p
