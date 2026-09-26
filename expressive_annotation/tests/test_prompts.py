"""Prompt assembly: the per-clip draw must be stable, and null tags must vanish.

The draw is what makes the caption pass resumable — an interrupted run has to give
each clip the same form and language on restart, or half a corpus ends up captioned
twice under different templates.
"""
from __future__ import annotations

import pytest

from continuo_expressive.prompts import (FORMS, LANGS, build_prompt, generation_seed,
                                          pick_example, pick_form, pick_lang)
from continuo_expressive.prompts.tags import tag_lines_en, tag_lines_zh

REC = {"id": "clip_0001", "gender": "male", "age_band": "middle",
       "accent": "Standard Mandarin", "pitch": "low", "volume": "normal",
       "speed": "measured", "emotion": "sad"}


# ------------------------------------------------------------------ the draw
def test_form_and_lang_are_stable_for_an_id():
    assert pick_form("clip_x") == pick_form("clip_x")
    assert pick_lang("clip_x") == pick_lang("clip_x")


def test_form_and_lang_are_independent_axes():
    # different key prefixes, so the two draws must not correlate by construction
    ids = [f"clip_{i}" for i in range(400)]
    pairs = {(pick_form(i), pick_lang(i)) for i in ids}
    assert len(pairs) == len(FORMS) * len(LANGS)


def test_the_draw_covers_every_form_and_language():
    ids = [f"clip_{i}" for i in range(200)]
    assert {pick_form(i) for i in ids} == set(FORMS)
    assert {pick_lang(i) for i in ids} == set(LANGS)


def test_seed_reshuffles_the_assignment():
    ids = [f"clip_{i}" for i in range(200)]
    assert [pick_form(i, 0) for i in ids] != [pick_form(i, 1) for i in ids]


def test_generation_seed_is_stable_and_in_torch_range():
    assert generation_seed("clip_x") == generation_seed("clip_x")
    assert 0 <= generation_seed("clip_x") < 2 ** 31


def test_example_rotation_is_stable_and_varies():
    pool = [f"example {i}" for i in range(6)]
    assert pick_example(pool, "a") == pick_example(pool, "a")
    assert len({pick_example(pool, f"clip_{i}") for i in range(200)}) == len(pool)


# ------------------------------------------------------------------ tag lines
def test_null_tags_produce_no_line():
    sparse = {"id": "x", "gender": "male", "volume": None, "emotion": None}
    lines = tag_lines_en(sparse)
    assert len(lines) == 1
    assert not any("None" in line or "unknown" in line.lower() for line in lines)


def test_gated_null_emotion_is_absent_from_both_languages():
    rec = dict(REC, emotion=None)
    assert not any("Emotion" in line for line in tag_lines_en(rec))
    assert not any("Emotion" in line for line in tag_lines_zh(rec))


def test_normal_volume_is_stated_as_moderate_not_normal():
    # the captioner is told the value verbatim; "normal" is not a word it should use
    line = [l for l in tag_lines_en(REC) if l.startswith("- Volume")][0]
    assert line == "- Volume: moderate"


def test_standard_mandarin_is_not_called_an_accent():
    line = [l for l in tag_lines_en(REC) if l.startswith("- Accent")][0]
    assert line == "- Accent: standard Mandarin"


def test_unmapped_accent_gets_the_generic_phrasing():
    line = [l for l in tag_lines_en(dict(REC, accent="Scottish")) if l.startswith("- Accent")][0]
    assert line == "- Accent: Scottish accent"


def test_chinese_output_prompt_receives_english_tag_data():
    lines = "\n".join(tag_lines_zh(REC))
    assert "Gender: male" in lines
    assert "Volume: moderate" in lines
    assert "Accent: standard Mandarin" in lines


# ------------------------------------------------------------------ full prompts
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("lang", LANGS)
def test_every_form_and_language_builds_a_prompt_carrying_the_tags(form, lang):
    prompt = build_prompt(REC, form=form, lang=lang)
    assert prompt.strip()
    assert "standard Mandarin" in prompt
    if lang == "zh":
        assert "Write the final answer entirely in Simplified Chinese" in prompt


@pytest.mark.parametrize("form", ["aps", "dsd", "free"])
def test_strict_forms_forbid_retelling_the_speech(form):
    en = build_prompt(REC, form=form, lang="en")
    zh = build_prompt(REC, form=form, lang="zh")
    assert "NEVER" in en or "Never" in en
    assert "NEVER" in zh or "Never" in zh


def test_rp_is_the_loose_form_but_still_bans_content_leakage():
    prompt = build_prompt(REC, form="rp", lang="en")
    assert "imagery" in prompt.lower()
    assert "never quote" in prompt.lower()


def test_rp_example_follows_the_clip_id():
    a = build_prompt(dict(REC, id="clip_a"), form="rp", lang="en")
    b = build_prompt(dict(REC, id="clip_b"), form="rp", lang="en")
    assert a == build_prompt(dict(REC, id="clip_a"), form="rp", lang="en")
    assert a != b            # different clips get different rotated examples


def test_free_example_is_opt_in():
    without = build_prompt(REC, form="free", lang="en")
    with_it = build_prompt(REC, form="free", lang="en", include_example=True)
    assert "Good Example" not in without
    assert "Good Example" in with_it


@pytest.mark.parametrize("bad", [("nope", "en"), ("free", "de")])
def test_unknown_form_or_language_is_rejected(bad):
    with pytest.raises(ValueError):
        build_prompt(REC, form=bad[0], lang=bad[1])


# ------------------------------------------------ captioning from text, without audio
def _rec(**over):
    rec = {"id": "x", "gender": "male", "age_band": "middle", "pitch": "medium",
           "volume": "normal", "speed": "measured",
           "txt": "Guten Morgen und danke für das Gespräch."}
    rec.update(over)
    return rec


def test_the_transcript_reaches_only_the_form_that_infers_a_scene():
    # free is the one form asked to infer an identity and a scene, so the words tell it
    # something the tags cannot. aps and dsd are acoustic descriptions with nothing to
    # gain, and rp must build its imagery from sound alone — given the transcript it
    # quoted the topics straight out of it, quotation marks and all.
    marker = "Guten Morgen und danke"
    for lang in ("en", "zh"):
        assert marker in build_prompt(_rec(), form="free", lang=lang, use_transcript=True)
        for form in ("aps", "dsd", "rp"):
            assert marker not in build_prompt(_rec(), form=form, lang=lang,
                                              use_transcript=True), (form, lang)


def test_the_transcript_needs_the_flag():
    assert "Guten Morgen" not in build_prompt(_rec(), form="free", lang="en")


def test_text_mode_drops_the_clause_it_cannot_follow():
    # "as if you did not understand the language" is unfollowable once the words are on
    # the page, and an unfollowable rule gets dropped whole rather than in part
    listening = build_prompt(_rec(txt=None), form="free", lang="en")
    reading = build_prompt(_rec(), form="free", lang="en", use_transcript=True)
    assert "ONLY loosely inform the GENERAL mood or register" in listening
    assert "ONLY loosely inform the GENERAL mood or register" not in reading
    assert "never be reproduced, quoted, or referred to specifically" in reading


def test_the_other_forms_keep_their_original_guard_wording():
    # they never see a transcript, so a guard that talks about one would be nonsense
    for form in ("aps", "dsd"):
        prompt = build_prompt(_rec(), form=form, lang="en", use_transcript=True)
        assert "as if you did not understand the language" in prompt, form
    rp = build_prompt(_rec(), form="rp", lang="en", use_transcript=True)
    assert "language you do not understand" in rp


def test_a_record_without_a_transcript_is_untouched_by_the_flag():
    for form in ("free", "aps", "dsd", "rp"):
        for lang in ("en", "zh"):
            plain = build_prompt(_rec(txt=None), form=form, lang=lang)
            flagged = build_prompt(_rec(txt=None), form=form, lang=lang, use_transcript=True)
            assert plain == flagged, (form, lang)


# ------------------------------------------------- the transformers text-only path
class _Recorder:
    """Stands in for a processor: records how it was called, returns a usable stub."""

    def __init__(self):
        self.chat_calls, self.processor_calls = [], []
        self.tokenizer = type("T", (), {"eos_token_id": 0})()

    def apply_chat_template(self, conversation, **kw):
        self.chat_calls.append(conversation)
        return "TEMPLATED"

    def __call__(self, **kw):
        self.processor_calls.append(kw)
        return {"input_ids": _FakeTensor()}

    def batch_decode(self, ids, **kw):
        return ["a caption"]


class _FakeTensor:
    shape = (1, 3)
    dtype = None

    def to(self, *a, **kw):
        return self

    def __getitem__(self, item):
        return self

    def keys(self):
        return []


def test_a_none_waveform_puts_no_audio_in_the_conversation():
    # captioning a long recording runs on text; the request must carry no audio at all
    # rather than an empty one, or the template inserts an audio token with nothing
    # behind it
    from continuo_expressive.captioners import VllmCaptioner

    cap = VllmCaptioner.__new__(VllmCaptioner)
    cap.processor = _Recorder()
    with_audio = cap._templated("P", with_audio=True)
    text_only = cap._templated("P", with_audio=False)
    assert with_audio == text_only == "TEMPLATED"
    kinds = [[c["type"] for c in call[0]["content"]] for call in cap.processor.chat_calls]
    assert kinds == [["audio", "text"], ["text"]]
