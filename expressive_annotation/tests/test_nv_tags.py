"""NVV tag extraction — the one piece of ``continuo-nv`` that runs without the model."""
import pytest

from continuo_expressive.cli.nv import split_tags

# Unicode escapes retain CJK parsing coverage without non-English source text.

@pytest.mark.parametrize("text, clean, tags", [
    ("[Uhm]\u5c31\u662f\u5927\u5bb6", "\u5c31\u662f\u5927\u5bb6", ["Uhm"]),
    ("\u4eca\u5929\u5f88\u5f00\u5fc3[Laughter]", "\u4eca\u5929\u5f88\u5f00\u5fc3", ["Laughter"]),
    ("\u666e\u901a\u4e00\u53e5\u8bdd", "\u666e\u901a\u4e00\u53e5\u8bdd", []),
    ("", "", []),
    # order is preserved: a laugh before a sentence and one after it are not the same
    ("[Laughter]\u597d[Cough]", "\u597d", ["Laughter", "Cough"]),
    # hyphenated categories are one tag, not two
    ("\u554a[Surprise-oh]", "\u554a", ["Surprise-oh"]),
    # a category outside the documented 13/7 still comes through, rather than being
    # dropped by a hardcoded vocabulary — the checkpoint emits [Surprise-yo] in practice
    ("\u54df[Surprise-yo]", "\u54df", ["Surprise-yo"]),
    # brackets that are not tags are left alone
    ("cost [3] dollars", "cost [3] dollars", []),
])
def test_split_tags(text, clean, tags):
    assert split_tags(text) == (clean, tags)


def test_english_spacing_is_collapsed():
    """Removing an inline tag must not leave a double space behind."""
    clean, tags = split_tags("this is [Breathing] a question")
    assert clean == "this is a question"
    assert tags == ["Breathing"]


def test_asr_ratio_is_none_without_a_reference():
    from continuo_expressive.cli.nv import asr_ratio
    assert asr_ratio("anything", None) is None
    assert asr_ratio("anything", "") is None
    assert asr_ratio("anything", "\uff0c\u3002\uff01") is None   # punctuation only leaves no reference


def test_asr_ratio_ignores_punctuation_on_both_sides():
    """The corpus punctuates, this model does not; that must not read as a collapse."""
    from continuo_expressive.cli.nv import asr_ratio
    assert asr_ratio("\u5728\u6211\u5fc3\u91cc", "\u5728\u6211\u5fc3\u91cc\u3002") == 1.0
    assert asr_ratio("And we're back", "And we're back.") == 1.0


def test_asr_ratio_spots_a_collapse():
    """The Crying failure: one event emitted instead of the words."""
    from continuo_expressive.cli.nv import asr_ratio
    assert asr_ratio("", "Oh oh oh oh oh oh oh oh.") == 0.0
    assert asr_ratio("\u4eca", "Come on, tonight, tonight, tonight.") < 0.1
