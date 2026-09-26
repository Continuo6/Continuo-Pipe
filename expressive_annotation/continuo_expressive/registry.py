"""The heads this pipeline can run: checkpoint id, label ordering, licence.

Kept as data rather than scattered constants so a run log can record exactly which
weights produced a field, and so swapping a checkpoint is a one-line edit.

Every neural head here is **non-commercial / research-only** (see ``licence``).
Nothing in this repo checks that for you — it is recorded so the obligation travels
with the code. See NOTICE.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HeadSpec:
    name: str
    module: str                       # continuo_expressive.heads.<module>
    cls: str                          # class inside that module
    model_id: str                     # HF id or local snapshot path
    attribute: str
    backbone: str = ""
    licence: str = ""
    repo: str = ""                    # "voxprofile" | "voxlect" | "" (none)
    extra: dict = field(default_factory=dict)


# Label orderings come from the model cards; the logits carry no names, so a wrong
# ordering silently mislabels everything.
VOXLECT_LABELS = {
    "english": ["East Asia", "English", "Germanic", "Irish", "North America",
                "Northern Irish", "Oceania", "Other", "Romance", "Scottish",
                "Semitic", "Slavic", "South African", "Southeast Asia",
                "South Asia", "Welsh"],
    "mandarin-cantonese": ["Jiang-Huai", "Jiao-Liao", "Ji-Lu", "Lan-Yin",
                           "Standard Mandarin", "Southwestern", "Zhongyuan", "Cantonese"],
    "arabic": ["Egyptian", "Levantine", "Maghrebi", "MSA", "Peninsular"],
}

#: Which of a head's labels a declared language may actually be given.
#:
#: Routing picks the head, but a head's label set is not always confined to one
#: language. ``mandarin-cantonese`` is the case that bites: Cantonese is ``yue``, a
#: different language from ``zh``, and a corpus that has already declared a clip
#: Chinese must not have it come back Cantonese — nor a Cantonese clip come back
#: Jiang-Huai. The declared language is the stronger evidence, so it constrains the
#: head rather than the other way round.
#:
#: Probabilities are renormalised over the admissible labels rather than the winner
#: being filtered out afterwards, so ``accent_top3`` stays a distribution and
#: ``accent_agreement`` stays comparable across clips.
ACCENT_LABELS_FOR_LANG: dict[str, tuple[str, ...]] = {
    "zh": tuple(l for l in VOXLECT_LABELS["mandarin-cantonese"] if l != "Cantonese"),
    "yue": ("Cantonese",),
    "en": tuple(VOXLECT_LABELS["english"]),
    "ar": tuple(VOXLECT_LABELS["arabic"]),
}


def restrict_accent(probs: dict[str, float], lang: str | None) -> dict[str, float]:
    """Renormalise an accent distribution onto the labels ``lang`` admits.

    An unlisted language, or a distribution with no admissible mass at all, is
    returned unchanged: inventing a constraint is worse than declining to apply one.
    """
    allowed = ACCENT_LABELS_FOR_LANG.get(lang or "")
    if not allowed or not probs:
        return probs
    kept = {label: p for label, p in probs.items() if label in allowed}
    total = sum(kept.values())
    if not kept or total <= 0:
        return probs
    return {label: p / total for label, p in kept.items()}

REGISTRY: dict[str, HeadSpec] = {
    "audeering_age_gender": HeadSpec(
        name="audeering_age_gender", module="audeering", cls="AgeGenderHead",
        model_id="audeering/wav2vec2-large-robust-24-ft-age-gender",
        attribute="gender+age", backbone="wav2vec2-large-robust",
        licence="CC-BY-NC-SA-4.0 (non-commercial)",
    ),
    "voxprofile_wavlm_age": HeadSpec(
        name="voxprofile_wavlm_age", module="voxprofile", cls="AgeSexHead",
        model_id="tiantiaf/wavlm-large-age-sex",
        attribute="age", backbone="wavlm-large",
        licence="OpenRAIL (non-commercial)", repo="voxprofile",
        extra={"backbone_cls": "wavlm", "apply_reg": True, "output_class_num": 2},
    ),
    "voxlect_english": HeadSpec(
        name="voxlect_english", module="voxlect", cls="DialectHead",
        model_id="tiantiaf/voxlect-english-dialect-whisper-large-v3",
        attribute="accent", backbone="whisper-large-v3",
        licence="OpenRAIL / CC-BY-NC (non-commercial)", repo="voxlect",
        extra={"backbone_cls": "whisper", "label_list": VOXLECT_LABELS["english"]},
    ),
    "voxlect_mandarin": HeadSpec(
        name="voxlect_mandarin", module="voxlect", cls="DialectHead",
        model_id="tiantiaf/voxlect-mandarin-cantonese-dialect-whisper-large-v3",
        attribute="accent", backbone="whisper-large-v3",
        licence="OpenRAIL / CC-BY-NC (non-commercial)", repo="voxlect",
        extra={"backbone_cls": "whisper",
               "label_list": VOXLECT_LABELS["mandarin-cantonese"]},
    ),
    # Weakest of the three, and knowingly so: ~0.35 UAR over 5 classes on ADI17
    # region-level, against a 0.20 chance floor. That is the same band as the English
    # head on hard conversational speech (EdAcc ~0.28), so it is wired up for coverage
    # — but an Arabic `accent` is a weak signal, not a finding.
    "voxlect_arabic": HeadSpec(
        name="voxlect_arabic", module="voxlect", cls="DialectHead",
        model_id="tiantiaf/voxlect-arabic-dialect-whisper-large-v3",
        attribute="accent", backbone="whisper-large-v3",
        licence="OpenRAIL / CC-BY-NC (non-commercial)", repo="voxlect",
        extra={"backbone_cls": "whisper", "label_list": VOXLECT_LABELS["arabic"]},
    ),
}


def get(name: str) -> HeadSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown head '{name}'; known: {sorted(REGISTRY)}")
    return REGISTRY[name]
