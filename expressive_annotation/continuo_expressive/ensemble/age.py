"""Age cascade: a classifier rescues children, a median handles adults.

Both age *regressions* over-estimate children badly — voxprofile by +13.66 years,
audeering by +3.74, which turns a 4-year-old into an 18-year-old — so no binning of
their output recovers kids (voxprofile child recall 0.11-0.43). The audeering gender
head's third class, ``child``, does: AUC 0.92-0.96, and it fires at ~0.95 recall.

So: if that classifier says ``child``, the answer is ``child``. Otherwise the two
regressions' median is binned, which is the best adult path measured (CREMA-D acc
0.71 / MAE 9.22). One path covers the whole range, which neither a single head nor a
plain median does (plain median gets child 0.63).

The 43/60 edges are a compromise leaning Chinese: English regressions calibrate well
and want lower edges, Chinese systematically over-estimates and wants higher ones —
one set of edges cannot serve both.

Both ends stay weak: teen recall 0.17-0.41, senior ~0.4. The trustworthy region is
the young/middle adult span. For labels you can lean on, collapse to
``young(<43) / middle(43-60) / old(>=60)`` or accept +-1 band (~0.95-0.98).
"""
from __future__ import annotations

from statistics import median

AGE_EDGES = (13, 18, 43, 60)
BANDS = ("child", "teen", "young", "middle", "senior")


def age_band(years: float | None, edges: tuple = AGE_EDGES) -> str | None:
    """Continuous years -> one of the five bands. None-safe."""
    if years is None:
        return None
    c1, c2, c3, c4 = edges
    return ("child" if years < c1 else "teen" if years < c2 else
            "young" if years < c3 else "middle" if years < c4 else "senior")


def cascade_age(aud_gender: str | None, aud_age: float | None = None,
                vox_age: float | None = None, edges: tuple = AGE_EDGES,
                adult: str = "median") -> str | None:
    """Five-band age.

    aud_gender : audeering gender label — ``female`` / ``male`` / ``child``.
    aud_age    : audeering continuous age in years, or None.
    vox_age    : voxprofile continuous age in years, or None.
    adult      : adult-path source — ``median`` (best), ``vox``, or ``aud``.

    Returns None only when there is no age signal at all.
    """
    if (aud_gender or "").lower() == "child":
        return "child"
    ages = [a for a in (aud_age, vox_age) if a is not None]
    if not ages:
        return None
    if adult == "vox" and vox_age is not None:
        years = vox_age
    elif adult == "aud" and aud_age is not None:
        years = aud_age
    else:
        years = median(ages)
    return age_band(years, edges)
