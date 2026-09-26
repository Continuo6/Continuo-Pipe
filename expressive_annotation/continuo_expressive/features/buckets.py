"""Where every continuous measurement becomes a three-way label.

All edges live here rather than next to the code that measures, because the edges
are the part that gets re-derived from data and argued about; the measurement is
just physics. Each set records what fitted it and what that makes it worth.
"""
from __future__ import annotations

import math

# --- volume: absolute LUFS thresholds --------------------------------------
# Fixed thresholds keep labels comparable across runs. Recalibrate for a new
# audio domain if its loudness distribution differs substantially.
VOLUME_EDGES = (-27.0, -19.0)
VOLUME_LABELS = ("soft", "normal", "loud")

# --- pitch: fixed F0 thresholds, split by gender ----------------------------
# Measure F0 with PENN so the estimator matches the scale of these boundaries.
PITCH_EDGES = {"male": (115.7, 149.7), "female": (141.6, 184.5)}
PITCH_LABELS = ("low", "medium", "high")

# --- speed: transcript characters per total second, per language ------------
# Pauses remain in the denominator. English thresholds are fitted to labeled
# speech-rate data; other languages use approximate scale transfers. Treat
# non-English labels as within-language rankings and refit with
# tools/fit_speed_edges.py when segmentation or population changes. Languages
# without an entry keep a null speed label.
SPEED_CPS_EDGES = {
    "en": (12.5, 19.8),
    "zh": (3.4, 5.4),
    "de": (14.2, 22.4),
    "ar": (10.4, 16.4),
    "ru": (8.0, 12.7),
    "pt": (10.2, 16.2),
    "ja": (5.9, 9.3),
}
SPEED_LABELS = ("slow", "measured", "fast")

#: en gold edges expressed as multiples of the en median -- the shape every non-en
#: entry above is transferred with. Used by tools/fit_speed_edges.py.
SPEED_SHAPE = (12.5 / 14.83, 19.8 / 14.83)      # (0.843, 1.335)
#: below this many clips a corpus median is too noisy to anchor edges on
SPEED_FIT_MIN_CLIPS = 200


def bucket(value: float | None, edges: tuple[float, float],
           labels: tuple[str, str, str]) -> str | None:
    """``< lo`` -> first label, ``> hi`` -> third, otherwise the middle one."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    lo, hi = edges
    return labels[0] if value < lo else labels[2] if value > hi else labels[1]


def pitch_edges_for(gender: str | None) -> tuple[float, float]:
    return PITCH_EDGES.get(gender or "", PITCH_EDGES["female"])
