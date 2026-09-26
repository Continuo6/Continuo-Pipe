"""Optional confidence gate for externally supplied emotion predictions.

Predictions come from the caller's emotion model. There is no default threshold, so
callers must choose one explicitly when they provide emotion predictions.
"""
from __future__ import annotations

#: Non-emotion labels that should not be emitted.
NON_EMOTION = {"unknown"}

FIELDS = ("emotion", "emotion_confidence", "emotion_top3")


def _top(scores: dict[str, float], k: int = 3) -> dict[str, float]:
    return {label: round(p, 3)
            for label, p in sorted(scores.items(), key=lambda kv: -kv[1])[:k]}


def gate_emotion(raw: dict | None, tau: float | None = None) -> dict:
    """External prediction record -> the three ``emotion*`` fields.

    ``raw`` contains ``label``, ``confidence`` and optional ``scores``,
    or is None when that clip has no external prediction.
    """
    if not raw:
        return dict.fromkeys(FIELDS)
    if tau is None or not 0.0 <= tau <= 1.0:
        raise ValueError("an explicit emotion threshold in [0, 1] is required")
    confidence = float(raw.get("confidence") or 0.0)
    label = raw.get("label")
    scores = raw.get("scores") or {}
    keep = confidence >= tau and label not in NON_EMOTION
    return {"emotion": label if keep else None,
            "emotion_confidence": round(confidence, 4),
            "emotion_top3": _top(scores) if scores else None}
