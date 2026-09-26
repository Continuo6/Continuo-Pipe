"""Chinese-dialect cascade: FireRedLID first, Voxlect-Mandarin to refine.

FireRedLID covers the broad Chinese varieties (mandarin / xinan / wu / min / xiang /
yue) and is Apache-2.0, but its ``north`` bucket collapses the northern Mandarin
subgroups (Zhongyuan, Ji-Lu, Jiao-Liao, Lan-Yin) and ``other`` swallows Jiang-Huai.
Voxlect-Mandarin has the finer 7-subgroup head but is weaker overall.

Take FireRedLID where it is confident and hand the coarse buckets to Voxlect.

The two models cannot share a process (different pinned dependency sets), so
FireRedLID runs as its own pass and this function fuses the two outputs offline.
"""
from __future__ import annotations

# FireRedLID token -> Voxlect Mandarin-subgroup label. FireRedLID is confident here.
FIRE_TO_SUBGROUP = {
    "mandarin": "Standard Mandarin",
    "xinan": "Southwestern",
    "yue": "Cantonese",
}
# Major varieties FireRedLID covers and Voxlect-Mandarin does not — keep its call.
FIRE_ONLY = ("wu", "min", "xiang")
# Coarse / ambiguous buckets — defer to Voxlect's finer head.
REFINE_BUCKETS = ("north", "other")


def cascade_dialect(firered_label: str | None, voxlect_label: str | None) -> str | None:
    """Fuse one FireRedLID output with one Voxlect-Mandarin output.

    firered_label : raw FireRedLID string, e.g. ``"zh mandarin"``, ``"zh north"``.
    voxlect_label : raw Voxlect label, e.g. ``"Zhongyuan"``, ``"Standard Mandarin"``.
    """
    f = (firered_label or "").lower()
    if any(bucket in f for bucket in REFINE_BUCKETS):
        return voxlect_label
    for bucket, subgroup in FIRE_TO_SUBGROUP.items():
        if bucket in f:
            return subgroup
    for major in FIRE_ONLY:
        if major in f:
            return major
    return voxlect_label            # unknown token / LID misfire -> trust Voxlect
