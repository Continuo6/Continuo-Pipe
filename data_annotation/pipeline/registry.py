"""Registry: maps config ``name`` strings to adapter classes.

Adding a new model implementation is exactly:

1. Implement an adapter under ``pipeline/adapters/`` that satisfies one of
   the ABCs in ``pipeline/stages/base.py``.
2. Define its params class in ``pipeline/config.py`` and add it to
   :data:`pipeline.config.PARAM_MODELS`.
3. Add a single line to the relevant registry dict below.

No other code changes.

Adapters are referenced lazily as ``(module_path, class_name)`` tuples
and only imported when ``build_adapter`` actually resolves them. That
keeps Phase 1 and Phase 2 environments independently slim — a Phase 2
env can omit ``audio-separator`` / DiariZen / FireRedASR2S / etc. and
still ``import pipeline.registry`` without crashing on a missing dep.
"""

from __future__ import annotations

import importlib
from typing import Any

from pipeline.config import PARAM_MODELS, StageRef
from pipeline.errors import StageConfigError


# Each entry is (module_path, class_name) — imported on demand by
# ``build_adapter``. Adding a new adapter is still one line per slot.
SEPARATOR_REGISTRY: dict[str, tuple[str, str]] = {
    "mel_band_roformer_kim_ft3": (
        "pipeline.adapters.mel_band_roformer_separator",
        "MelBandRoformerKimFT3Separator",
    ),
}
DIARIZER_REGISTRY: dict[str, tuple[str, str]] = {
    "diarizen": ("pipeline.adapters.diarizen_diarizer", "DiariZenDiarizer"),
}
VAD_REGISTRY: dict[str, tuple[str, str]] = {
    "firered_vad": ("pipeline.adapters.firered_vad", "FireRedVADAdapter"),
}
LID_REGISTRY: dict[str, tuple[str, str]] = {
    "firered_lid": ("pipeline.adapters.firered_lid", "FireRedLIDAdapter"),
}
ASR_REGISTRY: dict[str, tuple[str, str]] = {
    "qwen3_asr": ("pipeline.adapters.qwen3_asr", "Qwen3ASR"),
    "indic_conformer": (
        "pipeline.adapters.indic_conformer_asr", "IndicConformerASR",
    ),
}
SCORER_REGISTRY: dict[str, tuple[str, str]] = {
    "dnsmos": ("pipeline.adapters.dnsmos_adapter", "DNSMOSScorer"),
}
_REGISTRIES: dict[str, dict[str, tuple[str, str]]] = {
    "separator": SEPARATOR_REGISTRY,
    "diarizer": DIARIZER_REGISTRY,
    "vad": VAD_REGISTRY,
    "lid": LID_REGISTRY,
    "asr": ASR_REGISTRY,
    "scorer": SCORER_REGISTRY,
}


def build_adapter(slot: str, ref: StageRef, device: str) -> Any:
    """Resolve ``ref.name`` against the slot's registry and instantiate.

    The raw ``ref.params`` dict is coerced through the right pydantic model
    in :data:`pipeline.config.PARAM_MODELS`, so a missing or mistyped field
    fails fast at construction time. The adapter module is imported here
    on first use, so missing optional deps (vLLM in Phase 1 env,
    audio-separator in Phase 2 env) only fail when actually requested.
    """
    registry = _REGISTRIES.get(slot)
    if registry is None:
        raise StageConfigError(f"unknown stage slot: {slot}")
    entry = registry.get(ref.name)
    if entry is None:
        raise StageConfigError(
            f"unknown {slot} adapter {ref.name!r}; "
            f"available: {sorted(registry.keys())}"
        )
    module_path, class_name = entry
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise StageConfigError(
            f"adapter {ref.name!r} ({slot}) requires {module_path}, "
            f"which is not importable in this environment: {e}. "
            f"Check that the right Phase 1 / Phase 2 env is active."
        ) from e
    factory = getattr(module, class_name)

    param_cls = PARAM_MODELS.get(ref.name)
    if param_cls is None:
        raise StageConfigError(
            f"no param model registered for adapter {ref.name!r}; "
            f"add it to PARAM_MODELS in pipeline/config.py"
        )
    params = param_cls.model_validate(ref.params)
    return factory(params=params, device=device)
