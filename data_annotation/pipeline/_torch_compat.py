"""Cross-cutting torch shims for adapters that load legacy checkpoints.

PyTorch 2.6 flipped ``torch.load``'s default to ``weights_only=True``, which
rejects pyannote 3.x checkpoints (they pickle ``torch.torch_version.TorchVersion``
and friends that aren't on the safe-globals allowlist). lightning_fabric also
passes ``weights_only=True`` explicitly when loading from a path, so a mere
``setdefault`` isn't enough — we must *force* ``weights_only=False``.

Use :func:`legacy_torch_load_context` only around the audited legacy checkpoint
load; the process-wide default is restored immediately afterwards.
"""

from __future__ import annotations

from contextlib import contextmanager
import threading

_PATCH_LOCK = threading.RLock()

@contextmanager
def legacy_torch_load_context():
    """Temporarily force legacy checkpoint loading, then restore torch.load."""
    import torch
    with _PATCH_LOCK:
        original = torch.load

        def patched(*args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs["weights_only"] = False
            return original(*args, **kwargs)

        torch.load = patched  # type: ignore[assignment]
        try:
            yield
        finally:
            torch.load = original  # type: ignore[assignment]
