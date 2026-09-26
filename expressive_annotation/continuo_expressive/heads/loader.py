"""Instantiate a registry head, putting its vendored repo on ``sys.path`` first.

Vox-Profile and Voxlect are consumed as source checkouts, not packages, and both
expose their code under a top-level ``src``. That only coexists because neither ships
``src/__init__.py``: ``src`` and ``src.model`` are *namespace* packages, so their
``__path__`` is recomputed from ``sys.path`` and spans both checkouts at once. If
either upstream ever adds an ``__init__.py``, the first repo imported would shadow the
other and the second head would fail to import — :func:`add_repo` fails loudly on a
missing checkout so that shows up as a clear error rather than a wrong label.

Paths are resolved without ``os.chdir``-ing to the repo root at import time: a
library that changes the process's working directory breaks every relative path
its caller holds.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from .. import config
from ..registry import HeadSpec, get
from .base import Head

_REPO_RESOLVERS = {
    "voxprofile": (config.voxprofile_repo, "CONTINUO_EXPRESSIVE_VOXPROFILE_REPO",
                   "https://github.com/tiantiaf0627/vox-profile-release"),
    "voxlect": (config.voxlect_repo, "CONTINUO_EXPRESSIVE_VOXLECT_REPO",
                "https://github.com/tiantiaf0627/voxlect"),
}


def add_repo(kind: str) -> Path:
    """Put a vendored repo root on ``sys.path`` (idempotent) and return it."""
    resolve, env_var, url = _REPO_RESOLVERS[kind]
    root = resolve().resolve()
    if not (root / "src").is_dir():
        raise FileNotFoundError(
            f"{kind} checkout not found at {root} (no src/). "
            f"Run scripts/setup_models.sh, or set {env_var} to your clone of {url}.")
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return root


def build(name: str, device: str = "cuda", **overrides) -> Head:
    """Construct (but do not load) the head registered under ``name``."""
    spec: HeadSpec = get(name)
    if spec.repo:
        add_repo(spec.repo)
    module = importlib.import_module(f".{spec.module}", __package__)
    cls = getattr(module, spec.cls)
    kwargs = {**spec.extra, **overrides}
    return cls(model_id=spec.model_id, device=device, **kwargs)


def load(name: str, device: str = "cuda", quiet: bool = False, **overrides) -> Head:
    """Construct and load a head onto ``device``."""
    head = build(name, device, **overrides)
    head.load()
    if not quiet:
        print(f"  loaded {name:22s} <- {get(name).model_id}", file=sys.stderr)
    return head
