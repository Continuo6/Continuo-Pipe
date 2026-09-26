#!/usr/bin/env bash
# Bootstrap an isolated Phase-1 environment for Continuo-Pipe.
#
# Phase 1 (main.py) runs the feature stack: separator + diarizer + VAD +
# scoring + LID + exporter. It does *not* do ASR — Phase 2 (transcribe.py,
# Qwen3-ASR via vLLM) lives in a separate env to avoid dep conflicts
# (vLLM aggressively pins torch / transformers / CUDA; DiariZen's
# vendored pyannote-audio fork wants numpy<2 and an older torchaudio API).
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# A plain ``pip install -r`` does not reproduce a working pipeline. The
# DiariZen + Kim FT3 + FireRedASR2S stack needs:
#
#   1. A vendored pyannote-audio fork (DiariZen adds custom kwargs to
#      ``SpeakerDiarization.__init__``; stock pyannote 3.x rejects them).
#      The fork must be installed *non-editable* — its ``pyannote/__init__.py``
#      has license content that breaks namespace-package merging with
#      ``pyannote.core`` / ``pyannote.database`` when installed editable.
#
#   2. ``toml`` (DiariZen's setup.py forgot to declare it).
#
#   3. ``onnxruntime-gpu`` reinstalled last, because some transitive deps
#      pull in a CPU-only ``onnxruntime`` build whose package shadows the
#      GPU one and leaves the module with no ``__file__``.
#
#   4. ``numpy<2`` to keep ``np.NaN`` / ``np.NAN`` alive — the vendored
#      pyannote-audio fork still uses both. If numpy 2.x ends up installed,
#      the script applies a sed patch as a fallback.
#
#   5. FireRedASR2S cloned to ``third_party/FireRedASR2S``; the FireRedVAD /
#      FireRedLID adapters sys.path-prepend it at warmup. The repo's own
#      pyproject pins torch 2.10 / transformers 5.1 which doesn't match
#      this env, so we DO NOT pip-install it — we only clone for the
#      source tree.
#
#   6. cuDNN / cuBLAS shared libs on LD_LIBRARY_PATH at runtime so the GPU
#      onnxruntime CUDA provider actually loads. ``env_phase1.sh`` (sourced
#      before running) handles that — keep them as two separate concerns:
#      setup vs. invocation.
#
# USAGE
# -----
#   ./setup_phase1.sh                                  # create ./.venv-phase1 (default)
#   PYTHON_BIN=../.venv/custom/bin/python ./setup_phase1.sh
#                                                     # install into an existing interpreter
#
# After setup, source env_phase1.sh before running:
#   source env_phase1.sh
#   python main.py --config_path config.json

set -euo pipefail

cd "$(dirname "$0")"
HERE="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-}"          # if empty, create a fresh venv
VENV_DIR="${VENV_DIR:-$HERE/.venv-phase1}"
DIARIZEN_SRC="${DIARIZEN_SRC:-$HERE/third_party/DiariZen}"
FIRERED_SRC="${FIRERED_SRC:-$HERE/third_party/FireRedASR2S}"

log() { printf '\n\033[1;34m[setup-phase1]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[setup-phase1] %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Resolve target Python.
# ---------------------------------------------------------------------------
if [[ -z "$PYTHON_BIN" ]]; then
    if ! command -v python3.10 >/dev/null && ! command -v python3.11 >/dev/null && ! command -v python3.12 >/dev/null; then
        die "need python 3.10+ on PATH (audio-separator 0.44 requires it). \
Pass PYTHON_BIN explicitly."
    fi
    BASE_PY=$(command -v python3.10 || command -v python3.11 || command -v python3.12)
    log "creating venv at $VENV_DIR (base: $BASE_PY)"
    "$BASE_PY" -m venv "$VENV_DIR"
    PYTHON_BIN="$VENV_DIR/bin/python"
fi

"$PYTHON_BIN" --version || die "PYTHON_BIN=$PYTHON_BIN not runnable"

PY_VER=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
PY_MAJ=${PY_VER%.*}; PY_MIN=${PY_VER#*.}
if (( PY_MAJ < 3 )) || (( PY_MAJ == 3 && PY_MIN < 10 )); then
    die "audio-separator 0.44 requires Python >= 3.10 (got $PY_VER)"
fi

PIP=("$PYTHON_BIN" -m pip)
"${PIP[@]}" install --upgrade pip wheel setuptools >/dev/null

# ---------------------------------------------------------------------------
# 2. Base pinned deps (torch, numpy, audio_separator, ...).
# ---------------------------------------------------------------------------
log "installing pinned Phase-1 deps from requirements-phase1.txt"
"${PIP[@]}" install -r "$HERE/requirements-phase1.txt"

# ---------------------------------------------------------------------------
# 3. DiariZen + its vendored pyannote-audio fork.
# ---------------------------------------------------------------------------
log "cloning DiariZen (with submodules) → $DIARIZEN_SRC"
mkdir -p "$(dirname "$DIARIZEN_SRC")"
if [[ ! -d "$DIARIZEN_SRC/.git" ]]; then
    git clone --recurse-submodules https://github.com/BUTSpeechFIT/DiariZen.git "$DIARIZEN_SRC"
else
    log "  DiariZen already cloned; keeping its current revision"
    git -C "$DIARIZEN_SRC" submodule update --init --recursive
fi

log "installing diarizen package"
"${PIP[@]}" install "$DIARIZEN_SRC" --no-deps

# Vendored pyannote-audio fork. MUST be non-editable: the fork's
# pyannote/__init__.py has license content, which when installed editable
# turns ``pyannote`` from a namespace package into a regular package and
# hides ``pyannote.core``/``pyannote.database`` from import.
log "installing vendored pyannote-audio fork (non-editable)"
"${PIP[@]}" uninstall -y pyannote.audio || true
"${PIP[@]}" install "$DIARIZEN_SRC/pyannote-audio" --no-deps

# ---------------------------------------------------------------------------
# 4. Fallback patch for numpy 2.x compatibility.
# ---------------------------------------------------------------------------
NP_MAJ=$("$PYTHON_BIN" -c 'import numpy; print(numpy.__version__.split(".")[0])')
if (( NP_MAJ >= 2 )); then
    log "numpy $NP_MAJ.x detected — sed-patching np.NaN/np.NAN in pyannote.audio"
    # ``import pyannote.audio`` would itself raise on np.NaN here, so resolve
    # the install location via sysconfig instead.
    SITE_DIR=$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    PYANN_DIR="$SITE_DIR/pyannote/audio"
    [[ -d "$PYANN_DIR" ]] || die "pyannote.audio not found at $PYANN_DIR"
    find "$PYANN_DIR" -name "*.py" -exec sed -i \
        -e 's/np\.NaN/np.nan/g' \
        -e 's/np\.NAN/np.nan/g' {} +
    # Nuke __pycache__: pip's install produced .pyc files compiled from the
    # pre-patch sources; some .pyc headers (PEP 552 hash-based invalidation)
    # don't get re-validated on a mtime bump, so the patched .py is ignored
    # in favor of the cached bytecode unless we delete it.
    find "$PYANN_DIR" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 5. Reinstall onnxruntime-gpu LAST.
# ---------------------------------------------------------------------------
log "reinstalling onnxruntime-gpu (defensive)"
"${PIP[@]}" uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
"${PIP[@]}" install --force-reinstall --no-deps "onnxruntime-gpu==1.20.0"

# ---------------------------------------------------------------------------
# 6. FireRedASR2S source clone (used by VAD + LID adapters at runtime).
# ---------------------------------------------------------------------------
if [[ ! -d "$FIRERED_SRC/.git" ]]; then
    log "cloning FireRedASR2S → $FIRERED_SRC"
    git clone https://github.com/FireRedTeam/FireRedASR2S.git "$FIRERED_SRC" || \
        log "  clone failed; populate $FIRERED_SRC manually if you need FireRedVAD/LID"
else
    log "FireRedASR2S already present at $FIRERED_SRC"
fi

# ---------------------------------------------------------------------------
# 7. Verification.
# ---------------------------------------------------------------------------
log "verifying imports"
"$PYTHON_BIN" -c "
import torch, torchaudio
print(f'torch          {torch.__version__}  cuda_built={torch.version.cuda}')
import numpy; print(f'numpy          {numpy.__version__}')
from audio_separator.separator import Separator
print('audio_separator OK')
from diarizen.pipelines.inference import DiariZenPipeline
print('diarizen        OK')
import pyannote.audio, pyannote.core, pyannote.database
print(f'pyannote.audio {pyannote.audio.__version__} (pyannote.core {pyannote.core.__version__})')
import onnxruntime as ort
print(f'onnxruntime    {ort.__version__}  providers={ort.get_available_providers()}')
import sys; sys.path.insert(0, '$FIRERED_SRC')
try:
    from fireredasr2s.fireredvad.vad import FireRedVad, FireRedVadConfig
    from fireredasr2s.fireredlid.lid import FireRedLid, FireRedLidConfig
    print('FireRedASR2S    OK')
except ImportError as e:
    print(f'FireRedASR2S    not importable ({e}) — VAD/LID adapters will fail at warmup')
"

cat <<'EOM'

[setup-phase1] done.

Next steps:
  source env_phase1.sh         # exports LD_LIBRARY_PATH for cuDNN/cuBLAS
  python main.py --config_path config.json

Phase 2 (transcribe.py) runs in a separate env with vLLM installed —
keep them independent.

To verify CUDA actually works (requires GPU-visible shell, e.g. inside srun):
  source env_phase1.sh
  python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
EOM
