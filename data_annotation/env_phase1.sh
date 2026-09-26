# Phase 1 env helper: prepends NVIDIA wheel libs to LD_LIBRARY_PATH so
# onnxruntime-gpu's CUDAExecutionProvider can find cuDNN / cuBLAS at
# import time. Without this, DNSMOS scoring silently falls back to CPU
# (~10x slower) — we now fail-fast in models/dnsmos.py instead, so a
# misconfigured env will raise loudly.
#
# Usage:
#   source env_phase1.sh                          # uses ./.venv-phase1/bin/python
#   PYTHON_BIN=../.venv/custom/bin/python source env_phase1.sh
#
# Safe to source repeatedly; entries aren't duplicated.
#
# NOT for Phase 2 (transcribe.py / vLLM) — vLLM bundles its own CUDA
# libs and doesn't need this. If you have a separate env for Phase 2,
# don't source this script there.

_p1_python="${PYTHON_BIN:-$(dirname "${BASH_SOURCE[0]}")/.venv-phase1/bin/python}"
if [[ ! -x "$_p1_python" ]]; then
    echo "[env_phase1.sh] $_p1_python not executable; set PYTHON_BIN" >&2
    return 1 2>/dev/null || exit 1
fi

_p1_site=$("$_p1_python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")

_p1_prepend() {
    local d="$1"
    [[ -d "$d" ]] || return 0
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$d:"*) ;;
        *) LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}";;
    esac
}

for _lib in cudnn cublas cuda_runtime cuda_nvrtc cufft curand cusparse cusolver; do
    _p1_prepend "$_p1_site/nvidia/$_lib/lib"
done
export LD_LIBRARY_PATH

unset _p1_python _p1_site _p1_prepend _lib
