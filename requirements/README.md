# Environments

Create separate Python 3.10 or 3.11 environments. Phase 1 pins PyTorch 2.5.1;
verify its CUDA runtime matches the machine's driver before using its installer.
Install a driver-compatible Torch wheel first in the other GPU environments.
Do not install these files into one environment: data Phase 1 uses DiariZen's
pyannote fork, data ASR and captioning use separate vLLM stacks, and expressive
tags require Transformers 4.4x.

| Interpreter | Requirements | Purpose |
|---|---|---|
| `data_annotation/.venv-phase1/bin/python` | `data-phase1.txt` via `data_annotation/setup_phase1.sh` | separation, diarization, VAD, quality, carrier export |
| `.venv-data-asr/bin/python` | `data-asr.txt` | Qwen3-ASR / vLLM transcription |
| `.venv-expressive-tags/bin/python` | `expressive-tags.txt` | voice tags |
| `.venv-expressive-caption/bin/python` | `expressive-caption.txt` | instruction captions |
| `.venv-expressive-nv/bin/python` | `expressive-nv.txt` | NVASR, PANNs and NV verification |

Install the expressive package into each expressive environment with
`pip install -e ./expressive_annotation --no-deps`. The model source checkouts and
weights live outside version control. See the top-level README for setup and
model paths.

Run `python -m pip check` in each finished environment. Keep each environment's
resolved `pip freeze` with the run metadata when reproducibility matters; the
requirements files express compatibility boundaries rather than a universal CUDA
lockfile.
