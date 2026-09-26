# Continuo-Pipe

Continuo-Pipe has two independent speech-annotation stages and one optional
automation entry point. The **data annotation** stage turns raw recordings
into short utterances, long speech and dialogue, with transcription and
quality metadata. The **expressive annotation** stage reads a JSONL manifest
and adds voice tags, instruction captions and nonverbal vocalization (NV)
labels. You can run either stage on its own.

Each stage has its own script. The unified script runs them in sequence and,
by default, converts completed short utterances into an expressive manifest.
Use `--expressive-manifest` to annotate a different corpus.

| Entry point | Input | Main output |
|---|---|---|
| `scripts/run_data_annotation.py` | Recording folder + model config | Per-recording short, long and dialogue JSON, shared audio carrier |
| `scripts/run_expressive_annotation.py` | JSONL of clips or carrier ranges | `annotations.jsonl`, NV results, `final.jsonl` |
| `scripts/run_all.py` | Both inputs, or one folder for the default short-track bridge | Both stage outputs |

## Install in separate environments

Use Python 3.10 or 3.11. Keep the five environments separate. Phase 1 pins
PyTorch 2.5.1 and expects a compatible CUDA driver; check this before running
its installer. For the other GPU environments, install a PyTorch build
compatible with the machine's driver first. The DiariZen fork, the two vLLM
passes, the voice-tag heads and NVASR have incompatible or independently
constrained dependencies.
See [requirements/README.md](requirements/README.md) for the stage matrix.

```bash
bash data_annotation/setup_phase1.sh

python3.10 -m venv .venv-data-asr
.venv-data-asr/bin/pip install -r requirements/data-asr.txt

python3.10 -m venv .venv-expressive-tags
.venv-expressive-tags/bin/pip install -r requirements/expressive-tags.txt
.venv-expressive-tags/bin/pip install -e ./expressive_annotation --no-deps

python3.10 -m venv .venv-expressive-caption
.venv-expressive-caption/bin/pip install -r requirements/expressive-caption.txt
.venv-expressive-caption/bin/pip install -e ./expressive_annotation --no-deps

python3.10 -m venv .venv-expressive-nv
.venv-expressive-nv/bin/pip install -r requirements/expressive-nv.txt
.venv-expressive-nv/bin/pip install -e ./expressive_annotation --no-deps

bash expressive_annotation/scripts/setup_models.sh
```

The data config in `data_annotation/examples/qwen3_asr.json` is a template.
Copy it to `data_annotation/config.json` and fill the **public** VAD, LID,
separator and quality model locations you intend to use. That local config is
ignored by Git. Model source and weights are not included in this repository.
The default Phase 1 installer fetches the public DiariZen and FireRedASR2S
source trees into its ignored `third_party` directory. Install Qwen3-ASR and
vLLM only in the data ASR environment. `ffmpeg` and `ffprobe` must be on PATH.
For FireRedVAD, `model_dir` must name the directory containing both
`cmvn.ark` and `model.pth.tar` (the `VAD` subdirectory of its public checkpoint).
The expressive tags pass needs the public WavLM age/sex checkpoint, its
`microsoft/wavlm-large` backbone, the relevant Voxlect dialect checkpoint,
and PENN's `fcnf0++` checkpoint. An offline run needs all of them cached.

For expressive NV, set `CONTINUO_EXPRESSIVE_NVBENCH_REPO` and
`CONTINUO_EXPRESSIVE_NVASR_DIR` to the external model source and checkpoint.
The verified NV mode additionally needs `AUDIOLDM_ROOT`, `CKPT_ROOT`, and
`CONTINUO_EXPRESSIVE_PANNS_LABELS`. These point to external
assets and are never committed. `--nv-mode direct` runs NVASR without the
PANNs/verification passes. The verified path adds the sung-audio filter and
the verification stages.

## Run either stage

```bash
python scripts/run_data_annotation.py \
  --input audio --config data_annotation/config.json

python scripts/run_expressive_annotation.py \
  --manifest examples/manifest.example.jsonl --out-dir runs/expressive \
  --nv-mode direct
```

An expressive manifest is one JSON object per clip. Paths may be relative to
the manifest's directory or to `--audio-root`. A row may name a standalone
file or a sample range in a shared carrier:

```json
{"id":"clip_1","wav_path":"audio/clip_1.wav","txt":"Hello","lang":"en"}
{"id":"clip_2","wav_path":"audio/carrier.m4a","carrier_start_samples":44100,"carrier_end_samples":88200,"sample_rate":44100,"txt":"Hi","lang":"en"}
```

The expressive script defaults to tags, caption and verified NV. Choose a
subset with `--components tags,caption,nv`; for example, `--components nv`
only runs NV. Caption uses the completed tag file, so a caption-only rerun
requires an existing `annotations.jsonl`. Existing outputs resume by ID;
the script rejects an input manifest change in the same output directory.
The `zh` caption mode uses English-written instructions and tag data that ask
the model to answer in Simplified Chinese.
`final.jsonl` retains input fields and joins stage results by ID, failing on
missing or duplicate results rather than silently discarding clips.

## Run both automatically

```bash
python scripts/run_all.py \
  --input audio --data-config data_annotation/config.json \
  --out-dir runs/complete --nv-mode direct
```

The default bridge creates one expressive unit per finished short-track row.
Use `--tracks short,long,dialogue` to include long members and dialogue turns
as separate units; IDs encode their source track and parent recording. Shared
carrier audio is decoded once per cached recording and sliced at the original
sample rate. For an unrelated expressive corpus, supply
`--expressive-manifest examples/manifest.example.jsonl`; the data stage still runs
independently. Use `--data-stages none` with an existing `--processed-root`
or independent manifest to skip data inference. All three scripts offer
`--dry-run` where applicable.
