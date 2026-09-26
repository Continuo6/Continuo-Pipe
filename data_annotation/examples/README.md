# Data annotation configuration

Copy `qwen3_asr.json` to `../config.json`, then fill the public model source
and checkpoint paths required by the selected adapters. Keep that local file
out of version control. `stages.deepfake` is optional and left `null` in both
templates.

The `indic_conformer.json` example selects the optional Indic ASR adapter.
Both templates enable the short, long and dialogue tracks.

Run the data stage through `scripts/run_data_annotation.py` at the repository
root. It uses separate interpreters for feature extraction and ASR.
