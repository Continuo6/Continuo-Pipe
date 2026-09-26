# Expressive annotation

This component labels voice properties, writes instruction captions, and
annotates nonverbal vocalizations (NV). It accepts its own JSONL manifest;
it does not require the data annotation component. Use
`scripts/run_expressive_annotation.py` from the repository root.

Each row needs a unique `id` and either `wav_path` or `source_tar` plus
`source_member`. A row can also point to a shared carrier using
`carrier_start_samples`, `carrier_end_samples` and `sample_rate`.

Emotion labels are optional. Per-clip emotion predictions can be merged from a
JSONL with `--emotion-jsonl` together with an explicit `--emotion-tau`
threshold. Without them, instruction generation proceeds without emotion labels.

The verified NV pipeline additionally needs the NV-Bench and PANNs source
checkouts, weights, and an AudioSet label CSV. Direct NV annotation is
available with `--nv-mode direct`.
