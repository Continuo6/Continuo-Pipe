# Data annotation

This component runs the data feature and ASR passes. Phase 1 standardizes
audio, separates vocals, diarizes speakers, detects speech, scores quality,
and writes one audio carrier plus short, long and dialogue manifests. Phase 2
transcribes those manifests in a separate ASR environment. Both passes can
resume from completed per-recording files.

Use `scripts/run_data_annotation.py` from the repository root for a folder
of recordings. For a metadata/tar corpus, `run_pipeline_multi.py` remains
available and requires explicit metadata, output and config arguments.

Synthetic-speech detection is an optional stage. With `stages.deepfake` set
to `null` (the default), `is_fake` and `deepfake_score` stay unset and the
fake-related filters are skipped. A missing decision means the segment was
not evaluated.
