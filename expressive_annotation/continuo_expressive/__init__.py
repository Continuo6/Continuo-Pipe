"""Continuo-Pipe expressive annotation — paralinguistic tags + TTS instruction text from raw audio.

Three passes, each its own CLI because their dependency sets genuinely conflict:

``continuo-annotate``  gender, age, accent, volume, pitch, speed  (transformers ~4.46 env)
``continuo-caption``   instruction text from the tags             (Qwen3-Omni env)

They exchange JSONL keyed by clip ``id``, so any pass can be rerun, retuned, or
skipped without disturbing the others.

Importing this package pulls in no torch and no model code — the heavy imports live
inside each head's ``load()``.
"""

__version__ = "0.1.0"
