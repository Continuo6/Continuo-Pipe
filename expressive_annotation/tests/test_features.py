"""Speed derivation and the batch loader — the parts that need no model weights."""
from __future__ import annotations

import numpy as np
import pytest

from continuo_expressive.audio import BatchLoader, batched, truncate
from continuo_expressive.config import TARGET_SR
from continuo_expressive.features import speed

sf = pytest.importorskip("soundfile")


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * TARGET_SR), dtype=np.float32)


# ------------------------------------------------------------------ speed
def test_manifest_text_uses_the_whole_clip():
    # 40 chars over 10 s = 4.0 CPS
    out = speed.from_text("x" * 40, "zh", silence(10))
    assert out["speed_cps"] == 4.0
    assert out["speed"] == "measured"


def test_asr_text_caps_the_denominator_at_the_window_whisper_saw():
    # whisper only transcribes the first 30 s, so a 60 s clip must divide by 30,
    # not 60 — otherwise every long clip reads as artificially slow
    long_clip = silence(60)
    assert speed.from_asr("x" * 150, "zh", long_clip)["speed_cps"] == 5.0
    assert speed.from_text("x" * 150, "zh", long_clip)["speed_cps"] == 2.5


def test_no_cap_below_the_window():
    short = silence(10)
    assert speed.from_asr("x" * 40, "zh", short) == speed.from_text("x" * 40, "zh", short)


def test_language_without_edges_reports_the_rate_but_no_bucket():
    out = speed.from_text("x" * 40, "fr", silence(10))
    assert out["speed_cps"] == 4.0
    assert out["speed"] is None


def test_unknown_language_is_handled_like_an_unlisted_one():
    assert speed.from_text("x" * 40, None, silence(10))["speed"] is None


def test_clips_under_half_a_second_are_not_measured():
    assert speed.from_text("hello", "en", silence(0.3)) == {"speed_cps": None, "speed": None}


def test_empty_transcript_yields_nothing():
    assert speed.from_text("", "en", silence(10))["speed_cps"] is None
    assert speed.from_asr(None, "en", silence(10))["speed_cps"] is None


def test_character_count_is_raw_including_spaces():
    # the English edges were fit on PSC `transcription`, spaces and punctuation kept
    text = "a b, c."                      # 7 characters
    assert speed.from_text(text, "en", silence(1))["speed_cps"] == 7.0


# ------------------------------------------------------------------ loader
def test_truncate_caps_and_leaves_short_clips_alone():
    assert len(truncate(silence(20), 15.0)) == 15 * TARGET_SR
    assert len(truncate(silence(3), 15.0)) == 3 * TARGET_SR


def test_batched_covers_every_item():
    chunks = list(batched(list(range(10)), 3))
    assert [len(c) for c in chunks] == [3, 3, 3, 1]
    assert [x for c in chunks for x in c] == list(range(10))


def _corpus(tmp_path, n, seconds=1.0):
    rows = []
    for i in range(n):
        p = tmp_path / f"clip{i}.wav"
        sf.write(p, silence(seconds), TARGET_SR)
        rows.append({"id": f"clip{i}", "_path": p})
    return rows


def test_loader_yields_every_clip_in_order(tmp_path):
    rows = _corpus(tmp_path, 7)
    seen = [r["id"] for batch, _, _ in BatchLoader(rows, batch_size=3) for r in batch]
    assert seen == [r["id"] for r in rows]


def test_loader_runs_per_clip_work_and_aligns_it(tmp_path):
    rows = _corpus(tmp_path, 5)
    loader = BatchLoader(rows, batch_size=2,
                         per_clip=lambda row, wav: {"n": len(wav), "id": row["id"]})
    for batch, wavs, extras in loader:
        assert [e["id"] for e in extras] == [r["id"] for r in batch]
        assert [e["n"] for e in extras] == [len(w) for w in wavs]


def test_one_unreadable_clip_does_not_stop_the_run(tmp_path):
    rows = _corpus(tmp_path, 3)
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not audio at all")
    rows.insert(1, {"id": "broken", "_path": broken})

    failures = []
    loader = BatchLoader(rows, batch_size=2,
                         on_error=lambda row, exc: failures.append(row["id"]))
    seen = [r["id"] for batch, _, _ in loader for r in batch]
    assert "broken" not in seen
    assert len(seen) == 3
    assert failures == ["broken"]


def test_empty_input_yields_nothing(tmp_path):
    assert list(BatchLoader([], batch_size=4)) == []


# --------------------------------------- a mixed manifest must not lose its transcripts
class _StubTranscriber:
    """Records what it was asked to hear, and answers with a fixed transcript."""

    def __init__(self):
        self.calls = []

    def transcribe(self, waveforms):
        self.calls.append(len(waveforms))
        return [("asr words here", "fr")] * len(waveforms)


def test_only_rows_without_a_transcript_are_sent_to_asr():
    # a handful of rows missing `txt` — long-audio sub-windows, say — must not have
    # whisper overwrite every other row's own transcript, language, and speed denominator
    from continuo_expressive.cli import annotate as cli

    rows = [{"id": "a", "txt": "hello there", "lang": "en"},
            {"id": "b", "txt": "", "lang": None},
            {"id": "c", "txt": "more words", "lang": "en"}]
    wavs = [np.zeros(16000 * 3, dtype=np.float32) for _ in rows]
    stub = _StubTranscriber()

    class _Pool:
        age_gender = age_vox = None
    # exercise just the transcript branch rather than the whole batch path
    captured = {}

    def fake_accent(pool, waves, langs):
        captured["langs"] = list(langs)
        return [{"accent": None, "accent_top3": None, "accent_head": None} for _ in waves]

    original = cli._accent_for_batch
    cli._accent_for_batch = fake_accent
    try:
        class _Head:
            def predict(self, batch):
                from continuo_expressive.heads.base import Prediction
                return [Prediction(attribute="x", pred_label=None) for _ in batch]
        pool = _Pool()
        pool.age_gender = _Head()
        pool.age_vox = _Head()
        records = cli.annotate_batch(pool, stub, rows, wavs, [{}] * 3, None)
    finally:
        cli._accent_for_batch = original

    assert stub.calls == [1], "only the row without a transcript should reach ASR"
    assert captured["langs"] == ["en", "fr", "en"], "known languages must survive"
    assert records[0].get("asr_text") is None and records[2].get("asr_text") is None
    assert records[1]["asr_text"] == "asr words here"
