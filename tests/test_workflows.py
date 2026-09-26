from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "expressive_annotation"))
sys.path.insert(0, str(ROOT / "data_annotation"))


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def write_wav(path: Path, samples: np.ndarray, rate: int = 16000):
    with wave.open(str(path), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(rate)
        sink.writeframes(np.asarray(samples * 32767, dtype="<i2").tobytes())


def test_bridge_exports_all_tracks_as_independent_units(tmp_path):
    exporter = script("export_data_manifest")
    root = tmp_path / "processed"
    sid_dir = root / "ab" / "abc123"
    sid_dir.mkdir(parents=True)
    audio = sid_dir / "abc123.wav"
    write_wav(audio, np.zeros(16000), rate=16000)
    shared = {"carrier_path": audio.name, "sample_rate": 16000,
              "carrier_start_samples": 0, "carrier_end_samples": 16000,
              "start": 0, "end": 1}
    (sid_dir / "abc123.json").write_text(json.dumps([dict(shared, index="000", text="hello")]))
    (sid_dir / "abc123.long.json").write_text(json.dumps([
        dict(shared, index="L000", members=[{"start": 0.1, "end": 0.4, "text": "long"}])]))
    (sid_dir / "abc123.dialogue.json").write_text(json.dumps([
        dict(shared, index="D000", turns=[{"start": 0.5, "end": 0.8,
                                             "speaker": "S1", "text": "dialogue"}])]))
    out = tmp_path / "manifest.jsonl"
    assert exporter.export(root, out, ["short", "long", "dialogue"]) == 3
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len({row["id"] for row in rows}) == 3
    assert {row["source_track"] for row in rows} == {"short", "long", "dialogue"}
    assert [row for row in rows if row["source_track"] == "long"][0]["carrier_start_samples"] == 1600
    assert [row for row in rows if row["source_track"] == "dialogue"][0]["speaker"] == "S1"


def test_carrier_slice_uses_native_sample_bounds(tmp_path):
    from continuo_expressive.carrier_source import load_carrier_slice
    audio = tmp_path / "carrier.wav"
    samples = np.zeros(16000, dtype=np.float32)
    samples[4000:8000] = 0.5
    write_wav(audio, samples)
    row = {"wav_path": str(audio), "sample_rate": 16000,
           "carrier_start_samples": 4000, "carrier_end_samples": 8000}
    clip = load_carrier_slice(row, 16000)
    assert len(clip) == 4000
    assert np.allclose(clip, 0.5, atol=0.0001)


def test_caption_receives_each_shared_carrier_slice(tmp_path, monkeypatch):
    from continuo_expressive.cli import caption

    audio = tmp_path / "carrier.wav"
    write_wav(audio, np.r_[np.full(16000, 0.1), np.full(16000, 0.8)])
    annot = tmp_path / "annotations.jsonl"
    annot.write_text("".join(json.dumps({"id": clip_id, "txt": transcript, "lang": "en"}) + "\n"
                             for clip_id, transcript in (("a", "hello"), ("b", "world"))))
    manifest = tmp_path / "manifest.jsonl"
    with manifest.open("w") as out:
        for clip_id, start, end in (("a", 0, 16000), ("b", 16000, 32000)):
            out.write(json.dumps({"id": clip_id, "wav_path": str(audio),
                                  "carrier_start_samples": start,
                                  "carrier_end_samples": end,
                                  "sample_rate": 16000}) + "\n")

    class FakeCaptioner:
        chunk_size = 2

        def load(self):
            pass

        def generate(self, items):
            return [f"level={float(np.mean(wav)):.1f}" for _, wav, _ in items]

    monkeypatch.setattr(caption, "build_captioner", lambda *args, **kwargs: FakeCaptioner())
    assert caption.main(["--annot", str(annot), "--manifest", str(manifest),
                         "--workers", "1"]) == 0
    rows = [json.loads(line) for line in annot.read_text().splitlines()]
    assert [row["caption"] for row in rows] == ["level=0.1", "level=0.8"]


def test_merge_requires_complete_ids_and_preserves_source_text(tmp_path):
    merge = script("merge_results").merge
    manifest = tmp_path / "input.jsonl"
    manifest.write_text('{"id":"a","txt":"original"}\n{"id":"b","txt":"second"}\n')
    tags = tmp_path / "tags.jsonl"
    tags.write_text('{"id":"a","caption":"bright"}\n{"id":"b","caption":"quiet"}\n')
    nv = tmp_path / "nv.jsonl"
    nv.write_text('{"id":"a","text":"other","nv_tags":["Laughter"],"n_nv":1}\n')
    out = tmp_path / "final.jsonl"
    with pytest.raises(ValueError, match="missing NV result"):
        merge(manifest, out, tags, nv)
    assert not out.exists()
    nv.write_text(nv.read_text() + '{"id":"b","nv_tags":[],"n_nv":0}\n')
    assert merge(manifest, out, tags, nv) == 2
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["txt"] == "original"
    assert rows[0]["caption"] == "bright"
    assert rows[0]["nv_tags"] == ["Laughter"]


def test_default_data_config_leaves_optional_gates_unset():
    from pipeline.config import FilterConfig, LongChunkConfig, PipelineConfig
    config = json.loads((ROOT / "data_annotation/examples/qwen3_asr.json").read_text())
    parsed = PipelineConfig.model_validate(config)
    assert parsed.stages.deepfake is None
    assert FilterConfig().drop_fake_zh_en is False
    assert LongChunkConfig().max_fake_coverage is None


def test_data_helpers_skip_completed_audio_and_keep_filter_results(tmp_path):
    from utils.tool import calculate_audio_stats, get_audio_files, get_char_count

    source = tmp_path / "audio"
    source.mkdir()
    for name in ("one.mp3", "two.wav", "notes.txt"):
        (source / name).write_bytes(b"sample")
    assert {Path(path).name for path in get_audio_files(source)} == {"one.mp3", "two.wav"}

    completed = tmp_path / "audio_processed" / "one"
    completed.mkdir(parents=True)
    (completed / "one.json").write_text("{}")
    assert [Path(path).name for path in get_audio_files(source)] == ["two.wav"]

    assert get_char_count("A, é!") == 2
    accepted, all_rows = calculate_audio_stats([
        {"start": 0, "end": 2, "text": "ab", "dnsmos": 3.0},
        {"start": 0, "end": 3, "text": "xy", "dnsmos": 2.0},
    ])
    assert accepted == [(0, 2)]
    assert all_rows == [(0, 2), (1, 3)]


def test_timing_records_are_drained():
    import logging
    from utils.logger import drain_timed_spans, time_span

    drain_timed_spans()
    with time_span("decode", logger=logging.getLogger("test-timing")):
        pass
    records = drain_timed_spans()
    assert len(records) == 1 and records[0][0] == "decode"
    assert records[0][1] >= 0
    assert drain_timed_spans() == []


def test_optional_detector_fields_remain_unevaluated(tmp_path, monkeypatch):
    import logging

    from pipeline.config import FilterConfig, LongChunkConfig
    from pipeline.context import PipelineContext
    from pipeline.stages.long_chunk_stage import LongChunkStage
    from pipeline.stages.quality_filter_stage import QualityFilterStage
    from pipeline.types import Segment
    from utils.logger import Logger

    monkeypatch.setattr(Logger, "get_logger", lambda: logging.getLogger("test-public"))
    ctx = PipelineContext(audio_path=str(tmp_path / "in.wav"),
                          save_path=str(tmp_path), audio_name="sample")
    ctx.segments = [
        Segment(start=0, end=20, speaker="S1", index="000", quality=3.5),
        Segment(start=20, end=40, speaker="S1", index="001", quality=4.0),
    ]
    QualityFilterStage(FilterConfig()).run(ctx)
    LongChunkStage(LongChunkConfig()).run(ctx)
    assert all(row.extra["is_fake"] is None and row.extra["deepfake_score"] is None
               for row in ctx.segments)
    assert len(ctx.long_chunks) == 1
    assert ctx.long_chunks[0].extra["fake_coverage"] is None


def test_external_emotion_requires_explicit_threshold():
    from continuo_expressive.aggregate import aggregate
    from continuo_expressive.ensemble.emotion_gate import gate_emotion

    raw = {"label": "happy", "confidence": 0.8,
           "scores": {"happy": 0.8, "neutral": 0.2}}
    with pytest.raises(ValueError, match="explicit emotion threshold"):
        gate_emotion(raw)
    with pytest.raises(ValueError, match="explicit emotion threshold"):
        aggregate([{"rel_start": 0, "rel_end": 1, "dur_s": 1,
                    "emotion_scores": raw["scores"]}])
    assert gate_emotion(None)["emotion"] is None


def test_nv_verify_uses_public_checks_without_emotion_model(tmp_path):
    source = tmp_path / "nv.jsonl"
    source.write_text(json.dumps({"id": "one", "nv_tags": ["Laughter"],
                                  "nv_text": "hello[Laughter]", "txt": "hello",
                                  "asr_ratio": 1.0}) + "\n")
    result = tmp_path / "verified.jsonl"
    subprocess.run([sys.executable, str(ROOT / "expressive_annotation/tools/nv_verify.py"),
                    "--nv", str(source), "--out", str(result), "--no-panns"],
                   cwd=ROOT / "expressive_annotation", check=True, capture_output=True)
    row = json.loads(result.read_text().splitlines()[0])
    assert row["nv_verified"] == ["Laughter"]
    assert row["nv_rejected"] == {}


def test_missing_interpreter_fails_before_any_gpu_stage(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    missing = tmp_path / "missing-python"
    data = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_data_annotation.py"),
         "--input", str(inputs), "--config", str(ROOT / "data_annotation/examples/qwen3_asr.json"),
         "--stages", "both", "--phase1-python", sys.executable,
         "--asr-python", str(missing)], capture_output=True, text=True)
    assert data.returncode != 0 and "asr interpreter does not exist" in data.stderr
    assert not (tmp_path / "inputs_processed").exists()

    audio = tmp_path / "audio.wav"
    write_wav(audio, np.zeros(16000))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "one", "wav_path": str(audio)}) + "\n")
    expressive = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_expressive_annotation.py"),
         "--manifest", str(manifest), "--out-dir", str(tmp_path / "expressive"),
         "--components", "tags,caption", "--tags-python", sys.executable,
         "--caption-python", str(missing)], capture_output=True, text=True)
    assert expressive.returncode != 0 and "caption interpreter does not exist" in expressive.stderr
    assert not (tmp_path / "expressive").exists()


def test_release_scan_checks_staged_ignored_files(tmp_path, monkeypatch):
    scan = script("check_release")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("runs/\n")
    runtime = tmp_path / "runs"
    runtime.mkdir()
    private = runtime / "private.txt"
    private.write_text(str(Path("/") / "mnt" / "workspace" / "private")
                       + " " + "node" + "4\n")
    subprocess.run(["git", "add", "-f", "runs/private.txt"], cwd=tmp_path, check=True)
    private.write_text("safe working tree\n")
    monkeypatch.setattr(scan, "ROOT", tmp_path)
    assert any("runs/private.txt" in problem and "index" in problem
               for problem in scan.check())


def test_release_scan_rejects_non_english_source():
    scan = script("check_release")
    content = ("example " + chr(0x4E2D)).encode("utf-8")
    assert scan._scan(Path("example.txt"), content, "worktree")
