"""Re-splitting a half-finished pass must move rows, never duplicate or lose them."""
from __future__ import annotations

import importlib.util
import json
import zlib
from pathlib import Path

import pytest

from continuo_expressive.jsonl import load_manifest

_spec = importlib.util.spec_from_file_location(
    "reshard_annot", Path(__file__).resolve().parents[1] / "tools" / "reshard_annot.py")
reshard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reshard)


def build(tmp_path, n_rows=40, old_shards=5):
    """A manifest and the annot files a run on `old_shards` workers would have left."""
    rows = [{"id": f"p{i % 7}_{i:04d}", "parent_id": f"p{i % 7}",
             "source_tar": "t.tar", "source_member": f"{i}.m4a"} for i in range(n_rows)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    work = tmp_path / "work"
    work.mkdir()
    for i in range(old_shards):
        own = [r for r in rows if zlib.crc32(r["parent_id"].encode()) % old_shards == i]
        (work / f"annot{i:02d}.jsonl").write_text(
            "".join(json.dumps({"id": r["id"], "pitch_hz": 100.0}) + "\n" for r in own))
    return manifest, work, rows


def read_all(work):
    out = []
    for f in sorted(work.glob("annot[0-9][0-9].jsonl")):
        out += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return out


@pytest.mark.parametrize("new_shards", [1, 4, 5, 6, 8])
def test_every_row_survives_exactly_once(tmp_path, new_shards):
    manifest, work, rows = build(tmp_path)
    before = {r["id"] for r in read_all(work)}
    reshard.main(["--work", str(work), "--manifest", str(manifest),
                  "--shards", str(new_shards)])
    after = [r["id"] for r in read_all(work)]
    assert sorted(after) == sorted(before), "no row gained or lost"
    assert len(after) == len(set(after)), "no row duplicated"


@pytest.mark.parametrize("new_shards", [4, 6])
def test_each_worker_resumes_on_exactly_what_it_owns(tmp_path, new_shards):
    """The point of the exercise: worker i must find its own rows already done, so it
    re-annotates none of them and the others do not redo them either."""
    manifest, work, rows = build(tmp_path)
    reshard.main(["--work", str(work), "--manifest", str(manifest),
                  "--shards", str(new_shards)])
    for i in range(new_shards):
        mine = {r["id"] for r in load_manifest(manifest, shard=(i, new_shards))}
        done = {json.loads(l)["id"]
                for l in (work / f"annot{i:02d}.jsonl").read_text().splitlines() if l.strip()}
        assert done == mine, f"shard {i} resumes on rows it does not own (or misses its own)"


def test_files_above_the_new_count_are_removed(tmp_path):
    manifest, work, _ = build(tmp_path, old_shards=5)
    reshard.main(["--work", str(work), "--manifest", str(manifest), "--shards", "3"])
    assert sorted(f.name for f in work.glob("annot[0-9][0-9].jsonl")) == [
        "annot00.jsonl", "annot01.jsonl", "annot02.jsonl"]


def test_a_partly_finished_pass_keeps_only_what_was_done(tmp_path):
    manifest, work, rows = build(tmp_path)
    # drop half of shard 0's rows, as an interrupted worker would have
    f = work / "annot00.jsonl"
    kept = f.read_text().splitlines()[: len(f.read_text().splitlines()) // 2]
    f.write_text("".join(l + "\n" for l in kept))
    done_before = {r["id"] for r in read_all(work)}
    reshard.main(["--work", str(work), "--manifest", str(manifest), "--shards", "6"])
    assert {r["id"] for r in read_all(work)} == done_before


def test_dry_run_changes_nothing(tmp_path):
    manifest, work, _ = build(tmp_path)
    before = {f.name: f.read_text() for f in work.glob("annot*.jsonl")}
    reshard.main(["--work", str(work), "--manifest", str(manifest),
                  "--shards", "6", "--dry-run"])
    assert {f.name: f.read_text() for f in work.glob("annot*.jsonl")} == before
