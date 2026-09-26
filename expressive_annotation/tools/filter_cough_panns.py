#!/usr/bin/env python3
"""Make a high-precision Cough subset from a simple NV JSON array.

The input is the two-field export used for training::

    {"audio_path": ".../0000123_<clip-id>.m4a", "transcript": "a [Cough] b"}

``panns_filter.py score --topk 527`` supplies the score file.  Both ``Cough`` and
``Throat clearing`` count as acoustic support: the latter is a useful near-neighbour
for the annotation tag, which does not distinguish a cough from an ahem.  A rejected
tag is removed rather than automatically dropping the whole record; records carrying
another NV tag survive.

This tool is deliberately separate from the singing filter.  Its threshold must be
calibrated against a matched non-Cough control set, not copied from the rank-based
music rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path


COUGH_TAG = "[Cough]"
SCORE_LABELS = ("Cough", "Throat clearing")
NV_TAG = re.compile(r"\[[A-Za-z][A-Za-z0-9-]*\]")
EXPORT_PREFIX = re.compile(r"^[0-9]{7}_(.+)$")


def sha256(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clip_id(row: dict) -> str:
    if row.get("id"):
        return str(row["id"])
    raw = row.get("audio_path")
    if not isinstance(raw, str) or not raw:
        raise ValueError("a Cough record has neither id nor audio_path")
    stem = Path(raw).stem
    match = EXPORT_PREFIX.fullmatch(stem)
    if not match:
        raise ValueError(f"cannot recover clip id from audio_path {raw!r}")
    return match.group(1)


def load_scores(path: str | os.PathLike) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                top = row["top"]
                values = {name: float(value) for name, value in top}
                missing = [name for name in SCORE_LABELS if name not in values]
                if missing:
                    raise ValueError(
                        f"missing {missing}; score with --topk 527, not the usual top-3")
                ranks = {name: rank for rank, (name, _) in enumerate(top, 1)}
                item = {
                    "cough_score": values["Cough"],
                    "throat_clearing_score": values["Throat clearing"],
                    "max_score": max(values[name] for name in SCORE_LABELS),
                    "best_rank": min(ranks[name] for name in SCORE_LABELS),
                }
                rid = str(row["id"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{lineno}: unusable PANNs row ({exc})") from None
            if rid in out:
                raise ValueError(f"{path}:{lineno}: duplicate score id {rid!r}")
            out[rid] = item
    return out


def pass_rate(path: str | os.PathLike, threshold: float) -> dict:
    scores = load_scores(path)
    passed = sum(item["max_score"] >= threshold for item in scores.values())
    return {"records": len(scores), "passed": passed,
            "pass_rate": passed / len(scores) if scores else 0.0}


def atomic_json(path: str | os.PathLike, obj: object, *, indent: int | None = 2) -> None:
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=".cough-panns-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input", required=True, help="simple NV JSON array")
    ap.add_argument("--scores", required=True,
                    help="PANNs JSONL made with panns_filter.py score --topk 527")
    ap.add_argument("--out", required=True, help="filtered simple JSON array")
    ap.add_argument("--report", required=True)
    ap.add_argument("--evidence", default="",
                    help="optional compact JSONL with the decision for every Cough row")
    ap.add_argument("--control-scores", default="",
                    help="optional matched non-Cough PANNs JSONL used to report enrichment")
    ap.add_argument("--threshold", type=float, default=0.03,
                    help="keep Cough when max(Cough, Throat clearing) is at least this")
    ap.add_argument("--exact-transcript", action="store_true",
                    help="only review rows whose transcript is exactly [Cough]")
    ap.add_argument("--protect-first", type=int, default=0,
                    help="keep the first N reviewed rows without consulting PANNs")
    ap.add_argument("--backup", default="",
                    help="optional path to copy the input to before writing output")
    args = ap.parse_args(argv)
    if not 0 <= args.threshold <= 1:
        ap.error("--threshold must be between 0 and 1")

    if args.protect_first < 0:
        ap.error("--protect-first must be non-negative")

    input_sha256 = sha256(args.input)
    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise SystemExit(f"{args.input}: expected a JSON array of objects")
    scores = load_scores(args.scores)

    output: list[dict] = []
    evidence: list[dict] = []
    stats = {
        "input_records": len(records),
        "input_cough_records": 0,
        "input_cough_tags": 0,
        "manually_protected": 0,
        "panns_scored": 0,
        "cough_records_kept": 0,
        "cough_tags_kept": 0,
        "cough_records_rejected": 0,
        "cough_tags_removed": 0,
        "records_dropped": 0,
        "records_retained_without_cough": 0,
    }
    missing: list[str] = []
    for row in records:
        transcript = row.get("transcript")
        is_candidate = (transcript == COUGH_TAG if args.exact_transcript else
                        isinstance(transcript, str) and COUGH_TAG in transcript)
        if not is_candidate:
            output.append(row)
            continue
        stats["input_cough_records"] += 1
        n_tags = transcript.count(COUGH_TAG)
        stats["input_cough_tags"] += n_tags
        rid = clip_id(row)
        if stats["input_cough_records"] <= args.protect_first:
            output.append(row)
            stats["manually_protected"] += 1
            stats["cough_records_kept"] += 1
            stats["cough_tags_kept"] += n_tags
            evidence.append({"id": rid, "audio_path": row.get("audio_path"),
                             "decision": "manually_protected", "keep": True})
            continue
        score = scores.get(rid)
        if score is None:
            missing.append(rid)
            continue
        stats["panns_scored"] += 1
        keep = score["max_score"] >= args.threshold
        evidence.append({"id": rid, "audio_path": row.get("audio_path"), **score,
                         "threshold": args.threshold, "keep": keep})
        if keep:
            output.append(row)
            stats["cough_records_kept"] += 1
            stats["cough_tags_kept"] += n_tags
            continue

        stats["cough_records_rejected"] += 1
        stats["cough_tags_removed"] += n_tags
        cleaned = re.sub(r"[ \t]+", " ", transcript.replace(COUGH_TAG, " ")).strip()
        if NV_TAG.search(cleaned):
            new_row = dict(row)
            new_row["transcript"] = cleaned
            output.append(new_row)
            stats["records_retained_without_cough"] += 1
        else:
            stats["records_dropped"] += 1

    if missing:
        examples = ", ".join(missing[:5])
        raise SystemExit(
            f"{len(missing)} Cough record(s) have no PANNs score (for example {examples}); "
            "refusing to turn unscored into rejected")

    if args.backup:
        backup = os.path.abspath(args.backup)
        if os.path.exists(backup):
            raise SystemExit(f"backup already exists: {backup}")
        os.makedirs(os.path.dirname(backup) or ".", exist_ok=True)
        shutil.copy2(args.input, backup)
    atomic_json(args.out, output)
    if args.evidence:
        parent = os.path.dirname(os.path.abspath(args.evidence)) or "."
        os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=parent, prefix=".cough-panns-", suffix=".jsonl.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for row in evidence:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, args.evidence)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    report = {
        "criterion": {
            "tag": COUGH_TAG,
            "panns_labels": list(SCORE_LABELS),
            "score": "max(Cough, Throat clearing)",
            "threshold": args.threshold,
            "comparison": ">=",
            "transcript_match": "exact" if args.exact_transcript else "contains",
            "manually_protected_first": args.protect_first,
        },
        **stats,
        "output_records": len(output),
        "candidate_pass_rate": (stats["cough_records_kept"] /
                                stats["input_cough_records"]
                                if stats["input_cough_records"] else 0.0),
        "input": {"path": os.path.abspath(args.input), "sha256": input_sha256},
        "scores": {"path": os.path.abspath(args.scores), "sha256": sha256(args.scores)},
        "output": {"path": os.path.abspath(args.out), "sha256": sha256(args.out)},
    }
    if args.control_scores:
        control = pass_rate(args.control_scores, args.threshold)
        control.update({"path": os.path.abspath(args.control_scores),
                        "sha256": sha256(args.control_scores)})
        report["matched_non_cough_control"] = control
        if control["pass_rate"]:
            report["candidate_vs_control_enrichment"] = (
                report["candidate_pass_rate"] / control["pass_rate"])
    atomic_json(args.report, report)
    print(json.dumps({k: report[k] for k in (
        "input_records", "input_cough_records", "cough_records_kept",
        "manually_protected", "panns_scored", "cough_records_rejected", "records_dropped",
        "records_retained_without_cough", "output_records", "candidate_pass_rate")},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
