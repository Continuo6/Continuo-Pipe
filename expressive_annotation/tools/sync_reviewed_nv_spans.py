#!/usr/bin/env python3
"""Synchronize a reviewed simple span export into a dialogue NV-only JSONL.

Rows omitted from the review restore the corresponding corpus ASR segment.  Rows kept
in the review replace that segment's tagged transcript.  Aggregate NV fields and the
speaker-level copies are rebuilt, and records with no remaining NV event are omitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path


TAG_RE = re.compile(r"\[([A-Za-z][A-Za-z-]*)\]")


def sha256(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def span_key(start: object, end: object, speaker: object) -> tuple[float, float, str]:
    return (round(float(start), 6), round(float(end), 6), str(speaker or ""))


def tags_in(text: str) -> list[str]:
    return TAG_RE.findall(text)


def add_counts(total: Counter, row: dict, sign: int = 1) -> None:
    for tag, count in (row.get("nv_counts") or {}).items():
        total[tag] += sign * int(count)


def aggregate(obj: dict, spans: list[dict]) -> None:
    counts: Counter = Counter()
    texts: list[str] = []
    for span in spans:
        counts.update(span.get("tags") or [])
        if span.get("text"):
            texts.append(span["text"])
    obj["nv_tags"] = sorted(counts)
    obj["nv_counts"] = dict(counts.most_common())
    obj["n_nv"] = sum(counts.values())
    obj["nv_spans"] = spans
    if texts:
        obj["nv_text"] = texts[0] if len(texts) == 1 else texts
    else:
        obj.pop("nv_text", None)


def stitch(segments: list[dict], overrides: dict[tuple[float, float, str], str]) -> str:
    parts: list[str] = []
    last = None
    ordered = sorted(segments, key=lambda r: (
        float(r.get("rel_start") or 0.0), str(r.get("turn") or ""),
        str(r.get("txt") or ""), str(r.get("id") or "")))
    for seg in ordered:
        key = span_key(seg["rel_start"], seg["rel_end"], seg.get("turn"))
        body = overrides.get(key, seg.get("txt") or "")
        if not body:
            continue
        turn = seg.get("turn")
        if turn != last:
            if turn:
                parts.append(f"[{turn}]")
            last = turn
        parts.append(body)
    return " ".join(parts)


def atomic_json(path: str | os.PathLike, obj: object) -> None:
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".nv-sync-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
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
    ap.add_argument("--review", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--full-final", default="",
                    help="original full final.jsonl, used only if a removed span is re-added")
    ap.add_argument("--out", default="", help="candidate JSONL; omitted for dry-run")
    ap.add_argument("--report", required=True)
    args = ap.parse_args(argv)

    dry_run = not args.out
    review_sha = sha256(args.review)
    target_sha = sha256(args.target)

    with open(args.review, encoding="utf-8") as f:
        review_rows = json.load(f)
    if not isinstance(review_rows, list):
        raise SystemExit(f"{args.review}: expected a JSON array")
    review: dict[str, str] = {}
    for n, row in enumerate(review_rows, 1):
        try:
            audio_path, transcript = row["audio_path"], row["transcript"]
        except (KeyError, TypeError):
            raise SystemExit(f"review row {n}: expected audio_path and transcript") from None
        if not isinstance(audio_path, str) or not isinstance(transcript, str):
            raise SystemExit(f"review row {n}: audio_path/transcript must be strings")
        if audio_path in review:
            raise SystemExit(f"review row {n}: duplicate audio_path {audio_path}")
        review[audio_path] = transcript

    index_rows: list[dict] = []
    index_by_path: dict[str, dict] = {}
    index_by_parent: dict[str, dict[tuple[float, float, str], list[dict]]] = defaultdict(
        lambda: defaultdict(list))
    with open(args.index, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            path = row.get("audio_path")
            if not path or path in index_by_path:
                raise SystemExit(f"{args.index}:{lineno}: missing/duplicate audio_path")
            parent = row.get("parent_id")
            if not parent:
                raise SystemExit(f"{args.index}:{lineno}: missing parent_id")
            key = span_key(row["start"], row["end"], row.get("speaker"))
            index_rows.append(row)
            index_by_path[path] = row
            index_by_parent[parent][key].append(row)

    unknown = sorted(set(review) - set(index_by_path))
    if unknown:
        raise SystemExit(f"{len(unknown)} review path(s) absent from index; e.g. {unknown[:3]}")
    parents = set(index_by_parent)

    target_rows: dict[str, dict] = {}
    input_rows = 0
    input_tags: Counter = Counter()
    with open(args.target, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            input_rows += 1
            add_counts(input_tags, row)
            rid = row.get("id")
            if rid in parents:
                if rid in target_rows:
                    raise SystemExit(f"duplicate target id {rid}")
                target_rows[rid] = row

    segments: dict[str, list[dict]] = defaultdict(list)
    segment_by_key: dict[str, dict[tuple[float, float, str], dict]] = defaultdict(dict)
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            parent = row.get("parent_id")
            if parent not in parents:
                continue
            key = span_key(row["rel_start"], row["rel_end"], row.get("turn"))
            if key in segment_by_key[parent]:
                raise SystemExit(f"duplicate manifest span {parent} {key}")
            small = {k: row.get(k) for k in ("id", "rel_start", "rel_end", "turn", "txt")}
            segments[parent].append(small)
            segment_by_key[parent][key] = small

    desired: dict[str, dict[tuple[float, float, str], dict]] = defaultdict(dict)
    retained = corrected = 0
    deleted = len(index_rows) - len(review)
    collision_keys = 0
    for parent, keyed in index_by_parent.items():
        for key, items in keyed.items():
            seg = segment_by_key[parent].get(key)
            if seg is None:
                raise SystemExit(f"index span missing from manifest: {parent} {key}")
            choices: list[tuple[str, bool, str]] = []
            for item in items:
                path = item["audio_path"]
                keep = path in review
                text = review[path] if keep else (seg.get("txt") or "")
                choices.append((text, keep, path))
                if keep:
                    retained += 1
                    corrected += text != item.get("transcript", "")
            texts = {text for text, _, _ in choices}
            if len(texts) != 1:
                raise SystemExit(
                    f"conflicting decisions for {parent} {key}: "
                    f"{[(keep, path) for _, keep, path in choices]}")
            if len(items) > 1:
                collision_keys += 1
            text = choices[0][0]
            desired[parent][key] = {
                "text": text,
                "tags": tags_in(text),
                "retained": any(keep for _, keep, _ in choices),
                "paths": [path for _, _, path in choices],
            }

    need_original: set[str] = set()
    for parent, keyed in desired.items():
        row = target_rows.get(parent)
        present = ({span_key(s["start"], s["end"], s.get("speaker"))
                    for s in row.get("nv_spans", [])} if row else set())
        for key, decision in keyed.items():
            if decision["tags"] and key not in present:
                need_original.add(parent)

    original_rows: dict[str, dict] = {}
    if need_original:
        if not args.full_final:
            raise SystemExit(
                f"{len(need_original)} parent(s) need an original span re-added; pass --full-final")
        with open(args.full_final, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                rid = row.get("id")
                if rid in need_original:
                    original_rows[rid] = row
                    if len(original_rows) == len(need_original):
                        break
        missing = need_original - set(original_rows)
        if missing:
            raise SystemExit(f"parents missing from full final: {sorted(missing)[:5]}")

    changed_rows: dict[str, dict] = {}
    removed_rows: set[str] = set()
    readded_rows: dict[str, dict] = {}
    action_counts: Counter = Counter()
    tags_added: Counter = Counter()
    tags_removed: Counter = Counter()
    parents_considered = 0

    for parent in sorted(parents):
        current = target_rows.get(parent)
        if current is None and parent not in need_original:
            continue
        base = deepcopy(current if current is not None else original_rows[parent])
        parents_considered += 1
        spans = deepcopy(base.get("nv_spans") or [])
        keyed_spans: dict[tuple[float, float, str], dict] = {}
        for span in spans:
            key = span_key(span["start"], span["end"], span.get("speaker"))
            if key in keyed_spans:
                raise SystemExit(f"duplicate target span {parent} {key}")
            keyed_spans[key] = span

        original_keyed = {}
        if parent in original_rows:
            original_keyed = {
                span_key(s["start"], s["end"], s.get("speaker")): s
                for s in (original_rows[parent].get("nv_spans") or [])
            }

        for key, decision in desired[parent].items():
            old = keyed_spans.get(key)
            wanted_tags = decision["tags"]
            if wanted_tags:
                if old is None:
                    template = original_keyed.get(key)
                    if template is None:
                        raise SystemExit(f"no original span template for {parent} {key}")
                    old = deepcopy(template)
                    keyed_spans[key] = old
                    action_counts["readd_span"] += 1
                elif old.get("text") != decision["text"] or old.get("tags") != wanted_tags:
                    action_counts["replace_span"] += 1
                else:
                    action_counts["already_matching_span"] += 1
                old["text"] = decision["text"]
                old["tags"] = list(wanted_tags)
            elif old is not None:
                del keyed_spans[key]
                action_counts["remove_span"] += 1
            else:
                action_counts["already_absent_span"] += 1

        new_spans = sorted(keyed_spans.values(), key=lambda s: (
            float(s["start"]), float(s.get("end") or 0.0), str(s.get("speaker") or "")))
        aggregate(base, new_spans)

        speakers = base.get("speakers")
        if isinstance(speakers, dict):
            for speaker, info in speakers.items():
                aggregate(info, [s for s in new_spans if s.get("speaker") == speaker])

        overrides = {
            span_key(s["start"], s["end"], s.get("speaker")): s.get("text") or ""
            for s in new_spans
        }
        for key, decision in desired[parent].items():
            if decision["retained"]:
                overrides[key] = decision["text"]
        if base["n_nv"]:
            base["nv_txt"] = stitch(segments[parent], overrides)
        else:
            base.pop("nv_txt", None)

        if current is None:
            if base["n_nv"]:
                readded_rows[parent] = base
                add_counts(tags_added, base)
            continue
        if base == current:
            continue
        add_counts(tags_removed, current)
        if base["n_nv"]:
            add_counts(tags_added, base)
            changed_rows[parent] = base
        else:
            removed_rows.add(parent)

    tag_delta = Counter(tags_added)
    tag_delta.subtract(tags_removed)
    output_tags = Counter(input_tags)
    output_tags.update(tag_delta)
    output_tags += Counter()
    output_rows = input_rows - len(removed_rows) + len(readded_rows)

    report = {
        "status": "dry-run" if dry_run else "candidate_complete",
        "review_file": os.path.abspath(args.review),
        "review_sha256": review_sha,
        "target": os.path.abspath(args.target),
        "target_sha256": target_sha,
        "index": os.path.abspath(args.index),
        "manifest": os.path.abspath(args.manifest),
        "tag_parse_rule": TAG_RE.pattern,
        "baseline_items": len(index_rows),
        "retained_items": retained,
        "deleted_items": deleted,
        "corrected_items_vs_export": corrected,
        "collision_keys": collision_keys,
        "parents": len(parents),
        "parents_considered": parents_considered,
        "item_actions": dict(action_counts),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "records_updated": len(changed_rows),
        "records_removed": len(removed_rows),
        "records_readded": len(readded_rows),
        "tags_removed_from_changed_records": dict(tags_removed.most_common()),
        "tags_added_to_changed_records": dict(tags_added.most_common()),
        "tag_delta": {k: v for k, v in sorted(tag_delta.items()) if v},
        "input_tag_counts": dict(input_tags.most_common()),
        "output_tag_counts": dict(output_tags.most_common()),
        "note": ("Omitted review items restore manifest txt; retained items override nv_txt; "
                 "span tags are derived from the retained transcript."),
    }

    if not dry_run:
        out = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(out), prefix=".nv-sync-", suffix=".jsonl.tmp")
        wrote = 0
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as dst, open(
                    args.target, encoding="utf-8") as src:
                for line in src:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rid = row.get("id")
                    if rid in removed_rows:
                        continue
                    if rid in changed_rows:
                        dst.write(json.dumps(changed_rows[rid], ensure_ascii=False) + "\n")
                    else:
                        dst.write(line if line.endswith("\n") else line + "\n")
                    wrote += 1
                for rid in sorted(readded_rows):
                    dst.write(json.dumps(readded_rows[rid], ensure_ascii=False) + "\n")
                    wrote += 1
                dst.flush()
                os.fsync(dst.fileno())
            if wrote != output_rows:
                raise RuntimeError(f"wrote {wrote} rows, expected {output_rows}")
            if sha256(args.review) != review_sha:
                raise RuntimeError("review changed while candidate was built")
            if sha256(args.target) != target_sha:
                raise RuntimeError("target changed while candidate was built")
            os.replace(tmp, out)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        report["output"] = {"path": out, "sha256": sha256(out), "rows": wrote}

    atomic_json(args.report, report)
    print(json.dumps({k: report[k] for k in (
        "status", "baseline_items", "retained_items", "deleted_items",
        "corrected_items_vs_export", "item_actions", "input_rows", "output_rows",
        "records_updated", "records_removed", "records_readded", "tag_delta")},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
