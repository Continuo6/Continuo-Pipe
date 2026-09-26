#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


ALLOWED_TAGS = ("Laughter", "Cough", "Crying", "Sigh", "Breathing")
TAG_RE = re.compile(r"\[(?:Laughter|Cough|Crying|Sigh|Breathing)\]")
SPEAKER_RE = re.compile(r"\[(S\d+)\]")
BOUNDARY_RE = re.compile(r"(?<=[.!?\u3002\uff01\uff1f\uff1b;\uff0c,])\s*")

# Unicode escapes preserve CJK keyword matching while keeping source text English.
KEYWORDS = {
    "Laughter": (
        "haha", "laugh", "laughing", "funny", "joke", "hilarious", "amusing",
        "\u54c8\u54c8", "\u7b11", "\u597d\u7b11", "\u73a9\u7b11", "\u6ed1\u7a3d", "\u606d\u559c",
    ),
    "Crying": (
        "cry", "crying", "tears", "funeral", "murder", "killed", "bereaved",
        "heartbroken", "grief", "tragic", "tragedy", "sobbing", "\u54ed", "\u6cea", "\u75c5\u901d",
        "\u53bb\u4e16", "\u8eab\u4ea1", "\u60b2\u75db", "\u60b2\u4f24", "\u4e27\u751f", "\u9047\u96be", "\u8bc0\u522b", "\u6078\u54ed", "\u5fc3\u788e",
    ),
    "Sigh": (
        "sigh", "sorry", "regret", "difficult", "worry", "worried",
        "uncertain", "unfortunately", "disaster", "failed", "failure", "wrong", "struggle",
        "don.t know", "do not know", "\u62b1\u6b49", "\u9057\u61be",
        "\u4e0d\u77e5\u9053", "\u4e0d\u5bb9\u6613", "\u56f0\u96be", "\u62c5\u5fc3", "\u65e0\u5948", "\u53ef\u6015", "\u7cdf\u7cd5", "\u5931\u8d25", "\u95ee\u9898",
        "\u4e0d\u786e\u5b9a", "\u6ca1\u529e\u6cd5", "\u75db\u82e6", "\u96be\u8fc7",
    ),
}


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def target_times(duration, interval):
    targets = [round(interval * index, 3) for index in range(1, math.floor(duration / interval) + 1)]
    if targets and duration - targets[-1] < 10:
        lower_bound = targets[-2] + interval / 2 if len(targets) > 1 else interval / 2
        targets[-1] = round(max(lower_bound, duration - 10), 3)
    return targets


def source_utterances(manifest_rows, source_text):
    nonempty = [row for row in manifest_rows if row.get("txt")]
    utterances = []
    search_from = 0
    for row in nonempty:
        text = row["txt"]
        char_start = source_text.find(text, search_from)
        if char_start < 0:
            char_start = source_text.find(text)
        if char_start < 0:
            raise ValueError(f"Cannot locate segment text for {row['parent_id']}: {text[:80]!r}")
        search_from = char_start + len(text)
        duration = float(row.get("utt_seconds") or row.get("duration") or 0)
        start = float(row["rel_start"])
        end = min(float(row["parent_duration"]), start + duration)
        utterances.append({
            "start": start,
            "end": end,
            "text": text,
            "char_start": char_start,
            "char_end": char_start + len(text),
            "speaker": row.get("speaker") or row.get("turn"),
            "segment_id": row["id"],
        })
    return utterances


def choose_utterance(utterances, target):
    containing = [item for item in utterances if item["start"] <= target <= item["end"]]
    if containing:
        return min(containing, key=lambda item: item["end"] - item["start"])
    return min(
        utterances,
        key=lambda item: min(abs(target - item["start"]), abs(target - item["end"])),
    )


def boundary_positions(text):
    positions = {0, len(text)}
    positions.update(match.end() for match in BOUNDARY_RE.finditer(text))
    return sorted(positions)


def choose_position(utterance, target, occupied):
    span = max(utterance["end"] - utterance["start"], 0.001)
    ratio = min(1.0, max(0.0, (target - utterance["start"]) / span))
    ideal_local = round(ratio * len(utterance["text"]))
    candidates = boundary_positions(utterance["text"])
    candidates.sort(key=lambda value: (abs(value - ideal_local), value))
    for local_position in candidates:
        global_position = utterance["char_start"] + local_position
        if global_position not in occupied:
            inferred_time = utterance["start"] + span * local_position / max(len(utterance["text"]), 1)
            return global_position, inferred_time
    raise ValueError(f"No distinct insertion boundary in {utterance['segment_id']}")


def contains_keyword(text, keyword):
    if keyword.isascii() and keyword.replace("'", "").replace(" ", "").isalpha():
        return re.search(rf"(?<![A-Za-z]){re.escape(keyword)}(?![A-Za-z])", text, re.IGNORECASE) is not None
    return keyword.lower() in text.lower()


def choose_tag(context, record_id, event_index):
    scores = {
        tag: sum(contains_keyword(context, keyword) for keyword in keywords)
        for tag, keywords in KEYWORDS.items()
    }
    best = max(("Laughter", "Crying", "Sigh"), key=lambda tag: (scores[tag], -ALLOWED_TAGS.index(tag)))
    if scores[best] == 0:
        stable_key = f"{record_id}:{event_index}".encode()
        bucket = int.from_bytes(hashlib.sha256(stable_key).digest()[:4], "big") % 8
        if bucket in (0, 1, 2, 5):
            return "Breathing", "Neutral local context; add a breath at a natural pause."
        if bucket == 3:
            return "Laughter", "Light local context; add a soft laugh at a natural pause."
        if bucket == 4:
            return "Sigh", "Restrained local context; add a sigh at a natural pause."
        return "Cough", "Neutral local context; add a light cough at a natural pause."
    reasons = {
        "Laughter": "Light or humorous local context; add laughter at a pause.",
        "Crying": "Sad or distressed local context; add crying at a pause.",
        "Sigh": "Worried or resigned local context; add a sigh at a pause.",
    }
    return best, reasons[best]


def context_window(text, position, radius=80):
    return text[max(0, position - radius):min(len(text), position + radius)]


def speaker_at(text, position, fallback):
    matches = list(SPEAKER_RE.finditer(text, 0, position + 1))
    return matches[-1].group(1) if matches else fallback


def build_row(source_row, manifest_rows, interval):
    source_text = source_row["txt"]
    utterances = source_utterances(manifest_rows, source_text)
    occupied = set()
    insertions = []
    for event_index, target in enumerate(target_times(float(source_row["file_seconds"]), interval), 1):
        utterance = choose_utterance(utterances, target)
        position, inferred_time = choose_position(utterance, target, occupied)
        occupied.add(position)
        context = context_window(source_text, position)
        tag, reason = choose_tag(context, source_row["id"], event_index)
        speaker = speaker_at(source_text, position, utterance["speaker"])
        insertions.append({
            "event_index": event_index,
            "tag": tag,
            "source_char_index": position,
            "target_time": target,
            "inferred_text_time": round(inferred_time, 3),
            "time_error": round(abs(inferred_time - target), 3),
            "segment_id": utterance["segment_id"],
            "speaker": speaker,
            "anchor": context.strip(),
            "placement": "boundary",
            "reason": reason,
        })

    nv_text = source_text
    for insertion in sorted(insertions, key=lambda item: item["source_char_index"], reverse=True):
        position = insertion["source_char_index"]
        nv_text = nv_text[:position] + f"[{insertion['tag']}]" + nv_text[position:]

    result = dict(source_row)
    counts = Counter(item["tag"] for item in insertions)
    result.update({
        "nv_txt": nv_text,
        "nv_tags": [tag for tag in ALLOWED_TAGS if counts[tag]],
        "nv_counts": {tag: counts[tag] for tag in ALLOWED_TAGS if counts[tag]},
        "n_nv": len(insertions),
        "nv_source": "contextual_synthetic_30s",
        "nv_interval_seconds": interval,
        "nv_insertions": insertions,
    })

    speaker_data = result.get("speakers")
    if isinstance(speaker_data, (dict, list)):
        by_speaker = defaultdict(list)
        for insertion in insertions:
            by_speaker[insertion["speaker"]].append(insertion)
        if isinstance(speaker_data, dict):
            result["speakers"] = {name: dict(row) for name, row in speaker_data.items()}
            speaker_rows = result["speakers"].values()
        else:
            result["speakers"] = [dict(row) for row in speaker_data]
            speaker_rows = result["speakers"]
        for speaker_row in speaker_rows:
            speaker_insertions = by_speaker.get(speaker_row["speaker"], [])
            speaker_counts = Counter(item["tag"] for item in speaker_insertions)
            speaker_row.update({
                "nv_tags": [tag for tag in ALLOWED_TAGS if speaker_counts[tag]],
                "nv_counts": {tag: speaker_counts[tag] for tag in ALLOWED_TAGS if speaker_counts[tag]},
                "n_nv": len(speaker_insertions),
                "nv_source": "contextual_synthetic_30s",
                "nv_insertions": speaker_insertions,
            })
    return result


def validate(source_rows, output_rows, interval):
    if len(source_rows) != len(output_rows):
        raise ValueError("Row count changed")
    aggregate = Counter()
    errors = []
    for source, output in zip(source_rows, output_rows):
        if source["id"] != output["id"] or source["txt"] != output["txt"]:
            raise ValueError(f"Source identity/text changed for {source['id']}")
        if source.get("caption") != output.get("caption"):
            raise ValueError(f"Caption changed for {source['id']}")
        if TAG_RE.sub("", output["nv_txt"]) != source["txt"]:
            raise ValueError(f"NV text is not losslessly reversible for {source['id']}")
        expected = len(target_times(float(source["file_seconds"]), interval))
        found = TAG_RE.findall(output["nv_txt"])
        if output["n_nv"] != expected or len(found) != expected:
            raise ValueError(f"Unexpected tag count for {source['id']}: {len(found)} != {expected}")
        counts = Counter(tag[1:-1] for tag in found)
        if dict(counts) != output["nv_counts"]:
            raise ValueError(f"Count mismatch for {source['id']}")
        aggregate.update(counts)
        errors.extend(item["time_error"] for item in output["nv_insertions"])
        speaker_data = output.get("speakers")
        if isinstance(speaker_data, dict):
            speaker_rows = speaker_data.values()
        elif isinstance(speaker_data, list):
            speaker_rows = speaker_data
        else:
            speaker_rows = []
        if speaker_rows and sum(item["n_nv"] for item in speaker_rows) != output["n_nv"]:
            raise ValueError(f"Speaker NV counts mismatch for {source['id']}")
    return aggregate, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--long-manifest", type=Path, required=True)
    parser.add_argument("--dialogue-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()

    source_rows = read_jsonl(args.input)
    manifest_by_parent = defaultdict(list)
    for manifest_path in (args.long_manifest, args.dialogue_manifest):
        for row in read_jsonl(manifest_path):
            manifest_by_parent[row["parent_id"]].append(row)
    for rows in manifest_by_parent.values():
        rows.sort(key=lambda row: (row["rel_start"], row["id"]))

    output_rows = [build_row(row, manifest_by_parent[row["id"]], args.interval) for row in source_rows]
    counts, errors = validate(source_rows, output_rows, args.interval)
    write_jsonl(args.output, output_rows)

    plan_rows = []
    for row in output_rows:
        for insertion in row["nv_insertions"]:
            plan_rows.append({"id": row["id"], "file_seconds": row["file_seconds"], **insertion})
    write_jsonl(args.plan_output, plan_rows)

    report = {
        "source": str(args.input),
        "output": str(args.output),
        "records": len(output_rows),
        "interval_seconds": args.interval,
        "total_insertions": sum(counts.values()),
        "tag_counts": {tag: counts[tag] for tag in ALLOWED_TAGS},
        "mean_boundary_time_error": round(sum(errors) / len(errors), 3) if errors else 0,
        "max_boundary_time_error": round(max(errors), 3) if errors else 0,
        "validation": {
            "source_text_preserved": True,
            "caption_preserved": True,
            "only_allowed_nv_tags": True,
            "dialogue_speaker_counts_match": True,
        },
    }
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
