#!/usr/bin/env python3
"""Fold NV segment annotations back into one record per recording.

The NV passes work per segment, because that is the unit a sung filter and an utterance
ASR can be honest about. But a long corpus's record is the recording — runs/long-all/
final.jsonl is one row per file — so the segments have to be folded back before the
result can sit beside the tags, emotion and caption already there. Same shape the rest
of the corpus uses: counts for the whole recording, plus a span list carrying the
timeline, like ``emotion_spans`` and ``speed_spans``.

    python tools/merge_nv.py --records runs/long-all/final.jsonl \\
        --nv runs/nv-long/nv-verified.jsonl --sung runs/nv-long/manifest_sung.jsonl \\
        --out runs/nv-long/final.jsonl

Short clips need no folding — a clip *is* a record — so with ``--by id`` the NV fields
are joined straight onto the matching row instead.

**`--segments` gives back the text.** With the segment manifest the fold also writes
``nv_txt``: the recording's transcript rebuilt with the tags where they were said, and
`[S1]`/`[S2]` marked at every change of speaker for a dialogue — which is the form a
training set wants, rather than a span list to cross-reference. ``txt`` is left alone.

**A dialogue is folded twice**: once into the file (the totals and the timeline, each
span naming the speaker it came from) and once into each voice, because the corpus
record nests its speakers and a laugh belongs to whoever laughed. That needs the NV rows
to carry ``turn``; a record whose rows do not is reported rather than left with empty
per-speaker fields.

**Recordings with no NV are still written**, with ``n_nv: 0`` and an empty span list.
When you are sampling a training set, "examined, had none" and "never looked at" are
different facts, and only ``nv_segments`` tells them apart: 0 there means the recording
was never annotated (no segments survived the scope, or the pass has not run).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from continuo_expressive.jsonl import JsonlWriter  # noqa: E402


#: Keep only fields needed by the merged record to bound memory use.
_KEEP = ("id", "rel_start", "rel_end", "nv_verified", "nv_tags", "nv_text", "text",
         "nv_rejected", "asr_ratio", "turn", "speaker")

#: sentinel for --exclude-ids: everything this row claims is dropped
class _All(frozenset):
    def __contains__(self, _item) -> bool:
        return True
    def __bool__(self) -> bool:
        return True


ALL_TAGS = _All()


def load_nv(paths: list[str], key: str) -> dict[str, list[dict]]:
    """group key -> its NV rows, sorted by position in the recording."""
    out: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue          # a killed worker can leave a NUL hole mid-file
                k = r.get(key)
                if k:
                    out[k].append({f: r[f] for f in _KEEP if f in r})
    for rows in out.values():
        rows.sort(key=lambda r: (r.get("rel_start") or 0, r.get("rel_end") or 0))
    return out


def spans_for(rows: list[dict], verified_only: bool,
              drop: frozenset = frozenset()) -> tuple[list[dict], Counter, list[str], Counter]:
    """The segments that carry a tag, as a timeline, the tag counts, and the tagged
    transcripts — which are the annotation itself and must reach the record whether or
    not the rows have a position inside a recording to report."""
    spans, counts, texts, dropped = [], Counter(), [], Counter()
    for r in rows:
        tags = r.get("nv_verified") if verified_only and "nv_verified" in r else r.get("nv_tags")
        tags = tags or []
        if drop:
            for t in tags:
                if t in drop:
                    dropped[t] += 1
            tags = [t for t in tags if t not in drop]
        if not tags:
            continue
        counts.update(tags)
        if r.get("nv_text"):
            texts.append(r["nv_text"])
        if r.get("rel_start") is None:
            # A short clip is its own span: there is no position to report inside it, and
            # a span of null..null is worse than none — it reads as a timeline that has
            # been measured. nv_counts already says what the clip carried.
            continue
        span = {"start": r["rel_start"], "end": r.get("rel_end"),
                "tags": list(tags), "text": r.get("nv_text") or r.get("text") or ""}
        # a dialogue's segments are turns, so the span can say whose voice it was —
        # without this a laugh in a six-person conversation belongs to nobody
        if r.get("turn"):
            span["speaker"] = r["turn"]
        rejected = r.get("nv_rejected")
        if rejected:
            span["rejected"] = rejected
        if r.get("asr_ratio") is not None:
            span["asr_ratio"] = r["asr_ratio"]
        spans.append(span)
    return spans, counts, texts, dropped


def stitch(segments: list[tuple], tagged: dict[str, str], mark: bool) -> str:
    """The recording's transcript with the NV tags in place.

    A span list says a laugh happened at 41.2 s; what a training set wants is the line it
    happened in. So the transcript is rebuilt from the segments in time order — the same
    way ``aggregate_long`` builds ``txt``, `[S1]`/`[S2]` at every change of speaker — and
    a turn that carried a tag contributes **the tagged transcript instead of the corpus
    one**. That swap is deliberate and worth knowing about: 80% of tags land mid-turn
    (493 of 618 measured), so putting them at a turn boundary would misplace most of
    them, and only the NVASR transcript says where inside the turn the sound was. It has
    no punctuation or casing, so a tagged turn reads differently from its neighbours.
    """
    parts, last = [], None
    for _, turn, txt, sid in sorted(segments):
        body = tagged.get(sid) or txt or ""
        if not body:
            continue
        if mark and turn != last:
            parts.append(f"[{turn}]" if turn else "")
            last = turn
        parts.append(body)
    return " ".join(x for x in parts if x)


def by_speaker(rows: list[dict], tags: list) -> dict[str, list[dict]]:
    """The segments of a dialogue, split by the turn tag that says who spoke them."""
    out: dict[str, list[dict]] = {t: [] for t in tags}
    for r in rows:
        t = r.get("turn")
        if t in out:
            out[t].append(r)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--records", required=True,
                   help="the corpus rows to enrich (one per recording, or per clip)")
    p.add_argument("--nv", nargs="+", required=True, help="nv-verified.jsonl path(s)")
    p.add_argument("--segments", default="",
                   help="the segment manifest the NV pass ran on. With it every record "
                        "that carries a tag also gets `nv_txt`: its transcript rebuilt "
                        "with the tags in place, `[S1]`/`[S2]` marked for a dialogue. "
                        "Without it only the span timeline is written.")
    p.add_argument("--sung", nargs="*", default=[],
                   help="manifest_sung.jsonl: segments the filter routed out, counted "
                        "so a recording says how much of it was not annotated")
    p.add_argument("--out", required=True)
    p.add_argument("--out-tagged", default="",
                   help="also write just the records that carry an NV event. The full "
                        "file stays the corpus record; this is the subset worth "
                        "training on, and the two answer different questions")
    p.add_argument("--by", default="parent_id", choices=("parent_id", "id"),
                   help="parent_id folds segments into their recording (long audio); "
                        "id joins a clip's own row (short audio)")
    p.add_argument("--exclude-ids", default="",
                   help="file of clip/recording ids, one per line (# comments allowed), "
                        "whose NV tags were judged wrong by ear. Their tags are dropped "
                        "and counted in nv_dropped_tags, so a rerun keeps the judgement "
                        "instead of resurrecting it")
    p.add_argument("--drop-tags", default="",
                   help="comma-separated tag names to discard everywhere, for a class "
                        "that listening showed is not trustworthy. They are removed from "
                        "nv_tags/nv_counts/n_nv and from every span, and counted in "
                        "nv_dropped_tags so the record still says what was thrown away")
    p.add_argument("--all-tags", action="store_true",
                   help="count the tags NVASR produced rather than the ones that "
                        "survived verification")
    args = p.parse_args(argv)

    excluded: set[str] = set()
    if args.exclude_ids:
        for line in open(args.exclude_ids, encoding="utf-8"):
            line = line.split("#", 1)[0].strip()
            if line:
                excluded.add(line)
        print(f"  excluding {len(excluded)} id(s) listed in {args.exclude_ids}",
              file=sys.stderr)
    drop = frozenset(t.strip() for t in args.drop_tags.split(",") if t.strip())
    if drop:
        print(f"  dropping tag(s) {sorted(drop)} everywhere", file=sys.stderr)
    nv = load_nv(args.nv, args.by)
    # Only recordings with a surviving tag need their transcript rebuilt — the rest
    # would get a copy of `txt` under another name, at the price of holding the whole
    # corpus's segment text in memory.
    skeleton: dict[str, list[tuple]] = defaultdict(list)
    if args.segments:
        want_txt = {k for k, rows in nv.items()
                    if any(r.get("nv_verified") or r.get("nv_tags") for r in rows)}
        with open(args.segments, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = r.get(args.by)
                if k in want_txt:
                    skeleton[k].append((r.get("rel_start") or 0.0, r.get("turn"),
                                        r.get("txt") or "", r.get("id")))
        print(f"  transcripts: {len(skeleton)} recording(s) with a tag will be rebuilt "
              f"from {args.segments}", file=sys.stderr)
    sung: dict[str, int] = Counter()
    for path in args.sung:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    k = json.loads(line).get(args.by)
                    if k:
                        sung[k] += 1

    n = enriched = n_tagged = untagged_records = 0
    stray_turns: set = set()
    total_tags: Counter = Counter()
    with contextlib.ExitStack() as stack:
        f = stack.enter_context(open(args.records, encoding="utf-8"))
        out = stack.enter_context(JsonlWriter(args.out))
        tagged_out = (stack.enter_context(JsonlWriter(args.out_tagged))
                      if args.out_tagged else None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n += 1
            rows = nv.get(rec.get("id"), [])
            # An id on the exclusion list had its tags judged wrong by ear. Drop every
            # tag on it rather than the whole class: on the sample that produced this
            # list, 7 of 10 Sneeze clips were wrong and 3 were right.
            row_drop = ALL_TAGS if rec.get("id") in excluded else drop
            spans, counts, texts, dropped = spans_for(
                rows, verified_only=not args.all_tags, drop=row_drop)
            total_tags.update(counts)
            rec["nv_tags"] = sorted(counts)
            rec["nv_counts"] = dict(counts.most_common())
            rec["n_nv"] = sum(counts.values())
            rec["nv_spans"] = spans
            # The transcript with its tags inline is the annotation. A long recording
            # carries one per span; a short clip is a single segment, so it goes on the
            # record itself — dropping it there left rows saying `nv_tags: ["Uhm"]` with
            # no sign of where the Uhm was said.
            if texts:
                rec["nv_text"] = texts[0] if len(texts) == 1 else texts
            if dropped:
                rec["nv_dropped_tags"] = dict(dropped.most_common())
            rec["nv_segments"] = len(rows)
            rec["nv_sung_segments"] = sung.get(rec.get("id"), 0)
            segs = skeleton.get(rec.get("id"))
            if segs and rec["n_nv"]:
                tagged_text = {}
                for r in rows:
                    keep = r.get("nv_verified") if not args.all_tags and "nv_verified" in r \
                        else r.get("nv_tags")
                    keep = [t for t in (keep or []) if t not in row_drop]
                    if keep and r.get("nv_text") and r.get("id"):
                        tagged_text[r["id"]] = r["nv_text"]
                if tagged_text:
                    rec["nv_txt"] = stitch(segs, tagged_text,
                                           mark=bool(rec.get("speakers")))
            # A dialogue is several voices and the corpus record nests them, so the NV
            # goes on the voice as well as on the file: a laugh is an attribute of whoever
            # laughed. The dialogue-level fields above stay — they are the file's totals.
            speakers = rec.get("speakers")
            if isinstance(speakers, dict) and speakers:
                if rows and not any(r.get("turn") for r in rows):
                    untagged_records += 1
                else:
                    split = by_speaker(rows, list(speakers))
                    stray = {r.get("turn") for r in rows} - set(speakers) - {None}
                    if stray:
                        stray_turns.update(stray)
                    for tag, spk in speakers.items():
                        srows = split.get(tag, [])
                        sspans, scounts, stexts, _ = spans_for(
                            srows, verified_only=not args.all_tags, drop=row_drop)
                        spk["nv_tags"] = sorted(scounts)
                        spk["nv_counts"] = dict(scounts.most_common())
                        spk["n_nv"] = sum(scounts.values())
                        spk["nv_spans"] = sspans
                        spk["nv_segments"] = len(srows)
                        if stexts:
                            spk["nv_text"] = stexts[0] if len(stexts) == 1 else stexts
            if rows:
                enriched += 1
            out.write(rec)
            if tagged_out is not None and rec["n_nv"]:
                tagged_out.write(rec)
                n_tagged += 1

    if args.out_tagged:
        print(f"wrote {n_tagged} record(s) with an NV event -> {args.out_tagged}\n"
              f"  the {n - n_tagged} without one are still in {args.out}: most were "
              f"annotated and clean, which is a usable negative, and the rest were never "
              f"annotated at all (nv_segments 0). Those two are not the same thing.",
              file=sys.stderr)

    if untagged_records:
        print(f"  [warn] {untagged_records} record(s) nest speakers but their NV segments "
              f"carry no `turn`, so the tags could not be attributed to a voice — the "
              f"annotate pass needs --carry ...,speaker,turn (VIEW=dialogue sets it)",
              file=sys.stderr)
    if stray_turns:
        print(f"  [warn] turn tag(s) {sorted(stray_turns)[:8]} appear on NV segments but "
              f"on no record's speaker list; their tags stay in the file's totals only",
              file=sys.stderr)
    covered = f"{enriched}/{n} record(s) carry NV segments"
    print(f"wrote {n} record(s) -> {args.out}\n  {covered}; "
          f"{sum(total_tags.values())} tag(s) over {len(total_tags)} kind(s)",
          file=sys.stderr)
    if enriched < n:
        print(f"  [warn] {n - enriched} record(s) have nv_segments 0 — never annotated, "
              f"not 'annotated and clean'", file=sys.stderr)
    for tag, c in total_tags.most_common(10):
        print(f"    {tag:<20} {c}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
