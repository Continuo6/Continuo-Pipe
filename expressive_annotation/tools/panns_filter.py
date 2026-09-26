#!/usr/bin/env python3
"""Split a manifest into speech and sung/musical clips, with PANNs, before any pass runs.

nv_verify.py uses PANNs *after* the NVASR pass, to revoke tags on clips that turn out
to be sung. This tool is
the other order: score every clip first and hand the NVASR pass a manifest with the
singing already gone, so the model never sees the clips it is known to hallucinate on
(sustained vowels collapse into [Crying], melody reads as [Laughter]).

The verdict uses membership of SINGING|MUSIC in the PANNs top-k (k=3), rather
than an absolute score threshold.

Nothing is deleted: every clip is scored into panns.jsonl with its top-k as evidence,
and `split` routes whole manifest lines verbatim into speech/sung files. A clip whose
audio fails to decode gets no score row and stays in the speech manifest — the NVASR
pass will fail on it the same way and say so; silently dropping it would turn "never
looked at" into "examined".

    # score, one shard per GPU (scripts/run_nv_pipeline.sh drives this):
    python tools/panns_filter.py score --manifest m.jsonl --out work/panns00.jsonl \
        --shard 0/8 --device cuda:0 --resume
    # then split, cheap and CPU-only:
    python tools/panns_filter.py split --manifest m.jsonl --scores 'work/panns*.jsonl' \
        --out-speech manifest_speech.jsonl --out-sung manifest_sung.jsonl

Needs the nvbench env, and Cnn14's checkpoint under <ckpt-root>/ckpt (NV-Bench's repo
root has it); the tool chdirs there itself after resolving every path, because
audioldm_eval loads the checkpoint relative to the cwd.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nv_verify import MUSIC, SINGING  # noqa: E402  (the one source of the class lists)

from continuo_expressive.jsonl import JsonlWriter, done_ids, load_manifest  # noqa: E402


def load_cnn14(device: str, audioldm_root: str):
    """Cnn14 + its 527 AudioSet label names. Must be called with cwd at the ckpt root."""
    sys.path.insert(0, audioldm_root)
    import csv

    import torch

    from audioldm_eval.feature_extractors.panns.models import Cnn14

    labels_csv = os.environ.get("CONTINUO_EXPRESSIVE_PANNS_LABELS", "assets/class_labels_indices.csv")
    with open(labels_csv) as f:
        labels = [r["display_name"] for r in csv.DictReader(f)]
    if not os.path.isfile("ckpt/Cnn14_16k_mAP=0.438.pth"):
        raise SystemExit(f"no ckpt/Cnn14_16k_mAP=0.438.pth under cwd {os.getcwd()} — set --ckpt-root")
    model = Cnn14(features_list=["2048", "logits"], sample_rate=16000, window_size=512,
                  hop_size=160, mel_bins=64, fmin=50, fmax=8000, classes_num=527)
    ck = torch.load("ckpt/Cnn14_16k_mAP=0.438.pth", map_location="cpu", weights_only=False)
    missing, _ = model.load_state_dict(ck["model"], strict=False)
    real = [k for k in missing
            if not k.startswith(("spectrogram_extractor", "logmel_extractor", "spec_augmenter"))]
    if real:
        raise SystemExit(f"Cnn14 weights failed to load: {real[:5]}")
    return model.eval().to(device), labels


def cmd_score(args) -> int:
    import numpy as np
    import torch

    from continuo_expressive.tarsource import load_row

    if args.tar_dir:
        os.environ["CONTINUO_EXPRESSIVE_TAR_DIR"] = args.tar_dir

    shard = None
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/", 1))
        if not 0 <= i < n:
            raise SystemExit(f"--shard {args.shard}: need 0 <= i < n")
        shard = (i, n)
    rows = load_manifest(args.manifest, limit=args.limit, shard=shard)
    # Resolve before chdir: relative --out/--manifest must not silently move.
    out_path = os.path.abspath(args.out)
    if args.resume:
        # A score row is keyed by id and says nothing about which shard produced it, so
        # every sibling shard file is valid resume evidence. Reading only --out silently
        # re-scores completed clips when the shard count changes between runs.
        already = done_ids(args.out)
        for path in sorted(set(sum((glob.glob(p) for p in args.done_from), []))):
            if os.path.abspath(path) != out_path:
                already |= done_ids(path)
        if already:
            rows = [r for r in rows if r["id"] not in already]
            print(f"  resuming: {len(already)} done, {len(rows)} left", file=sys.stderr)
    if not rows:
        print("  nothing to do", file=sys.stderr)
        return 0

    # Keep only what the loader reads and what we write back. A manifest row
    # carries a caption paragraph, accent_top3, emotion_top3 and ~25 other fields — none
    # of which this pass touches, and holding a shard of them costs GB per worker for
    # nothing. cmd_split re-reads the original manifest, so the full rows survive there.
    keep = ("id", "source_tar", "source_member", "tar_offset", "tar_size",
            "rel_start", "rel_end", "wav_path", "_path", "carrier_start_samples",
            "carrier_end_samples", "sample_rate")
    rows = [{k: r[k] for k in keep if k in r} for r in rows]

    os.chdir(os.path.expanduser(args.ckpt_root))
    model, labels = load_cnn14(args.device, args.audioldm_root)
    print(f"  loaded Cnn14; {len(rows)} clip(s) to score", file=sys.stderr)

    def decode(r):
        try:
            return load_row(r)
        except Exception as e:
            print(f"  [warn] {r['id']}: {type(e).__name__}: {e}", file=sys.stderr)
            return None

    sung_labels = SINGING | MUSIC
    n_done = n_sung = 0
    pool = ThreadPoolExecutor(max_workers=args.workers)

    def decode_chunk(chunk):
        wavs, kept = [], []
        for r, w in zip(chunk, pool.map(decode, chunk)):
            if w is not None:
                wavs.append(w)
                kept.append(r)
        return wavs, kept

    # Decode the next chunk while the GPU works on this one. Without the overlap the
    # two alternate and leave either the decoder or GPU idle.
    ahead = ThreadPoolExecutor(max_workers=1)
    chunks = [rows[i:i + 64] for i in range(0, len(rows), 64)]
    try:
        with JsonlWriter(out_path, append=args.resume) as sink:
            pending = ahead.submit(decode_chunk, chunks[0])
            for ci in range(len(chunks)):
                wavs, kept = pending.result()
                if ci + 1 < len(chunks):
                    pending = ahead.submit(decode_chunk, chunks[ci + 1])
                for b in range(0, len(wavs), args.batch_size):
                    sub, subrows = wavs[b:b + args.batch_size], kept[b:b + args.batch_size]
                    # belt and braces: load_row now raises on an empty slice, but a batch
                    # padded to a longest length of 0 is a crash inside the model rather
                    # than a skipped clip, and that is too sharp an edge to leave
                    pairs = [(w, r) for w, r in zip(sub, subrows) if len(w)]
                    if len(pairs) != len(sub):
                        print(f"  [warn] dropped {len(sub) - len(pairs)} empty clip(s)",
                              file=sys.stderr)
                    if not pairs:
                        continue
                    sub = [w for w, _ in pairs]
                    subrows = [r for _, r in pairs]
                    n = max(len(w) for w in sub)
                    batch = np.zeros((len(sub), n), dtype="float32")
                    for j, w in enumerate(sub):
                        batch[j, :len(w)] = w
                    with torch.no_grad():
                        p = model(torch.from_numpy(batch).to(args.device))["clipwise_output"].cpu().numpy()
                    for r, scores in zip(subrows, p):
                        idx = scores.argsort()[-args.topk:][::-1]
                        top = [[labels[k], round(float(scores[k]), 4)] for k in idx]
                        sung = any(t[0] in sung_labels for t in top)
                        n_sung += sung
                        sink.write({"id": r["id"], "sung": bool(sung), "top": top})
                        n_done += 1
                if ci % 20 == 0:
                    print(f"  panns: {n_done}/{len(rows)} ({n_sung} sung)", file=sys.stderr, flush=True)
    finally:
        ahead.shutdown(wait=False)
        pool.shutdown(wait=False)
    print(f"  scored {n_done}/{len(rows)}; {n_sung} sung/musical", file=sys.stderr)
    return 0


def cmd_offsets(args) -> int:
    """Bake each row's tar_offset/tar_size in, optionally grouping rows by tar.

    Without the offsets every clip calls tarsource.load_index, whose cache is per
    process and per tar. Baking offsets once avoids repeated index loading.

    --sort-by-tar additionally groups the rows, which is what makes the loader's capped
    handle cache hit instead of reopening. Score in that order and split the original,
    since the verdicts are keyed by id and the next pass usually wants duration order.
    """
    from pathlib import Path

    from continuo_expressive.tarsource import _INDEX, load_index, resolve_tar, tar_dir

    root = Path(args.tar_dir).expanduser() if args.tar_dir else tar_dir()
    rows = [json.loads(line) for line in open(args.manifest)]
    print(f"  {len(rows)} row(s) from {args.manifest}", file=sys.stderr)

    # Group by tar first either way: keep one index in memory at a time.
    order = sorted(range(len(rows)), key=lambda i: rows[i].get("source_tar") or "")
    cur, idx, missing, had = None, {}, 0, 0
    for i in order:
        r = rows[i]
        tar = r.get("source_tar")
        if not tar or not r.get("source_member"):
            continue
        if r.get("tar_offset") is not None and r.get("tar_size") is not None:
            had += 1          # a long-audio segment manifest already carries these
            continue
        if tar != cur:
            _INDEX.clear()
            idx = load_index(resolve_tar(tar, root))
            cur = tar
        entry = idx.get(r["source_member"])
        if entry is None:
            missing += 1
        else:
            r["tar_offset"], r["tar_size"] = entry

    if args.sort_by_tar:
        # rel_start last so a long container's segments stay in time order inside the
        # group — they all share one tar_offset, and ordering them is what lets the
        # loader's capped handle cache hit for a whole container at a time.
        rows.sort(key=lambda r: (r.get("source_tar") or "", r.get("tar_offset") or 0,
                                 r.get("rel_start") or 0))
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows)} row(s)"
          + (f", {had} already had offsets" if had else "")
          + (f", {missing} without an index entry" if missing else "")
          + (" (grouped by tar)" if args.sort_by_tar else "")
          + f" -> {args.out}", file=sys.stderr)
    return 0


def cmd_split(args) -> int:
    verdict: dict[str, bool] = {}
    unparseable = 0
    for path in sorted(sum((glob.glob(p) for p in args.scores), [])):
        # Deliberately not read_jsonl: a worker killed mid-append leaves a run of NUL
        # bytes where a record was, and --resume already tolerates exactly this (see
        # jsonl.done_ids). Being strict here throws an hour of scoring away over one
        # hole — but it is reported, and a lost verdict resurfaces below as "unscored",
        # which routes the clip to speech rather than dropping it.
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    unparseable += 1
                    continue
                verdict[r["id"]] = r["sung"]
    if unparseable:
        print(f"  [warn] {unparseable} unparseable score line(s) skipped",
              file=sys.stderr)
    if not verdict:
        raise SystemExit(f"no score rows under {args.scores}")

    # Long audio only: a recording's segments are not independent the way short clips
    # are. If the recording *is* a song, scoring each segment alone leaks the ones PANNs
    # happens to miss, and those are exactly the clips the filter exists to keep out of
    # the ASR pass. So a recording whose sung fraction clears --sung-parent-frac is
    # routed out whole.
    #
    # A recording with a brief jingle or music bed should not be discarded whole.
    promoted: set[str] = set()
    if args.sung_parent_frac is not None:
        per_parent: dict[str, list[int]] = {}
        with open(args.manifest) as f:
            for line in f:
                row = json.loads(line)
                parent, v = row.get("parent_id"), verdict.get(row.get("id"))
                if parent is None or v is None:
                    continue
                c = per_parent.setdefault(parent, [0, 0])
                c[0] += bool(v)
                c[1] += 1
        promoted = {p for p, (s, t) in per_parent.items()
                    if t and s and s / t >= args.sung_parent_frac}
        if promoted:
            extra = sum(per_parent[p][1] - per_parent[p][0] for p in promoted)
            print(f"  {len(promoted)} recording(s) are >={args.sung_parent_frac:.0%} sung; "
                  f"routing out {extra} more segment(s) with them", file=sys.stderr)

    stats = Counter()
    with open(args.manifest) as f, \
         open(args.out_speech, "w") as speech, \
         open(args.out_sung, "w") as sung:
        for line in f:
            row = json.loads(line)
            v = verdict.get(row.get("id"))
            if v is None:
                stats["unscored"] += 1  # decode failed at score time; NVASR will say so too
            if not v and row.get("parent_id") in promoted:
                v = True
                stats["by_parent"] += 1
            (sung if v else speech).write(line)
            stats["sung" if v else "speech"] += 1
    total = stats["speech"] + stats["sung"]
    print(f"split {total} row(s): {stats['speech']} speech -> {args.out_speech}, "
          f"{stats['sung']} sung -> {args.out_sung}"
          + (f" ({stats['by_parent']} of them by their recording)" if stats["by_parent"] else "")
          + (f" ({stats['unscored']} unscored, kept in speech)" if stats["unscored"] else ""))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("score", help="PANNs top-k + sung verdict for every manifest row")
    sc.add_argument("--manifest", required=True)
    sc.add_argument("--out", required=True, help="scores JSONL (one row per clip)")
    sc.add_argument("--shard", default="", help="i/n by manifest line number")
    sc.add_argument("--resume", action="store_true")
    sc.add_argument("--done-from", nargs="*", default=[], metavar="GLOB",
                    help="extra score files whose ids also count as done under --resume; "
                         "pass the sibling shards so changing n does not re-score")
    sc.add_argument("--limit", type=int, default=0)
    sc.add_argument("--device", default="cuda:0")
    sc.add_argument("--batch-size", type=int, default=1,
                    help="clips per forward. 1 avoids padding-dependent sung verdicts")
    sc.add_argument("--workers", type=int, default=16, help="audio decode threads")
    sc.add_argument("--topk", type=int, default=3)
    sc.add_argument("--tar-dir", default="", help="corpus dir for source_tar rows (CONTINUO_EXPRESSIVE_TAR_DIR)")
    sc.add_argument("--audioldm-root", default="third_party/audioldm_eval")
    sc.add_argument("--ckpt-root", default="third_party/NV-Bench",
                    help="directory whose ./ckpt holds Cnn14_16k_mAP=0.438.pth")
    sc.set_defaults(fn=cmd_score)

    sp = sub.add_parser("split", help="route manifest lines by the score files' verdicts")
    sp.add_argument("--manifest", required=True)
    sp.add_argument("--scores", nargs="+", required=True, help="score JSONL path(s) or glob(s)")
    sp.add_argument("--out-speech", required=True)
    sp.add_argument("--out-sung", required=True)
    sp.add_argument("--sung-parent-frac", type=float, default=None, metavar="F",
                    help="long audio: route out every segment of a recording whose "
                         "segments are at least this fraction sung, so a song does not "
                         "leak the segments PANNs missed. Needs parent_id on the rows; "
                         "unset means judge each segment alone")
    sp.set_defaults(fn=cmd_split)

    op = sub.add_parser("offsets",
                        help="copy a manifest with tar_offset/tar_size baked in")
    op.add_argument("--manifest", required=True)
    op.add_argument("--out", required=True)
    op.add_argument("--sort-by-tar", action="store_true",
                    help="also group rows by (tar, offset) for sequential reads")
    op.add_argument("--tar-dir", default="", help="corpus dir (CONTINUO_EXPRESSIVE_TAR_DIR)")
    op.set_defaults(fn=cmd_offsets)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
