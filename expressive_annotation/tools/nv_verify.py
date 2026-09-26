#!/usr/bin/env python3
"""Verify NV tags using transcript consistency and public PANNs scores.

Rows retain their original tags and gain ``nv_verified`` and ``nv_rejected``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from continuo_expressive.cli.nv import asr_ratio  # noqa: E402
from continuo_expressive.jsonl import JsonlWriter, read_jsonl  # noqa: E402

# AudioSet classes that mean "this clip is sung or musical". Kept wide on purpose: a
# clip whose top-3 holds Lullaby or Mantra is not someone talking, whatever else it is.
SINGING = {
    "Singing", "Male singing", "Female singing", "Child singing", "Choir", "Yodeling",
    "Humming", "Synthetic singing", "Rapping", "Chant", "Mantra", "A capella",
    "Vocal music", "Lullaby",
}
MUSIC = {
    "Music", "Musical instrument", "Song", "Background music", "Theme music",
    "Singing bowl", "Pop music", "Rock music", "Classical music", "Electronic music",
    "Musical ensemble", "Soundtrack music",
}

# Tags whose credibility a sung clip destroys. Singing produces laughter-like and
# cry-like acoustics, so these are re-checked; a cough or a hesitation in a song is
# still a cough or a hesitation.
SINGING_SENSITIVE = {"Crying", "Laughter", "Sigh", "Surprise-ah", "Surprise-oh"}


def panns_scores(rows: list[dict], device: str, audioldm_root: str, topk: int,
                 workers: int = 16) -> dict[str, set]:
    """Clip id -> its top-k AudioSet class names.

    Imported lazily and only when the transcript check left something to check, because this is
    the only part that needs a GPU and a second dependency tree.
    """
    if not rows:
        return {}
    sys.path.insert(0, audioldm_root)
    import csv as _csv

    import numpy as np
    import torch

    from audioldm_eval.feature_extractors.panns.models import Cnn14

    from continuo_expressive.tarsource import load_row

    labels_csv = os.environ.get("CONTINUO_EXPRESSIVE_PANNS_LABELS", "assets/class_labels_indices.csv")
    with open(labels_csv) as f:
        labels = [r["display_name"] for r in _csv.DictReader(f)]

    # Cnn14.__init__ loads its own checkpoint from ./ckpt relative to the *cwd*, so the
    # caller has to be somewhere that has it. Say so plainly rather than let torch raise
    # a bare FileNotFoundError about a relative path.
    if not os.path.isfile("ckpt/Cnn14_16k_mAP=0.438.pth"):
        raise SystemExit(
            "audioldm_eval loads Cnn14 from ./ckpt relative to the current directory, and\n"
            "  ./ckpt/Cnn14_16k_mAP=0.438.pth is not there. Run this from a directory that\n"
            "  has it (NV-Bench's repo root does), or symlink it into place.")

    model = Cnn14(features_list=["2048", "logits"], sample_rate=16000, window_size=512,
                  hop_size=160, mel_bins=64, fmin=50, fmax=8000, classes_num=527)
    ck = torch.load("ckpt/Cnn14_16k_mAP=0.438.pth", map_location="cpu", weights_only=False)
    missing, _ = model.load_state_dict(ck["model"], strict=False)
    real = [k for k in missing
            if not k.startswith(("spectrogram_extractor", "logmel_extractor", "spec_augmenter"))]
    if real:
        raise SystemExit(f"Cnn14 weights failed to load: {real[:5]}")
    model = model.eval().to(device)

    # Decode clips concurrently so I/O can keep up with the model.
    from concurrent.futures import ThreadPoolExecutor

    def decode(r):
        try:
            return load_row(r)
        except Exception as e:
            print(f"  [warn] {r['id']}: {type(e).__name__}: {e}", file=sys.stderr)
            return None

    out: dict[str, set] = {}
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for i in range(0, len(rows), 64):
            chunk = rows[i:i + 64]
            wavs, kept = [], []
            for r, w in zip(chunk, pool.map(decode, chunk)):
                if w is not None:
                    wavs.append(w)
                    kept.append(r)
            if not wavs:
                continue
            # batch=1 on purpose. Batching pads every clip to the longest in its batch,
            # and padding can change the ranked classes. Score each clip without
            # batch padding to keep verification stable.
            for b in range(0, len(wavs), 1):
                sub, subrows = wavs[b:b + 1], kept[b:b + 1]
                n = max(len(w) for w in sub)
                batch = np.zeros((len(sub), n), dtype="float32")
                for j, w in enumerate(sub):
                    batch[j, :len(w)] = w
                with torch.no_grad():
                    p = model(torch.from_numpy(batch).to(device))["clipwise_output"].cpu().numpy()
                for r, scores in zip(subrows, p):
                    out[r["id"]] = {labels[k] for k in scores.argsort()[-topk:][::-1]}
            if (i // 64) % 20 == 0:
                print(f"  panns: {len(out)}/{len(rows)}", file=sys.stderr, flush=True)
    finally:
        pool.shutdown(wait=False)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nv", required=True, help="continuo-nv output JSONL")
    ap.add_argument("--out", required=True, help="where to write the verified rows")
    ap.add_argument("--suspect-below", type=float, default=0.6,
                    help="stage 1: reject tags on a row whose asr_ratio is under this")
    ap.add_argument("--topk", type=int, default=3,
                    help="stage 2: how many AudioSet classes count as 'what this clip is'")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=16,
                    help="stage 2 audio-decoding threads; the decode, not the network, "
                         "is this stage's cost")
    ap.add_argument("--audioldm-root", default="third_party/audioldm_eval",
                    help="checkout providing audioldm_eval.feature_extractors.panns")
    ap.add_argument("--tar-dir", default="", help="corpus dir for source_tar rows (CONTINUO_EXPRESSIVE_TAR_DIR)")
    ap.add_argument("--ckpt-root", default="",
                    help="directory holding ckpt/Cnn14_16k_mAP=0.438.pth; this is "
                         "chdir'd into for stage 2 (NV-Bench's repo root)")
    ap.add_argument("--no-panns", action="store_true", help="stop after stage 1")
    args = ap.parse_args(argv)
    if args.tar_dir:
        os.environ["CONTINUO_EXPRESSIVE_TAR_DIR"] = args.tar_dir

    # PANNs finds its checkpoint relative to cwd. Resolve paths before changing it.
    args.nv, args.out = os.path.abspath(args.nv), os.path.abspath(args.out)
    if not args.no_panns:
        if args.ckpt_root:
            os.chdir(os.path.expanduser(args.ckpt_root))
        if not os.path.isfile("ckpt/Cnn14_16k_mAP=0.438.pth"):
            raise SystemExit(
                f"stage 2 needs ckpt/Cnn14_16k_mAP=0.438.pth under {os.getcwd()}.\n"
                "  Pass --ckpt-root (NV-Bench's repo root has it), or --no-panns to stop\n"
                "  after stage 1.")

    rows = list(read_jsonl(args.nv))
    tagged = [r for r in rows if r.get("nv_tags")]
    print(f"stage 0: {len(rows)} row(s), {len(tagged)} carrying a tag", file=sys.stderr)

    rejected: dict[str, dict[str, str]] = defaultdict(dict)

    # ---- stage 1: the transcript collapsed, so its tags describe a failure to hear
    for r in tagged:
        ratio = r.get("asr_ratio")
        if ratio is None:
            # written before continuo-nv carried the field; recompute it from the same two
            # strings rather than skip the stage on an older run's output
            ratio = asr_ratio(r.get("text", ""), r.get("txt"))
            r["asr_ratio"] = ratio
        if ratio is not None and ratio < args.suspect_below:
            for t in set(r["nv_tags"]):
                rejected[r["id"]][t] = "collapse"
    n1 = sum(1 for r in tagged if rejected.get(r["id"]))
    print(f"stage 1 (collapse):  {n1} row(s) rejected outright", file=sys.stderr)

    # ---- stage 2: sung audio can cause false speech-event tags
    if not args.no_panns:
        survivors = [r for r in tagged
                     if any(t in SINGING_SENSITIVE and t not in rejected[r["id"]]
                            for t in set(r["nv_tags"]))]
        print(f"stage 2 (panns):     {len(survivors)} row(s) still need checking",
              file=sys.stderr)
        top = panns_scores(survivors, args.device, args.audioldm_root, args.topk, args.workers)
        n2 = 0
        for r in survivors:
            classes = top.get(r["id"])
            if not classes:
                continue
            hit = classes & (SINGING | MUSIC)
            if not hit:
                continue
            for t in set(r["nv_tags"]):
                if t in SINGING_SENSITIVE and t not in rejected[r["id"]]:
                    rejected[r["id"]][t] = f"singing({sorted(hit)[0]})"
                    n2 += 1
        print(f"stage 2 (panns):     {n2} tag(s) rejected as sung/musical", file=sys.stderr)

    # ---- write
    kept_counts: Counter = Counter()
    drop_counts: Counter = Counter()
    with JsonlWriter(args.out) as sink:
        for r in rows:
            bad = rejected.get(r["id"], {})
            # `_tar` / `_path` are decoder state that load_row caches onto the row, and a
            # PosixPath is not JSON. Private keys never belong in the output anyway.
            out = {k: v for k, v in r.items() if not k.startswith("_")}
            out["nv_verified"] = [t for t in r.get("nv_tags", []) if t not in bad]
            out["nv_rejected"] = bad
            kept_counts.update(out["nv_verified"])
            drop_counts.update(bad.keys())   # {tag: reason}, so count the keys
            sink.write(out)

    print(f"\n{'tag':22s} {'kept':>7s} {'dropped':>8s} {'kept%':>7s}", file=sys.stderr)
    for t in sorted(set(kept_counts) | set(drop_counts),
                    key=lambda x: -(kept_counts[x] + drop_counts[x])):
        k, d = kept_counts[t], drop_counts[t]
        print(f"  {t:20s} {k:7d} {d:8d} {100 * k / max(k + d, 1):6.1f}%", file=sys.stderr)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
