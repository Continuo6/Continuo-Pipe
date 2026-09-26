#!/usr/bin/env python3
"""Build a manifest that points into the tars, without cutting any audio.

``prepare_long.py`` decodes every container and writes a file per segment. For
standalone utterances that work buys nothing: a segment *is* a tar member, so the only
thing the pipeline actually needs is where the member starts and how long it is. This
reads the ``.tar.idx`` and each container's JSON sidecar — no ffmpeg, no audio written —
and emits rows :mod:`continuo_expressive.tarsource` can decode on demand.

    python tools/index_tars.py --tars 'corpus/continuo-*.tar' \
        --out runs/x/manifest.jsonl --languages zh,en

This avoids writing a separate decoded file for every utterance.
The trade is that each pass decodes the m4a itself rather than reading a prepared file,
so a three-pass run decodes three times instead of once — see the module docstring of
tarsource for what that costs and why the decode is equivalent.

Long and dialogue containers hold *many* utterances per member. Indexing them is
supported (``--container-types``) but every segment then decodes the whole container,
so for those ``prepare_long.py`` — which decodes once and slices many — is still the
right tool. This defaults to standalone only for that reason.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.prepare_long import load_idx, member_view, read_at  # noqa: E402


def sidecar_segments(local: dict, view: str) -> list[dict]:
    """The utterances a container's own sidecar declares, for the given view."""
    entries = local.get(view) or local.get("short") or []
    if isinstance(entries, dict):
        entries = entries.get("segments") or entries.get("members") or []
    return [e for e in entries if isinstance(e, dict)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="index_tars",
        description="Manifest rows pointing at tar members; no audio is written.")
    p.add_argument("--tars", default="", help="tar path or glob")
    p.add_argument("--tars-from", default="", help="file of tar paths, one per line")
    p.add_argument("--out", required=True, help="manifest destination")
    p.add_argument("--languages", default="",
                   help="keep only containers whose whole language list falls inside "
                        "these, e.g. zh,en")
    p.add_argument("--container-types", default="short",
                   help="short (default) | long | dialogue, comma-separated. Only "
                        "`short` is a member-per-utterance mapping; see the module "
                        "docstring before using the others.")
    p.add_argument("--min-seconds", type=float, default=0.5,
                   help="drop utterances shorter than this (default 0.5)")
    p.add_argument("--relative-tars", action="store_true",
                   help="write source_tar as a bare filename, so the manifest runs on "
                        "any machine that has the corpus under --tar-dir")
    args = p.parse_args(argv)

    named = set(glob.glob(args.tars)) if args.tars else set()
    if args.tars_from:
        named |= {ln.strip() for ln in Path(args.tars_from).read_text().splitlines()
                  if ln.strip() and not ln.strip().startswith("#")}
    tars = sorted({Path(x) for x in named if Path(x).suffix == ".tar"})
    if not tars:
        print("no tars matched", file=sys.stderr)
        return 2

    wanted = {x.strip() for x in args.languages.split(",") if x.strip()}
    views = {v.strip() for v in args.container_types.split(",") if v.strip()}
    written = skipped_short = no_idx = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as out:
        for n, tar in enumerate(tars, 1):
            idx_path = Path(str(tar) + ".idx")
            if not idx_path.is_file():
                print(f"[warn] {tar.name}: no .idx, skipped", file=sys.stderr)
                no_idx += 1
                continue
            idx = load_idx(idx_path)
            for member, (offset, size) in sorted(idx.items()):
                if not member.endswith(".json") or member_view(member) not in views:
                    continue
                audio = member[:-len(".json")] + ".m4a"
                if audio not in idx:
                    continue
                try:
                    local = json.loads(read_at(tar, offset, size))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                langs = {x for x in (local.get("languages") or []) if x}
                if wanted and (not langs or not langs <= wanted):
                    continue
                for i, seg in enumerate(sidecar_segments(local, "short")):
                    start = float(seg.get("rel_start") or 0.0)
                    end = seg.get("rel_end")
                    end = float(end) if end is not None else None
                    if end is not None and end - start < args.min_seconds:
                        skipped_short += 1
                        continue
                    a_off, a_size = idx[audio]
                    # the container's own sidecar id, exactly as prepare_long.py uses it,
                    # so a manifest built either way names the same clips
                    stem = local.get("id") or audio[:-len(".m4a")]
                    out.write(json.dumps({
                        "id": f"{stem}_{i:04d}",
                        "source_tar": tar.name if args.relative_tars else str(tar),
                        "source_member": audio,
                        "tar_offset": a_off, "tar_size": a_size,
                        "rel_start": start, "rel_end": end,
                        "duration": None if end is None else round(end - start, 3),
                        "txt": seg.get("text"),
                        "lang": seg.get("language") or (next(iter(langs)) if len(langs) == 1 else None),
                        "dnsmos": seg.get("dnsmos"),
                        "parent_id": stem,
                    }, ensure_ascii=False) + "\n")
                    written += 1
            out.flush()
            if n % 50 == 0 or n == len(tars):
                print(f"  {n}/{len(tars)} tar(s), {written} row(s)", flush=True)

    print(f"\n{written} row(s) -> {args.out}")
    if skipped_short:
        print(f"{skipped_short} utterance(s) below --min-seconds {args.min_seconds}")
    if no_idx:
        print(f"{no_idx} tar(s) had no .idx and were skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
