#!/usr/bin/env python3
"""Cut Continuo ``type=long`` containers into annotatable segments, from the tars.

A long container is a single-speaker recording — a podcast episode, a reading, a
monologue — running from tens of seconds to two hours. The annotation pipeline is
built for short utterances: its neural heads see the first 15 s of whatever they are
given and nothing warns you about the rest. Feeding a long container in directly
produces one label set describing its opening and stamped over the whole thing.

So the container is cut into its own utterances first, each of which the pipeline can
measure honestly, and the per-file answer is rebuilt afterwards by
``tools/aggregate_long.py``. This script is the cutting half.

**Two sources of segment boundaries**, and they are not equivalent:

``short``
    The ``short`` view inside each container's own JSON sidecar. Carries ``language``,
    ``speaker`` and ``dnsmos`` per entry — all three matter downstream, ``language``
    most of all, because it routes the accent head and selects the speed edges.

``metainfo``
    Separate metainfo tars can provide additional segment boundaries and cover
    containers whose short view is empty. They carry only ``start`` / ``end`` /
    ``text`` — no language, no speaker.

``union`` (the default) takes metainfo's boundaries and recovers the missing metadata by
matching each member back to the short entry it overlaps (IoU >= 0.5). Unmatched
segments get a language through the fallback chain in :func:`assign_languages`.
Short entries absent from metainfo are also retained.

A language that cannot be established is left ``None`` on purpose. That is the
pipeline's existing graceful degradation: no accent head runs and ``speed`` gets no
bucket, while ``speed_cps`` is still reported. Guessing would be worse.

Audio: each container is decoded once with ffmpeg and sliced many times — a two-hour
container holds thousands of segments and must not be decoded thousands of times. m4a
needs an edit-list-aware decoder or every offset shifts by ~2112 samples; ffmpeg is one.
Decoding goes straight to ``--segment-sr`` (16 kHz by default) because every model
downstream resamples there anyway, and storing 44.1 kHz would cost 2.7x the disk for
nothing.

    python tools/prepare_long.py \
        --tars 'corpus/audio-*.tar' \
        --metainfo-dir metainfo/long \
        --out-dir runs/long

Resumable: a segment whose FLAC already exists is not re-cut, and a container whose
segments are all present is never decoded.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from continuo_expressive.jsonl import ManifestError, safe_name

#: overlap needed to call a metainfo member and a short entry the same utterance
IOU_MATCH = 0.5
#: a short entry overlapping metainfo by less than this is treated as uncovered
ORPHAN_OVERLAP = 0.1
#: What a segment is written as. FLAC keeps the decode exactly; MP3 costs a second lossy
#: generation on top of the corpus's own 128 kbps AAC, and buys about 4.4x the disk.
#:
#: Lossy compression can change neural head predictions, even when simple DSP
#: measurements remain similar.
#:
#: Prefer reading the corpus straight out of its tars
#: (:mod:`continuo_expressive.tarsource`, ``tools/index_tars.py``): no second
#: generation, and nothing to store. Cut to FLAC if you need files.
SEGMENT_FORMATS = {"flac": (".flac", "FLAC", "PCM_16"), "mp3": (".mp3", "MP3", None)}

#: how a container's view shows up in a tar member name
VIEW_MARKERS = {"long": "_long_", "dialogue": "_dlg_"}
LONG_MARKER = VIEW_MARKERS["long"]


def member_view(member: str) -> str:
    """Which view a tar member belongs to, from its name.

    Standalone utterances are named ``<recording>_<nnn>_<nnn>_spkN`` with no marker, so
    anything that is not explicitly long or dialogue is a short container.
    """
    for view, marker in VIEW_MARKERS.items():
        if marker in member:
            return view
    return "short"
#: every model downstream consumes 16 kHz mono; decoding goes straight there
TARGET_SR = 16000


# --------------------------------------------------------------------------- io
def load_idx(idx_path: Path) -> dict[str, tuple[int, int]]:
    """``member -> (offset, size)`` from a WebDataset-style ``.tar.idx`` sidecar."""
    out: dict[str, tuple[int, int]] = {}
    with open(idx_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                out[parts[0]] = (int(parts[1]), int(parts[2]))
    return out


def read_at(tar_path: Path, offset: int, size: int) -> bytes:
    with open(tar_path, "rb") as f:
        f.seek(offset)
        return f.read(size)


def load_metainfo(meta_tar: Path) -> dict[str, dict]:
    """Whole metainfo tar -> ``{member_name: parsed json}``.

    Read once per tar in the parent: these are ~1.2 MB and reopening one per container
    would rescan it hundreds of times.
    """
    out: dict[str, dict] = {}
    with tarfile.open(meta_tar) as t:
        for member in t.getmembers():
            if not member.isfile():
                continue
            handle = t.extractfile(member)
            if handle is None:
                continue
            try:
                out[member.name] = json.load(handle)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # UnicodeDecodeError is not a JSONDecodeError, only a sibling under
                # ValueError. A corrupt sidecar should affect only its own container.
                continue
    return out


def decode(audio: bytes, sample_rate: int) -> np.ndarray:
    """m4a bytes -> mono float32 at ``sample_rate``, via ffmpeg.

    ffmpeg reads a temp file rather than stdin: the MP4 moov atom is not guaranteed to
    precede the media data, and a non-seekable input then fails outright.

    The whole container lands in memory, twice over for a moment — ffmpeg's output
    buffer and the array viewing it. At 16 kHz that is 64 kB per second, so the default
    five-minute cap costs about 19 MB per worker. Disabling the cap does not: the
    memory use grows with the longest container. Lower ``--workers`` if you
    raise ``--max-seconds``.
    """
    with tempfile.NamedTemporaryFile(suffix=".m4a") as tmp:
        tmp.write(audio)
        tmp.flush()
        cmd = ["ffmpeg", "-v", "error", "-i", tmp.name, "-f", "f32le",
               "-ac", "1", "-ar", str(sample_rate), "-"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: "
                           f"{proc.stderr.decode('utf-8', 'replace').strip()[:200]}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def write_audio(wav: np.ndarray, sr: int, dest: Path, fmt: str, quality: float) -> None:
    """Write one clip, via a temp file so a killed run never leaves a partial one."""
    suffix, container, subtype = SEGMENT_FORMATS[fmt]
    kwargs: dict = {"format": container}
    if subtype:
        kwargs["subtype"] = subtype
    else:
        kwargs["compression_level"] = quality
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(suffix + ".part")
    sf.write(tmp, wav, sr, **kwargs)
    tmp.replace(dest)


# ------------------------------------------------------------------- segmenting
def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi <= lo:
        return 0.0
    inter = hi - lo
    return inter / max((a[1] - a[0]) + (b[1] - b[0]) - inter, 1e-9)


def overlap_frac(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Fraction of ``a`` covered by ``b``."""
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return max(0.0, hi - lo) / max(a[1] - a[0], 1e-9)


def build_segments(local: dict, meta: dict | None, mode: str) -> list[dict]:
    """Merge the two boundary sources into one sorted segment list.

    Every segment carries ``lang`` (possibly None, filled in later by
    :func:`assign_languages`) and ``seg_source`` naming where its boundaries came from.
    """
    shorts = [{"start": float(x["rel_start"]), "end": float(x["rel_end"]),
               "text": x.get("text") or "", "lang": x.get("language"),
               "dnsmos": x.get("dnsmos"), "seg_source": "short",
               "speaker": x.get("speaker"), "turn": None}
              for x in (local.get("short") or [])
              if x.get("rel_start") is not None and x.get("rel_end") is not None]

    # A dialogue's metainfo members are turns: each carries the speaker that said it
    # and a per-turn language hint, which is how a multi-speaker file can be folded
    # back per speaker instead of per file. Long metainfo has none of these fields and
    # x.get() leaves them None, so long containers come out exactly as before.
    members = [{"start": float(x["start"]), "end": float(x["end"]),
                "text": x.get("text") or "", "lang": x.get("lang_hint"),
                "dnsmos": x.get("dnsmos"), "seg_source": "metainfo",
                "speaker": x.get("speaker"), "turn": x.get("tag")}
               for x in ((meta or {}).get("members") or [])
               if x.get("start") is not None and x.get("end") is not None]

    if mode == "short" or not members:
        return sorted(shorts, key=lambda s: (s["start"], s["end"]))

    # metainfo boundaries win; recover language/dnsmos from the short entry each
    # member overlaps, so the +22% segments do not cost the metadata the pipeline needs
    # Both lists are in time order, so the match is a sweep rather than a full
    # cross-product. It matters: the largest container in one shard holds 2154 members
    # against 2046 short entries, which is 4.4 M comparisons and about a second of CPU
    # for what a linear pass does in milliseconds.
    #
    # `first` only ever moves forward. A short entry that ends before the current
    # member starts cannot overlap any later member either, because member starts are
    # non-decreasing — which holds whether or not the short entries overlap each other.
    consumed: set[int] = set()
    first = 0
    for m in members:
        while first < len(shorts) and shorts[first]["end"] <= m["start"]:
            first += 1
        best, best_iou = -1, 0.0
        j = first
        while j < len(shorts) and shorts[j]["start"] < m["end"]:
            v = iou((m["start"], m["end"]), (shorts[j]["start"], shorts[j]["end"]))
            if v > best_iou:
                best, best_iou = j, v
            j += 1
        if best >= 0 and best_iou >= IOU_MATCH:
            m["lang"] = shorts[best]["lang"]
            m["dnsmos"] = shorts[best]["dnsmos"]
            consumed.add(best)

    segments = list(members)
    # A dialogue's fold is per speaker and keyed by the turn tag. Its sidecar's short
    # entries name speakers in another id space (`<id>_spk0` against the turns' `0`)
    # and carry no tag, so a stray one metainfo does not cover became a fifth "speaker"
    # keyed `('..._spk0',)` with one 0.9 s segment. Dialogue segmentation is metainfo's
    # alone; the shorts only lend their language and dnsmos above.
    if mode == "union" and any(m["turn"] for m in members):
        mode = "metainfo"
    if mode == "union":
        # short entries metainfo does not cover at all. Requiring near-zero overlap
        # keeps the result non-overlapping, so no audio is counted twice. Same sweep,
        # this time walking the members alongside the shorts.
        first = 0
        for j, s in enumerate(shorts):
            if j in consumed:
                continue
            while first < len(members) and members[first]["end"] <= s["start"]:
                first += 1
            k, covered = first, False
            while k < len(members) and members[k]["start"] < s["end"]:
                if overlap_frac((s["start"], s["end"]),
                                (members[k]["start"], members[k]["end"])) >= ORPHAN_OVERLAP:
                    covered = True
                    break
                k += 1
            if not covered:
                segments.append(s)

    return sorted(segments, key=lambda s: (s["start"], s["end"]))


def assign_languages(segments: list[dict], file_languages: list | None) -> None:
    """Fill every ``lang`` that the short-entry match could not supply, in place.

    Chain, most specific first:

    1. the segment's own language, from a matched short entry;
    2. the nearest segment in time that has one — a monologue rarely switches language
       mid-sentence, so a neighbour is a good guess;
    3. the container's ``languages`` list, **only when it holds exactly one**. 37% of
       long containers are multilingual (up to 16 languages in one file), so this list
       is not a per-segment answer and must not be used as one;
    4. None — no accent head, no speed bucket, ``speed_cps`` still reported.

    Each segment records which rung it landed on in ``lang_source``.
    """
    anchors = [(0.5 * (s["start"] + s["end"]), s["lang"]) for s in segments if s["lang"]]
    only = None
    langs = [x for x in (file_languages or []) if x]
    if len(langs) == 1:
        only = langs[0]

    for s in segments:
        if s["lang"]:
            s["lang_source"] = "segment"
            continue
        if anchors:
            mid = 0.5 * (s["start"] + s["end"])
            s["lang"] = min(anchors, key=lambda a: abs(a[0] - mid))[1]
            s["lang_source"] = "neighbour"
        elif only:
            s["lang"] = only
            s["lang_source"] = "file"
        else:
            s["lang"] = None
            s["lang_source"] = "none"


def split_windows(segments: list[dict], window: float, hop: float,
                  min_seconds: float) -> list[dict]:
    """Cut any segment longer than ``window`` into sub-windows.

    Only ~0.5% of segments are affected, but leaving them whole would silently discard
    everything past the heads' 15 s cap — the exact failure this script exists to fix.

    The transcript stays on the first sub-window. Splitting it would need alignment
    nobody has here, and duplicating it onto every sub-window would inflate each one's
    characters-per-second by the number of windows.

    Neither of those is a speaking rate, though: the window holding the text has only
    part of the audio (rate too high — measured median +32%, worst +100%), and the rest
    have no text at all. So every sub-window also carries ``utt_chars``/``utt_seconds``,
    the whole utterance's own numerator and denominator, and continuo-annotate prefers them.
    Speaking rate is a property of the utterance; its windows share it.

    ``window <= 0`` disables splitting, which is what the standalone-utterance path
    wants: one of those containers *is* one utterance, and cutting it yields two clips
    that are each not one — the first holding a whole transcript over fifteen seconds,
    the second holding none. There the documented behaviour is the right one: the heads
    see the first 15 s, the DSP sees all of it, and continuo-annotate says so.
    """
    out: list[dict] = []
    for index, s in enumerate(segments):
        if window <= 0 or s["end"] - s["start"] <= window:
            out.append({**s, "index": index, "sub": None})
            continue
        start, sub = s["start"], 0
        while start < s["end"] - 1e-6:
            end = min(start + window, s["end"])
            if end - start >= min_seconds:
                out.append({**s, "start": start, "end": end, "index": index, "sub": sub,
                            "text": s["text"] if sub == 0 else "",
                            "utt_chars": len(s["text"] or ""),
                            "utt_seconds": round(s["end"] - s["start"], 3)})
                sub += 1
            start += hop
    return out


def segment_id(parent_id: str, index: int, sub: int | None) -> str:
    return f"{parent_id}_{index:04d}" + ("" if sub is None else f".{sub}")


# ------------------------------------------------------------------ per container
def cut_container(job: dict) -> dict:
    """Decode one container and write every one of its segments. Runs in a worker."""
    local, meta = job["local"], job["meta"]
    out_dir = Path(job["out_dir"])
    sr = job["segment_sr"]
    parent_id = local.get("id") or job["member"]
    result = {"parent_id": parent_id, "rows": [], "skipped": [], "error": None,
              "empty": False, "invalid": False, "containers": 0, "too_long": 0.0,
              "lang_sources": Counter()}

    try:
        # the id becomes a directory and a filename; it comes from a JSON sidecar
        safe_name(parent_id, "container id")
    except ManifestError as e:
        result["error"] = str(e)
        return result

    if meta is not None and meta.get("is_valid") is False:
        result["invalid"] = True
        return result

    duration = local.get("duration") or (meta or {}).get("duration") or 0.0
    if job["max_seconds"] and duration > job["max_seconds"]:
        result["too_long"] = duration
        return result

    segments = build_segments(local, meta, job["segments"])
    if not segments:
        result["empty"] = True
        return result

    assign_languages(segments, (meta or local).get("languages"))
    segments = split_windows(segments, job["window_seconds"], job["window_hop"],
                             job["min_seconds"])

    parent_duration = local.get("duration") or (meta or {}).get("duration")
    if job.get("manifest_only"):
        # Name the segments instead of cutting them: the rows already carry
        # rel_start/rel_end and the container's tar member, which is everything
        # tarsource.load_row needs to slice a segment out of the tar at read time. No
        # wav_path, because is_tar_row() treats its presence as "there is a local copy".
        #
        # The trade is honest and worth stating: nothing here is decoded, so a segment
        # whose sidecar names a few frames past what the decoder actually yields is not
        # caught by comparing against the audio. What *is* checked is the container's own
        # declared duration, because a turn that begins after
        # the file ends slices to zero samples and can crash the PANNs pass in the
        # model's padding rather than being skipped as an unreadable clip. Clamp to the
        # duration, drop what starts beyond it. And a container is decoded once per
        # segment downstream rather than once in total, which is the price of not writing
        # a separate file per segment to disk.
        limit = float(parent_duration) if parent_duration else None
        rows = []
        for s in segments:
            start, end = s["start"], s["end"]
            if limit is not None:
                if start >= limit:
                    result["skipped"].append(
                        {"id": segment_id(parent_id, s["index"], s["sub"]),
                         "seconds": round(end - start, 3),
                         "reason": "starts past the container's duration"})
                    continue
                end = min(end, limit)
            duration = end - start
            if duration < job["min_seconds"]:
                result["skipped"].append({"id": segment_id(parent_id, s["index"], s["sub"]),
                                          "seconds": round(duration, 3),
                                          "reason": "shorter than --min-seconds"})
                continue
            rows.append({
                "id": segment_id(parent_id, s["index"], s["sub"]),
                "txt": s["text"], "lang": s["lang"], "lang_source": s["lang_source"],
                "dnsmos": s["dnsmos"], "duration": round(duration, 3),
                "parent_id": parent_id, "parent_duration": parent_duration,
                "rel_start": round(start, 3), "rel_end": round(end, 3),
                "seg_source": s["seg_source"],
                **({"speaker": s["speaker"], "turn": s.get("turn")}
                   if s.get("speaker") is not None else {}),
                **({"utt_chars": s["utt_chars"], "utt_seconds": s["utt_seconds"]}
                   if s.get("sub") is not None else {}),
                "source_tar": job["tar_name"], "source_member": job["audio_member"],
                "tar_offset": job["audio_offset"], "tar_size": job["audio_size"],
            })
            result["lang_sources"][s["lang_source"]] += 1
        result["rows"] = rows
        return result

    container_dest = (Path(job["container_dir"]) /
                      f"{parent_id}{SEGMENT_FORMATS[job['format']][0]}"
                      if job["container_dir"] else None)
    rows, pending = [], []
    for s in segments:
        duration = s["end"] - s["start"]
        if duration < job["min_seconds"]:
            result["skipped"].append({"id": segment_id(parent_id, s["index"], s["sub"]),
                                      "seconds": round(duration, 3),
                                      "reason": "shorter than --min-seconds"})
            continue
        seg_id = segment_id(parent_id, s["index"], s["sub"])
        dest = out_dir / parent_id / f"{seg_id}{SEGMENT_FORMATS[job['format']][0]}"
        rows.append({
            "id": seg_id,
            "wav_path": (str(dest.relative_to(job["relative_to"]))
                         if job["relative_to"] else str(dest)),
            "txt": s["text"],
            "lang": s["lang"],
            "lang_source": s["lang_source"],
            "dnsmos": s["dnsmos"],
            "duration": round(duration, 3),
            "parent_id": parent_id,
            "parent_duration": parent_duration,
            "rel_start": round(s["start"], 3),
            "rel_end": round(s["end"], 3),
            "seg_source": s["seg_source"],
            **({"speaker": s["speaker"], "turn": s.get("turn")}
               if s.get("speaker") is not None else {}),
            # only on sub-windows: the utterance this window was cut from, so its rate
            # can be measured on the text and the seconds that actually go together
            **({"utt_chars": s["utt_chars"], "utt_seconds": s["utt_seconds"]}
               if s.get("sub") is not None else {}),
            # where this came from, so the cut audio is reproducible and therefore
            # deletable. The corpus's own pack index maps ids to tars too, but its
            # member names predate the FLAC-to-m4a migration and no longer resolve.
            "source_tar": job["tar_name"],
            "source_member": job["audio_member"],
        })
        result["lang_sources"][s["lang_source"]] += 1
        if not dest.is_file():
            pending.append((dest, s["start"], s["end"]))

    want_container = container_dest is not None and not container_dest.is_file()
    if pending or want_container:                # decode once, slice many
        try:
            wav = decode(read_at(Path(job["audio_tar"]), job["audio_offset"],
                                 job["audio_size"]), sr)
        except Exception as e:
            result["error"] = f"{parent_id}: {e}"
            return result
        if want_container:
            # the whole recording on its own timeline, so the rel_start/rel_end in the
            # manifest — and the timestamps in an aggregated record — index straight
            # into it. Free: the container is already decoded for the slicing below.
            write_audio(wav, sr, container_dest, job["format"], job["quality"])
            result["containers"] = 1
        for dest, start, end in pending:
            a = max(0, min(int(round(start * sr)), len(wav)))
            b = max(a, min(int(round(end * sr)), len(wav)))
            chunk = wav[a:b]
            if len(chunk) < job["min_seconds"] * sr:
                # the sidecar can name a few frames past what the decoder yields; a
                # tail segment can clamp to nothing. Drop it rather than write silence.
                rows = [r for r in rows if r["wav_path"] != str(dest)]
                result["skipped"].append({"id": dest.stem, "seconds": len(chunk) / sr,
                                          "reason": "empty after clamping to decoded length"})
                continue
            write_audio(chunk, sr, dest, job["format"], job["quality"])

    result["rows"] = rows
    return result


def build_jobs(audio_tar: Path, metainfo_dir: Path | None, out_dir: Path,
               args, container_dir: Path | None = None) -> list[dict]:
    """One job per long container in one audio tar."""
    idx_path = audio_tar.with_suffix(audio_tar.suffix + ".idx")
    if not idx_path.is_file():
        # One unusable tar must not end an otherwise valid corpus extraction.
        print(f"[warn] {audio_tar.name}: no .idx sidecar, skipped", file=sys.stderr)
        return []
    idx = load_idx(idx_path)

    meta_by_member: dict[str, dict] = {}
    if metainfo_dir is not None:
        meta_tar = metainfo_dir / audio_tar.name
        if meta_tar.is_file():
            meta_by_member = load_metainfo(meta_tar)
        else:
            print(f"[warn] no metainfo tar for {audio_tar.name}; falling back to the "
                  "short view for this tar", file=sys.stderr)

    wanted = {x.strip() for x in args.languages.split(",") if x.strip()}
    views = {v.strip() for v in args.container_types.split(",") if v.strip()}
    unknown = views - set(VIEW_MARKERS) - {"short"}
    if unknown:
        raise SystemExit(f"unknown --container-types {sorted(unknown)}; "
                         f"expected any of short, long, dialogue")
    jobs = []
    for member, (offset, size) in sorted(idx.items()):
        if not member.endswith(".json") or member_view(member) not in views:
            continue
        audio_member = member[:-len(".json")] + ".m4a"
        if audio_member not in idx:
            continue
        try:
            local = json.loads(read_at(audio_tar, offset, size))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[warn] {audio_tar.name}:{member}: unreadable sidecar "
                  f"({type(e).__name__}), skipped", file=sys.stderr)
            continue
        if wanted:
            meta = meta_by_member.get(member)
            spoken = {x for x in ((meta or local).get("languages") or []) if x}
            # a container counts as this language only if it holds nothing else —
            # otherwise "the Chinese subset" quietly includes bilingual recordings
            if not spoken or not spoken <= wanted:
                continue
        jobs.append({
            "audio_tar": str(audio_tar),
            "tar_name": audio_tar.name,
            "audio_member": audio_member,
            "audio_offset": idx[audio_member][0],
            "audio_size": idx[audio_member][1],
            "member": member,
            "local": local,
            "meta": meta_by_member.get(member),
            "out_dir": str(out_dir),
            "segments": args.segments,
            "segment_sr": args.segment_sr,
            "min_seconds": args.min_seconds,
            "window_seconds": args.window_seconds,
            "window_hop": args.window_hop or args.window_seconds,
            "container_dir": str(container_dir) if container_dir else "",
            "max_seconds": args.max_seconds,
            "format": args.segment_format,
            "quality": args.segment_quality,
            "relative_to": str(out_dir.parent) if args.relative_paths else "",
            "manifest_only": args.manifest_only,
        })
    return jobs


# ---------------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prepare_long",
        description="Cut Continuo type=long containers into segments and write a manifest.")
    p.add_argument("--tars", default="",
                   help="audio tar path or glob, e.g. 'corpus/audio-*.tar'. "
                        "Each needs its .tar.idx beside it.")
    p.add_argument("--tars-from", default="",
                   help="file of tar paths, one per line (blank lines and #comments "
                        "ignored). For the set a glob cannot name: continuing a corpus "
                        "that has grown, where the tars still to do are whichever ones "
                        "the last run's manifest never mentions. Combines with --tars.")
    p.add_argument("--metainfo-dir", default="",
                   help="directory of same-named segment metainfo tars. Without it only "
                        "the containers' own short view is available (--segments short).")
    p.add_argument("--out-dir", required=True, help="destination for segments/ and the manifest")
    p.add_argument("--manifest-only", action="store_true",
                   help="write the segment manifest without cutting any audio; rows "
                        "carry rel_start/rel_end and the container's tar member, so "
                        "every pass reads segments straight out of the corpus tars")
    p.add_argument("--manifest", default="", help="manifest path (default: <out-dir>/manifest.jsonl)")
    p.add_argument("--tar-log", default="",
                   help="append each tar's path here once it is finished. The manifest "
                        "cannot serve as that record: a tar holding nothing in "
                        "--languages produces no rows, so a supervisor working from the "
                        "manifest alone would retry it forever. tools/pending_tars.py "
                        "reads this to decide what a restart still has to do.")
    p.add_argument("--append", action="store_true",
                   help="add to an existing manifest instead of refusing to overwrite "
                        "it. Rows are appended, so a row's line number never changes and "
                        "`continuo-annotate --shard i/n` stays disjoint across a restart. Only "
                        "pass tars the manifest does not already cover: a container's "
                        "rows are emitted whether or not its audio was already written, "
                        "so re-running a finished tar duplicates its rows.")
    p.add_argument("--segments", default="union", choices=("short", "metainfo", "union"),
                   help="boundary source: short = the container's own view; metainfo = the "
                        "sidecar tars (+22%% segments); union = metainfo plus the short "
                        "entries it misses (default)")
    p.add_argument("--segment-sr", type=int, default=16000,
                   help="sample rate for the written segments (default 16000, what every "
                        "model downstream uses)")
    p.add_argument("--relative-paths", action="store_true",
                   help="write wav_path relative to --out-dir instead of absolute. A "
                        "manifest of absolute paths only works on the machine that cut "
                        "it; with this, any machine can run it by passing the same "
                        "manifest and its own --audio-root, which continuo-annotate already "
                        "resolves against and refuses to let a path escape.")
    p.add_argument("--segment-format", default="flac", choices=tuple(SEGMENT_FORMATS),
                   help="flac keeps the decode exactly; mp3 is about 4.4x smaller and "
                        "costs a second lossy generation over the corpus's own AAC. "
                        "The measurements survive it — see SEGMENT_FORMATS.")
    p.add_argument("--segment-quality", type=float, default=0.7,
                   help="[mp3] libsndfile compression level, 0 best to 1 worst. On 16 kHz "
                        "mono speech 0.4 lands near 48 kbps, 0.5 near 42, 0.6 near 37 and "
                        "0.7 near 33 (default). It is VBR, so these are averages.")
    p.add_argument("--min-seconds", type=float, default=0.5,
                   help="drop segments shorter than this; below ~0.5 s neither loudness "
                        "nor speaking rate is measurable (default 0.5)")
    p.add_argument("--window-seconds", type=float, default=15.0,
                   help="split segments longer than this, so nothing is lost to the "
                        "neural heads' 15 s encoder cap (default 15). 0 disables it — "
                        "use that for standalone utterances, where the clip is the "
                        "unit and splitting it invents two that are not.")
    p.add_argument("--window-hop", type=float, default=0.0,
                   help="hop between sub-windows (default: --window-seconds, no overlap)")
    p.add_argument("--max-seconds", type=float, default=300.0,
                   help="skip containers longer than this (default 300, i.e. five "
                        "minutes). A recording is annotated as a whole, and past a few "
                        "minutes that stops meaning anything: a 49-minute lecture "
                        "yields a fifty-span timeline, which is a wall rather than a "
                        "description, and its transcript alone crowds the captioner's "
                        "context. 0 disables the cap. Expect this to cost real coverage "
                        "— the long tail is where the audio is; the run prints how much.")
    p.add_argument("--container-types", default="long",
                   help="which containers to cut the short view out of: long, "
                        "dialogue, short, or a comma-separated mix. `long` (default) "
                        "is the long-recording path. `short` takes the standalone "
                        "utterances, which need no cutting at all. Naming all three "
                        "gathers every annotated utterance in the corpus — on this "
                        "dataset that is where the hours are: standalone clips hold "
                        "about a tenth of the speech that the long and dialogue "
                        "containers do.")
    p.add_argument("--languages", default="",
                   help="comma-separated language codes; keep only containers whose "
                        "entire language list falls inside them. `--languages zh` is a "
                        "strictly monolingual Chinese subset, `--languages zh,en` also "
                        "admits the mixed ones. Empty keeps everything.")
    p.add_argument("--container-dir", default="",
                   help="also write each container's FULL audio here, one FLAC per "
                        "recording on its own timeline. Off by default because it "
                        "roughly doubles the disk; on when the deliverable is "
                        "(recording, instruction) pairs rather than segments.")
    p.add_argument("--workers", type=int, default=8, help="containers decoded in parallel")
    p.add_argument("--max-hours", type=float, default=0.0,
                   help="stop once this many hours of speech have been cut. The run "
                        "finishes the tar it is in, so the total overshoots slightly "
                        "rather than leaving a tar half-cut and unresumable.")
    p.add_argument("--limit", type=int, default=0, help="cap containers, for a smoke test")
    return p


def warn_memory(max_seconds: float, workers: int, budget_gb: float = 4.0) -> None:
    """Say so when the cap and the worker count together could exhaust memory.

    A container is decoded whole before it is sliced, so peak memory is roughly
    ``workers * max_seconds * 64 kB/s``, doubled while ffmpeg's buffer and the array
    over it both exist. Silence here would turn a raised cap into an OOM two hours
    into a run.
    """
    seconds = max_seconds if max_seconds > 0 else 4 * 3600      # the longest seen here
    gb = 2 * workers * seconds * TARGET_SR * 4 / 1e9
    if gb > budget_gb:
        print(f"[warn] --max-seconds {max_seconds:g} with --workers {workers} could need "
              f"~{gb:.1f} GB of memory: each container is decoded whole before it is "
              "sliced. Lower --workers, or keep the cap.", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    warn_memory(args.max_seconds, args.workers)

    if not (args.tars or args.tars_from):
        build_parser().error("give --tars or --tars-from")
    named = set(glob.glob(args.tars)) if args.tars else set()
    if args.tars_from:
        listed = [ln.strip() for ln in Path(args.tars_from).read_text().splitlines()]
        missing = [ln for ln in listed
                   if ln and not ln.startswith("#") and not Path(ln).is_file()]
        if missing:
            # a mistyped or half-copied list should not look like a short corpus
            print(f"[warn] {len(missing)} path(s) in {args.tars_from} do not exist, "
                  f"first: {missing[0]}", file=sys.stderr)
        named |= {ln for ln in listed if ln and not ln.startswith("#")}
    tars = sorted({Path(p) for p in named if Path(p).suffix == ".tar"})
    if not tars:
        print(f"no tars matched {args.tars!r} / {args.tars_from!r}", file=sys.stderr)
        return 2
    print(f"  {len(tars)} tar(s) to read", file=sys.stderr)

    metainfo_dir = Path(args.metainfo_dir).expanduser().resolve() if args.metainfo_dir else None
    if metainfo_dir is None and args.segments != "short":
        print(f"[warn] --segments {args.segments} needs --metainfo-dir; "
              "falling back to the short view", file=sys.stderr)

    out_dir = Path(args.out_dir).expanduser().resolve()
    segments_dir = out_dir / "segments"
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.jsonl"
    if args.manifest_only:
        out_dir.mkdir(parents=True, exist_ok=True)      # no segments/ to create
    else:
        segments_dir.mkdir(parents=True, exist_ok=True)

    container_dir = Path(args.container_dir).expanduser().resolve() if args.container_dir else None
    if container_dir:
        container_dir.mkdir(parents=True, exist_ok=True)

    # One tar at a time. A job carries its container's parsed sidecar and metainfo, so
    # building them all up front would hold every sidecar in the corpus in memory at
    # once. Rows stream to the manifest for the same reason.
    written = skipped_rows = 0
    stats = {"empty": 0, "invalid": 0, "containers": 0}
    errors: list[str] = []
    too_long: list[float] = []
    lang_sources: Counter = Counter()
    speech = 0.0
    parents: set[str] = set()
    seen_tars = 0

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and manifest_path.stat().st_size and not args.append:
        # Cutting a corpus takes many hours, so the second run over one --out-dir is
        # normally a continuation, not a redo — and opening the manifest "w" throws away
        # everything the first run recorded while its audio sits on disk unreferenced.
        raise SystemExit(
            f"{manifest_path} already holds {sum(1 for _ in open(manifest_path)):,} "
            "row(s). Opening it for writing would discard them. Pass --append to add to "
            "it, or --manifest to write the continuation somewhere else.")
    tar_log = open(args.tar_log, "a", encoding="utf-8") if args.tar_log else None

    def mark_done(tar: Path) -> None:
        if tar_log is not None:
            tar_log.write(f"{tar}\n")
            tar_log.flush()

    with open(manifest_path, "a" if args.append else "w", encoding="utf-8") as manifest:
        for tar in tars:
            if args.max_hours and speech >= args.max_hours * 3600:
                print(f"  reached --max-hours {args.max_hours:g} after {seen_tars} tar(s)",
                      flush=True)
                break
            remaining = (args.limit - len(parents)) if args.limit else None
            if remaining is not None and remaining <= 0:
                break
            jobs = build_jobs(tar, metainfo_dir, segments_dir, args, container_dir)
            if remaining is not None:
                jobs = jobs[:remaining]
            if not jobs:
                # nothing wanted in this tar — still finished, and recorded as such
                mark_done(tar)
                continue
            seen_tars += 1

            tar_rows: list[dict] = []
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for result in pool.map(cut_container, jobs):
                    if result["error"]:
                        errors.append(result["error"])
                    stats["empty"] += bool(result["empty"])
                    stats["invalid"] += bool(result["invalid"])
                    stats["containers"] += result["containers"]
                    if result["too_long"]:
                        too_long.append(result["too_long"])
                    tar_rows.extend(result["rows"])
                    skipped_rows += len(result["skipped"])
                    lang_sources.update(result["lang_sources"])

            # ids are unique within a shard by construction; a collision means two
            # containers claimed the same recording and one would silently win
            ids = {r["id"] for r in tar_rows}
            if len(ids) != len(tar_rows):
                print(f"[warn] {tar.name}: {len(tar_rows) - len(ids)} duplicate segment "
                      "id(s); the manifest would double-count them", file=sys.stderr)
            tar_rows.sort(key=lambda r: (r["parent_id"], r["rel_start"]))
            for row in tar_rows:
                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                speech += row["duration"]
                parents.add(row["parent_id"])
            # flush per tar: a run over a whole corpus takes hours, and a manifest that
            # only materialises at the end is a manifest a killed run does not have
            manifest.flush()
            mark_done(tar)
            written += len(tar_rows)
            print(f"  {seen_tars}/{len(tars)} tar(s), {written} segments", flush=True)

    if tar_log is not None:
        tar_log.close()

    if not written and not too_long:
        print(f"no {args.container_types} containers in {len(tars)} tar(s)",
              file=sys.stderr)
        return 2

    empty, invalid, containers = stats["empty"], stats["invalid"], stats["containers"]
    print(f"\n{written} segments ({speech / 3600:.2f} h) from {len(parents)} container(s) "
          f"-> {manifest_path}")
    if args.manifest_only:
        print("no audio cut: every row names its container's tar member and its "
              "rel_start/rel_end, and is read from the corpus tars")
    else:
        print(f"segments -> {segments_dir}")
    if container_dir:
        print(f"{containers} full recording(s) written -> {container_dir}")
    if lang_sources:
        total = sum(lang_sources.values())
        detail = ", ".join(f"{k} {v} ({100 * v / total:.1f}%)"
                           for k, v in lang_sources.most_common())
        print(f"language provenance: {detail}")
        if lang_sources.get("none"):
            print(f"  {lang_sources['none']} segment(s) have no language: they get no "
                  "accent head and no speed bucket (speed_cps is still reported)")
    if too_long:
        hours = sum(too_long) / 3600
        print(f"{len(too_long)} container(s) over --max-seconds {args.max_seconds:g} "
              f"were skipped, dropping {hours:.2f} h of recording "
              f"(longest {max(too_long) / 60:.0f} min). The long tail holds most of a "
              "corpus's audio; raise or disable the cap if that matters more than "
              "keeping each annotated unit readable.")
    if empty:
        print(f"{empty} container(s) carried no segments in either source and were skipped")
    if invalid:
        print(f"{invalid} container(s) marked is_valid=false in the metainfo were skipped")
    if skipped_rows:
        print(f"{skipped_rows} segment(s) dropped below --min-seconds {args.min_seconds}")
    for err in errors[:10]:
        print(f"[error] {err}", file=sys.stderr)
    if len(errors) > 10:
        print(f"[error] ... and {len(errors) - 10} more", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
