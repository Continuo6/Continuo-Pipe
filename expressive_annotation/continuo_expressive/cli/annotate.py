"""``continuo-annotate`` — the main pass: gender, age, accent, volume, pitch, speed.

One process holds the neural heads, the optional ASR, and the DSP, and streams a
manifest through them. Emotion predictions, when available, are merged by clip
id with an explicit threshold.

Five things make this cheap on a large corpus:

**Batching.** Every head runs on a batch, not a clip. The original ran each clip
alone through each head, which spends most of a GPU's time on kernel launches.

**Batched pitch.** F0 was the largest single cost here, and almost all of it was the
Viterbi decode running one clip at a time. :func:`~continuo_expressive.features.pitch.measure_batch`
decodes a whole batch in one call while keeping each clip's path its own.

**A capped thread pool.** torch takes one CPU thread per core by default, which
oversubscribes the small convolutions this pass runs on the CPU — most of all penn's
16k->8k resample. See ``--torch-threads`` to bound CPU oversubscription.

**Lazy accent heads.** Both dialect heads are whisper-large-v3-sized. A Chinese-only
corpus should never pay to load the English one, so each is loaded the first time a
clip actually needs it — on a single-language corpus that halves accent load time and
GPU residency.

**Overlapped decoding.** Audio decoding and loudness run on a thread pool one batch
ahead of the GPU, so disk I/O hides behind the previous batch's forward pass.

The manifest's own ``txt`` short-circuits the largest cost of all: when every row
already carries a transcript (the real large-corpus case), whisper is never loaded.

Output is one JSON object per clip. Any field can be null — too short to measure, no
bucket for that language, the model abstained, the gate rejected it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

from .. import config
from ..audio import BatchLoader, truncate
from ..ensemble.age import cascade_age
from ..ensemble.dialect import cascade_dialect
from ..ensemble.emotion_gate import gate_emotion
from ..features import pitch as pitch_feat
from ..features import speed as speed_feat
from ..features import volume as volume_feat
from ..heads import loader as head_loader
from ..jsonl import (JsonlWriter, ManifestError, done_ids, dumps, index_by_id,
                     load_manifest, resolve_audio_path)
from ..registry import ACCENT_LABELS_FOR_LANG, restrict_accent

#: language -> registry name of the dialect head that owns it. A language absent here
#: gets no accent, and that is not an oversight: accent is language-INTERNAL, so it
#: needs a head trained on that language's own dialects. German, Russian, Portuguese,
#: Japanese, Hindi, Thai and Korean have no such head in Voxlect, and inventing one by
#: pointing the English head at them would produce confident nonsense.
ACCENT_HEADS = {"en": "voxlect_english", "zh": "voxlect_mandarin",
                "yue": "voxlect_mandarin", "ar": "voxlect_arabic"}
HEAD_TAGS = {"voxlect_english": "voxlect-english", "voxlect_mandarin": "voxlect-mandarin",
             "voxlect_arabic": "voxlect-arabic"}

TRUNCATE_SECONDS = 15.0        # every neural head's encoder cap


class HeadPool:
    """Holds the always-on heads and loads dialect heads on first use."""

    def __init__(self, device: str, batch_age_head: bool = False):
        self.device = device
        self.age_gender = head_loader.load("audeering_age_gender", device)
        self.age_vox = head_loader.load("voxprofile_wavlm_age", device,
                                        allow_batching=batch_age_head)
        self._accent: dict[str, object] = {}

    def accent(self, name: str):
        """The dialect head for a language, or None if its checkpoint is not here.

        Dialect heads load on the first clip that needs one, which means a checkpoint
        that is absent may surface after startup. A missing language head should not
        interrupt annotation of the remaining clips.

        A language whose head is missing is treated exactly like one that has no head at
        all — German, Japanese, Korean and the rest already come out with ``accent: null``
        by design.
        """
        if name not in self._accent:
            try:
                self._accent[name] = head_loader.load(name, self.device)
            except Exception as e:
                print(f"  [warn] {name} unavailable ({type(e).__name__}); clips in that "
                      "language get no accent for the rest of this run", file=sys.stderr)
                self._accent[name] = None
        return self._accent[name]


def _accent_for_batch(pool: HeadPool, waves: list, langs: list[str | None]) -> list[dict]:
    """Run each needed dialect head once over the clips that route to it."""
    blank = {"accent": None, "accent_top3": None, "accent_head": None}
    out: list[dict] = [dict(blank) for _ in waves]

    groups: dict[str, list[int]] = {}
    for i, lang in enumerate(langs):
        name = ACCENT_HEADS.get(lang or "")
        if name:
            groups.setdefault(name, []).append(i)

    for name, idx in groups.items():
        head = pool.accent(name)
        if head is None:                 # checkpoint absent; those rows keep `blank`
            continue
        preds = head.predict([waves[i] for i in idx])
        for i, pred in zip(idx, preds):
            # the declared language constrains the head, not the other way round: a
            # clip the corpus already calls `zh` must not come back Cantonese
            probs = restrict_accent(pred.probs, langs[i])
            ranked = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
            out[i] = {"accent": ranked[0][0] if ranked else None,
                      "accent_top3": {label: round(p, 3) for label, p in ranked} or None,
                      "accent_head": HEAD_TAGS[name]}
    return out


def _speed_of(row: dict, wav) -> dict:
    """Speaking rate for one row, from the numbers that actually go together.

    A sub-window of a long utterance carries ``utt_chars``/``utt_seconds`` — the whole
    utterance's text length and span — because its own ``txt`` is either the utterance's
    entire transcript over a fraction of the audio, or empty. Either way the window's own
    pair is not a rate. See tools/prepare_long.split_windows.
    """
    if row.get("utt_chars") and row.get("utt_seconds"):
        return speed_feat.from_span(row["utt_chars"], row.get("lang"), row["utt_seconds"])
    return speed_feat.from_text(row.get("txt"), row.get("lang"), wav)


def _needs_asr(row: dict) -> bool:
    """A row needs transcribing only if nothing else can give it a rate."""
    if (row.get("txt") or "").strip():
        return False
    return not (row.get("utt_chars") and row.get("utt_seconds"))


def annotate_batch(pool: HeadPool, transcriber, rows: list[dict], wavs: list,
                   extras: list[dict], pitch_gpu: int | None,
                   carry: Sequence[str] = ()) -> list[dict]:
    """Everything the pipeline knows about one batch of clips."""
    truncated = [truncate(w, TRUNCATE_SECONDS) for w in wavs]

    ag_preds = pool.age_gender.predict(truncated)
    vox_preds = pool.age_vox.predict(truncated)

    # speed + language: prefer the manifest's transcript, else transcribe
    if transcriber is None:
        langs = [r.get("lang") for r in rows]
        speeds = [_speed_of(r, w) for r, w in zip(rows, wavs)]
        asr_texts: list[str | None] = [None] * len(rows)
    else:
        # Only the rows that actually lack a transcript go through ASR. A manifest where
        # a few rows are missing `txt` — long-audio sub-windows, say — would otherwise
        # have whisper overwrite every other row's own transcript, moving the speed
        # denominator to min(duration, 30 s) and replacing a known language with a
        # detected one, all for the sake of a handful of rows.
        need = [i for i, r in enumerate(rows) if _needs_asr(r)]
        heard = dict(zip(need, transcriber.transcribe([wavs[i] for i in need]))) if need else {}
        langs, asr_texts, speeds = [], [], []
        for i, row in enumerate(rows):
            if i in heard:
                text, lang = heard[i]
                langs.append(lang)
                asr_texts.append(text)
                speeds.append(speed_feat.from_asr(text, lang, wavs[i]))
            else:
                langs.append(row.get("lang"))
                asr_texts.append(None)
                speeds.append(_speed_of(row, wavs[i]))

    accents = _accent_for_batch(pool, truncated, langs)

    # Pitch for the whole batch in one pass. Framing and inference are per-frame, and the
    # Viterbi decode stays per clip, so this is the same computation reorganised — see
    # features.pitch.measure_batch. It is worth doing: decoding one clip at a time made
    # pitch the single largest cost in this pass.
    genders = [p.pred_label for p in ag_preds]
    pitches = pitch_feat.measure_batch(wavs, genders, gpu=pitch_gpu)

    records = []
    for i, row in enumerate(rows):
        gender = genders[i]
        aud_age = ag_preds[i].continuous.get("age_years")
        vox_age = vox_preds[i].continuous.get("age_years")
        aud_age = round(aud_age, 1) if aud_age is not None else None
        vox_age = round(vox_age, 1) if vox_age is not None else None

        rec = {
            "id": row["id"],
            # carried manifest fields go first, so a row reads source-then-prediction
            **{k: row[k] for k in carry if k in row},
            # what the models actually saw, in seconds. The manifest's own `duration`
            # is a claim about the file; this is measured off the decoded waveform, and
            # it is what any downstream weighting (see continuo_expressive.aggregate)
            # has to weight by.
            "dur_s": round(len(wavs[i]) / config.TARGET_SR, 3),
            "gender": gender,
            "age_years": aud_age,
            "age_vox_years": vox_age,
            "age_band": cascade_age(gender, aud_age, vox_age),
            **accents[i],
            **extras[i],                                   # volume, computed on the pool
            **pitches[i],
            "speed_cps": speeds[i]["speed_cps"],
            "speed": speeds[i]["speed"],
            "lang": langs[i],
        }
        if asr_texts[i] is not None:
            rec["asr_text"] = asr_texts[i]
        records.append(rec)
    return records


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="continuo-annotate",
        description="Paralinguistic annotation: gender, age, accent, volume, pitch, speed.")
    src = p.add_argument_group("input")
    src.add_argument("--manifest", help="JSONL; each row needs wav_path (id, txt, lang optional)")
    src.add_argument("--audio", help="annotate a single audio file")
    src.add_argument("--audio-root", default="",
                     help="resolve relative wav_path against this root, and refuse any "
                          "path that escapes it")
    src.add_argument("--limit", type=int, default=0, help="stop after N clips")
    src.add_argument("--tar-dir", default="",
                     help="where the corpus tars live, for a manifest whose rows name a "
                          "source_tar + source_member instead of a wav_path. Clips are "
                          "then decoded straight out of the tar and nothing is cut to "
                          "disk. Overrides CONTINUO_EXPRESSIVE_TAR_DIR.")
    src.add_argument("--shard", default="",
                     help="take one slice of the manifest, as `i/n` (0-based). Every "
                          "worker reads the same file and claims rows where "
                          "index %% n == i, so splitting a corpus across machines needs "
                          "no split files and no coordination — the slices are disjoint "
                          "by construction and each has its own --out to merge later.")
    src.add_argument("--strict", action="store_true",
                     help="abort on an unusable manifest row instead of skipping it")

    out = p.add_argument_group("output")
    out.add_argument("--out", default="", help="write JSONL here")
    out.add_argument("--carry", default="",
                     help="comma-separated manifest fields to copy into each output row "
                          "(e.g. wav_path,txt,speaker,duration). Without this the "
                          "annotation file holds only predictions, so nothing downstream "
                          "can tell what was said or which file it came from without "
                          "joining back to the manifest.")
    out.add_argument("--resume", action="store_true",
                     help="append to --out, skipping ids already recorded there")
    out.add_argument("--stdout", action="store_true",
                     help="also echo every record to stdout (default when --out is unset)")

    merge = p.add_argument_group("merge")
    merge.add_argument("--emotion-jsonl", default="",
                       help="optional external emotion predictions; merged by id and gated")
    merge.add_argument("--emotion-tau", type=float,
                       help="required with --emotion-jsonl; confidence gate in [0, 1]")
    merge.add_argument("--firered-jsonl", default="",
                       help="FireRedLID output {id, pred} from continuo-firered; upgrades zh "
                            "accent to the FireRedLID->Voxlect cascade")

    run = p.add_argument_group("runtime")
    run.add_argument("--device", default="cuda", help="torch device for the heads")
    run.add_argument("--batch-size", type=int, default=8, help="clips per forward pass")
    run.add_argument("--workers", type=int, default=4, help="audio-decoding threads")
    run.add_argument("--pitch-gpu", type=int, default=0,
                     help="GPU index for penn F0; -1 for CPU")
    run.add_argument("--follow", type=float, default=0.0,
                     help="keep the models loaded and re-read the manifest every N "
                          "seconds, for a manifest that is still being written. Loading "
                          "the heads costs minutes and annotating a shard costs minutes, "
                          "so a process that exits after each pass spends more than half "
                          "its life reading weights. 0 (default) runs once and exits.")
    run.add_argument("--follow-until", default="",
                     help="[--follow] stop once this file exists and a pass finds no new "
                          "rows — the writer's own completion marker. Without it "
                          "--follow runs until killed.")
    run.add_argument("--torch-threads", type=int, default=16,
                     help="cap torch's CPU thread pool (0 leaves it alone). torch "
                          "defaults to one thread per core, which on a many-core host "
                          "oversubscribes the small convolutions this pass runs on the "
                          "CPU. Multiple workers can otherwise each claim every "
                          "core. Results are unaffected by this setting.")
    run.add_argument("--batch-age-head", action="store_true",
                     help="batch the vox-profile age head too. Its upstream preprocessing "
                          "normalises over padded input, so age_vox_years can depend on "
                          "batch composition. Off by default so output is reproducible across "
                          "batch sizes.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.emotion_jsonl and (args.emotion_tau is None or not 0 <= args.emotion_tau <= 1):
        build_parser().error("--emotion-jsonl requires --emotion-tau in [0, 1]")
    if args.tar_dir:
        import os
        os.environ["CONTINUO_EXPRESSIVE_TAR_DIR"] = args.tar_dir
    endpoint = config.apply_hf_endpoint()
    if endpoint:
        print(f"  HF endpoint: {endpoint}", file=sys.stderr)

    if args.torch_threads > 0:
        import torch
        torch.set_num_threads(args.torch_threads)

    # ---- input ----
    shard: tuple[int, int] | None = None
    if args.shard:
        try:
            index, count = (int(x) for x in args.shard.split("/", 1))
        except ValueError:
            build_parser().error(f"--shard {args.shard!r} is not `i/n`")
        if not 0 <= index < count:
            build_parser().error(f"--shard {args.shard!r}: need 0 <= i < n")
        shard = (index, count)
    if args.resume and not args.out:
        build_parser().error("--resume needs --out to resume from")
    if args.follow and not args.manifest:
        build_parser().error("--follow needs --manifest")

    # ids already written, carried across --follow passes so the growing output file is
    # read from disk once rather than on every pass
    done: set[str] = done_ids(args.out) if args.resume else set()

    def select_rows(first: bool) -> list[dict]:
        if args.audio:
            path = resolve_audio_path(args.audio, args.audio_root or None)
            return [{"id": path.name, "wav_path": str(path), "_path": path}]
        if not args.manifest:
            build_parser().error("give --manifest or --audio")
        found = load_manifest(args.manifest, limit=args.limit,
                              audio_root=args.audio_root or None, strict=args.strict,
                              shard=shard)
        if shard is not None:
            if first:
                print(f"  shard {shard[0]}/{shard[1]}: {len(found)} clip(s) of this "
                      "manifest", file=sys.stderr)
        if done:
            before = len(found)
            found = [r for r in found if r["id"] not in done]
            if first:
                print(f"  resuming: {len(done)} clip(s) already in {args.out}, "
                      f"{before - len(found)} of this shard, {len(found)} left",
                      file=sys.stderr)
        return found

    rows = select_rows(first=True)
    if not rows and not args.follow:
        print("  nothing to do", file=sys.stderr)
        return 0

    # ---- sidecars ----
    emotions = index_by_id(args.emotion_jsonl) if args.emotion_jsonl else {}
    if emotions:
        print(f"  merging emotion <- {args.emotion_jsonl} ({len(emotions)} clips, "
              f"tau={args.emotion_tau})", file=sys.stderr)
    firered = {}
    if args.firered_jsonl:
        for cid, row in index_by_id(args.firered_jsonl).items():
            firered[cid] = row.get("pred") or row.get("lang")
        print(f"  merging FireRedLID <- {args.firered_jsonl} ({len(firered)} clips)",
              file=sys.stderr)

    # ---- models ----
    # A manifest that ships transcripts needs no ASR at all — that is the single
    # biggest saving available, so check before loading anything.
    # A sub-window that carries its utterance's chars/seconds needs no transcript of its
    # own, so it must not drag whisper into the run either.
    use_manifest_text = not any(_needs_asr(r) for r in rows)
    if use_manifest_text:
        print("  no row needs a transcript -> using manifest text for speed, "
              "ASR not loaded", file=sys.stderr)

    pool = HeadPool(args.device, batch_age_head=args.batch_age_head)
    transcriber = None
    if not use_manifest_text:
        from ..features.asr import Transcriber
        transcriber = Transcriber(device=args.device)
        transcriber.load()
        print(f"  loaded {'asr':22s} <- {transcriber.model_id}", file=sys.stderr)

    pitch_gpu = None if args.pitch_gpu < 0 else args.pitch_gpu
    carry = [f.strip() for f in args.carry.split(",") if f.strip()]
    if carry:
        print(f"  carrying manifest field(s) into the output: {', '.join(carry)}",
              file=sys.stderr)
    echo = args.stdout or not args.out

    def per_clip(row, wav):                 # CPU-only, runs in the decoder threads
        return volume_feat.measure(wav)

    def on_decode_error(row, exc):
        print(f"[warn] skip {row.get('id')}: {type(exc).__name__}: {exc}", file=sys.stderr)

    # ---- run ----
    started = time.time()
    written = 0
    over_cap = 0
    with JsonlWriter(args.out, append=args.resume) as sink:
      while True:
        loader = BatchLoader(rows, batch_size=args.batch_size, workers=args.workers,
                             per_clip=per_clip, on_error=on_decode_error)
        pass_start, pass_written = time.time(), 0
        for batch_rows, wavs, extras in loader:
            over_cap += sum(1 for w in wavs
                            if len(w) > TRUNCATE_SECONDS * config.TARGET_SR)
            records = annotate_batch(pool, transcriber, batch_rows, wavs, extras,
                                     pitch_gpu, carry)
            for rec in records:
                if emotions:
                    rec.update(gate_emotion(emotions.get(rec["id"]), args.emotion_tau))
                if firered and rec.get("accent_head") == "voxlect-mandarin" \
                        and rec["id"] in firered:
                    fused = cascade_dialect(firered[rec["id"]], rec["accent"])
                    # LID can disagree with the corpus about the language itself
                    # (`yue` for a clip declared `zh`); the declaration wins
                    allowed = ACCENT_LABELS_FOR_LANG.get(rec.get("lang") or "")
                    if not allowed or fused in allowed:
                        rec["accent"] = fused
                        rec["accent_head"] = "firered->voxlect-mandarin"
                sink.write(rec)
                done.add(rec["id"])
                if echo:
                    print(dumps(rec))
                written += 1
                pass_written += 1
            if not echo and pass_written % 200 < args.batch_size:
                rate = pass_written / max(time.time() - pass_start, 1e-6)
                print(f"  {pass_written}/{len(rows)} clips  ({rate:.1f}/s)",
                      file=sys.stderr)

        if not args.follow:
            break
        # The models stay resident across passes; that is the whole point. Loading them
        # costs minutes and a shard costs minutes, so a process that exited after each
        # pass spent more of its life reading weights than annotating.
        marked = bool(args.follow_until) and Path(args.follow_until).exists()
        if marked and not rows:
            print("  writer finished and nothing new: done", file=sys.stderr)
            break
        if not rows:
            time.sleep(args.follow)
        rows = select_rows(first=False)
        if rows:
            print(f"  follow: {len(rows)} new clip(s)", file=sys.stderr)

    elapsed = time.time() - started
    print(f"  done: {written} clip(s) in {elapsed:.1f}s ({written / max(elapsed, 1e-6):.1f}/s)",
          file=sys.stderr)
    if over_cap:
        # silent truncation is the failure mode this warning exists to surface: gender,
        # age and accent describe only the opening of an over-long clip, and nothing in
        # the output says so. On a long recording, segment it first — see
        # tools/prepare_long.py.
        print(f"  [warn] {over_cap}/{written} clip(s) longer than {TRUNCATE_SECONDS:g} s; "
              f"gender/age/accent were measured on the first {TRUNCATE_SECONDS:g} s only",
              file=sys.stderr)
    if args.out:
        print(f"  wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ManifestError as e:
        print(f"manifest error: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)
