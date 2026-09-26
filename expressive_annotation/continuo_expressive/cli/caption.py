"""``continuo-caption`` — turn the predicted tags into TTS instruction text.

Runs Qwen3-Omni-Captioner over the audio *plus* a per-clip text prompt built from the
tags ``continuo-annotate`` already produced. The tags are supplied to the model and it is
told to use them verbatim: the model is doing the writing, not the measuring, and it
must not override a metered ``loud`` with its own impression.

Each clip draws one of four forms (``free`` / ``aps`` / ``dsd`` / ``rp``) and one of
two caption languages, both hashed from the clip id — see
:mod:`continuo_expressive.prompts`.

Two inputs, joined by id: tags live in the annotation file, audio paths live in the
manifest.

**Backends.** ``--backend vllm`` (default) batches many clips through one engine;
``--backend transformers`` runs one clip per ``generate()`` call. On a 30B
mixture-of-experts the difference is large — see
:mod:`continuo_expressive.captioners`.

**Resumable.** Raw ``{id, caption, caption_form, caption_lang}`` records are appended
to ``--raw`` as each chunk completes, and ids already there are skipped. The merge into
the annotation file happens afterwards and is atomic — the merge target is usually the
annotation file itself, and a crash mid-write must not destroy it.

``--merge-only`` folds whatever is already in ``--raw`` into the annotation file
without generating anything or loading a model. On a long run that is how you get a
usable annotation file before the run ends, and it is how several sharded runs (one
per GPU, ``--no-merge``, raw files concatenated) become one file at the end. Clips with
no caption yet are left untouched rather than stamped with a null, so a later merge
fills them in.

**trust_remote_code.** Loading a model with ``trust_remote_code=True`` executes Python
from the model repo in this process. It is off unless you pass
``--trust-remote-code`` (or set ``CONTINUO_EXPRESSIVE_TRUST_REMOTE_CODE=1``); recent transformers
support Qwen3-Omni natively and do not need it.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Sequence

from .. import config
from ..audio import batched, load_wav, load_source
from ..captioners import build_captioner, MAX_NEW_TOKENS
from ..jsonl import (JsonlWriter, ManifestError, load_jsonl, read_jsonl,
                     resolve_audio_path, write_jsonl_atomic)
from ..prompts import FORMS, LANGS, build_prompt, pick_form, pick_lang

MAX_SECONDS = 30                 # model card: keep audio at or under 30 s
#: fields carried from the raw caption file into the annotation file. The last two are
#: provenance: how much audio the caption was actually written from, and whether that
#: was all of it. On a corpus of short clips they are constant; on anything longer they
#: are the difference between a caption of a recording and a caption of its opening.
CAPTION_FIELDS = ("caption", "caption_form", "caption_lang",
                  "caption_seconds", "caption_truncated")


def _load_chunk(rows: Sequence[dict], workers: int) -> tuple[list[dict], list, list[dict]]:
    """Decode a chunk's audio in parallel. -> (ok rows, waveforms, failed rows).

    Anything past ``MAX_SECONDS`` is dropped here. The full length is stashed on the
    row so the record can say so: a clip silently reduced to its first 30 s produces a
    caption describing an opening, and nothing else in the output would reveal that.
    """
    def read(row):
        try:
            wav = load_source(row)
        except Exception as e:
            print(f"  {row['id']} DECODE ERROR {type(e).__name__}: {e}", flush=True)
            return None
        row["_seconds"] = len(wav) / config.TARGET_SR
        return wav[: int(MAX_SECONDS * config.TARGET_SR)]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        wavs = list(pool.map(read, rows))
    ok_rows, ok_wavs, failed = [], [], []
    for row, wav in zip(rows, wavs):
        (failed if wav is None else ok_rows).append(row)
        if wav is not None:
            ok_wavs.append(wav)
    return ok_rows, ok_wavs, failed


def _chunks_with_prefetch(rows: Sequence[dict], size: int, workers: int) -> Iterator:
    """Yield decoded chunks, decoding the next one while the caller generates."""
    blocks = list(batched(list(rows), size))
    if not blocks:
        return
    with ThreadPoolExecutor(max_workers=1) as ahead:
        pending = ahead.submit(_load_chunk, blocks[0], workers)
        for nxt in blocks[1:] + [None]:
            current = pending.result()
            pending = ahead.submit(_load_chunk, nxt, workers) if nxt else None
            yield current
            if pending is None:
                break


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="continuo-caption",
        description="Generate instruction text from annotated tags with Qwen3-Omni-Captioner.")
    p.add_argument("--annot", required=True, help="annotation JSONL — tag source and merge target")
    p.add_argument("--manifest", default="",
                   help="JSONL with {id, wav_path}, joined by id. Not needed with --no-audio.")
    p.add_argument("--audio-root", default="", help="root for relative wav_path, enforced")
    p.add_argument("--tar-dir", default="",
                   help="where the corpus tars live, for a manifest whose rows name a "
                        "source_tar instead of a wav_path. Overrides CONTINUO_EXPRESSIVE_TAR_DIR.")
    p.add_argument("--raw", default="", help="raw caption JSONL (default: <annot>_captions_raw.jsonl)")
    p.add_argument("--out", default="", help="merged output (default: --annot, rewritten in place)")
    p.add_argument("--limit", type=int, default=0, help="cap clips, for a smoke test")
    p.add_argument("--no-merge", action="store_true", help="write --raw only, skip the merge")
    p.add_argument("--merge-only", action="store_true",
                   help="merge existing --raw captions into --annot and stop; generates "
                        "nothing and loads no model. Safe to run mid-run, and how "
                        "sharded runs are combined.")
    p.add_argument("--form", default="random", choices=("random",) + FORMS,
                   help="caption form: random per clip (default) | free | aps | dsd | rp")
    p.add_argument("--lang", default="random", choices=("random",) + LANGS,
                   help="caption language: random per clip (default) | en | zh")
    p.add_argument("--seed", type=int, default=0, help="seed for the per-clip form/lang draw")
    p.add_argument("--example", action="store_true",
                   help="[free form] append the worked example (default: off)")
    p.add_argument("--no-audio", action="store_true",
                   help="caption from the tags and the record's own `txt` transcript, "
                        "with no audio at all. This is the honest way to caption a long "
                        "recording: the model takes 30 s, and feeding it an excerpt has "
                        "it describe minutes it never heard. --manifest is then optional.")
    p.add_argument("--describe-variation", action="store_true",
                   help="[long recordings] let the *_spans timeline contribute a line "
                        "about how volume, rate and emotion move across the file, and "
                        "ask the caption to mention it. No effect on single clips.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="sampling temperature; 0 = greedy (diversity comes from the "
                        "rotated examples)")
    p.add_argument("--model", default="", help=f"captioner model id (default: {config.captioner_model()})")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="allow the model repo to execute its own Python in this process")
    p.add_argument("--backend", default="vllm", choices=("vllm", "transformers"),
                   help="vllm batches many clips through one engine (default); "
                        "transformers runs one clip per generate() call")
    p.add_argument("--chunk-size", type=int, default=256,
                   help="[vllm] clips per engine call; also the checkpoint interval")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                   help="generation cap per caption. A single voice fits the default "
                        "several times over; a dialogue writes one line per speaker "
                        "(measured up to 64 tokens each, 10 speakers in a container), "
                        "so a dialogue run wants ~1024. A caption that hits the cap is "
                        "cut mid-sentence and reported as a [warn] line, not marked on "
                        "the record.")
    p.add_argument("--max-chunk-failures", type=int, default=3,
                   help="consecutive chunk failures before the engine is treated as "
                        "dead and the process exits non-zero (default 3)")
    p.add_argument("--tensor-parallel", type=int, default=1,
                   help="shard the engine across this many GPUs (vllm backend). Needed "
                        "when the model does not fit one card: the 30B captioner is 60 GB "
                        "and a 24 GB card needs 4.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                   help="[vllm] fraction of the device given to weights + KV cache")
    p.add_argument("--workers", type=int, default=8, help="audio-decoding threads")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.tar_dir:
        import os
        os.environ["CONTINUO_EXPRESSIVE_TAR_DIR"] = args.tar_dir
    config.apply_hf_endpoint()

    raw_path = args.raw or (args.annot[:-6] if args.annot.endswith(".jsonl") else args.annot) \
        + "_captions_raw.jsonl"

    if not args.manifest and not (args.no_audio or args.merge_only):
        build_parser().error("give --manifest, or --no-audio to caption from text")

    annot = load_jsonl(args.annot, required=("id",), limit=args.limit)
    # id -> where the audio is: a path, or the tar member it lives in. Only the
    # fields needed to read it are kept; the manifest is millions of rows.
    sources: dict[str, dict] = {}
    if args.manifest:
        for row in read_jsonl(args.manifest, required=("id",)):
            if row.get("source_tar") and row.get("source_member"):
                sources[row["id"]] = {k: row[k] for k in
                                      ("source_tar", "source_member", "tar_offset",
                                       "tar_size", "rel_start", "rel_end")
                                      if k in row}
            elif row.get("wav_path"):
                sources[row["id"]] = {k: row[k] for k in
                                      ("wav_path", "carrier_start_samples",
                                       "carrier_end_samples", "sample_rate") if k in row}

    done: dict[str, dict] = {}
    try:
        for row in read_jsonl(raw_path, required=("id",)):
            done[row["id"]] = {k: row.get(k) for k in CAPTION_FIELDS}
    except FileNotFoundError:
        pass

    todo = []
    for row in (() if args.merge_only else annot):
        if row["id"] in done:
            continue
        if args.no_audio:
            if not (row.get("txt") or "").strip():
                print(f"[warn] {row['id']}: no txt to caption from, skipping", file=sys.stderr)
                continue
            todo.append(row)
            continue
        src = sources.get(row["id"])
        if not src:
            print(f"[warn] {row['id']}: not in the manifest, skipping", file=sys.stderr)
            continue
        if "wav_path" in src:
            try:
                row["_path"] = resolve_audio_path(src["wav_path"], args.audio_root or None)
            except ManifestError as e:
                print(f"[warn] {row['id']}: {e}", file=sys.stderr)
                continue
        row.update(src)
        todo.append(row)

    if args.merge_only:
        print(f"{len(annot)} clips, {len(done)} captioned in {raw_path}; merge only",
              flush=True)
    else:
        print(f"{len(annot)} clips, {len(done)} already captioned, {len(todo)} to do "
              f"(backend={args.backend}, form={args.form}, lang={args.lang})", flush=True)

    if todo:
        captioner = build_captioner(
            args.backend, args.model,
            trust_remote_code=args.trust_remote_code or config.trust_remote_code(),
            temperature=args.temperature, chunk_size=args.chunk_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel,
            text_only=args.no_audio, max_new_tokens=args.max_new_tokens)
        captioner.load()

        started, written, failures = time.time(), 0, 0
        with JsonlWriter(raw_path, append=True) as sink:
            batches = (((chunk, [None] * len(chunk), []) for chunk in
                        batched(todo, captioner.chunk_size)) if args.no_audio
                       else _chunks_with_prefetch(todo, captioner.chunk_size, args.workers))
            for rows, wavs, failed in batches:
                items, meta = [], []
                for row, wav in zip(rows, wavs):
                    clip_id = row["id"]
                    form = pick_form(clip_id, args.seed) if args.form == "random" else args.form
                    lang = pick_lang(clip_id, args.seed) if args.lang == "random" else args.lang
                    # a dialogue record is captioned in the drawn form too, one line
                    # per speaker; `speakers` on the record is what says it is one
                    items.append((clip_id, wav,
                                  build_prompt(row, form=form, lang=lang,
                                               include_example=args.example,
                                               describe_variation=args.describe_variation,
                                               use_transcript=args.no_audio)))
                    meta.append((clip_id, form, lang, row.get("_seconds")))
                try:
                    captions = captioner.generate(items)
                    failures = 0
                except Exception as e:
                    # One chunk failing is survivable — a clip that upsets the engine
                    # should not cost the run — but a *dead* engine fails every chunk
                    # from then on, instantly, forever. vLLM's EngineCore dies on its own
                    # OOM and every later generate() raises EngineDeadError; catching
                    # that as "pending for the next run" turns a crash into a process
                    # that looks alive, holds its GPUs and writes nothing, which is worse
                    # than crashing. Give up after a few in a row and let the supervisor
                    # restart the engine.
                    failures += 1
                    print(f"  chunk failed ({type(e).__name__}: {e}); "
                          "its clips stay pending for the next run", flush=True)
                    if failures >= args.max_chunk_failures:
                        print(f"  {failures} chunk(s) failed in a row — treating the "
                              "engine as dead and exiting non-zero so it is restarted",
                              file=sys.stderr, flush=True)
                        return 1
                    continue

                for (clip_id, form, lang, clip_seconds), text in zip(meta, captions):
                    record = {"id": clip_id, "caption": text,
                              "caption_form": form, "caption_lang": lang,
                              "caption_seconds": round(min(clip_seconds, MAX_SECONDS), 2)
                              if clip_seconds else None,
                              "caption_truncated": bool(clip_seconds
                                                        and clip_seconds > MAX_SECONDS)}
                    done[clip_id] = {k: record[k] for k in CAPTION_FIELDS}
                    sink.write(record)
                    written += 1
                # a clip that could not be decoded is recorded with a null caption, so
                # a resumed run does not retry an unreadable file forever
                for row in failed:
                    record = {"id": row["id"], **dict.fromkeys(CAPTION_FIELDS)}
                    done[row["id"]] = {k: record[k] for k in CAPTION_FIELDS}
                    sink.write(record)

                rate = written / max(time.time() - started, 1e-6)
                sample = next((c for c in captions if c), "")
                print(f"  {written}/{len(todo)}  ({rate:.2f} clips/s)  {sample[:120]}",
                      flush=True)

        elapsed = time.time() - started
        print(f"raw captions -> {raw_path}  ({written} in {elapsed:.0f}s, "
              f"{written / max(elapsed, 1e-6):.2f} clips/s)", flush=True)

    if args.no_merge and not args.merge_only:
        return 0

    # Re-read the full annotation file: --limit may have truncated `annot` above, and
    # the merge must not drop the rows it never looked at.
    out_path = args.out or args.annot
    merged = 0

    def rows_with_captions():
        nonlocal merged
        for row in read_jsonl(args.annot, required=("id",)):
            if row["id"] in done:
                row.update(done[row["id"]])
                merged += 1
            yield row

    total = write_jsonl_atomic(out_path, rows_with_captions())
    print(f"merged {merged}/{total} captions -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ManifestError as e:
        print(f"manifest error: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)
