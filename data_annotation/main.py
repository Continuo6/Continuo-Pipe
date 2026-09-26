"""Entry point for Continuo-Pipe — Phase 1 only.

Stage order is hardcoded here so the pipeline shape is obvious from one
file. The config only picks *which* adapter sits in each slot.

Phase 1: standardize → separate vocals → diarize → VAD → score (DNSMOS)
→ quality-filter → identify language → write per-segment WAV files +
``<sid>.partial.json`` manifest. **No transcription happens here.**

Phase 2 (``transcribe.py``) reads the partial manifests, runs ASR
batched across all files, applies post-ASR language gating, and writes
the final ``<sid>.json``. Splitting the two means ASR can saturate the
GPU on its own batched workload instead of context-switching with the
per-file feature stack.

CLI: ``--config_path`` and ``--global-size`` / ``--local-index`` shard
knobs. Per-segment ASR batch and ASR adapter settings are read from
config but only used by Phase 2.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
import warnings

import torch
import tqdm

from pipeline.builder import build_pipeline
from pipeline.config import PipelineConfig
from utils.logger import Logger
from utils.tool import check_env, detect_gpu, get_audio_files

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path",
        type=str,
        default="",
        help="input folder path; overrides config.io.input_folder_path when set",
    )
    parser.add_argument(
        "--config_path", type=str, default="config.json", help="config path"
    )
    parser.add_argument(
        "--global-size", dest="global_size", type=int, default=None,
        help="override runtime.global_size",
    )
    parser.add_argument(
        "--local-index", dest="local_index", type=int, default=None,
        help="override runtime.local_index",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.load(args.config_path)
    if args.input_folder_path:
        cfg.io.input_folder_path = args.input_folder_path
    if args.global_size is not None:
        cfg.runtime.global_size = args.global_size
    if args.local_index is not None:
        cfg.runtime.local_index = args.local_index

    logger = Logger.get_logger()

    if detect_gpu():
        logger.info("Using GPU")
        device = "cuda"
    else:
        logger.info("Using CPU")
        device = "cpu"

    check_env(logger)

    logger.debug("Building pipeline...")
    pipeline = build_pipeline(cfg, device)
    logger.debug("Warming up pipeline (loading models)...")
    pipeline.warmup()
    logger.debug("All stages ready")

    input_folder_path = cfg.io.input_folder_path
    if not os.path.exists(input_folder_path):
        raise FileNotFoundError(f"input_folder_path: {input_folder_path} not found")

    audio_paths = get_audio_files(input_folder_path)
    logger.debug(f"Scanning {len(audio_paths)} audio files in {input_folder_path}")

    gs = cfg.runtime.global_size
    li = cfg.runtime.local_index
    logger.debug(f"global size {gs}, local index {li}")
    shard_len = len(audio_paths) // gs if gs else len(audio_paths)
    batch = audio_paths[shard_len * li: shard_len * (li + 1)][::-1]

    # Per-shard structured failure log. Failed samples land here as JSONL so
    # an operator can re-queue them later; the loop never crashes on a single
    # bad file but also never silently swallows the traceback (legacy behavior
    # was a bare ``print(e)`` that lost the stack and didn't survive log
    # rotation).
    processed_root = input_folder_path.rstrip("/") + "_processed"
    os.makedirs(processed_root, exist_ok=True)
    failed_log_path = os.path.join(processed_root, "_failed.jsonl")
    n_failed = 0

    for path in tqdm.tqdm(batch, desc="audio files"):
        try:
            pipeline.process(path)
        except Exception as e:  # noqa: BLE001 - keep the batch loop alive
            n_failed += 1
            logger.exception(f"pipeline failed on {path}: {e}")
            with open(failed_log_path, "a") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "audio_path": path,
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                    "traceback": traceback.format_exc(),
                    "shard": {"global_size": gs, "local_index": li},
                }, ensure_ascii=False) + "\n")
        gc.collect()
        torch.cuda.empty_cache()

    if n_failed:
        logger.warning(
            f"{n_failed} sample(s) failed in this shard; see {failed_log_path}"
        )


if __name__ == "__main__":
    main()
