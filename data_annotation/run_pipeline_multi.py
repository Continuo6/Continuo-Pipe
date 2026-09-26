"""Multi-GPU / multi-node Phase-1 launcher over tar metadata.

DECOUPLED from Phase-2: this only produces ``<sid>.partial.json`` (+ wavs,
all_shorts, long_chunks). Phase-2 (``transcribe.py``) consumes those via the
filesystem, separately — this launcher never references Phase-2.

Modes (Phase-1 and Phase-2 are independent processes, never welded):
  single-GPU Phase-1:  python run_pipeline_multi.py --gpus 7
  multi-GPU  Phase-1:  python run_pipeline_multi.py --gpus 0,1,2,3,4,5,6
  two nodes:           (node A) ... --nodes 2 --node-id 0
                       (node B) ... --nodes 2 --node-id 1"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_ATTEMPT = 3
PARTIAL_SUFFIX = ".partial.json"


def _out_dir(out_root: str, sid: str) -> Path:
    # 2-char fan-out so a 100k+ run doesn't pile millions of dirs in one node.
    return Path(out_root) / sid[:2] / sid


def _read_attempt(apath: Path) -> int:
    try:
        return int(apath.read_text())
    except Exception:  # noqa: BLE001
        return 0


def _record_fail(failed_log: Path, e: dict, stage: str, err: Exception, apath: Path) -> None:
    apath.parent.mkdir(parents=True, exist_ok=True)
    apath.write_text(str(_read_attempt(apath) + 1))  # bump attempt
    rec = {
        "sample_id": e.get("sample_id"),
        "tar_path": e.get("tar_path"),
        "member": e.get("member"),
        "stage": stage,
        "error_type": type(err).__name__,
        "error_msg": str(err)[:500],
        "ts": time.time(),
    }
    with open(failed_log, "a") as f:  # atomic-ish append
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Process one tar (shared by static-shard and dynamic-claim paths)
# ---------------------------------------------------------------------------

def _process_tar(pipe, tar_path, entries, out_root, failed_log, wid, tmp_dir,
                 logger, heartbeat=None, prefetch=2,
                 make_reader=None, publish=None) -> tuple[int, int, int]:

    done = skip = fail = 0
    if make_reader is None:
        from pipeline.io.tar_source import TarReader
        make_reader = lambda unit: TarReader(unit, tmp_dir)  # noqa: E731
    try:
        reader = make_reader(tar_path)
    except Exception as ex:  # noqa: BLE001 - whole unit unreadable
        logger.exception(f"[w{wid}] source open failed: {tar_path}")
        for e in entries:
            _record_fail(failed_log, e, "tar_open", ex,
                         _out_dir(out_root, e["sample_id"]) / ".attempt")
            fail += 1
        return done, skip, fail

    def _prep_task(e, od):
        """Extract + standardization on the prefetch thread. Never raises —
        carries any error back so the main thread records it + cleans tmp."""
        sid = e["sample_id"]
        tmp = None
        try:
            tmp = reader.extract(e["member"], sid)
            res = pipe.prep(tmp, save_path=str(od), audio_name=sid)
            return tmp, res, None
        except Exception as ex:  # noqa: BLE001
            return tmp, None, ex


    tail_ex = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix=f"p1tail-w{wid}")
    tails: deque = deque()          # (entry, out_dir, future)

    def _drain_tails(all_of_them: bool) -> None:
        nonlocal done, fail
        while tails and (all_of_them or len(tails) > 1 or tails[0][2].done()):
            te, tod, tfut = tails.popleft()
            try:
                tfut.result()
                done += 1
                if publish:

                    publish(te["sample_id"], long=True)
            except Exception as ex:  # noqa: BLE001 - keep batch alive
                fail += 1
                logger.exception(f"[w{wid}] tail failed: {te['sample_id']}")
                _record_fail(failed_log, te, "tail", ex, tod / ".attempt")

    try:
        with reader, ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"p1prep-w{wid}") as prep_ex:
            work = iter(entries)
            inflight: deque = deque()

            def _submit_next() -> bool:
                nonlocal skip
                for e in work:
                    sid = e["sample_id"]
                    od = _out_dir(out_root, sid)
                    if (od / f"{sid}{PARTIAL_SUFFIX}").exists():
                        skip += 1  # done OR 5h/PCM abort (empty partial)


                        if publish:
                            publish(sid, long=True)
                        continue
                    if _read_attempt(od / ".attempt") >= MAX_ATTEMPT:
                        skip += 1  # permanent fail
                        continue
                    inflight.append((e, od, prep_ex.submit(_prep_task, e, od)))
                    return True
                return False

            for _ in range(max(1, prefetch)):
                if not _submit_next():
                    break

            while inflight:
                e, od, fut = inflight.popleft()
                _submit_next()  # keep the prefetch pipeline full (overlap next front)
                sid = e["sample_id"]
                tmp, res, prep_exc = fut.result()
                try:
                    if prep_exc is not None:
                        raise prep_exc  # extract / standardization error
                    if res is None:
                        skip += 1  # already processed (race with a peer)
                    else:
                        ctx, status = res
                        if status == "ok":


                            if pipe.finish_gpu(ctx) == "ok":
                                tails.append((e, od, tail_ex.submit(pipe.tail, ctx)))
                            else:
                                done += 1
                                if publish:
                                    publish(sid, long=False)
                        else:
                            done += 1
                            if publish:
                                publish(sid, long=False)
                except Exception as ex:  # noqa: BLE001 - keep batch alive
                    fail += 1
                    logger.exception(f"[w{wid}] process failed: {sid}")
                    _record_fail(failed_log, e, "process", ex, od / ".attempt")
                finally:
                    if tmp and os.path.exists(tmp):
                        os.remove(tmp)
                    gc.collect()
                _drain_tails(all_of_them=False)
                if heartbeat:
                    heartbeat()
    finally:
        _drain_tails(all_of_them=True)
        tail_ex.shutdown(wait=True)
    return done, skip, fail


# ---------------------------------------------------------------------------
# Dynamic pull-based loop: claim the earliest free tar, repeat until all done
# ---------------------------------------------------------------------------


EXIT_STOPPED = 3


def classify_worker_exits(codes: "list[int]") -> "tuple[str, list[int]]":

    failed = [c for c in codes if c not in (0, EXIT_STOPPED)]
    if failed:
        return "failed", failed
    if EXIT_STOPPED in codes:
        return "stopped", [c for c in codes if c == EXIT_STOPPED]
    return "done", []


def run_dynamic(all_tars, claims, owner, process_fn, progress_cb=None,
                sleep_fn=time.sleep, should_stop=None,
                now_fn=time.time) -> tuple[int, int, int]:
    """Work-stealing scan: grab the earliest tar that's neither done nor
    live-claimed, process it, mark done; loop until every tar is done.

    ``process_fn(tar_path, entries) -> (done, skip, fail)`` does the work (and
    its own heartbeat). Kept free of pipeline imports so it's unit-testable
    with a mock ``process_fn`` and many concurrent callers."""
    done = skip = fail = 0
    # Memoize tars we've confirmed done (ours or a peer's) so we never re-stat
    # them on later passes — without this, each pass re-stats all ~6k .done
    # files (a 14-worker NAS metadata storm over a 10-day run). Only the
    # still-unfinished pool is re-checked each pass.
    seen_done: set[str] = set()
    # Seed the memo from ONE bulk scandir of _claims/ instead of stat-ing every
    # tar's .done individually on the first pass (~57k stats at startup, paid
    # again on every worker relaunch). Best-effort: the per-tar is_done/acquire
    # checks below still gate correctness, so a stale/empty seed is harmless.
    try:
        done_names = claims.done_names()
        if done_names:
            for te in all_tars:
                if Path(te[0]).name in done_names:
                    seen_done.add(te[0])
    except Exception:  # noqa: BLE001 - seeding is an optimization, never fatal
        pass

    deferred: dict = {}
    cooldown_s = claims.lease_s / 3.0

    def _attempt(tar_path, entries) -> bool:

        nonlocal done, skip, fail
        if claims.is_done(tar_path):
            seen_done.add(tar_path)
            deferred.pop(tar_path, None)
            return False  # finished by another worker mid-scan
        if not claims.acquire(tar_path, owner):


            deferred.pop(tar_path, None)
            return False
        d, s, f = process_fn(tar_path, entries)
        done += d; skip += s; fail += f
        if f:
            # Keep transient failures retryable. _process_tar increments
            # .attempt; after MAX_ATTEMPT the sample is an explicit skip
            # and a subsequent clean pass can publish the tar as done.
            claims.release(tar_path, owner)
            deferred[tar_path] = (now_fn() + cooldown_s, entries)
        else:
            claims.mark_done(tar_path)
            seen_done.add(tar_path)
            deferred.pop(tar_path, None)
        if progress_cb:
            progress_cb(tar_path, done, skip, fail)
        return True

    def _drain_deferred() -> bool:

        did_local = False
        now = now_fn()
        for tar_path in [t for t, (ready, _) in deferred.items() if now >= ready]:
            item = deferred.pop(tar_path, None)
            if item is None:
                continue
            if _attempt(tar_path, item[1]):
                did_local = True
        return did_local

    while True:
        if should_stop is not None and should_stop():
            break
        remaining = []
        for te in all_tars:
            if te[0] in seen_done:
                continue
            if claims.is_done(te[0]):
                seen_done.add(te[0])
                deferred.pop(te[0], None)
                continue
            remaining.append(te)
        if not remaining:
            break
        did = False
        stopped = False
        for tar_path, entries in remaining:
            if should_stop is not None and should_stop():
                stopped = True
                break
            if _drain_deferred():
                did = True
            if tar_path in deferred:
                continue
            if _attempt(tar_path, entries):
                did = True
        if stopped:
            break
        if not did:
            # everything left is live-claimed elsewhere; wait for it to finish
            # or go stale (crash) so we can steal it. Bounded tail-idle.

            wait = claims.lease_s / 3
            if deferred:
                wait = max(1.0, min(
                    min(r for r, _ in deferred.values()) - now_fn(), wait))
            sleep_fn(wait)
    return done, skip, fail


def run_static(units, claims, process_fn, progress_cb=None, should_stop=None,
               sleep_fn=time.sleep, monotonic_fn=time.monotonic,
               logger=None) -> "tuple[int, int, int, list]":

    done = skip = fail = 0
    pending = list(units)
    cooldown_s = claims.lease_s / 3.0
    round_t0 = monotonic_fn()
    stopped = False
    for round_i in range(MAX_ATTEMPT + 1):
        if not pending or stopped:
            break
        if round_i:
            waited = monotonic_fn() - round_t0
            if waited < cooldown_s:
                if logger:
                    logger.info(f"STATIC retry round {round_i}: {len(pending)} failed "
                                f"units; waiting {cooldown_s - waited:.0f}s")
                sleep_fn(cooldown_s - waited)
        round_t0 = monotonic_fn()
        retry: list = []
        for tar_path, entries in pending:
            if claims.is_done(tar_path):
                continue
            if should_stop is not None and should_stop():


                if logger:
                    logger.info("stop marker found; accepting no new units")
                stopped = True
                break
            d, s, f = process_fn(tar_path, entries)
            done += d; skip += s; fail += f
            if f:
                retry.append((tar_path, entries))
            else:
                claims.mark_done(tar_path)
            if progress_cb:
                progress_cb(tar_path, done, skip, fail)
        pending = retry
    return done, skip, fail, pending


# ---------------------------------------------------------------------------
# Worker entry: build pipeline, then static-shard or dynamic-claim
# ---------------------------------------------------------------------------

def worker_main(args) -> None:
    sys.path.insert(0, HERE)
    from pipeline.io.tar_source import (
        resolve_snapshot, shard_ordered, all_tars_ordered, ClaimDir)
    from pipeline.builder import build_pipeline
    from pipeline.config import PipelineConfig
    from utils.logger import Logger

    logger = Logger.get_logger()
    # Cap torch CPU intra-op threads (it ignores OMP_NUM_THREADS by default and
    # would grab all cores). With many workers per node this is what causes CPU
    # oversubscription / thrashing on the diar-clustering & decode stages.
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "12")))
    cfg = PipelineConfig.load(args.config_path)
    logger.info(f"[w{args.worker_id}] building pipeline...")
    pipe = build_pipeline(cfg, "cuda")
    pipe.warmup()

    snapshot = resolve_snapshot(args.metadata)
    out_root = args.output_root
    wid = args.worker_id


    stopped_by_signal = False
    ns = f"n{args.node_id}_w{wid}"
    failed_log = Path(out_root) / f"_failed_{ns}.jsonl"
    prog = Path(out_root) / f"_worker_{ns}.progress"
    tmp_dir = f"{args.local_tmp}/w{wid}"
    # Clear stale temp audio left by a hard kill (OOM / TaskStop bypasses the
    # per-sample finally cleanup). These live in the configured scratch directory and a leaked
    # 96k/5h file is GB-sized, so wipe this worker's own tmp on every start.
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)


    if args.source == "loose":
        if not args.raw_root:
            raise SystemExit("--source loose requires --raw-root")
        from pipeline.io.loose_source import LooseReader, all_units_ordered
        ordered_units = all_units_ordered(snapshot, args.raw_root)
        make_reader = lambda unit: LooseReader(args.raw_root, tmp_dir)  # noqa: E731
    else:
        ordered_units = all_tars_ordered(snapshot)
        make_reader = None      # _process_tar defaults to TarReader


    from pipeline.io import work_queue


    base = args.output_root if args.batch_recordings > 0 else ""
    if base:
        from pipeline.io import batches
        batches.init(base)
        logger.info(f"[w{wid}] batch mode: base={base}, "
                    f"{args.batch_recordings} recordings per batch")

    _root_ready: set[str] = set()

    def out_root_for(tar_path, entries) -> str:

        if not base:
            return out_root
        bi = batches.assign(base, tar_path, len(entries), args.batch_recordings)
        r = batches.batch_root(base, bi)
        if r not in _root_ready:
            os.makedirs(r, exist_ok=True)
            if args.queue:
                work_queue.enable(r)
            _root_ready.add(r)
            logger.info(f"[w{wid}] assigned to batch {batches.batch_name(bi)}")
        return r


    _queue_on: dict[str, bool] = {}

    def queue_on(root: str) -> bool:
        if root not in _queue_on:
            _queue_on[root] = bool(args.queue) and work_queue.is_enabled(root)
            logger.info(f"[w{wid}] phase2 work-queue @{root}: "
                        f"{'ON' if _queue_on[root] else 'off (scan)'}")
        return _queue_on[root]

    def proc(tar_path, entries, heartbeat=None):
        root = out_root_for(tar_path, entries)

        def publish(sid: str, long: bool = False) -> None:
            work_queue.publish(root, sid, short=True, long=long)

        return _process_tar(pipe, tar_path, entries, root, failed_log,
                            wid, tmp_dir, logger, heartbeat,
                            make_reader=make_reader,
                            publish=publish if queue_on(root) else None)

    # Progress is refreshed per-sample but THROTTLED to ~1/min: a worker shows
    # up in monitoring within a minute (not only after finishing a whole tar,
    # which can take ~1h) and its ts stays fresh enough for stale-worker
    # detection. During a tar the counts reflect completed tars + current tar.
    state = {"done": 0, "skip": 0, "fail": 0, "tar": "-"}
    _last_write = [0.0]

    def write_prog(force=False):
        now = time.time()
        if not force and now - _last_write[0] < 60.0:
            return
        _last_write[0] = now
        prog.write_text(json.dumps({**state, "ts": now}))


    stop_file = Path(args.claims_dir or out_root) / "_BUDGET_STOP"


    wstop_file = Path(out_root) / f"_STOP_n{args.node_id}_W{wid}"

    def budget_stop() -> bool:

        nonlocal stopped_by_signal
        if stop_file.exists() or wstop_file.exists():
            stopped_by_signal = True
            return True
        return False

    if args.static_shard:


        static_claims = ClaimDir(out_root, lease_s=args.lease_s,
                                 claims_dir=args.claims_dir or None)
        weights = ([float(x) for x in args.worker_weights.split(",")]
                   if args.worker_weights else None)
        my = shard_ordered(ordered_units, wid, args.num_workers, weights)
        logger.info(f"[w{wid}/{args.num_workers}] STATIC {len(my)} units, "
                    f"{sum(len(es) for _, es in my)} samples")

        def static_sample(t):
            state["tar"] = os.path.basename(t)
            write_prog()

        def static_progress(tar_path, d, s, f):
            state.update(done=d, skip=s, fail=f,
                         tar=os.path.basename(tar_path))
            write_prog(force=True)

        done, skip, fail, pending = run_static(
            my, static_claims,
            process_fn=lambda t, es: proc(t, es, lambda: static_sample(t)),
            progress_cb=static_progress, should_stop=budget_stop, logger=logger)


        if stopped_by_signal:
            logger.info(f"[w{wid}] stop marker found; STATIC stops after this unit")
        if pending:


            logger.error(
                f"[w{wid}] STATIC: {len(pending)} units failed after "
                f"{MAX_ATTEMPT + 1} attempts; no .done marker, so their batches "
                f"cannot seal and need review: "
                + ", ".join(os.path.basename(t) for t, _ in pending[:5])
                + (" ..." if len(pending) > 5 else ""))
    else:
        all_tars = ordered_units
        claims = ClaimDir(out_root, lease_s=args.lease_s,
                          claims_dir=args.claims_dir or None)
        owner = ClaimDir.owner_id(wid)
        logger.info(f"[w{wid}] DYNAMIC claim over {len(all_tars)} units "
                    f"(lease={args.lease_s}s, owner={owner})")

        def dyn_sample(t):
            claims.heartbeat(t, owner)            # lease (self-throttled)
            state["tar"] = os.path.basename(t)
            write_prog()                          # progress (throttled 60s)

        def tar_done(tar_path, done, skip, fail):
            state.update(done=done, skip=skip, fail=fail,
                         tar=os.path.basename(tar_path))
            write_prog(force=True)
        done, skip, fail = run_dynamic(
            all_tars, claims, owner,
            process_fn=lambda t, es: proc(t, es, lambda: dyn_sample(t)),
            progress_cb=tar_done, should_stop=budget_stop)


        if stopped_by_signal:
            which = (wstop_file.name if wstop_file.exists()
                     else "_BUDGET_STOP" if stop_file.exists() else "cleared marker")
            logger.info(f"[w{wid}] {which}; exiting after this unit")

    logger.info(f"[w{wid}] FINISHED done={done} skip={skip} fail={fail}")
    if stopped_by_signal:


        # classify_worker_exits.
        raise SystemExit(EXIT_STOPPED)


# ---------------------------------------------------------------------------
# Launcher: spawn one worker subprocess per GPU
# ---------------------------------------------------------------------------

def clear_stop_sentinels(out_root: str, node_id: int) -> "list[str]":

    import glob as _glob
    pat = os.path.join(out_root, f"_STOP_n{node_id}_W*")
    cleared = []
    for f in sorted(_glob.glob(pat)):
        try:
            os.unlink(f)
            cleared.append(os.path.basename(f))
        except FileNotFoundError:
            pass
        except OSError as ex:
            print(f"[launch] cannot remove stop marker {os.path.basename(f)} "
                  f"({type(ex).__name__}: {ex}); the worker would exit immediately")
    return cleared


def claims_seed_warning(n_done: int, n_units: int) -> "str | None":

    if n_done == 0 and n_units >= 1000:
        return (
            f"[launch] warning: claims table is empty, but metadata contains "
            f"{n_units} units.\n"
            f"[launch] Previously completed units would be processed again.\n"
            f"[launch] For resume or incremental runs, set CLAIMS_DIR to the "
            f"existing claims directory.\n"
            f"[launch] Ignore this warning for a new corpus. Continuing in 5 seconds..."
        )
    return None


def launch(args) -> int:
    gpus = [int(g) for g in str(args.gpus).split(",") if g.strip() != ""]
    if not gpus:
        raise ValueError("--gpus must list at least one GPU id")
    if args.nodes < 1:
        raise ValueError(f"--nodes must be >= 1 (got {args.nodes})")
    if not (0 <= args.node_id < args.nodes):
        raise ValueError(f"--node-id {args.node_id} outside --nodes={args.nodes}")
    num_local = len(gpus)
    node_worker_counts = (
        [int(x) for x in args.node_worker_counts.split(",")]
        if args.node_worker_counts else None)
    if node_worker_counts is not None:
        if len(node_worker_counts) != args.nodes or any(n < 1 for n in node_worker_counts):
            raise ValueError("--node-worker-counts must contain one positive count per node")
        if node_worker_counts[args.node_id] != num_local:
            raise ValueError(
                f"this node lists {num_local} GPUs but --node-worker-counts says "
                f"{node_worker_counts[args.node_id]} for node {args.node_id}")
    elif args.static_shard and args.nodes > 1:
        raise ValueError(
            "multi-node --static-shard requires --node-worker-counts (for example "
            "8,6); deriving global ids from each node's local GPU count creates "
            "overlapping/missing shards on heterogeneous nodes")
    counts = node_worker_counts or [num_local] * args.nodes
    global_num_workers = sum(counts)
    worker_id_base = sum(counts[:args.node_id])
    os.makedirs(args.output_root, exist_ok=True)

    # Default = DYNAMIC pull-based claiming: workers grab the earliest free
    # tar off the shared NAS, so a faster machine self-balances with NO preset
    # ratio (and crashed tars get re-claimed after the lease). --static-shard
    # falls back to fixed weighted sharding (needs --node-speeds calibration).
    weights_arg = ",".join(["1.0"] * global_num_workers)
    if args.static_shard:
        if args.node_speeds:
            speeds = [float(x) for x in args.node_speeds.split(",")]
            if len(speeds) != args.nodes:
                raise ValueError(f"--node-speeds has {len(speeds)} entries, "
                                 f"expected --nodes={args.nodes}")
        else:
            speeds = [1.0] * args.nodes
        # Each GPU on a node inherits its node's speed. ``counts`` makes the
        # flattened schedule correct for heterogeneous 8+6 style topologies.
        weights = [speeds[n] for n in range(args.nodes) for _ in range(counts[n])]
        weights_arg = ",".join(str(w) for w in weights)
        share = (counts[args.node_id] * speeds[args.node_id]
                 / sum(counts[n] * speeds[n] for n in range(args.nodes)))
        print(f"[launch] STATIC shard: node-speeds={speeds} → "
              f"node {args.node_id} takes ~{share*100:.0f}% of tars")
    else:
        if args.node_speeds:
            print("[launch] note: --node-speeds ignored in DYNAMIC mode "
                  "(claiming auto-balances; use --static-shard to force a ratio)")
        print(f"[launch] DYNAMIC claim mode (lease={args.lease_s}s, self-balancing)")


    sys.path.insert(0, HERE)
    from pipeline.io import nodes
    node_owner = nodes.register(
        args.output_root, args.node_id, lease_s=args.node_lease_s,
        expected_nodes=args.nodes)


    _cleared = clear_stop_sentinels(args.output_root, args.node_id)
    if _cleared:
        print(f"[launch] removed {len(_cleared)} stale stop markers: "
              f"{' '.join(_cleared)}", flush=True)


    sys.path.insert(0, HERE)
    from pipeline.io import work_queue
    if args.queue and args.batch_recordings <= 0:
        work_queue.enable(args.output_root)
        print(f"[launch] phase2 work-queue enabled at "
              f"{work_queue.queue_root(args.output_root)}")


    sealer_stop = None
    if args.batch_recordings > 0:
        from pipeline.io import batches
        batches.init(args.output_root)
        claims_root = args.claims_dir or os.path.join(args.output_root, "_claims")
        sealer_owner = f"{socket.gethostname()}:{os.getpid()}"

        def _terminal(tar_name: str) -> bool:
            return os.path.exists(os.path.join(claims_root, tar_name + ".done"))

        def _seal_sweep() -> int:


            if not batches.sealer_lease(args.output_root, owner=sealer_owner):
                return 0
            n = 0
            for i in batches.all_batches(args.output_root):
                if batches.is_sealed(args.output_root, i):
                    continue
                if batches.try_seal(args.output_root, i, _terminal):
                    n += 1
                    print(f"[launch] batch {batches.batch_name(i)} sealed")
            return n

        import threading
        sealer_stop = threading.Event()

        def _sealer():
            while not sealer_stop.wait(args.seal_interval_s):
                try:
                    _seal_sweep()
                except Exception as ex:
                    print(f"[launch] seal sweep error: {type(ex).__name__}: {ex}")

        threading.Thread(target=_sealer, daemon=True).start()
        print(f"[launch] batch mode: {args.batch_recordings} recordings per batch; "
              f"sealing check every {args.seal_interval_s}s")

    print(f"[launch] node {args.node_id}/{args.nodes}, gpus={gpus}, "
          f"global_workers={global_num_workers}, out={args.output_root}")

    # Pre-build the compact source-index ONCE per node before fanning out
    # workers. Otherwise all N per-GPU workers cold-start on the same missing
    # cache and each re-parse the multi-GB metadata / re-walk the raw tree (an
    # 8x startup NFS storm). Best-effort: on error workers build it themselves.
    n_units = 0
    try:
        sys.path.insert(0, HERE)
        from pipeline.io.tar_source import resolve_snapshot, all_tars_ordered
        t0 = time.time()
        if args.source == "loose":
            from pipeline.io.loose_source import load_or_build_index
            _units, stats = load_or_build_index(
                resolve_snapshot(args.metadata), args.raw_root)
            n_units = len(_units)
            print(f"[launch] loose-index ready in {time.time()-t0:.1f}s: {stats}")
        else:
            n_units = len(all_tars_ordered(resolve_snapshot(args.metadata)))
            print(f"[launch] tar-index ready ({n_units} tars) "
                  f"in {time.time()-t0:.1f}s")
    except Exception as ex:  # noqa: BLE001 - never block the run on the cache
        print(f"[launch] source-index prebuild skipped: {type(ex).__name__}: {ex}")


    try:
        from pipeline.io.tar_source import ClaimDir
        cd = ClaimDir(args.output_root, claims_dir=args.claims_dir or None)
        n_done = len(cd.done_names())
        print(f"[launch] claims table {cd.dir}: {n_done} .done markers / "
              f"{n_units} metadata units")
        warn = claims_seed_warning(n_done, n_units)
        if warn:
            print(warn, flush=True)
            time.sleep(5)
    except Exception as ex:
        print(f"[launch] skipped claims-table safety check: {type(ex).__name__}: {ex}")

    procs = []
    for local_id, gpu in enumerate(gpus):
        gwid = worker_id_base + local_id
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            sys.executable, os.path.abspath(__file__), "--worker",
            "--worker-id", str(gwid),
            "--node-id", str(args.node_id),
            "--num-workers", str(global_num_workers),
            "--worker-weights", weights_arg,
            "--lease-s", str(args.lease_s),
            "--config-path", args.config_path,
            "--metadata", args.metadata,
            "--output-root", args.output_root,
            "--local-tmp", args.local_tmp,
            "--source", args.source,
        ] + (["--raw-root", args.raw_root] if args.raw_root else []) \
          + (["--claims-dir", args.claims_dir] if args.claims_dir else []) \
          + (["--static-shard"] if args.static_shard else []) \
          + (["--queue"] if args.queue else []) \
          + (["--batch-recordings", str(args.batch_recordings)]
             if args.batch_recordings > 0 else [])
        log_path = os.path.join(args.output_root, f"_worker_n{args.node_id}_gpu{gpu}.log")
        lf = open(log_path, "a", buffering=1)
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        procs.append(p)
        print(f"[launch] -> gpu={gpu} worker={gwid} pid={p.pid} log={log_path}")


    codes = [p.wait() for p in procs]

    if sealer_stop is not None:
        sealer_stop.set()

    verdict, detail = classify_worker_exits(codes)
    if verdict != "done":
        if args.batch_recordings > 0:
            batches.release_sealer_lease(args.output_root, sealer_owner)
        nodes.unregister(args.output_root, args.node_id, node_owner)
        if verdict == "failed":
            print(f"[launch] workers exited {codes}; NOT writing phase1 done marker")
            return max(detail)


        print(f"[launch] interrupted (exit codes {codes}): leaving current batch "
              f"open and not writing _PHASE1_DONE_n{args.node_id}")
        return 0


    if args.batch_recordings > 0:
        cur = batches.current(args.output_root)
        batches.close_current(args.output_root)
        print(f"[launch] phase1 finalization: closing batch {batches.batch_name(cur)}")


        deadline = time.time() + max(
            batches.SEAL_GRACE_S, batches.SEALER_LEASE_S) + 120
        while True:
            _seal_sweep()
            left = [i for i in batches.all_batches(args.output_root)
                    if not batches.is_sealed(args.output_root, i)]
            if not left:
                break
            if time.time() >= deadline:


                print(f"[launch] final sealing timed out; unsealed batches: "
                      f"{[batches.batch_name(i) for i in left]} "
                      f"(another active node may seal them during finalization)")
                break
            time.sleep(15)

    # All local workers exited cleanly. Only then publish completion.
    nodes.mark_done(args.output_root, args.node_id, node_owner)
    print(f"[launch] all workers finished (exit codes {codes}); "
          f"wrote _PHASE1_DONE_n{args.node_id}")
    if args.batch_recordings > 0:
        batches.release_sealer_lease(args.output_root, sealer_owner)


    nodes.unregister(args.output_root, args.node_id, node_owner)
    return 0


def nodes_lease_default() -> float:

    sys.path.insert(0, HERE)
    from pipeline.io import nodes
    return nodes.LEASE_S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help="internal: run as a worker")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--worker-weights", default="",
                    help="internal: comma-separated per-worker weights")
    ap.add_argument("--config-path", default="config.json")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--local-tmp", default=".continuo-tmp")
    ap.add_argument("--source", choices=("tar", "loose"), default="tar",
                    help="tar: metadata points to tar members (default); loose: "
                         "metadata is an identity list and audio is under --raw-root")
    ap.add_argument("--raw-root", default="",
                    help="root directory of loose audio files")
    ap.add_argument("--claims-dir", default="",
                    help="claims table directory (default <output-root>/_claims). "
                         "Point nodes at the same shared directory to avoid duplicate "
                         "work when they write separate output trees. Dynamic mode only")
    ap.add_argument("--batch-recordings", type=int, default=0,
                    help="recordings per sealed batch; 0 disables batch mode. "
                         "Each batch gets its own output tree beneath --output-root. "
                         "Phase 1 writes _SEALED after its members finish; phase 2 "
                         "writes _PHASE2_DONE after its backlog is empty")
    ap.add_argument(
        "--node-lease-s", type=float, default=nodes_lease_default(),
        help=("node-id ownership lease in seconds. An active heartbeat prevents "
              "another launcher from using the same id; stale registrations expire. "
              "RUN.json keeps the fixed topology for phase-two completion."))
    ap.add_argument("--seal-interval-s", type=float, default=300.0,
                    help="batch-mode sealing check interval in seconds")
    ap.add_argument("--queue", action="store_true",
                    help="enable the optional phase-two work queue for incremental "
                         "discovery; see pipeline/io/work_queue.py")
    ap.add_argument("--gpus", default="0", help="comma-separated local GPU ids")
    ap.add_argument("--nodes", type=int, default=1,
                    help="number of machines in the cluster, used for static sharding")
    ap.add_argument("--node-id", type=int, default=0,
                    help="unique machine id. Worker names and completion markers "
                         "derive from it; duplicate active ids are rejected")
    ap.add_argument(
        "--node-worker-counts", default="",
        help=("comma-separated worker counts by node-id, for example '8,6'. "
              "Required for multi-node static sharding and optional for dynamic claims."))
    ap.add_argument("--node-speeds", default="",
                    help="STATIC mode only: relative speed per node-id, e.g. "
                         "'1,3' if node 1 is 3x faster. Pass the SAME value on "
                         "every machine. Ignored in dynamic (default) mode.")
    ap.add_argument("--static-shard", action="store_true",
                    help="fixed weighted sharding instead of dynamic claiming "
                         "(fallback; dynamic self-balances without a preset ratio)")
    ap.add_argument("--lease-s", type=float, default=1800.0,
                    help="dynamic mode: tar-claim lease seconds; a lock older "
                         "than this is stale (crashed) and gets re-claimed")
    args = ap.parse_args()

    if args.worker:
        worker_main(args)
    else:
        sys.exit(launch(args))


if __name__ == "__main__":
    main()
