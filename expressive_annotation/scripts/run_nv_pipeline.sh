#!/usr/bin/env bash
# Nonverbal-vocalization annotation for a whole corpus, one command.
#
#   scripts/run_nv_pipeline.sh runs/short-all/final.jsonl runs/nv-all
#
# Six stages, in the order that matters:
#
#   prep    bake tar_offset/tar_size in, grouped by tar        cheap, once
#   panns   score every clip, route the sung ones out          multi-GPU, supervised
#   nvasr   continuo-nv over the speech manifest only                multi-GPU, supervised
#   verify  transcript consistency and PANNs on surviving tags   one GPU
#   final   fold back to one record per corpus row             the deliverable
#   stats   what came out
#
# PANNs runs first because sung audio can make NVASR emit spurious speech-event tags.
# Filtering such clips before NVASR also avoids unnecessary inference.
#
# Every stage is resumable and skipped when its output is already complete, so a rerun
# after any interruption picks up where it stopped. STAGES= runs a subset:
#
#   STAGES=verify scripts/run_nv_pipeline.sh corpus.jsonl runs/nv-all   # just re-verify
#   STAGES=panns,nvasr ...                                              # skip verify
#   DRY_RUN=1 ...                                                       # plan only
#
# **Both GPU passes are supervised**, because at corpus scale each of these happened:
#
#   * A worker dies. rc=137 may indicate an OOM kill, so the supervisor detects
#     failures and reports progress before retrying.
#   * A worker hangs. The corpus is on a shared filesystem; a worker stuck in a read
#     never exits and every GPU idles behind it. The watchdog kills a run whose output
#     has not grown for STALL_KILL seconds — growth in bytes via stat, which is free,
#     where counting rows rereads gigabytes.
#   * Two fleets on one work dir interleave writes into the same shard files. The flock
#     refuses the second start, and children inherit the fd, so it holds even if the
#     supervisor dies and leaves its fleet running.
#   * Killing the supervisor used to orphan the fleet, so each fleet runs in its own
#     process group and TERM/INT/EXIT forward a group kill.
#
# Restarts are conditional on progress: free when --resume skips finished ids, but
# restarting forever into the same wall is not. No new rows MAX_STALL times means a
# human should read the log.
#
# Outputs, all under <out-dir>:
#
#   manifest_speech.jsonl / manifest_sung.jsonl   the routing, whole input lines
#   panns.jsonl                                   the evidence: top-3 classes per clip
#   nv-speech.jsonl                               continuo-nv over the speech manifest
#   nv-verified.jsonl                             + nv_verified / nv_rejected, per segment
#   final.jsonl                                   **read this one**: your input rows with
#                                                 nv_tags / nv_counts / n_nv / nv_spans /
#                                                 nv_segments / nv_sung_segments added
#   final-nv-only.jsonl                           the same rows, only those that carry an
#                                                 event — the subset worth training on
#
# Env: PY_NV, CONTINUO_EXPRESSIVE_NVBENCH_REPO, CONTINUO_EXPRESSIVE_NVASR_DIR, CKPT_ROOT, AUDIOLDM_ROOT, CORPUS/CONTINUO_EXPRESSIVE_TAR_DIR,
#      GPUS, BUSY_MIB, FORCE,
#      STAGES, DRY_RUN, NO_PANNS_VERIFY,
#      PANNS_PER_GPU (3) PANNS_DECODERS (32) PANNS_BATCH (1) TOPK (3),
#      NV_PER_GPU (1) NV_DECODERS (32) NV_BATCH (8) LANGUAGE CARRY TIMESTAMPS
#      SUSPECT_BELOW, VERIFY_WORKERS (32), EXCLUDE_IDS DROP_TAGS,
#      SEGMENT_SHARDS (12) PREPARE_WORKERS (8) METAINFO_DIR MAX_SECONDS,
#      MAX_DEATHS (10) MAX_STALL (3) COOLDOWN (60) PAUSE (10) STALL_KILL (1800)
#      WATCH_EVERY (15) HEARTBEAT_EVERY (300).
set -uo pipefail

ROOT="${CONTINUO_EXPRESSIVE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ ! -d "$ROOT/tools" ] || [ ! -d "$ROOT/continuo_expressive" ]; then
  echo "[fatal] $ROOT is not the repo root (no tools/ or continuo_expressive/); set CONTINUO_EXPRESSIVE_ROOT" >&2
  exit 1
fi
cd "$ROOT" || exit 1

PY_NV=${PY_NV:-$(command -v python)}
export CONTINUO_EXPRESSIVE_NVBENCH_REPO="${CONTINUO_EXPRESSIVE_NVBENCH_REPO:-$ROOT/third_party/NV-Bench}"
export CONTINUO_EXPRESSIVE_NVASR_DIR="${CONTINUO_EXPRESSIVE_NVASR_DIR:-$ROOT/checkpoints/Multilingual-NVASR}"
CKPT_ROOT=${CKPT_ROOT:-$CONTINUO_EXPRESSIVE_NVBENCH_REPO}
AUDIOLDM_ROOT=${AUDIOLDM_ROOT:-$ROOT/third_party/audioldm_eval}
CORPUS="${CORPUS:-}"
[ -d "$CORPUS" ] && export CONTINUO_EXPRESSIVE_TAR_DIR="${CONTINUO_EXPRESSIVE_TAR_DIR:-$CORPUS}"
# LONG=1 annotates Continuo `type=long` containers instead of short clips, and then the
# first argument is a tar glob (or a list, TARS_FROM) rather than a manifest. The unit
# has to be the segment, not the recording: containers can be long and NVASR
# is an utterance model, but more to the point the sung test is calibrated on singing
# lasting the whole clip — over minutes that premise is false and the rule stops meaning
# anything. Segments are named, never cut: every row carries rel_start/rel_end and its
# container's tar member, so the passes slice out of the corpus tars and no disk is
# spent on separate files for every segment. The cost is that a container is decoded once per segment.
LONG=${LONG:-}
# VIEW=dialogue reads Continuo `type=dialogue` containers instead, the same three things
# changing as in the long-audio workflow: the metainfo lives in `dialog_dedup/`, the cut is
# told which container type to take, and each segment carries the speaker that said it.
# Everything after that is identical — a turn is a segment like any other, and the sung
# filter and NVASR do not care whose voice it is. What the speaker buys is the fold:
# merge_nv puts the tags on the voice as well as on the file.
VIEW=${VIEW:-long}
case "$VIEW" in
  long)     META_SUB=long;         EXTRA_CARRY="" ;;
  dialogue) META_SUB=dialog_dedup; EXTRA_CARRY=",speaker,turn" ;;
  *) echo "[fatal] VIEW=$VIEW; expected long or dialogue" >&2; exit 1 ;;
esac
PY_PREPARE=${PY_PREPARE:-$PY_NV}
METAINFO_DIR=${METAINFO_DIR:-none}
TARS_FROM=${TARS_FROM:-}
SEGMENTS_MODE=${SEGMENTS_MODE:-union}
MAX_SECONDS=${MAX_SECONDS:-300}
WINDOW_SECONDS=${WINDOW_SECONDS:-15}
MIN_SECONDS=${MIN_SECONDS:-0.5}
LANGUAGES=${LANGUAGES:-}
PREPARE_WORKERS=${PREPARE_WORKERS:-8}
SEGMENT_SHARDS=${SEGMENT_SHARDS:-12}

if [ -n "$LONG" ]; then
  # No prep: the segments stage already bakes tar_offset in and emits its rows grouped by
  # tar, so prep would only order them by offset *within* a tar — and it does that by
  # holding every row in memory at once, to
  # buy back seeks inside a file the loader keeps open anyway. Add it explicitly if you
  # ever want it.
  STAGES=${STAGES:-segments,panns,nvasr,verify,final,stats}
  # A short clip *is* the unit, so an NV row needs nothing to place it. A segment does:
  # without parent_id and rel_start/rel_end a tag cannot be put back on its recording's
  # timeline or grouped with its siblings, which is the whole point of annotating long
# audio.
  # ...and for a dialogue, `speaker`/`turn` as well: without the turn tag a laugh cannot
  # be put on the voice that made it, and merge_nv's per-speaker fold has nothing to key
  # on. It says so rather than emitting empty per-speaker fields, but the fix is here.
  CARRY=${CARRY:-txt,lang,duration,source_tar,source_member,parent_id,parent_duration,rel_start,rel_end,seg_source,wav_path,carrier_start_samples,carrier_end_samples,sample_rate$EXTRA_CARRY}
else
  STAGES=${STAGES:-prep,panns,nvasr,verify,final,stats}
  CARRY=${CARRY:-txt,lang,duration,source_tar,source_member,wav_path,carrier_start_samples,carrier_end_samples,sample_rate}
fi

# Size worker counts from peak inference memory, not idle model memory.
# Keep one NVASR worker per GPU to prevent OOM and incomplete batches.
# batch=1 for PANNs is not a performance choice: batching pads every clip to the longest
# in its batch and the zeros reshuffle low-rank scores — 3% of verdicts flipped.
PANNS_PER_GPU=${PANNS_PER_GPU:-3}; PANNS_DECODERS=${PANNS_DECODERS:-32}
PANNS_BATCH=${PANNS_BATCH:-1};     TOPK=${TOPK:-3}
NV_PER_GPU=${NV_PER_GPU:-1};       NV_DECODERS=${NV_DECODERS:-32}
NV_BATCH=${NV_BATCH:-8}            # Adjust for available GPU memory.
LANGUAGE=${LANGUAGE:-auto}
# CARRY's default is set per mode above — a long-audio row needs its timeline fields.
TIMESTAMPS=${TIMESTAMPS:-}         # off: the CTC aligner cannot batch, ~6x the cost
SUSPECT_BELOW=${SUSPECT_BELOW:-0.6}
VERIFY_WORKERS=${VERIFY_WORKERS:-32}
# Clips whose tags listening has rejected, one id a line. Per clip, not per class: on the
# sample that started this list 7 of 10 Sneeze clips were wrong and 3 were right, so
# dropping the class would have thrown away the good ones with the bad. A rerun reads the
# file, so a judgement made once stays made.
EXCLUDE_IDS=${EXCLUDE_IDS:-}
DROP_TAGS=${DROP_TAGS:-}

BUSY_MIB=${BUSY_MIB:-2000};  FORCE=${FORCE:-};      DRY_RUN=${DRY_RUN:-}
MAX_DEATHS=${MAX_DEATHS:-10}
MAX_STALL=${MAX_STALL:-3};   COOLDOWN=${COOLDOWN:-60};   PAUSE=${PAUSE:-10}
STALL_KILL=${STALL_KILL:-1800}; WATCH_EVERY=${WATCH_EVERY:-15}
HEARTBEAT_EVERY=${HEARTBEAT_EVERY:-300}

say() { printf '%s  %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "${LOG:-/dev/null}"; }
die() { printf '%s  [fatal] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "${LOG:-/dev/null}" >&2; exit 1; }
want() { [[ ",$STAGES," == *",$1,"* ]]; }
rows() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
# printf "%.0f", and neither of the two obvious alternatives. Plain `print` switches to
# scientific notation for large values (OFMT is %.6g), so the result is a
# syntax error rather than a comparison; printf "%d" instead clamps at INT_MAX, so the
# value freezes at 2147483647. Either way the watchdog stops seeing growth and kills a
# healthy fleet every STALL_KILL seconds.
bytes() { stat -c %s "$@" 2>/dev/null | awk '{s+=$1} END{printf "%.0f", s+0}'; }

# ============================================================ one sharded fleet
# Re-entered as a child process group by supervise(); everything else comes from the
# environment it inherits. KIND picks the worker command and the shard-file prefix.
fleet() {
  local kind=$1 manifest=$2 work=$3 prefix per_gpu
  case $kind in
    panns) prefix=panns; per_gpu=$PANNS_PER_GPU ;;
    nvasr) prefix=nv;    per_gpu=$NV_PER_GPU ;;
    *) die "fleet: unknown kind $kind" ;;
  esac
  mkdir -p "$work"
  LOG=$work/shards.log

  pick_gpus
  local nwork=$((NGPU * per_gpu))
  local total; total=$(rows "$manifest")
  say "$kind: manifest=$manifest rows=$total gpus=$GPUS$AUTO ngpu=$NGPU per_gpu=$per_gpu workers=$nwork"

  # One worker: run it, and restart it as long as it is still writing rows. Every
  # restart is free because --resume drops the finished ids before loading the model.
  worker() {
    local gpu=$1 shard_i=$2 n out log deaths=0
    n=$(printf %02d "$shard_i")
    out=$work/$prefix$n.jsonl log=$work/$prefix$n.log
    while :; do
      local before after rc
      before=$(rows "$out")
      case $kind in
        panns)
          CUDA_VISIBLE_DEVICES=$gpu "$PY_NV" tools/panns_filter.py score \
            --manifest "$manifest" --shard "$shard_i/$nwork" --out "$out" --resume \
            --done-from "$work/$prefix[0-9]*.jsonl" \
            --device cuda:0 --batch-size "$PANNS_BATCH" --workers "$PANNS_DECODERS" \
            --topk "$TOPK" --ckpt-root "$CKPT_ROOT" --audioldm-root "$AUDIOLDM_ROOT" \
            >> "$log" 2>&1 ;;
        nvasr)
          local ts=(); [ -n "$TIMESTAMPS" ] && ts=(--timestamps)
          CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 \
            "$PY_NV" -m continuo_expressive.cli.nv \
            --manifest "$manifest" --shard "$shard_i/$nwork" --out "$out" --resume \
            --done-from "$work/$prefix[0-9]*.jsonl" \
            --batch-size "$NV_BATCH" --workers "$NV_DECODERS" --device cuda:0 \
            --language "$LANGUAGE" --carry "$CARRY" \
            --suspect-below "$SUSPECT_BELOW" "${ts[@]}" >> "$log" 2>&1 ;;
      esac
      rc=$?
      after=$(rows "$out")
      [ "$rc" = 0 ] && { echo "[gpu $gpu] done at $after rows" >> "$log"; return 0; }
      deaths=$((deaths + 1))
      # rc 137 is SIGKILL; check system memory for a possible OOM kill.
      local why=""; [ "$rc" = 137 ] && why=" (SIGKILL — check free(1) for the OOM killer)"
      echo "[gpu $gpu] exited rc=$rc$why at $after rows (was $before); restart #$deaths" >> "$log"
      if [ "$after" = "$before" ]; then
        [ "$deaths" -ge "$MAX_DEATHS" ] && {
          echo "[gpu $gpu] giving up after $deaths restart(s) with no progress" >> "$log"
          return 1; }
        sleep 60
      else
        deaths=0
        sleep 10
      fi
    done
  }

  # One line every HEARTBEAT_EVERY with the counts that move. Counting rows rereads
  # every output file, which at corpus scale is GBs of the filesystem the decode threads
  # are already waiting on — so beat rarely rather than estimate. supervise() answers
  # "is it alive" from byte growth every WATCH_EVERY seconds anyway.
  heartbeat() {
    local last=0 now rate
    while :; do
      now=$(cat "$work"/$prefix[0-9]*.jsonl 2>/dev/null | wc -l)
      rate=$(( (now - last) / HEARTBEAT_EVERY ))
      printf '%s  rows=%s/%s (%s%%) %s clips/s gpu=%s\n' "$(date -u +%FT%TZ)" \
        "$now" "$total" "$(( total ? 100 * now / total : 0 ))" "$rate" \
        "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd/)" \
        >> "$work/progress.log"
      last=$now
      sleep "$HEARTBEAT_EVERY"
    done
  }

  heartbeat & local hb=$!
  trap 'kill '"$hb"' 2>/dev/null' EXIT
  local i rc=0
  for ((i = 0; i < nwork; i++)); do worker "${GPU_LIST[$((i % NGPU))]}" "$i" & done
  say "$kind: $nwork worker(s) on GPU(s) ${GPU_LIST[*]}; logs $work/${prefix}NN.log"
  local job
  for job in $(jobs -p); do [ "$job" = "$hb" ] || wait "$job" || rc=1; done
  kill "$hb" 2>/dev/null
  return "$rc"
}

# GPUS explicit, or every card idle enough to take one. Which cards to use is an input,
# not 0..n-1: this box runs training on the same GPUs, and a run that assumed the whole
# machine would land on top of it. Card count is not stable here either — it was 8, then
# 3, then 4 within one day — so the shard index comes from a card's position in the
# list, not its number, and the output files keep their names when the list changes.
AUTO=""
pick_gpus() {
  if [ -z "${GPUS:-}" ]; then
    GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null |
           awk -F', *' -v b="$BUSY_MIB" '$2 <= b {printf "%s%s", (n++ ? "," : ""), $1}')
    [ -n "$GPUS" ] || die "every GPU is above ${BUSY_MIB} MiB; name cards with GPUS=... FORCE=1"
    AUTO=" (auto-selected)"
  else
    local want_g used busy=()
    IFS=, read -ra want_g <<< "$GPUS"
    for g in "${want_g[@]}"; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null) \
        || die "no such GPU: $g"
      [ "${used:-0}" -gt "$BUSY_MIB" ] && busy+=("gpu$g=${used}MiB")
    done
    if [ ${#busy[@]} -gt 0 ]; then
      [ -n "$FORCE" ] || die "these cards are in use: ${busy[*]} — pick others, or FORCE=1"
      say "[warn] sharing busy card(s): ${busy[*]}"
    fi
  fi
  IFS=, read -ra GPU_LIST <<< "$GPUS"
  NGPU=${#GPU_LIST[@]}
}

# ---- internal re-entry: `$0 --fleet <kind> <manifest> <workdir>` is one fleet in its
# own process group, started by supervise() below. Not a public entry point.
if [ "${1:-}" = "--fleet" ]; then
  fleet "$2" "$3" "$4"
  exit $?
fi

# ============================================================ supervisor
# Keep a fleet alive across a multi-hour pass without turning a real fault into a crash
# loop. Returns 0 when the fleet completed, 1 when it stopped making progress.
supervise() {
  local kind=$1 manifest=$2 work=$3 prefix
  case $kind in panns) prefix=panns ;; nvasr) prefix=nv ;; esac
  mkdir -p "$work" || die "cannot create $work"

  exec 9>"$work/.supervise.lock"
  flock -n 9 || die "another supervisor (or its surviving fleet) holds $work/.supervise.lock"

  local CHILD=
  cleanup() {
    [ -n "$CHILD" ] || return 0
    kill -- -"$CHILD" 2>/dev/null && sleep 2
    kill -KILL -- -"$CHILD" 2>/dev/null
    CHILD=
  }
  trap cleanup EXIT
  trap 'say "stopped by SIGTERM"; exit 143' TERM
  trap 'say "stopped by SIGINT"; exit 130' INT

  # One attempt: start the fleet in its own group, watch it, reap it. 124 means the
  # watchdog killed a run that was alive but no longer writing.
  run_once() {
    setsid "$0" --fleet "$kind" "$manifest" "$work" >> "$work/run.out" 2>&1 &
    CHILD=$!
    local quiet=0 last_b b rc
    last_b=$(bytes "$work"/$prefix[0-9]*.jsonl)
    while kill -0 "$CHILD" 2>/dev/null; do
      sleep "$WATCH_EVERY"
      b=$(bytes "$work"/$prefix[0-9]*.jsonl)
      if [ "$b" -gt "$last_b" ]; then last_b=$b quiet=0
      else quiet=$((quiet + WATCH_EVERY)); fi
      if [ "$STALL_KILL" -gt 0 ] && [ "$quiet" -ge "$STALL_KILL" ]; then
        say "$kind: no output growth in ${quiet}s but still alive — killing it (hung worker?)"
        kill -- -"$CHILD" 2>/dev/null; sleep 5
        kill -KILL -- -"$CHILD" 2>/dev/null
        wait "$CHILD" 2>/dev/null
        CHILD=
        return 124
      fi
    done
    wait "$CHILD"; rc=$?
    CHILD=
    return "$rc"
  }

  local total attempt=0 stall=0 last now rc note
  total=$(rows "$manifest")
  last=$(cat "$work"/$prefix[0-9]*.jsonl 2>/dev/null | wc -l)
  say "$kind: supervising as pid $$; $last/$total row(s) already done"
  while :; do
    attempt=$((attempt + 1))
    run_once; rc=$?
    now=$(cat "$work"/$prefix[0-9]*.jsonl 2>/dev/null | wc -l)
    if [ "$rc" = 0 ]; then
      say "$kind: fleet exited 0 at $now/$total row(s) — pass complete"
      trap - EXIT; return 0
    fi
    note=""; [ "$rc" = 124 ] && note=" (watchdog kill)"
    if [ "$now" -gt "$last" ]; then
      stall=0
      say "$kind: attempt $attempt rc=$rc$note after $((now - last)) new row(s); retry in ${PAUSE}s"
      sleep "$PAUSE"
    else
      stall=$((stall + 1))
      say "$kind: attempt $attempt rc=$rc$note with no new rows ($stall/$MAX_STALL)"
      if [ "$stall" -ge "$MAX_STALL" ]; then
        say "$kind: giving up after $stall attempt(s) with no progress. Tail of $work/run.out:"
        tail -20 "$work/run.out" | tee -a "$LOG"
        local lastlog; lastlog=$(ls -t "$work"/$prefix[0-9]*.log 2>/dev/null | head -1)
        [ -n "$lastlog" ] && { say "and of $lastlog:"; tail -20 "$lastlog" | tee -a "$LOG"; }
        trap - EXIT; return 1
      fi
      sleep $((COOLDOWN * stall))
    fi
    last=$now
  done
}

# ============================================================ the pipeline
MANIFEST=${1:?usage: run_nv_pipeline.sh <manifest.jsonl> <out-dir>}
OUT=${2:?usage: run_nv_pipeline.sh <manifest.jsonl> <out-dir>}
mkdir -p "$OUT" || exit 1
LOG="$OUT/pipeline.log"
# <out-dir>/rejected-ids.txt is the run's own list, so a rerun keeps the judgements made
# against it without anyone having to remember the flag.
[ -n "$EXCLUDE_IDS" ] || [ ! -s "$OUT/rejected-ids.txt" ] || EXCLUDE_IDS="$OUT/rejected-ids.txt"

# ---- preflight: every stage's prerequisites, before any of them runs. The verify stage
# used to discover its checkpoint was unreachable only after earlier passes completed.
# The same applies to a wrong interpreter, which surfaces per-worker and looks like a fleet
# problem rather than a one-line fix.
[ -x "$PY_NV" ] || die "interpreter missing: $PY_NV"
if [ -n "$LONG" ]; then
  # MANIFEST is a tar glob here, so it cannot be stat'd; the segments stage validates it.
  want segments || [ -s "$OUT/manifest.jsonl" ] \
    || die "LONG without the segments stage needs $OUT/manifest.jsonl from an earlier run"
  if want segments; then
    [ -x "$PY_PREPARE" ] || die "interpreter missing: PY_PREPARE=$PY_PREPARE"
    # Count the tars, do not just test the directory. A METAINFO_DIR that exists but is
    # nearly empty is the dangerous case: prepare_long falls back to each container's own
    # short view per tar, silently producing ids that no longer match the corpus.
    n_meta=$(ls "$METAINFO_DIR"/*.tar 2>/dev/null | wc -l)
    if [ "$n_meta" = 0 ] && [ "$METAINFO_DIR" != none ]; then
      die "METAINFO_DIR=$METAINFO_DIR holds no .tar — segmentation would fall back to
       each container's own short view. For VIEW=long that produces ids which no longer
       match the corpus; for VIEW=dialogue it produces **nothing at all**, because a
       dialogue sidecar's short view is empty and its turns exist only in the metainfo.
       Set METAINFO_DIR to the matching metainfo directory, or none to
       accept the fallback deliberately."
    fi
    say "segments: view=$VIEW METAINFO_DIR=$METAINFO_DIR has $n_meta metainfo tar(s)"
  fi
else
  [ -s "$MANIFEST" ] || die "empty or missing manifest: $MANIFEST"
fi
for v in PANNS_PER_GPU NV_PER_GPU MAX_STALL COOLDOWN PAUSE STALL_KILL WATCH_EVERY \
         HEARTBEAT_EVERY MAX_DEATHS; do
  case ${!v} in '' | *[!0-9]*) die "$v='${!v}' is not a non-negative integer" ;; esac
done
[ "$WATCH_EVERY" -ge 1 ] || die "WATCH_EVERY must be >= 1"
if [ -n "$LONG" ] && [ ! -d "$CORPUS" ]; then
  say "[warn] CORPUS=$CORPUS is not a directory; long-recording tar input will fail"
fi
if want nvasr; then
  [ -f "$CONTINUO_EXPRESSIVE_NVBENCH_REPO/model.py" ] || die "no model.py under CONTINUO_EXPRESSIVE_NVBENCH_REPO=$CONTINUO_EXPRESSIVE_NVBENCH_REPO"
  [ -f "$CONTINUO_EXPRESSIVE_NVASR_DIR/model.pt" ]    || die "no model.pt under CONTINUO_EXPRESSIVE_NVASR_DIR=$CONTINUO_EXPRESSIVE_NVASR_DIR"
  "$PY_NV" -c "import funasr, torch; assert torch.cuda.is_available()" 2>>"$LOG" \
    || die "$PY_NV cannot import funasr+torch with CUDA — wrong environment? see $LOG"
fi
if want panns || { want verify && [ -z "${NO_PANNS_VERIFY:-}" ]; }; then
  [ -f "$CKPT_ROOT/ckpt/Cnn14_16k_mAP=0.438.pth" ]   || die "no Cnn14 ckpt under $CKPT_ROOT/ckpt"
  [ -f "${CONTINUO_EXPRESSIVE_PANNS_LABELS:-assets/class_labels_indices.csv}" ] || die "set CONTINUO_EXPRESSIVE_PANNS_LABELS to an AudioSet labels CSV"
  "$PY_NV" -c "import sys; sys.path.insert(0,'$AUDIOLDM_ROOT')
from audioldm_eval.feature_extractors.panns.models import Cnn14" 2>>"$LOG" \
    || die "$PY_NV cannot import Cnn14 from $AUDIOLDM_ROOT — see $LOG"
fi
if [ -n "$DRY_RUN" ]; then
  pick_gpus
  say "DRY_RUN: ${LONG:+$VIEW }input=$MANIFEST out=$OUT stages=$STAGES metainfo=$METAINFO_DIR"
  say "DRY_RUN: gpus=$GPUS$AUTO -> $((NGPU * PANNS_PER_GPU)) panns / $((NGPU * NV_PER_GPU)) nvasr worker(s)"
  exit 0
fi

# A recording manifest as input means "annotate these recordings" — the corpus's own
# long-audio output (runs/long-all/final.jsonl) is one row per recording, and scoping to
# it is what keeps the NV rows joinable to the annotations already there. It is also the
# record list the final stage folds the segments back into, so it is resolved here
# rather than inside the segments stage, which STAGES= may skip.
SCOPE=""
SCOPE_KEY=""                      # set to parent_id when the scope has no source_tar
if [ -n "$LONG" ]; then
  case "$MANIFEST" in *.jsonl) SCOPE="$MANIFEST" ;; esac
else
  SCOPE="$MANIFEST"
fi

# ---- segments (long only): name every segment, cut nothing
if [ -n "$LONG" ] && want segments; then
  SEGMANIFEST="$OUT/manifest.jsonl"
  if [ -s "$SEGMANIFEST" ]; then
    say "segments: $SEGMANIFEST already has $(rows "$SEGMANIFEST") row(s), keeping it"
  else
    # The join key is (source_tar, source_member), not the recording id: ids carry a _pN
    # part number that depends on how the container was split, so they drift with the
    # segmentation while the tar member does not.
    case "$MANIFEST" in
      *.jsonl)
        [ -s "$MANIFEST" ] || die "recording manifest is empty or missing: $MANIFEST"
        SCOPE="$MANIFEST"
        tarlist="$OUT/scope-tars.txt"
        "$PY_NV" - "$MANIFEST" "$CORPUS" > "$tarlist" <<'PY' || die "reading $MANIFEST failed"
import json, os, sys
seen = set()
for line in open(sys.argv[1]):
    t = json.loads(line).get("source_tar")
    if t and t not in seen:
        seen.add(t)
        print(t if os.path.isabs(t) else os.path.join(sys.argv[2], t))
PY
        if [ -s "$tarlist" ]; then
          say "segments: $MANIFEST names $(rows "$MANIFEST") recording(s) across $(rows "$tarlist") tar(s)"
        else
          # A scope that does not name its tars — runs/dialogue-all/final.jsonl is one,
          # its records are built field by field and used to drop the provenance. Fall
          # back to the id: read every tar the metainfo covers and keep the segments
          # whose parent_id the scope names. Same result, more tars opened. TARS_FROM
          # narrows it if you only want part of the corpus.
          SCOPE_KEY=parent_id
          comm -12 <(ls "$METAINFO_DIR"/*.tar 2>/dev/null | xargs -r -n1 basename | sort -u) \
                   <(ls "$CORPUS"/*.tar 2>/dev/null | xargs -r -n1 basename | sort -u) \
            | sed "s#^#$CORPUS/#" > "$tarlist"
          [ -s "$tarlist" ] || die "$MANIFEST names no source_tar, and no tar is covered by
       both METAINFO_DIR=$METAINFO_DIR and CORPUS=$CORPUS to fall back to."
          say "segments: $MANIFEST names $(rows "$MANIFEST") recording(s) but no source_tar;
       scoping by parent_id over the $(rows "$tarlist") tar(s) the metainfo covers"
        fi
        tars_args=(--tars-from "$tarlist")
        ;;
      *)
        tarlist="$OUT/scope-tars.txt"
        ls -1 $MANIFEST > "$tarlist" 2>/dev/null
        [ -s "$tarlist" ] || die "no tar matched $MANIFEST"
        ;;
    esac
    [ -n "$TARS_FROM" ] && cp "$TARS_FROM" "$tarlist"

    # prepare_long walks tars one at a time in the parent — reading each .idx and each
    # metainfo tar is sequential, and with --manifest-only there is no decoding left to
    # hide it behind. Shard the tar list so independent reads run concurrently.
    segwork="$OUT/segments-work"
    mkdir -p "$segwork"
    n_tars=$(rows "$tarlist")
    shards=$SEGMENT_SHARDS
    [ "$shards" -gt "$n_tars" ] && shards=$n_tars
    [ "$shards" -lt 1 ] && shards=1
    say "segments: naming segments from $n_tars tar(s) in $shards shard(s), no audio cut"
    rm -f "$segwork"/tars.* "$segwork"/manifest.*.jsonl
    split -n "l/$shards" -d -a 3 "$tarlist" "$segwork/tars."
    langs=(); [ -n "$LANGUAGES" ] && langs=(--languages "$LANGUAGES")
    meta=(); [ "$METAINFO_DIR" != none ] && meta=(--metainfo-dir "$METAINFO_DIR")
    seg_rc=0 seg_pids=()
    for part in "$segwork"/tars.*; do
      [ -s "$part" ] || continue
      n=${part##*.}
      "$PY_PREPARE" tools/prepare_long.py --tars-from "$part" "${meta[@]}" "${langs[@]}" \
        --out-dir "$segwork" --manifest "$segwork/manifest.$n.jsonl" --manifest-only \
        --container-types "$VIEW" \
        --segments "$SEGMENTS_MODE" --max-seconds "$MAX_SECONDS" \
        --window-seconds "$WINDOW_SECONDS" --min-seconds "$MIN_SECONDS" \
        --workers "$PREPARE_WORKERS" > "$segwork/seg.$n.log" 2>&1 &
      seg_pids+=($!)
    done
    # Wait on these pids, not `jobs -p`: that would also reap anything else this shell
    # has in the background and report its exit as a shard failure.
    for pid in "${seg_pids[@]}"; do wait "$pid" || seg_rc=1; done
    [ "$seg_rc" = 0 ] || die "a segments shard failed — see $segwork/seg.NNN.log"
    cat "$segwork"/manifest.*.jsonl > "$SEGMANIFEST"
    say "segments: $(rows "$SEGMANIFEST") segment(s) from $shards shard(s)"
    [ -s "$SEGMANIFEST" ] || die "segments stage produced no $SEGMANIFEST"
    if [ -n "$SCOPE" ]; then
      # A tar holds containers the scope does not name — other views, or recordings the
      # long run excluded. Drop their segments rather than annotate past the corpus.
      "$PY_NV" - "$SEGMANIFEST" "$SCOPE" "${SCOPE_KEY:-source}" <<'PY' 2>&1 | tee -a "$LOG" || die "scoping failed"
import json, os, sys
seg, scope, mode = sys.argv[1], sys.argv[2], sys.argv[3]
# (source_tar, source_member) is the join wherever the scope carries it: recording ids
# hold a _pN part number that moves with the segmentation, while the tar member does not.
# A scope built without that provenance joins on parent_id instead, which is stable for
# a container even though it is not for a part of one.
key = (lambda r: r.get("parent_id")) if mode == "parent_id" else \
      (lambda r: (r.get("source_tar"), r.get("source_member")))
scope_key = (lambda r: r.get("id")) if mode == "parent_id" else key
want = set()
for line in open(scope):
    want.add(scope_key(json.loads(line)))
kept = dropped = 0
seen = set()
tmp = seg + ".scoped"
with open(seg) as f, open(tmp, "w") as out:
    for line in f:
        r = json.loads(line)
        k = key(r)
        if k in want:
            out.write(line)
            kept += 1
            seen.add(k)
        else:
            dropped += 1
os.replace(tmp, seg)
missing = len(want) - len(seen)
print(f"  scoped to {scope}: kept {kept} segment(s) over {len(seen)} recording(s), "
      f"dropped {dropped} outside it")
if missing:
    print(f"  {missing} of the scope's {len(want)} recording(s) produced no segment. "
          f"Expected when the tar list is a subset of the corpus (TARS_FROM, or a glob); "
          f"if it was meant to be the whole corpus, check METAINFO_DIR — a container "
          f"whose short view is empty can only be segmented from the metainfo.")
PY
      [ -s "$SEGMANIFEST" ] || die "scoping left no segments; is $SCOPE from this corpus?"
    fi
  fi
  MANIFEST="$SEGMANIFEST"
elif [ -n "$LONG" ]; then
  MANIFEST="$OUT/manifest.jsonl"
fi

total=$(rows "$MANIFEST")
say "manifest=$MANIFEST rows=$total out=$OUT stages=$STAGES"

# ---- prep
SCORE_MANIFEST="$MANIFEST"
if want prep; then
  prepped="$OUT/manifest-tarorder.jsonl"
  if [ "$(rows "$prepped")" = "$total" ]; then
    say "prep: $prepped already has $total row(s), keeping it"
  else
    say "prep: baking tar offsets, grouping by tar -> $prepped"
    "$PY_NV" tools/panns_filter.py offsets --manifest "$MANIFEST" --out "$prepped" \
      --sort-by-tar --tar-dir "$CORPUS" 2>&1 | tee -a "$LOG" || die "prep failed"
  fi
  SCORE_MANIFEST="$prepped"
fi

# ---- panns
SPEECH="$OUT/manifest_speech.jsonl"
SUNG="$OUT/manifest_sung.jsonl"
if want panns; then
  if [ "$(( $(rows "$SPEECH") + $(rows "$SUNG") ))" = "$total" ]; then
    say "panns: already split $total row(s), keeping $SPEECH"
  else
    supervise panns "$SCORE_MANIFEST" "$OUT/panns-work" \
      || die "panns stage stopped making progress — see $OUT/panns-work/pannsNN.log"
    # Score the tar-grouped copy for the reads, split the original so the NVASR pass
    # gets its own order back; the verdicts are keyed by id, so the two need not match.
    # Read the shards rather than a cat of them: a worker killed mid-append leaves a NUL
    # hole, and cat welds a truncated tail onto the next file's head.
    say "panns: splitting $MANIFEST by verdict"
    "$PY_NV" tools/panns_filter.py split --manifest "$MANIFEST" \
      --scores "$OUT/panns-work/panns[0-9]*.jsonl" \
      --out-speech "$SPEECH" --out-sung "$SUNG" 2>&1 | tee -a "$LOG" || die "split failed"
    cat "$OUT"/panns-work/panns[0-9]*.jsonl > "$OUT/panns.jsonl"
  fi
  # Falling back to the unfiltered manifest here would annotate the very clips this
  # stage exists to remove, and the run would still look like a success. It is an error.
  [ -s "$SPEECH" ] || die "panns stage left no $SPEECH — refusing to annotate unfiltered"
  say "panns: $(rows "$SPEECH") speech / $(rows "$SUNG") sung"
fi
# Only when the filter was deliberately skipped: STAGES without panns means the caller
# is annotating whatever they passed in.
[ -s "$SPEECH" ] || SPEECH="$MANIFEST"

# ---- nvasr
NV="$OUT/nv-speech.jsonl"
if want nvasr; then
  speech_rows=$(rows "$SPEECH")
  if [ "$(rows "$NV")" = "$speech_rows" ] && [ "$speech_rows" != 0 ]; then
    say "nvasr: $NV already has all $speech_rows row(s), keeping it"
  else
    supervise nvasr "$SPEECH" "$OUT/nv-work" \
      || die "nvasr stage stopped making progress — see $OUT/nv-work/nvNN.log"
    cat "$OUT"/nv-work/nv[0-9]*.jsonl > "$NV" || die "merging nv shards failed"
    if [ "$(rows "$NV")" != "$speech_rows" ]; then
      # A clip goes unwritten when its audio fails to decode *or* when its batch died —
      # a CUDA OOM leaves the whole batch pending, unretried. Report both causes.
      oom=$(grep -c 'batch of .* failed (OutOfMemoryError' "$OUT"/nv-work/nv[0-9]*.log 2>/dev/null \
            | awk -F: '{s+=$NF} END{printf "%.0f", s+0}')
      say "[warn] $NV has $(rows "$NV")/$speech_rows row(s); $oom batch(es) died of CUDA OOM" \
          "(lower NV_PER_GPU or NV_BATCH). Rerunning resumes into exactly the gap."
    fi
  fi
fi

# ---- verify
VERIFIED="$OUT/nv-verified.jsonl"
if want verify; then
  if [ "$(rows "$VERIFIED")" = "$(rows "$NV")" ] && [ "$(rows "$NV")" != 0 ]; then
    say "verify: $VERIFIED already covers $(rows "$NV") row(s), keeping it"
  else
    say "verify: transcript and PANNs checks over $(rows "$NV") row(s)"
    nop=(); [ -n "${NO_PANNS_VERIFY:-}" ] && nop=(--no-panns)
    "$PY_NV" tools/nv_verify.py --nv "$NV" --out "$VERIFIED" \
      --workers "$VERIFY_WORKERS" --audioldm-root "$AUDIOLDM_ROOT" \
      --tar-dir "$CORPUS" --ckpt-root "$CKPT_ROOT" "${nop[@]}" \
      2>&1 | tee -a "$LOG" || die "verify stage failed"
  fi
fi

# ---- final: one record per corpus row, the file everything downstream should read
# The passes work per segment because that is the unit a sung filter and an utterance
# ASR can be honest about, but the corpus's record is a recording (long) or a clip
# (short). nv-verified.jsonl is the evidence; this is the deliverable.
FINAL="$OUT/final.jsonl"
if want final; then
  if [ ! -s "$VERIFIED" ]; then
    say "[warn] final: no $VERIFIED yet, skipping"
  elif [ "$(rows "$FINAL")" = "$(rows "$SCOPE")" ] && [ "$(rows "$FINAL")" != 0 ]; then
    say "final: $FINAL already covers $(rows "$SCOPE") record(s), keeping it"
  else
    by=id; [ -n "$LONG" ] && by=parent_id
    records="$SCOPE"
    if [ -z "$records" ] || [ ! -s "$records" ]; then
      # A long run given a tar glob has no record list to fold into, so make one from
      # the segments themselves: a row per recording, carrying what the segments agree on.
      records="$OUT/records.jsonl"
      say "final: no record manifest given, deriving one from the segments"
      "$PY_NV" - "$OUT/manifest.jsonl" "$records" <<'PY' || die "deriving records failed"
import json, sys
seen = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    p = r.get("parent_id")
    if p and p not in seen:
        seen[p] = {"id": p, "source_tar": r.get("source_tar"),
                   "source_member": r.get("source_member"),
                   "file_seconds": r.get("parent_duration"), "lang": r.get("lang")}
with open(sys.argv[2], "w") as out:
    for row in seen.values():
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"  {len(seen)} recording(s)")
PY
    fi
    sung_args=(); [ -s "$SUNG" ] && sung_args=(--sung "$SUNG")
    "$PY_NV" tools/merge_nv.py --records "$records" --nv "$VERIFIED" \
      --out "$FINAL" --out-tagged "$OUT/final-nv-only.jsonl" --by "$by" \
      ${MANIFEST:+--segments "$MANIFEST"} \
      ${DROP_TAGS:+--drop-tags "$DROP_TAGS"} \
      ${EXCLUDE_IDS:+--exclude-ids "$EXCLUDE_IDS"} "${sung_args[@]}" \
      2>&1 | tee -a "$LOG" || die "final stage failed"
  fi
fi

# ---- stats
if want stats; then
  say "stats:"
  "$PY_NV" - "$VERIFIED" "$SUNG" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
from collections import Counter

path, sung_path = sys.argv[1], sys.argv[2]
n = tagged = kept_rows = 0
kept, dropped, by_stage = Counter(), Counter(), Counter()
try:
    fh = open(path)
except OSError as e:
    sys.exit(f"  no verified output to summarise: {e}")
for line in fh:
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    n += 1
    if r.get("nv_tags"):
        tagged += 1
    v = r.get("nv_verified") or []
    if v:
        kept_rows += 1
    kept.update(v)
    for tag, why in (r.get("nv_rejected") or {}).items():
        dropped[tag] += 1
        by_stage[why.split("(")[0]] += 1

try:
    n_sung = sum(1 for _ in open(sung_path))
except OSError:
    n_sung = 0

print(f"  rows annotated      {n}")
print(f"  routed out as sung  {n_sung}")
print(f"  rows tagged by ASR  {tagged}  ({tagged*100/max(n,1):.2f}%)")
print(f"  rows tagged after   {kept_rows}  ({kept_rows*100/max(n,1):.2f}%)")
print(f"  tags kept/dropped   {sum(kept.values())} / {sum(dropped.values())}")
if by_stage:
    print("  dropped by stage    " + ", ".join(f"{k}={v}" for k, v in by_stage.most_common()))
print("  surviving tags:")
for tag, c in kept.most_common():
    d = dropped[tag]
    print(f"    {tag:<20} {c:>8}   (dropped {d}, {d*100/max(c+d,1):.1f}%)")
PY
fi

say "done -> $OUT"
