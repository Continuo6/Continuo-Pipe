#!/usr/bin/env bash
# Run the public NV pipeline over long-recording or dialogue segments.
# VIEW=long (default) or VIEW=dialogue selects the source layout. The pipeline
# performs PANNs routing, NVASR inference, transcript/PANNs verification, and
# merges results back by recording id. No emotion model is used.
#
# Usage: scripts/run_nv_long_pipeline.sh <manifest-or-tar-glob> <out-dir>
set -uo pipefail

TARS=${1:?usage: run_nv_long_pipeline.sh <tar-glob> <out-dir>}
OUT=${2:?usage: run_nv_long_pipeline.sh <tar-glob> <out-dir>}

ROOT="${CONTINUO_EXPRESSIVE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -x "$ROOT/scripts/run_nv_pipeline.sh" ] || {
  echo "[fatal] $ROOT/scripts/run_nv_pipeline.sh missing" >&2; exit 1; }
VIEW="${VIEW:-long}"
case "$VIEW" in
  long|dialogue) ;;
  *) echo "[fatal] VIEW=$VIEW; expected long or dialogue" >&2; exit 1 ;;
esac
exec env LONG=1 VIEW="$VIEW" "$ROOT/scripts/run_nv_pipeline.sh" "$TARS" "$OUT"
