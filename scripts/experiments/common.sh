#!/usr/bin/env bash
# Shared interpreter/environment setup. The experiment steps live in each RQ script.
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
loris_python="${LORIS_PYTHON:-python}"
loris_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
loris_source_root="$(cd -- "$loris_script_dir/../.." && pwd)"
export PYTHONPATH="$loris_source_root${PYTHONPATH:+:$PYTHONPATH}"
loris_stage() { "$loris_python" -m loris.experiments workflow "$@"; }
require_workspace() {
  if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 BOUND_WORKSPACE" >&2
    exit 2
  fi
  loris_workspace="$1"
}
