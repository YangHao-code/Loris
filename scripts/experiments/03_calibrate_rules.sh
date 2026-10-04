#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 BOUND_WORKSPACE [--conflict-policy unknown_arbitrate|halt] [--dry-run]" >&2
  exit 2
fi
require_workspace "$1"
shift
# Fixed checkpoint + separate independent calibration context, with scored truth masked.
# Each rule: distinct verifiable conclusions; q=(C+1)/(N+2). No optimizer or BO.
# Main and halt_conflicts use distinct real calibration runs and output directories.
loris_stage calibrate --workspace "$loris_workspace" "$@"
