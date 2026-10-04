#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_workspace "$@"
# Use a fresh workspace for this RQ. All variants start with identical assets/Gamma.
loris_stage audit --workspace "$loris_workspace"
loris_stage rules --workspace "$loris_workspace"
loris_stage calibrate --workspace "$loris_workspace"
# Frozen keeps the same declared external permissions while fixing omega/kappa.
# Adaptive requests can differ: this interaction comparison does not isolate refinement.
for variant in frozen self_train main; do
  loris_stage label --workspace "$loris_workspace" --variant "$variant"
  # Score the actual FINAL_P from that run, never an initial-model re-Chase surrogate.
  loris_stage evaluate --workspace "$loris_workspace" --variant "$variant"
done
loris_stage collect --workspace "$loris_workspace"
