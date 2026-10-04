#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_workspace "$@"
loris_stage audit --workspace "$loris_workspace"
loris_stage rules --workspace "$loris_workspace"
loris_stage calibrate --workspace "$loris_workspace"
# Manifest must supply the same actual answer provider and nonzero budget for all.
# One unresolved document per step: closure saturation -> RDG/random/uncertainty -> answer.
for variant in nollm main random uncertainty; do
  loris_stage label --workspace "$loris_workspace" --variant "$variant"
  loris_stage evaluate --workspace "$loris_workspace" --variant "$variant"
done
loris_stage collect --workspace "$loris_workspace"
