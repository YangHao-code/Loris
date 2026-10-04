#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_workspace "$@"
# Independent workspace, rules_mode=discover, and the same supplied Pi/Si/Theta.
# The rules stage rejects reuse; random joint proposals use zero fit initializers.
loris_stage audit --workspace "$loris_workspace"
loris_stage rules --workspace "$loris_workspace" --proposal-strategy random_joint
# The newly discovered program needs its own independent rule reliability scores.
loris_stage calibrate --workspace "$loris_workspace"
loris_stage label --workspace "$loris_workspace" --variant main
loris_stage evaluate --workspace "$loris_workspace" --variant main
loris_stage collect --workspace "$loris_workspace"
