#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_workspace "$@"
loris_stage audit --workspace "$loris_workspace"
loris_stage rules --workspace "$loris_workspace"
loris_stage calibrate --workspace "$loris_workspace"
# Same rules/model/Gamma/query seed and update budget; vary closure maintenance only.
# retain_consequences: NoInc scans every rule on the SAME retained history.
# It must not clear historical consequences and recompute from Gamma.
# The legacy full_rebuild option still cannot establish an incremental speedup.
for variant in main noinc; do
  loris_stage label --workspace "$loris_workspace" --variant "$variant"
  loris_stage evaluate --workspace "$loris_workspace" --variant "$variant"
done
# Assert final signed facts, retained history, Gamma and supervision events.
"$loris_python" "$loris_script_dir/check_noinc_equivalence.py" --workspace "$loris_workspace"
# Any mismatch exits before reporting equivalence or measured costs.
loris_stage collect --workspace "$loris_workspace"
