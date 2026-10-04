#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_workspace "$@"
loris_stage audit --workspace "$loris_workspace"
loris_stage rules --workspace "$loris_workspace"
loris_stage calibrate --workspace "$loris_workspace"
# Main uses the declared external provider and budget (unlimited by default).
# Set LORIS_SUPERVISION_PROTOCOL=fixed_gamma only for the named zero-query control.
supervision_protocol="${LORIS_SUPERVISION_PROTOCOL:-as_declared}"
case "$supervision_protocol" in
  as_declared|fixed_gamma) ;;
  *) echo 'Expected as_declared or fixed_gamma' >&2; exit 2 ;;
esac
# All strategies share unknown_arbitrate; their effective quarantine views may differ.
# Main discounts current direct-rule precision by model disagreement (default c=.5).
# Pure direct-rule precision and complete-proof scores/discount are extra controls.
# The named transfer ablation lets affected ensemble disagreements supervise Aux.
# NoCheck only removes the consistency check; its quality need not differ from main.
variants=(frozen supervised self_train main direct_rule_precision uniform complete_provenance complete_provenance_discount confidence_only nocheck trusted_replay exclude_receiver affected_channels_to_aux frozen_prior)
for variant in "${variants[@]}"; do
  loris_stage label --workspace "$loris_workspace" --variant "$variant" --supervision-protocol "$supervision_protocol"
  loris_stage evaluate --workspace "$loris_workspace" --variant "$variant" --require-round-metrics
done
# Verify actual final Gamma and external counts, not merely matching declared budgets.
if [[ "$supervision_protocol" == fixed_gamma ]]; then
  "$loris_python" "$loris_script_dir/check_fixed_gamma.py" --workspace "$loris_workspace" --variants "${variants[@]}"
fi
loris_stage collect --workspace "$loris_workspace"
