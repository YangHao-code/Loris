#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
  echo "Usage: $0 ASSET_MAP_DIRECTORY NEW_OUTPUT_ROOT rq1_quality|rq1_random_joint|rq2_feedback|rq3_annotations|rq4_incremental [REPEATS=3]" >&2
  exit 2
fi
maps="$1"
output_root="$2"
rq="$3"
repeats="${4:-3}"
case "$rq" in rq1_quality|rq1_random_joint|rq2_feedback|rq3_annotations|rq4_incremental) ;; *) exit 2 ;; esac
if [[ ! "$repeats" =~ ^[1-9][0-9]*$ ]]; then exit 2; fi
# Paper reports three runs. Require independently prepared seed-specific assets.
# These maps freeze model training/selector/BO seeds; changing one CLI seed is insufficient.
for ((run_index=0; run_index<repeats; run_index++)); do
  seed=$((42+run_index))
  if [[ ! -f "$maps/seed_${seed}.json" ]]; then
    echo "Missing independently bound seed_${seed}.json; no duplicated run substitutes for it." >&2
    exit 2
  fi
done
workspaces=()
for ((run_index=0; run_index<repeats; run_index++)); do
  seed=$((42+run_index))
  workspace="$output_root/seed_${seed}"
  loris_stage bind --assets "$maps/seed_${seed}.json" --output "$workspace"
  bash "$loris_script_dir/${rq}.sh" "$workspace"
  workspaces+=("$workspace")
done
"$loris_python" "$loris_script_dir/aggregate_repeats.py" --workspaces "${workspaces[@]}" --output "$output_root/aggregate"
