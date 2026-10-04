#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "Usage: $0 PREPARED_SWEEP.json NEW_OUTPUT_ROOT [--dry-run]" >&2
  exit 2
fi
# Every point must supply complete actual data/model/calibration/program assets.
# The declared document/rule/label lists are validated against actual context/program arrays.
exec "$loris_python" "$loris_script_dir/run_scaling.py" --sweep "$1" --output "$2" "${@:3}"
