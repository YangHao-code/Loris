#!/usr/bin/env bash
set -euo pipefail
# Existing complete BO -> preserve committed prefix -> approximate admission.
# SOURCE remains read-only. OUTPUT must be a separate new directory.
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 MANIFEST SOURCE_RUN NEW_OUTPUT [extra admit-pool options]" >&2
  exit 2
fi
manifest=$1
source_run=$2
output=$3
shift 3
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
python -m loris.discovery admit-pool --manifest "$manifest" --source "$source_run" \
  --output "$output" --admission-workers "${ADMISSION_WORKERS:-8}" \
  --admission-batch-size "${ADMISSION_BATCH_SIZE:-8}" "$@"
