#!/usr/bin/env bash
set -euo pipefail
# Requires an already prepared, explicit oracle and labeling manifest.
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 LABELING_MANIFEST NEW_OUTPUT [extra labeling options]" >&2
  exit 2
fi
manifest=$1
output=$2
shift 2
python -m loris.labeling run --manifest "$manifest" --output "$output" \
  --query-cadence each_closure --query-answer-mode complete \
  --query-batch-fraction "${QUERY_BATCH_FRACTION:-0.01}" \
  --query-batch-min "${QUERY_BATCH_MIN:-32}" \
  --new-verified-supervision full_batch --label-budget "${LABEL_BUDGET:-unlimited}" \
  --intermediate-exports periodic --history-export-interval "${HISTORY_EXPORT_INTERVAL:-10}" "$@"
