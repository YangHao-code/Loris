#!/usr/bin/env bash
# Fresh complete task: each stage records its commands, result, and live log.
# Retry a failed stage with run_stages.py; do not recreate completed upstreams.
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "usage: bash scripts/discovery/run_all.sh CONFIG [--variables 2 --models learned --bo-trials B --search-seed S]" >&2
  exit 2
fi
config=$1
shift
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for stage in supervision pool-freeze pool-cpu pool-gpu pool-handoff features context selector-inputs selector patterns prewarm bind check bo status archive; do
  python "$script_dir/run_stages.py" --config "$config" --stage "$stage" --execute "$@"
done
