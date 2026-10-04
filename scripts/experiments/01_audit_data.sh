#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_workspace "$@"
# Check every row, ordered label axis, complete split count and cross-split duplicate text.
loris_stage audit --workspace "$loris_workspace"
