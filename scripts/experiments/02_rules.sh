#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_workspace "$@"
# rules_mode=reuse validates the entire ordered program, closure, BO counts and receipt.
# rules_mode=discover starts fresh BO from a complete prepared upstream, in score order.
# An incomplete cached program fails; it never silently launches discovery.
loris_stage rules --workspace "$loris_workspace"
