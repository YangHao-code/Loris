#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 ASSET_MAP.json NEW_WORKSPACE (optional LORIS_ASSET_ROOT)" >&2
  exit 2
fi
# Resolve relocatable paths and freeze SHA256 identities. No experiment is launched.
loris_stage bind --assets "$1" --output "$2"
