#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Loris
export LORIS_PIPELINE_ENTRY=tests.golden._golden_entry
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
python tests/golden/run_golden.py "$@"
