#!/usr/bin/env bash
# Phase 3 launcher — SINGLE sequential lane over all 5 datasets on the GPU.
# Rationale: the selector pool's encoder now drops on no-raw-text datasets
# (rcv1), so rcv1 is fast and needs no separate CPU lane. A single lane avoids
# the CPU contention that two concurrent lanes caused (both are CPU-heavy on
# snuba/selectors). rcv1's transformer TEXT baselines still skip by design.
set -u
cd /root/autodl-tmp/Loris

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p logs experiments/baselines

SEEDS="${SEEDS:-0,1,2}"
DATASETS="${DATASETS:-reuters21578,aapd,arxiv,bgc,rcv1}"

echo "[$(date '+%F %T')] Phase 3 single-lane: datasets=$DATASETS seeds=$SEEDS"
CUDA_VISIBLE_DEVICES=0 python run_phase3.py \
    --datasets "$DATASETS" --seeds "$SEEDS" \
    > logs/phase3_all.log 2>&1
RC=$?
echo "[$(date '+%F %T')] Phase 3 DONE rc=$RC"
