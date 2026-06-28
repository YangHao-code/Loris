#!/usr/bin/env bash
# Phase 3 launcher — single GPU + concurrent CPU lane for rcv1.
# GPU lane: 4 text datasets, run sequentially in ONE process (single GPU, no contention).
# CPU lane: rcv1 (linear-only, no GPU) runs concurrently for a free speedup.
set -u
cd /root/autodl-tmp/Loris

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p logs experiments/baselines

SEEDS="${SEEDS:-0,1,2}"

echo "[$(date '+%F %T')] launching GPU lane (reuters,aapd,bgc,arxiv) seeds=$SEEDS"
CUDA_VISIBLE_DEVICES=0 python run_phase3.py \
    --datasets reuters21578,aapd,bgc,arxiv --seeds "$SEEDS" \
    > logs/phase3_gpu.log 2>&1 &
GPU_PID=$!

echo "[$(date '+%F %T')] launching CPU lane (rcv1) seeds=$SEEDS"
CUDA_VISIBLE_DEVICES="" python run_phase3.py \
    --datasets rcv1 --seeds "$SEEDS" \
    > logs/phase3_cpu.log 2>&1 &
CPU_PID=$!

echo "GPU_PID=$GPU_PID CPU_PID=$CPU_PID"
wait $GPU_PID; GPU_RC=$?
wait $CPU_PID; CPU_RC=$?
echo "[$(date '+%F %T')] DONE gpu_rc=$GPU_RC cpu_rc=$CPU_RC"
