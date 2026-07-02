#!/usr/bin/env bash
# Seed-0 gate launcher for the LORIS-integrated Group A+C matrix.
# Detached (setsid) so it survives a REPL/session restart. Writes a PID file.
set -u
cd /root/autodl-tmp/Loris
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p logs experiments/baselines_loris

echo "[$(date '+%F %T')] PHASE3-LORIS seed0 START pid=$$" > logs/phase3_loris_seed0.log
python run_phase3_loris.py \
    --group both --datasets reuters21578,aapd,arxiv,bgc,rcv1 \
    --seeds 0 --k 3 >> logs/phase3_loris_seed0.log 2>&1
echo "[$(date '+%F %T')] PHASE3-LORIS seed0 DONE rc=$?" >> logs/phase3_loris_seed0.log
