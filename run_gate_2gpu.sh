#!/usr/bin/env bash
# Seed-0 gate, 2-GPU split. Two parallel lanes, each pinned to one RTX 5090,
# over disjoint datasets. Detached (setsid) so it survives a REPL restart.
#
#   GPU0 lane: bgc, arxiv         (the two heaviest: subset+test capped)
#   GPU1 lane: reuters, aapd, rcv1
#
# Both lanes write to the SAME experiments/baselines_loris/ (filenames are
# per-dataset so no collision). Each lane logs separately.
set -u
cd /root/autodl-tmp/Loris
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Thread caps: 32 physical cores, 2 lanes. Match the project's own scripts
# (run_aapd_final.sh uses 4). Without this, torch/BLAS defaults to 32/proc ->
# 2 lanes x 32 = 64 threads thrashing 32 cores, crawling the chase. 12/lane
# keeps both lanes < core count with headroom for joblib.
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
mkdir -p logs experiments/baselines_loris

SEEDS="${SEEDS:-0}"

echo "[$(date '+%F %T')] GATE 2-GPU START seeds=$SEEDS" | tee logs/phase3_loris_gate.log

CUDA_VISIBLE_DEVICES=0 python run_phase3_loris.py \
    --group both --datasets bgc,arxiv --seeds "$SEEDS" --k 3 \
    > logs/phase3_loris_gpu0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python run_phase3_loris.py \
    --group both --datasets reuters21578,aapd,rcv1 --seeds "$SEEDS" --k 3 \
    > logs/phase3_loris_gpu1.log 2>&1 &
P1=$!

echo "[$(date '+%F %T')] GPU0 lane pid=$P0 (bgc,arxiv) | GPU1 lane pid=$P1 (reuters,aapd,rcv1)" | tee -a logs/phase3_loris_gate.log
wait $P0; R0=$?
wait $P1; R1=$?
echo "[$(date '+%F %T')] GATE DONE gpu0_rc=$R0 gpu1_rc=$R1" | tee -a logs/phase3_loris_gate.log
