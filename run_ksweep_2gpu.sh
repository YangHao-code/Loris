#!/usr/bin/env bash
# Varying-K sweep, 2-GPU: aapd (GPU0, the case where router lost) + reuters (GPU1).
# K=1..5 x 5 selectors x 1 dataset per lane = 25 cells/lane. Detached via setsid.
set -u
cd /root/autodl-tmp/Loris
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12
mkdir -p logs experiments/baselines_loris/ksweep

SEEDS="${SEEDS:-0}"; KGRID="${KGRID:-1,2,3,4,5}"
echo "[$(date '+%F %T')] KSWEEP START seeds=$SEEDS kgrid=$KGRID" | tee logs/ksweep_gate.log

CUDA_VISIBLE_DEVICES=0 python run_ksweep_loris.py --datasets aapd --seeds "$SEEDS" --kgrid "$KGRID" \
    > logs/ksweep_gpu0_aapd.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python run_ksweep_loris.py --datasets reuters21578 --seeds "$SEEDS" --kgrid "$KGRID" \
    > logs/ksweep_gpu1_reuters.log 2>&1 &
P1=$!
echo "[$(date '+%F %T')] GPU0 aapd pid=$P0 | GPU1 reuters pid=$P1" | tee -a logs/ksweep_gate.log
wait $P0; R0=$?; wait $P1; R1=$?
echo "[$(date '+%F %T')] KSWEEP DONE gpu0_rc=$R0 gpu1_rc=$R1" | tee -a logs/ksweep_gate.log
