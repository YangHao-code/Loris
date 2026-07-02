#!/usr/bin/env bash
# =============================================================================
# Phase 3 (v2) — LORIS-INTEGRATED baselines, Group A (model selection) + Group C
# (pattern selection). Paper protocol: swap ONE LORIS component, report LORIS's
# downstream Macro-F1. Canonical caps. SINGLE GPU, sequential.
#
# Run order (user decision): SEED 0 FIRST as a sanity gate, then seeds 1,2 after
# review. This script runs ONE seed (default 0); pass SEED=1 / SEED=2 to scale.
#
# What each seed runs (driver = run_phase3_loris.py):
#   Group A: --selector {router,random_ms,indiv_ms,hybrid_llm,caas}  (5)  × 5 datasets = 25 cells
#            ('router' = the LORIS reference row, router ON at K)
#   Group C: --pattern_select {filter_mi,filter_chi2,weshap,localboost} (4) × 5 datasets = 20 cells
#            ('loris' reference reuses A_router, so it's skipped here)
#   => 45 LORIS runs per seed.
#
# Canonical caps (from loris.baselines.common.DATASET_DEFAULTS, applied by the driver):
#   reuters21578: full       aapd: subset 15k    rcv1: subset 20k (encoder auto-dropped, no raw text)
#   bgc: subset 12k/test 8k  arxiv: subset 20k/test 8k
# Chase bounded: --max_trials 40 --no_adaptive_trials (else 15*30=450 trials/cluster),
#   applied EQUALLY to every cell incl. the LORIS reference row (fair comparison).
#
# Each cell shells `python -m loris ... --selector/--pattern_select ...`, parses
# final_macro_f1 from metrics.json, writes experiments/baselines_loris/<name>__<ds>__seedN.json
# =============================================================================
set -u
cd /root/autodl-tmp/Loris

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"     # <-- GPU pinned here
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p logs experiments/baselines_loris

SEED="${SEED:-0}"
DATASETS="${DATASETS:-reuters21578,aapd,arxiv,bgc,rcv1}"
K="${K:-3}"

echo "[$(date '+%F %T')] Phase3-LORIS seed=$SEED gpu=$CUDA_VISIBLE_DEVICES datasets=$DATASETS K=$K"
echo "[$(date '+%F %T')] Group A (model selection) + Group C (pattern selection), canonical caps"

python run_phase3_loris.py \
    --group both \
    --datasets "$DATASETS" \
    --seeds "$SEED" \
    --k "$K" \
    > "logs/phase3_loris_seed${SEED}.log" 2>&1
RC=$?
echo "[$(date '+%F %T')] Phase3-LORIS seed=$SEED DONE rc=$RC"
echo "Results: experiments/baselines_loris/*.json   Log: logs/phase3_loris_seed${SEED}.log"
