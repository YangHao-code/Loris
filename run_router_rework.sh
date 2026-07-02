#!/usr/bin/env bash
# Router rework tune + re-run, 2-GPU. For each dataset: (1) per-dataset oracle
# tuning (chase-free) → experiments/router_tuning/, then (2) the LORIS router cell
# with per-cluster routing + perturbed backend + tuned hp → a FRESH output root
# (experiments/baselines_loris_rework/) so the old seed-0 runs are preserved.
#
# Only the A_router cell is (re)run; the four Group-A baselines are unaffected and
# stay in experiments/baselines_loris/. Seed 0. rcv1 excluded (tokenized text).
# Smallest datasets first in each lane for an early end-to-end signal.
set -u
cd /root/autodl-tmp/Loris
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12
mkdir -p logs experiments/router_tuning experiments/baselines_loris_rework

MAXCFG="${MAXCFG:-18}"
OUT_ROOT="${OUT_ROOT:-experiments/baselines_loris_rework}"

run_lane () {
  local gpu="$1"; shift
  local lane="$1"; shift
  local datasets=("$@")
  local log="logs/rework_gpu${gpu}_${lane}.log"
  echo "[$(date '+%F %T')] LANE $lane gpu$gpu datasets=${datasets[*]} maxcfg=$MAXCFG" | tee "$log"
  for ds in "${datasets[@]}"; do
    echo "[$(date '+%F %T')] === $ds : TUNE ===" | tee -a "$log"
    CUDA_VISIBLE_DEVICES=$gpu python -m loris.selection.router_tuning \
      --dataset "$ds" --seed 0 --backend perturbed --max_configs "$MAXCFG" \
      --out experiments/router_tuning >> "$log" 2>&1 \
      || echo "[$(date '+%F %T')] TUNE $ds FAILED (run uses default router hp)" | tee -a "$log"
    echo "[$(date '+%F %T')] === $ds : RUN ===" | tee -a "$log"
    CUDA_VISIBLE_DEVICES=$gpu python run_phase3_loris.py --group A --datasets "$ds" \
      --seeds 0 --only_selectors router --use_tuned_router --out_root "$OUT_ROOT" \
      >> "$log" 2>&1 \
      || echo "[$(date '+%F %T')] RUN $ds FAILED" | tee -a "$log"
  done
  echo "[$(date '+%F %T')] LANE $lane DONE" | tee -a "$log"
}

run_lane 0 A reuters21578 bgc pubmed &
P0=$!
run_lane 1 B goodreads aapd arxiv hupd &
P1=$!
wait $P0; wait $P1
echo "[$(date '+%F %T')] REWORK ALL DONE"
