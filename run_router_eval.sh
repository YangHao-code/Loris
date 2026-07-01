#!/usr/bin/env bash
# Router TEST-set oracle-gap eval across the 7 suitable datasets, 2-GPU.
# Run AFTER run_router_rework.sh completes (needs the GPUs free). For each
# dataset it fits the pool, trains the router with the tuned config, and reports
# val→test hit@K/MRR + the fixed-selection-vs-per-doc-oracle gap → experiments/router_eval/.
set -u
cd /root/autodl-tmp/Loris
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12
mkdir -p logs experiments/router_eval

lane () {
  local gpu="$1"; shift
  local log="logs/router_eval_gpu${gpu}.log"
  : > "$log"
  for ds in "$@"; do
    echo "[$(date '+%F %T')] eval $ds (gpu$gpu)" | tee -a "$log"
    CUDA_VISIBLE_DEVICES=$gpu python -m loris.selection.router_eval \
      --dataset "$ds" --seed 0 --backend perturbed \
      --tuned experiments/router_tuning --out experiments/router_eval \
      >> "$log" 2>&1 || echo "[$(date '+%F %T')] eval $ds FAILED" | tee -a "$log"
  done
  echo "[$(date '+%F %T')] lane gpu$gpu DONE" | tee -a "$log"
}

lane 0 reuters21578 aapd pubmed goodreads &
P0=$!
lane 1 bgc arxiv hupd &
P1=$!
wait $P0; wait $P1
echo "[$(date '+%F %T')] ROUTER EVAL ALL DONE"
