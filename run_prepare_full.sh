#!/usr/bin/env bash
# Download + process the FULL/million-scale datasets (auth-free sources) into
# data/<ds>_full/processed/. IO/CPU-bound (no GPU). Two tracks to overlap
# download with processing without over-committing memory.
set -u
cd /root/autodl-tmp/Loris
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=0
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs

prep(){
  local ds="$1"; local log="logs/prep_${ds}.log"
  echo "[$(date '+%F %T')] START $ds" | tee "$log"
  if python -m loris.data.prepare_full --dataset "$ds" >> "$log" 2>&1; then
    echo "[$(date '+%F %T')] DONE $ds" | tee -a "$log"
  else
    echo "[$(date '+%F %T')] FAILED $ds (see $log)" | tee -a "$log"
  fi
}

( prep pubmed_full; prep goodreads_full ) &
T1=$!
( prep arxiv_full; prep hupd_full ) &
T2=$!
wait $T1; wait $T2
echo "[$(date '+%F %T')] ALL FULL PREP DONE"
