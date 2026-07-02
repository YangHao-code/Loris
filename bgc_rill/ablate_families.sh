#!/bin/bash
# Per-predicate-family ablation on BGC, embedding-only pool (no tfidf).
# Fixed config; vary only the textual family kept. Delta = final - baseline(val-selected model).
set -u
cd /root/autodl-tmp/Loris
export PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
COMMON="--dataset bgc --subset_size 4000 --top_labels 20 --max_trials 60 --no_adaptive_trials \
 --model_pool embedding --no_encoder --two_val --rule_strategy batch --pattern_mode full \
 --track1_baseline model"
OUT=experiments/bgc_abl
mkdir -p $OUT
for FAM in all match freq before cooccur; do
  PF=""; [ "$FAM" != "all" ] && PF="--predicate_families $FAM"
  echo "===== FAMILY=$FAM ====="
  python3 -m loris $COMMON $PF --exp_dir $OUT/$FAM > $OUT/$FAM.log 2>&1
  echo "FAMILY=$FAM done (exit $?)"
done
echo "ALL ABLATION RUNS DONE"
