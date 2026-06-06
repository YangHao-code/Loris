#!/usr/bin/env bash
# ===========================================================================
# End-to-end staged-redesign runs: AAPD + BGC, default vs enriched sim bins.
#
# Each run: blank/staged pipeline (Stage 0 ML -> 1 FN-add -> 2 FP-remove ->
# 3 propagation), encoder + tfidf in the pool, all stages on, with the RILL
# human-label budget sweep (bounded via --rill_sweep_max_docs). The "enriched"
# arm adds finer/denser similarity-graph degree bins (--sim_target_degrees) to
# measure the enrich-similarity lever as an A/B against the default arm.
#
# Runs SEQUENTIALLY (single GPU; avoids encoder-training OOM/contention) and is
# CHECKPOINTED + resumable: re-run this script to resume any interrupted arm
# (auto-detects the prior exp_dir -> --resume_mode all).
#
# Usage:  bash run_staged_e2e.sh           # all 4 arms
#         ARMS="aapd_default bgc_default" bash run_staged_e2e.sh   # subset
# ===========================================================================
set -uo pipefail
cd /root/autodl-tmp/Loris

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONHASHSEED=0
unset CUDA_VISIBLE_DEVICES || true          # GPU on for the encoder

COMMON=(
  --top_labels 30
  --max_trials 30
  --two_val
  --rule_strategy batch
  --cluster_model_selection global
  --batch_metric_mode global_macro
  --track1_baseline blank
  --pattern_mode sim
  --rill_budget_sweep 0,10,25,50,100
  --rill_sweep_max_docs 2000
)
ENRICH=(--sim_target_degrees 3,5,8,12,20,30,40,60)

# arm name -> dataset-specific args (subset/test caps chosen for ~comparable
# scale to the prior methodology results: AAPD 15k, BGC 12k/8k-test).
declare -A ARM_ARGS=(
  [aapd_default]="--dataset aapd --subset_size 15000"
  [aapd_enriched]="--dataset aapd --subset_size 15000 ENRICH"
  [bgc_default]="--dataset bgc --subset_size 12000 --max_test_docs 8000"
  [bgc_enriched]="--dataset bgc --subset_size 12000 --max_test_docs 8000 ENRICH"
)
ARMS="${ARMS:-aapd_default aapd_enriched bgc_default bgc_enriched}"

run_arm () {
  local name="$1"; local spec="${ARM_ARGS[$name]}"
  local exp="experiments/e2e_${name}"
  local extra=(); for tok in $spec; do
    if [ "$tok" = "ENRICH" ]; then extra+=("${ENRICH[@]}"); else extra+=("$tok"); fi
  done
  local resume=()
  if compgen -G "$exp/*/model_pool.pkl" > /dev/null 2>&1; then
    local prior; prior="$(ls -dt "$exp"/*/ 2>/dev/null | head -1)"; prior="${prior%/}"
    [ -n "$prior" ] && resume=(--resume_exp_dir "$prior" --resume_mode all) && \
      echo "[$name] resuming from $prior"
  fi
  echo "############ $name @ $(date) | commit $(git rev-parse --short HEAD) ############"
  python3 -m loris "${COMMON[@]}" "${extra[@]}" --exp_dir "$exp" \
      > "experiments/e2e_${name}.log" 2>&1
  echo "[$name] exit=$? @ $(date)"
}

for arm in $ARMS; do run_arm "$arm"; done

echo "######## ALL E2E ARMS DONE @ $(date) ########"
for arm in $ARMS; do
  M=$(ls experiments/e2e_${arm}/*/metrics.json 2>/dev/null | head -1)
  S=$(ls experiments/e2e_${arm}/*/rill_budget_sweep.json 2>/dev/null | head -1)
  [ -n "$M" ] && python3 -c "import json;d=json.load(open('$M'));print('  [$arm] base micro',round(d['baseline_test_micro_f1'],4),'macro',round(d['baseline_test_macro_f1'],4),'-> +rules micro',round(d['final_micro_f1'],4),'(d',round(d['micro_f1_delta'],4),') macro',round(d['final_macro_f1'],4),'(d',round(d['macro_f1_delta'],4),') n_rules',d.get('n_rules'))"
  [ -n "$S" ] && python3 -c "import json;d=json.load(open('$S'));print('     RILL sweep:',[(r['budget'],round(r['macro_f1'],4)) for r in d['budgets']])" 2>/dev/null
done
