#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Final end-to-end AAPD run for the paper-alignment refactor (B-3 .. C-8).
#
# Design goals (per user request):
#   * Run the AAPD dataset end-to-end and leave reviewable metrics + rules.
#   * CHECKPOINT intermediate state so the run resumes after an interruption
#     (dropped connection / restart). The pipeline already persists, into
#     $EXP_DIR/<run>/ :
#         embeddings_*.npy        (sentence embeddings cache)
#         model_pool.pkl          (trained ML model pool)
#         cluster_*_store.json     (abstracted pattern stores per cluster)
#         optuna_chase_*.log       (Optuna journal — BO trials, resumable)
#         rules.json / rules_readable.txt / metrics.json   (final outputs)
#   * On re-launch, if a prior (interrupted) run exists under $EXP_DIR, resume
#     from it with --resume_mode all (reuse model pool + patterns + trials,
#     skipping the expensive phases that already finished).
#
# Why CPU (CUDA_VISIBLE_DEVICES=""): the per-cluster Bayesian-optimization chase
# is numpy/CPU-bound; the GPU only accelerates the one-off embedding pass, and
# running the parallel BO workers on the 24GB GPU exhausts it (CUDA OOM). CPU is
# the reliable path (this is also what the golden harness uses). The
# torch.backends.mps shim now lives in loris/__init__, so CPU runs work directly.
# ---------------------------------------------------------------------------
set -uo pipefail
cd /root/autodl-tmp/Loris

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0

EXP_DIR="experiments/aapd_final"

ARGS=(
  --dataset aapd
  --subset_size 5000
  --top_labels 30
  --max_trials 20
  --no_encoder
  --two_val
  --rule_strategy batch
  --cluster_model_selection global
  --batch_metric_mode global_macro
  --track1_baseline blank
  --pattern_mode sim
  --exp_dir "$EXP_DIR"
)

# ── Resume detection ──────────────────────────────────────────────────────
# If a previous run left a model pool under $EXP_DIR, resume from the newest one
# (reuse model pool + per-cluster pattern stores + Optuna trials).
RESUME_ARGS=()
if compgen -G "$EXP_DIR/*/model_pool.pkl" > /dev/null 2>&1; then
    PRIOR_DIR="$(ls -dt "$EXP_DIR"/*/ 2>/dev/null | head -1)"
    PRIOR_DIR="${PRIOR_DIR%/}"
    if [ -n "$PRIOR_DIR" ]; then
        echo "[resume] found prior run: $PRIOR_DIR — resuming (--resume_mode all)"
        RESUME_ARGS=(--resume_exp_dir "$PRIOR_DIR" --resume_mode all)
    fi
fi

echo "============================================================"
echo "=== AAPD final run (paper-alignment B-3..C-8)            ==="
echo "=== commit: $(git rev-parse --short HEAD)                 ==="
echo "=== started: $(date)                                      ==="
echo "=== args: ${ARGS[*]} ${RESUME_ARGS[*]+${RESUME_ARGS[*]}}  ==="
echo "============================================================"

python3 -m loris "${ARGS[@]}" ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
RC=$?
echo "[done] exit=$RC at $(date)"
exit $RC
