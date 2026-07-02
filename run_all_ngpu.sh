#!/usr/bin/env bash
# Launch the full LORIS paper experiment matrix (Exp 1-5, seed 0) across N GPUs.
#
#   ./run_all_ngpu.sh [NGPU]         # default: all visible GPUs
#
# Jobs are per-dataset so they parallelise; they are distributed round-robin to
# GPU lanes and each lane runs its jobs sequentially, setsid-detached so the run
# survives a session restart. Logs: logs/all_gpu<g>.log ; PIDs: logs/all.pids.
#
# Prereqs: ./prep_models.sh once (LoRA SLMs + deps). For the LLM-annotator arm,
# export OPENAI_API_KEY / OPENAI_BASE_URL / LORIS_LLM_MODEL before launching.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs

NGPU="${1:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[ "$NGPU" -ge 1 ] 2>/dev/null || NGPU=1
echo "Launching across NGPU=$NGPU"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
PY=python3

# All datasets, rcv1 LAST (hashed-TFIDF, slowest/special — deprioritised per request).
ALLDS="reuters21578 aapd bgc arxiv pubmed hupd rcv1"

# ── Build the job list (each entry = one shell command) ──────────────────────
# LoRA arm only if BOTH SLMs are cached (else run base now, add lora after prep).
MAIN_ARM=base
if [ -d "$HF_HOME/hub/models--mistralai--Mistral-7B-v0.1" ] && \
   [ -d "$HF_HOME/hub/models--Qwen--Qwen2-7B" ]; then
  MAIN_ARM=both
fi
echo "main arm = $MAIN_ARM (both ⇒ LoRA SLMs cached; base ⇒ run --arm lora later)"

JOBS=()
# Exp-1a main table — one job per dataset (rcv1 last)
for ds in $ALLDS; do
  JOBS+=("$PY run_main_loris.py --datasets $ds --arm $MAIN_ARM")
done
# Exp-1b standalone research baselines — only the NEW datasets (others done, seed0)
JOBS+=("$PY run_phase3.py --datasets pubmed --seeds 0 --skip_sweeps")
JOBS+=("$PY run_phase3.py --datasets hupd --seeds 0 --skip_sweeps")
# Exp-4a model-selection gate — new datasets (others done seed0)
JOBS+=("$PY run_phase3_loris.py --group both --datasets pubmed --seeds 0 --use_tuned_router")
JOBS+=("$PY run_phase3_loris.py --group both --datasets hupd --seeds 0 --use_tuned_router")
# Exp-4b Varying-K — all datasets, complete grid (rcv1 last)
for ds in $ALLDS; do
  JOBS+=("$PY run_ksweep_loris.py --datasets $ds --kgrid 1,2,3,4,5")
done
# Exp-4c Varying-|M| — all datasets (rcv1 last)
for ds in $ALLDS; do
  JOBS+=("$PY run_msweep_loris.py --datasets $ds --mgrid 2,3,4,5,6")
done
# Exp-2 human cost (Γ sweep + RILL human-only) — all datasets (rcv1 last). The LLM
# arm is launched separately (API-latency bound) after a one-dataset validation.
for ds in $ALLDS; do
  JOBS+=("$PY run_humancost.py --datasets $ds --pool base --arms no_llm")
done
# Exp-3 scalability (|D|,|Σ|,noInc) — all datasets (rcv1 last)
for ds in $ALLDS; do
  JOBS+=("$PY run_scalability.py --datasets $ds --pool cpu")
done
# Exp-5 ablations (noS/noL/noInc + stages) — all datasets, base pool (rcv1 last)
for ds in $ALLDS; do
  JOBS+=("$PY run_ablations.py --datasets $ds --pool base")
done

echo "Total jobs: ${#JOBS[@]}"

# ── Distribute round-robin to GPU lanes ──────────────────────────────────────
: > logs/all.pids
for ((g=0; g<NGPU; g++)); do
  lane_script="logs/_lane_$g.sh"
  {
    echo "#!/usr/bin/env bash"
    echo "cd '$(pwd)'"
    echo "export HF_HOME='$HF_HOME' HF_HUB_OFFLINE='$HF_HUB_OFFLINE'"
    echo "export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1"
    echo "export CUDA_VISIBLE_DEVICES=$g"
    # carry the LLM annotator env through if set
    [ -n "${OPENAI_API_KEY:-}" ]  && echo "export OPENAI_API_KEY='$OPENAI_API_KEY'"
    [ -n "${OPENAI_BASE_URL:-}" ] && echo "export OPENAI_BASE_URL='$OPENAI_BASE_URL'"
    [ -n "${LORIS_LLM_MODEL:-}" ] && echo "export LORIS_LLM_MODEL='$LORIS_LLM_MODEL'"
    for ((j=g; j<${#JOBS[@]}; j+=NGPU)); do
      echo "echo '### [gpu$g] START: ${JOBS[$j]}'"
      echo "${JOBS[$j]}"
      echo "echo '### [gpu$g] END rc=\$? : ${JOBS[$j]}'"
    done
    echo "$PY run_influence_mrr.py   # cpu-only, cheap; harmless if repeated"
    echo "echo '### [gpu$g] LANE COMPLETE'"
  } > "$lane_script"
  chmod +x "$lane_script"
  setsid bash "$lane_script" > "logs/all_gpu$g.log" 2>&1 &
  echo "gpu$g PID=$!" | tee -a logs/all.pids
done

echo "Launched. Monitor: tail -f logs/all_gpu*.log"
echo "When complete, build the report:  python gen_report_v2.py"
echo "(swap to results_auto.tex via  LORIS_REPORT_TEX=results_auto.tex python gen_report_v2.py )"
