#!/usr/bin/env bash
# Wave-2: LoRA main arm (Mistral-7B + Qwen2-7B in pool) + human-cost LLM arm.
# Shares the 4 GPUs with the running base matrix (2 LORIS jobs/GPU max here).
# Idempotent: refuses to double-launch if logs/wave2_gpu*.log already exist.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
if ls logs/wave2_gpu*.log >/dev/null 2>&1; then
  echo "wave2 already launched (logs/wave2_gpu*.log exist) — abort."; exit 0
fi
NGPU=4
PY=python3
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}" HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OPENAI_BASE_URL="https://api.siliconflow.cn/v1"
export OPENAI_API_KEY="sk-qtvgmcnfolftsuvjmfjvvvibkytpgyhpuqcbmxostpinfrpt"
export LORIS_LLM_MODEL="deepseek-ai/DeepSeek-V3"

ALLDS="reuters21578 aapd bgc arxiv pubmed hupd rcv1"   # rcv1 last
JOBS=()
# LoRA main arm first (GPU-heavy), then the API-bound LLM human-cost arm.
for ds in $ALLDS; do JOBS+=("$PY run_main_loris.py --datasets $ds --arm lora"); done
for ds in $ALLDS; do JOBS+=("$PY run_humancost.py --datasets $ds --pool base --arms llm"); done
echo "wave2 total jobs: ${#JOBS[@]}"

: > logs/wave2.pids
for ((g=0; g<NGPU; g++)); do
  ls="logs/_wave2_lane_$g.sh"
  {
    echo "#!/usr/bin/env bash"
    echo "cd '$(pwd)'"
    echo "export HF_HOME='$HF_HOME' HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1"
    echo "export CUDA_VISIBLE_DEVICES=$g"
    echo "export OPENAI_BASE_URL='$OPENAI_BASE_URL' OPENAI_API_KEY='$OPENAI_API_KEY' LORIS_LLM_MODEL='$LORIS_LLM_MODEL'"
    for ((j=g; j<${#JOBS[@]}; j+=NGPU)); do
      echo "echo \"### [w2-gpu$g] START: ${JOBS[$j]}\""
      echo "${JOBS[$j]}"
      echo "rc=\$?; echo \"### [w2-gpu$g] END rc=\$rc : ${JOBS[$j]}\""
    done
    echo "echo '### [w2-gpu$g] WAVE2 LANE COMPLETE'"
  } > "$ls"
  chmod +x "$ls"
  setsid bash "$ls" > "logs/wave2_gpu$g.log" 2>&1 &
  echo "w2-gpu$g PID=$!" | tee -a logs/wave2.pids
done
echo "wave2 launched. Monitor: tail -f logs/wave2_gpu*.log"
