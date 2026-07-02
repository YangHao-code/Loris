#!/usr/bin/env bash
# Prerequisites for the LORIS paper experiment matrix.
#  1. Python deps for the LoRA SLMs, the LLM annotator, and PEFT/QLoRA.
#  2. Download the two LoRA SLMs (Mistral-7B ungated; Llama-3-8B GATED — needs a
#     licensed HF token, else the LoRA arm falls back to Mistral-only + a logged note).
# Run ONCE before run_all_ngpu.sh.  Uses hf-mirror for the download.
set -uo pipefail
cd "$(dirname "$0")"

echo "== installing python deps =="
pip install -q "peft>=0.11" "bitsandbytes>=0.43" "openai>=1.0" "accelerate>=0.30" || \
  echo "WARN: pip install had issues — check manually"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
echo "== HF_ENDPOINT=$HF_ENDPOINT  HF_HOME=$HF_HOME =="

dl() {  # $1 = repo id
  echo "== downloading $1 =="
  huggingface-cli download "$1" --exclude "*.pth" "original/*" 2>&1 | tail -2 || \
    python - "$1" <<'PY'
import sys
from huggingface_hub import snapshot_download
try:
    snapshot_download(sys.argv[1], ignore_patterns=["*.pth","original/*"])
    print("OK", sys.argv[1])
except Exception as e:
    print("FAILED", sys.argv[1], e)
PY
}

dl "mistralai/Mistral-7B-v0.1"
dl "Qwen/Qwen2-7B"
# Llama-3-8B is gated (application pending). Once granted, add it to the pool via
# LORA_BOTH in exp_common.py and re-run: dl "meta-llama/Meta-Llama-3-8B" (needs HF_TOKEN).
if [ -n "${HF_TOKEN:-}" ]; then
  dl "meta-llama/Meta-Llama-3-8B"
fi
echo "== done. LLM annotator: set OPENAI_BASE_URL=https://api.siliconflow.cn/v1,"
echo "   OPENAI_API_KEY=<key>, LORIS_LLM_MODEL=deepseek-ai/DeepSeek-V3 for the LLM arm. =="
