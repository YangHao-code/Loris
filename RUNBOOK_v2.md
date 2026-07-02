# LORIS paper — experiment matrix runbook (Exp 1–5, seed 0, 7 datasets)

Datasets: `reuters21578, aapd, rcv1, bgc, arxiv, pubmed, hupd` (goodreads dropped; MIMIC unavailable).

## GPU count — recommendation

Rough seed-0 total ≈ **~130 GPU-hours** (dominated by the K/|M| sweeps, human-cost, and the dual-LoRA main runs).

| GPUs | wall-clock |
| :-- | :-- |
| 1 | ~5.5 days |
| 2 | ~2.7 days |
| **4 (recommended)** | **~1.5 days** |
| 8 | ~1 day (diminishing; only 7 datasets) |

**Rent 4× 32 GB GPUs (RTX 5090 / A100-class).** Each QLoRA SLM (Mistral-7B / Llama-3-8B, 4-bit) fits in ~16 GB; the pool trains them one at a time. The LLM-annotator arm calls an API (SiliconFlow), not the GPU.

## 0. One-time prerequisites
```bash
./prep_models.sh                    # installs peft/bitsandbytes/openai; downloads Mistral-7B (+ Llama-3-8B if HF_TOKEN set)
# LLM annotator (SiliconFlow, for the human-cost LLM arm):
export OPENAI_BASE_URL=https://api.siliconflow.cn/v1
export OPENAI_API_KEY=<your_siliconflow_key>
export LORIS_LLM_MODEL=deepseek-ai/DeepSeek-V3     # fallback: Qwen/Qwen2.5-72B-Instruct
```
- **Llama-3-8B is gated**: without `HF_TOKEN` the main LoRA arm falls back to Mistral-only (disclosed).
- Without an API key the LLM arm runs mock (empty → 100 % human fallback) — only useful as a plumbing check.

## 1. Launch everything
```bash
./run_all_ngpu.sh 4                 # N = number of rented GPUs
tail -f logs/all_gpu*.log           # monitor
```
Jobs are per-dataset, distributed round-robin to GPU lanes, `setsid`-detached (survive session restart). PIDs in `logs/all.pids`.

## 2. What runs (all seed 0, canonical config: `--track1_baseline blank --pattern_mode full`, per-cluster oracle-tuned router)
| Exp | Driver | Scope |
| :-- | :-- | :-- |
| 1 main table | `run_main_loris.py` | 7 datasets × {base, pool+LoRA} → `experiments/main_7ds/` |
| 1 baselines | `run_phase3.py` | pubmed, hupd (others already done) → `experiments/baselines/` |
| 1c influence/MRR | `run_influence_mrr.py` | aggregates router_tuning + RDG timing → `experiments/influence/` |
| 2 human cost | `run_humancost.py` | Γ sweep + RILL LLM/noLLM → `experiments/humancost/` |
| 3 scalability | `run_scalability.py` | \|D\|,\|Σ\|,noInc (bgc,aapd) → `experiments/scalability/` |
| 4 gate | `run_phase3_loris.py` | pubmed, hupd (others done) → `experiments/baselines_loris/` |
| 4 K-sweep | `run_ksweep_loris.py` | aapd,reuters,bgc × K=1..5 → `experiments/baselines_loris/ksweep/` |
| 4 \|M\|-sweep | `run_msweep_loris.py` | aapd,reuters,bgc × M=2..6 → `experiments/baselines_loris/msweep/` |
| 5 ablations | `run_ablations.py` | noS/noL/noInc + stages (aapd,bgc) → `experiments/ablations/` |

## 3. Build the report
```bash
python gen_report_v2.py                                  # → paper/results_auto_v2.tex (idempotent; placeholders for pending)
LORIS_REPORT_TEX=results_auto.tex python gen_report_v2.py # once complete, overwrite the headline tex
# PDF: pdflatex is NOT installed here — build paper/*.tex on a LaTeX machine, or `apt-get install texlive`.
```

## Notes / hygiene
- Disk: each cell purges `model_pool.pkl` + `*.npy` after parsing metrics (122 GB free now).
- If a LORIS process is killed, also kill orphaned loky workers (`pkill -f loky`) to avoid CPU starvation.
- Baseline substitutions (disclosed in the report): RAL→CoMAL, standalone LocalBoost→RuleCleaner (KDD'25), RulePrompt→reimpl. Group-C `localboost` = in-pipeline LocalBoost reimpl (distinct from standalone RuleCleaner).
- Everything is seed 0 (per decision). To add seeds later, pass `--seeds 0,1,2` to the phase3 drivers and add a `--seed` loop to the others.
