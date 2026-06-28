# Baseline runbook — remote server (step-by-step commands)

Companion to `BASELINE_PLAN.md`. Dev box = where the code/data/weights live now
(`/root/autodl-tmp`). Remote = `ssh -p 22324 root@connect.bjb2.seetacloud.com`.
Code travels via git; **data + HF weights travel out-of-band** (they are not in git).

---

## Step 1 — Push the code (branch `baselines`)

The commit is already made locally on branch `baselines`. Push it (run on the dev box):

```bash
cd /root/autodl-tmp/Loris
git push -u origin baselines        # enter GitHub username + Personal Access Token when prompted
```

(There is no credential configured in the agent sandbox, so this push must be run
by you — or paste a PAT and the agent can push for you.)

---

## Step 2 — Get the code on the remote

```bash
ssh -p 22324 root@connect.bjb2.seetacloud.com
# if the repo already exists on the remote:
cd /root/autodl-tmp/Loris && git fetch origin && git checkout baselines && git pull
# otherwise clone it fresh:
# cd /root/autodl-tmp && git clone https://github.com/YangHao-code/Loris.git && cd Loris && git checkout baselines
```

---

## Step 3 — Transfer datasets (run ON the dev box; ~214 MB)

Only the `processed/` CSVs are needed (that's all `load_data` reads):

```bash
cd /root/autodl-tmp/Loris
for d in reuters21578 aapd rcv1 bgc arxiv; do
  rsync -avz --relative -e "ssh -p 22324" \
    data/$d/processed \
    root@connect.bjb2.seetacloud.com:/root/autodl-tmp/Loris/
done
```

`--relative` preserves the `data/<name>/processed/` layout on the remote.

---

## Step 4 — Transfer HF weights (run ON the dev box)

Needed for the baselines: **roberta-base** (encoder_head roberta_* variants) and
**all-MiniLM-L6-v2** (used by some LORIS embeddings). DeBERTa-v3 is downloaded on
the remote (Step 5). **Mistral-7B (14 GB) is optional** — only if you also run the
LoRA / LORIS `pool+LoRA` reference, not for the baselines themselves.

```bash
cd /root/autodl-tmp
rsync -avz -e "ssh -p 22324" \
  hf_cache/hub/models--roberta-base \
  hf_cache/hub/models--sentence-transformers--all-MiniLM-L6-v2 \
  root@connect.bjb2.seetacloud.com:/root/autodl-tmp/hf_cache/hub/

# optional (14 GB) — only if running LoRA/LORIS reference:
# rsync -avz -e "ssh -p 22324" \
#   hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2 \
#   root@connect.bjb2.seetacloud.com:/root/autodl-tmp/hf_cache/hub/
```

---

## Step 5 — Environment on the remote

```bash
ssh -p 22324 root@connect.bjb2.seetacloud.com
cd /root/autodl-tmp/Loris

# base deps (skip any already present in the env)
pip install -r requirements.txt
# baseline extras: xgboost, sentencepiece (DeBERTa-v3 tokenizer), openai
pip install -r requirements_baselines.txt

# point HF at the copied weights; download DeBERTa-v3 (needs network this once)
export HF_HOME=/root/autodl-tmp/hf_cache
python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/deberta-v3-base')"

# LLM baselines (gpt4, ruleprompt) — paper model gpt-4.1-2025-04-14, temp=0
export OPENAI_API_KEY=sk-...            # required for real GPT4/RulePrompt numbers
# export OPENAI_BASE_URL=...            # only if using a proxy/gateway
```

Quick check the suite imports and lists all 18 baselines:

```bash
export HF_HOME=/root/autodl-tmp/hf_cache
python -m loris.baselines.run_baselines --baseline list
```

---

## Step 6 — Run the baselines

Always run from the repo root with `HF_HOME` exported. Results are written to
`experiments/baselines/<baseline>__<dataset>__seedN.json`.

```bash
export HF_HOME=/root/autodl-tmp/hf_cache

# everything non-LLM on one dataset (model-selection + Group C + DeBERTa/snuba/self_pretrain/HITL):
python -m loris.baselines.run_baselines --baseline all --dataset bgc

# specific baselines:
python -m loris.baselines.run_baselines --baseline indiv_ms,random_ms,hybrid_llm,caas --dataset aapd --k 3
python -m loris.baselines.run_baselines --baseline deberta_svm,deberta_xgboost --dataset reuters21578
python -m loris.baselines.run_baselines --baseline filter_mi,filter_chi2,weshap,localboost --dataset aapd --top_k 200
python -m loris.baselines.run_baselines --baseline besra,ral --dataset bgc --budget 400 --batch 50

# LLM baselines (cost-gated: subsample the test set; --max_test default 1000):
OPENAI_API_KEY=sk-... python -m loris.baselines.run_baselines --baseline gpt4 --dataset reuters21578 --max_test 1000
OPENAI_API_KEY=sk-... python -m loris.baselines.run_baselines --baseline ruleprompt --dataset aapd --max_test 1000

# multi-seed (selectors / HITL report σ): rerun with --seed 1, --seed 2 …
```

**Per-dataset caps are applied automatically** (`reuters21578` full, `rcv1` 20k,
`aapd` 15k, `bgc` 12k/test 8k, `arxiv` 20k/test 8k). Override with
`--top_labels/--subset_size/--max_test_docs` if needed.

**Skips by design:** transformer/LLM baselines (deberta_*, gpt4, ruleprompt) raise
`NotImplementedError` on **rcv1** (hashed TF-IDF, no raw text) — that's expected;
`tfidf_svm` / the filter/snuba baselines are the legitimate rcv1 comparison.

Across all GPUs, run one dataset per GPU lane in parallel; LLM jobs are API-bound
(run them on a CPU lane). See `BASELINE_PLAN.md` §7 for the full schedule.

---

## Step 7 — Aggregate into a comparison table

```bash
python aggregate_baselines.py --csv baseline_results.csv
```

Prints macro/micro-F1 by baseline × dataset (averaged over seeds) and writes a CSV.
Fold these rows into `gen_report.py` / `paper/results_auto.tex` to fill the paper's
`XX` baseline cells (Figs. 4–5 / Exp-5). See `BASELINE_PLAN.md` §8 for the mapping.

---

## Notes / caveats (see `BASELINE_PLAN.md` §9 for detail)

- The research-method ports (snuba, besra, ral, ruleprompt, weshap, localboost) are
  **first-pass, runnable** implementations — fidelity to the original papers should
  be validated before camera-ready.
- The Group-A selectors here are the **ensemble-of-selected proxy**; the
  paper-faithful "LORIS-with-selector" path (feed the K selected models into the
  chase) is a follow-up `--selector` flag on the orchestrator.
- `gpt4`/`ruleprompt` run in **mock mode** (empty preds) if `OPENAI_API_KEY` is
  unset — do not report mock numbers.
