# LORIS Baseline Experiment Plan (remote server)

> **Goal.** Run **all** paper baselines (Groups A/B/C of §7) on **5 datasets**, on a separate
> (multi-GPU) server, and fill the `XX` / "TO BE FILLED" placeholders in the paper's results
> figures/tables. This document is the runbook: what each baseline is, what already exists in the
> repo, what must be built, how to keep numbers comparable to LORIS, what to ship to the new server,
> and the run order.
>
> **Locked decisions (2026-06-28):**
> - **Datasets:** `reuters21578`, `aapd`, `rcv1`, `bgc`, `arxiv` (all currently-prepared corpora).
>   *Note:* the paper's Table 2 names `MIMIC-III-50 / Reuters-21578 / RCV1-V2 / arXiv-Full`, but
>   `data/mimic` and `data/eurlex` are **empty** and `data/arxiv` is a **cs-papers sample**, not the
>   2.9 M-doc arXiv-Full. The big results table (`paper/results_auto.tex`) already uses
>   `reuters/aapd/rcv1/bgc`; we extend it with `arxiv`. (If the camera-ready must match Table 2
>   verbatim, MIMIC-III-50 needs credentialed PhysioNet access + a `prepare_mimic` — out of scope here.)
> - **LLM annotator:** OpenAI `gpt-4.1-2025-04-14`, `temperature=0` (paper setting). `LLMOracle` is a
>   stub today and `openai` is not installed — both must be set up on the new server.
> - **Hardware:** multi-GPU — parallelize datasets across GPUs; LoRA/Mistral runs comfortably.

---

## 1. Baseline inventory — what the paper asks for vs. what exists

The paper (`§7`, Figs. 4–5) compares LORIS against **three** baseline families. **None** of the named
comparison methods are implemented yet — the repo only has LORIS's *own* components plus the
single-model *reference* rows already in `results_auto.tex` (tfidf_svm, textcnn, bilstm, RoBERTa,
Mistral-LoRA). Verified by name-grep across `loris/`, `*.py`, `*.sh`.

### Group A — Model selection (swap LORIS's router; report downstream labeling Macro-F1)
Protocol (paper §7 "model selection"): *keep the full LORIS pipeline, replace only the model-selection
step with the baseline selector, run rule discovery + labeling, report Macro-F1.* Integration point is
`register_selected_models(pool, selected_idx, label_names)` (`loris/pipeline/orchestrator.py:1058`).

| Baseline | What it does | Repo status | Build effort |
|---|---|---|---|
| **Random_MS** | pick K models uniformly at random | **absent** (trivial) | **low** |
| **Indiv_MS** | pick K best by individual val accuracy | **partial** — per-model val acc already computed (the reference rows / `gen_report.py`); just needs a selector wrapper | **low** |
| **Hybrid_LLM** [Ding et al., *Hybrid LLM*] | route inputs to a model via an auxiliary *difficulty* predictor | **absent** | **medium** |
| **CAAS** [contextual-bandit selection] | model selection as a contextual bandit (LinUCB-style) | **absent** | **medium** |

LORIS's own selector to compare against = `loris/selection/dynamic_router.py` (the stochastic-smoothing
selection net). The base pool it selects from is fixed (see §3).

### Group B — End-to-end labeling (standalone methods; report Macro-F1)
| Baseline | What it does | Repo status | Build effort | LLM? |
|---|---|---|---|---|
| **Snuba** [Varma & Ré] | synthesize weak heuristics over features, train a label model | **absent** | **high** | no |
| **Self-Pretraining** [semi-sup self-training] | seed-train base model, pseudo-label unlabeled, retrain | **partial** — self-training arms exist in `loris/pipeline/iterative_orchestrator.py` / `loris/chase/iterative_chase.py` (the ②/②m arms) | **medium** | no |
| **RulePrompt** [Li et al., WWW'24] | prompt a PLM with self-iterative logical rules | **absent** | **high** | **yes** |
| **DeBERTa_SVM** | DeBERTa embeddings → SVM head | **partial** — `PretrainedEncoderClassifier` supports configurable `model_name` + heads; needs DeBERTa weights + SVM head | **low–med** | no |
| **DeBERTa_XGBoost** | DeBERTa embeddings → XGBoost head | **partial** — `classifier_head="xgboost"` exists (`requirements.txt:19`); needs `xgboost` install + DeBERTa weights | **low–med** | no |
| **GPT4** | zero-shot multi-label classification via `gpt-4.1` | **absent** (`LLMOracle` is a stub, `loris/rill/oracle.py:133-189`) | **medium** | **yes** |
| **BESRA** [Tan et al., AAAI'24] | HITL active learning, beta-scoring acquisition, human=GT | **absent** | **high** | no¹ |
| **RAL** [Wertz et al.] | reinforced active learning (RL acquisition), human=GT | **absent** | **high** | no¹ |

¹ BESRA/RAL query a *human*, simulated by revealing ground truth (paper §7: "we simulated human
labeling by assigning the documents their corresponding ground truth labels"). They do **not** need the
LLM at inference. Only **GPT4** and **RulePrompt** consume `gpt-4.1`.

### Group C — Pattern-selection ablation (swap LORIS's pattern selection; report Macro-F1)
Protocol (paper §7 "Varying pattern selection methods"): *keep LORIS, replace the pattern-selection step
inside rule discovery.* Integration point is the pattern/anchor selection in the `PatternAbstractor` →
rule-discovery hand-off (`loris/patterns/`, `loris/rules/`).

| Baseline | What it does | Repo status | Build effort |
|---|---|---|---|
| **Filter_MI** | select patterns by mutual information | **absent** | **low** |
| **Filter_χ²** | select patterns by chi-square | **partial** — `chi2` already used in `loris/models/intermediate/feature_density.py` | **low** |
| **WeShap** [Shapley pattern value] | Shapley-value pattern selection | **absent** | **med–high** |
| **LocalBoost** [Zhang et al., KDD'23] | iterative boosting-style pattern selection | **absent** | **med–high** |

### Plus: ablation variants the paper also reports (Exp-5, mostly already in `results_auto.tex` Table `tab:abl`)
`without LLM`, `without incremental chasing`, `without gradient approximation`, `without imitation loss`,
`pool = embedding only`, `gt-leak oracle`. These are **LORIS internal flags**, not external baselines —
already partially captured. Listed here for completeness; covered by `--no_*` flags + ablation runner.

---

## 2. Current repo state (verified)

**Entrypoint.** `python -m loris --dataset <name> [flags]` → `loris/pipeline/orchestrator.py:parse_args`
(`loris/__main__.py`). Canonical main-table flags (`run_main_table.sh:14`):
```
--top_labels 30 --max_trials 100 --two_val --no_router --rule_strategy batch \
--cluster_model_selection global --batch_metric_mode global_macro --track1_baseline blank
```
`--track1_baseline blank` ⇒ LORIS labels **from scratch**; models enter only as the rule predicate
`M(x,τ)`. Per-dataset caps used today (`run_main_table.sh:21-24`):

| Dataset | flag | test size | train cap | notes |
|---|---|---|---|---|
| reuters21578 | (full) | 1,985 | full | real text |
| rcv1 | `--subset_size 20000` | 10,000 | 20,000 | **hashed TF-IDF only, no raw text** → transformers/LLM inapplicable; `tfidf_svm` is the legit baseline (`results_auto.tex:12-14,45`) |
| aapd | `--subset_size 15000` | 987 | 15,000 | real text |
| bgc | `--subset_size 12000 --max_test_docs 8000 --big` | 32,840 | 12,000 | real text; largest test set |
| **arxiv** | *TBD* (propose `--subset_size 20000 --top_labels 30`) | TBD | TBD | cs-papers sample, author join-key; **sanity-check label cardinality in smoke phase** |

**Model pool** (`loris/pipeline/shared.py:111-180`, `init_models`): `tfidf_svm_unigram`,
`tfidf_svm_bigram`, `tfidf_lr_bigram`, `textcnn`, `bilstm`, `encoder_mlp` (RoBERTa-base) = **6 "base
pool"**; `+ lora_slm` (Mistral-7B, needs `--lora_model` and VRAM ≥ 20 GB) = **7 "pool + LoRA"**.
Encoder auto-picks first available of `roberta-base / distilbert / bert-base`. `--model_pool embedding`
drops the TF-IDF models; `--no_encoder` drops the encoder.

**Metric.** Micro/Macro-F1 via `sklearn.f1_score` at a global threshold chosen to max micro-F1 on val
(`loris/pipeline/orchestrator.py:561-579`, `loris/eval/rill_efficiency.py:283-308`). **Macro-F1 is the
headline.** All baselines must report the **same** metric on the **same** test split.

**Datasets.** `DATASET_REGISTRY` (`loris/data/prepare.py:981-1032`) covers
`aapd, eurlex, rcv1, bgc, reuters21578, arxiv`. On-disk readiness (`data/`):
`aapd` 58 M ✅, `bgc` 291 M ✅, `rcv1` 15 M ✅, `reuters21578` 56 M ✅, `arxiv` 289 M ✅ (incl.
`processed/train.csv`,`test.csv`); `eurlex` 0 ❌, `mimic` 0 ❌. Loading: `loris/data/loader.py:load_data`
(stable train/val/test split + initial-Γ sampling; `val_ratio` default 0.40).

**Environment (this box — must be reproduced on the new server):**
- Python **3.8.10**, torch **1.11.0+cu113** (CUDA 11.3), transformers **4.46.3**, sklearn **1.3.2**,
  numpy **1.22.4**, peft 0.13.2, accelerate 1.0.1, sentence-transformers 3.2.1 — present.
- **MISSING (must install for baselines):** `xgboost`, `openai`, `bitsandbytes`, (and `sentencepiece`
  for DeBERTa-v3). `vllm` absent (Mistral runs via HF, bf16 — not 4-bit; see `shared.py:170`).
- **HF weights cached** at `/root/autodl-tmp/hf_cache/hub`: `roberta-base`, `Mistral-7B-Instruct-v0.2`,
  `all-MiniLM-L6-v2`. **Not cached:** DeBERTa-v3, Llama-3-8B. (`HF_HOME` is set via run scripts/profile
  to `/root/autodl-tmp/hf_cache` — confirm on the new server; weights are loaded offline.)

---

## 3. Experimental protocol (keep baselines comparable)

All baselines **must reuse LORIS's data layer** so splits/seeds/labels are identical:
1. **Splits & labels:** call `loris.data.loader.load_data(...)` with the **same** `--dataset`,
   `--top_labels 30`, `--subset_size`, `--val_ratio`, and seed as the LORIS run. Never re-split.
2. **Metric:** Macro-F1 (headline) **and** Micro-F1, computed by the **same** global-threshold helper
   (factor it out of `orchestrator.py` into `loris/eval/metrics.py` and import everywhere).
3. **Supervision regime:** **full-supervision** for the main comparison (matches `results_auto.tex`
   Table `tab:big`). Weak-supervision / data-efficiency sweeps are a separate, later pass.
4. **Group A** (model selection): fix `K` and the pool `|M|` to LORIS's values; selector returns
   `selected_idx`; everything downstream is unchanged LORIS. Also run the paper's **Varying K**
   (K = 1…|M|) and **Varying |M|** sweeps (Fig. 5b,c).
5. **Group B** standalone classifiers (DeBERTa_*, Snuba, Self-Pretraining, GPT4): predict labels
   directly, score on the same test set. **rcv1 is excluded** for transformer/LLM ones (hashed text);
   report `tfidf_svm` there per the paper footnote.
6. **HITL (BESRA, RAL):** human = GT (simulated). Sweep human budget `K ∈ {0,50,100,200,400}` (same grid
   as LORIS `tab:human`); report Macro-F1 **and** #annotations (the human-cost axis, Fig. 4c–e).
7. **Group C** (pattern selection): fix everything except the pattern selector; report Macro-F1 on the
   same test set (paper compares LORIS pattern selection vs the 4 variants).
8. **Seeds:** ≥3 seeds where the paper reports σ (Group A K-sweep, HITL). Single seed acceptable for the
   deterministic main rows to start, then add seeds.

---

## 4. Code to add (new modules)

Keep all new code under a **`loris/baselines/`** package so it's isolated and golden-neutral (does not
touch the LORIS golden path unless a baseline flag is passed).

```
loris/baselines/
  __init__.py
  selectors.py        # Group A: Selector ABC + Random_MS, Indiv_MS, Hybrid_LLM, CAAS
                      #   each: select(pool, val_X, val_Y, K, docs) -> List[int] (model indices)
  snuba.py            # Group B: heuristic synthesis + label model over TF-IDF/anchor primitives
  self_pretrain.py    # Group B: standard self-training wrapper around a base classifier
  ruleprompt.py       # Group B: PLM-prompted self-iterative rules (uses LLM client)
  encoder_head.py     # Group B: DeBERTa(+RoBERTa) embeddings -> {svm, xgboost} head
  gpt4_zeroshot.py    # Group B: per-doc multi-label zero-shot via gpt-4.1
  hitl.py             # Group B: BESRA (beta-scoring AL) + RAL (RL acquisition), human=GT
  pattern_select.py   # Group C: Filter_MI, Filter_chi2, WeShap, LocalBoost selectors
  llm_client.py       # shared OpenAI gpt-4.1 client (retry, temp=0, JSON parse, cost log)
  run_baselines.py    # unified CLI: --baseline <name> --dataset <ds> [shared LORIS flags]
```

**Wiring:**
- **`llm_client.py`** also makes `LLMOracle` real (`loris/rill/oracle.py:133`): implement `query`,
  `query_with_evidence`, `query_paraphrased` against `gpt-4.1` (paper §6.2 prompt protocol).
- **Selectors (Group A):** add `--selector {router,random_ms,indiv_ms,hybrid_llm,caas}` to
  `orchestrator.parse_args`; when set, replace the router's `selected_idx` before
  `register_selected_models(...)` (`orchestrator.py:1058`). `router` = current behavior.
- **Pattern select (Group C):** add `--pattern_select {loris,filter_mi,filter_chi2,weshap,localboost}`;
  branch at the anchor→pattern hand-off in `loris/patterns/` so only the selection rule changes.
- **Standalone baselines (B):** `run_baselines.py` loads data via `load_data`, fits the baseline,
  writes the **same** results JSON schema as a LORIS run (see §8) so the aggregator is uniform.

**Reference implementations to port (do not reinvent):** Snuba → HazyResearch `reef`; RulePrompt →
authors' repo; LocalBoost / WeShap / BESRA / RAL → their papers' released code. Keep each a thin,
self-contained adapter over our `load_data` + metric.

---

## 5. Environment & dependency setup (new server)

1. **Match the core stack** (Python 3.8, torch 1.11+cu113 *or* a clean upgrade to torch 2.x+cu118 — if
   you upgrade, re-run the LORIS smoke test first to confirm parity). Easiest: clone the working venv.
2. **Install the baseline extras** into that env:
   ```
   pip install xgboost openai sentencepiece           # DeBERTa_XGBoost, GPT4/RulePrompt, DeBERTa tokenizer
   pip install bitsandbytes                            # optional: 4-bit LoRA if torch/cuda supports it
   # plus any reference-repo deps for snuba / besra / ral / localboost when you port them
   ```
   Capture as `requirements_baselines.txt`.
3. **Secrets / env:**
   ```
   export OPENAI_API_KEY=...           # gpt-4.1 access (GPT4 + RulePrompt)
   export HF_HOME=/path/to/hf_cache    # point at the copied weights
   export HF_HUB_OFFLINE=1             # weights are pre-copied; avoid network fetch (unset only to pull DeBERTa)
   ```
4. **HF weights to provision on the new server:**
   - Copy from here: `roberta-base`, `Mistral-7B-Instruct-v0.2`, `all-MiniLM-L6-v2`.
   - **Download there:** `microsoft/deberta-v3-base` (~440 MB) — needed for DeBERTa_SVM/XGBoost.
   - Only if a fine-tuned-SLM baseline is wanted beyond Mistral: `meta-llama/Meta-Llama-3-8B` (gated).

---

## 6. Transfer package (what to copy)

**Ship (≈ 16 GB, dominated by Mistral weights):**
```
Loris/                      # full repo MINUS the heavy dirs below
  loris/  *.py  *.sh  pattern_extraction/  paper/  method_docs/  requirements*.txt  pyproject.toml
data/reuters21578/ data/aapd/ data/rcv1/ data/bgc/ data/arxiv/   # ~710 MB (processed/ is the essential part)
hf_cache/hub/models--roberta-base
hf_cache/hub/models--sentence-transformers--all-MiniLM-L6-v2
hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2          # ~14 GB
```
**Exclude (regenerate or unneeded):** `Loris/experiments/` (170 MB+ of logs/old runs),
`Loris/.git` (optional), `data/{mimic,eurlex,arxiv_fake,mimic_fake,ogb,goodreads,arxiv_cite}`,
`data/corpus_*.txt`, all `__pycache__`.

**Recipe:**
```bash
cd /root/autodl-tmp
tar --exclude='Loris/experiments' --exclude='Loris/.git' --exclude='**/__pycache__' \
    -czf loris_code.tgz Loris
tar -czf loris_data.tgz data/reuters21578 data/aapd data/rcv1 data/bgc data/arxiv
tar -czf loris_hf.tgz hf_cache/hub/models--roberta-base \
    hf_cache/hub/models--sentence-transformers--all-MiniLM-L6-v2 \
    hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2
sha256sum loris_code.tgz loris_data.tgz loris_hf.tgz > SHA256SUMS   # verify after transfer
```
On the new server: extract preserving the `Loris/ + data/ + hf_cache/` layout (the run scripts and
`DATASET_REGISTRY` expect `data/<name>/processed/...` relative to repo root; set `HF_HOME` to the
extracted `hf_cache`). **First action there:** run the LORIS smoke (§7 Phase 0) to confirm parity.

---

## 7. Run order, parallelization, runtime

**Phase 0 — Smoke & parity (½ day).** On the new server, reproduce **one** LORIS main-table cell
(e.g. `aapd`) and confirm Macro-F1 ≈ `results_auto.tex` (aapd 0.647). Validate `arxiv` loads and has
sane label cardinality; finalize its `--subset_size`/`--top_labels`. Confirm `gpt-4.1` round-trips one
prompt and DeBERTa-v3 loads.

**Phase 1 — Reference single-model rows (cheap).** Re-confirm the per-model reference accuracies
(tfidf_svm, textcnn, bilstm, RoBERTa, Mistral-LoRA) on all 5 datasets — these feed Indiv_MS/Random_MS
and the table's reference block.

**Phase 2 — Non-LLM baselines (bulk; fully parallel across GPUs/datasets):**
- Group A: Random_MS, Indiv_MS, Hybrid_LLM, CAAS (+ Varying-K, Varying-|M| sweeps).
- Group B (no LLM): DeBERTa_SVM, DeBERTa_XGBoost, Snuba, Self-Pretraining.
- Group C: Filter_MI, Filter_χ², WeShap, LocalBoost.
- HITL: BESRA, RAL (human=GT; K-budget sweep).

**Phase 3 — LLM baselines (rate-limited, cost-gated):** GPT4, RulePrompt. Run **last** so prompts/parsing
are stable. **Exclude rcv1** (no raw text).

**GPU lanes (multi-GPU):** one dataset family per GPU; LoRA/Mistral and DeBERTa runs serialize within a
lane (24 GB each). LLM phases are API-bound, not GPU-bound — run on a CPU lane in parallel with GPU work.

**Rough runtime / cost:**
- LORIS-style runs: minutes (reuters/aapd) to ~1 h (bgc 12k) per cell; ×5 datasets ×~15 baselines, but
  embarrassingly parallel across GPUs → **~1–2 days** wall-clock for Phases 1–2 on a few GPUs.
- **GPT4 cost:** zero-shot hits **every** test doc — bgc 32,840 + rcv1(excl.) + aapd 987 + reuters 1,985
  + arxiv. At ~1 k input tokens/doc that is tens of millions of tokens. **Recommendation:** for the GPT4
  and RulePrompt rows, **subsample the test set** (e.g. 1,000–2,000 docs/dataset, fixed seed) and
  disclose it, or budget explicitly. Confirm the cap before launching Phase 3.

---

## 8. Outputs, metric collection, table filling

**Per-run results JSON** (uniform schema, written by every baseline; mirror what LORIS already emits):
```json
{"baseline": "snuba", "dataset": "bgc", "seed": 0, "regime": "full",
 "k_models": 3, "macro_f1": 0.xxx, "micro_f1": 0.xxx,
 "n_annotations": 0, "n_test": 32840, "wall_sec": 0, "notes": "..."}
```
Write under `experiments/baselines/<baseline>/<dataset>_<seed>.json`.

**Aggregation → paper.** Add `aggregate_baselines.py` (mirror `aggregate_main_table.py`) that collects
the JSONs into one matrix, then extend `gen_report.py` so `paper/results_auto.tex` gains:
- a **Group-B end-to-end** block (Snuba / Self-Pretraining / RulePrompt / DeBERTa_SVM /
  DeBERTa_XGBoost / GPT4 / BESRA / RAL rows) alongside the LORIS row → fills **Fig. 4a** (`tab:big` ext).
- a **model-selection** comparison (Random_MS / Indiv_MS / Hybrid_LLM / CAAS vs LORIS) → **Fig. 5a–c**.
- a **pattern-selection** comparison (Filter_MI / Filter_χ² / WeShap / LocalBoost vs LORIS) → **Exp-5**.
- **human-cost** curves (BESRA / RAL vs LORIS, Macro-F1 vs #annotations) → **Fig. 4c–e**.

Each filled cell maps 1:1 to a current `XX` in the paper's `figure*{Performance evaluation}` /
`figure*{model selection}` placeholders.

---

## 9. Risks & open items

- **rcv1 has no raw text** (sklearn hashed TF-IDF): GPT4, RulePrompt, DeBERTa_* and any
  transformer/LLM baseline are **inapplicable** there — report `tfidf_svm` as the legit baseline and
  footnote it (as LORIS already does).
- **arxiv is a cs-sample**, not arXiv-Full; **MIMIC-III-50 absent**. If reviewers require Table-2-exact
  datasets, that is a separate data-prep task (PhysioNet credential for MIMIC; full arXiv dump for
  arXiv-Full). Flagged, not in this scope.
- **GPT4/RulePrompt cost** (see §7) — decide test-subsample cap before Phase 3.
- **torch 1.11 vs 2.x:** `bitsandbytes` 4-bit is unavailable on cu113 (Mistral runs bf16). If the new
  server upgrades torch, re-run Phase 0 parity before trusting numbers.
- **Determinism:** fix seeds; for stochastic baselines (Random_MS, bandit, RL) report ≥3 seeds ± σ.
- **Port fidelity:** Snuba/BESRA/RAL/WeShap/LocalBoost should be ported from authors' code, not
  re-derived, to be defensible in review.

---

## 10. Priority / sequencing (build order, fastest defensible coverage first)

1. **Shared infra:** `loris/eval/metrics.py` (factor out F1), `llm_client.py`, `run_baselines.py`, results
   JSON + `aggregate_baselines.py`. *(unblocks everything)*
2. **Low-effort, high-value:** Random_MS, Indiv_MS (Group A); Filter_MI, Filter_χ² (Group C);
   DeBERTa_SVM, DeBERTa_XGBoost (Group B). → fills a large fraction of the table cheaply.
3. **Medium:** Hybrid_LLM, CAAS (Group A); Self-Pretraining (reuse iterative arms); GPT4 (Group B).
4. **High (port from reference code):** Snuba, RulePrompt, BESRA, RAL, WeShap, LocalBoost.
5. **Sweeps & seeds:** Varying-K, Varying-|M|, human-budget curves, multi-seed σ.
