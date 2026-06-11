# Rule-Enrichment Levers — Implementation Status

Goal: admit more genuinely-useful Track-1 ("add") and Track-2 (propagation) rules
by making predicates **deeper/stronger** (they fail the precision gate because
they're too shallow), drawing on LBoost / SALT / REGAL / Gandalf. Every lever
defaults OFF (golden-neutral); the golden-master regression is **ALL PASS** with
flags off.

## DONE + validated (no LLM needed)

### Lever B — richer high-precision Track-1 "add" LFs  `--fn_add_richer`
SALT/REGAL template in `_synthesize_fn_add_rules` ([loris/pipeline/steps.py](loris/pipeline/steps.py)): on the
FN residual it also mines **2-phrase conjunctions** (`CooccurPredicate`, kept only
if they beat the best single-phrase precision), **negative guards** (`¬match`), and
**cross-representation ML guards** (an `MLThresholdPredicate` from a *different*
model family than the base — the anti-redundancy lever). Helpers `_pick_guard_models`,
`_best_ml_guard`. Tests: `tests/unit/test_fn_add_richer.py` (3, green).

### Lever D — multi-literal conjunctive joins  `--enable_multiattr_joins`
`x.A=y.A ∧ x.B=y.B → +τ` — two group stars AND'd, far more label-coherent than a
shallow single attribute (the direct fix for the diagnosis). Discovery: Phase 1b in
`discover_group_rules` ([loris/rules/group_propagation.py](loris/rules/group_propagation.py)) pairs signal-bearing
attributes from **different families** (`_attr_family`), capped to `max_multiattr_signal`,
keeps a pair only if the conjunction clears the gate AND beats both singles. Chase:
`_eval_group_rules`/`_eval_sim_rules` ([loris/chase/multi_chase.py](loris/chase/multi_chase.py)) now AND **all**
group/sim literals (was `next()` → first-only, silently dropping the 2nd). `batch_select`
already loops all body predicates (no change). Tests: `tests/unit/test_multiattr_joins.py`
(4, green — incl. the chase enforcing both literals).

### Regime reporting
- `Staged final` log now prints `[richer-LF=N, multiattr-join=M]` provenance.
- `--diagnostic_max_rules` (`_apply_diagnostic_overrides` in [loris/pipeline/orchestrator.py](loris/pipeline/orchestrator.py))
  composes existing knobs to admit far more rules for an **illustrative** writeup arm
  (prop_admit_on_precision + lower gates + uncapped + richer/multiattr). Logged loudly
  as "does NOT raise F1".

## DONE — scaffolding ready (needs the user's local Mistral-7B for the real GO/NO-GO)

### Lever C — LLM closed-ontology join attributes  `--enable_llm_attrs --llm_attr_cache PATH`
Architecture is **decoupled**: the LLM runs OFFLINE (heavy model can't share the
4090 with the encoder) and writes a cache `{doc_hash: [ontology values]}`; LORIS only
reads it (no vLLM dep added).
- `compute_llm_attributes` + `llm_doc_hash` ([loris/rules/virtual_attributes.py](loris/rules/virtual_attributes.py)),
  threaded through `compute_all_virtual_attributes` (3-tuple return unchanged when OFF →
  golden-neutral), re-exported in the legacy shim `rule_discovery/virtual_attributes.py`.
  The `llm` attribute auto-enrols into group/equal/**multi-literal** discovery.
- Offline producer `precompute_llm_attributes.py` — providers `transformers`
  (Mistral-7B-Instruct, fp16, greedy/deterministic, closed-ontology prompt; mirrors
  `LBoost/LLMAug/infer.py`) and `dummy` (deterministic keyword overlap, for plumbing).
- Probe arm `llm_attr_graph` in `propagation_joinkey_probe.py` (`--llm_cache`) — the
  **acceptance gate**: integrate only if `llm` PROP-GAIN ≫ `embed`, toward `label_oracle`.
- Tests: `tests/unit/test_llm_attributes.py` (4, green). Plumbing validated end-to-end
  with a dummy BGC cache (91k docs, 93% covered).

**To run the real GO/NO-GO (user's box):**
```
# 1) produce the cache with the local Mistral-7B
HF_ENDPOINT=https://hf-mirror.com python precompute_llm_attributes.py \
    --dataset bgc --provider transformers --model mistralai/Mistral-7B-Instruct-v0.2 \
    --ontology_from labels --k 3 --out experiments/llm_attr_bgc.json
# (needs: pip install accelerate; a torch new enough for Mistral — the current
#  env has torch 1.11 which is likely too old, hence vLLM in the LBoost env is preferred)

# 2) PROBE — accept only if `llm` ≫ embed toward oracle
python propagation_joinkey_probe.py --dataset bgc \
    --llm_cache experiments/llm_attr_bgc.json --out experiments/probe_llm_bgc.json

# 3) if it passes, run the pipeline with it
python3 -m loris ... --enable_llm_attrs --llm_attr_cache experiments/llm_attr_bgc.json \
    --enable_multiattr_joins   # the llm attr becomes a conjoinable literal
```

## DEFERRED — designed, not built (real LLM unavailable here to validate)

### Lever E — LLM-as-predicate `M_LLM(x,τ)` guard + CaMVo/LCB trust gate
The deepest precision guard. **Recommended low-risk design:** register the offline
LLM judgments (`{doc_hash: {label: confidence}}`, a new `--mode judgments` in
`precompute_llm_attributes.py`) as an **`LLMJudgeModel` in the ML model pool** so the
EXISTING `MLThresholdPredicate(llm_judge, τ, t)` reuses every eval path (batch_select,
`_vectorized_staged_predict`, chase) with no new predicate class — a shallow-but-high-
recall body becomes high-precision when AND-gated by the LLM judge. CaMVo = a Wilson/LCB
floor on the cached confidence (reuse the working 3-layer `TrustChecker`, [loris/rill/rill.py:225](loris/rill/rill.py)).
Deferred because it touches core eval paths and can't be validated without a runnable LLM.

### Lever F — weak-LF fusion via a generative label model
Stop demanding each rule independently beat the base; fuse many sub-threshold LFs with
a Snorkel-style `LabelModel` (Gandalf co-occurrence). The most weak-supervision-faithful
re-architecture; build only if B/D/C don't meet the count/macro goal.

## E2E findings (weak-Γ, --no_encoder, subset 3000, top 20, Γ=1000)
Mechanisms run correctly in the real pipeline; admission depends on whether the
data has the signal — the honest, multiply-confirmed wall.
- **BGC, all levers ON (dummy LLM cache):** `compute_llm_attributes → attr llm: 90 values`
  (Lever C plumbing ✓, 31 virtual attrs), `Phase 1b` runs over 620 candidates → **0**
  multiattr joins; `Staged final: 74 rules [richer-LF=0, multiattr-join=0]`; base→+rules
  micro **0.5324→0.5586 (+0.0262)**. Lever B = 0 here because BGC single-phrase genre
  keywords are already precise (conjunctions can't beat them → correct, not a bug).
- **BGC, `--diagnostic_max_rules` (gates→0.40, uncapped, prop_admit):** still **0**
  multiattr/richer (75 rules). ⇒ on BGC the binding constraint is **signal coherence,
  NOT the gate** — cluster/phrase/dummy-LLM attrs aren't label-coherent enough for even
  a 0.40-floor conjunction. This is precisely why the **real** probe-gated LLM is decisive.
- **AAPD, `--fn_add_richer`:** **Lever B FIRES** — `Staged final: 63 rules
  [richer-LF=1, multiattr-join=0]`; the admitted rule = a 2-phrase `CooccurPredicate`
  → `cs.lg` at prec 0.75 (AAPD's weaker tfidf base leaves a conjunction-coverable
  residual, as prior analysis predicted). Positive on-real-data proof for Lever B.
- **Takeaway:** B helps where single phrases are weak (AAPD ✓, not BGC). D needs a
  label-coherent join key, which current attrs lack everywhere → **run the Lever C
  probe with the real Mistral-7B; if `llm` doesn't beat `embed`/`prec_phrase` toward
  the oracle, D/C won't pay and the propagation headline should stay dropped.** Unit
  tests prove all mechanisms on synthetic data where the signal exists.

## Lever C REAL-Mistral-7B probe verdict (2026-06-11) — **NO-GO**
Ran the real Mistral-7B-Instruct-v0.2 (in BASE, torch 1.11 + transformers 4.46 — vLLM/torch2 NOT
needed; weights on the 50G `/root/autodl-tmp`) to assign each BGC test doc ≤3 values from the closed
145-genre label ontology → cache `experiments/llm_attr_bgc.json` (2000 docs, **89% non-empty**, 127
distinct genres). Probe `experiments/probe_llm_bgc.json`:

| B | direct macro | embed | prec_phrase | pseudo_label | **llm** | oracle (ceiling) |
|---|---|---|---|---|---|---|
| 50  | 0.091 | +0.005 | −0.012 | +0.012 | **+0.018** | +0.127 |
| 100 | 0.103 | −0.008 | −0.017 | +0.022 | **+0.016** | +0.206 |
| 300 | 0.154 | +0.039 | −0.027 | +0.037 | **+0.029** | +0.375 |
| 1000| 0.207 | +0.057 | +0.007 | +0.069 | **+0.049** | +0.509 |
(`llm` Δmicro ≈ 0/negative throughout: −0.028 → −0.002.)

**NO-GO** by the pre-registered rule: `llm` beats `embed` only at B≤100 (loses at B≥300), is **beaten by
the free `pseudo_label` baseline** at B≥100, recovers only **~8–14% of the oracle ceiling**, and hurts
micro. A real 7B LLM's genre assignment is NOT label-coherent enough to be a useful propagation join key
— the same info wall every prior text-derived key hit. ⇒ **Do not integrate Lever C; keep the
propagation headline dropped.** The probe gate worked: rejected before any pipeline cost. Repro:
`python precompute_llm_attributes.py --dataset bgc --provider transformers --texts_file
experiments/probe_te_bgc.json --ontology_from labels --texts_file ... ` then
`python propagation_joinkey_probe.py --dataset bgc --llm_cache experiments/llm_attr_bgc.json`.
(AAPD not run — weaker cross-doc structure ⇒ expected NO-GO too; env is ready if wanted.)

**Net standing after this run:** ship **Lever B** (Track-1 depth; fires on AAPD: +1 cooccur rule) + the
weak-Γ/macro/tail human-efficiency curve. Lever C/D cross-doc propagation do not pay on these datasets
(signal-limited, now confirmed with a real LLM). Lever E/F remain deferred (would face the same wall).

## The other question (why "full" baseline looked lower)
`--no_router` (single best model vs router-weighted ensemble, ~−0.016) + `--subset_size
12000` ("full labels on a 12k subset", not all 58k). Not a rule issue. Re-run only `gfull`
with the router on for the higher headline.
