# LORIS staged-redesign — end-to-end results (overnight A/B)

_Core goal: **human-efficient labeling** — (1) logic+ML rules lift the ML baseline, (2) the staged pipeline learns a real rule mix (not just ML), (3) RILL propagates a small human-label budget across the corpus via the similarity engine. Read the RILL curves as the headline._

_Generated 2026-06-07 10:12 from `experiments/e2e_*`._


## 1. Headline: baseline ML vs model+rules (test)

| Arm | base micro | +rules micro | Δmicro | base macro | +rules macro | Δmacro | n_rules |
|---|---|---|---|---|---|---|---|
| aapd_default | 0.6818 | 0.6997 | 0.0179 | 0.5463 | 0.6347 | 0.0885 | 90 |
| aapd_enriched | 0.6818 | 0.6466 | -0.0352 | 0.5463 | 0.6314 | 0.0851 | 89 |
| bgc_default | 0.7612 | 0.7777 | 0.0166 | 0.5384 | 0.6854 | 0.1470 | 85 |
| bgc_enriched | -- | -- | -- | -- | -- | -- | -- |

## 2. Rule mix per stage (did we learn more than ML?)

  - leave-one-out per-stage macro-F1 marginal: {'stage0_ml': {'n': 77, 'Σmarginal_macroF1': 0.333}, 'stage1_fn_add': {'n': 3, 'Σmarginal_macroF1': 0.0004}, 'stage2_fp_remove': {'n': 10, 'Σmarginal_macroF1': 0.0001}}
  - leave-one-out per-stage macro-F1 marginal: {'stage0_ml': {'n': 76, 'Σmarginal_macroF1': 0.355}, 'stage1_fn_add': {'n': 1, 'Σmarginal_macroF1': -0.0}, 'stage2_fp_remove': {'n': 10, 'Σmarginal_macroF1': 0.0013}, 'stage3_prop': {'n': 2, 'Σmarginal_macroF1': 0.0}}

## 3. Human efficiency — RILL F1 vs human-label budget (the paper's core)

- **aapd_default** (graph avg-degree {}):
  | budget | queries | micro | macro |
  |---|---|---|---|
  | 0 | 0 | 0.6997 | 0.6347 |
  | 10 | 10 | 0.7047 | 0.6405 |
  | 25 | 14 | 0.7070 | 0.6451 |
  | 50 | 14 | 0.7070 | 0.6451 |
  | 100 | 14 | 0.7070 | 0.6451 |
- **aapd_enriched** (graph avg-degree {'0.477': 8.9, '0.459': 12.0}):
  | budget | queries | micro | macro |
  |---|---|---|---|
  | 0 | 0 | 0.6466 | 0.6314 |
  | 10 | 0 | 0.6466 | 0.6314 |
  | 25 | 0 | 0.6466 | 0.6314 |
  | 50 | 0 | 0.6466 | 0.6314 |
  | 100 | 0 | 0.6466 | 0.6314 |
- **bgc_default**: no RILL sweep recorded.
- **bgc_enriched**: no RILL sweep recorded.

## 4. Enrichment A/B — does `--sim_target_degrees` help propagation?

- **AAPD**: enriched − default = Δmicro -0.0531, Δmacro -0.0034  (default 0.6997/0.6347 → enriched 0.6466/0.6314). Enriched rules: None.
  - RILL best-budget macro: default 0.6451 vs enriched 0.6314 (Δ -0.0138) — the propagation-engine comparison.
- **BGC**: incomplete (default=done, enriched=no_metrics).

## 5. To analyze tomorrow

- Does +rules beat the ML baseline on **macro** (long-tail) F1? (LORIS's claimed strength.)
- Is the rule mix balanced (ML + FN-add + FP-remove + propagation), or still ML-dominated? (per-stage LOO marginals above.)
- **RILL curves**: does macro-F1 rise with budget (propagation working), and does the enriched-bins arm propagate better? If curves are flat with 0 queries, Stage-3 produced no `sim` rules to propagate through — the lever is then lowering sim thresholds / `prop_label_source=gt`.
- Per-arm artifacts: `experiments/e2e_<arm>/<run>/` (metrics.json, rules.json, staged_pipeline_logs.json, rule_analysis*.json, rill_budget_sweep.json).
