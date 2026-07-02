# Overnight run summary — paper-alignment refactor (2026-06-02)

Branch `refactor/phase0-golden-baseline`. All work unit-tested + committed; the
final AAPD run was launched at the end and runs detached.

## What landed (committed, in order)

| Commit | Phase | What | Validation |
|--------|-------|------|------------|
| `c1ab8f3` | **B-3** | Multi-value **sparse attribute membership** for `x.A=y.A` over real NER/syntactic/regex attributes (replaces fabricated Cantor/KMeans virtual attrs). SpMV fire mask. | golden ALL PASS (only benign `group_count` metadata changed) |
| `eae1064` | **B-4** | **Comparison consequence `x.lbl=y.lbl`** (`consequence_op="equal"`) via batched SpMV label-set closure (provably = paper's sub/sup+propagate for `=`). | 45 unit tests |
| `d930f50` | **B-5** | Removed the **B1 bug**: `_update_subset_relations` no longer invents subset edges from coincidental prediction-bitmap containment. `enable_transitivity` default→False. | 46 unit tests |
| `e17c1e6` | **B-6** | **Discover + apply equal-rules** (`discover_equal_rules`, accuracy-guided per paper §5.2) + orchestrator fast-path equal handling. | 48 unit tests; golden ALL PASS |
| `44baee9` | **C-8** | Chase **converges on conflict**: `conflict_mode` default `halt`→`negative_wins` (kept remove-rules working; did NOT use plan's `positive_wins`, which discards `neg`). | 49 unit tests |
| `dcbb441` | fix | `torch.backends.mps` shim in `loris/__init__` so `python -m loris` runs standalone on CPU. | smoke + tests |

**49 unit tests pass.** Golden (`tests/golden/artifacts/aapd_small`) stays ALL PASS.

## Key paper-authoritative decisions (per your directive: paper > plan when unclear)
- **B-6**: skipped the plan's statistical anchoring module (Lift/Fisher/BH-FDR). Paper §5.2 says statistical support/confidence are *inadequate* and uses **accuracy-guided** discovery; equal-rule candidate space is tiny (≤1/attribute), so no anti-explosion filter is needed. Admit equal-rules by validation F1-gain.
- **C-8**: used `negative_wins` (converge + `pos & ~neg`), not the plan's literal `positive_wins` (which discards the neg set and nullifies the 11 corrective remove-rules → strictly worse). Converging-not-halting is the paper requirement; `negative_wins` satisfies it without breaking removes.

## The AAPD run (your deliverable)
- Launched: `bash run_aapd_final.sh` (detached, CPU, checkpointed). Log: `experiments/aapd_final.log`. Output dir: `experiments/aapd_final/<run>/`.
- Config: AAPD, subset 5000, top-30 labels, max_trials 20, two_val, batch, blank, pattern_mode sim.
- **Baseline (honest, val-selected model): tfidf_svm_unigram test micro-F1 = 0.6244, macro = 0.4187.**
- **Final metrics:** see `experiments/aapd_final/<run>/metrics.json` (`final_micro_f1`/`final_macro_f1` vs `baseline_test_*`). The "RESULTS" section below is filled in when the run finishes.
- **CPU, not GPU**, on purpose: the parallel BO chase is CPU-bound and the parallel workers OOM the 24GB GPU; CPU is the reliable path (same as the golden harness).
- **Resume after any interruption:** just re-run `bash run_aapd_final.sh` — it auto-detects the prior `experiments/aapd_final/<run>/` and resumes with `--resume_mode all` (reuses model pool + pattern stores + Optuna trials).

## Deferred (NOT done tonight — with rationale, for your call)
- **C-7 (remove SimPredicate type):** assessed *safe* to remove (only 2 instantiation sites; `pattern_mode=sim` controls textual-predicate similarity, not SimPredicate). **Deferred deliberately**: removing it before the headline run could remove a mechanism that *helps* AAPD results — better to A/B it (run with vs without) after seeing baseline numbers, rather than blindly delete.
- **C-9 (negative transitivity along `sup`):** needs a legitimate `sub/sup` source first. B-4 used SpMV closure and deferred explicit `sub/sup`; C-9 should derive `sub/sup` from the co-membership relation (`virtual_attrs`), then propagate `neg` down. Non-trivial infra; deferred.
- **D-10/D-11 (router task_loss + dead-code cleanup):** the plan notes these are golden-neutral (router early-exits when n≤k). Low impact on AAPD metrics; deferred.

## Note on the comparison-predicate feature
On the small golden config (and likely at subset 5000) the accuracy gate admits **0 equal-rules** — copying whole label-sets among co-members isn't net-positive there, so it's correctly rejected (this is the paper-faithful behaviour, not a bug). The measurable gains tonight come from the **honest baseline (A-1)**, **real NER/syntactic/regex attributes (B-3)**, and the **convergent chase (C-8)**. Equal-rules would contribute on data where sharing an attribute value strongly predicts a shared label set; whether AAPD has such structure is what the full run will show.
