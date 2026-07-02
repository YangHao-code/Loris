# LORIS audit — verified fix plan (35 bugs + 20 opts vs current code)

_Triaged against current code this session (`AUDIT_TRIAGE.json` = raw). Source audit: `AUDIT_full_findings.json`. **Core goal: human-efficient labeling** — RILL doc-selection + propagation correctness is the priority cluster._

_Counts: 54 unique items — already_fixed 7, Tier1 6, Tier2 15, Tier3 14, Tier4 12._


## Execution constraint
The 4-arm e2e A/B is running; arms launch fresh `python -m loris`, so **no prediction-changing `loris/` edits until the A/B finishes** (would corrupt the comparison). Tier-1/2 safe + non-prediction-changing fixes go in an isolated git worktree (unit-tested, golden deferred); Tier-3 prediction-changing fixes are batched for one golden recapture AFTER the A/B; Tier-4 deferred.


## Tier 1 — paper-core, SAFE, non-prediction-changing (worktree NOW)  (6)

### opt#19 — Select eval threshold on held-out TRAIN (not full pool)
- status **still_open** | risk safe | PAPER
- evidence: `loris/eval/rill_efficiency.py:353 'sel_Y = Y[nonseed]' (held-out non-seed pool, not pure train); not per-label`
- fix: Split train into seed + hold-out; select threshold on hold-out; evaluate on test; implement per-label thresholding for multi-label

### bug#14 — Livelock: doc with empty oracle labels is never injected AND never added to skip_set → re-selected forever
- status **still_open** | risk safe | PAPER
- evidence: `loris/rill/rill.py:658-687: if labels_star is empty/falsy, skips trust_check block (line 658) and injection block (line 679); never reaches skip_set.add(x_star) unless trust_check failed; constant influence (BUG #1) makes U[0] re-selected i`
- fix: After oracle query, if labels_star is empty or all labels are filtered at line 682-684, add x_star to skip_set and continue to next iteration

### opt#10 — Wilson/Beta lower-bound confidence instead of Laplace+1 point estimate for corr_prec
- status **still_open** | risk safe | PAPER
- evidence: `discovery.py:1463,2957,3109 (all use Laplace: n_improved/(n_improved+n_worsened+1)). No Wilson or Beta bound anywhere in codebase. Removes need for _adaptive_corr_prec floor-lowering.`
- fix: Replace corr_prec = n_improved/(n_improved+n_worsened+1) with: from scipy.stats import binom; z=1.96; p_hat=n_improved/(n_improved+n_worsened); lower = (p_hat + z²/2n - z*sqrt(p_hat(1-p_hat)/n + z²/4n²)) / (1+z²/n) where n=n_improved+n_worsened. Or use a simpler Beta quantile: scipystats.beta.ppf(0.

### opt#8 — Round-bounded / anytime chase cascade (K~20) with set-equality early-stop
- status **partial** | risk safe | PAPER
- evidence: `multi_chase.py:883-886, 1177, 1327, 1472: max_rounds cap is in place (default 100) with fixpoint detection (status='fixpoint' check). However, the discovery validation at group_propagation.py:438 uses max_rounds=5, and orchestrator.py:1100 `
- fix: Unify: (1) pass max_rounds through orchestrator instead of hardcoding range(5); (2) add set-equality convergence detection (np.array_equal check) to multi_chase.py main loops, not just count-based termination; (3) consider K~20 as a reasonable default instead of 100 to match LBoost anytime semantics

### bug#26 — corr_prec computed only over FLIPPED docs + Laplace+1 + _adaptive_corr_prec lowers floor for small labels → tiny-evidence rules pass
- status **still_open** | risk moderate | PAPER
- evidence: `discovery.py:1447-1463 (corr_prec = n_improved/(n_improved+n_worsened+1) over will_change subset only); 116-123 (_adaptive_corr_prec lowers floor to 0.70/0.60 for labels with ≤1000/≤100 support). No support gate on flip set; only min_n_chan`
- fix: Add gate: n_changes >= min(k, 0.05*n_fire) to ensure the flip set is non-trivial relative to fires. Replace Laplace+1 with Wilson/Beta lower bound (closed-form, ~3 lines). Remove the _adaptive_corr_prec floor-lowering entirely or gate it: never lower precision floor below 0.65 regardless of support.

### bug#13 — Evidence-consistency check rejects any label not produced by TEXT-bodied rule — cross-doc propagated labels always rejected
- status **still_open** | risk moderate | PAPER
- evidence: `loris/rill/rill.py:281-312: _check_evidence_consistency filters related_rules to only text_preds (line 300-301), then requires rule.consequence==label AND all_text_fire. Propagation rules (SimPredicate/GroupPredicate/LabelPredicate bodies) `
- fix: For evidence checks on propagation consequences, accept rules producing the label regardless of text predicate presence; or route evidence through a proxy text rule if one exists upstream


## Tier 2 — safe quick wins / perf / dead-code (no golden)  (15)

### bug#35 — sim_graph antipodal sentinel -1.0 collides with real cosine=-1 pairs
- status **partial** | risk safe | 
- evidence: `sim_graph.py:99-102 (fill_diagonal -1.0 sentinel); connectivity floor added in commit 1e7d79b but antipodal collision unfixed`
- fix: Replace sentinel -1.0 with -2.0 (outside [-1,1] cosine range), or use separate boolean self-loop mask instead of sentinel in similarity matrix.

### bug#33 — Dead/inconsistent plumbing: legacy chase_inference import + minus branches
- status **still_open** | risk safe | 
- evidence: `loris/rules/rdl.py:254 'from chase_inference.multi_chase import MultiChase' (should be loris.chase); 'minus' op removed from discovery (5303302) but dead branches in multi_chase.py:346-356 remain`
- fix: Repoint rdl.py:254 to 'from loris.chase.multi_chase import MultiChase'; verify all 'minus' references deleted from multi_chase.py, discovery.py

### bug#32 — FreqPredicate eta hard-coded as train median; mostly redundant with MatchPredicate
- status **still_open** | risk safe | 
- evidence: `pattern_abstractor.py:1236-1244 (inline eta), :1324-1354 (dead _freq_eta method)`
- fix: Delete dead _freq_eta method. Document inline eta as NOT Optuna-optimized (fixed heuristic). Consider dropping FreqPredicate if ablations show no improvement over MatchPredicate.

### bug#28 — Default conflict_mode='negative_wins' silently resolves conflicts (pos & ~neg) instead of surfacing ⊥, undermining the paper's validity guarantee; conflicts are also duplicate-recorded every round
- status **partial** | risk safe | 
- evidence: `multi_chase.py:177 (default conflict_mode='negative_wins'); 668-688 (_handle_conflicts appends every (doc,label) pair for every round it is re-examined with no dedup); 696-698 (_build_predictions applies conflict_mode='negative_wins' silent`
- fix: For duplication: (1) track a `seen_conflicts: Set[Tuple[int,int]]` in _handle_conflicts and append only on first observation, OR (2) compute conflicts once at finalization from `np.where(lbl.pos & lbl.neg)`. For semantics documentation: add explicit comments that 'negative_wins' is a non-paper engin

### bug#27 — Transitivity machinery (sub/sup, _propagate_transitivity) is entirely inert; enable_transitivity=True changes nothing
- status **still_open** | risk safe | 
- evidence: `loris/chase/multi_chase.py:621-662: _update_subset_relations documented no-op (B-5 removed the only sub/sup source), so sub/sup always empty; _propagate_transitivity loops over empty sub/sup; paper's subset-transitivity propagation absent`
- fix: Either implement sub/sup population from comparison consequence (deferred C-9), or remove the enable_transitivity flag and dead loop code (lines 627-645, 858-861, 927-930)

### bug#25 — _DIAG_COUNTERS global race under ThreadPoolExecutor parallel clusters
- status **still_open** | risk safe | 
- evidence: `discovery.py:94-100 (_DIAG_COUNTERS module global), 2555 (_diag_reset at run_bo start), _diag_inc throughout; steps.py:858-865 (parallel run_bo in ThreadPoolExecutor). Each thread's _diag_reset() wipes in-flight counts of every other thread`
- fix: Make _DIAG_COUNTERS thread-local (use threading.local or pass as instance variable self._diag per run_bo invocation); aggregate after ThreadPoolExecutor.join(). Or: pass a fresh dict to evaluate_chase_configuration per run_bo and collect results after join().

### opt#20 — Delete dead code en masse
- status **still_open** | risk safe | 
- evidence: `perturbations.py still present; dynamic_router.py:324-352 single-label task_loss still defined; multi_chase.py:346-356 minus branches; discovery.py 'minus' refs`
- fix: Delete perturbations.py; remove single-label task_loss from HybridLoss; delete minus/replace branches from multi_chase/discovery; remove inert _structure_cache/_cache_key sites

### opt#17 — Dedup predicates by fire-set BEFORE screening; thread fire_counts_map into Step 3
- status **still_open** | risk safe | 
- evidence: `pattern_abstractor.py:441 (dedup by equality not fire-set), :1198-1234 (fire_counts_map computed but unused in Step 3), :771-791 (spaCy anchors not DF-pruned before O(k²))`
- fix: 1. Thread fire_counts_map to Step 3, dedup by fire-set (frozenset of fire indices) → keep one predicate per unique coverage pattern. 2. Add DF-prune on spaCy anchors before pair generation.

### opt#16 — Memoize estimate_influence(τ, U) within each RILL loop iteration — depends only on U and τ, not per-doc
- status **still_open** | risk safe | 
- evidence: `loris/rill/rill.py:147-150: estimate_total_influence loops over candidates and calls estimate_influence(τ_name, U_indices) for each label; called per-doc at rank_unlabeled line 160; U_indices is constant so each (τ, U) pair computed repeate`
- fix: Add @lru_cache(maxsize=1024) to estimate_influence or maintain a dict cache keyed by (label, tuple(U_indices)); reset cache at iteration boundary

### opt#13 — Pass pos_weight to imitation BCE; expose epsilon/sigma separately
- status **still_open** | risk safe | 
- evidence: `loris/pipeline/shared.py:375 HybridLoss(...) no pos_weight arg; loris/selection/dynamic_router.py:105,108 epsilon/sigma collapsed`
- fix: Pass pos_weight=(n_models/K-1) to HybridLoss; expose epsilon and sigma as separate hyperparams in config

### opt#12 — Vectorize group fire mask across all labels at once
- status **still_open** | risk safe | 
- evidence: `loris/rules/group_propagation.py:88-91 computes per-label iteratively; no single SpMV for all labels`
- fix: Compute M @ membership @ Mᵀ once for all labels; self-exclude diagonal via mask

### opt#11 — Delta-propagate SpMV for equal/group rules using only newly-changed label columns
- status **still_open** | risk safe | 
- evidence: `multi_chase.py:560 (Mᵀ@pos recomputed over FULL pos every round), 496-497 (same for group). The module docstring at line 21 mentions 'new-item tracking, delta-document filtering' but this is not implemented for the global passes (_eval_equa`
- fix: Track which (doc, label) pairs changed in the previous round and restrict the SpMV to only those columns: delta_labels = set(newly changed in last round); for attr in [attrs with active labels in delta_labels], recompute M @ M.T restricted to delta. Requires a delta_pos / delta_neg set maintained pe

### opt#9 — Seeded influence ranking with reproducible RNG instead of unseeded random sampling for large U
- status **still_open** | risk safe | 
- evidence: `loris/rill/rill.py:625: np.random.choice(len(U), ...) is unseeded; fixing requires storing a reproducible RNG state in RILLController.__init__`
- fix: Add self.rng = np.random.RandomState(seed or 42) to RILLController.__init__; replace np.random.choice with self.rng.choice

### opt#7 — Hoist all_old_f1s = _fast_per_label_f1(val_labels, existing_predictions) out of per-trial loop
- status **still_open** | risk safe | 
- evidence: `discovery.py:1517,1543,1561 (all_old_f1s recomputed every trial in evaluate_chase_configuration); also 3013 in batch_select. Should be hoisted once per round before the trial loop since it depends only on (val_labels, existing_predictions).`
- fix: Before the trial loop (before line ~1500 in evaluate_chase_configuration, and before line ~3000 in batch_select), compute all_old_f1s = _fast_per_label_f1(val_labels, existing_predictions) once and pass it as a parameter to the trial evaluation function, or store as self.all_old_f1s. Removes O(n_lab

### opt#6 — Batch ML proba per model in chase (not per-doc)
- status **still_open** | risk safe | 
- evidence: `loris/chase/multi_chase.py:291 calls predict_proba_single per doc; discovery.py:262 already batches at model level`
- fix: Pre-compute model proba (n_docs,n_labels,n_models) once per chase; slice instead of per-doc inference


## Tier 3 — prediction-changing (batch → ONE golden recapture, after A/B)  (14)

### bug#24 — Step-3 screening computes coverage+entropy on TRAINING corpus, not validation
- status **still_open** | risk moderate | PAPER PRED-CHG
- evidence: `pattern_abstractor.py:1413-1490`
- fix: Split pattern abstraction into fit(train_X, train_y) → generate patterns, then separate screen_patterns(val_X, val_y) → filter by coverage/entropy on validation set.

### bug#18 — Missing paper predicates: not_contains + ⊆/⊂ comparison consequences
- status **partial** | risk moderate | PAPER PRED-CHG
- evidence: `loris/predicates/_core.py:717 _VALID_LABEL_OPS={contains,eq,subset,strict_subset} but 'not_contains' absent; ComparisonPredicate (two-variable) not defined; deferred per user`
- fix: Add 'not_contains' op to LabelPredicate validated against lbl.neg; implement ComparisonPredicate(x,y,op∈{=,⊆,⊂}) for cross-doc label ops

### bug#17 — Equal-rule closure validated at 5 rounds (discovery) but applied to fixpoint (chase max_rounds=100); orchestrator fast-path runs range(5) while full chase runs to convergence → apply set can exceed validated set AND identical rule sets give different predictions across code paths
- status **partial** | risk moderate | PAPER PRED-CHG
- evidence: `group_propagation.py:438 (max_rounds=5 in discover_equal_rules); multi_chase.py:175 (max_rounds=100 default in MultiChase); orchestrator.py:1100 (fast-path Pass-3 equal loop uses `for _round in range(5)` hardcoded). Steps.py calls discover_`
- fix: Iterate discover_equal_rules to fixpoint (already does via the break condition) and ALSO apply a consistent max_rounds cap at test time. Either (1) pass max_rounds=None to discover_equal_rules and write `while True` with break on convergence (not range(max_rounds)), or (2) use a bounded-K approach (

### opt#15 — Coverage-greedy seeding (SALT select_initial_labeled_by_coverage + sqrt-coverage weights) wired into production RILLController
- status **still_open** | risk moderate | PAPER PRED-CHG
- evidence: `loris/rill/rill.py: no mention of 'coverage' or 'seeds_coverage'; eval/rill_efficiency.py:198-241 implements coverage-greedy but it's evaluation-only, not used in runtime RILLController`
- fix: Extract seeds_coverage and _coverage_sets from eval/rill_efficiency.py into loris/rill/oracle.py or a seeding module; call it in RILLController.run() if a seeding strategy is specified; track coverage weights for sample weighting

### opt#14 — Feed ML probability as soft literal into noisy-OR
- status **still_open** | risk moderate | PAPER PRED-CHG
- evidence: `loris/predicates/_core.py:688-697 MLThresholdPredicate collapses to 0/1; no soft probability fed into chase confidence`
- fix: Pass model proba directly into noisy-OR confidence; remove thresholding to 0/1 bit

### bug#12 — Global macro-F1 admission gate dilutes rare/tail-label corrections below threshold
- status **partial** | risk moderate | PAPER PRED-CHG
- evidence: `group_propagation.py:53-61,120-125 (global gate via min_f1_gain=0.0005); discovery.py:3020-3025 (per-label gate already implemented); group_propagation.py:267-270 (_eff_gate asymmetric gate exists but may not be used everywhere)`
- fix: Verify group_propagation admission always uses per-label _eff_gate(label_idx) at every decision point (lines 302-303, 385); if any site uses global min_f1_gain instead, redirect to per-label gate. Confirm group_propagation lines 303,385 are the only entry points and both use _eff_gate

### bug#11 — TrustChecker sandbox only handles LabelPredicate op=='contains', ignoring eq/subset/strict_subset
- status **still_open** | risk moderate | PAPER PRED-CHG
- evidence: `loris/rill/rill.py:410-414: mini-Chase loop checks `if pred.op == "contains":` and skips all other LabelPredicate ops (eq/subset/strict_subset); no handling for SimPredicate or GroupPredicate`
- fix: Extend sandbox mini-Chase to evaluate all LabelPredicate ops and cross-doc predicates (SimPredicate via text_fire_cache + sim_graph, GroupPredicate via membership); match the real chase _check_label_predicates logic

### bug#9 — Negative class [x.lbl]≠ never propagated across co-members; equal/group consequence writes only pos → manufactured conflicts + positive labels leak past corrective remove-rules
- status **still_open** | risk moderate | PAPER PRED-CHG
- evidence: `multi_chase.py:557-565 (_eval_equal_rules only does `lbl.pos |= new_pos`, never touches lbl.neg); 496-503 (same for group rules). The newly-added labels are never checked against lbl.neg, so a remove-rule's negation is overridden by equal/g`
- fix: In _eval_equal_rules and _eval_group_rules: (1) propagate neg symmetrically via neg_closure = (M @ (Mᵀ @ neg)) > 0; (2) guard newly &= ~lbl.neg before asserting, so a label in neg is not re-added; (3) optionally keep pos and neg as separate transitive classes per paper semantics. One-line fix for (2

### opt#5 — Add discriminative cluster attributes (TF-IDF / label-conditional n-grams)
- status **still_open** | risk moderate | PAPER PRED-CHG
- evidence: `loris/patterns/virtual_attributes.py:36-81 enable_cluster_attrs=False by default; no TF-IDF or label-conditional attrs implemented`
- fix: Add shared-phrase TF-IDF vocab and label-conditional n-gram attributes; wire into comparison consequence (x.A=y.A joins)

### opt#4 — Greedy max-coverage RILL document selection (submodular 1−1/e) — currently not implemented in RILLController
- status **still_open** | risk moderate | PAPER PRED-CHG
- evidence: `loris/rill/rill.py:614-648: rank_unlabeled() argmaxes influence scores; no coverage set tracking or submodular greedy. eval/rill_efficiency.py:198-241 has seeds_coverage() but it's not wired into RILLController.run()`
- fix: Port seeds_coverage logic from eval/rill_efficiency.py into InfluenceEstimator; track covered document sets per selected seed; update U to exclude covered docs after each injection

### bug#19 — RILL large-U sampling uses unseeded np.random.choice — query order non-reproducible when |U|>50
- status **still_open** | risk safe | PRED-CHG
- evidence: `loris/rill/rill.py:625: sample_idx = np.random.choice(len(U), sample_size, replace=False) uses global unseeded RNG, breaking determinism contract when |U|>50`
- fix: Create a seeded np.random.RandomState in __init__ or run(); use self.rng.choice instead of unseeded np.random.choice

### bug#31 — _corpus_vocab mutable instance state causes cross-cluster contamination
- status **still_open** | risk moderate | PRED-CHG
- evidence: `pattern_abstractor.py:1206 (reset per cluster), :1125 (read across all clusters)`
- fix: Make _corpus_vocab local to _step2_generate_patterns: compute per cluster, pass as parameter to morph regex builders instead of instance state.

### bug#30 — auto_regex emits over-broad unanchored patterns (e.g. \b[a-z]{3}\d\b)
- status **still_open** | risk moderate | PRED-CHG
- evidence: `auto_regex.py:264-303`
- fix: Add min-occurrence-per-cluster gate: only emit auto-regex if pattern matches ≥min_cluster_coverage fraction of source cluster. Anchor shapes to their source cluster context.

### bug#23 — _tree_seeded_rules drops 'absent' branch of decision-tree splits → body omits guard literal, generalizing rule
- status **still_open** | risk moderate | PRED-CHG
- evidence: `discovery.py:3438-3444 (tree path conversion handles go_left+thresh>=0.5 and not go_left+thresh>=0.5 but not 'not go_left and thresh<0.5' case, which is unhandled → predicate is dropped)`
- fix: In the tree path loop at discovery.py ~3435-3465, add an explicit check: if a path requires a predicate ABSENT (the right child of a <0.5 split where threshold says 'no match') and that predicate cannot be negated, skip the leaf via `continue` rather than emitting an incomplete body. Alternative: in


## Tier 4 — risky / research-grade (defer; need runtime measurement)  (12)

### bug#34 — Single-label task_loss (softmax+NLL) in multi-label LORIS
- status **still_open** | risk risky | PAPER PRED-CHG
- evidence: `loris/selection/dynamic_router.py:424 HybridLoss.forward() calls self.task_loss (single-label softmax+NLL at :324-352) instead of task_loss_multilabel (multi-label BCE at :354-382); LORIS is multi-label`
- fix: Route forward() to task_loss_multilabel; delete unused single-label task_loss path OR add a config flag to choose

### bug#15 — Pattern preselection scores by phi×coverage, NOT marginal F1 per paper
- status **still_open** | risk risky | PAPER PRED-CHG
- evidence: `shared.py:523-543`
- fix: Replace lines 523-543 with per-label F1 computation: for each candidate p, compute F1(p) = TP/(TP+FP+FN) per label, gate on min(F1_per_label) >= threshold. Remove phi/cov hybrid scoring.

### bug#8 — Imitation loss unweighted BCE; oracle FP-blind
- status **still_open** | risk risky | PAPER PRED-CHG
- evidence: `loris/pipeline/shared.py:375 calls HybridLoss(lambda_task=0.1, lambda_ent=0.1) with no pos_weight; line 306-307 oracle uses TP-overlap without FP penalty`
- fix: Pass pos_weight=torch.tensor([n_models/K-1]) to HybridLoss.__init__; replace oracle overlap with Jaccard (TP/(TP+FP+FN))

### bug#4 — _vectorized_staged_predict silently DROPS every Sim/Label/Group propagation rule → staged predictions omit Track-2 entirely
- status **still_open** | risk risky | PAPER PRED-CHG
- evidence: `discovery.py:3569-3600: SimPredicate/LabelPredicate set _skip=True (lines 3595-3596); GroupPredicate falls through to else→_skip=True (line 3597); if _skip: continue at line 3599 drops the whole rule. Used at steps.py:1444,1698-1700 for sta`
- fix: Route rule sets with Sim/Label/Group through MultiChase.chase_predict (the full chase), OR vectorize these predicates inside the fast-path: for Sim compute sim_mask inline, for Label/Group route to the chase's _check_label_predicates/_eval_group_rules logic with sim_graphs/label_state threaded throu

### bug#3 — LBL_0 seeded from MODEL predictions, not validated Γ — voids paper Validity (Thm 4)
- status **still_open** | risk risky | PAPER PRED-CHG
- evidence: `loris/rill/rill.py:561-572: MultiChase.run_persistent(docs, base_predictions) initializes LBL from base_predictions (model) without truth validation; paper §6.1 Validity requires Γ=ground_truth, but RILL propagates from model FPs/FNs`
- fix: Either (a) require ground-truth seed for RILL's Validity proof, or (b) modify Validity theorem to admit model-seeded chase with confidence bounds; document the departure from paper assumption

### opt#3 — Replace phi×coverage pattern pre-selector with marginal-per-label F1
- status **still_open** | risk risky | PAPER PRED-CHG
- evidence: `shared.py:535-543 (hybrid scoring); discovery.py:3024-3026 already per-label but pattern abstraction is not`
- fix: Refactor _select_top_predicates to compute F1_per_label = TP/(TP+FP+FN), gate on min(F1_per_label) >= threshold, rank by mean(F1_per_label). Align with discovery's per-label gate.

### opt#2 — Port LBoost noisy-OR: replace boolean OR with confidence accumulation
- status **still_open** | risk risky | PAPER PRED-CHG
- evidence: `loris/chase/multi_chase.py:565 'lbl.pos |= new_pos  # monotone OR' (hard boolean); no ∏(1-c_i) confidence fusion anywhere`
- fix: Maintain float (n_docs,n_labels) confidence matrix; update via conf_new = 1-∏(1-w_ij·conf_j); threshold at finalization

### bug#1 — RDG omits cross-document (sim/group/equal) edges — influence estimation blind to propagation rules
- status **still_open** | risk risky | PAPER PRED-CHG
- evidence: `loris/chase/rdg.py:46-60: _build() only scans isinstance(pred, LabelPredicate), ignoring SimPredicate/GroupPredicate/equal rules that carry labels across documents; estimate_influence(rill.py:102-109) only uses LabelPredicate downstream`
- fix: Extend RDG._build() to include SimPredicate/GroupPredicate/equal-consequence rules as edges; generalize bfs_from_label to traverse cross-doc paths via sim_graph/membership adjacency; recompute estimate_influence with actual reachable U doc set per cross-doc rule

### bug#22 — batch_select fire-mask maps predicates by id(pred) → after pickle/resume all lookups miss, silent O(n_docs) fallback
- status **still_open** | risk moderate | 
- evidence: `discovery.py:2746-2752 (_pred_to_idx[id(_p)]), 2824-2828 (lookup _pred_to_idx.get(id(pred))), 2918-2923 (fallback to per-doc bool(pred(...))); steps.py:903 (pickles all_trials), :844-846 (reloads); after unpickle predicates are new instance`
- fix: Replace id()-based keying with structural hash: compute predicate_key = hash(predicate_to_dict(pred)) or hash(repr(pred)) (excluding non-deterministic fields like sim_text); use this key everywhere _pred_to_idx is built and queried. Survives pickling and dedups value-equal predicates.

### opt#18 — Add load-balancing aux loss + antithetic variates to router MC
- status **still_open** | risk moderate | 
- evidence: `loris/selection/dynamic_router.py:104-118 no auxiliary loss; no antithetic variates in MC sampling`
- fix: Add aux_loss = -entropy(model_freq) to router objective; implement antithetic pairs in reparameterized sampling

### bug#16 — 'replace' consequence op is non-monotonic (evicts current pos snapshot to neg) → output depends on rule order, violating Church-Rosser (Thm 5)
- status **still_open** | risk moderate | 
- evidence: `multi_chase.py:604-613: replace branch does `current_pos = np.where(lbl.pos[doc_idx])[0]; for existing_lidx in current_pos: lbl.neg[doc_idx, existing_lidx] = True` — which depends on what was added before. The order-dependence empirical tes`
- fix: Remove the replace op entirely (verified: discovery never emits consequence_op='replace', only add/equal/remove, so it is inert in production). If it must be preserved, redefine monotonically (e.g., as add-L0 + explicit remove of a FIXED set independent of current pos) and add an order-independence 

### bug#21 — ML label-index mismatch: proba[:,l_idx] paired with label_list[l_idx] (learner order) not model order
- status **still_open** | risk risky | PRED-CHG
- evidence: `discovery.py:650-670 (greedy instantiation uses l_idx to index both proba and label_list; lines 669: label=label_list[l_idx]); 885-905 (beam same pattern); 2795-2796 (batch_select re-derives via model.label_index, so BO-scored mask differs `
- fix: At greedy/beam ML instantiation (discovery.py ~650,885): retrieve the model's label_list via _get_ml_model(model_name).label_list or model.label_index(); only iterate over labels that exist in BOTH model.label_list and label_list; build predicate with the label name, not via label_list[l_idx].


## Already fixed this session (verify, no action)

- bug#6 RILL final predictions = chase._build_predictions(lbl) using pos & ~neg — FIXED — loris/rill/rill.py:728: predictions = chase._build_predictions(chase.lbl); loris/chase/multi_chase.py:1298-1300: _build_
- bug#7 RDLSet.predict()/predict_on_base() treat replace/equal/unknown consequence_op as — rdl.py:165-166, 195-196: both predict() methods now explicitly check `if rule.consequence_op not in ('add', 'remove'): c
- bug#10 Consequence label/op NOT Optuna params + _trial_to_rdl KeyError + _cache_key Nam — discovery.py:1143-1160 (consequence_label/op now suggest_categorical), :1169 (_cache_key now assigned), :1141-1142 (comm
- opt#1 Make consequence label/op Optuna-suggested so TPE focuses budget on few improvab — discovery.py:1143-1160 (consequence_op and consequence_label are now trial.suggest_categorical with static candidate lis
- bug#29 Anchor ordering non-deterministic via set iteration (FIXED) — pattern_abstractor.py:813 (sorted(anchors) confirmed in commit 5303302)
- bug#5 Negated rescue predicate loses negation between discovery and application — loris/predicates/_core.py:352 defines negate:bool field; loris/rules/group_propagation.py:397 emits MatchPredicate(negat
- bug#20 best_f1 threshold tuned on test set — loris/eval/rill_efficiency.py:285 'HONEST (Phase-4 fix): threshold chosen to maximise micro-F1 on SEPARATE selection set
