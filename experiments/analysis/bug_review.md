# LORIS bug review — findings & triage

Adversarial full-repo + router-deep-dive review. Candidates: **29**; verified **10 CONFIRMED**, **8 PLAUSIBLE**, 11 refuted.

Two independent verifiers per finding (refute-by-code + construct-failing-input); CONFIRMED = both agreed.

## Confirmed findings

### [FIXED] HIGH · router-selection · `loris/pipeline/shared.py:385`
**Defect:** hybrid_llm selector's argmax tie-break awards spurious 'wins' to the pool's first model (index 0), systematically biasing K-model selection toward it.
**Failure scenario:** In multi-label data many val docs have all-zero per-model Dice-F1 (every model wrong on that doc). `f1_dm.argmax(axis=1)` returns index 0 on these ties, so `wins[0]` is inflated by every such doc. Because `key = wins*1e6 + ...` makes `wins` the dominant term, the first pool model (e.g. tfidf_svm_unigram, or textcnn under drop_tfidf) is selected regardless of quality, and a genuinely better model at a higher index is dropped. Reproduced: with model 3 truly best on 50 docs and 50 all-zero docs, hybrid_llm ranks model 0 above every non-winner purely from tie mass.
**Triage:** hybrid_llm now credits wins only on docs with signal (has_signal mask).

### [FIXED] HIGH · router-selection · `loris/pipeline/shared.py:364`
**Defect:** _baseline_select_models catches every fit/predict exception and silently scores the model as val-macro 0 with all-zero preds, masking real failures.
**Failure scenario:** If clf.fit or clf.predict raises for any model (encoder OOM, HF download failure, shape error), the bare `except Exception` sets vp=zeros, vm=0.0 with no logging. For indiv_ms the model is merely ranked last, but for hybrid_llm/caas its f1_dm column is all zeros, and via the argmax tie-bias (line 385) an all-zero (broken) model can still be selected. The failure is invisible: the pipeline reports a normal LORIS run on a silently-degraded model set.
**Triage:** Selector fit/predict failures now logged (no silent val-macro-0).

### [DEFERRED] HIGH · rule-discovery · `loris/rules/discovery.py:1391`
**Defect:** In split mode, the val-side conjunction mask for a FreqPredicate is reconstructed from fire_masks[sidx] (the candidate's ORIGINAL op/eta), not from the grid-selected variant op/eta that beam_instantiate actually chose and stored in the rule body.
**Failure scenario:** Any run with fit_docs set (the orchestrator/pipeline split-mode path) and FreqPredicate candidates (pattern_mode=full). beam_instantiate picks e.g. variant (op='<=', eta=5.0) but records the candidate index whose precomputed mask is (op='>=', eta=2.0). At line 1391 fires uses fire_masks[sidx] = the wrong (op='>=',eta=2.0) mask. Verified concretely: for docs with 'the' counts [4,1,0] the candidate mask is [True,False,False] while the variant (op='<=',eta=5) mask is [True,True,True] — completely different. This yields wrong `fires`, wrong coverage, wrong n_improved/n_worsened and wrong F1-gain, and the _body_cache stores the variant predicate while it was scored on the unrelated candidate mask, so an emitted rule's body does not match the mask it was accepted on.
**Triage:** Deep beam_instantiate/FreqPredicate split-mode internals; affects rare Freq predicates (19/3572). Fix needs careful variant-mask threading — recommend a focused follow-up.

### [DEFERRED] HIGH · rule-discovery · `loris/rules/discovery.py:3037`
**Defect:** The batch_select accept-cross-check fires loop does not special-case SimPredicate/GroupPredicate/LabelPredicate as the Pass-1 val loop does, so propagation rules are evaluated with wrong (stub) semantics on accept_docs.
**Failure scenario:** batch_select called with accept_docs and a propagation rule whose body contains SimPredicate/GroupPredicate (+ LabelPredicate). id(pred) is not in _pred_to_idx (those masks are computed via SpMV in Pass-1, not stored in acc_pred_masks), so the else-branch calls pred(_make_proxy(d)). SimPredicate.__call__ and GroupPredicate.__call__ are stubs returning True (see loris/predicates/_core.py:842,883), so the sim/group step imposes no constraint, and any LabelPredicate is checked against the doc's OWN label set instead of neighbor labels via label_state. The accept fires mask thus uses semantics entirely different from the Pass-1 val fires mask, so the generalization gate accepts/rejects propagation rules based on a mask that does not represent the rule.
**Triage:** Propagation predicate stub semantics in batch_select accept cross-check; deep. Recommend focused follow-up.

### [DEFERRED] HIGH · chase-inference · `loris/chase/multi_chase.py:1040`
**Defect:** Fixpoint write-back sets doc.lbl from lbl.pos ignoring lbl.neg, mutating the caller's input Documents with labels that were removed; a subsequent chase/RILL run on the same docs re-seeds those removed labels as positive.
**Failure scenario:** orchestrator calls final_rdl_set.chase_predict(test_docs, ...) (line 1291) which at lines 1040-1045 does doc.lbl = {label_names[j] for pos[i,j]} — a removed label (neg set but pos still set) is written into doc.lbl. Then RILL's _rill.run(test_docs, base_predictions=combined) (line 1319) constructs a fresh MultiChase that seeds lbl.pos from doc.lbl (multi_chase lines 795-800), re-asserting the removed label as positive. The RILL predictions therefore differ from and contradict the chase predictions for the same rule set — a non-idempotent, order-dependent contamination between the two evaluation passes on shared mutable Document objects.
**Triage:** Chase write-back reseeds removed labels via lbl.pos ignoring lbl.neg; touches REMOVE semantics — risky to change blindly. See P0/P1/P3 (same area).

### [FIXED] HIGH · baselines · `loris/baselines/encoder_head.py:360`
**Defect:** The SVM encoder head returns raw LinearSVC decision_function margins, but the shared val threshold tuner only scans thresholds in [0.05, 0.95], so the flagship deberta_svm/roberta_svm baselines are thresholded at a wrong, non-comparable operating point.
**Failure scenario:** deberta_svm/roberta_svm: decision_scores() -> OneVsRestClassifier(LinearSVC).decision_function() returns raw margins roughly in [-3,+3] (natural decision boundary is margin>=0). These raw margins flow into score_from_scores -> common.tune_threshold (common.py:142), which only tries t in np.linspace(0.05,0.95,19). No threshold <=0 (the SVM's true boundary) or >0.95 can ever be chosen, so labels whose margin is in (0,0.05) are dropped and the cutoff sits far from the SVM optimum. The logreg/xgboost heads return proper [0,1] probabilities, so this systematically biases the SVM-head encoder baselines downward and makes them non-apples-to-apples with the other heads and with LORIS. Fix: squash SVM margins (e.g. expit) before thresholding, as loris/models/tfidf_classifier.py:238 already does, or tune the threshold over the margin's actual range.
**Triage:** SVM margins squashed via logistic to (0,1) for valid thresholding. NOTE: regenerate deberta_svm/roberta_svm baseline numbers.

### [FIXED] MEDIUM · rule-discovery · `loris/rules/rdl.py:347`
**Defect:** _is_redundant compares only the consequence label, ignoring consequence_op, so an ADD rule and a REMOVE rule with the same label and >80% body overlap are wrongly deduplicated, dropping a distinct opposite-effect rule.
**Failure scenario:** consequence_op_mode='both' (or a mix of add/remove rules). A rule body B -> +L already exists; a distinct rule with the same body B -> -L (remove) is a candidate. In discover() (discovery.py:2403) or batch_select() (discovery.py:3120), _is_redundant returns True because rule.consequence == candidate.consequence and intersection/union > 0.8, so the REMOVE rule is silently discarded even though it has the opposite effect and is not redundant.
**Triage:** _is_redundant now also compares consequence_op (ADD vs REMOVE not merged).

### [FIXED] LOW · router-selection · `loris/selection/perturbations.py:294`
**Defect:** sample_noise_with_gradients_special is called without the caller-supplied device, so noise is always created on cuda:0 regardless of the input tensor's device.
**Failure scenario:** perturbed_special/perturbed_extended pass no device to sample_noise_with_gradients_special, which defaults to cuda:0 when CUDA is available. If input scores live on cpu (CPU-only inference, or device explicitly set to cpu while a GPU exists), forward crashes at `input_tensor.unsqueeze(0) + sigma * additive_noise` with 'Expected all tensors to be on the same device'. Reproduced with a cpu score tensor. Only reachable via backend='perturbed'; the router hardcodes backend='custom', so it does not trigger in the active pipeline.
**Triage:** Noise now pinned to input_tensor.device (was cuda:0).

### [DEFERRED] LOW · rule-discovery · `loris/rules/discovery.py:1911`
**Defect:** _trial_to_rdl reconstructs the label-predicate structure from params['n_label_contains'], but for track='propagation' the objective sets LabelPredicate_contains=1 directly without ever calling suggest_int('n_label_contains'), so the key is absent and reconstruction drops the label predicate.
**Failure scenario:** track='propagation' and a _body_cache miss (fallback path at discovery.py:2400 / 2642, which logs a warning). evaluate_chase_configuration set structure['LabelPredicate_contains']=1 (line 1103) without a matching Optuna param, so trial.params has no 'n_label_contains'. _trial_to_rdl gets params.get('n_label_contains',0)=0, so beam_instantiate skips Phase-3 label selection and the reconstructed body omits the LabelPredicate that was present during evaluation, producing an RDL whose body/coverage differ from what was scored. (The docstring's claim that greedy is deterministic is also violated by feature dropout _dropout_pools.)
**Triage:** Propagation label-predicate reconstruction drops predicate when n_label_contains absent; rare path.

### [DEFERRED] LOW · data-pipeline · `loris/data/prepare.py:459`
**Defect:** In _convert_eurlex_json the first split-detection loop leaves json_path bound to the last iteration's value, so the None-guard at line 470 can raise (or pass) based on split ORDER rather than actual presence; the loop's result is dead and immediately re-derived.
**Failure scenario:** The loop at lines 459-468 sets json_path per split but only the final iteration's binding survives. If train.json exists but test.json does not, the loop ends with json_path=None (test was last) and line 470 raises FileNotFoundError even though train data is present and the intended 80/20 fallback (lines 511-517) would have handled the missing test split. Conversely if only test.json exists, the guard passes and the real loading (lines 481-506) then finds no train data and raises at line 509. The first loop is redundant with the second (481-491) and its early-exit logic is incorrect.
**Triage:** EUR-Lex dead code; eurlex not in the active dataset set.

## Plausible findings (one verifier; not yet acted on)

- **CRITICAL** · chase-inference · `loris/chase/multi_chase.py:359` — REMOVE consequences never suppress a label during chase propagation: every downstream label consumer reads only lbl.pos and never masks against lbl.neg, so a removed label still satisfies contains-predicates, group/sim y_masks, and equal-rule copies until the very last _build_predictions step.
- **HIGH** · chase-inference · `loris/chase/multi_chase.py:569` — _eval_equal_rules copies the full positive label set across co-members but never AND-masks against lbl.neg, so labels that a remove rule negated are still propagated to every co-member, and it also has no evaluated/dedup guard so it re-runs a full (n_docs x n_labels) SpMV every round until no new bit appears.
- **MEDIUM** · router-selection · `loris/pipeline/shared.py:315` — _build_multi_label_oracle_mask fills the imitation-target oracle with arbitrary/failed models via argpartition tie-breaking when overlap scores tie (including exception-zeroed columns).
- **MEDIUM** · chase-inference · `loris/chase/multi_chase.py:657` — The 'replace' consequence is order-dependent and non-confluent: it evicts whatever labels happen to be in pos at the instant it fires to neg, but labels added by later-firing rules in the same or later rounds are not evicted, so the final label set depends on rule application order.
- **MEDIUM** · baselines · `loris/baselines/selectors.py:95` — A model that raises during pool fit is silently replaced with all-zero test predictions and val_macro=0, so it can be swept into the majority-vote ensemble and vote 'no label' on every document, corrupting the Group A comparison rather than being excluded.
- **MEDIUM** · data-pipeline · `loris/pipeline/steps.py:933` — All post-split randomness (KMeans clustering, silhouette sampling, pattern abstraction, Track1/Track2 BO seeds) is hardcoded to 42 and never derived from hp.seed, so the paper's '3 runs with distinct seeds' only varies the data partition, not clustering/BO/pattern extraction.
- **LOW** · router-selection · `loris/selection/perturbations.py:274` — perturbed_special's func=None decorator branch returns functools.partial(perturbed, ...) — the WRONG base function — and drops hard_fwd.
- **LOW** · baselines · `loris/baselines/pattern_select.py:98` — The _proba rebuild logic and its comment assume OneVsRestClassifier drops single-class label columns, but for multilabel-indicator targets it always returns all n_labels columns, so the classes_-based remap fallback is dead code and would misplace columns if it ever ran.
