# Post-V1 source updates

The retained rule-discovery program remains usable unchanged. A separately
selected Aux can contribute an adaptive suffix without updating any old model
alias, ensemble, arbitration table or fallback. Existing manifests preserve
their historical defaults; the options below must be declared explicitly.

## Signed correction and calibration

`prepare-aux-extension` retains the original rules and prepares independent Aux
rules. The correction APIs also support negative heads through a separately
bound complement table. Feedback and threshold calibration account for the
polarity of each actual receiver occurrence, rather than treating every
predicate as a positive-label channel.

For saved checkpoints, [the residual Aux workflow](residual_aux.md) fits sparse
per-label positive and negative proposals, then checks their actual complete
Chase endpoint. The original endpoint provides frozen present/absent guards;
original trusted Gamma cells are outside the adaptive receiver scope. Negative
conclusions use the declared conflict/arbitration protocol: they are not an
unconditional deletion of an existing positive conclusion.

Calibration gates retain the previous committed state when a candidate fails.
The saved-checkpoint workflow requires both calibration macro and micro F1 to
not decrease and a strict gain in their sum. The ordinary training loop also
offers `loop.commit_gate_metric_policy: "aux_loris_joint"`, requiring an
independent-calibration gate, an isolated Aux, explicit disjoint fit/score row
IDs, and `loop.conflict_update_policy: "fresh_from_gamma"`. It checks the Aux
and deployed LORIS endpoint together. Completeness and FN tolerances are explicit
settings. The historical `mean_f1` gate remains the default.

`fresh_from_gamma` rebuilds a model-update closure from trusted roots under the
retained-unknown conflict protocol. It is a declared alternative to retaining
the previous derivation history, not an exact replacement for that history
policy. Proposal checks continue to use the configured answer-check protocol.

No gate uses unqueried test answers. Reusing calibration data repeatedly can
overfit that data; accepted calibration updates do not guarantee increasing
test F1. Optional SSL controls retain their explicit manifests and are not
enabled as a default accuracy improvement.

## Answer replay and querying

`feedback.verified_replay_policy: "cumulative_acquired"` replays accumulated
complete authorized answers plus the current observed partial answers. It
requires `new_verified_supervision: "full_batch"`. Targets come from Gamma;
the acquired-answer ledger avoids scanning every training cell just to identify
a small number of new supervised rows. `pending_batch` remains the default.

[Review queries](QUERY_POOL_ZH.md) can inspect already covered documents using
model/rule disagreement, unresolved conflicts and near-threshold missing-label
evidence. Cardinality alone is insufficient in the optional `evidence` mode.
This mode preserves the original training, answer validation and budget protocol.

## Performance APIs

- Calibration refreshes can reuse authenticated static lexical primitives.
  Dynamic model tables, rule outputs and reliability records remain separately
  bound and recomputed as required.
- Feedback validation checks shared probability tables once per object and
  polarity. Calibration computes a checked-batch identity once for all receiver
  occurrences.
- `prediction_logits_only: true` optionally skips unused training projection
  outputs during transformer prediction. Its default is false and historical
  config digests remain compatible.
- `loris_ssl.aux_feature_cache.load_frozen_parent` and `predict_cache` provide
  a prediction-only frozen checkpoint cache. One encoder pass returns original
  probabilities and normalized first-token features with restored row order.
  They verify model tensors, tokenizer, text order, label order and checkpoint
  identity. Float16 feature storage is an explicit approximation; probabilities
  remain float32. These APIs neither train a model nor certify that cached
  features constitute a trainable clone, and are not automatically enabled by
  `loris-label run`.

Backend source hashes are maintained in `SOURCE_MANIFEST.json`. Regression tests
cover signed channels, calibration binding, complete versus partial answers,
query budgets, frozen original models and the new proposal policies. Datasets,
checkpoints, learned rules, annotation services and experiment artifacts remain
caller-owned assets.
