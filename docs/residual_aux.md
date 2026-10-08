# Independent Aux with signed residual rules

This is an opt-in workflow. Existing `loris-label run` manifests retain their
behavior. An Aux is a separately owned model selected by the caller; the old
program can continue to reference any of its frozen ML models. Training the new
Aux does not refresh an old alias, ensemble, arbitration model or fallback.

Start with a frozen program and its original model tables. Compute its complete
endpoint once for each context. The added ordinary RDL rules use that endpoint
as a frozen model guard:

- Positive rules apply where the original endpoint lacks the label.
- Negative rules apply where the original endpoint contains the label.
- Known original Gamma cells are excluded from the new receiver scope.

The old endpoint is not a new Label premise or a Gamma fact. The new Aux's
complement table is rebuilt together with its probabilities for every
checkpoint. Negative rules use that current complement. Frozen guard tables
remain separate from the two changing Aux tables.

Per-label cuts are proposed on complete calibration answers. The backward
compatible default, `coarse`, uses the original fixed finite grid. The opt-in
`quantile41_rare_allowance_one` policy uses 41 quantiles of adjacent probability
midpoints separately for original-present and original-absent cells, plus fixed
low and broad probability anchors. It requires at least three removals and
allows at most the greater of one true-label loss or ten percent of the label's
calibration positives. The one-label allowance is explicit; the coarse default
still requires five removals and has no rare-label allowance.

Grid counts are computed with matrices; ties keep the higher positive cut, then
the lower negative cut. Labels are considered in decreasing per-label F1 gain,
and proposed changes must not reduce the running micro F1. Probabilities and
midpoint cuts stay in float64 after widening native float32 scores. The negative
complement is rebuilt without rounding back to float32 and is checked against
the intended inclusive `P <= cut` condition; an unrepresentable numerical
boundary fails instead of silently changing a rule.

These counts only screen proposals. The proposal can differ from actual inference: relation rules,
conflicts, the original arbitration and the original fallback still affect the
endpoint. Each candidate therefore runs a fresh complete cold Chase. It replaces
the incumbent only when calibration macro F1 and micro F1 both do not decrease
and their sum strictly increases (numerical tolerance `1e-12`). Equality retains
the previous checkpoint. No all-label positive union is enabled by default.

Application endpoints are computed for every candidate, including rejected
candidates. They are reporting outputs. Calibration alone chooses the committed
checkpoint. A candidate's current endpoint and the retained deployment endpoint
are separate; rejecting a model preserves the previous rules and probabilities.
Calibration and test scores describe different data scopes. Calibration gains
do not establish monotonic test improvement, and repeatedly reused calibration
data are not a fresh holdout.

## Python integration

`ResidualAuxWorkflow` accepts the same context dictionaries and typed `Rule`
objects used by the ordinary labeling engine. Contexts require ordered documents,
row IDs, native label names, original model tables and signed Gamma. The
calibration answers must be excluded from Gamma and from upstream model fitting.
Declare the original endpoint policy explicitly; fallback cuts must already be
fixed. The policy is used unchanged for the baseline and every candidate.

```python
from loris.discovery.manifest import file_record
from loris.labeling.residual_aux import ResidualAuxWorkflow

workflow = ResidualAuxWorkflow(
    contexts={"calibration": calibration_context, "application": application_context},
    rules=frozen_rules,
    alias="independent_aux",
    receiver_masks={"calibration": calibration_receiver_mask,
                    "application": application_receiver_mask},
    calibration_rows=calibration_A_mask,
    calibration_truth=complete_A_targets,  # Rows follow the masked context order.
    proposal_policy="quantile41_rare_allowance_one",  # Omit to keep coarse.
    policy={"conflict_policy": "unknown_arbitrate", "gamma_wins": True,
            "arbitration": frozen_arbitration_config,
            "fallback": frozen_fallback_config},
    cache_dir="run/cache",
)

# The existing backend owns training, saving and newly predicting this checkpoint.
# Both arrays must follow their complete context's row and native label order.
decision = workflow.consider(
    "round_1_student", file_record(saved_checkpoint),
    {"calibration": new_calibration_probabilities,
     "application": new_application_probabilities},
    training_receipt=fit_receipt,
)
current_prediction = workflow.current_outputs["application"]
deployed_prediction = workflow.outputs["application"]
```

`policy` has exactly `conflict_policy`, `gamma_wins`, `arbitration` and `fallback`.
The conflict policy is `halt` or `unknown_arbitrate`. Each optional model policy
is either `None` or a map with `alias`, `default_threshold` and `thresholds`.
Its alias must identify an original frozen table. The new alias and its generated
complement/guard aliases must be absent from the original context and program.

The proposal policy can also be an explicit map with `name`,
`minimum_removals` and `maximum_true_loss`. Its named quantile count and
rare-label allowance are fixed. Non-default screening bounds describe a
different declared experiment. The normalized configuration is part of the
frozen workflow binding, every candidate record and `COMPLETE.json`; changing
it during a run is rejected. Calibration answers alone determine the cuts.

The backend can use a strong pretrained checkpoint for the new Aux. Preserve its
actual architecture, complete encoder/head state, tokenizer, native label order,
supported-label mask, token policy and loss. Validate an independent zero-update
clone against the original model before training it. Reusing a frozen teacher's
prediction table does not prove that a trainable clone exists. This workflow
binds caller-supplied checkpoint and prediction records; it does not itself load
the checkpoint or certify that its forward pass produced those predictions.

Trusted full replay is the recommended supervised control. Conservative SSL is
optional and should be compared at the same update budget, seed, trusted order
and optimizer settings. Certify positive and negative teacher cells separately
without test answers. Unknown cells have zero gradient, and pseudo labels never
become Gamma. `masked_teacher_bce` provides a detached, confidence-weighted hard
BCE normalized by **all** batch cells; the supervised loss remains separate.
This optional loss does not certify a teacher or guarantee an accuracy gain.

## Portable saved-checkpoint CLI

```bash
loris-label residual-aux --protocol residual_protocol.json --output new_run
```

The JSON protocol has these fields:

```json
{
  "schema": "loris_residual_aux_protocol_v1",
  "contexts": {
    "calibration": {"context": {"path": "calibration.pkl", "sha256": "FILL_SHA256"},
                    "receiver_row_ids": ["cal_A_1", "cal_B_1"]},
    "application": {"context": {"path": "application.pkl", "sha256": "FILL_SHA256"},
                    "receiver_row_ids": ["app_1"]}
  },
  "program": {"path": "frozen_rules.json", "sha256": "FILL_SHA256"},
  "alias": "independent_aux",
  "proposal_policy": "quantile41_rare_allowance_one",
  "policy": {"conflict_policy": "unknown_arbitrate", "gamma_wins": true,
             "arbitration": {"alias": "old_model", "default_threshold": 0.5,
                             "thresholds": {}}, "fallback": null},
  "calibration_answers": {"path": "calibration_A.npz", "sha256": "FILL_SHA256"},
  "candidates": [
    {"candidate_id": "round_1_student",
     "checkpoint": {"path": "round_1.pt", "sha256": "FILL_SHA256"},
     "probabilities": {
       "calibration": {"path": "round_1_cal.npz", "sha256": "FILL_SHA256"},
       "application": {"path": "round_1_app.npz", "sha256": "FILL_SHA256"}
     }}
  ]
}
```

Paths resolve relative to the protocol. Fill hashes from the actual caller-owned
files. Context pickles use the project's authenticated local context format.
Each prediction archive contains exactly `row_ids`, `labels`, `probabilities`;
probabilities are finite float32/float64 values on the entire context axis.
The calibration archive contains exactly `row_ids`, `labels`, Boolean `targets`
and Boolean `observed`. Its rows are the complete selection subset in context
order, and every cell must be observed. There is no test-answer field.
`proposal_policy` is optional; existing protocols continue to use `coarse`.

Declare candidate order and training parameters before running the experiment.
The command authenticates each checkpoint and saved prediction file, records
every actual complete candidate closure, and writes the initial outputs,
per-candidate current and retained outputs/programs, final accepted program and
`COMPLETE.json`. Outputs are fresh: an existing directory is never overwritten.
This command does not train, request annotations, or evaluate test answers.
Use the saved initial/current/retained endpoints to report a fixed test trajectory
after the calibration decisions are locked; never choose a variant on that report.
