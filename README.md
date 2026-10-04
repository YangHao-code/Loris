# LORIS

LORIS is a logic-reinforced multi-label document labelling toolkit. This
checkout is a source-only submission: datasets, labels, model weights, learned
rules, experiment outputs, credentials, and machine-specific launchers are
provided separately by the caller.

## Install

```bash
python -m pip install -e ".[dev]"
export LORIS_SSL_SOURCE="$PWD/backends/transformer/src"
python -m pytest tests
```

The optional `aligned`, `torch`, `embed`, `datasets`, and `report` extras add
the dependencies needed by the corresponding model and preparation paths. The
package declares Python 3.8 or newer. Pretrained models must be available from
the caller's local model store; this source tree does not download or contain
weights.

## Current protocol defaults

- Discovery fits the complete supplied training split (`1/1`), preserving the
  caller's row order and explicit development exclusion manifest.
- Chase queries use one percent of the initial eligible pool, with a minimum of
  32 rows. Answers are complete vectors on the supplied label axis; the human
  path must provide the full ground-truth vector, while SiliconFlow may use the
  named `positive_labels` response format.
- The transformer Chase/Aux path selects BF16 on supported CUDA hardware and
  uses no autocast on CPU. Unsupported BF16 hardware is rejected explicitly;
  callers may select FP16 or FP32 explicitly.
- Initializer archives default to `evaluated_with_analytic_skips`: only
  candidates with a recorded analytic zero-positive certificate are omitted
  from the materialized archive. The logical search domain, seed stream, and BO
  budget are unchanged; use `all` when every evaluated candidate must be
  materialized.

Approximate admission remains an explicit opt-in. A caller must bind manifests,
assets, output directories, and any external annotation broker explicitly; no
server path or secret is assumed by the library.

## Aux initialization

`loris-label run --manifest manifest.json --output run --aux-initialization m0`
selects a fixed-initial-checkpoint experiment. Each round reloads the original
Aux checkpoint, then fine-tunes it using all accumulated authorized answers
and the currently valid logical corrections. Corrections are selected against
M0 probabilities with the current predicate thresholds, so correcting a cell
in an earlier model does not silently remove it from the next M0 training set.
Rule reliability and logical eligibility are recomputed for the current Chase;
obsolete or conflicting weak feedback is not blindly replayed. Unknown cells
have zero loss weight. This mode enforces one update per main closure and full
answer replay; it does not use the incremental changed-feedback filter.

`--aux-initialization previous` continues the preceding committed model and
uses the manifest's existing feedback/replay policy. Omitted flags preserve
historical manifest behavior. Thresholds and acquired answers accumulate in
both modes. Successful M0 candidates commit even when their discrete decisions
are unchanged; evaluation scores do not select or reject candidates. Learning
rates, epochs, BCE weighting, and final EMA selection remain manifest settings.
The M0 option reproduces the initialization idea, not the old ASL/three-epoch
experiment or its development-selected threshold protocol.

## Aux training controls

`loris-experiments aux-tuning` runs explicit train-only controls for 1, 3, or
5 epochs, supervised replay, fixed-confidence pseudo labels, and a per-label,
per-polarity adaptive-confidence variant. The adaptive variant borrows the
class-progress threshold idea from FlexMatch; it is offline multilabel
self-training, not a reproduction of FlexMatch's online weak/strong-view loss.
Unknown cells retain zero weight and verified labels override pseudo labels.

The registered multi-round controls compare fixed M0, previous weights with
full replay, and previous weights with changed feedback plus 5% trusted replay.
All arms use the same predeclared train-document query roster and label budget.
This is a controlled training screen, separate from full RDG/Chase experiments.
By default a frozen tuning partition selects the configuration. A second disjoint
partition confirms the selected configuration once after the search, with all
positive and negative tuning results preserved. This default mode excludes test
labels from hyperparameter selection. Inputs, original checkpoint and baseline probabilities are
bound by hashes; no dataset, server path, or credentials are assumed.

An explicitly declared `official_test_selection` manifest instead ranks the
controls on the complete official test set. Its receipts mark test use for
parameter selection and do not claim independent test generalization. Test
cells still have zero training weight in this offline screen. Interactive
Chase results separately identify acquired test answers and unqueried rows.

## Layout

`loris/` contains the package, `backends/transformer/src/loris_ssl/` contains
the retained transformer backend, `configs/templates/` contains caller-filled
templates, and `scripts/` contains portable preparation, discovery, experiment,
and validation entry points. Tests use synthetic fixtures and do not ship
learned experiment artifacts.

`backends/transformer/SOURCE_MANIFEST.json` records the retained backend source
hashes. Recompute its release hashes whenever backend source is changed.
