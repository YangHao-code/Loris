import copy
from types import SimpleNamespace

import numpy as np
import pytest

from loris.labeling.cli import validate
from loris.labeling.feedback import Batch, incremental_batch
from loris.labeling.runner import LabelingRun
from tests.test_batch_query_and_exports import configured
from tests.test_labeling import synthetic_same_trainer


def test_delta_feedback_retains_fresh_answers_and_five_percent_old_replay():
    shape = (104, 2)
    positive = np.zeros(shape, bool)
    negative = np.zeros(shape, bool)
    positive[:101, 0] = True
    negative[:101, 1] = True
    gamma = SimpleNamespace(pos=positive, neg=negative)
    weak = np.zeros(shape, bool)
    weak[101:] = True
    targets = np.ones(shape, np.float32)
    targets[102, 0] = 0
    previous = np.zeros(shape, np.int8)
    previous[101:103] = 1
    batch = Batch(targets, weak.copy(), np.ones(shape, np.float32),
                  np.zeros(shape, bool), weak, {})
    filtered = incremental_batch(batch, gamma, np.ones(104, bool), [100], .05, previous)
    assert filtered.true_mask[100].all()
    assert np.count_nonzero(filtered.true_mask[:100].any(1)) == 5
    assert filtered.audit["old_trusted_rows_available"] == 100
    assert not filtered.weak_mask[101].any()
    assert filtered.weak_mask[102].tolist() == [True, False]
    assert filtered.weak_mask[103].all()
    assert (filtered.weights[~filtered.observed] == 0).all()
    assert (filtered.targets[filtered.true_mask] == positive[filtered.true_mask]).all()
    assert not (filtered.true_mask & filtered.weak_mask).any()


def test_each_closure_queries_before_one_update_and_keeps_previous_weights(tmp_path):
    context, rules, manifest = configured(tmp_path)
    manifest["loop"].update(query_batch_size=1, query_cadence="each_closure", max_inner_rounds=1)
    manifest["feedback"].update(rule_feedback_scope="new_or_changed",
                                trusted_replay=True, trusted_replay_fraction=.05)
    validate(manifest)
    seen = []

    def train(**kwargs):
        seen.append((kwargs["checkpoint"], copy.deepcopy(kwargs["batch"].audit)))
        return synthetic_same_trainer(**kwargs)

    run = LabelingRun(context=context, rules=rules, manifest=manifest,
                     output=tmp_path / "run", trainer=train)
    report = run.run()
    assert report["actual_label_requests"] == 2
    assert report["query_policy"]["batches"] == report["training_attempts"] == 2
    assert report["committed_model_updates"] == 2
    assert report["status"] == "all_documents_resolved"
    assert seen[0][0] == "fixture" and seen[1][0] == "synthetic"
    assert seen[0][1]["fresh_verified_batch_rows"] == [2]
    assert seen[1][1]["fresh_verified_batch_rows"] == [3]
    events = [event["kind"] for event in run.events]
    assert events.index("query_batch_completed") < events.index("supervision")
    assert seen[1][1]["unchanged_rule_feedback_cells_skipped"] == 1


@pytest.mark.parametrize("fraction", [-.01, 1.01, float("nan"), True])
def test_invalid_replay_fraction_is_rejected(tmp_path, fraction):
    _, _, manifest = configured(tmp_path)
    manifest["feedback"]["trusted_replay_fraction"] = fraction
    with pytest.raises(ValueError, match="trusted_replay_fraction"):
        validate(manifest)
