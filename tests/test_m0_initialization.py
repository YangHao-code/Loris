import copy

import numpy as np
import pytest

from loris.labeling.cli import configure_aux_initialization, validate
from loris.labeling.runner import LabelingRun, Version
from tests.test_batch_query_and_exports import configured
from tests.test_labeling import synthetic_same_trainer


def test_m0_reloads_initial_checkpoint_and_replays_accumulated_answers(tmp_path):
    context, rules, manifest = configured(tmp_path)
    manifest["loop"]["query_batch_size"] = 1
    configure_aux_initialization(manifest, "m0")
    validate(manifest)
    seen = []

    def train(**kwargs):
        seen.append((kwargs["checkpoint"], copy.deepcopy(kwargs["batch"])))
        return synthetic_same_trainer(**kwargs)

    run = LabelingRun(context=context, rules=rules, manifest=manifest,
                     output=tmp_path / "run", trainer=train)
    report = run.run()
    assert report["committed_model_updates"] == report["training_attempts"] == 2
    assert report["actual_label_requests"] == 2
    assert [parent for parent, _ in seen] == ["fixture", "fixture"]
    assert seen[0][1].true_mask[2].all()
    assert seen[1][1].true_mask[2:4].all()
    assert seen[1][1].weak_mask[1, 0]
    assert seen[1][1].audit["selection_probability_reference"] == "m0_with_current_thresholds"
    starts = [event for event in run.events if event["kind"] == "model_training_started"]
    assert starts[1]["previous_committed_checkpoint"]["path"] == "synthetic"
    assert starts[1]["training_parent"]["path"] == "fixture"


def test_m0_reselects_current_logic_even_after_current_model_learned_it(tmp_path):
    context, rules, manifest = configured(tmp_path)
    configure_aux_initialization(manifest, "m0")
    run = LabelingRun(context=context, rules=rules, manifest=manifest,
                     output=tmp_path / "run", trainer=synthetic_same_trainer)
    probabilities = np.array(run.version.probabilities, copy=True)
    probabilities[:2, 0] = .9
    current = Version(probabilities, {"path": "later", "sha256": "later"},
                      run.version.channels, "later")
    result, session, _ = run.chase(current, phase="m0_selection_test")
    batch = run.feedback(result, current, session=session)
    assert batch.weak_mask[1, 0]
    assert batch.targets[1, 0] == 1
    assert run.initial_version.probabilities[1, 0] < .8
    assert not batch.observed[2:].any()
    run.release(session)


@pytest.mark.parametrize("field,value", [
    ("trusted_replay_fraction", .05),
    ("trusted_replay", False),
    ("rule_feedback_scope", "new_or_changed"),
])
def test_m0_rejects_forgetting_previous_supervision(tmp_path, field, value):
    _, _, manifest = configured(tmp_path)
    configure_aux_initialization(manifest, "m0")
    manifest["feedback"][field] = value
    with pytest.raises(ValueError, match="m0 requires"):
        validate(manifest)


def test_invalid_initialization_is_rejected(tmp_path):
    _, _, manifest = configured(tmp_path)
    manifest["dynamic"]["initialization"] = "random"
    with pytest.raises(ValueError, match="Aux initialization"):
        validate(manifest)
