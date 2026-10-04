"""Runtime evidence for idempotence checks and explicitly scoped rule revision."""

import copy
import json

import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, ML, Rule
from loris.labeling.runner import FeedbackCheckInvariantError, LabelingRun
from tests.test_labeling import fixture, synthetic_same_trainer


def test_checked_and_nocheck_have_identical_supervision_and_model_state(tmp_path):
    runs = []
    for policy in ("checked", "nocheck"):
        context, rules, manifest = fixture()
        manifest["feedback"]["candidate_check"] = policy
        run = LabelingRun(
            context=context,
            rules=rules,
            manifest=manifest,
            output=tmp_path / policy,
            trainer=synthetic_same_trainer,
        )
        report = run.run()
        with np.load(run.output / "supervision_00000/ARRAYS.npz") as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        runs.append((report, arrays, run))
    checked, unchecked = runs
    for key in checked[1]:
        np.testing.assert_array_equal(checked[1][key], unchecked[1][key])
    for key in (
        "checkpoint",
        "version",
        "state",
        "Gamma",
        "training_attempts",
        "committed_model_updates",
        "actual_label_requests",
        "normal_metrics",
    ):
        assert checked[0][key] == unchecked[0][key]
    assert checked[0]["actual_Chase_calls"] == unchecked[0]["actual_Chase_calls"] + 1
    assert (
        checked[0]["consistency_rejected_cells"] == unchecked[0]["consistency_rejected_cells"] == 0
    )
    assert not next(e for e in checked[2].events if e["kind"] == "supervision")[
        "candidate_check_is_independent_correctness_validation"
    ]


@pytest.mark.parametrize("failure_policy", ["halt", "reject_batch"])
@pytest.mark.parametrize("failure_kind", ["conflict", "facts_changed"])
def test_unexpected_check_failure_is_diagnostic_or_explicit_policy(
    tmp_path, failure_policy, failure_kind
):
    context, rules, manifest = fixture()
    manifest["feedback"]["check_failure_policy"] = failure_policy
    calls = []

    def trainer(**kwargs):
        calls.append(True)
        return synthetic_same_trainer(**kwargs)

    run = LabelingRun(
        context=context,
        rules=rules,
        manifest=manifest,
        output=tmp_path / failure_policy,
        trainer=trainer,
    )
    actual_chase = run.chase

    def injected_failure(*args, **kwargs):
        result, session, folder = actual_chase(*args, **kwargs)
        if kwargs["phase"] == "feedback_check":
            result = copy.copy(result)
            if failure_kind == "conflict":
                result.status = "conflict"
            else:
                result.positive_facts = result.positive_facts.copy()
                result.positive_facts[2, 0] = True
        return result, session, folder

    run.chase = injected_failure
    if failure_policy == "halt":
        with pytest.raises(FeedbackCheckInvariantError, match="entailed W failed"):
            run.run()
        assert not calls
        assert (run.output / "FAILED.json").is_file()
    else:
        report = run.run()
        assert report["consistency_rejected_cells"] == 1
        assert len(calls) == 1  # Only verified Gamma remains in this replay fixture.
        with np.load(run.output / "supervision_00000/ARRAYS.npz") as saved:
            assert not saved["weak_mask"].any()
            assert saved["true_mask"].sum() == 2
    diagnostic = next(run.output.glob("feedback_check_failure_*/DIAGNOSTIC.json"))
    record = json.loads(diagnostic.read_text())
    assert record["W_subset_successful_closure"]
    assert record["candidate_cells"] == 1
    assert record["policy"] == failure_policy
    assert (
        record["interpretation"]
        == "unexpected_closure_invariant_failure_not_evidence_of_wrong_labels"
    )
    np.testing.assert_array_equal(run.gamma.pos, context["gamma"].pos)
    np.testing.assert_array_equal(run.gamma.neg, context["gamma"].neg)


def test_default_fixed_program_conflict_never_queries_or_disables_instances(tmp_path):
    context, rules, manifest = fixture()
    # Permission and budget alone must not implicitly activate rule revision.
    manifest["strategy"] = "frozen"
    manifest["oracle"]["allow_known_source_review"] = True
    manifest["loop"].update(label_budget=3, source_review_budget=3)
    context["model_probabilities"]["aux"][0, 1] = 0.9
    rules += (Rule((Doc("x"), ML("x", "aux", "B", 0.8)), Label("x", "B")),)
    run = LabelingRun(
        context=context,
        rules=rules,
        manifest=manifest,
        output=tmp_path / "halt",
        trainer=synthetic_same_trainer,
    )
    report = run.run()
    assert report["status"] == "pending_initial_conflict"
    assert report["actual_source_reviews"] == report["actual_label_requests"] == 0
    assert not report["effective_ground_program_changed"]
    assert not run.decisions
    assert run.gamma.neg[0, 1]


def test_explicit_instance_review_reports_effective_program_change(tmp_path):
    context, rules, manifest = fixture()
    manifest["strategy"] = "frozen"
    manifest["oracle"]["allow_known_source_review"] = True
    manifest["loop"].update(source_review_budget=1, conflict_policy="review_instances")
    context["model_probabilities"]["aux"][0, 1] = 0.9
    rules += (Rule((Doc("x"), ML("x", "aux", "B", 0.8)), Label("x", "B")),)
    run = LabelingRun(
        context=context,
        rules=rules,
        manifest=manifest,
        output=tmp_path / "review",
        trainer=synthetic_same_trainer,
    )
    report = run.run()
    assert report["rule_revision_extension_used"]
    assert report["effective_ground_program_changed"]
    assert report["active_disabled_rule_instances"] == 1
    assert report["current_Gamma_has_valid_closure"]
    assert [r.to_dict() for r in run.rules] == [r.to_dict() for r in rules]
    assert run.gamma.neg[0, 1]


def test_frozen_acquires_annotation_with_unchanged_model_and_thresholds(tmp_path):
    context, rules, manifest = fixture()
    manifest["strategy"] = "frozen"
    manifest["loop"]["label_budget"] = 1
    answers = tmp_path / "answers"
    answers.mkdir()
    (answers / "00000002.json").write_text(
        json.dumps(dict(row_id="r2", labels=["A", "B"], targets=[1, 0]))
    )
    manifest["oracle"].update(kind="directory_simulated_human", directory=str(answers))

    def forbidden_training(**kwargs):
        pytest.fail("Frozen must not train")

    run = LabelingRun(
        context=context,
        rules=rules,
        manifest=manifest,
        output=tmp_path / "frozen",
        trainer=forbidden_training,
    )
    initial_version = run.version.identity
    report = run.run()
    assert report["actual_label_requests"] == 1
    assert report["version"] == initial_version
    assert report["training_attempts"] == report["committed_model_updates"] == 0
    assert report["inference_counts"]["chaseInc"] == 1
    assert run.gamma.pos[2, 0] and run.gamma.neg[2, 1]
