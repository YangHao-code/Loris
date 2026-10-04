"""Complete empty external label sets use real Chase, never fabricated positives.

All providers and documents below are synthetic. These are interface/termination
regressions, not dataset experiments or model-quality tests.
"""

import json

import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, RefGamma, Rule, Text
from loris.labeling.runner import LabelingRun
from loris.predicates import MatchPredicate
from tests.test_labeling import fixture


def one_pending_document():
    context, rules, manifest = fixture()
    positive = context["gamma"].pos.copy()
    positive[3, 0] = True  # The scoring-only document is already resolved.
    context["gamma"] = RefGamma(positive, context["gamma"].neg.copy())
    manifest["strategy"] = "frozen"
    manifest["query"]["strategy"] = "random"
    return context, rules, manifest


def forbid_training(**kwargs):
    pytest.fail("these frozen annotation regressions must not train a model")


def complete_empty_gamma(context, row=2):
    positive, negative = context["gamma"].pos.copy(), context["gamma"].neg.copy()
    positive[row] = False
    negative[row] = True
    context["gamma"] = RefGamma(positive, negative)


@pytest.mark.parametrize("budget", [1, 2])
@pytest.mark.parametrize(
    "kind", ["directory_llm_then_human", "directory_human", "directory_simulated_human"]
)
def test_full_zero_answer_finishes_once_with_or_without_spare_budget(tmp_path, budget, kind):
    context, rules, manifest = one_pending_document()
    manifest["loop"]["label_budget"] = budget
    directory = tmp_path / "answers"
    directory.mkdir()
    manifest["oracle"].update(kind=kind, directory=str(directory))
    answer = dict(row_id="r2", labels=["A", "B"], targets=[0, 0])
    if kind == "directory_simulated_human":
        filename = "00000002.json"
    else:
        stage = "llm" if kind == "directory_llm_then_human" else "human"
        filename = "00000002." + stage + ".json"
        answer.update(provider="simulated_" + stage, provenance="Synthetic empty-answer regression")
        # A successful LLM check must not consult the human fallback. NoLLM must
        # likewise ignore the unrelated malformed LLM proposal.
        ignored_stage = "human" if stage == "llm" else "llm"
        (directory / ("00000002." + ignored_stage + ".json")).write_text("invalid JSON")
    (directory / filename).write_text(json.dumps(answer))

    run = LabelingRun(
        context=context, rules=rules, manifest=manifest, output=tmp_path / "run",
        trainer=forbid_training,
    )
    report = run.run()

    assert report["status"] == "all_documents_resolved" and report["task_complete"]
    assert report["actual_label_requests"] == 1 and report["acquired_documents"] == [2]
    assert report["training_attempts"] == report["actual_optimizer_updates"] == 0
    assert not run.gamma.pos[2].any() and run.gamma.neg[2].all() and 2 in run.answered
    requests = [event for event in run.events if event["kind"] == "label_request"]
    assert len(requests) == 1 and requests[0]["row"] == 2
    with np.load(run.output / "TRUSTED_GAMMA_FINAL.npz") as saved:
        assert not saved["positive"][2].any() and saved["negative"][2].all()
    with np.load(run.output / "FINAL_P.npz") as saved:
        assert not saved["logical_positive"][2].any()
        assert saved["logical_negative"][2].all()
    with pytest.raises(ValueError, match="known complete"):
        run.oracle_answer(2, "would_repeat_completed_empty_answer")

    if kind != "directory_simulated_human":
        receipt = json.loads((run.output / "EXTERNAL_00001.json").read_text())
        is_llm = kind == "directory_llm_then_human"
        assert receipt["llm_consistency_checks"] == receipt["llm_accepted"] == int(is_llm)
        assert receipt["status"] == ("accepted_llm" if is_llm else "accepted_human")
        checks = [event for event in run.events if event.get("phase") == "LLM_proposal_validation"]
        assert len(checks) == int(is_llm)


def test_initial_complete_empty_gamma_is_resolved_without_a_request(tmp_path):
    context, rules, manifest = one_pending_document()
    complete_empty_gamma(context)
    run = LabelingRun(
        context=context, rules=rules, manifest=manifest, output=tmp_path / "run",
        trainer=forbid_training,
    )
    report = run.run()
    assert report["status"] == "all_documents_resolved" and report["task_complete"]
    assert report["actual_label_requests"] == 0 and report["actual_Chase_calls"] == 1
    assert report["acquired_documents"] == [] and 2 in run.initial_answers


def test_complete_empty_gamma_does_not_hide_an_active_unknown_conflict(tmp_path):
    context, rules, manifest = one_pending_document()
    complete_empty_gamma(context)
    manifest["loop"].update(
        conflict_policy="unknown_arbitrate", unknown_semantics="retain_consequences"
    )
    manifest["arbitration"] = dict(alias="aux", default_threshold=0.5, thresholds={})
    rules += (Rule((Doc("x"), Text("x", MatchPredicate("cnt", "blue"))), Label("x", "A")),)
    run = LabelingRun(
        context=context, rules=rules, manifest=manifest, output=tmp_path / "run",
        trainer=forbid_training,
    )
    report = run.run()
    assert report["status"] == "conflict_unknown_saturated" and not report["task_complete"]
    assert report["actual_label_requests"] == 0
    assert not run.gamma.pos[2].any() and run.gamma.neg[2].all()
    with np.load(run.output / "FINAL_P.npz") as saved:
        assert saved["conflict_unknown"][2, 0]


def test_rules_deriving_every_negative_do_not_certify_a_complete_external_answer(tmp_path):
    context, _, manifest = one_pending_document()
    context["gamma"] = RefGamma.empty(4, 2)
    rules = tuple(Rule((Doc("x"),), Label("x", label, "neg")) for label in context["labels"])
    run = LabelingRun(
        context=context, rules=rules, manifest=manifest, output=tmp_path / "run",
        trainer=forbid_training,
    )
    report = run.run()
    assert report["status"] == "label_budget_exhausted" and not report["task_complete"]
    assert not run.answered and not run.gamma.pos.any() and not run.gamma.neg.any()
    with np.load(run.output / "FINAL_P.npz") as saved:
        assert saved["logical_negative"].all() and not saved["logical_positive"].any()


def test_complete_trusted_negative_labels_remain_observed_feedback(tmp_path):
    context, rules, manifest = one_pending_document()
    complete_empty_gamma(context)
    manifest["strategy"] = "paper"
    run = LabelingRun(
        context=context, rules=rules, manifest=manifest, output=tmp_path / "run",
        trainer=forbid_training,
    )
    try:
        result, session, _ = run.chase(run.version, phase="negative_feedback_fixture")
        batch = run.feedback(result, run.version, session=session)
        assert batch.true_mask[2].all() and batch.observed[2].all()
        assert not batch.targets[2].any() and not batch.weak_mask[2].any()
        np.testing.assert_array_equal(batch.weights[2], np.ones(2))
        assert not run.gamma.pos[2].any() and run.gamma.neg[2].all()
        assert run.training_attempts == 0
    finally:
        for runner in run.sessions:
            runner.close()


def test_rdg_does_not_count_a_trusted_empty_answer_as_uncovered(tmp_path, monkeypatch):
    from loris._learning.rdg import RuleDependencyGraph

    context, rules, manifest = one_pending_document()
    complete_empty_gamma(context)
    positive = context["gamma"].pos.copy()
    positive[3] = False  # A different document still needs an answer.
    context["gamma"] = RefGamma(positive, context["gamma"].neg.copy())
    manifest["domains"]["query_roles"] = ["train", "dev"]
    manifest["query"]["strategy"] = "rdg"
    seen = []
    original = RuleDependencyGraph.estimate_batch

    def observe(estimator, session, result, queries, uncovered):
        queries = list(queries)
        seen.extend((row, label, list(uncovered)) for row, label in queries)
        return original(estimator, session, result, queries, uncovered)

    monkeypatch.setattr(RuleDependencyGraph, "estimate_batch", observe)
    run = LabelingRun(
        context=context, rules=rules, manifest=manifest, output=tmp_path / "run",
        trainer=forbid_training,
    )
    try:
        result, session, _ = run.chase(run.version, phase="negative_answer_rdg_fixture")
        assert run.rdg(result, session) == 3
        assert seen == [(3, 0, [3]), (3, 1, [3])]
        assert run.queries == 0  # RDG estimates do not themselves disclose truth.
    finally:
        for runner in run.sessions:
            runner.close()
