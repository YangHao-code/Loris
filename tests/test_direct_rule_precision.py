"""Revised author weighting: immediate rules, all supports, real Chase counts."""

import copy
import json

import numpy as np
import pytest

from loris.document import Document
from loris._engine.rdl_reference import Doc, Label, ML, Rule, Text, RefGamma
from loris._storage.discovery_factory import make_discovery_runner
from loris.labeling import cli
from loris.labeling.feedback import assignments, canonical_weighting
from loris.labeling.reliability import direct_rule_weights, lineage_scores, main_calibrate
from loris.predicates import MatchPredicate
from tests.test_fit_reliability import calibration_fixture
from tests.test_labeling_reliability_weights import artifact_for, evidence, runtime_manifest


def context(labels=("A", "B", "C"), documents=("red",)):
    return dict(
        docs=[Document(cnt=text, ttl="", mtd="") for text in documents],
        labels=list(labels),
        row_ids=["r%d" % i for i in range(len(documents))],
        row_roles=["train"] * len(documents),
        gamma=RefGamma.empty(len(documents), len(labels)),
        model_probabilities={"aux": np.full((len(documents), len(labels)), 0.2, np.float32)},
        model_decisions={},
    )


def test_main_direct_rule_weight_does_not_take_minimum_upstream(tmp_path):
    ctx = context()
    rules = (
        Rule((Doc("x"), Text("x", MatchPredicate("cnt", "red"))), Label("x", "A")),
        Rule((Doc("x"), Label("x", "A")), Label("x", "B")),
        Rule((Doc("x"), Label("x", "C")), Label("x", "C")),
    )
    artifact = artifact_for(rules, [(8, 1), (8, 8), (98, 98)])
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        closure = runner(rules, ctx["gamma"])
        weights, audit = direct_rule_weights(runner, rules, closure, ctx["gamma"], artifact)
        old, _ = lineage_scores(
            lambda program, seed: runner(program, seed), rules, ctx["gamma"], artifact
        )
    finally:
        runner.close()
    assert np.allclose(weights, [[0.2, 0.9, 0]])
    assert np.allclose(old, [[0.2, 0.2, 0]])
    assert audit["upstream_rule_scores_used"] is False
    assert audit["closure_calls"] == 0 and not audit["first_witness_only"]
    assert audit["unsupported_nonGamma_labels"] == 0


@pytest.mark.parametrize("reverse", [False, True])
def test_all_direct_supports_order_invariance_duplicate_instances_and_negative(tmp_path, reverse):
    ctx = context(labels=("A",), documents=("red", "blue"))
    base = (
        Rule((Doc("x"),), Label("x", "A", polarity="neg")),
        Rule((Doc("x"), Doc("y")), Label("x", "A", polarity="neg")),
    )
    counts = [(8, 1), (8, 8)]
    rules = base[::-1] if reverse else base
    artifact = artifact_for(rules, counts[::-1] if reverse else counts)
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        closure = runner(rules, ctx["gamma"])
        weights, audit = direct_rule_weights(runner, rules, closure, ctx["gamma"], artifact)
    finally:
        runner.close()
    assert np.allclose(weights, 0.9)
    assert closure.proof_dag["neg:0:0"]["rule_id"] == 0
    assert audit["multiply_direct_supported_labels"] == 2
    high = next(x for x in audit["rules"] if x["q"] == 0.9)
    assert high["valid_instances"] == 4 and high["distinct_supported_labels"] == 2


def test_only_current_valid_and_allowed_direct_instances_count(tmp_path):
    ctx = context(labels=("A",))
    rules = (
        Rule((Doc("x"),), Label("x", "A")),
        Rule((Doc("x"), ML("x", "aux", "A", 0.8)), Label("x", "A")),
        Rule((Doc("x"),), Label("x", "A")),
    )
    artifact = artifact_for(rules, [(8, 1), (98, 98), (8, 8)])
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        closure = runner(rules, ctx["gamma"])
        weights, audit = direct_rule_weights(
            runner, rules, closure, ctx["gamma"], artifact, excluded_rule_ids=(2,)
        )
        assert np.allclose(weights, 0.2)
        assert audit["rules"][1]["valid_instances"] == 0
        original = runner.application_allowed
        runner.application_allowed = lambda ri, rule, valuation: ri != 2 and original(
            ri, rule, valuation
        )
        scoped, _ = direct_rule_weights(runner, rules, closure, ctx["gamma"], artifact)
        assert np.allclose(scoped, 0.2)
    finally:
        runner.close()


def test_trusted_relation_transfers_Gamma_label_support_without_dropping_correction(tmp_path):
    ctx = context(labels=("A",), documents=("seed", "target"))
    gp = np.array([[True], [False]])
    ctx["gamma"] = RefGamma(gp, np.zeros_like(gp), ((0, 1),), ())
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    artifact = artifact_for()
    try:
        closure = runner((), ctx["gamma"])
        weights, audit = direct_rule_weights(runner, (), closure, ctx["gamma"], artifact)
    finally:
        runner.close()
    assert np.array_equal(weights, [[1.0], [1.0]])
    assert audit["unsupported_nonGamma_labels"] == 0
    data = dict(
        gamma=ctx["gamma"],
        closure=closure,
        independent=None,
        probabilities=ctx["model_probabilities"]["aux"],
        channels={(0, 0): (0, 0.5)},
        rows=np.ones(2, bool),
        strategy="paper",
        weighting="direct_rule_precision",
        reliability=artifact,
        reliability_scores=weights,
        reliability_audit=audit,
    )
    batch = assignments(**data)
    assert batch.weak_mask[1, 0] and batch.weights[1, 0] == 1
    # Training-row permissions remain independent of support attribution.
    data["rows"][1] = False
    batch = assignments(**data)
    assert batch.true_mask[0, 0] and batch.weights[0, 0] == 1
    assert not batch.observed[1, 0]


def test_direct_main_ignores_model_disagreement_discount_and_aliases_are_explicit():
    data = evidence()
    batch = assignments(**data, weighting="direct_rule_precision", disagreement_discount=1.0)
    assert batch.weights[1, 0] == pytest.approx(0.8)
    assert not batch.audit["confidence_applied"]
    assert batch.audit["weight_formula"] == "max_direct_support_rules(q_rule)"
    for alias, target in (
        ("reliability_only", "complete_provenance"),
        ("provenance", "complete_provenance_discount"),
    ):
        assert canonical_weighting(alias) == target
        old = assignments(**data, weighting=alias)
        named = assignments(**data, weighting=target)
        assert np.array_equal(old.weights, named.weights)
        assert old.audit["weighting"] == target and old.audit["weighting_alias"] == alias
    data["strategy"] = "self_train"
    with pytest.raises(ValueError, match="SelfTrain predictions"):
        assignments(**data, weighting="direct_rule_precision")


def test_actual_calibration_q_and_public_cli_canonical_alias_receipts(tmp_path):
    args, _, _ = calibration_fixture(tmp_path)
    assert main_calibrate(args) == 0
    artifact_path = tmp_path / "calibrated/RELIABILITY.json"
    effective, ctx, rules, artifact, receipt = cli.load(
        args.manifest, weighting="direct_rule_precision", source_reliability=artifact_path
    )
    assert effective["weighting"] == receipt["effective_weighting"] == "direct_rule_precision"
    assert receipt["weighting_alias"] is None
    assert [(r["support"], r["correct"], r["q"]) for r in artifact["rule_scores"]] == [
        (0, 0, 0.5),
        (18, 17, 0.9),
    ]
    runner = make_discovery_runner(ctx, tmp_path / "weight_cache")
    try:
        closure = runner(rules, ctx["gamma"])
        weights, _ = direct_rule_weights(runner, rules, closure, ctx["gamma"], artifact)
    finally:
        runner.close()
    assert weights[1, 0] == pytest.approx(0.9)
    assert weights[0, 0] == 1  # Verified root stays trusted.
    old, _, _, _, old_receipt = cli.load(
        args.manifest, weighting="provenance", source_reliability=artifact_path
    )
    assert old["weighting"] == "complete_provenance_discount"
    assert old_receipt["requested_weighting"] == old_receipt["weighting_alias"] == "provenance"


def test_cli_conflict_and_failed_check_policies_require_explicit_extensions(tmp_path):
    manifest, data = runtime_manifest(tmp_path)
    loaded, _, _, _, receipt = cli.load(manifest)
    assert loaded["loop"]["conflict_policy"] == receipt["conflict_policy"] == "halt"
    assert loaded["feedback"]["check_failure_policy"] == "halt"
    data["loop"]["source_review_budget"] = 1
    with pytest.raises(ValueError, match="explicit conflict_policy"):
        cli.validate(data)
    data["loop"]["conflict_policy"] = "review_instances"
    data["feedback"]["check_failure_policy"] = "reject_batch"
    cli.validate(data)
    manifest.write_text(json.dumps(data))
    loaded, _, _, _, _ = cli.load(
        manifest, source_review_budget=0, conflict_policy="halt", check_failure_policy="halt"
    )
    assert loaded["loop"]["conflict_policy"] == "halt"
    assert loaded["feedback"]["check_failure_policy"] == "halt"


def test_main_weight_runs_current_checkpoint_calibration_transaction(tmp_path):
    from loris.labeling.runner import LabelingRun
    from tests.test_current_model_reliability import initial, scripted_trainer

    manifest, ctx, rules, artifact, _ = initial(tmp_path)
    manifest["weighting"] = "direct_rule_precision"
    trainer, seen = scripted_trainer(ctx)
    run = LabelingRun(
        context=ctx,
        rules=rules,
        manifest=manifest,
        output=tmp_path / "run",
        trainer=trainer,
        reliability=artifact,
    )
    report = run.run()
    assert report["committed_model_updates"] == 1
    assert report["inference_counts"]["calibration"] == 1
    assert run.reliability["rule_scores"][0]["q"] == pytest.approx(18 / 22)
    assert all("prediction_sets" in call for call in seen)
    assert seen[0]["batch"].weights[1, 0] == pytest.approx(0.9)
    audit = seen[0]["batch"].audit["reliability_derivation_audit"]
    assert audit["algorithm"] == "all_current_direct_Label_and_relation_transfer_instances"
    assert audit["closure_calls"] == 0 and not audit["upstream_rule_scores_used"]
    assert run.reliability["binding"]["checkpoint_sha256"] == report["checkpoint"]["sha256"]


def test_main_weight_missing_independent_predictions_rolls_back_model_and_q(tmp_path):
    from loris.labeling.runner import LabelingRun
    from tests.test_current_model_reliability import initial, scripted_trainer

    manifest, ctx, rules, artifact, _ = initial(tmp_path)
    manifest["weighting"] = "direct_rule_precision"
    trainer, _ = scripted_trainer(ctx, missing_predictions=True)
    run = LabelingRun(
        context=ctx,
        rules=rules,
        manifest=manifest,
        output=tmp_path / "run",
        trainer=trainer,
        reliability=artifact,
    )
    report = run.run()
    assert report["committed_model_updates"] == 0
    assert report["checkpoint"] == manifest["dynamic"]["checkpoint"]
    assert report["final_reliability_artifact"] == manifest["source_reliability"]
    assert run.reliability["rule_scores"] == artifact["rule_scores"]
    assert report["status"] == "update_failed_retaining_zero_update_state"


def test_receiver_exclusion_direct_weight_uses_independent_premises(tmp_path):
    from loris.labeling.runner import LabelingRun
    from tests.test_labeling import fixture, synthetic_same_trainer

    ctx, _, manifest = fixture()
    ctx["gamma"] = RefGamma.empty(4, 2)
    rules = (
        Rule((Doc("x"), ML("x", "aux", "A", 0.1)), Label("x", "A")),
        Rule((Doc("x"), Label("x", "A")), Label("x", "B")),
        Rule((Doc("x"), Text("x", MatchPredicate("cnt", "red"))), Label("x", "B")),
        Rule((Doc("x"), ML("x", "aux", "B", 0.8)), Label("x", "B")),
    )
    artifact = artifact_for(rules, [(8, 8), (8, 8), (8, 1), (8, 8)])
    manifest["weighting"] = "direct_rule_precision"
    manifest["feedback"]["source_filter"] = "exclude_receiver"
    manifest["feedback"][
        "rule_reliability_update"
    ] = "frozen_prior"  # Only enumerate support; no fit.
    run = LabelingRun(
        context=ctx,
        rules=rules,
        manifest=manifest,
        output=tmp_path / "run",
        trainer=synthetic_same_trainer,
        reliability=artifact,
    )
    try:
        closure, session, _ = run.chase(run.version, phase="weight_source_filter_test")
        assert closure.positive_facts[1].all()
        batch = run.feedback(closure, run.version, session=session)
        assert batch.weak_mask[1, 1]
        # B is independently supported by red->B at q=.2. The high-q A->B
        # instance is invalid after removing the only producer of A (Aux).
        assert batch.weights[1, 1] == pytest.approx(0.2)
    finally:
        for session in run.sessions:
            session.close()
