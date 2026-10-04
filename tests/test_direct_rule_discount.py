"""Main confidence factor uses immediate q, not a complete-chain minimum."""

import copy

import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, Rule, Text
from loris._storage.discovery_factory import make_discovery_runner
from loris.labeling import cli
from loris.labeling.feedback import assignments
from loris.labeling.reliability import direct_rule_weights, lineage_scores, main_calibrate
from loris.labeling.runner import LabelingRun
from loris.predicates import MatchPredicate
from tests.test_direct_rule_precision import context
from tests.test_labeling_reliability_weights import artifact_for, evidence


def test_main_both_signs_confidence_and_q_without_changing_candidates():
    data = evidence()
    pure = assignments(**data, weighting="direct_rule_precision")
    combined = assignments(**data, weighting="direct_rule_precision_discount")
    assert combined.weights[1, 0] == pytest.approx(0.48)  # A=1,p=.2,q=.8
    assert combined.weights[2, 0] == pytest.approx(0.48)  # A=0,p=.8,q=.8
    assert combined.weights[3, 1] == pytest.approx(0.275)
    np.testing.assert_array_equal(combined.weak_mask, pure.weak_mask)
    np.testing.assert_array_equal(combined.observed, pure.observed)
    np.testing.assert_array_equal(combined.targets, pure.targets)
    assert np.array_equal(combined.weights[0], [1.0, 1.0])
    assert not combined.weights[4].any()  # unknown remains unobserved
    assert combined.audit["confidence_applied"] and combined.audit["reliability_applied"]
    assert not pure.audit["confidence_applied"]


def test_discount_keeps_high_direct_q_despite_low_upstream_q(tmp_path):
    ctx = context(labels=("A", "B"))
    rules = (
        Rule((Doc("x"), Text("x", MatchPredicate("cnt", "red"))), Label("x", "A")),
        Rule((Doc("x"), Label("x", "A")), Label("x", "B")),
    )
    artifact = artifact_for(rules, [(8, 1), (8, 8)])
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        closure = runner(rules, ctx["gamma"])
        direct, audit = direct_rule_weights(runner, rules, closure, ctx["gamma"], artifact)
        complete, _ = lineage_scores(
            lambda rs, gamma: runner(rs, gamma), rules, ctx["gamma"], artifact
        )
    finally:
        runner.close()
    data = dict(
        gamma=ctx["gamma"],
        closure=closure,
        independent=None,
        probabilities=ctx["model_probabilities"]["aux"],
        channels={(0, 1): (1, 0.5)},
        rows=np.ones(1, bool),
        strategy="paper",
        reliability=artifact,
        reliability_scores=direct,
        reliability_audit=audit,
    )
    main = assignments(**data, weighting="direct_rule_precision_discount")
    pure = assignments(**data, weighting="direct_rule_precision")
    data["reliability_scores"] = complete
    old = assignments(**data, weighting="complete_provenance_discount")
    assert main.weights[0, 1] == pytest.approx(0.54)
    assert pure.weights[0, 1] == pytest.approx(0.9)
    assert old.weights[0, 1] == pytest.approx(0.12)
    assert not audit["upstream_rule_scores_used"]


def test_coefficient_zero_is_pure_q_and_probability_before_update_matters():
    data = evidence()
    pure = assignments(**data, weighting="direct_rule_precision")
    zero = assignments(**data, weighting="direct_rule_precision_discount", disagreement_discount=0)
    np.testing.assert_array_equal(pure.weights, zero.weights)
    before = assignments(**data, weighting="direct_rule_precision_discount")
    changed = copy.deepcopy(data)
    changed["probabilities"][1, 0] = 0.4  # still disagrees with A=1 at .5
    after = assignments(**changed, weighting="direct_rule_precision_discount")
    assert before.weights[1, 0] == pytest.approx(0.48)
    assert after.weights[1, 0] == pytest.approx(0.56)
    for coefficient in (-1.0, 2.0, float("nan"), True):
        with pytest.raises(ValueError, match="discount"):
            assignments(
                **data,
                weighting="direct_rule_precision_discount",
                disagreement_discount=coefficient,
            )


def test_public_cli_and_runtime_use_direct_discount_without_chain_chases(tmp_path):
    from tests.test_current_model_reliability import initial, scripted_trainer

    manifest, ctx, rules, artifact, _ = initial(tmp_path)
    manifest["weighting"] = "direct_rule_precision_discount"
    cli.validate(manifest)
    trainer, calls = scripted_trainer(ctx)
    run = LabelingRun(
        context=ctx,
        rules=rules,
        manifest=manifest,
        output=tmp_path / "run",
        trainer=trainer,
        reliability=artifact,
    )
    result = run.run()
    first = calls[0]["batch"]
    assert first.weights[1, 0] == pytest.approx(0.54)
    assert first.audit["weighting"] == "direct_rule_precision_discount"
    assert first.audit["confidence_applied"]
    assert first.audit["reliability_derivation_audit"]["closure_calls"] == 0
    assert not first.audit["reliability_derivation_audit"]["upstream_rule_scores_used"]
    assert result["committed_model_updates"] == 1
    assert result["inference_counts"]["calibration"] == 1
    assert not any(e.get("phase") == "derivation_reliability" for e in run.events)
    assert run.model_predicate_map()["discount_active"]


def test_new_mode_is_an_explicit_cli_choice(tmp_path):
    from tests.test_fit_reliability import calibration_fixture

    args, _, _ = calibration_fixture(tmp_path)
    assert main_calibrate(args) == 0
    manifest, _, _, _, receipt = cli.load(
        args.manifest,
        weighting="direct_rule_precision_discount",
        source_reliability=tmp_path / "calibrated/RELIABILITY.json",
    )
    assert (
        manifest["weighting"] == receipt["effective_weighting"] == "direct_rule_precision_discount"
    )
    assert receipt["weighting_alias"] is None
