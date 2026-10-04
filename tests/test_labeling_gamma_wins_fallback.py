"""Validated-Gamma-wins retention, undecided-cell fallback and kappa options."""
import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, ML, RefGamma, Rule, Text
from loris._engine.typed_runtime import IndexedChaseRunner
from loris.document import Document
from loris.labeling import fallback
from loris.labeling.feedback import Batch, calibrate
from loris.labeling.unknown_incremental import retaining_chase
from loris.predicates import MatchPredicate


def context(n=3, labels=("A", "B"), p=0.9):
    return dict(
        docs=[Document(cnt="node%d" % i, ttl="", mtd="") for i in range(n)],
        labels=list(labels),
        gamma=RefGamma.empty(n, len(labels)),
        model_probabilities={"aux": np.full((n, len(labels)), p)},
        model_decisions={},
    )


def everyone(head, *body):
    return Rule((Doc("x"), Text("x", MatchPredicate("cnt", "^node"))) + body, head)


def test_quarantine_versus_gamma_wins(tmp_path):
    ctx = context()
    neg = np.zeros((3, 2), bool)
    neg[1, 0] = True  # validated: row 1 is not A
    gamma = RefGamma(np.zeros((3, 2), bool), neg)
    rules = (everyone(Label("x", "A")), everyone(Label("x", "B"), Label("x", "A")))
    runner = IndexedChaseRunner(ctx, tmp_path / "c")
    try:
        old = retaining_chase(runner, rules, gamma)
        assert old.conflict_unknown[1, 0]  # the validated cell was quarantined
        new = retaining_chase(runner, rules, gamma, gamma_wins=True)
        assert new.status == "fixpoint" and not new.conflict_unknown.any()
        assert new.negative_facts[1, 0] and not new.positive_facts[1, 0]
        assert new.positive_facts[[0, 2], 0].all()
        # The blocked A at row 1 cannot trigger B there.
        assert not new.positive_facts[1, 1] and new.positive_facts[[0, 2], 1].all()
        assert new.gamma_blocked == 1 and new.gamma_wins
    finally:
        runner.close()


def test_new_answer_withdraws_contradicted_derivation_and_descendants(tmp_path):
    ctx = context()
    gamma = ctx["gamma"]
    rules = (everyone(Label("x", "A")), everyone(Label("x", "B"), Label("x", "A")))
    runner = IndexedChaseRunner(ctx, tmp_path / "c")
    try:
        before = retaining_chase(runner, rules, gamma, gamma_wins=True)
        assert before.positive_facts.all()
        neg = np.zeros((3, 2), bool)
        neg[2, 0] = True
        answered = RefGamma(np.zeros((3, 2), bool), neg)
        later = IndexedChaseRunner(dict(ctx, gamma=answered), tmp_path / "d")
        try:
            after = retaining_chase(later, rules, answered, before=before, old_gamma=gamma,
                                    update_kind="chaseInc", gamma_wins=True)
            fresh = retaining_chase(later, rules, answered, gamma_wins=True)
            for a, b in ((after, fresh),):
                np.testing.assert_array_equal(a.positive_facts, b.positive_facts)
                np.testing.assert_array_equal(a.negative_facts, b.negative_facts)
            assert not after.positive_facts[2, 0] and not after.positive_facts[2, 1]
            assert after.negative_facts[2, 0] and not after.conflict_unknown.any()
        finally:
            later.close()
        with pytest.raises(ValueError):
            retaining_chase(runner, rules, gamma, before=before, old_gamma=gamma,
                            update_kind="chaseInc", gamma_wins=False)
    finally:
        runner.close()


def test_fallback_fills_only_undecided_cells():
    positive = np.array([[1, 0, 0, 0]], bool)
    negative = np.array([[0, 1, 0, 0]], bool)
    unknown = np.array([[0, 0, 1, 0]], bool)
    p = np.array([[0.9, 0.9, 0.9, 0.9]])
    filled = fallback.fill(positive, negative, unknown, np.zeros((1, 4), bool), p,
                           np.array([0.5, 0.5, 0.5, 0.5]))
    assert filled.tolist() == [[False, False, False, True]]
    filled = fallback.fill(positive, negative, unknown, np.zeros((1, 4), bool), p,
                           np.array([0.5, 0.5, 0.5, 0.95]))
    assert not filled.any()


def test_fallback_calibration_is_f1_optimal_and_conservative():
    n = 40
    rng = np.random.default_rng(0)
    truth = rng.random((n, 1)) < 0.5
    p = np.where(truth, 0.7, 0.3) + rng.normal(0, 0.01, (n, 1))
    zero = np.zeros((n, 1), bool)
    manifest = dict(dynamic=dict(alias="aux"),
                    fallback=dict(enabled=True, alias="aux", calibration="independent_calibration",
                                  default_threshold=0.5, thresholds={}))
    config = fallback.configuration(manifest)
    thresholds, gate, audit = fallback.calibrate(
        positive=zero, negative=zero, unknown=zero, probabilities=p,
        arbitration_kappa=np.array([0.5]), truth_positive=truth, truth_negative=~truth,
        rows=np.ones(n, bool), config=config, labels=["A"])
    assert gate["micro_f1"] == 1.0
    # Every grid point in (0.31, 0.69] separates perfectly; the largest wins.
    assert thresholds["A"] == 0.65


def batch(targets, true_mask, weights=None):
    targets = np.asarray(targets, np.float32)
    true_mask = np.asarray(true_mask, bool)
    observed = np.ones_like(true_mask)
    weights = np.ones(targets.shape, np.float32) if weights is None else np.asarray(weights, np.float32)
    return Batch(targets, observed, weights, true_mask, observed & ~true_mask, {})


def test_kappa_trusted_only_and_max_step():
    p = np.array([[0.2], [0.4], [0.6], [0.8]])
    # Trusted rows say 0.2/0.4 negative, 0.6/0.8 positive; weak rows flip 0.4.
    b = batch([[0], [1], [1], [1]], [[1], [0], [1], [1]])
    channels = {(0, 0): (0, 0.7)}
    default, _ = calibrate(p, b, channels, grid=[0.3, 0.5, 0.7], minimum_support_each_sign=1)
    trusted, _ = calibrate(p, b, channels, grid=[0.3, 0.5, 0.7], minimum_support_each_sign=1,
                           cells="trusted_only")
    stepped, receipt = calibrate(p, b, channels, grid=[0.3, 0.5, 0.7], minimum_support_each_sign=1,
                                 cells="trusted_only", max_step=0.1)
    assert default[(0, 0)][1] == 0.3      # pulled down by the weak positive
    assert trusted[(0, 0)][1] == 0.5
    assert stepped[(0, 0)][1] == 0.7      # 0.5 is outside the allowed step
    assert receipt["cells"] == "trusted_only" and receipt["max_step"] == 0.1


def test_workflow_variants_keep_output_controls_well_defined(tmp_path):
    import copy
    from loris.experiments.workflow import _adapt_output_controls, _needs_independent_outputs

    base = dict(
        strategy="paper", weighting="direct_rule_precision_discount",
        feedback=dict(rule_reliability_update="current_model"), dynamic=dict(alias="aux"),
        loop=dict(conflict_policy="unknown_arbitrate", unknown_semantics="retain_consequences",
                  gamma_conflict_policy="validated_gamma_wins", commit_gate="independent_calibration"),
        fallback=dict(enabled=True, alias="aux", calibration="independent_calibration",
                      default_threshold=1.0, thresholds={}))
    main = copy.deepcopy(base)
    _adapt_output_controls(main)
    assert main == base and _needs_independent_outputs(main)
    halt = copy.deepcopy(base)
    halt["loop"]["conflict_policy"] = "halt"
    _adapt_output_controls(halt)
    assert halt["loop"]["gamma_conflict_policy"] == "quarantine"
    # 2026-10-02: comparison variants keep the same gate and per-model
    # fallback recalibration as the main protocol.
    for strategy, weighting in (("frozen", "uniform"), ("paper", "uniform"), ("self_train", "uniform"),
                                ("supervised", "uniform"), ("paper", "confidence_only")):
        control = copy.deepcopy(base)
        control.update(strategy=strategy, weighting=weighting)
        expected = copy.deepcopy(control)
        _adapt_output_controls(control)
        assert control == expected and _needs_independent_outputs(control)
    plain = copy.deepcopy(base)
    plain.pop("fallback")
    plain["loop"]["commit_gate"] = "none"
    assert not _needs_independent_outputs(plain)


def test_end_to_end_calibrated_fallback_commit_gate_and_gamma_wins(tmp_path):
    import json
    from pathlib import Path
    from loris.discovery.manifest import file_record
    from loris.labeling import cli
    from loris.labeling.reliability import main_calibrate
    from loris.labeling.runner import LabelingRun
    from tests.test_fit_reliability import calibration_fixture
    from tests.test_current_model_reliability import scripted_trainer

    args, _, _ = calibration_fixture(tmp_path)
    data = json.loads(Path(args.manifest).read_text())
    data["loop"].update(conflict_policy="unknown_arbitrate", unknown_semantics="retain_consequences",
                        gamma_conflict_policy="validated_gamma_wins", commit_gate="independent_calibration",
                        commit_gate_tolerance=0.0)
    data["arbitration"] = dict(alias=data["dynamic"]["alias"], default_threshold=0.5, thresholds={})
    data["fallback"] = dict(enabled=True, alias=data["dynamic"]["alias"],
                            calibration="independent_calibration", default_threshold=1.0, thresholds={})
    data["calibration"].update(cells="trusted_only", max_step=0.1, minimum_support_each_sign=1)
    Path(args.manifest).write_text(json.dumps(data))
    assert main_calibrate(args) == 0
    record = json.loads((tmp_path / "calibrated/FALLBACK_CALIBRATION.json").read_text())
    assert set(record["thresholds"]) == set(data["labels"] if "labels" in data else record["thresholds"])
    assert record["answers_entered_inference"] is False and "mean_f1" in record["gate"]
    manifest, ctx, rules, artifact, _ = cli.load(
        args.manifest, weighting="direct_rule_precision_discount",
        source_reliability=tmp_path / "calibrated/RELIABILITY.json")
    trainer, calls = scripted_trainer(ctx)
    run = LabelingRun(context=ctx, rules=rules, manifest=manifest, output=tmp_path / "run",
                      trainer=trainer, reliability=artifact)
    result = run.run()
    assert result["fallback"]["calibration"] == "independent_calibration"
    assert result["gamma_conflict_policy"] == "validated_gamma_wins"
    assert result["commit_gate"]["policy"] == "independent_calibration"
    final = np.load(tmp_path / "run/FINAL_P.npz")
    fallback_positive = final["fallback_positive"]
    assert not (fallback_positive & (final["logical_positive"] | final["logical_negative"]
                                     | final["conflict_unknown"])).any()
    assert np.array_equal(final["positive"], final["logical_positive"] | final["arbitration_positive"]
                          | fallback_positive)
