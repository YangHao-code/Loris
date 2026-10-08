"""Cumulative real Gamma feedback, independent A/B gate, and atomic rejection."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from loris._engine.rdl_reference import RefGamma
from loris.labeling.feedback import Batch, cumulative_verified_rows, incremental_batch, limit_training_rows
from loris.labeling.fallback import auxiliary_quality, joint_gate_decision, output_quality, record_binding
from loris.labeling.reliability import _fallback_calibration
from loris.labeling.runner import LabelingRun
from tests.test_batch_query_and_exports import configured
from tests.test_labeling import synthetic_same_trainer


def test_all_acquired_answers_remain_required_after_pending_was_cleared():
    positive = np.tile([True, False], (104, 1))
    negative = ~positive
    gamma = RefGamma(positive, negative)
    acquired, required = cumulative_verified_rows(gamma, np.ones(104, bool), [100, 101], [101])
    assert acquired.tolist() == required.tolist() == [100, 101]
    batch = Batch(positive.astype(np.float32), np.ones_like(positive), np.ones_like(positive, np.float32),
                  np.ones_like(positive), np.zeros_like(positive), {})
    batch = incremental_batch(batch, gamma, np.ones(104, bool), required, .01, None)
    batch = limit_training_rows(batch, 2, required, None, seed=42)
    assert np.flatnonzero(batch.observed.any(1)).tolist() == [100, 101]
    assert batch.true_mask[100:102].all() and not batch.weak_mask.any()
    np.testing.assert_array_equal(batch.targets[100:102], positive[100:102])
    np.testing.assert_array_equal(gamma.pos, positive)


def test_cumulative_pool_excludes_partial_and_unauthorized_rows_but_keeps_new_partial():
    gp = np.array([[1, 0], [1, 0], [1, 0], [1, 0]], bool)
    gn = np.array([[0, 1], [0, 0], [0, 1], [0, 1]], bool)
    acquired, required = cumulative_verified_rows(RefGamma(gp, gn), np.array([1, 1, 0, 1], bool),
                                                  [0, 1, 2], [1])
    assert acquired.tolist() == [0] and required.tolist() == [0, 1]


def test_cumulative_answers_cannot_be_silently_dropped_by_row_budget():
    shape = (3, 2)
    batch = Batch(np.zeros(shape), np.ones(shape, bool), np.ones(shape),
                  np.ones(shape, bool), np.zeros(shape, bool), {})
    with pytest.raises(ValueError, match="cannot discard"):
        limit_training_rows(batch, 1, [0, 1], None, seed=0)


def test_two_actual_query_batches_retrain_both_answers_and_clear_only_pending(tmp_path):
    ctx, rules, manifest = configured(tmp_path)
    manifest["loop"].update(query_batch_size=1, query_cadence="each_closure", max_inner_rounds=1)
    manifest["feedback"].update(rule_feedback_scope="new_or_changed", trusted_replay=True,
                                trusted_replay_fraction=.05, max_training_rows=2,
                                verified_replay_policy="cumulative_acquired")
    seen = []
    def train(**kwargs):
        seen.append(copy.deepcopy(kwargs["batch"]))
        return synthetic_same_trainer(**kwargs)
    run = LabelingRun(context=ctx, rules=rules, manifest=manifest, output=tmp_path / "run", trainer=train)
    result = run.run()
    assert result["actual_label_requests"] == 2 and len(seen) == 2
    assert seen[0].audit["cumulative_verified_training_rows"] == [2]
    assert seen[1].audit["cumulative_verified_training_rows"] == [2, 3]
    assert seen[1].audit["fresh_verified_batch_rows"] == [3]
    assert seen[1].true_mask[2:4].all() and not seen[1].weak_mask[2:4].any()
    assert not run.pending_verified_rows and run.answered - run.initial_answers == {2, 3}


def test_auxiliary_thresholds_are_fit_on_A_never_on_B():
    y = np.array([[1], [0]], bool)
    score = auxiliary_quality(np.array([[.7], [.3]]), y, ~y,
                             fit_probabilities=np.array([[.9], [.1]]), fit_positive=y,
                             fit_negative=~y, grid=[.2, .8])
    assert score["thresholds"] == [.8]
    assert score["FN"] == 1 and score["micro_f1"] == 0
    assert score["macro_average_precision"] == 1


def test_aux_AP_ties_match_independent_sklearn_reference():
    from sklearn.metrics import average_precision_score
    rng = np.random.RandomState(403)
    p = rng.choice([.1, .3, .7, .9], (43, 4))
    y = rng.random((43, 4)) > .7
    mask = rng.random((43, 4)) > .2
    result = auxiliary_quality(p, y & mask, ~y & mask, fit_probabilities=p[:13],
                               fit_positive=y[:13], fit_negative=~y[:13], grid=[.2, .5, .8])
    reference = np.mean([average_precision_score(y[mask[:, c], c], p[mask[:, c], c]) for c in range(4)])
    assert result["macro_average_precision"] == pytest.approx(reference)


def test_partial_observations_do_not_make_unknown_negative_or_full_Comp_document():
    truth = np.array([[1, 1], [1, 1]], bool)
    observed = np.array([[1, 0], [1, 1]], bool)
    prediction = np.array([[1, 0], [1, 0]], bool)
    score = output_quality(truth, observed, prediction, np.ones(2, bool))
    assert score["FN"] == 1 and score["completeness_documents"] == 1
    assert score["multi_label_completeness"] == .5


def parent_quality():
    value = dict(macro_f1=.6, micro_f1=.7, macro_average_precision=.8, FN=20,
                 multi_label_completeness=.85, completeness_documents=10, rows=12, positive_cells=60)
    return dict(auxiliary=copy.deepcopy(value), loris=copy.deepcopy(value))


@pytest.mark.parametrize("family, field, value, reason", [
    ("auxiliary", "micro_f1", .69, "micro_f1_decreased"),
    ("auxiliary", "macro_average_precision", .79, "average_precision_decreased"),
    ("loris", "macro_f1", .59, "macro_f1_decreased"),
    ("loris", "FN", 21, "false_negatives_increased"),
    ("loris", "multi_label_completeness", .84, "completeness_decreased"),
])
def test_joint_gate_does_not_compensate_bad_component_with_mean_gain(family, field, value, reason):
    parent = parent_quality()
    candidate = copy.deepcopy(parent)
    candidate["loris"]["micro_f1"] += .1
    candidate[family][field] = value
    decision = joint_gate_decision(parent, candidate)
    assert not decision["accepted"] and any(reason in item for item in decision["failures"])


def test_joint_gate_ties_retain_parent_but_real_nonregressing_gain_passes():
    parent = parent_quality()
    assert not joint_gate_decision(parent, copy.deepcopy(parent))["accepted"]
    candidate = copy.deepcopy(parent)
    candidate["auxiliary"]["micro_f1"] += .01
    assert joint_gate_decision(parent, candidate)["accepted"]


def joint_record_inputs():
    labels = ["A", "B"]
    p = np.array([[.9, .9], [.1, .1], [.7, .7], [.3, .3]])
    truth = np.array([[1, 1], [0, 0], [1, 1], [0, 0]], bool)
    selected = np.ones(4, bool)
    context = dict(row_ids=["a1", "a2", "b1", "b2"], labels=labels,
                   model_probabilities={"old": p, "new": p})
    manifest = dict(dynamic=dict(alias="new"), aux_extension=dict(source_alias="old"),
                    loop=dict(commit_gate="independent_calibration", commit_gate_metric_policy="aux_loris_joint",
                              commit_gate_aux_grid=[.2, .8],
                              joint_gate_partition=dict(fit_row_ids=["a1", "a2"], score_row_ids=["b1", "b2"])),
                    fallback=dict(enabled=True, alias="old", calibration="fixed", default_threshold=.5, thresholds={}))
    closure = SimpleNamespace(positive_facts=np.zeros_like(truth), negative_facts=np.zeros_like(truth),
                              conflict_unknown=np.zeros_like(truth))
    return manifest, context, closure, truth, selected


def test_bound_gate_record_uses_A_for_Aux_thresholds_and_same_B_for_both_outputs():
    manifest, ctx, closure, truth, selected = joint_record_inputs()
    record = _fallback_calibration(manifest, ctx, ctx, closure, truth, ~truth, selected)
    assert record["joint_gate"]["auxiliary"]["thresholds"] == [.8, .8]
    assert record["joint_gate"]["auxiliary"]["FN"] == 2
    assert record["gate"]["TP"] == 2 and record["gate"]["FN"] == 0
    assert record["gate"]["rows"] == record["joint_gate"]["auxiliary"]["rows"] == 2
    before = record_binding(manifest, ctx["labels"], {}, {})
    manifest["loop"]["joint_gate_partition"]["fit_row_ids"] = ["a2"]
    assert record_binding(manifest, ctx["labels"], {}, {}) != before


def test_mutable_fallback_fits_only_A_before_scoring_B():
    manifest, ctx, closure, truth, selected = joint_record_inputs()
    manifest["fallback"].update(calibration="independent_calibration", grid=[.2, .8], minimum_positive_support=1)
    record = _fallback_calibration(manifest, ctx, ctx, closure, truth, ~truth, selected)
    assert record["thresholds"] == {"A": .8, "B": .8}
    assert record["gate"]["TP"] == 0 and record["gate"]["FN"] == 2
    assert record["audit"]["thresholds_source"] == "fit_partition"


@pytest.mark.parametrize("invalid", ["overlap", "outside", "unselected"])
def test_joint_gate_partition_rejects_overlap_unknown_or_undeclared_rows(invalid):
    manifest, ctx, closure, truth, selected = joint_record_inputs()
    if invalid == "overlap":
        manifest["loop"]["joint_gate_partition"]["fit_row_ids"] = ["b1"]
    elif invalid == "outside":
        manifest["loop"]["joint_gate_partition"]["fit_row_ids"] = ["doesnotexist"]
    else:
        selected[0] = False
    with pytest.raises(ValueError, match="overlap|declared calibration"):
        _fallback_calibration(manifest, ctx, ctx, closure, truth, ~truth, selected)


def test_actual_runner_rejects_bad_aux_and_preserves_model_thresholds_and_answers(tmp_path):
    from loris.discovery.manifest import file_record
    from loris._engine.typed_session import digest
    from loris.labeling import cli
    from loris.labeling.aux_extension import prepare
    from loris.labeling.reliability import main_calibrate
    from tests.test_fit_reliability import calibration_fixture

    args, calibration, _ = calibration_fixture(tmp_path)
    prepared = prepare(args.manifest, tmp_path / "isolated", protocol_path=args.protocol,
                       label_budget=1, query_batch_size=1)
    manifest_path = Path(prepared["labeling_manifest"]["path"])
    data = json.loads(manifest_path.read_text())
    answers = tmp_path / "answers"
    answers.mkdir()
    for row, target in ((1, [1, 0]), (2, [0, 1])):
        (answers / ("%08d.json" % row)).write_text(json.dumps(dict(
            row_id="r%d" % row, labels=["A", "B"], targets=target)))
    data["oracle"] = dict(kind="directory_simulated_human", directory=str(answers),
                          allow_known_source_review=False)
    data["loop"].update(commit_gate="independent_calibration", commit_gate_metric_policy="aux_loris_joint",
                        commit_gate_tolerance=0,
                        conflict_policy="unknown_arbitrate", unknown_semantics="retain_consequences",
                        gamma_conflict_policy="validated_gamma_wins", conflict_update_policy="fresh_from_gamma",
                        joint_gate_partition=dict(fit_row_ids=calibration["row_ids"][:10],
                                                  score_row_ids=calibration["row_ids"][10:]))
    data["arbitration"] = dict(alias="aux", default_threshold=.5, thresholds={})
    data["feedback"]["verified_replay_policy"] = "cumulative_acquired"
    manifest_path.write_text(json.dumps(data))
    cal = tmp_path / "strict_cal"
    assert main_calibrate(SimpleNamespace(manifest=str(manifest_path),
        protocol=prepared["reliability_protocol"]["path"], output=str(cal))) == 0
    manifest, ctx, rules, artifact, _ = cli.load(manifest_path, source_reliability=cal / "RELIABILITY.json")
    def bad_aux(**kwargs):
        folder = Path(kwargs["output"])
        folder.mkdir()
        checkpoint = folder / "new.pt"
        checkpoint.write_bytes(b"fixture bad Aux, not a trained scientific result")
        p = np.full_like(ctx["model_probabilities"][manifest["dynamic"]["alias"]], .99)
        extra = np.full((20, 2), .1, np.float32)
        extra[10:17, 0] = .01
        extra[17:, 0] = .99
        extra[10:15, 1] = .01
        extra[15:, 1] = .99
        np.savez(folder / "calibration.npz", probabilities=extra)
        fit = dict(optimizer_updates=1, checkpoint=file_record(checkpoint), prediction_sets={
            "independent_calibration": dict(file=file_record(folder / "calibration.npz"), rows=20,
                labels=list(ctx["labels"]), texts_id=digest(kwargs["prediction_sets"]["independent_calibration"]),
                checkpoint_sha256=file_record(checkpoint)["sha256"], supervision_used=False)})
        return p, fit
    run = LabelingRun(context=ctx, rules=rules, manifest=manifest, reliability=artifact,
                      output=tmp_path / "run", trainer=bad_aux)
    before = copy.deepcopy(run.version)
    before_q = copy.deepcopy(run.reliability)
    result = run.run()
    assert result["training_attempts"] >= 1 and result["committed_model_updates"] == 0
    assert result["actual_label_requests"] == 1 and len(run.answered - run.initial_answers) == 1
    assert (run.gamma.pos | run.gamma.neg)[list(run.answered - run.initial_answers)].all()
    assert run.version.checkpoint == before.checkpoint and run.version.channels == before.channels
    np.testing.assert_array_equal(run.version.probabilities, before.probabilities)
    assert run.reliability == before_q
    decisions = list((tmp_path / "run").glob("update_*/JOINT_GATE_DECISION.json"))
    assert decisions and all(not json.loads(p.read_text())["decision"]["accepted"] for p in decisions)


def test_dormant_positive_channel_is_initially_equal_then_new_Aux_can_add_label(tmp_path):
    from loris._engine.rdl_reference import Doc, Label, ML, Rule
    from loris._storage.discovery_factory import make_discovery_runner
    from loris.labeling.aux_extension import dormant_positive_channels, attach_context, extend_program
    from loris.labeling.unknown_incremental import retaining_chase
    from tests.test_labeling import fixture
    ctx, _, manifest = fixture()
    p = np.array([[.99, .1], [.99, .1], [.1, .1], [.1, .1]], np.float32)
    ctx["model_probabilities"]["aux"] = p
    old = (Rule((Doc("x"), ML("x", "aux", "A", .8)), Label("x", "A")),)
    old_runner = make_discovery_runner(ctx, tmp_path / "old")
    try:
        original = retaining_chase(old_runner, old, ctx["gamma"], gamma_wins=True)
    finally:
        old_runner.close()
    y = np.array([[1, 0], [1, 0]], bool)
    admission, receipt = dormant_positive_channels(ctx["labels"],
        [(p, original.positive_facts), (p, original.positive_facts)], p[:2], y, ~y,
        grid=[.9, .95], minimum_support=2)
    assert admission["positive_active"] == ["A"] and admission["thresholds"] == {"A": .9}
    manifest["dynamic"]["alias"] = "adaptive"
    manifest["aux_extension"] = dict(source_alias="aux", alias="adaptive", default_threshold=.95,
        thresholds=admission["thresholds"], correction=dict(complement_alias="adaptive_low",
            positive_active=admission["positive_active"], negative_rule_indices=[],
            negative_default_threshold=.1, negative_thresholds={}, min_gap=.05))
    extended = attach_context(ctx, manifest)
    rules = extend_program(old, manifest, ctx["labels"])
    runner = make_discovery_runner(extended, tmp_path / "new")
    try:
        same = retaining_chase(runner, rules, extended["gamma"], gamma_wins=True)
        np.testing.assert_array_equal(same.positive_facts, original.positive_facts)
        np.testing.assert_array_equal(same.negative_facts, original.negative_facts)
        np.testing.assert_array_equal(same.conflict_unknown, original.conflict_unknown)
    finally:
        runner.close()
    changed = dict(extended, model_probabilities=dict(extended["model_probabilities"]))
    new_p = p.copy()
    new_p[2, 0] = .99
    changed["model_probabilities"]["adaptive"] = new_p
    runner = make_discovery_runner(changed, tmp_path / "changed")
    try:
        improved = retaining_chase(runner, rules, changed["gamma"], gamma_wins=True)
        assert improved.positive_facts[2, 0] and not original.positive_facts[2, 0]
        assert changed["model_probabilities"]["aux"] is p
        assert rules[:1] == old and receipt["application_truth_used"] is False
    finally:
        runner.close()


@pytest.mark.parametrize("update", [None, "retain_history"])
def test_joint_gate_rejects_retained_history_before_calibration_or_runner_artifacts(tmp_path, update):
    from loris.labeling.cli import validate
    from tests.test_aux_correction_v3 import corrected
    context, rules, manifest = corrected()
    manifest["loop"].update(commit_gate="independent_calibration", commit_gate_metric_policy="aux_loris_joint",
        conflict_policy="unknown_arbitrate", unknown_semantics="retain_consequences",
        joint_gate_partition=dict(fit_row_ids=["r2"],score_row_ids=["r3"]))
    manifest["arbitration"] = dict(alias="aux",default_threshold=.5,thresholds={})
    if update is not None:
        manifest["loop"]["conflict_update_policy"] = update
    with pytest.raises(ValueError, match="joint gate requires conflict_update_policy=fresh_from_gamma"):
        validate(manifest)
    with pytest.raises(ValueError, match="joint gate requires conflict_update_policy=fresh_from_gamma"):
        LabelingRun(context=context,rules=rules,manifest=manifest,output=tmp_path/"rejected",trainer=synthetic_same_trainer)
    assert not (tmp_path/"rejected").exists()
    manifest["loop"]["conflict_update_policy"] = "fresh_from_gamma"
    assert validate(manifest)["loop"]["conflict_update_policy"] == "fresh_from_gamma"
