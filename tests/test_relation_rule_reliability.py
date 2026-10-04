"""Relation precision counts transferred target labels and reaches optimization."""

import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from loris._engine.rdl_reference import (
    Doc,
    Text,
    Label,
    LabelCompare,
    AttrEqual,
    ML,
    Rule,
    RefGamma,
)
from loris._storage.discovery_factory import make_discovery_runner
from loris.discovery.manifest import file_record
from loris.labeling.reliability import (
    RELATION_SCORING,
    count_rule_conclusions,
    direct_rule_weights,
    lineage_scores,
    main_calibrate,
    validate_artifact,
)
from loris.predicates import MatchPredicate
from tests.test_direct_rule_precision import context
from tests.test_labeling_reliability_weights import artifact_for


def relation(source="source", target="target", op="subset", extra=()):
    return Rule(
        (
            Doc("x"),
            Doc("y"),
            Text("x", MatchPredicate("cnt", source)),
            Text("y", MatchPredicate("cnt", target)),
            AttrEqual("x", "mtd.group", "y", "mtd.group"),
        )
        + tuple(extra),
        LabelCompare("x", op, "y"),
    )


def ctx_for(texts, labels=("A", "B")):
    ctx = context(labels=labels, documents=texts)
    ctx["attribute_values"] = {"mtd.group": ["same"] * len(texts)}
    ctx["attribute_presence"] = {"mtd.group": np.ones(len(texts), bool)}
    return ctx


def scored_artifact(rules, counts):
    value = artifact_for(rules, counts)
    value["relation_scoring"] = copy.deepcopy(RELATION_SCORING)
    for rule, record in zip(rules, value["rule_scores"]):
        record["head_kind"] = type(rule.head).__name__
        record["conclusion_unit"] = (
            "propagated_distinct_target_Label"
            if isinstance(rule.head, LabelCompare)
            else "distinct_signed_Label_head"
        )
    return value


@pytest.mark.parametrize("op", ["eq", "subset", "strict_subset"])
def test_relation_q_counts_partial_target_labels_not_whole_sets(tmp_path, op):
    ctx = ctx_for(("source", "target", "other source"))
    gp = np.zeros((3, 2), bool)
    gp[0, 0] = gp[2, 0] = True
    gn = np.zeros_like(gp)
    gn[0, 1] = True
    ctx["gamma"] = RefGamma(gp, gn)
    rules = (relation(op=op),)
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        closure = runner(rules, ctx["gamma"])
        assert closure.positive_facts[1, 0] and not closure.negative_facts[1, 1]
        # Only one target label is checkable. Neither complete endpoint vector
        # nor any source labels are supplied as calibration answers.
        truth_pos = np.zeros_like(gp)
        truth_neg = np.zeros_like(gp)
        truth_pos[1, 0] = True
        selected = np.array([False, True, False])
        counts = count_rule_conclusions(runner, closure, rules, truth_pos, truth_neg, selected)
        assert (counts[0]["support"], counts[0]["correct"], counts[0]["q"]) == (1, 1, 2 / 3)
        assert counts[0]["matching_instances"] == 2  # same target from two sources deduplicated
        truth_pos[1, 0] = False
        truth_neg[1, 0] = True
        wrong = count_rule_conclusions(runner, closure, rules, truth_pos, truth_neg, selected)
        assert (wrong[0]["support"], wrong[0]["correct"], wrong[0]["q"]) == (1, 0, 1 / 3)
        truth_neg[1, 0] = False
        unknown = count_rule_conclusions(runner, closure, rules, truth_pos, truth_neg, selected)
        assert unknown[0]["support"] == 0
    finally:
        runner.close()


def test_eq_both_directions_and_alternative_support_not_first_fire(tmp_path):
    ctx = ctx_for(("source", "target"))
    rules = (
        Rule((Doc("x"), Text("x", MatchPredicate("cnt", "source"))), Label("x", "A")),
        Rule((Doc("x"), Text("x", MatchPredicate("cnt", "target"))), Label("x", "B")),
        relation(op="eq"),
    )
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        closure = runner(rules, ctx["gamma"])
        assert closure.positive_facts.all()
        pos = np.ones((2, 2), bool)
        neg = np.zeros_like(pos)
        counts = count_rule_conclusions(runner, closure, rules, pos, neg, np.ones(2, bool))
        # Count every current transfer-supported target, even if another rule
        # generated it first. The result cannot depend on a first-fire order.
        assert counts[2]["support"] == counts[2]["correct"] == 4
        reversed_rules = rules[::-1]
        other = runner(reversed_rules, ctx["gamma"])
        reversed_counts = count_rule_conclusions(
            runner, other, reversed_rules, pos, neg, np.ones(2, bool)
        )
        assert reversed_counts[0]["support"] == 4
        artifact = scored_artifact(rules, [(8, 1), (8, 1), (8, 8)])
        weights, _ = direct_rule_weights(runner, rules, closure, ctx["gamma"], artifact)
        assert np.allclose(weights, 0.9)
    finally:
        runner.close()


@pytest.mark.parametrize("op", ["subset", "strict_subset"])
def test_directed_relation_does_not_transfer_against_its_declared_direction(tmp_path, op):
    ctx = ctx_for(("source", "target"))
    gp = np.zeros((2, 2), bool)
    gp[0, 0] = True
    ctx["gamma"] = RefGamma(gp, np.zeros_like(gp))
    forward = relation(op=op)
    backward = Rule(forward.body, LabelCompare("y", op, "x"))
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        closure = runner((backward,), ctx["gamma"])
        assert not closure.positive_facts[1].any()
        weights, audit = direct_rule_weights(
            runner, (backward,), closure, ctx["gamma"], scored_artifact((backward,), [(8, 8)])
        )
        assert not weights[1].any() and audit["unsupported_nonGamma_labels"] == 0
        counts = count_rule_conclusions(
            runner,
            closure,
            (backward,),
            np.ones((2, 2), bool),
            np.zeros((2, 2), bool),
            np.ones(2, bool),
        )
        assert counts[0]["support"] == 0
    finally:
        runner.close()


def test_relation_identity_edges_do_not_calibrate_fake_label_transfer(tmp_path):
    ctx = ctx_for(("source",))
    gp = np.array([[True, False]])
    ctx["gamma"] = RefGamma(gp, np.zeros_like(gp))
    rule = Rule((Doc("x"),), LabelCompare("x", "eq", "x"))
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        closure = runner((rule,), ctx["gamma"])
        counts = count_rule_conclusions(runner, closure, (rule,), gp, ~gp, np.ones(1, bool))
        assert counts[0]["matching_instances"] == 1 and counts[0]["support"] == 0
    finally:
        runner.close()


def test_last_learned_primitive_edge_not_upstream_min_or_hidden_transitive_edge(tmp_path):
    ctx = ctx_for(("source", "middle", "target"))
    gp = np.zeros((3, 2), bool)
    gp[0, 0] = True
    ctx["gamma"] = RefGamma(gp, np.zeros_like(gp))
    rules = (relation("source", "middle"), relation("middle", "target"))
    artifact = scored_artifact(rules, [(8, 1), (8, 8)])
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        closure = runner(rules, ctx["gamma"])
        assert closure.subset_facts.rows[0] & (1 << 2)
        weights, audit = direct_rule_weights(runner, rules, closure, ctx["gamma"], artifact)
        assert np.allclose(weights[:, 0], [1, 0.2, 0.9])
        old, _ = lineage_scores(
            lambda selected, g: runner(selected, g), rules, ctx["gamma"], artifact
        )
        assert np.allclose(old[:, 0], [1, 0.2, 0.2])
        truth_pos = np.ones_like(gp)
        truth_neg = np.zeros_like(gp)
        counts = count_rule_conclusions(
            runner, closure, rules, truth_pos, truth_neg, np.ones(3, bool)
        )
        assert [r["support"] for r in counts] == [1, 1]
        assert not audit["upstream_rule_scores_used"]
    finally:
        runner.close()


@pytest.mark.parametrize("reverse", [False, True])
def test_alternative_relation_rules_max_q_order_independent(tmp_path, reverse):
    ctx = ctx_for(("source", "target"))
    gp = np.zeros((2, 2), bool)
    gp[0, 0] = True
    ctx["gamma"] = RefGamma(gp, np.zeros_like(gp))
    rules = (relation(), relation(extra=(Doc("z"),)))
    counts = [(8, 1), (8, 8)]
    if reverse:
        rules = rules[::-1]
        counts = counts[::-1]
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        closure = runner(rules, ctx["gamma"])
        weights, audit = direct_rule_weights(
            runner, rules, closure, ctx["gamma"], scored_artifact(rules, counts)
        )
        assert weights[1, 0] == pytest.approx(0.9)
        assert audit["multiply_direct_supported_labels"] == 1
        assert next(r for r in audit["rules"] if r["q"] == 0.9)["distinct_supported_labels"] == 1
    finally:
        runner.close()


def test_relation_scope_and_current_model_body_control_weight_support(tmp_path):
    ctx = ctx_for(("source", "target"))
    gp = np.zeros((2, 2), bool)
    gp[0, 0] = True
    ctx["gamma"] = RefGamma(gp, np.zeros_like(gp))
    rules = (relation(), relation(extra=(ML("x", "aux", "A", 0.8),)), relation(extra=(Doc("z"),)))
    artifact = scored_artifact(rules, [(8, 1), (98, 98), (8, 8)])
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        closure = runner(rules, ctx["gamma"])
        original = runner.application_allowed
        runner.application_allowed = lambda ri, r, v: ri != 2 and original(ri, r, v)
        weights, audit = direct_rule_weights(runner, rules, closure, ctx["gamma"], artifact)
        assert weights[1, 0] == pytest.approx(0.2)
        assert audit["rules"][1]["valid_instances"] == audit["rules"][2]["valid_instances"] == 0
    finally:
        runner.close()


def test_Gamma_transport_preserves_last_relation_or_Label_source_and_calibration_targets(tmp_path):
    ctx = ctx_for(("source", "middle", "target", "end"))
    gp = np.zeros((4, 2), bool)
    gp[0, 0] = True
    ctx["gamma"] = RefGamma(gp, np.zeros_like(gp), ((1, 2), (2, 3)), ())
    rules = (relation("source", "middle"),)
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        closure = runner(rules, ctx["gamma"])
        weights, _ = direct_rule_weights(
            runner, rules, closure, ctx["gamma"], scored_artifact(rules, [(8, 8)])
        )
        assert np.allclose(weights[:, 0], [1, 0.9, 0.9, 0.9])
        pos = np.zeros_like(gp)
        pos[1:3, 0] = True
        neg = np.zeros_like(gp)
        neg[3, 0] = True
        counts = count_rule_conclusions(
            runner, closure, rules, pos, neg, np.array([False, True, True, True])
        )
        assert (counts[0]["support"], counts[0]["correct"], counts[0]["q"]) == (3, 2, 0.6)
        # No learned relation: a learned Label's q travels through Gamma edges.
        label_rules = (
            Rule((Doc("x"), Text("x", MatchPredicate("cnt", "middle"))), Label("x", "A")),
        )
        other = runner(label_rules, ctx["gamma"])
        labels, _ = direct_rule_weights(
            runner, label_rules, other, ctx["gamma"], artifact_for(label_rules, [(8, 1)])
        )
        assert np.allclose(labels[:, 0], [1, 0.2, 0.2, 0.2])
    finally:
        runner.close()


def test_relation_cycles_without_labels_do_not_create_support_or_denominator(tmp_path):
    ctx = ctx_for(("source", "target"))
    rules = (relation(), relation("target", "source"))
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        closure = runner(rules, ctx["gamma"])
        weights, audit = direct_rule_weights(
            runner, rules, closure, ctx["gamma"], scored_artifact(rules, [(8, 8), (8, 8)])
        )
        assert not weights.any() and audit["unsupported_nonGamma_labels"] == 0
        counts = count_rule_conclusions(
            runner, closure, rules, np.ones((2, 2), bool), np.zeros((2, 2), bool), np.ones(2, bool)
        )
        assert [r["support"] for r in counts] == [0, 0]
    finally:
        runner.close()


def test_legacy_relation_whole_set_artifact_rejected_but_Label_artifact_compatible(tmp_path):
    ctx = ctx_for(("source", "target"))
    rules = (relation(),)
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        closure = runner(rules, ctx["gamma"])
        old = artifact_for(rules, [(8, 8)])
        with pytest.raises(ValueError, match="recalibrated"):
            direct_rule_weights(runner, rules, closure, ctx["gamma"], old)
        with pytest.raises(ValueError, match="recalibrated"):
            lineage_scores(lambda selected, g: runner(selected, g), rules, ctx["gamma"], old)
        direct_rule_weights(runner, (), runner((), ctx["gamma"]), ctx["gamma"], artifact_for())
    finally:
        runner.close()


@pytest.mark.parametrize("retained", [False, True])
def test_actual_relation_calibration_correction_discount_and_REAL_training(tmp_path, retained):
    from tests.test_fit_reliability import calibration_fixture
    from tests.test_labeling_actual_aux import assets, Tokenizer, Encoder
    from loris.labeling import cli
    from loris.labeling.runner import LabelingRun
    from loris.labeling.trainer import train_actual

    source, parent, configuration = assets(tmp_path)
    args, calibration, protocol = calibration_fixture(tmp_path)
    manifest = json.loads(Path(args.manifest).read_text())
    if retained:
        from loris.labeling.reliability import RETAINED_CHASE_SEMANTICS

        manifest["loop"].update(
            conflict_policy="unknown_arbitrate", unknown_semantics="retain_consequences"
        )
        manifest["arbitration"] = dict(alias="aux", default_threshold=0.5, thresholds={})
        protocol["model_binding"]["chase_semantics"] = RETAINED_CHASE_SEMANTICS
    context_path = Path(manifest["context"]["path"])
    ctx = pickle.loads(context_path.read_bytes())
    ctx["docs"][0].cnt = "seed application"
    ctx["docs"][1].cnt = "target application"
    ctx["attribute_values"] = {"mtd.group": ["red", "red", "blue", "blank"]}
    ctx["attribute_presence"] = {"mtd.group": np.ones(4, bool)}
    context_path.write_bytes(pickle.dumps(ctx))
    manifest["context"] = file_record(context_path)
    rules = (
        Rule((Doc("x"), ML("x", "aux", "A", 0.8)), Label("x", "A")),
        Rule((Doc("x"), Text("x", MatchPredicate("cnt", "seed"))), Label("x", "A")),
        Rule(
            (Doc("x"), Doc("y"), AttrEqual("x", "mtd.group", "y", "mtd.group")),
            LabelCompare("x", "eq", "y"),
        ),
    )
    program = Path(manifest["program"]["path"])
    program.write_text(json.dumps([r.to_dict() for r in rules]))
    manifest["program"] = file_record(program)
    manifest["dynamic"].update(
        checkpoint=file_record(parent), backend_source=source, configuration=configuration
    )
    Path(args.manifest).write_text(json.dumps(manifest))
    calibration["docs"][0].cnt = "seed independent zero"
    calibration["attribute_values"] = {"mtd.group": ["calibration-group"] * 20}
    calibration["attribute_presence"] = {"mtd.group": np.ones(20, bool)}
    cp = Path(protocol["context"]["path"])
    cp.write_bytes(pickle.dumps(calibration))
    protocol["context"] = file_record(cp)
    protocol["model_binding"]["checkpoint_sha256"] = manifest["dynamic"]["checkpoint"]["sha256"]
    Path(args.protocol).write_text(json.dumps(protocol))
    assert main_calibrate(args) == 0
    artifact_path = Path(args.output) / "RELIABILITY.json"
    effective, ctx, rules, artifact, _ = cli.load(
        args.manifest, weighting="direct_rule_precision_discount", source_reliability=artifact_path
    )
    record = artifact["rule_scores"][2]
    assert (record["support"], record["correct"], record["q"]) == (20, 17, 18 / 22)
    from loris.labeling.reliability import relation_scoring, chase_semantics

    assert artifact["relation_scoring"] == relation_scoring(chase_semantics(effective))
    validate_artifact(artifact, ctx, rules, effective)
    old = copy.deepcopy(artifact)
    old.pop("relation_scoring")
    with pytest.raises(ValueError, match="recalibrated"):
        validate_artifact(old, ctx, rules, effective)
    batches = []

    def train(**kwargs):
        batches.append(kwargs["batch"])
        return train_actual(
            **kwargs, tokenizer_factory=lambda _: Tokenizer(), model_factory=lambda _, k: Encoder(k)
        )

    run = LabelingRun(
        context=ctx,
        rules=rules,
        manifest=effective,
        output=tmp_path / "run",
        trainer=train,
        reliability=artifact,
    )
    report = run.run()
    assert report["actual_optimizer_updates"] >= 1
    assert report[
        (
            "current_history_has_quarantined_fixpoint"
            if retained
            else "current_Gamma_has_valid_closure"
        )
    ]
    batch = batches[0]
    assert batch.weak_mask[1, 0] and batch.targets[1, 0] == 1
    expected = (18 / 22) * (1 - 0.5 * abs(0.2 - 1))
    assert batch.weights[1, 0] == pytest.approx(expected)
    trace = json.loads((tmp_path / "run/update_00001/ACTUAL_TRAINING_TRACE.json").read_text())
    assert trace["windows"][0]["weak_mass"] == pytest.approx(expected)
    assert set(trace["sampled_rows"]) == {0, 1}
