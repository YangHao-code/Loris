"""Author-added q protocol counts only retained, arbitration-free conclusions."""

import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, Rule, Text, RefGamma
from loris._engine.typed_session import digest
from loris._storage.discovery_factory import make_discovery_runner
from loris.discovery.manifest import file_record
from loris.labeling.quarantine import quarantine_chase
from loris.labeling.reliability import (
    QUARANTINE_CHASE_SEMANTICS,
    STRICT_CHASE_SEMANTICS,
    _estimate,
    count_rule_conclusions,
    direct_rule_weights,
    lineage_scores,
    validate_artifact,
)
from loris.predicates import MatchPredicate
from tests.test_fit_reliability import calibration_fixture
from tests.test_relation_rule_reliability import ctx_for, relation, scored_artifact


def label_rule(text, label, sign="pos"):
    return Rule((Doc("x"), Text("x", MatchPredicate("cnt", text))), Label("x", label, sign))


def artifact(rules, counts):
    value = scored_artifact(rules, counts)
    value["chase_semantics"] = QUARANTINE_CHASE_SEMANTICS
    return value


def test_actual_conflicting_chase_excludes_unknown_and_retains_other_relation_targets(tmp_path):
    ctx = ctx_for(("source", "target"))
    rules = (
        label_rule("source", "A"),
        label_rule("target", "A", "neg"),
        label_rule("source", "B"),
        relation(),
    )
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        assert runner(rules, ctx["gamma"]).status == "conflict"
        closure = quarantine_chase(runner, rules, ctx["gamma"])
        assert closure.status == "fixpoint"
        assert closure.conflict_unknown.tolist() == [[False, False], [True, False]]
        counts = count_rule_conclusions(
            runner,
            closure,
            rules,
            np.ones((2, 2), bool),
            np.zeros((2, 2), bool),
            np.ones(2, bool),
        )
        assert [(r["support"], r["correct"]) for r in counts] == [(1, 1), (0, 0), (1, 1), (1, 1)]
        weights, audit = direct_rule_weights(
            runner, rules, closure, ctx["gamma"], artifact(rules, [(8, 1), (8, 8), (8, 4), (8, 7)])
        )
        assert np.allclose(weights, [[0.2, 0.5], [0, 0.8]])
        assert audit["unsupported_nonGamma_labels"] == 0
        assert audit["arbitration_conclusions_used"] is False
        assert audit["conflict_unknown_pairs"] == 1
    finally:
        runner.close()


def test_quarantined_Gamma_root_has_no_weight_or_transport_but_ledger_is_unchanged(tmp_path):
    ctx = ctx_for(("source", "target"))
    pos = np.array([[True, True], [False, False]])
    ctx["gamma"] = RefGamma(pos, np.zeros_like(pos), ((0, 1),))
    rules = (label_rule("source", "A", "neg"),)
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        closure = quarantine_chase(runner, rules, ctx["gamma"])
        assert closure.conflict_unknown[0, 0]
        assert not closure.positive_facts[:, 0].any()
        weights, audit = direct_rule_weights(
            runner, rules, closure, ctx["gamma"], artifact(rules, [(8, 8)])
        )
        assert np.allclose(weights, [[0, 1], [0, 1]])
        assert ctx["gamma"].pos[0].all()
        assert audit["unsupported_nonGamma_labels"] == 0
    finally:
        runner.close()


@pytest.mark.parametrize("trusted_tail", [False, True])
def test_valid_relation_path_can_bypass_unknown_intermediate_without_reviving_it(
    tmp_path, trusted_tail
):
    ctx = ctx_for(("source", "middle", "target"))
    gp = np.zeros((3, 2), bool)
    gp[0, 0] = True
    ctx["gamma"] = RefGamma(gp, np.zeros_like(gp), ((1, 2),) if trusted_tail else ())
    rules = (relation("source", "middle"), label_rule("middle", "A", "neg"))
    if not trusted_tail:
        rules += (relation("middle", "target"),)
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        closure = quarantine_chase(runner, rules, ctx["gamma"])
        assert closure.conflict_unknown[1, 0]
        assert closure.positive_facts[:, 0].tolist() == [True, False, True]
        counts = count_rule_conclusions(
            runner,
            closure,
            rules,
            np.ones((3, 2), bool),
            np.zeros((3, 2), bool),
            np.ones(3, bool),
        )
        assert [r["support"] for r in counts] == ([1, 0] if trusted_tail else [0, 0, 1])
        weights, audit = direct_rule_weights(
            runner,
            rules,
            closure,
            ctx["gamma"],
            artifact(rules, [(8, 1), (8, 4)] if trusted_tail else [(8, 1), (8, 4), (8, 8)]),
        )
        assert np.allclose(weights[:, 0], [1, 0, 0.2 if trusted_tail else 0.9])
        assert audit["unsupported_nonGamma_labels"] == 0
    finally:
        runner.close()


def test_mask_does_not_hide_unrelated_missing_closure_fact_or_temporary_arbitration(tmp_path):
    ctx = ctx_for(("source",))
    rules = (label_rule("source", "A"), label_rule("source", "A", "neg"), label_rule("source", "B"))
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        closure = quarantine_chase(runner, rules, ctx["gamma"])
        missing = copy.copy(closure)
        missing.positive_facts = closure.positive_facts.copy()
        missing.positive_facts[0, 1] = False
        with pytest.raises(ValueError, match="absent"):
            count_rule_conclusions(
                runner,
                missing,
                rules,
                np.ones((1, 2), bool),
                np.zeros((1, 2), bool),
                np.ones(1, bool),
            )
        temporary = copy.copy(closure)
        temporary.positive_facts = closure.positive_facts.copy()
        temporary.positive_facts[0, 0] = True
        with pytest.raises(ValueError, match="arbitration-free"):
            count_rule_conclusions(
                runner,
                temporary,
                rules,
                np.ones((1, 2), bool),
                np.zeros((1, 2), bool),
                np.ones(1, bool),
            )
    finally:
        runner.close()


def test_runtime_artifact_semantics_must_match_even_when_current_closure_has_no_collision(tmp_path):
    ctx = ctx_for(("source",))
    rules = (label_rule("source", "A"),)
    new = artifact(rules, [(8, 8)])
    old = copy.deepcopy(new)
    del old["chase_semantics"]
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        quarantine = quarantine_chase(runner, rules, ctx["gamma"])
        strict = runner(rules, ctx["gamma"])
        with pytest.raises(ValueError, match="chase_semantics"):
            direct_rule_weights(runner, rules, quarantine, ctx["gamma"], old)
        with pytest.raises(ValueError, match="chase_semantics"):
            direct_rule_weights(runner, rules, strict, ctx["gamma"], new)
        direct_rule_weights(runner, rules, strict, ctx["gamma"], old)
    finally:
        runner.close()


def test_complete_provenance_ablation_keeps_the_full_program_quarantine_fixed(tmp_path):
    ctx = ctx_for(("source",))
    rules = (label_rule("source", "A"), label_rule("source", "A", "neg"), label_rule("source", "B"))
    runner = make_discovery_runner(ctx, tmp_path / "cache")
    try:
        full = quarantine_chase(runner, rules, ctx["gamma"])
        scores = artifact(rules, [(8, 8), (8, 1), (8, 4)])
        values, audit = lineage_scores(
            lambda chosen, gamma: quarantine_chase(
                runner, chosen, gamma, initial_unknown=full.conflict_unknown
            ),
            rules,
            ctx["gamma"],
            scores,
        )
        assert np.allclose(values, [[0, 0.5]])
        assert audit["chase_semantics"] == QUARANTINE_CHASE_SEMANTICS
        with pytest.raises(ValueError, match="fixed full-program quarantine"):
            lineage_scores(
                lambda chosen, gamma: quarantine_chase(runner, chosen, gamma),
                rules,
                ctx["gamma"],
                scores,
            )
    finally:
        runner.close()


def actual_estimate(tmp_path, *, quarantine=True, conflict=True):
    args, calibration, protocol = calibration_fixture(tmp_path)
    manifest = json.loads(Path(args.manifest).read_text())
    manifest["loop"]["conflict_policy"] = "unknown_arbitrate" if quarantine else "halt"
    with Path(manifest["context"]["path"]).open("rb") as stream:
        target = pickle.load(stream)
    rules = (label_rule("red", "B"),)
    if conflict:
        rules += (label_rule("red", "B", "neg"),)
    rules += (Rule((Doc("x"),), Label("x", "A")),)
    status = _estimate(
        manifest,
        target,
        rules,
        protocol,
        calibration,
        calibration["gamma"].pos,
        calibration["gamma"].neg,
        np.ones(20, bool),
        Path(args.output),
        Path(protocol["context"]["path"]),
        Path(protocol["answers"]["path"]),
        {},
    )
    assert status == 0
    output = Path(args.output)
    value = json.loads((output / "RELIABILITY.json").read_text())
    return value, manifest, target, rules, output


def test_actual_new_calibration_uses_quarantine_counts_without_arbitration_and_binds_receipt(
    tmp_path,
):
    value, manifest, target, rules, output = actual_estimate(tmp_path)
    assert [r["support"] for r in value["rule_scores"]] == [0, 0, 20]
    assert [r["correct"] for r in value["rule_scores"]] == [0, 0, 17]
    assert value["rule_scores"][2]["q"] == 18 / 22
    assert value["chase_semantics"] == QUARANTINE_CHASE_SEMANTICS
    receipt = json.loads((output / "RESULTS.json").read_text())
    assert receipt["conflict_unknown_pairs"] == 18
    assert receipt["calibration_inference"] == "quarantine_without_arbitration"
    assert receipt["arbitration_conclusions_used"] is False
    assert receipt["answers_entered_inference"] is False
    assert receipt["model_updates"] == receipt["rule_learning_calls"] == 0
    validate_artifact(value, target, rules, manifest)
    tampered = copy.deepcopy(value)
    del tampered["calibration_binding"]["model_binding"]["chase_semantics"]
    with pytest.raises(ValueError, match="chase_semantics"):
        validate_artifact(tampered, target, rules, manifest)
    receipt["chase_semantics"] = STRICT_CHASE_SEMANTICS
    (output / "RESULTS.json").write_text(json.dumps(receipt))
    value["estimation_receipt"] = file_record(output / "RESULTS.json")
    with pytest.raises(ValueError, match="receipt"):
        validate_artifact(value, target, rules, manifest)


def test_legacy_strict_artifact_still_valid_but_cannot_be_relabelled_as_new_protocol(tmp_path):
    value, manifest, target, rules, output = actual_estimate(
        tmp_path, quarantine=False, conflict=False
    )
    # Recreate a historically authenticated strict artifact without any of the
    # newly introduced semantics fields, including its protocol and receipt.
    del value["chase_semantics"]
    del value["binding"]["chase_semantics"]
    del value["calibration_binding"]["chase_semantics"]
    del value["calibration_binding"]["model_binding"]["chase_semantics"]
    protocol = json.loads((output / "PROTOCOL.json").read_text())
    del protocol["model_binding"]["chase_semantics"]
    (output / "PROTOCOL.json").write_text(json.dumps(protocol))
    value["protocol"] = file_record(output / "PROTOCOL.json")
    receipt = json.loads((output / "RESULTS.json").read_text())
    del receipt["chase_semantics"]
    receipt["binding_id"] = digest(value["binding"])
    receipt["protocol"] = value["protocol"]
    receipt["calibration_binding"] = copy.deepcopy(value["calibration_binding"])
    (output / "RESULTS.json").write_text(json.dumps(receipt))
    value["estimation_receipt"] = file_record(output / "RESULTS.json")
    validate_artifact(value, target, rules, manifest)
    manifest["loop"]["conflict_policy"] = "unknown_arbitrate"
    with pytest.raises(ValueError, match="chase_semantics"):
        validate_artifact(value, target, rules, manifest)
    value["chase_semantics"] = QUARANTINE_CHASE_SEMANTICS
    with pytest.raises(ValueError, match="chase_semantics"):
        validate_artifact(value, target, rules, manifest)
