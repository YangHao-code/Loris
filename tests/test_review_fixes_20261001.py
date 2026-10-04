"""Regression tests for the 2026-10-01 review of the validated-Gamma-wins and soft-conflict paths."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, LabelCompare, ML, RefGamma, Rule, Text
from loris._engine.typed_runtime import IndexedChaseRunner
from loris._storage.discovery_factory import make_discovery_runner
from loris._storage.selection_execution import make_selection_runner
from loris.document import Document
from loris.labeling.runner import LabelingRun
from loris.labeling.unknown_incremental import retaining_chase
from loris.predicates import MatchPredicate
from tests.test_conflict_protocol_loop import close, forbidden_training, rule, setup


def at(var, row):
    return Text(var, MatchPredicate("cnt", "^node%d$" % row))


def labeling_run(tmp_path, rules, policy, gamma=None):
    ctx, manifest = setup()
    if gamma is not None:
        ctx["gamma"] = gamma
    manifest["loop"]["unknown_semantics"] = "retain_consequences"
    manifest["loop"]["gamma_conflict_policy"] = policy
    return LabelingRun(context=ctx, rules=rules, manifest=manifest, output=tmp_path / "run",
                       trainer=forbidden_training)


def llm_answer(run, directory, row, targets):
    directory.mkdir(exist_ok=True, parents=True)
    run.manifest["loop"]["label_budget"] = None
    run.manifest["loop"]["query_answer_mode"] = "allow_partial"
    run.manifest["oracle"].update(kind="directory_llm_then_human", directory=str(directory))
    (directory / ("%08d.llm.json" % row)).write_text(json.dumps(dict(
        row_id="r%d" % row, labels=list(run.labels), targets=targets, provider="llm",
        provenance="synthetic review regression")))


def propose(run, tmp_path, row, targets, result, session):
    llm_answer(run, tmp_path / "answers", row, targets)
    return bool(run.oracle_answer(row, "review", result=result, session=session, version=run.version))


@pytest.mark.parametrize("policy", ["quarantine", "validated_gamma_wins"])
def test_llm_proposal_contradicting_its_own_consequence_is_rejected(tmp_path, policy):
    run = labeling_run(tmp_path, (rule("B", Label("x", "A"), sign="neg"),), policy)
    try:
        result, session, _ = run.chase(run.version, phase="initial_zero_update")
        assert not propose(run, tmp_path, 0, [1, 1, None, None], result, session)
        assert not run.gamma.pos[0].any()
    finally:
        close(run)


@pytest.mark.parametrize("policy", ["quarantine", "validated_gamma_wins"])
def test_llm_proposal_whose_consequence_contradicts_trusted_gamma_is_rejected(tmp_path, policy):
    pos = np.zeros((2, 4), bool)
    pos[0, 2] = True
    run = labeling_run(tmp_path, (rule("C", Label("x", "A"), sign="neg"),), policy,
                       gamma=RefGamma(pos, np.zeros_like(pos)))
    try:
        result, session, _ = run.chase(run.version, phase="initial_zero_update")
        assert not propose(run, tmp_path, 0, [1, None, None, None], result, session)
        assert not run.gamma.pos[0, 0]
    finally:
        close(run)


def test_answers_accepted_earlier_in_the_batch_are_not_attributed_to_a_later_proposal(tmp_path):
    # doc0 derives A; a human answer accepted earlier in the same batch says
    # doc0 has no A. A later consistent proposal for doc1 is still validated.
    run = labeling_run(tmp_path, (rule("A"),), "validated_gamma_wins")
    try:
        result, session, _ = run.chase(run.version, phase="initial_zero_update")
        assert result.positive_facts[0, 0]
        neg = run.gamma.neg.copy()
        neg[0] = True
        run.gamma = RefGamma(run.gamma.pos, neg)
        assert propose(run, tmp_path, 1, [0, 0, 0, 1], result, session)
        assert run.gamma.pos[1, 3]
    finally:
        close(run)


def test_proposal_contradicting_an_earlier_answer_through_a_rule_is_rejected(tmp_path):
    # doc1 has A -> doc0 has B; an earlier accepted answer says doc0 has no B.
    cross = Rule((Doc("x"), Doc("y"), at("x", 1), at("y", 0), Label("x", "A")), Label("y", "B"))
    run = labeling_run(tmp_path, (cross,), "validated_gamma_wins")
    try:
        result, session, _ = run.chase(run.version, phase="initial_zero_update")
        neg = run.gamma.neg.copy()
        neg[0] = True
        run.gamma = RefGamma(run.gamma.pos, neg)
        assert not propose(run, tmp_path, 1, [1, None, None, None], result, session)
    finally:
        close(run)


def retaining_case(gamma_wins, tmp_path):
    rules = (Rule((Doc("x"), at("x", 0), ML("x", "aux", "A", .5)), Label("x", "A")),
             Rule((Doc("x"), Doc("y"), at("x", 0), at("y", 1), Label("x", "A")), Label("y", "A")),
             Rule((Doc("x"), at("x", 0), Label("x", "A")), Label("x", "B")),
             Rule((Doc("x"), at("x", 0), Label("x", "B")), Label("x", "A", "neg")))
    ctx = dict(docs=[Document(cnt="node%d" % i, ttl="", mtd="") for i in range(2)], labels=["A", "B"],
               gamma=RefGamma.empty(2, 2), model_probabilities={"aux": np.full((2, 2), .9)}, model_decisions={})
    first = IndexedChaseRunner(ctx, tmp_path / "a")
    before = retaining_chase(first, rules, ctx["gamma"], gamma_wins=gamma_wins)
    neg = np.zeros((2, 2), bool)
    neg[0] = True
    answered = RefGamma(np.zeros((2, 2), bool), neg)
    later = IndexedChaseRunner(dict(ctx, gamma=answered), tmp_path / "b")
    inc = retaining_chase(later, rules, answered, before=before, old_gamma=ctx["gamma"], update_kind="chaseInc",
                          initial_unknown=np.zeros((2, 2), bool), gamma_wins=gamma_wins)
    fresh = retaining_chase(later, rules, answered, gamma_wins=gamma_wins)
    first.close(); later.close()
    return before, inc, fresh


def test_new_answer_withdraws_descendants_of_isolated_refuted_versions(tmp_path):
    before, inc, fresh = retaining_case(True, tmp_path)
    assert before.conflict_unknown[0, 0] and before.positive_facts[1, 0]
    for a, b in ((inc, fresh),):
        np.testing.assert_array_equal(a.positive_facts, b.positive_facts)
        np.testing.assert_array_equal(a.negative_facts, b.negative_facts)
    assert not inc.positive_facts[1, 0]
    assert inc.reference_audit["maintenance"]["gamma_refuted_isolated_versions_withdrawn"] >= 1


def test_original_quarantine_semantics_keep_their_approximation(tmp_path):
    _, inc, _ = retaining_case(False, tmp_path)
    assert inc.positive_facts[1, 0]  # unchanged original behaviour: descendants retained
    assert inc.reference_audit["maintenance"]["gamma_refuted_isolated_versions_withdrawn"] == 0


def test_check_candidates_are_validated(tmp_path):
    ctx = dict(docs=[Document(cnt="node0", ttl="", mtd="")], labels=["A"], gamma=RefGamma.empty(1, 1),
               model_probabilities={}, model_decisions={})
    runner = IndexedChaseRunner(ctx, tmp_path / "c")
    try:
        rules = (Rule((Doc("x"), at("x", 0)), Label("x", "A")),)
        before = retaining_chase(runner, rules, ctx["gamma"], gamma_wins=True)
        mask = np.ones((1, 1), bool)
        with pytest.raises(ValueError):
            retaining_chase(runner, rules, ctx["gamma"], gamma_wins=True, check_candidates=(mask, ~mask))
        with pytest.raises(ValueError):  # not cells of the checked Gamma
            retaining_chase(runner, rules, ctx["gamma"], before=before, update_kind="check",
                            gamma_wins=True, check_candidates=(mask, ~mask))
        plain = retaining_chase(runner, rules, ctx["gamma"], before=before, update_kind="check", gamma_wins=True)
        assert plain.candidate_contradictions is None
    finally:
        runner.close()


def edge(a, b):
    return Rule((Doc("x"), Doc("y"), at("x", a), at("y", b)), LabelCompare("x", "subset", "y"))


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_selection_relation_batching_copies_labels_blocked_on_the_chain(tmp_path, order):
    # doc0 has A, doc1 is validated not-A: 0 <= 1 <= 2 must still copy A to doc2.
    pos = np.zeros((3, 1), bool); neg = np.zeros((3, 1), bool)
    pos[0, 0] = True; neg[1, 0] = True
    ctx = dict(docs=[Document(cnt="node%d" % i, ttl="", mtd="") for i in range(3)], labels=["A"],
               gamma=RefGamma(pos, neg), model_probabilities={}, model_decisions={})
    rules = tuple((edge(0, 1), edge(1, 2))[i] for i in order)
    runners = dict(batched=make_selection_runner(ctx, tmp_path / "s", gamma_protection=True),
                   scalar=make_selection_runner(ctx, tmp_path / "t", relation_batching=False, gamma_protection=True),
                   proof=make_discovery_runner(ctx, tmp_path / "p", gamma_protection=True))
    try:
        results = {name: r(rules, gamma=ctx["gamma"]) for name, r in runners.items()}
        for result in results.values():
            assert result.status == "fixpoint"
            assert result.positive_facts[:, 0].tolist() == [True, False, True]
    finally:
        for r in runners.values():
            r.close()


def test_original_semantics_variant_and_its_calibration_directory(tmp_path):
    from loris.experiments.workflow import _calibration_directory, _original_semantics

    data = dict(loop=dict(conflict_policy="unknown_arbitrate", unknown_semantics="retain_consequences",
                          gamma_conflict_policy="validated_gamma_wins", commit_gate="independent_calibration",
                          commit_gate_tolerance=0.005),
                fallback=dict(enabled=True), calibration=dict(grid=[0.3, 0.5], minimum_support_each_sign=10,
                                                              cells="trusted_only", max_step=0.1))
    _original_semantics(data)
    assert data["loop"]["gamma_conflict_policy"] == "quarantine"
    assert data["loop"]["commit_gate"] == "none" and "commit_gate_tolerance" not in data["loop"]
    assert "fallback" not in data
    assert data["calibration"] == dict(grid=[0.3, 0.5], minimum_support_each_sign=1)
    assert _calibration_directory(tmp_path, data, "original_semantics").name == "calibration-original"
    assert _calibration_directory(tmp_path, data, "calibration_original").name == "calibration-original"
    assert _calibration_directory(tmp_path, data).name == "calibration"


def valid_manifest():
    from loris.labeling import cli

    _, manifest = setup()
    manifest["feedback"].update(trusted_replay=True, trusted_replay_fraction=0.05)
    cli.validate(manifest)
    return manifest


def test_auto_amp_resolution_and_manifest_validation():
    from loris.labeling import cli
    from loris.labeling.trainer import resolve_amp

    assert resolve_amp(dict(amp="auto", seed=1), "cpu") == dict(amp="none", seed=1)
    assert resolve_amp(dict(amp="fp16"), "cuda") == dict(amp="fp16")
    good = valid_manifest()
    good["dynamic"]["configuration"]["amp"] = "auto"
    good["dynamic"]["amp_init_scale"] = 1024
    cli.validate(good)
    for key, value in (("amp", "fp8"), ("amp_init_scale", 0), ("amp_init_scale", float("inf"))):
        bad = copy.deepcopy(good)
        if key == "amp":
            bad["dynamic"]["configuration"]["amp"] = value
        else:
            bad["dynamic"][key] = value
        with pytest.raises(ValueError):
            cli.validate(bad)


def test_calibration_anchor_option_was_removed():
    from loris.labeling import cli
    from loris.labeling.feedback import calibrate
    import inspect

    assert "anchor" not in inspect.signature(calibrate).parameters
    manifest = valid_manifest()
    manifest["calibration"]["anchor"] = "initial"
    with pytest.raises(ValueError):
        cli.validate(manifest)
