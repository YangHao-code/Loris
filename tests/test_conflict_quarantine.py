"""Complete new quarantine semantics, independent of first-conflict Chase."""

import itertools

import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, LabelCompare, ML, RefGamma, Rule, Text
from loris._engine.typed_runtime import IndexedChaseRunner
from loris.document import Document
from loris.labeling.quarantine import SEMANTICS, quarantine_chase
from loris.predicates import MatchPredicate


def context(n=1, labels=("A", "B", "C", "D")):
    return dict(
        docs=[Document(cnt="node%d" % i, ttl="", mtd="") for i in range(n)],
        labels=list(labels),
        gamma=RefGamma.empty(n, len(labels)),
        model_probabilities={"aux": np.full((n, len(labels)), 0.8)},
        model_decisions={},
    )


def at(var, row):
    return Text(var, MatchPredicate("cnt", "^node%d$" % row))


def local(head, *body):
    return Rule((Doc("x"),) + body, head)


def assert_proof(result):
    proof = result.proof_dag
    active = set()
    done = set()

    def visit(key):
        assert key in proof
        if key in done:
            return
        assert key not in active
        active.add(key)
        node = proof[key]
        if node["kind"] in ("pos", "neg"):
            assert not result.conflict_unknown[node["row"], node["column"]]
            matrix = result.positive_facts if node["kind"] == "pos" else result.negative_facts
            assert matrix[node["row"], node["column"]]
        for dep in node["dependencies"]:
            visit(dep)
        active.remove(key)
        done.add(key)

    for key in proof:
        visit(key)
    assert not (result.positive_facts & result.negative_facts).any()
    assert not ((result.positive_facts | result.negative_facts) & result.conflict_unknown).any()


def test_all_rule_orders_isolate_both_signs_before_rule_body_propagation(tmp_path):
    ctx = context()
    rules = (
        local(Label("x", "A")),
        local(Label("x", "A", "neg")),
        local(Label("x", "B"), Label("x", "A")),
        local(Label("x", "C"), Label("x", "A", "neg")),
    )
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        for order in itertools.permutations(rules):
            result = quarantine_chase(runner, order, ctx["gamma"])
            assert result.status == "fixpoint"
            assert np.array_equal(result.conflict_unknown, [[True, False, False, False]])
            assert not result.positive_facts.any() and not result.negative_facts.any()
            assert result.reference_audit["semantics"] == SEMANTICS
            assert_proof(result)
    finally:
        runner.close()


def test_same_batch_collects_every_collision_without_calling_old_Chase(tmp_path, monkeypatch):
    ctx = context()
    rules = tuple(local(Label("x", label, sign)) for label in ("A", "B") for sign in ("pos", "neg"))

    def forbidden(*args, **kwargs):
        pytest.fail("quarantine must not consume the old first-conflict partial chase")

    monkeypatch.setattr(IndexedChaseRunner, "__call__", forbidden)
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        result = quarantine_chase(runner, rules, ctx["gamma"])
        np.testing.assert_array_equal(result.conflict_unknown, [[True, True, False, False]])
        assert result.status == "fixpoint"
        events = result.reference_audit["maintenance"]["quarantines"]
        assert len(events) == 1 and events[0]["newly_quarantined"] == [[0, 0], [0, 1]]
        assert_proof(result)
    finally:
        runner.close()


def test_candidates_stream_against_unchanged_snapshot_while_live_heads_are_applied(
    tmp_path, monkeypatch
):
    from loris.labeling import quarantine

    ctx = context(n=3)
    original_valuations, original_apply = quarantine.valuations, quarantine.apply_head
    applied = []
    observed = []

    def tracking_apply(runner, rule, ri, valuation, state):
        applied.append(id(state))
        return original_apply(runner, rule, ri, valuation, state)

    def tracking_valuations(runner, rule, prepared, state, **kwargs):
        before_pos = state.pos.copy()
        before_bits = dict(state.label_bits)
        before_relations = tuple(
            (name, tuple(getattr(state, name))) for name in ("out", "inc", "proper", "proper_inc")
        )
        for valuation in original_valuations(runner, rule, prepared, state, **kwargs):
            count = len(applied)
            yield valuation
            # Materializing all candidates before application would fail here.
            assert len(applied) == count + 1 and applied[-1] != id(state)
            np.testing.assert_array_equal(state.pos, before_pos)
            assert state.label_bits == before_bits
            assert before_relations == tuple(
                (name, tuple(getattr(state, name)))
                for name in ("out", "inc", "proper", "proper_inc")
            )
            observed.append(True)

    monkeypatch.setattr(quarantine, "apply_head", tracking_apply)
    monkeypatch.setattr(quarantine, "valuations", tracking_valuations)
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        result = quarantine_chase(runner, (local(Label("x", "A")),), ctx["gamma"])
        assert len(observed) == 3 and result.positive_facts[:, 0].all()
        assert (
            result.reference_audit["maintenance"]["candidate_storage"]
            == "streamed_from_frozen_valuation_indexes"
        )
        assert_proof(result)
    finally:
        runner.close()


@pytest.mark.parametrize("alternate", [False, True])
def test_late_collision_retracts_downstream_and_unrooted_cycle_but_restores_alternative(
    tmp_path, alternate
):
    ctx = context()
    # A -> B -> C -> !A discovers the collision only after B/C existed.
    rules = [
        local(Label("x", "A")),
        local(Label("x", "B"), Label("x", "A")),
        local(Label("x", "C"), Label("x", "B")),
        local(Label("x", "A", "neg"), Label("x", "C")),
        local(Label("x", "B"), Label("x", "C")),
    ]
    if alternate:
        rules.append(local(Label("x", "C"), ML("x", "aux", "D", 0.5)))
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        result = quarantine_chase(runner, rules, ctx["gamma"])
        assert result.status == "fixpoint" and result.conflict_unknown[0, 0]
        assert bool(result.positive_facts[0, 1]) == alternate
        assert bool(result.positive_facts[0, 2]) == alternate
        assert result.reference_audit["maintenance"]["rebuilds"] == 2
        assert_proof(result)
    finally:
        runner.close()


def test_verified_Gamma_record_survives_while_execution_pair_is_quarantined(tmp_path):
    ctx = context()
    pos = ctx["gamma"].pos.copy()
    pos[0, 0] = True
    ctx["gamma"] = RefGamma(pos, ctx["gamma"].neg)
    original_pos = ctx["gamma"].pos.copy()
    rules = (local(Label("x", "A", "neg")), local(Label("x", "B"), Label("x", "A")))
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        result = quarantine_chase(runner, rules, ctx["gamma"])
        assert result.status == "fixpoint" and result.conflict_unknown[0, 0]
        assert not result.positive_facts.any() and not result.negative_facts.any()
        np.testing.assert_array_equal(ctx["gamma"].pos, original_pos)
        np.testing.assert_array_equal(runner.gamma.pos, original_pos)
        assert result.reference_audit["maintenance"]["Gamma_execution_quarantined_cells"] == 1
        assert_proof(result)
    finally:
        runner.close()


def test_relation_copy_masks_unknown_target_and_preserves_transitive_alternative(tmp_path):
    ctx = context(n=3)
    pos = ctx["gamma"].pos.copy()
    pos[0, 0] = True
    neg = ctx["gamma"].neg.copy()
    neg[1, 0] = True
    ctx["gamma"] = RefGamma(pos, neg, subset=((0, 1), (1, 2)))
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        result = quarantine_chase(runner, (), ctx["gamma"])
        assert result.status == "fixpoint"
        assert np.array_equal(result.conflict_unknown[:, 0], [False, True, False])
        assert np.array_equal(result.positive_facts[:, 0], [True, False, True])
        assert not result.negative_facts.any()
        assert (0, 2) in result.subset_facts
        # The direct transitive source 0 is still usable; no unknown label at
        # row1 is used as an intermediate factual premise.
        assert "pos:1:0" not in result.proof_dag and "neg:1:0" not in result.proof_dag
        assert_proof(result)
    finally:
        runner.close()


def test_relation_derived_from_quarantined_premise_is_retracted_with_copied_labels(tmp_path):
    ctx = context(n=2)
    rules = (
        local(Label("x", "A"), at("x", 0)),
        Rule(
            (Doc("x"), Doc("y"), at("x", 0), at("y", 1), Label("x", "A")),
            LabelCompare("x", "subset", "y"),
        ),
        Rule((Doc("x"), Doc("y"), at("x", 0), at("y", 1), Label("y", "A")), Label("x", "A", "neg")),
    )
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        result = quarantine_chase(runner, rules, ctx["gamma"])
        assert result.status == "fixpoint" and result.conflict_unknown[0, 0]
        assert not result.positive_facts.any()
        assert (0, 1) not in result.subset_facts
        assert_proof(result)
    finally:
        runner.close()


def test_temporary_roots_and_initial_unknown_are_not_promoted_or_modified(tmp_path):
    ctx = context()
    positive = np.array([[True, True, False, False]])
    negative = np.array([[True, False, False, False]])
    unknown = np.array([[False, False, True, False]])
    original = (positive.copy(), negative.copy(), unknown.copy())
    rules = (local(Label("x", "C")), local(Label("x", "D"), Label("x", "B")))
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        result = quarantine_chase(
            runner,
            rules,
            ctx["gamma"],
            initial_positive=positive,
            initial_negative=negative,
            initial_unknown=unknown,
        )
        assert result.status == "fixpoint"
        np.testing.assert_array_equal(result.conflict_unknown, [[True, False, True, False]])
        np.testing.assert_array_equal(result.positive_facts, [[False, True, False, True]])
        assert result.proof_dag["pos:0:1"]["origin"] == "temporary_seed"
        assert result.proof_dag["pos:0:1"]["persistent_Gamma"] is False
        assert not ctx["gamma"].pos.any() and not ctx["gamma"].neg.any()
        for actual, before in zip((positive, negative, unknown), original):
            np.testing.assert_array_equal(actual, before)
        assert_proof(result)
    finally:
        runner.close()


def test_fixed_unknown_replay_drops_temporary_only_descendants_and_keeps_Gamma_record(tmp_path):
    ctx = context()
    pos = ctx["gamma"].pos.copy()
    pos[0, 0] = True
    ctx["gamma"] = RefGamma(pos, ctx["gamma"].neg)
    temporary = np.array([[False, True, False, False]])
    rules = (local(Label("x", "A", "neg")), local(Label("x", "C"), Label("x", "B")))
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        full = quarantine_chase(runner, rules, ctx["gamma"], initial_positive=temporary)
        evidence_only = quarantine_chase(
            runner, rules, ctx["gamma"], initial_unknown=full.conflict_unknown
        )
        assert full.positive_facts[0, 1] and full.positive_facts[0, 2]
        assert not evidence_only.positive_facts.any()
        np.testing.assert_array_equal(evidence_only.conflict_unknown, full.conflict_unknown)
        assert ctx["gamma"].pos[0, 0]
        assert_proof(full)
        assert_proof(evidence_only)
    finally:
        runner.close()


@pytest.mark.parametrize("cycle", [True, False])
def test_independent_strict_relations_still_fail_with_clear_diagnostic(tmp_path, cycle):
    ctx = context(n=3, labels=("A",))
    ctx["gamma"] = RefGamma(
        ctx["gamma"].pos,
        ctx["gamma"].neg,
        subset=((1, 0),) if cycle else (),
        strict_subset=((0, 1),) if cycle else ((0, 1), (1, 2)),
    )
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        result = quarantine_chase(runner, (), ctx["gamma"])
        assert result.status == "conflict" and not result.conflict_unknown.any()
        assert result.comparison_conflicts
        kind = result.comparison_conflicts[0]["kind"]
        assert kind == (
            "strict_inclusion_cycle" if cycle else "finite_label_comparisons_unsatisfiable"
        )
        assert (
            result.reference_audit["independent_relation_conflict"] == result.comparison_conflicts
        )
        assert_proof(result)
    finally:
        runner.close()


def test_no_conflict_matches_complete_existing_chase_with_relations_and_negative_body(tmp_path):
    ctx = context(n=2)
    rules = (
        local(Label("x", "A"), at("x", 0), ML("x", "aux", "A", 0.5)),
        local(Label("x", "B", "neg"), at("x", 0)),
        local(Label("x", "C"), Label("x", "B", "neg")),
        Rule((Doc("x"), Doc("y"), at("x", 0), at("y", 1)), LabelCompare("x", "subset", "y")),
    )
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        expected = runner(rules, ctx["gamma"])
        actual = quarantine_chase(runner, rules, ctx["gamma"])
        assert actual.status == expected.status == "fixpoint"
        np.testing.assert_array_equal(actual.positive_facts, expected.positive_facts)
        np.testing.assert_array_equal(actual.negative_facts, expected.negative_facts)
        assert set(actual.subset_facts) == set(expected.subset_facts)
        assert not actual.conflict_unknown.any()
        assert_proof(actual)
    finally:
        runner.close()
