"""Small actual retaining-history executions; no training or external data."""

import copy
import itertools

import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, LabelCompare, ML, RefGamma, Rule, Text
from loris._engine.typed_runtime import IndexedChaseRunner, ProofDAG, _plain
from loris.document import Document
from loris.labeling.unknown_incremental import retaining_chase, recorded_feedback_view
from loris.predicates import MatchPredicate


def context(n=1, labels=("A", "B", "C", "D"), p=0.9):
    return dict(
        docs=[Document(cnt="node%d" % i, ttl="", mtd="") for i in range(n)],
        labels=list(labels),
        gamma=RefGamma.empty(n, len(labels)),
        model_probabilities={"aux": np.full((n, len(labels)), p)},
        model_decisions={},
    )


def at(var, row):
    return Text(var, MatchPredicate("cnt", "^node%d$" % row))


def local(head, *body):
    return Rule((Doc("x"),) + body, head)


def edge(a, b, *body, op="subset"):
    return Rule((Doc("x"), Doc("y"), at("x", a), at("y", b)) + body, LabelCompare("x", op, "y"))


def same(a, b):
    assert a.status == b.status
    np.testing.assert_array_equal(a.positive_facts, b.positive_facts)
    np.testing.assert_array_equal(a.negative_facts, b.negative_facts)
    np.testing.assert_array_equal(a.conflict_unknown, b.conflict_unknown)
    assert set(a.subset_facts) == set(b.subset_facts)
    assert set(a.strict_subset_facts) == set(b.strict_subset_facts)


@pytest.mark.parametrize("execution", ["incremental", "full_recompute"])
def test_synchronous_label_and_copy_candidates_do_not_depend_on_rule_order(tmp_path, execution):
    ctx = context(2)
    gp = ctx["gamma"].pos.copy()
    gp[0, 0] = True
    ctx["gamma"] = RefGamma(gp, ctx["gamma"].neg, ((0, 1),))
    rules = (
        local(Label("x", "A", "neg"), at("x", 1)),
        local(Label("x", "B"), Label("x", "A")),
        local(Label("x", "C"), Label("x", "A", "neg")),
    )
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        results = [
            retaining_chase(runner, order, ctx["gamma"], execution=execution)
            for order in itertools.permutations(rules)
        ]
        for result in results:
            assert result.conflict_unknown[1, 0]
            # The copy and negative rule collide in one frozen batch. Neither
            # sign at document 1 can trigger B/C; B may later arrive from doc0.
            assert not result.positive_facts[1, 2]
            assert not result.negative_facts.any()
            same(result, results[0])
    finally:
        runner.close()


def test_unknown_keeps_history_descendants_and_relation_for_future_labels(tmp_path):
    ctx = context(2)
    rules = (
        local(Label("x", "A"), at("x", 0), ML("x", "aux", "A", 0.5)),
        local(Label("x", "B"), Label("x", "A")),
        local(Label("x", "A", "neg"), Label("x", "B")),
        local(Label("x", "C"), Label("x", "B")),
        edge(0, 1, Label("x", "A")),
    )
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        before = retaining_chase(runner, rules, ctx["gamma"])
        assert before.status == "fixpoint" and before.conflict_unknown[:, 0].all()
        assert before.positive_facts[:, 1:3].all() and (0, 1) in before.subset_facts
        gp = ctx["gamma"].pos.copy()
        gp[0, 3] = True
        gamma = RefGamma(gp, ctx["gamma"].neg)
        after = retaining_chase(
            runner, rules, gamma, before=before, old_gamma=ctx["gamma"], update_kind="chaseInc"
        )
        assert after.positive_facts[:, 3].all()
        assert after.conflict_unknown[:, 0].all() and (0, 1) in after.subset_facts
        assert not ctx["gamma"].pos.any() and gamma.pos.sum() == 1
        view = recorded_feedback_view(runner, rules, after, gamma)
        assert view.recorded_source_view
        assert not view.positive_facts[:, 1:3].any()
        assert (0, 1) not in view.subset_facts
        assert view.positive_facts[0, 3] and not view.positive_facts[1, 3]
        assert after.positive_facts[:, 1:4].all()
        assert view.reference_audit["recorded_source_check"]["extra_Chase_calls"] == 0
    finally:
        runner.close()


def test_hidden_ml_ancestor_still_retracts_preserved_descendants_after_model_change(tmp_path):
    ctx = context()
    rules = (
        local(Label("x", "A"), ML("x", "aux", "A", 0.5)),
        local(Label("x", "B"), Label("x", "A")),
        local(Label("x", "A", "neg"), Label("x", "B")),
        local(Label("x", "C"), Label("x", "B")),
    )
    old = IndexedChaseRunner(ctx, tmp_path / "old")
    newctx = context(p=0.1)
    new = IndexedChaseRunner(newctx, tmp_path / "new")
    try:
        before = retaining_chase(old, rules, ctx["gamma"])
        assert before.conflict_unknown[0, 0] and before.positive_facts[0, 1:3].all()
        after = retaining_chase(
            new,
            rules,
            newctx["gamma"],
            before=before,
            old_runner=old,
            old_rules=rules,
            update_kind="chaseDelta",
        )
        assert not after.positive_facts.any() and not after.negative_facts.any()
        assert not after.conflict_unknown.any()
        assert after.reference_audit["maintenance"]["active_facts_removed_for_model_change"] >= 2
        assert before.positive_facts[0, 1:3].all()
    finally:
        old.close()
        new.close()


def test_unchanged_alternative_rederives_in_same_model_update(tmp_path):
    labels = ("A", "B", "C", "X", "Y", "Z")
    ctx = context(labels=labels)
    rules = (
        local(Label("x", "A"), ML("x", "aux", "A", 0.5)),
        local(Label("x", "B"), Label("x", "A")),
        local(Label("x", "C"), Label("x", "B")),
        local(Label("x", "Z")),
        local(Label("x", "Y"), Label("x", "Z")),
        local(Label("x", "X"), Label("x", "Y")),
        local(Label("x", "B"), Label("x", "X")),
    )
    old = IndexedChaseRunner(ctx, tmp_path / "old")
    newctx = context(labels=labels, p=0.1)
    new = IndexedChaseRunner(newctx, tmp_path / "new")
    try:
        before = retaining_chase(old, rules, ctx["gamma"])
        assert before.proof_dag["pos:0:1"]["rule_id"] == 1
        after = retaining_chase(
            new,
            rules,
            newctx["gamma"],
            before=before,
            old_runner=old,
            old_rules=rules,
            update_kind="chaseDelta",
        )
        assert not after.positive_facts[0, 0] and after.positive_facts[0, 1:].all()
        assert after.proof_dag["pos:0:1"]["rule_id"] == 6
        assert after.reference_audit["maintenance"]["restored_same_update"] >= 2
    finally:
        old.close()
        new.close()


def test_builtin_alternative_path_recovers_deleted_relation_and_copy(tmp_path):
    ctx = context(3)
    gp = ctx["gamma"].pos.copy()
    gp[0, 3] = True
    ctx["gamma"] = RefGamma(gp, ctx["gamma"].neg, ((1, 2),))
    rules = (
        edge(0, 2, ML("x", "aux", "A", 0.5)),
        local(Label("x", "B"), at("x", 0)),
        edge(0, 1, Label("x", "B")),
    )
    old = IndexedChaseRunner(ctx, tmp_path / "old")
    newctx = context(3, p=0.1)
    newctx["gamma"] = ctx["gamma"]
    new = IndexedChaseRunner(newctx, tmp_path / "new")
    try:
        before = retaining_chase(old, rules, ctx["gamma"])
        assert before.proof_dag["subset:0:2"]["rule_id"] == 0
        after = retaining_chase(
            new,
            rules,
            ctx["gamma"],
            before=before,
            old_runner=old,
            old_rules=rules,
            update_kind="chaseDelta",
        )
        assert (0, 2) in after.subset_facts and after.positive_facts[:, 3].all()
        assert after.proof_dag["subset:0:2"]["rule_id"] is None
    finally:
        old.close()
        new.close()


def test_gamma_protection_and_temporary_seed_replacement(tmp_path):
    ctx = context()
    gp = ctx["gamma"].pos.copy()
    gp[0, 1] = True
    ctx["gamma"] = RefGamma(gp, ctx["gamma"].neg)
    rules = (
        local(Label("x", "B"), Label("x", "A")),
        local(Label("x", "C"), Label("x", "B")),
        local(Label("x", "D"), Label("x", "A", "neg")),
    )
    old = IndexedChaseRunner(ctx, tmp_path / "old")
    newctx = context(p=0.1)
    newctx["gamma"] = ctx["gamma"]
    new = IndexedChaseRunner(newctx, tmp_path / "new")
    pos = np.zeros_like(gp)
    pos[0, 0] = True
    try:
        before = retaining_chase(old, rules, ctx["gamma"], initial_positive=pos)
        view = recorded_feedback_view(old, rules, before, ctx["gamma"])
        assert not view.positive_facts[0, 0] and view.positive_facts[0, 1:3].all()
        after = retaining_chase(
            new,
            rules,
            ctx["gamma"],
            before=before,
            old_runner=old,
            old_rules=rules,
            update_kind="chaseDelta",
            initial_negative=pos,
        )
        assert not after.positive_facts[0, 0] and after.negative_facts[0, 0]
        assert after.positive_facts[0, 1:].all()
        assert ctx["gamma"].pos.tolist() == [[False, True, False, False]]
    finally:
        old.close()
        new.close()


def test_next_round_rechecks_unchanged_negative_and_relation_sources(tmp_path):
    ctx = context(2)
    gp = ctx["gamma"].pos.copy()
    gp[0, 0] = True
    ctx["gamma"] = RefGamma(gp, ctx["gamma"].neg, ((0, 1),))
    rules = (local(Label("x", "A", "neg"), at("x", 1)),)
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        before = retaining_chase(runner, rules, ctx["gamma"])
        assert before.conflict_unknown[1, 0]
        after = retaining_chase(
            runner,
            rules,
            ctx["gamma"],
            before=before,
            update_kind="chaseInc",
            initial_unknown=np.zeros_like(gp),
        )
        assert after.conflict_unknown[1, 0] and not after.positive_facts[1, 0]
        with pytest.raises(ValueError, match="cannot unlock"):
            retaining_chase(
                runner,
                rules,
                ctx["gamma"],
                before=before,
                update_kind="check",
                initial_unknown=np.zeros_like(gp),
            )
    finally:
        runner.close()


@pytest.mark.parametrize("op", ["eq", "subset", "strict_subset"])
def test_relations_keep_direction_and_strict_constraints(tmp_path, op):
    ctx = context(2)
    gp = ctx["gamma"].pos.copy()
    gp[0, 0] = True
    gp[1, 1] = True
    ctx["gamma"] = RefGamma(gp, ctx["gamma"].neg)
    rules = (edge(0, 1, op=op),)
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        result = retaining_chase(runner, rules, ctx["gamma"])
        assert result.status == "fixpoint" and result.positive_facts[1, 0]
        assert bool(result.positive_facts[0, 1]) is (op == "eq")
        if op == "strict_subset":
            failed = retaining_chase(runner, rules + (edge(1, 0),), ctx["gamma"])
            assert failed.status == "conflict" and failed.comparison_conflicts
    finally:
        runner.close()


def test_unrooted_cycles_do_not_create_facts(tmp_path):
    ctx = context()
    rules = (local(Label("x", "A"), Label("x", "B")), local(Label("x", "B"), Label("x", "A")))
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        result = retaining_chase(runner, rules, ctx["gamma"])
        assert not result.positive_facts.any()
    finally:
        runner.close()


def test_random_multi_round_incremental_and_full_scan_share_retained_semantics(tmp_path):
    rng = np.random.RandomState(9011)
    for trial in range(6):
        ctx = context(3)
        ctx["model_probabilities"]["aux"] = rng.rand(3, 4)
        rules = (
            local(Label("x", "A"), ML("x", "aux", "A", 0.5)),
            local(Label("x", "B"), Label("x", "A")),
            local(Label("x", "A", "neg"), Label("x", "B"), ML("x", "aux", "B", 0.4)),
            local(Label("x", "C"), Label("x", "B")),
            edge(0, 1, Label("x", "A")),
            edge(1, 2, op="eq"),
        )
        old = IndexedChaseRunner(ctx, tmp_path / ("old%d" % trial))
        try:
            inc = retaining_chase(old, rules, ctx["gamma"])
            full = retaining_chase(old, rules, ctx["gamma"], execution="full_recompute")
            same(inc, full)
            for step in range(3):
                nxt = context(3)
                nxt["model_probabilities"]["aux"] = rng.rand(3, 4)
                new = IndexedChaseRunner(nxt, tmp_path / ("next%d-%d" % (trial, step)))
                try:
                    inc = retaining_chase(
                        new,
                        rules,
                        nxt["gamma"],
                        before=inc,
                        old_runner=old,
                        old_rules=rules,
                        update_kind="chaseDelta",
                    )
                    full = retaining_chase(
                        new,
                        rules,
                        nxt["gamma"],
                        before=full,
                        old_runner=old,
                        old_rules=rules,
                        update_kind="chaseDelta",
                        execution="full_recompute",
                    )
                    same(inc, full)
                except BaseException:
                    new.close()
                    raise
                old.close()
                old = new
        finally:
            old.close()


def test_versioned_withdrawal_does_not_retire_replacement_witness(tmp_path):
    ctx = context()
    rules = (
        local(Label("x", "A"), ML("x", "aux", "A", 0.5)),
        local(Label("x", "B"), Label("x", "A")),
        local(Label("x", "C"), Label("x", "B")),
        local(Label("x", "B", "neg"), Label("x", "C"), ML("x", "aux", "D", 0.5)),
    )
    old = IndexedChaseRunner(ctx, tmp_path / "old")
    middle_ctx = context()
    middle_ctx["model_probabilities"]["aux"][0, 3] = 0.1
    middle = IndexedChaseRunner(middle_ctx, tmp_path / "middle")
    final_ctx = context(p=0.1)
    final = IndexedChaseRunner(final_ctx, tmp_path / "final")
    temporary = np.zeros((1, 4), bool)
    temporary[0, 1] = True
    try:
        before = retaining_chase(old, rules, ctx["gamma"])
        old_b = before.history_latest["pos:0:1"]
        assert before.conflict_unknown[0, 1]
        second = retaining_chase(
            middle,
            rules,
            ctx["gamma"],
            before=before,
            old_runner=old,
            old_rules=rules,
            update_kind="chaseDelta",
            initial_positive=temporary,
        )
        new_b = second.history_current["pos:0:1"]
        assert new_b != old_b and second.history_nodes[new_b]["origin"] == "temporary_seed"
        third = retaining_chase(
            final,
            rules,
            ctx["gamma"],
            before=second,
            old_runner=middle,
            old_rules=rules,
            update_kind="chaseDelta",
            initial_positive=temporary,
        )
        assert old_b in third.history_retired
        assert new_b not in third.history_retired
        assert third.history_current["pos:0:1"] == new_b
        assert third.positive_facts[0, 1:3].all()
        assert third.history_nodes[third.history_current["pos:0:2"]]["history_dependencies"] == (
            new_b,
        )
    finally:
        old.close()
        middle.close()
        final.close()


def test_recorded_feedback_handles_deep_history_and_current_scope(tmp_path):
    labels = tuple("L%d" % i for i in range(1050))
    ctx = context(labels=labels)
    rules = (local(Label("x", labels[0]), ML("x", "aux", labels[0], 0.5)),) + tuple(
        local(Label("x", labels[i]), Label("x", labels[i - 1])) for i in range(1, len(labels))
    )
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        result = retaining_chase(runner, rules, ctx["gamma"])
        assert result.positive_facts.all()
        mask = np.zeros((1, len(labels)), bool)
        mask[0, -1] = True
        view = recorded_feedback_view(runner, rules, result, ctx["gamma"], candidate_mask=mask)
        assert view.positive_facts[0, -1]
        assert view.reference_audit["recorded_source_check"]["visited_history_nodes"] >= len(labels)
        runner.application_allowed = lambda ri, rule, valuation: ri != 0
        excluded = recorded_feedback_view(runner, rules, result, ctx["gamma"], candidate_mask=mask)
        assert not excluded.positive_facts.any() and result.positive_facts.all()
    finally:
        runner.close()


@pytest.mark.parametrize("defect", ["cycle", "missing"])
def test_recorded_feedback_rejects_incomplete_or_cyclic_history_without_chase(tmp_path, defect):
    ctx = context()
    rules = (local(Label("x", "A"), ML("x", "aux", "A", 0.5)),)
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        result = retaining_chase(runner, rules, ctx["gamma"])
        identifier = result.history_current["pos:0:0"]
        nodes = {key: _plain(node) for key, node in result.history_nodes.items()}
        node = nodes[identifier]
        node.update(
            origin="inclusion_label_copy",
            rule_id=None,
            dependencies=["pos:0:0"],
            history_dependencies=[identifier if defect == "cycle" else "missing"],
        )
        result.history_nodes = ProofDAG(nodes)
        view = recorded_feedback_view(runner, rules, result, ctx["gamma"])
        assert not view.positive_facts.any() and result.positive_facts[0, 0]
        reasons = view.reference_audit["recorded_source_check"]["rejected_reasons"]
        assert reasons["cycle" if defect == "cycle" else "missing_or_mismatched_history_parents"]
    finally:
        runner.close()


@pytest.mark.parametrize("execution", ["incremental", "full_recompute"])
def test_actual_firing_ledger_keeps_later_unknown_and_retained_relation_copies(tmp_path, execution):
    ctx = context(2)
    rules = (
        local(Label("x", "A"), at("x", 0), ML("x", "aux", "A", 0.5)),
        local(Label("x", "B"), Label("x", "A")),
        local(Label("x", "A", "neg"), Label("x", "B")),
        local(Label("x", "C"), Label("x", "B")),
        edge(0, 1, Label("x", "A")),
        local(Label("x", "A"), at("x", 0)),
    )
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        result = retaining_chase(
            runner, rules, ctx["gamma"], execution=execution, collect_rule_conclusions=True
        )
        assert result.rule_conclusions_enabled and result.conflict_unknown[:, 0].all()
        assert result.rule_conclusions[0]["pos"] == ((0, 0),)
        assert result.rule_conclusions[5]["pos"] == ((0, 0),)  # both direct supports count
        assert result.rule_conclusions[2]["neg"] == ((0, 0), (1, 0))
        assert result.rule_conclusions[4]["pos"] == ((1, 0), (1, 1), (1, 2))
        assert result.rule_conclusions[1]["pos"] == ((0, 1), (1, 1))
        assert all(result.rule_conclusion_instances[ri] >= 1 for ri in range(len(rules)))
        assert (
            sum(result.rule_conclusion_instances.values())
            == result.reference_audit["maintenance"]["fired_instances"]
        )
        with pytest.raises(ValueError, match="fresh invocation"):
            retaining_chase(
                runner,
                rules,
                ctx["gamma"],
                before=result,
                update_kind="check",
                collect_rule_conclusions=True,
            )
    finally:
        runner.close()


@pytest.mark.parametrize("op", ["eq", "subset", "strict_subset"])
def test_relation_calibration_ledger_transports_last_learned_source_through_gamma(tmp_path, op):
    ctx = context(3)
    gp = ctx["gamma"].pos.copy()
    gp[0, 0] = True
    gp[1, 1] = True
    ctx["gamma"] = RefGamma(gp, ctx["gamma"].neg, ((1, 2),))
    rules = (edge(0, 1, op=op),)
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        result = retaining_chase(runner, rules, ctx["gamma"], collect_rule_conclusions=True)
        assert result.status == "fixpoint"
        targets = set(result.rule_conclusions[0]["pos"])
        assert {(1, 0), (2, 0)} <= targets
        assert ((0, 1) in targets) is (op == "eq")
        assert len(result.rule_conclusions[0]["pos"]) == len(targets)
    finally:
        runner.close()


@pytest.mark.parametrize("op", ["eq", "subset", "strict_subset"])
def test_no_conflict_matches_independent_strict_executor(tmp_path, op):
    ctx = context(3)
    gp = ctx["gamma"].pos.copy()
    gp[0, 0] = True
    gp[1, 3] = True
    ctx["gamma"] = RefGamma(gp, ctx["gamma"].neg, ((1, 2),))
    rules = (
        edge(0, 1, op=op),
        local(Label("x", "B"), Label("x", "A")),
        local(Label("x", "C"), Label("x", "B")),
    )
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        strict = runner(rules, ctx["gamma"])
        retained = retaining_chase(runner, rules, ctx["gamma"])
        assert strict.status == retained.status == "fixpoint"
        np.testing.assert_array_equal(strict.positive_facts, retained.positive_facts)
        np.testing.assert_array_equal(strict.negative_facts, retained.negative_facts)
        assert set(strict.subset_facts) == set(retained.subset_facts)
        assert set(strict.strict_subset_facts) == set(retained.strict_subset_facts)
    finally:
        runner.close()


def test_gamma_pair_can_be_isolated_without_changing_persistent_storage(tmp_path):
    from loris.labeling.unknown_incremental import export_history

    ctx = context()
    gp = ctx["gamma"].pos.copy()
    gp[0, 0] = True
    ctx["gamma"] = RefGamma(gp, ctx["gamma"].neg)
    rules = (local(Label("x", "A", "neg")), local(Label("x", "B"), Label("x", "A")))
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        result = retaining_chase(runner, rules, ctx["gamma"])
        assert result.conflict_unknown[0, 0] and not result.positive_facts[0, 0]
        assert result.positive_facts[0, 1] and ctx["gamma"].pos[0, 0]
        assert not recorded_feedback_view(runner, rules, result, ctx["gamma"]).positive_facts.any()
        exported = export_history(result)
        assert exported["input_gamma_binding"]["positive"]["shape"] == [1, 4]
        assert "pos" not in exported["input_gamma_binding"]
        assert set(exported) >= {"nodes", "current", "latest", "retired", "temporary_seed_binding"}
    finally:
        runner.close()


def test_relation_ledger_uses_last_learned_rule_with_gamma_suffix(tmp_path):
    ctx = context(4)
    gp = ctx["gamma"].pos.copy()
    gp[0, 0] = True
    ctx["gamma"] = RefGamma(gp, ctx["gamma"].neg, ((2, 3),))
    rules = (edge(0, 1), edge(1, 2))
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        result = retaining_chase(runner, rules, ctx["gamma"], collect_rule_conclusions=True)
        assert result.rule_conclusions[0]["pos"] == ((1, 0),)
        assert result.rule_conclusions[1]["pos"] == ((2, 0), (3, 0))
        runner.application_allowed = lambda ri, rule, valuation: ri != 1
        restricted = retaining_chase(runner, rules, ctx["gamma"], collect_rule_conclusions=True)
        assert restricted.rule_conclusions[1]["pos"] == ()
        assert restricted.rule_conclusion_instances[1] == 0
        assert not restricted.positive_facts[2:, 0].any()
    finally:
        runner.close()


@pytest.mark.parametrize("execution", ["incremental", "full_recompute"])
@pytest.mark.parametrize("verified_b", [False, True])
def test_nonwitness_false_instance_withdraws_head_and_respects_gamma(
    tmp_path, execution, verified_b
):
    ctx = context()
    if verified_b:
        gp = ctx["gamma"].pos.copy()
        gp[0, 1] = True
        ctx["gamma"] = RefGamma(gp, ctx["gamma"].neg)
    rules = (
        local(Label("x", "A")),
        local(Label("x", "B"), Label("x", "A")),
        local(Label("x", "D")),
        local(Label("x", "C"), Label("x", "D")),
        local(Label("x", "A", "neg"), Label("x", "C")),
        local(Label("x", "B"), Label("x", "C"), ML("x", "aux", "B", 0.5)),
    )
    old = IndexedChaseRunner(ctx, tmp_path / "old")
    newctx = context(p=0.1)
    newctx["gamma"] = ctx["gamma"]
    new = IndexedChaseRunner(newctx, tmp_path / "new")
    try:
        before = retaining_chase(old, rules, ctx["gamma"], execution=execution)
        assert before.conflict_unknown[0, 0] and before.positive_facts[0, 1]
        assert before.proof_dag["pos:0:1"]["rule_id"] == (None if verified_b else 1)
        after = retaining_chase(
            new,
            rules,
            ctx["gamma"],
            before=before,
            old_runner=old,
            old_rules=rules,
            update_kind="chaseDelta",
            execution=execution,
            initial_unknown=np.zeros((1, 4), bool),
        )
        assert after.conflict_unknown[0, 0]
        # A and negative A now collide in one batch. With A isolated, its old
        # recorded B derivation cannot recover the head of the failed ML rule.
        assert bool(after.positive_facts[0, 1]) is verified_b
        assert after.positive_facts[0, 2:].all()
        assert bool(ctx["gamma"].pos[0, 1]) is verified_b
        assert after.reference_audit["maintenance"]["temporarily_withdrawn"] == (
            0 if verified_b else 1
        )
    finally:
        old.close()
        new.close()


@pytest.mark.parametrize("kind", ["chaseInc", "check"])
def test_same_model_updates_reject_different_frozen_context_but_accept_new_gamma(tmp_path, kind):
    ctx = context()
    rules = (local(Label("x", "A"), ML("x", "aux", "A", 0.5)),)
    old = IndexedChaseRunner(ctx, tmp_path / "old")
    wrong = IndexedChaseRunner(context(p=0.1), tmp_path / "wrong")
    gp = ctx["gamma"].pos.copy()
    gp[0, 1] = True
    gamma = RefGamma(gp, ctx["gamma"].neg)
    newctx = dict(ctx, gamma=gamma)
    same_model = IndexedChaseRunner(newctx, tmp_path / "same_model")
    try:
        before = retaining_chase(old, rules, ctx["gamma"])
        with pytest.raises(ValueError, match="frozen context"):
            retaining_chase(wrong, rules, gamma, before=before, update_kind=kind)
        assert old.context_id == same_model.context_id
        after = retaining_chase(same_model, rules, gamma, before=before, update_kind=kind)
        assert after.positive_facts[0, :2].all()
    finally:
        old.close()
        wrong.close()
        same_model.close()


def test_model_delta_rejects_mismatched_old_context_and_program(tmp_path):
    ctx = context()
    rules = (local(Label("x", "A"), ML("x", "aux", "A", 0.5)),)
    old = IndexedChaseRunner(ctx, tmp_path / "old")
    new = IndexedChaseRunner(context(p=0.1), tmp_path / "new")
    try:
        before = retaining_chase(old, rules, ctx["gamma"])
        with pytest.raises(ValueError, match="old context/program"):
            retaining_chase(
                new,
                rules,
                ctx["gamma"],
                before=before,
                old_runner=new,
                old_rules=rules,
                update_kind="chaseDelta",
            )
        different = (local(Label("x", "A"), ML("x", "aux", "A", 0.4)),)
        with pytest.raises(ValueError, match="old context/program"):
            retaining_chase(
                new,
                rules,
                ctx["gamma"],
                before=before,
                old_runner=old,
                old_rules=different,
                update_kind="chaseDelta",
            )
    finally:
        old.close()
        new.close()


@pytest.mark.parametrize("execution", ["incremental", "full_recompute"])
@pytest.mark.parametrize("replace_unknown_version", [False, True])
@pytest.mark.parametrize("verified_b", [False, True])
def test_false_nonwitness_unknown_head_covers_older_live_descendants(
    tmp_path, execution, replace_unknown_version, verified_b
):
    ctx = context()
    if verified_b:
        gp = ctx["gamma"].pos.copy()
        gp[0, 1] = True
        ctx["gamma"] = RefGamma(gp, ctx["gamma"].neg)
    rules = (
        local(Label("x", "A")),
        local(Label("x", "B"), Label("x", "A")),
        local(Label("x", "D")),
        local(Label("x", "C"), Label("x", "D")),
        local(Label("x", "A", "neg"), Label("x", "C")),
        local(Label("x", "A"), Label("x", "C"), ML("x", "aux", "A", 0.5)),
    )
    old = IndexedChaseRunner(ctx, tmp_path / "old")
    newctx = context(p=0.1)
    newctx["gamma"] = ctx["gamma"]
    new = IndexedChaseRunner(newctx, tmp_path / "new")
    seed = np.zeros((1, 4), bool)
    seed[0, 0] = True
    try:
        before = retaining_chase(old, rules, ctx["gamma"], execution=execution)
        oldest_a = before.history_latest["pos:0:0"]
        oldest_b = before.history_current["pos:0:1"]
        assert before.conflict_unknown[0, 0] and "pos:0:0" not in before.history_current
        assert before.history_nodes[oldest_a]["rule_id"] == 0
        if replace_unknown_version:
            before = retaining_chase(
                old,
                rules,
                ctx["gamma"],
                before=before,
                update_kind="chaseInc",
                initial_unknown=np.zeros((1, 4), bool),
                initial_positive=seed,
                execution=execution,
            )
            latest_a = before.history_latest["pos:0:0"]
            assert latest_a != oldest_a
            assert before.history_nodes[latest_a]["origin"] == "temporary_seed"
            assert before.history_current["pos:0:1"] == oldest_b
            if not verified_b:
                assert before.history_nodes[oldest_b]["history_dependencies"] == (oldest_a,)
        after = retaining_chase(
            new,
            rules,
            ctx["gamma"],
            before=before,
            old_runner=old,
            old_rules=rules,
            update_kind="chaseDelta",
            execution=execution,
            initial_unknown=np.zeros((1, 4), bool),
            initial_negative=seed,
        )
        assert after.conflict_unknown[0, 0]
        assert bool(after.positive_facts[0, 1]) is verified_b
        assert after.positive_facts[0, 2:].all()
        audit = after.reference_audit["maintenance"]
        assert audit["newly_false_old_valid_instance_heads"] == 1
        assert audit["failed_instance_head_history_versions"] == (
            2 if replace_unknown_version else 1
        )
        if not verified_b:
            assert oldest_a in after.history_retired and oldest_b in after.history_retired
        else:
            assert after.history_current["pos:0:1"] == oldest_b
    finally:
        old.close()
        new.close()
