"""Relation conclusions propagate actual facts and RDG sees their copy events.

The finite reference is an independent full-closure oracle. Influence remains
an estimate in general; the exact counts below concern these explicit small
positive propagation chains, not a claim of equality for every rule program.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from loris.document import Document
from loris.predicates import MatchPredicate
from loris._engine.rdl_reference import (
    AttrEqual, Doc, Label, LabelCompare, RefGamma, ReferenceChase, Rule, Text,
)
from loris._engine.typed_runtime import IndexedChaseRunner
from loris._learning.rdg import RuleDependencyGraph
from loris.labeling.inference import chase_inc
from loris.rule_forms import validate_program_rule_forms


def make_context(n, labels):
    return dict(
        docs=[Document(cnt="document", ttl=str(i), mtd="same") for i in range(n)],
        labels=labels, gamma=RefGamma.empty(n, len(labels)),
        model_probabilities={}, model_decisions={},
    )


def at(var, row):
    return Text(var, MatchPredicate("ttl", "^%d$" % row))


def relation_rule(op, source=0, target=1, gate=False):
    body = (Doc("x"), Doc("y"), at("x", source), at("y", target),
            AttrEqual("x", "mtd", "y", "mtd"))
    if gate:
        body += (Label("x", "G"),)
    return Rule(body, LabelCompare("x", op, "y"))


def compare_oracle(context, rules, gamma, actual):
    expected = ReferenceChase(context["labels"]).run(context["docs"], rules, gamma)
    assert actual.status == expected.status
    if actual.status == "fixpoint":
        np.testing.assert_array_equal(actual.positive_facts, expected.positive_facts)
        np.testing.assert_array_equal(actual.negative_facts, expected.negative_facts)
        assert set(actual.subset_facts) == set(expected.subset_facts)
        assert set(actual.strict_subset_facts) == set(expected.strict_subset_facts)
        proof = actual.proof_dag
        done, visiting = set(), set()

        def visit(key):
            assert key in proof and key not in visiting
            if key in done:
                return
            visiting.add(key)
            for parent in proof[key].get("dependencies", ()):
                visit(parent)
            visiting.remove(key)
            done.add(key)

        for key in proof:
            visit(key)
    return expected


def save_evidence(path, result, **extra):
    path.write_text(json.dumps(dict(
        status=result.status,
        positive=result.positive_facts.astype(int).tolist(),
        negative=result.negative_facts.astype(int).tolist(),
        subset=sorted(list(edge) for edge in result.subset_facts),
        strict_subset=sorted(list(edge) for edge in result.strict_subset_facts),
        **extra,
    ), indent=2))


@pytest.mark.parametrize("op", ["subset", "eq", "strict_subset"])
@pytest.mark.parametrize("sign", ["pos", "neg"])
@pytest.mark.parametrize("seed_row", [0, 1])
@pytest.mark.parametrize("incremental", [False, True])
def test_relation_head_both_root_directions_and_explicit_signs(
    tmp_path, op, sign, seed_row, incremental,
):
    context = make_context(2, ["A", "seen", "strict_witness"])
    # The third rule form is legal without a single-label body premise.
    rules = [relation_rule(op), Rule((Doc("z"), Label("z", "A", sign)), Label("z", "seen"))]
    assert validate_program_rule_forms(rules)["invalid_count"] == 0
    old = context["gamma"]
    pos, neg = old.pos.copy(), old.neg.copy()
    (pos if sign == "pos" else neg)[seed_row, 0] = True
    gamma = RefGamma(pos, neg)
    runner = IndexedChaseRunner(context, tmp_path / "cache")
    try:
        before = runner(rules)
        result = (chase_inc(runner, rules, before, old, gamma) if incremental
                  else runner(rules, gamma))
        compare_oracle(context, rules, gamma, result)
        reached = {seed_row}
        if sign == "pos" and (op == "eq" or seed_row == 0):
            reached = {0, 1}
        assert set(np.flatnonzero((result.positive_facts if sign == "pos"
                                  else result.negative_facts)[:, 0])) == reached
        # Negative premises activate ordinary RDLs, but relation closure does
        # not invent negative copies or contrapose a positive inclusion.
        expected_seen = set(reached)
        if 0 in expected_seen:
            expected_seen.add(1)
        if op == "eq":
            expected_seen = {0, 1}
        assert set(np.flatnonzero(result.positive_facts[:, 1])) == expected_seen
        save_evidence(tmp_path / "evidence.json", result, operation=op,
                      sign=sign, seed_row=seed_row, incremental=incremental)
    finally:
        runner.close()


@pytest.mark.parametrize("edges", [((0, 1),), ((0, 1), (1, 2)),
                                   ((0, 1), (1, 0), (1, 2))])
def test_rdg_counts_builtin_positive_copy_from_existing_relations(tmp_path, edges):
    context = make_context(3, ["A"])
    old = context["gamma"]
    context["gamma"] = RefGamma(old.pos, old.neg, subset=edges)
    runner = IndexedChaseRunner(context, tmp_path / "cache")
    try:
        before = runner(())
        rdg = RuleDependencyGraph(())
        session = SimpleNamespace(runner=runner)
        scope = rdg.activation_scope(session, before, [("pos", 0, 0)])
        pos = old.pos.copy()
        pos[0, 0] = True
        gamma = RefGamma(pos, old.neg, subset=edges)
        actual = runner((), gamma)
        compare_oracle(context, (), gamma, actual)
        reached = np.flatnonzero(actual.positive_facts.any(1)).tolist()
        assert scope["support_documents"] == reached
        assert rdg.estimate(session, before, 0, 0, [0, 1, 2]) == len(reached)
        assert scope["rule_indices"] == [] and not before.positive_facts.any()
        negative = rdg.activation_scope(session, before, [("neg", 0, 0)])
        assert negative["support_documents"] == [0]
        save_evidence(tmp_path / "evidence.json", actual, scope=scope,
                      estimated_influence=len(reached), negative_scope=negative)
    finally:
        runner.close()


@pytest.mark.parametrize("op", ["subset", "eq", "strict_subset"])
@pytest.mark.parametrize("source,target", [(0, 1), (1, 0)])
def test_new_relation_head_copies_transitively_and_activates_label_consumers(
    tmp_path, op, source, target,
):
    context = make_context(4, ["A", "G", "B", "removed", "strict_witness"])
    old = context["gamma"]
    pos = old.pos.copy()
    pos[source, 0] = True
    context["gamma"] = old = RefGamma(pos, old.neg, subset=((target, 2),))
    rules = [
        relation_rule(op, source, target, gate=True),
        Rule((Doc("x"), Doc("y"), at("x", 2), at("y", 3),
              AttrEqual("x", "mtd", "y", "mtd"), Label("x", "A")), Label("y", "B")),
        Rule((Doc("x"), at("x", 2), Label("x", "A")), Label("x", "removed", "neg")),
    ]
    validate_program_rule_forms(rules)
    runner = IndexedChaseRunner(context, tmp_path / "cache")
    try:
        before = runner(rules)
        rdg = RuleDependencyGraph(rules)
        session = SimpleNamespace(runner=runner)
        scope = rdg.activation_scope(session, before, [("pos", source, 1)])
        # Target 2 has no explicit positive Label head: it is a built-in copy.
        assert scope["support_documents"] == [0, 1, 2, 3]
        assert scope["rule_indices"] == [0, 1, 2]
        pos = old.pos.copy()
        pos[source, 1] = True
        gamma = RefGamma(pos, old.neg, subset=old.subset)
        actual = chase_inc(runner, rules, before, old, gamma)
        compare_oracle(context, rules, gamma, actual)
        assert actual.positive_facts[target, 0] and actual.positive_facts[2, 0]
        assert actual.positive_facts[3, 2] and actual.negative_facts[2, 3]
        assert not actual.negative_facts[:2, 3].any()
        assert (source, 2) in set(actual.subset_facts)
        if op == "strict_subset":
            assert (source, 2) in set(actual.strict_subset_facts)
        save_evidence(tmp_path / "evidence.json", actual, scope=scope,
                      operation=op, source=source, target=target)
    finally:
        runner.close()


@pytest.mark.parametrize("op", ["subset", "eq", "strict_subset"])
def test_relation_events_trigger_comparison_body_consumers(tmp_path, op):
    context = make_context(3, ["G", "seen", "strict_witness"])
    old = context["gamma"]
    rules = [relation_rule(op, gate=True)]
    # Equality needs both directions; proper inclusion must also satisfy a
    # non-strict body. Reversing the equality body tests both event bindings.
    comparisons = [LabelCompare("x", "subset", "y")]
    if op == "eq":
        comparisons.append(LabelCompare("y", "eq", "x"))
    if op == "strict_subset":
        comparisons.append(LabelCompare("x", "strict_subset", "y"))
    for atom in comparisons:
        rules.append(Rule((Doc("x"), Doc("y"), at("x", 0), at("y", 1),
                           atom, Label("x", "G")), Label("y", "seen")))
    validate_program_rule_forms(rules)
    runner = IndexedChaseRunner(context, tmp_path / "cache")
    try:
        before = runner(rules)
        scope = RuleDependencyGraph(rules).activation_scope(
            SimpleNamespace(runner=runner), before, [("pos", 0, 0)])
        assert scope["rule_indices"] == list(range(len(rules)))
        pos = old.pos.copy()
        pos[0, 0] = True
        gamma = RefGamma(pos, old.neg)
        result = chase_inc(runner, rules, before, old, gamma)
        compare_oracle(context, rules, gamma, result)
        assert result.positive_facts[1, 1]
        save_evidence(tmp_path / "evidence.json", result, scope=scope, operation=op)
    finally:
        runner.close()


def test_relation_copy_conflict_is_reported_without_contrapositive_repair(tmp_path):
    context = make_context(2, ["A", "strict_witness"])
    pos, neg = context["gamma"].pos.copy(), context["gamma"].neg.copy()
    pos[0, 0], neg[1, 0] = True, True
    context["gamma"] = RefGamma(pos, neg)
    rule = relation_rule("subset")
    runner = IndexedChaseRunner(context, tmp_path / "cache")
    try:
        result = runner([rule])
        compare_oracle(context, [rule], context["gamma"], result)
        assert result.status == "conflict"
        assert result.positive_facts[1, 0] and result.negative_facts[1, 0]
        assert not result.negative_facts[0, 0]
        save_evidence(tmp_path / "evidence.json", result)
    finally:
        runner.close()
