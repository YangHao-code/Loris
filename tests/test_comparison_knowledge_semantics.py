"""Partial label sets never create cross-document comparison premises."""
import numpy as np
import pytest

from loris.document import Document
from loris.predicates import LabelPredicate, predicate_from_dict, predicate_to_dict
from loris._engine.rdl_reference import Doc, Label, LabelCompare, RefGamma, Rule
from loris._engine.typed_runtime import IndexedChaseRunner
from loris.labeling.inference import chase_inc
from loris.rule_forms import validate_program_rule_forms
from tests.test_relation_propagation_rdg import (
    at, compare_oracle, make_context, relation_rule,
)


def guarded_consumer(op):
    # The independent positive anchor makes the cross rule legal without
    # supplying either of the two compared documents with a label.
    return Rule((Doc("x"), Doc("y"), Doc("z"), at("x", 0), at("y", 1), at("z", 2),
                 Label("z", "A"), LabelCompare("x", op, "y")), Label("y", "B"))


@pytest.mark.parametrize("op", ["eq", "subset", "strict_subset"])
@pytest.mark.parametrize("observations", ["unknown", "same_positive", "contained_positive", "known_empty"])
def test_coincidental_partial_or_complete_sets_do_not_establish_relations(tmp_path, op, observations):
    context = make_context(3, ["A", "B", "spare"])
    pos, neg = np.zeros((3, 3), bool), np.zeros((3, 3), bool)
    pos[2, 0] = True
    if observations == "same_positive":
        pos[:2, 0] = True
    elif observations == "contained_positive":
        pos[1, 0] = True
    elif observations == "known_empty":
        neg[:2] = True
    gamma = context["gamma"] = RefGamma(pos, neg)
    rules = (guarded_consumer(op),)
    assert validate_program_rule_forms(rules)["invalid_count"] == 0
    runner = IndexedChaseRunner(context, tmp_path / "cache")
    try:
        result = runner(rules)
        compare_oracle(context, rules, gamma, result)
        assert result.status == "fixpoint" and not result.positive_facts[:, 1].any()
        assert set(result.subset_facts) == {(0, 0), (1, 1), (2, 2)}
        assert not len(result.strict_subset_facts)
    finally:
        runner.close()


@pytest.mark.parametrize("op", ["eq", "subset", "strict_subset"])
def test_verified_relation_roots_support_premises_even_before_any_positive_at_endpoints(tmp_path, op):
    context = make_context(3, ["A", "B", "spare"])
    pos = np.zeros((3, 3), bool)
    pos[2, 0] = True
    edges = ((0, 1), (1, 0)) if op == "eq" else ((0, 1),)
    gamma = context["gamma"] = RefGamma(pos, np.zeros_like(pos), subset=edges,
                                        strict_subset=((0, 1),) if op == "strict_subset" else ())
    rules = (guarded_consumer(op),)
    runner = IndexedChaseRunner(context, tmp_path / "cache")
    try:
        result = runner(rules)
        compare_oracle(context, rules, gamma, result)
        assert result.status == "fixpoint" and result.positive_facts[1, 1]
        assert bool(result.positive_facts[0, 1]) is (op == "eq")
    finally:
        runner.close()


@pytest.mark.parametrize("op", ["eq", "subset", "strict_subset"])
@pytest.mark.parametrize("reverse_rules", [False, True])
def test_relation_head_on_unknown_documents_propagates_later_labels_in_same_update(tmp_path, op, reverse_rules):
    context = make_context(2, ["A", "B", "spare"])
    rules = [relation_rule(op), Rule((Doc("z"), at("z", 1), Label("z", "A")), Label("z", "B"))]
    if reverse_rules:
        rules.reverse()
    assert validate_program_rule_forms(rules)["invalid_count"] == 0
    old = context["gamma"]
    runner = IndexedChaseRunner(context, tmp_path / "cache")
    try:
        before = runner(rules)
        compare_oracle(context, rules, old, before)
        assert not before.positive_facts.any() and (0, 1) in before.subset_facts
        pos = old.pos.copy()
        pos[0, 0] = True
        gamma = RefGamma(pos, old.neg)
        after = chase_inc(runner, rules, before, old, gamma)
        compare_oracle(context, rules, gamma, after)
        assert after.positive_facts[:, 0].all() and after.positive_facts[1, 1]
        assert bool(after.positive_facts[0, 1]) is (op == "eq")
    finally:
        runner.close()


@pytest.mark.parametrize("op", ["eq", "subset", "strict_subset"])
def test_same_document_reflexivity_is_identity_not_an_empty_set_comparison(tmp_path, op):
    context = make_context(2, ["A"])
    rules = (Rule((Doc("x"), LabelCompare("x", op, "x")), Label("x", "A")),)
    assert validate_program_rule_forms(rules)["invalid_count"] == 0
    runner = IndexedChaseRunner(context, tmp_path / "cache")
    try:
        result = runner(rules)
        compare_oracle(context, rules, context["gamma"], result)
        assert bool(result.positive_facts.all()) is (op != "strict_subset")
    finally:
        runner.close()


@pytest.mark.parametrize("op", ["eq", "subset", "strict_subset"])
@pytest.mark.parametrize("literal", ["", "A", "A,B"])
def test_public_literal_comparison_cannot_bypass_the_knowledge_state(op, literal):
    with pytest.raises(ValueError, match="explicit relation facts"):
        LabelPredicate(literal, op)
    with pytest.raises(ValueError, match="explicit relation facts"):
        predicate_from_dict(dict(type="LabelPredicate", label=literal, op=op))
    legacy = LabelPredicate(literal)
    object.__setattr__(legacy, "op", op)  # Simulate an older pickle constructor bypass.
    with pytest.raises(ValueError, match="LabelCompare"):
        legacy(Document(cnt="text"))


def test_public_membership_remains_a_pure_read_and_round_trips():
    doc = Document(cnt="text", lbl={"A"})
    predicate = predicate_from_dict(predicate_to_dict(LabelPredicate("A")))
    assert predicate(doc) and not LabelPredicate("B")(doc)
    assert doc.lbl == {"A"}
