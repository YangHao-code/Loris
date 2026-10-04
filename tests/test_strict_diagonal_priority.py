"""Small real semantics checks for the isolated runtime005 execution order."""
import numpy as np
import pytest

from loris.document import Document
from loris.predicates import MatchPredicate
from loris._engine.rdl_reference import (
    Doc, Text, ML, Label, AttrEqual, LabelCompare, Rule, RefGamma, ReferenceChase)
from loris._engine.typed_runtime import IndexedChaseRunner
from loris._engine.instance_scope import FrozenInstanceScope, scope_corpus_identity
from loris._engine.array_cache import sha
from loris._storage.selection_execution import make_selection_runner
from loris._storage.discovery_factory import make_discovery_runner


def context():
    n, k = 4, 3
    p = np.zeros((n, k), np.float64)
    p[2] = .9
    pos = np.zeros((n, k), bool)
    neg = pos.copy()
    return dict(docs=[Document('row' + str(i)) for i in range(n)],
        labels=['A', 'B', 'C'], gamma=RefGamma(pos, neg),
        model_probabilities={'M': p},
        attribute_values={'mtd.g': ['same'] * n,
                          'mtd.a': ['a'] * n, 'mtd.b': ['b'] * n})


def reference(ctx, rules):
    return ReferenceChase(ctx['labels'], model_probabilities=ctx['model_probabilities'],
        attribute_values=ctx['attribute_values']).run(ctx['docs'], rules, ctx['gamma'])


def same_success(a, b):
    assert a.status == b.status == 'fixpoint'
    for field in ('positive_facts', 'negative_facts', 'predictions'):
        assert np.array_equal(getattr(a, field), getattr(b, field))
    for field in ('subset_facts', 'strict_subset_facts'):
        assert set(getattr(a, field)) == set(getattr(b, field))


def no_milp(*args, **kwargs):
    raise AssertionError('strict diagonal must fail without MILP')


def test_real_diagonal_proof_and_selection_fail_without_milp(tmp_path, monkeypatch):
    ctx = context()
    pos, neg = ctx['gamma'].pos.copy(), ctx['gamma'].neg.copy()
    pos[2, 0] = True
    neg[2, 1] = True
    ctx['gamma'] = RefGamma(pos, neg)
    body = (Doc('x'), Doc('y'), Text('x', MatchPredicate('cnt', '^row')),
        ML('y', 'M', 'A', .8, dynamic=True),
        ML('y', 'M', 'B', .8, dynamic=True),
        ML('y', 'M', 'C', .8, dynamic=True),
        Label('y', 'A'), Label('x', 'B', 'neg'),
        AttrEqual('x', 'mtd.g', 'y', 'mtd.g'), LabelCompare('x', 'subset', 'y'))
    rule = Rule(body, LabelCompare('y', 'strict_subset', 'x'))
    assert reference(ctx, [rule]).status == 'conflict'
    monkeypatch.setattr('scipy.optimize.milp', no_milp)
    for name, factory in [('typed', IndexedChaseRunner),
                          ('proof', make_discovery_runner), ('score', make_selection_runner)]:
        runner = factory(ctx, tmp_path / name)
        try:
            result = runner([rule])
            assert result.status == 'conflict'
            assert result.reference_audit['finite_comparison_solver_calls'] == 0
            assert result.comparison_conflicts == [dict(kind='strict_inclusion_cycle', source=2, target=2)]
            assert runner.stats['strict_diagonal_body_witnesses'] == 1
            assert set(result.strict_subset_facts) == {(2, 2)}
            if name != 'score':
                proof = result.proof_dag['strict_subset:2:2']
                assert proof['origin'] == 'rule_application'
                assert dict(proof['valuation']) == {'x': 2, 'y': 2}
                assert set(proof['dependencies']) == {'pos:2:0', 'neg:2:1', 'subset:2:2'}
                assert len(proof['static_evidence']) == 5
                for dependency in proof['dependencies']:
                    assert dependency in result.proof_dag
            else:
                assert not hasattr(result, 'proof_dag')
        finally:
            runner.close()


@pytest.mark.parametrize('case', ['ml', 'text', 'positive_label', 'negative_label',
    'missing_attribute', 'different_attribute', 'strict_relation', 'third_variable'])
def test_incomplete_body_cannot_create_false_diagonal(tmp_path, case):
    ctx = context()
    body = [Doc('x'), Doc('y'), ML('y', 'M', 'A', .8, dynamic=True),
        AttrEqual('x', 'mtd.g', 'y', 'mtd.g')]
    if case == 'ml': body.append(ML('x', 'M', 'A', .95, dynamic=True))
    if case == 'text': body.append(Text('x', MatchPredicate('cnt', '^absent$')))
    if case == 'positive_label': body.append(Label('x', 'A'))
    if case == 'negative_label': body.append(Label('x', 'A', 'neg'))
    if case == 'missing_attribute': ctx['attribute_values']['mtd.g'] = [None] * 4
    if case == 'different_attribute': body.append(AttrEqual('x', 'mtd.a', 'y', 'mtd.b'))
    if case == 'strict_relation': body.append(LabelCompare('x', 'strict_subset', 'y'))
    if case == 'third_variable': body.extend([Doc('z'), Label('z', 'A')])
    rule = Rule(tuple(body), LabelCompare('y', 'strict_subset', 'x'))
    expected = reference(ctx, [rule])
    for name, factory in [('proof', make_discovery_runner), ('score', make_selection_runner)]:
        runner = factory(ctx, tmp_path / name)
        try:
            same_success(expected, runner([rule]))
            assert runner.stats['strict_diagonal_body_witnesses'] == 0
        finally:
            runner.close()


@pytest.mark.parametrize('op', ['subset', 'eq', 'strict_subset'])
def test_successful_non_diagonal_full_fixedpoint_matches_reference(tmp_path, op):
    ctx = context()
    pos = ctx['gamma'].pos.copy()
    pos[2, 0] = True
    ctx['gamma'] = RefGamma(pos, ctx['gamma'].neg)
    relation = Rule((Doc('x'), Doc('y'), Text('x', MatchPredicate('cnt', '^row0$')),
        ML('y', 'M', 'A', .8, dynamic=True), AttrEqual('x', 'mtd.g', 'y', 'mtd.g')),
        LabelCompare('y', op, 'x'))
    # Put the dependent rule first to require an additional fixed-point round.
    rules = [Rule((Doc('x'), Label('x', 'A')), Label('x', 'B')), relation]
    expected = reference(ctx, rules)
    for name, factory in [('proof', make_discovery_runner), ('score', make_selection_runner)]:
        runner = factory(ctx, tmp_path / name)
        try:
            actual = runner(rules)
            same_success(expected, actual)
            assert actual.positive_facts[0, 0] and actual.positive_facts[0, 1]
            assert runner.stats['strict_diagonal_body_witnesses'] == 0
        finally:
            runner.close()


def test_scope_disabled_self_instance_uses_original_authorized_join(tmp_path):
    ctx = context()
    rule = Rule((Doc('x'), Doc('y'),
        ML('x', 'M', 'A', .8, dynamic=True), ML('y', 'M', 'A', .8, dynamic=True)),
        LabelCompare('y', 'strict_subset', 'x'))
    decision = dict(status='approved', action='disable_rule_instance', reason='fixture',
        approval_receipt_id='fixture', reviewer='fixture',
        authority_scope='exact_instance_same_corpus_program', parent_conflict_id='fixture',
        corpus_id=scope_corpus_identity(ctx), program_id=sha([rule.to_dict()]),
        rule_index=0, rule_sha256=sha(rule.to_dict()), valuation={'x': 2, 'y': 2},
        provider='development_truth_review', development_only=True)
    scope = FrozenInstanceScope(ctx, [rule], [decision])
    runner = make_discovery_runner(ctx, tmp_path / 'scoped', application_scope=scope)
    try:
        result = runner([rule])
        assert result.status == 'fixpoint' and not len(result.strict_subset_facts)
        assert runner.stats['strict_diagonal_body_witnesses'] == 0
        assert runner.stats['approved_rule_instances_skipped'] == 1
    finally:
        runner.close()


def test_newly_derived_label_can_activate_diagonal_on_next_round(tmp_path):
    ctx = context()
    strict = Rule((Doc('x'), Doc('y'), Label('x', 'A'), Label('y', 'A')),
        LabelCompare('x', 'strict_subset', 'y'))
    producer = Rule((Doc('x'), ML('x', 'M', 'A', .8, dynamic=True)), Label('x', 'A'))
    rules = [strict, producer]
    assert reference(ctx, rules).status == 'conflict'
    runner = make_discovery_runner(ctx, tmp_path / 'derived')
    try:
        result = runner(rules)
        assert result.status == 'conflict' and result.n_rounds == 2
        assert result.proof_dag['strict_subset:2:2']['dependencies'] == ('pos:2:0', 'pos:2:0')
        assert result.proof_dag['pos:2:0']['origin'] == 'rule_application'
    finally:
        runner.close()
