"""Soft Gamma conflicts (2026-09-30): validated Gamma wins over derived facts."""
import numpy as np
import pytest

from loris.document import Document
from loris.predicates import MatchPredicate
from loris._engine.rdl_reference import (Doc, Text, Label, AttrEqual, LabelCompare, Rule, RefGamma)
from loris._engine.rlearner import TwoFilterEvaluator
from loris._storage.selection_execution import make_selection_runner
from loris._storage.discovery_factory import make_discovery_runner

LABELS = ['A', 'B']


def context():
    # rows 0-3 are fit rows with complete Gamma; rows 4-5 are unknown.
    texts = ['alpha', 'alpha', 'alpha', 'beta', 'alpha', 'beta']
    n, k = len(texts), len(LABELS)
    pos = np.zeros((n, k), bool)
    neg = np.zeros((n, k), bool)
    pos[0, 0] = pos[1, 0] = True          # alpha rows 0,1 are A
    neg[2, 0] = neg[3, 0] = True          # alpha row 2 is not A
    neg[0:4, 1] = True
    return dict(docs=[Document(cnt=t, ttl='t%d' % i, mtd='g%d' % (i // 2)) for i, t in enumerate(texts)],
                labels=list(LABELS), gamma=RefGamma(pos, neg), model_probabilities={}, model_decisions={})


def alpha_to_A():
    return Rule((Doc('x'), Text('x', MatchPredicate('cnt', 'alpha'))), Label('x', 'A'))


@pytest.mark.parametrize('factory', [make_selection_runner, make_discovery_runner])
def test_strict_conflict_versus_soft_blocking(tmp_path, factory):
    ctx = context()
    strict = factory(ctx, tmp_path / 'strict')
    soft = factory(ctx, tmp_path / 'soft', gamma_protection=True)
    try:
        assert strict(( alpha_to_A(),), gamma=ctx['gamma']).status == 'conflict'
        result = soft((alpha_to_A(),), gamma=ctx['gamma'])
        assert result.status == 'fixpoint'
        # Gamma negative at row 2 is kept; the unknown alpha row 4 is derived.
        assert not result.positive_facts[2, 0] and result.negative_facts[2, 0]
        assert result.positive_facts[4, 0] and result.positive_facts[:, 0].tolist() == [1, 1, 0, 0, 1, 0]
        assert np.array_equal(result.negative_facts, ctx['gamma'].neg)
        assert strict.context_id != soft.context_id
        if factory is make_selection_runner:
            assert result.gamma_protection == {'blocked': 1, 'agreements': 2}
    finally:
        strict.close()
        soft.close()


def test_selection_and_proof_runners_agree_under_protection(tmp_path):
    ctx = context()
    rules = (alpha_to_A(),
             Rule((Doc('x'), Doc('y'), Label('x', 'A'), AttrEqual('x', 'mtd', 'y', 'mtd')), Label('y', 'A')))
    selection = make_selection_runner(ctx, tmp_path / 's', gamma_protection=True)
    proof = make_discovery_runner(ctx, tmp_path / 'p', gamma_protection=True)
    try:
        a, b = selection(rules, gamma=ctx['gamma']), proof(rules, gamma=ctx['gamma'])
        assert a.status == b.status == 'fixpoint'
        assert np.array_equal(a.positive_facts, b.positive_facts)
        assert np.array_equal(a.negative_facts, b.negative_facts)
        # Propagation onto the Gamma-negative row 3 (same mtd group as row 2) is blocked too.
        assert not a.positive_facts[3, 0] and a.positive_facts[5, 0]
    finally:
        selection.close()
        proof.close()


def evaluator(runner, floor):
    ctx = context()
    return TwoFilterEvaluator(runner, labels=LABELS, receiver_indices=[4, 5],
                              truth=np.array([[1, 0], [0, 0]], bool), gamma=ctx['gamma'],
                              context_id='t', line7_mode='positive_macro_f1',
                              head_info=lambda r: (r.head.label, r.head.polarity == 'neg'),
                              fit_precision_floor=floor)


def test_fit_precision_floor_gates_standalone_and_admission(tmp_path):
    ctx = context()
    runner = make_selection_runner(ctx, tmp_path / 'r', gamma_protection=True)
    try:
        # precision 2/3 on validated fit cells
        low = evaluator(runner, 0.9)
        assert low.standalone(alpha_to_A())['score'] == -1.
        assert low.standalone(alpha_to_A())['metric'] == 'fit_precision_below_floor'
        decision = low.admit((), alpha_to_A(), epsilon=0.0)
        assert not decision['accepted'] and decision['reason'] == 'fit_precision_below_floor'
        high = evaluator(runner, 0.6)
        assert high.standalone(alpha_to_A())['score'] > 0
        assert high.admit((), alpha_to_A(), epsilon=0.0)['accepted']
    finally:
        runner.close()


def test_precision_floor_requires_protection(tmp_path):
    runner = make_selection_runner(context(), tmp_path / 'r')
    try:
        with pytest.raises(ValueError):
            evaluator(runner, 0.9)
        assert evaluator(runner, None).standalone(alpha_to_A())['score'] == -1.
    finally:
        runner.close()
