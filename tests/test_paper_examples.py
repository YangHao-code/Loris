"""Literal paper Examples 2, 6 and 7 checked against the finite oracle.

These small documents exercise the promised semantics rather than merely
checking implementation metadata. The indexed executor and ReferenceChase
must agree on all explicit positive, negative and comparison facts.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from loris.document import Document
from loris.predicates import MatchPredicate, CooccurPredicate, BeforePredicate, FreqPredicate
from loris._engine.rdl_reference import (
    Doc, Label, ML, Text, AttrEqual, LabelCompare, Rule, RefGamma, ReferenceChase,
)
from loris._engine.typed_runtime import IndexedChaseRunner
from loris._learning.rdg import RuleDependencyGraph
from loris.labeling.inference import chase_inc


def oracle(context, rules, gamma=None):
    return ReferenceChase(context['labels'],
        model_probabilities=context.get('model_probabilities', {}),
        model_decisions=context.get('model_decisions', {}),
        static_model_probabilities=context.get('static_model_probabilities'),
        attribute_values=context.get('attribute_values'),
        attribute_presence=context.get('attribute_presence')).run(
            context['docs'], rules, context['gamma'] if gamma is None else gamma)


def assert_same_facts(actual, expected):
    assert actual.status == expected.status
    if actual.status == 'fixpoint':
        np.testing.assert_array_equal(actual.positive_facts, expected.positive_facts)
        np.testing.assert_array_equal(actual.negative_facts, expected.negative_facts)
        assert set(actual.subset_facts) == set(expected.subset_facts)
        assert set(actual.strict_subset_facts) == set(expected.strict_subset_facts)


def execute(context, rules, tmp_path):
    runner = IndexedChaseRunner(context, tmp_path/'indexed')
    try:
        result = runner(rules)
        assert_same_facts(result, oracle(context, rules))
        return result
    finally:
        runner.close()


def context(docs, labels, probabilities=None):
    return dict(docs=docs, labels=labels, gamma=RefGamma.empty(len(docs),len(labels)),
                model_probabilities=probabilities or {}, model_decisions={})


def test_example2_joint_complementary_labels_require_both_models_and_context(tmp_path):
    labels=['SE','IF']
    docs=[Document(cnt='please return the form',ttl='settle') for _ in range(3)]
    docs.append(Document(cnt='please return the form',ttl='unrelated'))
    ctx=context(docs,labels,dict(
        llm=np.array([[0,.9],[0,.1],[0,.9],[0,.9]]),
        bert=np.array([[.9,0],[.9,0],[.1,0],[.9,0]])))
    body=(Doc('x'),Text('x',MatchPredicate('ttl','settle')),
          Text('x',CooccurPredicate('cnt','return','form')),
          ML('x','llm','IF',.8),ML('x','bert','SE',.8))
    rules=[Rule(body,Label('x',label)) for label in labels]
    result=execute(ctx,rules,tmp_path)
    np.testing.assert_array_equal(result.positive_facts,[[1,1],[0,0],[0,0],[0,0]])
    assert not result.negative_facts.any()  # Unsupported is unknown, not false.


def test_example2_flight_and_date_guards_are_both_required(tmp_path):
    docs=[Document(text) for text in ('DL123 on 2026-09-22', 'DL123 without date',
                                    'date 2026-09-22 only','DL123 on 2026-09-22')]
    ctx=context(docs,['SE'],{'llm':np.array([[.9],[.9],[.9],[.1]])})
    rule=Rule((Doc('x'),Text('x',MatchPredicate('cnt',r'DL\d{1,4}')),
               Text('x',MatchPredicate('cnt',r'\d{4}-\d{2}-\d{2}')),
               ML('x','llm','SE',.8)),Label('x','SE'))
    np.testing.assert_array_equal(execute(ctx,[rule],tmp_path).positive_facts[:,0],[1,0,0,0])


def test_example2_order_frequency_and_model_are_conjoined(tmp_path):
    docs=[Document(text) for text in (
        'refuse cancellation 14 days before depart; notice 14 days',
        'refuse depart 14 days before cancellation; notice 14 days',
        'refuse cancellation 14 days before depart',
        'refuse cancellation 14 days before depart; notice 14 days')]
    ctx=context(docs,['DE'],{'bert':np.array([[.9],[.9],[.9],[.1]])})
    rule=Rule((Doc('x'),Text('x',MatchPredicate('cnt','refuse')),
               Text('x',BeforePredicate('cnt','cancellation','depart')),
               Text('x',FreqPredicate('cnt','14 days','>=',2)),
               ML('x','bert','DE',.8)),Label('x','DE'))
    np.testing.assert_array_equal(execute(ctx,[rule],tmp_path).positive_facts[:,0],[1,0,0,0])


@pytest.mark.parametrize('reverse_order',[False,True])
def test_example2_rejects_model_paid_label_and_adds_denial_without_overwriting_gamma(tmp_path,reverse_order):
    ctx=context([Document('already paid; we must decline')],['PD','DE'],
                {'rule_model':np.array([[.99,0.]])})
    body=(Doc('x'),ML('x','rule_model','PD',.5),
          Text('x',CooccurPredicate('cnt','already paid','decline')))
    rules=[Rule(body,Label('x','PD','neg')),Rule(body,Label('x','DE'))]
    if reverse_order:rules.reverse()
    result=execute(ctx,rules,tmp_path/'clean')
    np.testing.assert_array_equal(result.positive_facts,[[0,1]])
    np.testing.assert_array_equal(result.negative_facts,[[1,0]])
    positive=ctx['gamma'].pos.copy();positive[0,0]=True
    ctx['gamma']=RefGamma(positive,ctx['gamma'].neg)
    conflict=execute(ctx,rules,tmp_path/'conflict')
    assert conflict.status=='conflict'
    assert ctx['gamma'].pos[0,0]  # A validated root cannot be silently withdrawn.


def test_unknown_is_not_an_explicit_negative_rule_premise(tmp_path):
    ctx=context([Document('unknown'),Document('explicit negative')],['PD','DE'])
    negative=ctx['gamma'].neg.copy();negative[1,0]=True
    ctx['gamma']=RefGamma(ctx['gamma'].pos,negative)
    rule=Rule((Doc('x'),Label('x','PD','neg')),Label('x','DE'))
    result=execute(ctx,[rule],tmp_path)
    np.testing.assert_array_equal(result.positive_facts,[[0,0],[0,1]])
    np.testing.assert_array_equal(result.negative_facts,[[0,0],[1,0]])


def test_fixed_model_still_compares_its_kappa_and_keeps_original_probabilities(tmp_path):
    ctx=context([Document('same fixed model, different thresholds')],['SE','IF','DE'],
                {'m':np.array([[.95,0.,0.]])})
    ctx['static_model_probabilities']={'m':np.array([[.7,0.,0.]])}
    ctx['model_decisions']={'m':np.ones((1,3),bool)}  # Must not override kappa.
    rules=[Rule((Doc('x'),ML('x','m','SE',.5,dynamic=False)),Label('x','SE')),
           Rule((Doc('x'),ML('x','m','SE',.8,dynamic=False)),Label('x','IF')),
           Rule((Doc('x'),ML('x','m','SE',.8,dynamic=True)),Label('x','DE'))]
    result=execute(ctx,rules,tmp_path)
    np.testing.assert_array_equal(result.positive_facts,[[1,0,1]])


def test_example6_label_equality_propagates_verified_labels_through_relation_closure(tmp_path):
    docs=[Document('hurricane',mtd='incident1'),Document('hurricane',mtd='incident1'),
          Document('hurricane',mtd='incident2')]
    ctx=context(docs,['SE'])
    positive=ctx['gamma'].pos.copy();positive[0,0]=True
    ctx['gamma']=RefGamma(positive,ctx['gamma'].neg)
    rule=Rule((Doc('x'),Doc('y'),Text('y',MatchPredicate('cnt','hurricane')),
               AttrEqual('x','mtd','y','mtd')),LabelCompare('x','eq','y'))
    result=execute(ctx,[rule],tmp_path)
    np.testing.assert_array_equal(result.positive_facts[:,0],[1,1,0])
    assert (0,1) in set(result.subset_facts) and (1,0) in set(result.subset_facts)


def test_example7_influence_counts_self_one_and_shared_flight_two(tmp_path):
    docs=[Document('T1',mtd='independent'),Document('T2',mtd='flight2'),
          Document('T2-prime',mtd='flight2')]
    ctx=context(docs,['SE'])
    rules=[Rule((Doc('x'),Doc('y'),AttrEqual('x','mtd','y','mtd'),Label('x','SE')),
                Label('y','SE'))]
    runner=IndexedChaseRunner(ctx,tmp_path/'rdg')
    try:
        before=runner(rules)
        assert_same_facts(before,oracle(ctx,rules))
        rdg=RuleDependencyGraph(rules);session=SimpleNamespace(runner=runner)
        estimates=[rdg.estimate(session,before,row,0,[0,1,2]) for row in range(3)]
        assert estimates==[1,2,2]
        for row,expected in enumerate(estimates):
            positive=ctx['gamma'].pos.copy();positive[row,0]=True
            gamma=RefGamma(positive,ctx['gamma'].neg)
            after=chase_inc(runner,rules,before,ctx['gamma'],gamma)
            exact=oracle(ctx,rules,gamma)
            assert_same_facts(after,exact)
            assert int(exact.positive_facts.any(1).sum())==expected
        assert not before.positive_facts.any()  # Hypothetical selection is read-only.
    finally:
        runner.close()
