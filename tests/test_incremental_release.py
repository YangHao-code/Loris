import numpy as np
import pytest

from loris.document import Document
from loris.predicates import MatchPredicate
from loris._engine.rdl_reference import Doc, Label, ML, Text, LabelCompare, Rule, RefGamma
from loris._engine.typed_runtime import IndexedChaseRunner
from loris.labeling.inference import chase_delta, chase_inc


def ctx(p):
    return dict(docs=[Document('red'), Document('blue')], labels=['A', 'B', 'C'],
        gamma=RefGamma.empty(2, 3), model_probabilities={'m': np.asarray(p, float)},
        model_decisions={})


def same(a, b):
    assert a.status == b.status
    if a.status == 'fixpoint':
        assert np.array_equal(a.positive_facts, b.positive_facts)
        assert np.array_equal(a.negative_facts, b.negative_facts)
        assert set(a.subset_facts) == set(b.subset_facts)
        assert set(a.strict_subset_facts) == set(b.strict_subset_facts)


@pytest.mark.parametrize('alternative', [False, True])
def test_withdraw_descendants_and_restore_unchanged_alternative_now(tmp_path, alternative):
    old, new = ctx([[.9,0,0],[0,0,0]]), ctx([[.1,0,0],[0,0,0]])
    rules = [Rule((Doc('x'), ML('x','m','A',.5)), Label('x','A')),
        Rule((Doc('x'), Label('x','A')), Label('x','B')),
        Rule((Doc('x'), Label('x','B')), Label('x','C')),
        Rule((Doc('x'), Label('x','C')), Label('x','B'))]
    if alternative:
        rules.append(Rule((Doc('x'), Text('x',MatchPredicate('cnt','red'))),Label('x','A')))
    a, b = IndexedChaseRunner(old,tmp_path/'a'), IndexedChaseRunner(new,tmp_path/'b')
    try:
        before = a(rules)
        after = chase_delta(a,b,rules,rules,before,old['gamma'])
        same(after, b(rules))
        assert after.positive_facts[0].all() == alternative
        audit = after.reference_audit['maintenance']
        assert audit['temporarily_withdrawn'] == 3
        assert audit['restored_same_update'] == (3 if alternative else 0)
        assert not after.reference_audit['full_chase_call_performed_for_this_update']
    finally:
        a.close(); b.close()


def test_verified_fact_promoted_before_delta_is_never_retracted(tmp_path):
    old, new = ctx([[.9,0,0],[0,0,0]]), ctx([[.1,0,0],[0,0,0]])
    rules = [Rule((Doc('x'),ML('x','m','A',.5)),Label('x','A')),
        Rule((Doc('x'),Label('x','A')),Label('x','B'))]
    a,b = IndexedChaseRunner(old,tmp_path/'a'),IndexedChaseRunner(new,tmp_path/'b')
    try:
        before=a(rules); p=old['gamma'].pos.copy();p[0,0]=True
        gamma=RefGamma(p,old['gamma'].neg)
        verified=chase_inc(a,rules,before,old['gamma'],gamma)
        assert verified.proof_dag['pos:0:0']['origin']=='validated_Gamma'
        after=chase_delta(a,b,rules,rules,verified,gamma)
        same(after,b(rules,gamma))
        assert after.positive_facts[0,0] and after.positive_facts[0,1]
    finally:a.close();b.close()


def test_relation_alternative_and_new_instance_and_negative(tmp_path):
    old,new=ctx([[.9,0,0],[0,0,0]]),ctx([[.1,.9,0],[0,0,0]])
    p=old['gamma'].pos.copy();p[0,2]=True
    old['gamma']=new['gamma']=RefGamma(p,old['gamma'].neg)
    rules=[Rule((Doc('x'),Doc('y'),ML('x','m','A',.5),
        Text('y',MatchPredicate('cnt','blue'))),LabelCompare('x','subset','y')),
        Rule((Doc('x'),Doc('y'),Text('x',MatchPredicate('cnt','red')),
        Text('y',MatchPredicate('cnt','blue'))),LabelCompare('x','subset','y')),
        Rule((Doc('x'),ML('x','m','B',.5)),Label('x','B','neg'))]
    a,b=IndexedChaseRunner(old,tmp_path/'a'),IndexedChaseRunner(new,tmp_path/'b')
    try:
        after=chase_delta(a,b,rules,rules,a(rules),old['gamma'])
        same(after,b(rules));assert after.positive_facts[1,2] and after.negative_facts[0,1]
    finally:a.close();b.close()


def test_inc_chained_insertions_and_conflict(tmp_path):
    c=ctx([[0,0,0],[0,0,0]])
    rules=[Rule((Doc('x'),Label('x','B')),Label('x','C')),
        Rule((Doc('x'),Label('x','A')),Label('x','B'))]
    r=IndexedChaseRunner(c,tmp_path/'r')
    try:
        before=r(rules);p=c['gamma'].pos.copy();p[1,0]=True
        g=RefGamma(p,c['gamma'].neg)
        after=chase_inc(r,rules,before,c['gamma'],g);same(after,r(rules,g))
        n=g.neg.copy();n[1,2]=True; bad=RefGamma(g.pos,n)
        same(chase_inc(r,rules,after,g,bad),r(rules,bad))
    finally:r.close()


def test_random_signed_program_deltas_match_full_closure(tmp_path):
    rng=np.random.RandomState(24)
    for trial in range(15):
        old,new=ctx(rng.rand(2,3)),ctx(rng.rand(2,3))
        rules=[Rule((Doc('x'),ML('x','m',label,.5)),Label('x',label)) for label in old['labels']]
        rules += [Rule((Doc('x'),Label('x',old['labels'][int(rng.randint(3))])),
                       Label('x',old['labels'][int(rng.randint(3))])) for _ in range(4)]
        a,b=IndexedChaseRunner(old,tmp_path/('a%d'%trial)),IndexedChaseRunner(new,tmp_path/('b%d'%trial))
        try:same(chase_delta(a,b,rules,rules,a(rules),old['gamma']),b(rules))
        finally:a.close();b.close()
