"""Exact unary block execution: protected roots, partial conflicts and fallbacks."""
import numpy as np
import pytest
from loris.chase import GammaSeed
from loris.document import Document
from loris.predicates import MatchPredicate
from loris._engine.rdl_reference import Doc, Label, LabelCompare, AttrEqual, ML, Text, RefGamma, Rule
from loris._storage.selection_execution import make_selection_runner


def equal(a,b,left,right):
    assert a.status==b.status
    for key in ('positive_facts','negative_facts','predictions'):
        np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    assert a.n_rounds==b.n_rounds
    assert a.conflicts==b.conflicts
    assert a.comparison_conflicts==b.comparison_conflicts
    assert a.gamma_protection==b.gamma_protection
    for key in ('subset_facts','strict_subset_facts'):
        assert set(getattr(a,key))==set(getattr(b,key))
    for key in ('matched_valuations','indexed_prefixes','strict_diagonal_body_witnesses'):
        assert left.audit()['logical_stats'].get(key,0)==right.audit()['logical_stats'].get(key,0)


@pytest.mark.parametrize('protected',[False,True])
@pytest.mark.parametrize('prefix',[False,True])
def test_random_unary_programs_preserve_facts_conflicts_counts_rounds(tmp_path,protected,prefix):
    for seed in range(50):
        rng=np.random.RandomState(seed);n,k=23,5
        pos=rng.rand(n,k)<.22;neg=(rng.rand(n,k)<.25)&~pos
        ctx=dict(docs=[Document('a' if i%3 else 'b') for i in range(n)],labels=list('ABCDE'),
                 gamma=GammaSeed(pos,neg),model_probabilities={'M':rng.rand(n,k)})
        runners=[make_selection_runner(ctx,tmp_path/(str(seed)+'_'+str(flag)),
                 gamma_protection=protected,prefix_reuse=prefix,unary_label_batching=flag) for flag in (False,True)]
        rules=[]
        try:
            for step in range(6):
                body=[Doc('x')]
                if step%3!=0:body.append(Label('x',chr(65+rng.randint(k)),'neg' if step%2 else 'pos'))
                if step%2:body.append(ML('x','M',chr(65+rng.randint(k)),.4,dynamic=True))
                if step%4==0:body.append(Text('x',MatchPredicate('cnt','a')))
                rules.append(Rule(tuple(body),Label('x',chr(65+rng.randint(k)),'neg' if rng.rand()<.4 else 'pos')))
                a,b=[r(rules) for r in runners];equal(a,b,*runners)
        finally:
            for runner in runners:runner.close()


def test_first_conflict_keeps_earlier_blocked_cells_and_partial_labels(tmp_path):
    pos=np.zeros((8,2),bool);neg=pos.copy();neg[1,0]=True
    ctx=dict(docs=[Document(str(i)) for i in range(8)],labels=['A','B'],gamma=GammaSeed(pos,neg))
    # Rule 1 adds derived negative A at row 4. Rule 2 first adds A at row 0,
    # blocks row 1 by Gamma, adds rows 2/3, then conflicts at row 4.
    rules=[Rule((Doc('x'),Text('x',MatchPredicate('cnt','4'))),Label('x','A','neg')),
           Rule((Doc('x'),),Label('x','A'))]
    runners=[make_selection_runner(ctx,tmp_path/str(flag),gamma_protection=True,
        unary_label_batching=flag) for flag in (False,True)]
    try:
        a,b=[r(rules) for r in runners];equal(a,b,*runners)
        assert a.status=='conflict' and a.positive_facts[:,0].tolist()==[True,False,True,True,True,False,False,False]
    finally:
        for runner in runners:runner.close()


@pytest.mark.parametrize('edge_kind',['subset','strict_subset'])
def test_existing_relations_retain_scalar_propagation_and_feasibility(tmp_path,edge_kind):
    pos=np.zeros((5,3),bool)
    gamma=RefGamma(pos,pos.copy(),**{edge_kind:((0,1),)})
    ctx=dict(docs=[Document(str(i)) for i in range(5)],labels=list('ABC'),gamma=gamma)
    rules=[Rule((Doc('x'),Text('x',MatchPredicate('cnt','0'))),Label('x','A'))]
    runners=[make_selection_runner(ctx,tmp_path/str(flag),unary_label_batching=flag) for flag in (False,True)]
    try:
        a,b=[r(rules) for r in runners];equal(a,b,*runners)
        assert a.positive_facts[1,0]
        assert runners[1].stats['unary_label_blocks']==0
    finally:
        for runner in runners:runner.close()


def test_group_rule_after_batched_unary_uses_exact_updated_row_bits(tmp_path):
    pos=np.zeros((12,4),bool);neg=pos.copy();neg[2,0]=True
    ctx=dict(docs=[Document(str(i)) for i in range(12)],labels=list('ABCD'),gamma=GammaSeed(pos,neg),
        attribute_values={'mtd.group':['a']*6+['b']*6})
    rules=[Rule((Doc('x'),Text('x',MatchPredicate('cnt','0|7'))),Label('x','A')),
        Rule((Doc('x'),Doc('y'),AttrEqual('x','mtd.group','y','mtd.group'),Label('x','A')),LabelCompare('x','eq','y')),
        Rule((Doc('x'),Label('x','A')),Label('x','B'))]
    runners=[make_selection_runner(ctx,tmp_path/str(flag),gamma_protection=True,
        unary_label_batching=flag) for flag in (False,True)]
    try:
        a,b=[r(rules) for r in runners];equal(a,b,*runners)
        assert a.positive_facts[:,0].sum()==11
        assert runners[1].stats['unary_label_blocks']>0
    finally:
        for runner in runners:runner.close()


def test_execution_binding_separates_unary_policy(tmp_path):
    pos=np.zeros((2,1),bool);ctx=dict(docs=[Document('x')]*2,labels=['A'],gamma=GammaSeed(pos,pos.copy()))
    runners=[make_selection_runner(ctx,tmp_path/str(flag),unary_label_batching=flag) for flag in (False,True)]
    try:
        assert runners[0].selection_execution_binding!=runners[1].selection_execution_binding
        assert runners[0].program_key([])!=runners[1].program_key([])
    finally:
        for runner in runners:runner.close()
