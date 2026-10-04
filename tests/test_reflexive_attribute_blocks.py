"""Skipping re-added reflexivity preserves every ordered observable."""
import numpy as np
import pytest

from loris.document import Document
from loris._engine.rdl_reference import Doc, Label, AttrEqual, LabelCompare, ML, RefGamma, Rule
from loris._storage import selection_execution as execution


def context(pos, neg, left, right=None, probabilities=None, edges=(), strict=()):
    return dict(docs=[Document(str(i)) for i in range(len(pos))], labels=list('ABC')[:pos.shape[1]],
                gamma=RefGamma(pos, neg, edges, strict),
                attribute_values={'mtd.a':left, 'mtd.b':left if right is None else right},
                model_probabilities={} if probabilities is None else {'M':probabilities})


def relation(op='subset', right='mtd.a', extra=()):
    return Rule((Doc('x'),Doc('y'),AttrEqual('x','mtd.a','y',right))+tuple(extra),LabelCompare('x',op,'y'))


def compare(old, new, program):
    a,b=old(program),new(program)
    assert a.status==b.status and a.conflicts==b.conflicts
    assert a.comparison_conflicts==b.comparison_conflicts and a.n_rounds==b.n_rounds
    for name in ('positive_facts','negative_facts','predictions'):
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    for name in ('subset_facts','strict_subset_facts'):
        assert getattr(a,name).identity()==getattr(b,name).identity()
    assert a.gamma_protection==b.gamma_protection
    assert old.stats==new.stats
    assert a.reference_audit['proof_materialized'] is b.reference_audit['proof_materialized'] is False
    aa={k:v for k,v in a.reference_audit.items() if k!='context_id'}
    bb={k:v for k,v in b.reference_audit.items() if k!='context_id'}
    assert aa==bb
    return a,b


def runners(ctx,path,**kwargs):
    return [execution.make_selection_runner(ctx,path/str(flag),skip_reflexive_attribute_blocks=flag,**kwargs) for flag in (False,True)]


@pytest.mark.parametrize('protected',[False,True])
@pytest.mark.parametrize('direction',['standard','paper_p7_reverse'])
def test_random_partial_groups_cross_attributes_gamma_and_all_counters(tmp_path,protected,direction):
    for seed in range(40):
        rng=np.random.RandomState(seed);n=11;k=3
        pos=rng.rand(n,k)<.23;neg=(rng.rand(n,k)<.19)&~pos
        left=[None if i%7==0 else str(x) for i,x in enumerate(rng.randint(0,8,n))]
        right=[None if i%6==0 else str(x) for i,x in enumerate(rng.randint(0,8,n))]
        ctx=context(pos,neg,left,right,rng.rand(n,k))
        old,new=runners(ctx,tmp_path/str(seed),gamma_protection=protected,subset_direction=direction)
        program=[]
        try:
            for step in range(4):
                extra=[]
                if step%2:extra.append(ML('x','M','A',.4,dynamic=True))
                if step%3:extra.append(Label('y','B','neg' if step==2 else 'pos'))
                program.append(relation('eq' if step%2 else 'subset','mtd.b' if step%3 else 'mtd.a',extra))
                compare(old,new,program)
        finally:old.close();new.close()


@pytest.mark.parametrize('op',['subset','eq'])
def test_same_row_singletons_really_skip_bitmap_and_closure_work(tmp_path,monkeypatch,op):
    pos=np.zeros((9,1),bool);pos[0]=True
    ctx=context(pos,np.zeros_like(pos),[str(i) for i in range(9)])
    old,new=runners(ctx,tmp_path,gamma_protection=True)
    calls=[];original=execution.row_bits
    def counted(rows):
        calls.append(len(rows));return original(rows)
    monkeypatch.setattr(execution,'row_bits',counted)
    try:
        a=old([relation(op)]);old_count=len(calls);calls.clear();b=new([relation(op)])
        assert old_count==18 and calls==[]
        assert old.stats==new.stats and new.stats['attribute_blocks_matched']==9
        assert a.subset_facts.identity()==b.subset_facts.identity()
        np.testing.assert_array_equal(a.positive_facts,b.positive_facts)
    finally:old.close();new.close()


@pytest.mark.parametrize('op',['subset','eq'])
@pytest.mark.parametrize('protected',[False,True])
def test_cross_attribute_singletons_at_different_rows_still_propagate(tmp_path,op,protected):
    pos=np.array([[True],[False],[False]],bool);neg=np.array([[False],[True],[False]],bool)
    old,new=runners(context(pos,neg,['a','b','c'],['b','a','c']),tmp_path,gamma_protection=protected)
    try:
        a,b=compare(old,new,[relation(op,'mtd.b')])
        assert (0,1) in b.subset_facts
        if protected:assert b.gamma_protection['blocked']==1 and not b.positive_facts[1,0]
        else:assert b.status=='conflict'
    finally:old.close();new.close()


def test_multirow_groups_filter_to_same_singleton_without_losing_later_blocks(tmp_path):
    pos=np.array([[True],[False],[False],[False]],bool)
    probs=np.array([[.9],[.1],[.1],[.9]])
    ctx=context(pos,np.zeros_like(pos),['g','g','h','h'],probabilities=probs)
    old,new=runners(ctx,tmp_path)
    try:
        compare(old,new,[relation(extra=(ML('x','M','A',.5,dynamic=True),ML('y','M','A',.5,dynamic=True)))])
        assert new.stats['attribute_blocks_matched']==2 and not new.stats['attribute_relation_blocks']
    finally:old.close();new.close()


@pytest.mark.parametrize('op',['subset','eq'])
def test_nonidentity_gamma_preserves_existing_scalar_fallback(tmp_path,op):
    pos=np.array([[True],[False],[False],[False]],bool);neg=np.array([[False],[True],[False],[False]],bool)
    old,new=runners(context(pos,neg,list('abcd'),edges=((0,1),(1,2))),tmp_path,gamma_protection=True)
    try:
        compare(old,new,[relation(op)])
        assert new.stats['attribute_blocks_matched']==0
    finally:old.close();new.close()


def test_strict_self_relation_is_not_skipped(tmp_path):
    pos=np.zeros((3,1),bool);old,new=runners(context(pos,pos.copy(),list('abc')),tmp_path)
    try:
        a,b=compare(old,new,[relation('strict_subset')])
        assert b.status=='conflict' and not new.stats['attribute_blocks_matched']
    finally:old.close();new.close()


def test_strict_preexisting_relation_keeps_ordinary_fallback(tmp_path):
    pos=np.zeros((3,1),bool);ctx=context(pos,pos.copy(),list('abc'),edges=((0,1),),strict=((0,1),))
    old,new=runners(ctx,tmp_path)
    try:
        compare(old,new,[relation()]);assert not new.stats['attribute_blocks_matched']
    finally:old.close();new.close()


def test_policy_is_explicit_bound_and_cannot_be_mutated(tmp_path):
    pos=np.zeros((1,1),bool);ctx=context(pos,pos.copy(),['a'])
    with pytest.raises(ValueError):execution.make_selection_runner(ctx,tmp_path/'bad',skip_reflexive_attribute_blocks=1)
    runner=execution.make_selection_runner(ctx,tmp_path/'good')
    try:
        assert runner.selection_execution_binding['attribute_relation_blocks']['skip_reflexive_singletons'] is True
        runner._skip_reflexive_attribute_blocks=False
        with pytest.raises(RuntimeError,match='bound reflexive'):runner([relation()])
    finally:runner.close()


def test_reflexive_skip_after_learned_relation_prefix(tmp_path):
    pos=np.array([[True],[False],[False],[False]],bool)
    ctx=context(pos,np.zeros_like(pos),list('abcd'),list('bacd'))
    old,new=runners(ctx,tmp_path,gamma_protection=True)
    try:
        a,b=compare(old,new,[relation('eq','mtd.b'),relation('subset')])
        assert (0,1) in b.subset_facts and b.positive_facts[1,0]
        assert new.stats['attribute_blocks_matched']>=8
    finally:old.close();new.close()
