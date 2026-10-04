"""Exact ordered composite joins in the frozen retained-chase snapshot."""
from functools import partial
import numpy as np
import pytest
from loris._engine.rdl_reference import Doc, Label, LabelCompare, AttrEqual, Rule, RefGamma
from loris._engine.typed_runtime import _plain
from loris._learning.incremental import EventState, valuations
from loris._storage.discovery_factory import make_discovery_runner
from loris.labeling import unknown_incremental as retained
from loris.labeling.reliability import count_rule_conclusions
from tests.test_relation_rule_reliability import ctx_for


def make(seed,n=7):
    rng=np.random.default_rng(seed)
    ctx=ctx_for(['d%d'%i for i in range(n)],labels=('A','B'))
    ctx['attribute_values']={f'mtd.{a}':['v%d'%v for v in rng.integers(0,3,n)] for a in 'abcd'}
    ctx['attribute_presence']={f'mtd.{a}':rng.random(n)>.15 for a in 'abcd'}
    known=rng.integers(0,4,(n,2));ctx['gamma']=RefGamma(known==1,known==2)
    return ctx,rng


@pytest.mark.parametrize('known',[False,True])
@pytest.mark.parametrize('reverse',[False,True])
def test_composite_valuations_equal_original_with_anchors_scope_and_missing(tmp_path,known,reverse):
    for seed in range(25):
        ctx,rng=make(seed)
        r=make_discovery_runner(ctx,tmp_path/str(seed))
        try:
            docs=(Doc('y'),Doc('x')) if reverse else (Doc('x'),Doc('y'))
            atoms=(AttrEqual('x','mtd.a','y','mtd.b'),AttrEqual('y','mtd.c','x','mtd.d'))
            rule=Rule(docs+atoms+(Label('y','A'),),Label('x','B','neg'))
            prep=r._prepare(*r._rule(rule));state=EventState(r,ctx['gamma'])
            # Domain masks and receiver-only application scopes must remain exact.
            prep['domains']['x'] &= sum(1<<int(i) for i in np.flatnonzero(rng.random(r.n)>.2))
            r._application_scope={'fixture':True}
            r.application_allowed=lambda ri,rule,v:(v['x']+v['y'])%3!=0
            for fixed in ({},{'x':0},{'y':2},{'x':3,'y':4}):
                for allowed in (None,[0,1,3,5]):
                    kwargs=dict(fixed=fixed,allowed_rows=allowed,known_head=known,rule_index=0,frozen_state=True)
                    old=list(valuations(r,rule,prep,state,multi_attribute_join=False,**kwargs))
                    new=list(valuations(r,rule,prep,state,multi_attribute_join=True,**kwargs))
                    assert new==old
        finally:r.close()


def test_missing_components_do_not_match_and_branch_avoids_general_domains(tmp_path,monkeypatch):
    ctx,_=make(1,4)
    ctx['attribute_values']={'mtd.a':['z']*4,'mtd.b':['x','x','y','y']}
    ctx['attribute_presence']={'mtd.a':np.array([1,1,0,0],bool),'mtd.b':np.ones(4,bool)}
    ctx['gamma']=RefGamma.empty(4,2)
    r=make_discovery_runner(ctx,tmp_path/'cache')
    try:
        rule=Rule((Doc('x'),Doc('y'),AttrEqual('x','mtd.a','y','mtd.a'),AttrEqual('x','mtd.b','y','mtd.b')),Label('x','A'))
        prep=r._prepare(*r._rule(rule));state=EventState(r,ctx['gamma'])
        monkeypatch.setattr(r,'_relation_domain',lambda *a:pytest.fail('composite join fell back'))
        result=list(valuations(r,rule,prep,state,known_head=True,frozen_state=True))
        assert result==[{'x':x,'y':y} for x in (0,1) for y in (0,1)]
    finally:r.close()


@pytest.mark.parametrize('gamma_wins',[False,True])
@pytest.mark.parametrize('execution',['incremental','full_recompute'])
def test_retained_programs_preserve_all_facts_q_proofs_and_counters(tmp_path,monkeypatch,gamma_wins,execution):
    for seed in range(35):
        ctx,rng=make(300+seed,6);r=make_discovery_runner(ctx,tmp_path/str(seed))
        try:
            rules=[]
            for ri in range(5):
                attrs=[AttrEqual('x','mtd.a','y','mtd.b'),AttrEqual('y','mtd.c','x','mtd.d')]
                if ri%2:attrs.reverse()
                if ri==2:attrs.append(attrs[0])  # Duplicate equalities must not duplicate valuations.
                body=(Doc('x'),Doc('y'),*attrs,Label('y',str(rng.choice(['A','B'])),str(rng.choice(['pos','neg']))))
                head=Label('x',str(rng.choice(['A','B'])),str(rng.choice(['pos','neg'])))
                rules.append(Rule(body,head))
            rules.append(Rule((Doc('x'),Label('x','A')),Label('x','B')))
            observed=rng.random((6,2))>.3;truth=rng.random((6,2))>.5;selected=rng.random(6)>.25
            pos=observed&truth;neg=observed&~truth
            kw=dict(gamma_wins=gamma_wins,execution=execution,collect_rule_conclusions=True)
            monkeypatch.setattr(retained,'valuations',partial(valuations,multi_attribute_join=False))
            old=retained.retaining_chase(r,rules,ctx['gamma'],**kw)
            monkeypatch.setattr(retained,'valuations',valuations)
            new=retained.retaining_chase(r,rules,ctx['gamma'],**kw)
            for name in ['positive_facts','negative_facts','conflict_unknown']:
                assert np.array_equal(getattr(old,name),getattr(new,name)),name
            assert old.n_rounds==new.n_rounds
            assert old.history_identity==new.history_identity
            assert _plain(dict(old.proof_dag.items()))==_plain(dict(new.proof_dag.items()))
            assert _plain(old.reference_audit)==_plain(new.reference_audit)
            assert _plain(old.rule_conclusions)==_plain(new.rule_conclusions)
            assert _plain(old.rule_conclusion_instances)==_plain(new.rule_conclusion_instances)
            assert count_rule_conclusions(r,old,rules,pos,neg,selected)==count_rule_conclusions(r,new,rules,pos,neg,selected)
        finally:r.close()
