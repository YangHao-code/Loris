"""Calibration observes receivers without restricting any inference source."""
import numpy as np
import pytest
from loris._engine.rdl_reference import Doc, Label, LabelCompare, AttrEqual, ML, Rule, RefGamma
from loris._engine.typed_runtime import _plain
from loris._storage.discovery_factory import make_discovery_runner
from loris.labeling.unknown_incremental import retaining_chase
from loris.labeling.reliability import count_rule_conclusions
from tests.test_relation_rule_reliability import ctx_for


def compare(ctx,rules,pos,neg,selected,path,gamma_wins=True,execution='incremental'):
    scope=(pos|neg)&selected[:,None]
    runner=make_discovery_runner(ctx,path)
    try:
        full=retaining_chase(runner,rules,ctx['gamma'],collect_rule_conclusions=True,
                             gamma_wins=gamma_wins,execution=execution)
        pruned=retaining_chase(runner,rules,ctx['gamma'],collect_rule_conclusions=True,
                               rule_conclusion_observed=scope,gamma_wins=gamma_wins,execution=execution)
        for name in ['positive_facts','negative_facts','conflict_unknown']:
            assert np.array_equal(getattr(full,name),getattr(pruned,name)),name
        for name in ['subset_facts','strict_subset_facts']:
            assert list(getattr(full,name).rows)==list(getattr(pruned,name).rows),name
        assert full.n_rounds==pruned.n_rounds
        assert full.history_identity==pruned.history_identity
        assert _plain(dict(full.proof_dag.items()))==_plain(dict(pruned.proof_dag.items()))
        assert dict(full.relation_source_markers)==dict(pruned.relation_source_markers)
        assert dict(full.label_relation_markers)==dict(pruned.label_relation_markers)
        a=count_rule_conclusions(runner,full,rules,pos,neg,selected)
        b=count_rule_conclusions(runner,pruned,rules,pos,neg,selected)
        for x,y in zip(a,b):
            for key in ['rule_id','support','correct','q','checked_conclusions_id']:
                assert x[key]==y[key],(key,x,y)
            assert y['matching_instances']<=x['matching_instances']
        assert not pruned.rule_conclusion_observed.flags.writeable
        return full,pruned,a,b
    finally:runner.close()


def test_train_source_is_retained_when_only_dev_receiver_is_observed(tmp_path):
    ctx=ctx_for(['train source','dev receiver'],labels=('A',))
    ctx['gamma']=RefGamma(np.array([[1],[0]],bool),np.zeros((2,1),bool))
    rule=Rule((Doc('x'),Doc('y'),AttrEqual('x','mtd.group','y','mtd.group'),Label('y','A')),Label('x','A'))
    pos=np.array([[0],[1]],bool);neg=np.zeros_like(pos)
    full,pruned,a,b=compare(ctx,(rule,),pos,neg,np.array([0,1],bool),tmp_path/'cache')
    assert b[0]['support']==1 and b[0]['correct']==1
    assert pruned.positive_facts[:,0].all()
    assert b[0]['matching_instances']<a[0]['matching_instances']
    assert (0,0) not in pruned.rule_conclusions[0]['pos']


@pytest.mark.parametrize('gamma_wins',[False,True])
@pytest.mark.parametrize('execution',['incremental','full_recompute'])
def test_random_programs_preserve_facts_witnesses_and_observed_q(tmp_path,gamma_wins,execution):
    labels=('A','B','C');rng=np.random.default_rng(18931)
    for seed in range(60):
        n=5;ctx=ctx_for(['d%d'%i for i in range(n)],labels=labels)
        ctx['attribute_values']={'mtd.group':['g%d'%v for v in rng.integers(0,2,n)]}
        ctx['attribute_presence']={'mtd.group':rng.random(n)>.1}
        ctx['model_probabilities']['aux']=rng.random((n,3)).astype(np.float32)
        known=rng.integers(0,4,(n,3));ctx['gamma']=RefGamma(known==1,known==2)
        rules=[]
        for ri in range(7):
            binary=bool(rng.integers(2));body=[Doc('x')]+([Doc('y')] if binary else [])
            if binary and rng.random()<.75:body.append(AttrEqual('x','mtd.group','y','mtd.group'))
            if rng.random()<.8:body.append(Label('y' if binary else 'x',str(rng.choice(labels)),polarity=str(rng.choice(['pos','neg']))))
            if rng.random()<.4:body.append(ML('x','aux',str(rng.choice(labels)),.5))
            if binary and rng.random()<.2:head=LabelCompare('x',str(rng.choice(['subset','eq'])),'y')
            else:head=Label(str(rng.choice(['x','y'])) if binary else 'x',str(rng.choice(labels)),polarity=str(rng.choice(['pos','neg'])))
            rules.append(Rule(tuple(body),head))
        observed=rng.random((n,3))>.3;truth=rng.random((n,3))>.5
        pos=observed&truth;neg=observed&~truth;selected=rng.random(n)>.4
        compare(ctx,tuple(rules),pos,neg,selected,tmp_path/str(seed),gamma_wins,execution)


def test_scope_cannot_be_broadened_or_used_outside_fresh_collection(tmp_path):
    ctx=ctx_for(['a','b'],labels=('A',));rule=Rule((Doc('x'),),Label('x','A'))
    runner=make_discovery_runner(ctx,tmp_path/'cache')
    try:
        with pytest.raises(ValueError,match='requires calibration'):
            retaining_chase(runner,(rule,),ctx['gamma'],rule_conclusion_observed=np.zeros((2,1),bool))
        closure=retaining_chase(runner,(rule,),ctx['gamma'],collect_rule_conclusions=True,
            rule_conclusion_observed=np.array([[0],[1]],bool))
        with pytest.raises(ValueError,match='exceeds'):
            count_rule_conclusions(runner,closure,(rule,),np.ones((2,1),bool),np.zeros((2,1),bool),np.ones(2,bool))
        with pytest.raises(ValueError):
            retaining_chase(runner,(rule,),ctx['gamma'],collect_rule_conclusions=True,rule_conclusion_observed=np.zeros((1,1),bool))
    finally:runner.close()


def test_later_unknown_keeps_earlier_observed_conclusions_and_unobserved_new_heads(tmp_path):
    ctx=ctx_for(['train','dev'],labels=('A','B'))
    ctx['gamma']=RefGamma(np.array([[1,1],[0,0]],bool),np.zeros((2,2),bool))
    rules=(Rule((Doc('x'),),Label('x','A')),
           Rule((Doc('x'),Label('x','A')),Label('x','B')),
           Rule((Doc('x'),Label('x','B')),Label('x','A',polarity='neg')))
    pos=np.array([[0,0],[1,0]],bool);neg=np.zeros_like(pos)
    full,pruned,a,b=compare(ctx,rules,pos,neg,np.array([0,1],bool),tmp_path/'cache',False)
    assert pruned.conflict_unknown[1,0] and pruned.positive_facts[1,1]
    assert b[0]['support']==b[2]['support']==1
    assert b[0]['correct']==1 and b[2]['correct']==0
    assert (1,1) in pruned.rule_conclusions[1]['pos']  # Unobserved new B still drives inference.
