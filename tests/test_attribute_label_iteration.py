"""Target-first signed attribute-copy joins preserve ordered scalar semantics."""
import numpy as np
import pytest

from loris.document import Document
from loris._engine.rdl_reference import Doc, Label, AttrEqual, LabelCompare, Rule, RefGamma
from loris._engine.typed_runtime import IndexedChaseRunner
from loris._storage.selection_execution import make_selection_runner


def context(pos, neg, attributes, presence=None, edges=()):
    n=len(pos)
    return dict(docs=[Document('record %d'%i) for i in range(n)],labels=list('AB')[:pos.shape[1]],
                gamma=RefGamma(pos,neg,edges),attribute_values={'mtd.authors':attributes},
                attribute_presence={'mtd.authors':np.ones(n,bool) if presence is None else presence})


def rule(label='A',sign='pos',reverse=False):
    source,target=('x','y') if reverse else ('y','x')
    return Rule((Doc('x'),Doc('y'),Label(source,label,sign),
                 AttrEqual('x','mtd.authors','y','mtd.authors')),Label(target,label,sign))


def same(a,b,old,new):
    assert a.status==b.status and a.conflicts==b.conflicts and a.n_rounds==b.n_rounds
    for field in ('positive_facts','negative_facts','predictions'):
        np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
    for field in ('subset_facts','strict_subset_facts'):
        assert getattr(a,field).identity()==getattr(b,field).identity()
    assert getattr(a,'gamma_protection',None)==getattr(b,'gamma_protection',None)
    assert a.comparison_conflicts==b.comparison_conflicts
    if hasattr(a,'proof_dag'):
        assert a.proof_dag.identity()==b.proof_dag.identity()
        assert a.provenance==b.provenance
    assert old.stats==new.stats
    # The enabled flag deliberately changes context identity, not execution.
    aa={k:v for k,v in a.reference_audit.items() if k!='context_id'}
    bb={k:v for k,v in b.reference_audit.items() if k!='context_id'}
    assert aa==bb


@pytest.mark.parametrize('selection',[False,True])
@pytest.mark.parametrize('protected',[False,True])
@pytest.mark.parametrize('sign',['pos','neg'])
def test_random_signed_joins_facts_proofs_conflicts_and_all_counters(tmp_path,selection,protected,sign):
    factory=make_selection_runner if selection else IndexedChaseRunner
    for seed in range(35):
        rng=np.random.RandomState(seed);n=12
        pos=rng.rand(n,2)<.23;neg=(rng.rand(n,2)<.24)&~pos
        attrs=[str(x) for x in rng.randint(0,4,n)]
        present=rng.rand(n)>.15
        ctx=context(pos,neg,attrs,present)
        old=factory(ctx,tmp_path/(str(seed)+'old'),gamma_protection=protected,attribute_label_iteration=False)
        new=factory(ctx,tmp_path/(str(seed)+'new'),gamma_protection=protected,attribute_label_iteration=True)
        try:
            program=[]
            for candidate in (rule(sign=sign),rule('B',sign),rule('A',sign,reverse=True)):
                program.append(candidate);same(old(program),new(program),old,new)
        finally:old.close();new.close()


@pytest.mark.parametrize('selection',[False,True])
def test_protected_target_retains_every_source_grounding(tmp_path,selection):
    pos=np.array([[0],[1],[1],[0]],bool);neg=np.array([[1],[0],[0],[0]],bool)
    ctx=context(pos,neg,['g']*4)
    factory=make_selection_runner if selection else IndexedChaseRunner
    old=factory(ctx,tmp_path/'old',gamma_protection=True,attribute_label_iteration=False)
    new=factory(ctx,tmp_path/'new',gamma_protection=True)
    try:
        a,b=old([rule()]),new([rule()]);same(a,b,old,new)
        assert a.positive_facts[:,0].tolist()==[False,True,True,True]
        if selection: assert a.gamma_protection['blocked']==1
        # First round: blocked row0 has two sources; row3 one. Second round:
        # protected row0 now has three sources. No blocked grounding is lost.
        assert old.stats['matched_valuations']==6
    finally:old.close();new.close()


def test_new_earlier_target_is_later_targets_first_proof_witness(tmp_path):
    pos=np.array([[0],[0],[1],[0]],bool);neg=np.zeros_like(pos)
    ctx=context(pos,neg,['g']*4)
    old=IndexedChaseRunner(ctx,tmp_path/'old',attribute_label_iteration=False)
    new=IndexedChaseRunner(ctx,tmp_path/'new')
    try:
        a,b=old([rule()]),new([rule()]);same(a,b,old,new)
        assert dict(b.proof_dag['pos:0:0']['valuation'])=={'x':0,'y':2}
        assert dict(b.proof_dag['pos:1:0']['valuation'])=={'x':1,'y':0}
        assert dict(b.proof_dag['pos:3:0']['valuation'])=={'x':3,'y':0}
    finally:old.close();new.close()


@pytest.mark.parametrize('selection',[False,True])
def test_first_conflict_leaves_same_partial_facts_and_proofs(tmp_path,selection):
    pos=np.array([[0],[0],[1],[0]],bool);neg=np.array([[0],[1],[0],[0]],bool)
    factory=make_selection_runner if selection else IndexedChaseRunner
    ctx=context(pos,neg,['g']*4)
    old=factory(ctx,tmp_path/'old',attribute_label_iteration=False)
    new=factory(ctx,tmp_path/'new')
    try:
        a,b=old([rule()]),new([rule()]);same(a,b,old,new)
        assert a.status=='conflict';assert a.positive_facts[:,0].tolist()==[True,True,True,False]
    finally:old.close();new.close()


@pytest.mark.parametrize('selection',[False,True])
def test_nonidentity_relations_and_richer_bodies_fall_back(tmp_path,selection):
    pos=np.zeros((6,2),bool);pos[4,0]=True;pos[0,1]=True
    ctx=context(pos,np.zeros_like(pos),['g']*6,edges=((0,2),))
    factory=make_selection_runner if selection else IndexedChaseRunner
    old=factory(ctx,tmp_path/'old',attribute_label_iteration=False)
    new=factory(ctx,tmp_path/'new')
    richer=Rule(rule().body+(Label('x','B'),),rule().head)
    try:same(old([richer,rule()]),new([richer,rule()]),old,new)
    finally:old.close();new.close()


def test_bound_option_is_explicit_and_mutation_rejected(tmp_path):
    pos=np.zeros((3,1),bool);ctx=context(pos,pos.copy(),['g']*3)
    with pytest.raises(TypeError):IndexedChaseRunner(ctx,tmp_path/'bad',attribute_label_iteration=1)
    runner=IndexedChaseRunner(ctx,tmp_path/'good')
    try:
        assert runner._binding['attribute_label_iteration']['enabled'] is True
        runner.attribute_label_iteration=False
        with pytest.raises(RuntimeError,match='bound attribute label'):runner([rule()])
    finally:runner.close()


@pytest.mark.parametrize('present',[False,True])
def test_absent_attributes_and_singletons_keep_outer_prefix_counts(tmp_path,present):
    pos=np.zeros((8,1),bool);pos[2,0]=True;neg=np.zeros_like(pos)
    ctx=context(pos,neg,[str(i) for i in range(8)],np.full(8,present,bool))
    old=IndexedChaseRunner(ctx,tmp_path/'old',attribute_label_iteration=False)
    new=IndexedChaseRunner(ctx,tmp_path/'new')
    try:
        a,b=old([rule()]),new([rule()]);same(a,b,old,new)
        assert old.stats['indexed_prefixes']==7
        assert old.stats['matched_valuations']==0
    finally:old.close();new.close()


def test_minimal_fragment_really_enters_specialized_iterator(tmp_path):
    from loris._engine.typed_runtime import _State
    pos=np.array([[False],[False],[True]],bool)
    ctx=context(pos,np.zeros_like(pos),['g']*3)
    runner=IndexedChaseRunner(ctx,tmp_path)
    try:
        parsed,key=runner._rule(rule());prepared=runner._prepare(parsed,key)
        values=runner._attribute_label_valuations(parsed,prepared,_State(runner,runner.gamma))
        assert values is not None
        assert next(values)=={'x':0,'y':2}
        reverse,key=runner._rule(rule(reverse=True))
        assert runner._attribute_label_valuations(reverse,runner._prepare(reverse,key),_State(runner,runner.gamma)) is None
    finally:runner.close()


@pytest.mark.parametrize('selection',[False,True])
@pytest.mark.parametrize('protected',[False,True])
@pytest.mark.parametrize('sign',['pos','neg'])
def test_static_text_model_domains_strict_random_differential(tmp_path,selection,protected,sign):
    from loris._engine.rdl_reference import Text,ML
    from loris.predicates import MatchPredicate
    factory=make_selection_runner if selection else IndexedChaseRunner
    for seed in range(30):
        rng=np.random.RandomState(seed+100);n=18
        pos=rng.rand(n,2)<.25;neg=(rng.rand(n,2)<.25)&~pos
        ctx=context(pos,neg,[str(i) for i in rng.randint(0,5,n)],rng.rand(n)>.1)
        ctx['docs']=[Document(('source ' if rng.rand()<.65 else '')+('target' if rng.rand()<.75 else '')) for _ in range(n)]
        ctx['model_probabilities']={'M':rng.rand(n,2)}
        old=factory(ctx,tmp_path/(str(seed)+'old'),gamma_protection=protected,attribute_label_iteration=False)
        new=factory(ctx,tmp_path/(str(seed)+'new'),gamma_protection=protected)
        try:
            program=[]
            for step in range(4):
                base=rule('AB'[step%2],sign,reverse=step==3)
                source,target=('x','y') if step==3 else ('y','x')
                static=[ML(source,'M','A',float(rng.rand()),dynamic=True)]
                if step%2==0:static.append(Text(target,MatchPredicate('cnt','target')))
                if step%3==0:static.append(Text(source,MatchPredicate('cnt','source')))
                if step%2:static.append(ML(target,'M','B',float(rng.rand()),dynamic=True))
                program.append(Rule(base.body+tuple(static),base.head))
                same(old(program),new(program),old,new)
        finally:old.close();new.close()


def test_new_head_outside_source_static_domain_is_not_a_first_witness(tmp_path):
    from loris._engine.rdl_reference import Text,ML
    from loris.predicates import MatchPredicate
    from loris._engine.typed_runtime import _State
    pos=np.zeros((6,1),bool);pos[4,0]=True
    ctx=context(pos,np.zeros_like(pos),['g']*6)
    ctx['docs']=[Document(x) for x in ('source target','target','source target','target','source target','source')]
    ctx['model_probabilities']={'M':np.array([[0.],[1.],[1.],[0.],[1.],[1.]])}
    base=rule();candidate=Rule(base.body+(Text('y',MatchPredicate('cnt','source')),
        ML('y','M','A',.5,dynamic=True),Text('x',MatchPredicate('cnt','target'))),base.head)
    old=IndexedChaseRunner(ctx,tmp_path/'old',attribute_label_iteration=False)
    new=IndexedChaseRunner(ctx,tmp_path/'new')
    try:
        parsed,key=new._rule(candidate)
        specialized=new._attribute_label_valuations(parsed,new._prepare(parsed,key),_State(new,new.gamma))
        assert specialized is not None and next(specialized)=={'x':0,'y':4}
        # Drop only explicit-hit matching instrumentation. Keep the already
        # computed static masks in the audit, as the cold old runner will do.
        new.stats.pop('indexed_prefixes',None)
        new.stats.pop('matched_valuations',None)
        a,b=old([candidate]),new([candidate]);same(a,b,old,new)
        assert a.positive_facts[:,0].tolist()==[True,True,True,True,True,False]
        for target,source in ((0,4),(1,4),(2,4),(3,2)):
            assert dict(b.proof_dag['pos:%d:0'%target]['valuation'])=={'x':target,'y':source}
    finally:old.close();new.close()


@pytest.mark.parametrize('selection',[False,True])
def test_ineligible_source_does_not_activate_its_group(tmp_path,selection):
    from loris._engine.rdl_reference import ML
    pos=np.array([[1],[0],[0],[1],[0]],bool);neg=np.zeros_like(pos)
    ctx=context(pos,neg,['a','a','a','b','b'])
    ctx['model_probabilities']={'M':np.array([[0.],[1.],[1.],[1.],[1.]])}
    base=rule();candidate=Rule(base.body+(ML('y','M','A',.5,dynamic=True),),base.head)
    factory=make_selection_runner if selection else IndexedChaseRunner
    old=factory(ctx,tmp_path/'old',attribute_label_iteration=False)
    new=factory(ctx,tmp_path/'new')
    try:
        a,b=old([candidate]),new([candidate]);same(a,b,old,new)
        assert a.positive_facts[:,0].tolist()==[True,False,False,True,True]
    finally:old.close();new.close()
