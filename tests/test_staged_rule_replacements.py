import sys
from pathlib import Path
import numpy as np
import pytest
from loris.discovery.staged import search as s
from loris.discovery.staged import inventory as prior
from loris.discovery.staged import program as rb
from loris.document import Document
from loris.predicates import FreqPredicate
from loris._engine.rdl_reference import Rule,Doc,ML,Text,Label,AttrEqual,RefGamma,ReferenceChase
from loris._storage.discovery_factory import make_discovery_runner


def setup(tmp_path,n=12,k=3):
    rng=np.random.RandomState(28)
    ctx=dict(docs=[Document(cnt='red '*i+' blue',mtd=str(i%3)) for i in range(n)],row_ids=list(map(str,range(n))),
        labels=list('ABC')[:k],gamma=RefGamma(np.zeros((n,k),bool),np.zeros((n,k),bool)),model_probabilities={'m':rng.rand(n,k).astype('float32')})
    return ctx,make_discovery_runner(ctx,tmp_path/'cache',gamma_protection=True)


@pytest.mark.parametrize('op',['>','>=','<','<=','==','!='])
def test_complement_native_including_threshold_equalities(tmp_path,op):
    ctx,runner=setup(tmp_path)
    try:
        p=FreqPredicate(attr='cnt',r='red',op=op,eta=3)
        prog=rb.Program(runner,[]);fac=prior.Factory(ctx,runner,prog,[])
        inv=s.complete_complement(fac,p);assert inv is not None
        a=runner._static(Text('x',p));b=runner._static(Text('x',inv))
        assert a^b==(1<<prog.n)-1 and not a&b
    finally:runner.close()


def test_missing_attribute_is_not_a_complement_partition(tmp_path):
    ctx,runner=setup(tmp_path);runner.close()
    ctx['attribute_presence']={'cnt':np.array([False]+[True]*11)}
    runner=make_discovery_runner(ctx,tmp_path/'other',gamma_protection=True)
    try:
        prog=rb.Program(runner,[]);fac=prior.Factory(ctx,runner,prog,[])
        assert s.complete_complement(fac,FreqPredicate(attr='cnt',r='red',op='>=',eta=3)) is None
    finally:runner.close()


def test_removed_support_and_descendants_retracted(tmp_path):
    ctx,runner=setup(tmp_path)
    rules=[Rule((Doc('x'),ML('x','m','A',.1)),Label('x','A')),
        Rule((Doc('x'),Label('x','A')),Label('x','B')),
        Rule((Doc('x'),Label('x','B')),Label('x','C')),
        Rule((Doc('x'),ML('x','m','A',.9)),Label('x','A'))]
    try:
        prog=rb.Program(runner,rules);ids=s.mutate([0,1,2],dict(remove=[0],add=[3]))
        old=prog.prediction(prog.closure(prog.seed,[0,1,2]),np.arange(prog.n))
        new,_=s.Evaluator(prog,np.arange(prog.n),old)(ids)
        new=prog.prediction(new,np.arange(prog.n))
        oracle=ReferenceChase(ctx['labels'],model_probabilities=ctx['model_probabilities']).run(ctx['docs'],[rules[i] for i in ids],ctx['gamma'])
        assert np.array_equal(new,oracle.positive_facts) and (old&~new).any()
        assert (new[:,0]==new[:,1]).all() and (new[:,1]==new[:,2]).all()
        assert s.mutate(ids,dict(remove=[0],add=[3])) is None
    finally:runner.close()


@pytest.mark.parametrize('seed',range(8))
def test_random_replacements_match_native_reference(tmp_path,seed):
    ctx,runner=setup(tmp_path,n=9);rng=np.random.RandomState(seed)
    rules=[Rule((Doc('x'),ML('x','m',l,float(c))),Label('x',l)) for l in ctx['labels'] for c in [.3,.7]]
    rules += [Rule((Doc('x'),Label('x','A'),Text('x',FreqPredicate(attr='cnt',r='red',op='>=',eta=3))),Label('x','B')),
        Rule((Doc('x'),Doc('y'),AttrEqual('x','mtd','y','mtd'),Label('y','B')),Label('x','C'))]
    try:
        prog=rb.Program(runner,rules)
        for _ in range(12):
            ids=np.flatnonzero(rng.rand(len(rules))>.4).tolist()
            got=prog.prediction(prog.closure(prog.seed,ids),np.arange(prog.n))
            expected=ReferenceChase(ctx['labels'],model_probabilities=ctx['model_probabilities']).run(ctx['docs'],[rules[i] for i in ids],ctx['gamma'])
            assert expected.status!='conflict' and np.array_equal(got,expected.positive_facts)
    finally:runner.close()


def test_redundant_rules_do_not_count_as_effective_diversity(tmp_path):
    ctx,runner=setup(tmp_path)
    a=Rule((Doc('x'),ML('x','m','A',.1)),Label('x','A'))
    b=Rule(a.body+(Text('x',FreqPredicate(attr='cnt',r='red',op='>=',eta=3)),),a.head)
    try:
        prog=rb.Program(runner,[a,b]);f=s.effective_families(prog,[a,b],[0,1],np.arange(prog.n))
        assert f['model_text']['necessary_facts']==0
    finally:runner.close()


def test_small_complete_bo_and_admission(tmp_path):
    ctx,runner=setup(tmp_path,n=30,k=2);ctx['model_probabilities']['loris_calibrated_pool']=ctx['model_probabilities']['m']
    runner.close();runner=make_discovery_runner(ctx,tmp_path/'all',gamma_protection=True)
    rules=[Rule((Doc('x'),ML('x','m',l,.5)),Label('x',l)) for l in ctx['labels']]
    try:
        prog=rb.Program(runner,rules);fac=prior.Factory(ctx,runner,prog,rules)
        rows=np.arange(20);y=ctx['model_probabilities']['m'][rows]>.7
        pattern=FreqPredicate(attr='cnt',r='red',op='>=',eta=5)
        matrix=np.array([[i>=5] for i in rows])
        bundles,record=s.proposals(ctx,fac,[0,1],rows,y,['m'],{'m':[.7,.7]},[pattern],matrix,[])
        assert bundles and record['final_rule_cap_per_label'] is None
        hold=np.arange(20,30);hy=ctx['model_probabilities']['m'][hold]>.7
        ids,trace=s.accept(prog,rules,[0,1],bundles,rows,y,hold,hy,diverse=True)
        assert trace
        got=prog.prediction(prog.closure(prog.seed,ids),np.arange(prog.n))
        expected=ReferenceChase(ctx['labels'],model_probabilities=ctx['model_probabilities']).run(ctx['docs'],[rules[i] for i in ids],ctx['gamma'])
        assert np.array_equal(got,expected.positive_facts)
    finally:runner.close()
