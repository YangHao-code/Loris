"""Complete original-order proof comparison for compact unary candidate iteration."""
import numpy as np
import pytest
from loris.document import Document
from loris.predicates import MatchPredicate
from loris._engine.rdl_reference import Doc,Label,Text,ML,Rule,RefGamma,LabelCompare,AttrEqual
from loris._engine.typed_runtime import IndexedChaseRunner


class ScalarUnaryRunner(IndexedChaseRunner):
    """The pre-optimization single-variable visit loop, retained as test oracle."""
    def _valuations(self,rule,prepared,state):
        head=rule.head
        if not isinstance(head,Label) or len(rule.variables)!=1 or prepared['binary']:
            yield from super()._valuations(rule,prepared,state)
            return
        var=rule.variables[0];previous=-1
        while True:
            domain=prepared['domains'][var]&~((1<<(previous+1))-1)
            for atom in prepared['unary'][var]:
                domain&=state.label_bits[atom.polarity][self.label_index[atom.label]]
            domain&=~state.label_bits[head.polarity][self.label_index[head.label]]
            if not domain:return
            low=domain&-domain;previous=low.bit_length()-1
            self.stats['indexed_prefixes']+=1
            self.stats['matched_valuations']+=1
            yield {var:previous}


def same(a,b,old,new):
    assert a.status==b.status and a.conflicts==b.conflicts and a.n_rounds==b.n_rounds
    for field in ('positive_facts','negative_facts','predictions'):
        np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
    for field in ('subset_facts','strict_subset_facts'):
        assert getattr(a,field).identity()==getattr(b,field).identity()
    assert a.proof_dag.identity()==b.proof_dag.identity()
    assert a.provenance==b.provenance
    assert a.reference_audit == b.reference_audit
    assert old.stats==new.stats


@pytest.mark.parametrize('protected',[False,True])
@pytest.mark.parametrize('relations',[False,True])
def test_unary_generator_preserves_actual_proof_dag_conflicts_and_counts(tmp_path,protected,relations):
    for seed in range(30):
        rng=np.random.RandomState(seed);n,k=9,4
        pos=rng.rand(n,k)<.16;neg=(rng.rand(n,k)<.2)&~pos
        edges=((0,3),(3,6),(0,7)) if relations else ()
        ctx=dict(docs=[Document('a' if i%3 else 'b') for i in range(n)],labels=list('ABCD'),
            gamma=RefGamma(pos,neg,edges),model_probabilities={'M':rng.rand(n,k)})
        old=ScalarUnaryRunner(ctx,tmp_path/(str(seed)+'old'),gamma_protection=protected)
        new=IndexedChaseRunner(ctx,tmp_path/(str(seed)+'new'),gamma_protection=protected)
        rules=[]
        try:
            for step in range(5):
                body=[Doc('x')]
                if step%3:body.append(Label('x',chr(65+rng.randint(k)),'neg' if step%2 else 'pos'))
                if step%2:body.append(ML('x','M',chr(65+rng.randint(k)),.3,dynamic=True))
                if step==2:body.append(Text('x',MatchPredicate('cnt','a')))
                rules.append(Rule(tuple(body),Label('x',chr(65+rng.randint(k)),'neg' if rng.rand()<.4 else 'pos')))
                same(old(rules),new(rules),old,new)
        finally:old.close();new.close()


def test_propagation_skips_known_future_rows_without_changing_first_witness(tmp_path):
    pos=np.zeros((8,2),bool);ctx=dict(docs=[Document('a')]*8,labels=['A','B'],gamma=RefGamma(pos,pos.copy(),((0,2),(0,5),(3,7))))
    old=ScalarUnaryRunner(ctx,tmp_path/'old');new=IndexedChaseRunner(ctx,tmp_path/'new')
    try:
        rule=Rule((Doc('x'),),Label('x','A'));a,b=old([rule]),new([rule]);same(a,b,old,new)
        assert old.stats['matched_valuations']==5
        assert a.proof_dag['pos:2:0']['origin']=='inclusion_label_copy'
        assert a.proof_dag['pos:5:0']['origin']=='inclusion_label_copy'
    finally:old.close();new.close()


def test_first_signed_conflict_does_not_yield_later_candidates(tmp_path):
    pos=np.zeros((5,2),bool);neg=pos.copy();neg[2,0]=True
    ctx=dict(docs=[Document('a')]*5,labels=['A','B'],gamma=RefGamma(pos,neg))
    old=ScalarUnaryRunner(ctx,tmp_path/'old');new=IndexedChaseRunner(ctx,tmp_path/'new')
    try:
        rule=Rule((Doc('x'),),Label('x','A'));a,b=old([rule]),new([rule]);same(a,b,old,new)
        assert a.status=='conflict' and old.stats['matched_valuations']==3
        assert a.positive_facts[:,0].tolist()==[True,True,True,False,False]
    finally:old.close();new.close()


def test_uncertified_positive_closure_keeps_dynamic_domain_refresh(tmp_path):
    from types import SimpleNamespace
    pos=np.zeros((5,2),bool);pos[0,0]=True;neg=np.zeros_like(pos)
    ctx=dict(docs=[Document('a')]*5,labels=['A','B'],gamma=RefGamma(pos,neg))
    runner=IndexedChaseRunner(ctx,tmp_path)
    try:
        rule=Rule((Doc('x'),Label('x','A')),Label('x','B'))
        parsed,key=runner._rule(rule);prepared=runner._prepare(parsed,key)
        state=SimpleNamespace(pos=pos,neg=neg,label_bits={'pos':[1,0],'neg':[0,0]},_positive_closed=False)
        values=runner._valuations(parsed,prepared,state)
        assert next(values)=={'x':0}
        # Simulate an explicitly uncertified/deferred body-column update.
        state.pos[3,0]=True;state.label_bits['pos'][0]|=1<<3
        assert list(values)==[{'x':3}]
    finally:runner.close()
