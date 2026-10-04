import itertools

import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, LabelCompare, RefGamma, Rule
from loris._engine.typed_runtime import IndexedChaseRunner
from loris._learning.incremental import EventState, valuations
from loris.document import Document


@pytest.mark.parametrize('head', [Label('y','A'), Label('x','A','neg'),
    LabelCompare('x','subset','y'), LabelCompare('x','eq','y'),
    LabelCompare('x','strict_subset','y'), LabelCompare('x','subset','x')])
@pytest.mark.parametrize('known_head', [False, True])
def test_pruned_join_matches_exhaustive_finite_valuations(tmp_path,head,known_head):
    n=4;pos=np.array([[1],[0],[1],[0]],bool);neg=~pos
    gamma=RefGamma(pos,neg,((0,2),(2,0),(1,3)),())
    ctx=dict(docs=[Document(cnt=str(i),ttl='',mtd='')for i in range(n)],labels=['A'],gamma=gamma,model_probabilities={})
    runner=IndexedChaseRunner(ctx,tmp_path/'cache')
    try:
        state=EventState(runner,gamma)
        rule=Rule((Doc('x'),Doc('y'),Label('x','A')),head)
        prep=runner._prepare(*runner._rule(rule))
        for fixed in ({},{'x':0},{'y':3},{'x':2,'y':1}):
            expected=[]
            for x,y in itertools.product(range(n),repeat=2):
                assignment={'x':x,'y':y}
                if any(assignment[k]!=v for k,v in fixed.items()) or not state.pos[x,0]:continue
                if isinstance(head,Label):
                    matrix=state.pos if head.polarity=='pos' else state.neg
                    present=bool(matrix[assignment[head.var],0])
                else:
                    a,b=assignment[head.left_var],assignment[head.right_var]
                    ab=bool(state.out[a]&(1<<b));ba=bool(state.out[b]&(1<<a))
                    present=(ab and ba)if head.op=='eq' else bool(state.proper[a]&(1<<b))if head.op=='strict_subset' else ab
                if known_head or not present:expected.append(assignment)
            assert list(valuations(runner,rule,prep,state,fixed,known_head=known_head))==expected
    finally:runner.close()


def test_existing_head_rows_do_not_expand_cartesian_tail(tmp_path):
    n=100;pos=np.ones((n,1),bool);pos[0,0]=False
    ctx=dict(docs=[Document(cnt=str(i),ttl='',mtd='')for i in range(n)],labels=['A'],gamma=RefGamma(pos,np.zeros_like(pos)),model_probabilities={})
    runner=IndexedChaseRunner(ctx,tmp_path/'cache')
    try:
        state=EventState(runner,ctx['gamma']);rule=Rule((Doc('x'),Doc('y')),Label('x','A'))
        prep=runner._prepare(*runner._rule(rule));calls=[]
        runner.application_allowed=lambda ri,r,v: calls.append(dict(v)) or True
        result=list(valuations(runner,rule,prep,state))
        assert result==[dict(x=0,y=y)for y in range(n)]
        assert len(calls)==n # previously n*n complete leaves before head rejection
        assert len(list(valuations(runner,rule,prep,state,known_head=True)))==n*n
    finally:runner.close()
