"""Exact retained-state caching and empty-mask shortcuts; no scientific cutoff."""
import copy
import gc
import weakref
import numpy as np
import pytest
from loris.labeling import unknown_incremental as e
from loris._engine.typed_runtime import bits, IndexedChaseRunner
from loris._engine.rdl_reference import RefGamma, Label, ML
from tests.test_unknown_incremental import context, local, at, edge, same


def immutable(a):
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def test_columns_immutable_private_lists_and_rebound_storage():
    a=immutable(np.array([[1,0],[0,1]],bool))
    first=e._label_columns(a); first[0]=0
    assert e._label_columns(a)==[1,2]
    with pytest.raises(ValueError): a.setflags(write=True)
    with pytest.warns(DeprecationWarning): a.data=memoryview(bytes([0,1,1,0]))
    a.setflags(write=False)
    assert e._label_columns(a)==[2,1]
    a.shape=(4,1)
    assert e._label_columns(a)==[6]


def test_mutable_and_readonly_view_aliases_are_never_cached():
    a=np.array([[1,0],[0,1]],bool); v=a.view();v.setflags(write=False)
    assert e._label_columns(v)==[1,2]
    a[:]=~a
    assert e._label_columns(v)==[2,1]
    assert id(v) not in e._LABEL_COLUMN_CACHE


def test_weak_cache_released_and_bounded():
    a=immutable(np.zeros((3,2),bool));e._label_columns(a);key=id(a);ref=weakref.ref(a)
    del a;gc.collect();assert ref() is None and key not in e._LABEL_COLUMN_CACHE
    arrays=[immutable(np.zeros((3,2),bool)) for _ in range(20)]
    for a in arrays:e._label_columns(a)
    assert len(e._LABEL_COLUMN_CACHE)<=8


@pytest.mark.parametrize('shape',[(0,3),(4,0),(5,3)])
def test_changed_cells_empty_or_nonempty_exact(shape):
    m=np.zeros(shape,bool)
    np.testing.assert_array_equal(e._changed_cells(m),np.argwhere(m))
    if m.size:
        m[-1,-1]=True;m[0,0]=True
        np.testing.assert_array_equal(e._changed_cells(m),np.argwhere(m))


def test_retained_snapshots_are_immutable_but_branch_matrices_are_private(tmp_path):
    c=context(4);r=IndexedChaseRunner(c,tmp_path/'cache')
    try:
        b=e.retaining_chase(r,(),c['gamma'],root_storage='compact')
        for a in (b.positive_facts,b.negative_facts):
            with pytest.raises(ValueError):a.setflags(write=True)
        a=e._HistoryState(r,b);z=e._HistoryState(r,b)
        a.pos[1,0]=True;a.label_bits['pos'][0]|=2
        assert not b.positive_facts[1,0] and not z.pos[1,0]
        assert z.label_bits['pos'][0]==0
        # Public field replacement is allowed but cannot reuse the old packed bits.
        c=copy.copy(b);c.positive_facts=b.positive_facts.copy();c.positive_facts[2,0]=True
        assert e._HistoryState(r,c).label_bits['pos'][0]==4
    finally:r.close()


@pytest.mark.parametrize('gamma_wins',[False,True])
@pytest.mark.parametrize('execution',['incremental','full_recompute'])
def test_random_checks_match_uncached_and_full_relation_validation(tmp_path,monkeypatch,gamma_wins,execution):
    rng=np.random.RandomState(23)
    c=context(7);r=IndexedChaseRunner(c,tmp_path/'cache')
    original_columns=e._label_columns;original_invalid=e._HistoryState.invalid;original_changed=e._changed_cells
    try:
        for turn in range(20):
            values=rng.choice(3,(7,4),p=[.7,.15,.15]);g=RefGamma(values==1,values==2)
            rules=(local(Label('x','B'),Label('x','A')),
                   local(Label('x','C','neg'),Label('x','B'),ML('x','aux','C',.5)),
                   edge(1,3,Label('x','A'),op='subset'))
            b=e.retaining_chase(r,rules,g,gamma_wins=gamma_wins,execution=execution,root_storage='compact')
            p,n=g.pos.copy(),g.neg.copy();row,col=np.argwhere(~(p|n))[0];p[row,col]=True
            checked=RefGamma(p,n);candidate=(p&~g.pos,n&~g.neg) if gamma_wins else None
            kw=dict(before=b,update_kind='check',gamma_wins=gamma_wins,execution=execution,check_candidates=candidate)
            fast=e.retaining_chase(r,rules,checked,decision_only=True,**kw)
            with monkeypatch.context() as m:
                m.setattr(e,'_label_columns',lambda matrix,known=None:[bits(matrix[:,j]) for j in range(matrix.shape[1])])
                m.setattr(e._HistoryState,'invalid',e.EventState.invalid)
                m.setattr(e,'_changed_cells',np.argwhere)
                slow=e.retaining_chase(r,rules,checked,decision_only=True,**kw)
            same(fast,slow);assert fast.n_rounds==slow.n_rounds
            assert fast.candidate_contradictions==slow.candidate_contradictions
            assert fast.reference_audit==slow.reference_audit
            assert e.history_digest(b)==b.history_identity
    finally:r.close()


def test_strict_relation_validation_uses_original_conflict_path(tmp_path):
    c=context(3);r=IndexedChaseRunner(c,tmp_path/'cache')
    try:
        a=e._HistoryState(r);a.proper[0]=1<<1;a.out[1]|=1
        expected=copy.copy(a);expected.last_valid_version=-1
        assert e.EventState.invalid(expected) and a.invalid()
        assert a.conflict==expected.conflict
        assert a.feasibility_checks==expected.feasibility_checks
    finally:r.close()


def test_empty_seed_buffer_rebinding_does_not_inject_facts(tmp_path):
    c=context(3);r=IndexedChaseRunner(c,tmp_path/'cache')
    try:
        e._HistoryState(r)
        seed=r._retained_empty_seed[0]
        with pytest.warns(DeprecationWarning): seed.pos.data=memoryview(bytes([1]+[0]*11))
        seed.pos.setflags(write=False)
        result=e._HistoryState(r)
        assert not result.pos.any()
        assert r._retained_empty_seed[0] is not seed
    finally:r.close()


def test_head_rule_index_preserves_order_polarity_and_scope_indices(tmp_path,monkeypatch):
    c=context(3);r=IndexedChaseRunner(c,tmp_path/'cache')
    try:
        rules=(edge(0,1),local(Label('x','A','neg')),local(Label('x','A'),at('x',2)),
               local(Label('x','B')),local(Label('x','A'),at('x',1)))
        prep=[r._prepare(*r._rule(rule)) for rule in rules];state=e._HistoryState(r)
        index=e._head_rule_index(r,rules)
        assert [i for i,_ in index['pos',0]]==[2,4]
        seen=[];actual=e.valuations
        def tracked(*args,**kw):
            seen.append(kw['rule_index']);yield from actual(*args,**kw)
        monkeypatch.setattr(e,'valuations',tracked)
        assert e._derivable(r,rules,prep,state,'pos:1:0',head_rules=index)
        assert seen==[2,4];seen.clear()
        assert e._derivable(r,rules,prep,state,'pos:1:0') and seen==[2,4]
        for sign in ('pos','neg'):
            for row in range(3):
                for label in range(4):
                    key='%s:%d:%d'%(sign,row,label)
                    assert e._derivable(r,rules,prep,state,key,head_rules=index)==e._derivable(r,rules,prep,state,key)
    finally:r.close()
