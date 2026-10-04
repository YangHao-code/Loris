"""Ordered copy joins may intersect label columns which this rule cannot change."""
import numpy as np
import pytest

from loris._engine.rdl_reference import Label, Rule
from loris._engine.typed_runtime import IndexedChaseRunner, _State
from loris._storage.selection_execution import make_selection_runner
from tests.test_attribute_label_iteration import context, rule, same


@pytest.mark.parametrize('selection', [False, True])
@pytest.mark.parametrize('protected', [False, True])
@pytest.mark.parametrize('sign', ['pos', 'neg'])
def test_fixed_other_label_columns_preserve_complete_ordered_semantics(tmp_path, selection, protected, sign):
    factory = make_selection_runner if selection else IndexedChaseRunner
    for seed in range(30):
        rng = np.random.RandomState(seed + 391)
        n = 14
        pos = rng.rand(n, 2) < .28
        neg = (rng.rand(n, 2) < .29) & ~pos
        ctx = context(pos, neg, [str(v) for v in rng.randint(0, 4, n)], rng.rand(n) > .1)
        old = factory(ctx, tmp_path/(str(seed)+'old'), gamma_protection=protected,
                      attribute_label_iteration=False)
        new = factory(ctx, tmp_path/(str(seed)+'new'), gamma_protection=protected)
        source_condition = Label('y', 'B', 'neg' if seed % 2 else 'pos')
        target_condition = Label('x', 'B', 'pos' if seed % 3 else 'neg')
        a = rule(sign=sign)
        b = rule('B', 'pos')
        program = []
        try:
            # The B rule changes a formerly fixed column only between
            # enumerations. The next A enumeration must take a fresh snapshot.
            for candidate in (Rule(a.body+(source_condition,), a.head),
                              Rule(b.body+(Label('y', 'A', sign),), b.head),
                              Rule(a.body+(source_condition,target_condition), a.head)):
                program.append(candidate)
                same(old(program), new(program), old, new)
        finally:
            old.close(); new.close()


def test_new_sources_must_also_satisfy_fixed_negative_column(tmp_path):
    pos = np.zeros((5, 2), bool); pos[2, 0] = True
    neg = np.zeros_like(pos); neg[[1,2,4], 1] = True
    ctx = context(pos, neg, ['group']*5)
    copied = rule(); candidate = Rule(copied.body+(Label('y', 'B', 'neg'),),copied.head)
    old = IndexedChaseRunner(ctx, tmp_path/'old', attribute_label_iteration=False)
    new = IndexedChaseRunner(ctx, tmp_path/'new')
    try:
        a,b = old([candidate]),new([candidate]); same(a,b,old,new)
        assert b.proof_dag['pos:1:0']['valuation']['y'] == 2
        assert b.proof_dag['pos:3:0']['valuation']['y'] == 1
        assert all(b.positive_facts[:,0])
    finally:
        old.close(); new.close()


def test_fixed_columns_branch_is_used_and_head_column_extras_fall_back(tmp_path):
    pos = np.zeros((4,2),bool); pos[2] = True
    ctx = context(pos,np.zeros_like(pos),['group']*4)
    runner = IndexedChaseRunner(ctx,tmp_path/'cache')
    try:
        base = rule()
        candidate = Rule(base.body+(Label('y','B'),),base.head)
        canonical,key = runner._rule(candidate)
        prepared = runner._prepare(canonical,key)
        value = runner._attribute_label_valuations(canonical,prepared,_State(runner,runner.gamma))
        assert value is not None and next(value) == {'x':0,'y':2}
        for condition in (Label('x','A'),Label('x','A','neg'),Label('y','A','neg')):
            candidate = Rule(base.body+(condition,),base.head)
            canonical,key = runner._rule(candidate)
            assert runner._attribute_label_valuations(canonical,runner._prepare(canonical,key),
                                                      _State(runner,runner.gamma)) is None
    finally:
        runner.close()


def test_fixed_target_and_false_source_columns_do_not_activate_groups(tmp_path):
    pos = np.zeros((5,2),bool); pos[2,0] = True
    # Only row1 satisfies the target condition; no A source satisfies -B.
    pos[1,1] = True
    neg = np.zeros_like(pos); neg[4,1] = True
    ctx = context(pos,neg,['group']*5)
    base = rule()
    candidate = Rule(base.body+(Label('y','B','neg'),Label('x','B','pos')),base.head)
    old = IndexedChaseRunner(ctx,tmp_path/'old',attribute_label_iteration=False)
    new = IndexedChaseRunner(ctx,tmp_path/'new')
    try:
        a,b = old([candidate]),new([candidate]);same(a,b,old,new)
        assert not b.positive_facts[1,0]
        assert new.stats['indexed_prefixes'] == 1
        assert new.stats['matched_valuations'] == 0
    finally:
        old.close();new.close()
