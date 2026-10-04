"""Exact group closure, protected Gamma, and scalar fallback regressions."""
from types import SimpleNamespace

import numpy as np
import pytest

from loris.document import Document
from loris._engine.rdl_reference import Doc, AttrEqual, Label, LabelCompare, ML, RefGamma, Rule
from loris._engine.relation_rows import RelationRows
from loris._storage.selection_execution import SelectionFactsState, make_selection_runner


def bitset(rows):
    return sum(1 << int(row) for row in rows)


def same_state(left, right):
    for name in ('pos', 'neg'):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    for name in ('out', 'inc', 'proper', 'proper_inc', 'label_bits', '_positive_row_bits',
                 'version', 'relation_version', 'blocked_bits', 'gamma_blocked'):
        assert getattr(left, name) == getattr(right, name), name
    assert left.stats['new_subset'] == right.stats['new_subset']
    assert left.invalid() == right.invalid()


@pytest.mark.parametrize('protected', [False, True])
@pytest.mark.parametrize('equality', [False, True])
def test_random_group_closure_matches_scalar_cells_relations_and_counts(protected, equality):
    for seed in range(25):
        rng = np.random.RandomState(seed)
        pos = rng.rand(12, 7) < .16
        neg = (rng.rand(12, 7) < .22) & ~pos if protected else np.zeros_like(pos)
        gamma = RefGamma(pos, neg)
        owner = SimpleNamespace(_relation_batching=False, gamma_protection=protected)
        scalar, block = (SelectionFactsState(owner, gamma) for _ in range(2))
        for _ in range(5):
            a, b = rng.randint(12, size=2).tolist()
            scalar.add_edge(a, b, False)
            block.add_edge(a, b, False)
        sources = np.flatnonzero(rng.rand(12) < .4).tolist()
        targets = np.flatnonzero(rng.rand(12) < .4).tolist()
        for a in sources:
            for b in targets:
                scalar.add_edge(a, b, False)
                if equality:
                    scalar.add_edge(b, a, False)
        block.add_edge_block(bitset(sources), bitset(targets), equality=equality)
        same_state(scalar, block)
        version = block.version
        assert not block.add_edge_block(bitset(sources), bitset(targets), equality=equality)
        assert block.version == version


def test_protected_intermediate_does_not_hide_predecessor_label():
    pos = np.array([[1], [0], [0], [0]], bool)
    neg = np.array([[0], [1], [0], [0]], bool)
    owner = SimpleNamespace(_relation_batching=False, gamma_protection=True)
    state = SelectionFactsState(owner, RefGamma(pos, neg))
    state.add_edge(0, 1, False)
    assert not state.pos[1, 0]
    state.add_edge_block(1 << 1, (1 << 2) | (1 << 3))
    assert state.pos[2:, 0].all()
    assert state.gamma_blocked == 1


@pytest.mark.parametrize('op', ['subset', 'eq'])
@pytest.mark.parametrize('direction', ['standard', 'paper_p7_reverse'])
@pytest.mark.parametrize('body_case', ['plain', 'ml', 'negative', 'positive', 'cross_attribute'])
def test_runner_blocks_match_original_join(tmp_path, op, direction, body_case):
    rng = np.random.RandomState(42)
    n, k = 15, 4
    pos = rng.rand(n, k) < .3
    neg = (rng.rand(n, k) < .3) & ~pos
    ctx = dict(docs=[Document(str(i)) for i in range(n)], labels=list('ABCD'),
        gamma=RefGamma(pos, neg), model_probabilities={'M': rng.rand(n, k)},
        attribute_values={'mtd.a': ['x', 'y', None, 'y', 'z'] * 3,
                          'mtd.b': ['y', None, 'z', 'x', 'y'] * 3})
    right_attr = 'mtd.b' if body_case == 'cross_attribute' else 'mtd.a'
    body = [Doc('x'), Doc('y'), AttrEqual('y', 'mtd.a', 'x', right_attr)]
    if body_case == 'ml':
        body += [ML('x', 'M', 'A', .4, dynamic=True), ML('y', 'M', 'B', .3)]
    if body_case == 'negative': body += [Label('x', 'A', 'neg')]
    if body_case == 'positive': body += [Label('x', 'A')]
    rules = [Rule((Doc('x'), Label('x', 'B')), Label('x', 'D')),
             Rule(tuple(body), LabelCompare('x', op, 'y'))]
    runners = [make_selection_runner(ctx, tmp_path / str(flag), relation_batching=flag,
        gamma_protection=True, subset_direction=direction) for flag in (False, True)]
    try:
        a, b = [runner(rules) for runner in runners]
        assert a.status == b.status == 'fixpoint'
        for field in ('positive_facts', 'negative_facts', 'predictions'):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        for field in ('subset_facts', 'strict_subset_facts'):
            assert set(getattr(a, field)) == set(getattr(b, field))
        assert a.gamma_protection == b.gamma_protection
        assert runners[1].stats['attribute_blocks_matched'] > 0
    finally:
        for runner in runners: runner.close()


def test_strict_state_rejects_direct_block_and_strict_head_keeps_fallback(tmp_path):
    pos = np.zeros((3, 2), bool)
    ctx = dict(docs=[Document(str(i)) for i in range(3)], labels=['A', 'B'],
        gamma=RefGamma(pos, pos.copy()), attribute_values={'mtd.a': ['x'] * 3})
    runner = make_selection_runner(ctx, tmp_path)
    try:
        state = SelectionFactsState(runner, runner.gamma)
        state.add_edge(0, 1, True)
        with pytest.raises(ValueError): state.add_edge_block(1, 4)
        rule = Rule((Doc('x'), Doc('y'), AttrEqual('x', 'mtd.a', 'y', 'mtd.a')),
                    LabelCompare('x', 'strict_subset', 'y'))
        assert runner([rule]).status == 'conflict'
        assert not runner.stats['attribute_blocks_matched']
    finally:
        runner.close()


def test_shared_relation_row_encoding_remains_exact_and_snapshot_private():
    rows = RelationRows(73500, reflexive=True)
    selected = (73000, 73010, 73040)
    value = bitset(selected)
    rows.set_many(selected, value)
    assert rows.union_rows(selected) == value
    assert rows._data[selected[0]] is rows._data[selected[1]]
    snapshot = rows.copy(frozen=True)
    rows[selected[0]] = 1 << selected[0]
    assert snapshot[selected[0]] == value
    assert rows.union_rows([1, 2]) == 6
    with pytest.raises(TypeError): snapshot.set_many([1], value)
    with pytest.raises(IndexError): rows.set_many([73500], value)


@pytest.mark.parametrize('direction', ['standard', 'paper_p7_reverse'])
def test_positive_domains_refresh_across_rule_prefixes_and_join_blocks(tmp_path, direction):
    for seed in range(40):
        rng = np.random.RandomState(seed)
        n, k = 12, 5
        pos = rng.rand(n, k) < .2
        neg = (rng.rand(n, k) < .18) & ~pos
        context = dict(docs=[Document(str(i)) for i in range(n)], labels=list('ABCDE'),
            gamma=RefGamma(pos, neg), model_probabilities={'M': rng.rand(n, k)},
            attribute_values={'mtd.a': rng.randint(3, size=n).astype(str).tolist(),
                              'mtd.b': rng.randint(3, size=n).astype(str).tolist()})
        runners = [make_selection_runner(context, tmp_path / (str(seed) + str(flag)),
            gamma_protection=True, subset_direction=direction, relation_batching=flag)
            for flag in (False, True)]
        rules = []
        try:
            for round_number in range(4):
                body = [Doc('x'), Doc('y'), AttrEqual('x', 'mtd.a', 'y',
                        'mtd.b' if seed % 3 else 'mtd.a'), Label('x', chr(65 + rng.randint(k)))]
                if seed % 4:
                    body.append(Label('y', chr(65 + rng.randint(k)),
                                      'neg' if seed % 4 == 1 else 'pos'))
                if seed % 5:
                    body.append(ML('x', 'M', chr(65 + rng.randint(k)), .4, dynamic=True))
                rules.append(Rule(tuple(body), LabelCompare('x',
                    'eq' if round_number % 2 else 'subset', 'y')))
                a, b = [runner(rules) for runner in runners]
                assert a.status == b.status == 'fixpoint'
                np.testing.assert_array_equal(a.positive_facts, b.positive_facts)
                np.testing.assert_array_equal(a.negative_facts, b.negative_facts)
                assert set(a.subset_facts) == set(b.subset_facts)
                assert set(a.strict_subset_facts) == set(b.strict_subset_facts)
                assert a.gamma_protection == b.gamma_protection
        finally:
            for runner in runners:
                runner.close()


@pytest.mark.parametrize('direction', ['standard', 'paper_p7_reverse'])
def test_positive_attribute_closure_matches_complete_proof_score(tmp_path, direction):
    from loris._storage.discovery_factory import make_discovery_runner
    from loris._storage.selection_evaluator import semantic_result_projection
    from loris._engine.rlearner import TwoFilterEvaluator
    for seed in range(12):
        rng = np.random.RandomState(seed)
        n, k = 10, 4
        pos = rng.rand(n, k) < .23
        neg = (rng.rand(n, k) < .2) & ~pos
        pos[-2:] = neg[-2:] = False
        context = dict(docs=[Document(str(i)) for i in range(n)], labels=list('ABCD'),
            gamma=RefGamma(pos, neg),
            attribute_values={'mtd.a': rng.randint(3, size=n).astype(str).tolist(),
                              'mtd.b': rng.randint(3, size=n).astype(str).tolist()})
        runners = [factory(context, tmp_path / (str(seed) + str(i)),
                   gamma_protection=True, subset_direction=direction)
                   for i, factory in enumerate((make_selection_runner, make_discovery_runner))]
        evaluators = [TwoFilterEvaluator(r, labels=context['labels'], receiver_indices=[8, 9],
            truth=np.zeros((2, k), bool), gamma=context['gamma'], context_id='dynamic-block-proof',
            line7_mode='positive_macro_f1', head_info=lambda r: (None, False)) for r in runners]
        rules = []
        try:
            for step in range(3):
                body = (Doc('x'), Doc('y'), AttrEqual('x', 'mtd.a', 'y', 'mtd.b'),
                        Label('x', chr(65 + rng.randint(k))))
                rules.append(Rule(body, LabelCompare('x', 'eq' if step % 2 else 'subset', 'y')))
                a, b = [e.evaluate(rules) for e in evaluators]
                assert semantic_result_projection(a) == semantic_result_projection(b)
        finally:
            for runner in runners:
                runner.close()
