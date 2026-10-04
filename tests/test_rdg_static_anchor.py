"""An event anchor narrows immutable equality without reordering live joins."""
from collections import ChainMap
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from loris._engine.rdl_reference import AttrEqual, Doc, Label, LabelCompare, RefGamma, Rule, Text
from loris._engine.typed_runtime import IndexedChaseRunner
from loris._learning.incremental import EventState, valuations
from loris._learning.rdg import RuleDependencyGraph
from loris._engine.typed_session import complete_state_identity
from loris.document import Document
from loris.predicates import MatchPredicate


@contextmanager
def original_lookup(runner):
    """Exactly the pre-fix AttrEqual lookup receives the current assignment only."""
    lookup = runner._relation_domain
    def old(atom, var, assignment, state):
        if isinstance(atom, AttrEqual) and isinstance(assignment, ChainMap):
            assignment = assignment.maps[0]
        return lookup(atom, var, assignment, state)
    runner._relation_domain = old
    try:
        yield
    finally:
        runner._relation_domain = lookup


def context(seed=0, n=6):
    rng = np.random.default_rng(seed)
    known = rng.integers(0, 3, (n, 2))
    return dict(docs=[Document(cnt='yes' if i % 2 else 'no', ttl=str(i), mtd='') for i in range(n)],
        labels=['A', 'B'], gamma=RefGamma(known == 1, known == 2), model_probabilities={},
        attribute_values={'mtd.a': [str(x) for x in rng.integers(0, 3, n)],
                          'mtd.b': [str(x) for x in rng.integers(0, 3, n)]},
        attribute_presence={'mtd.a': rng.random(n) > .2, 'mtd.b': rng.random(n) > .2})


@pytest.mark.parametrize('known_head', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_ordered_anchored_valuations_equal_old_with_masks_missing_and_scope(tmp_path, known_head, reverse):
    for seed in range(30):
        ctx = context(seed)
        runner = IndexedChaseRunner(ctx, tmp_path / str(seed))
        try:
            docs = (Doc('y'), Doc('x')) if reverse else (Doc('x'), Doc('y'))
            body = docs + (Label('y', 'A', 'pos' if seed % 2 else 'neg'),
                           AttrEqual('x', 'mtd.a', 'y', 'mtd.b'))
            if seed % 3 == 0:
                body += (AttrEqual('y', 'mtd.a', 'x', 'mtd.b'),)
            if seed % 4 == 0:
                body += (Text('y', MatchPredicate('cnt', 'yes')),)
            rule = Rule(body, Label('x', 'B', 'pos' if seed % 2 else 'neg'))
            prep = runner._prepare(*runner._rule(rule))
            runner.application_allowed = lambda ri, r, val: (val['x'] + val['y']) % 3 != 0
            for fixed in ({}, {'x': 0}, {'y': 4}, {'x': 2, 'y': 1}, {'y': 7}, {'ignored': 1}):
                for allowed in (None, [0, 2, 4, 5]):
                    kw = dict(fixed=fixed, known_head=known_head, allowed_rows=allowed, rule_index=0)
                    with original_lookup(runner):
                        old = list(valuations(runner, rule, prep, EventState(runner, ctx['gamma']), **kw))
                    new = list(valuations(runner, rule, prep, EventState(runner, ctx['gamma']), **kw))
                    assert new == old
        finally:
            runner.close()


def test_target_first_static_anchor_is_used_before_target_enumeration(tmp_path):
    ctx = context(n=100)
    ctx['attribute_values'] = {'mtd.a': [str(i) for i in range(100)]}
    ctx['attribute_presence'] = {'mtd.a': np.ones(100, bool)}
    pos = np.zeros((100, 2), bool); pos[99, 0] = True
    ctx['gamma'] = RefGamma(pos, np.zeros_like(pos))
    runner = IndexedChaseRunner(ctx, tmp_path)
    try:
        rule = Rule((Doc('x'), Doc('y'), Label('y', 'A'),
                     AttrEqual('x', 'mtd.a', 'y', 'mtd.a')), Label('x', 'A'))
        prep = runner._prepare(*runner._rule(rule))
        lookups = []
        real = runner._relation_domain
        def observed(atom, var, assignment, state):
            lookups.append((var, dict(assignment)))
            return real(atom, var, assignment, state)
        runner._relation_domain = observed
        actual = list(valuations(runner, rule, prep, EventState(runner, ctx['gamma']), {'y': 99}, known_head=True))
        assert actual == [{'x': 99, 'y': 99}]
        assert lookups == [('x', {'y': 99}), ('y', {'x': 99, 'y': 99})]
    finally:
        runner.close()


@pytest.mark.parametrize('head', [Label('x', 'B'), Label('x', 'B', 'neg'), LabelCompare('x', 'subset', 'y')])
def test_live_generator_label_and_relation_mutations_keep_original_sequence(tmp_path, head):
    ctx = context(n=5)
    ctx['attribute_values'] = {'mtd.a': ['same'] * 5}
    ctx['attribute_presence'] = {'mtd.a': np.ones(5, bool)}
    pos = np.zeros((5, 2), bool); pos[4, 0] = True
    ctx['gamma'] = RefGamma(pos, np.zeros_like(pos))
    runner = IndexedChaseRunner(ctx, tmp_path)
    try:
        rule = Rule((Doc('x'), Doc('y'), AttrEqual('x', 'mtd.a', 'y', 'mtd.a'),
                     Label('y', 'A')), head)
        prep = runner._prepare(*runner._rule(rule))
        def consume():
            state = EventState(runner, ctx['gamma'])
            values = []
            for val in valuations(runner, rule, prep, state, {'y': 4}, known_head=False):
                values.append(val)
                state.set_label('pos', min(val['x'] + 1, 4), 1, 'test_callback')
                state.remember('subset', val['x'], 4, 'test_callback')
                state.add_edge(val['x'], 4, False)
            return values, state.pos.copy(), state.neg.copy(), list(state.out)
        with original_lookup(runner):
            old = consume()
        new = consume()
        assert new[0] == old[0] and new[3] == old[3]
        assert np.array_equal(new[1], old[1]) and np.array_equal(new[2], old[2])
    finally:
        runner.close()


@pytest.mark.parametrize('op', ['subset', 'eq', 'strict_subset'])
def test_dynamic_comparison_never_receives_future_anchor(tmp_path, op):
    ctx = context(n=4)
    runner = IndexedChaseRunner(ctx, tmp_path)
    try:
        rule = Rule((Doc('x'), Doc('y'), AttrEqual('x', 'mtd.a', 'y', 'mtd.b'),
                     LabelCompare('x', op, 'y')), Label('x', 'A'))
        prep = runner._prepare(*runner._rule(rule))
        original = runner._relation_domain
        seen = []
        def inspect(atom, var, assignment, state):
            if isinstance(atom, LabelCompare):
                assert not isinstance(assignment, ChainMap)
                if var == 'x':
                    assert 'y' not in assignment
                seen.append(var)
            return original(atom, var, assignment, state)
        runner._relation_domain = inspect
        list(valuations(runner, rule, prep, EventState(runner, ctx['gamma']), {'y': 2}, known_head=True))
        assert seen
    finally:
        runner.close()


def test_same_variable_attribute_and_false_source_mask_equal_old(tmp_path):
    ctx = context(n=5)
    runner = IndexedChaseRunner(ctx, tmp_path)
    try:
        rule = Rule((Doc('x'), Doc('y'), AttrEqual('x', 'mtd.a', 'x', 'mtd.b'),
                     AttrEqual('x', 'mtd.a', 'y', 'mtd.a'),
                     Text('y', MatchPredicate('cnt', '^never$'))), Label('x', 'A'))
        prep = runner._prepare(*runner._rule(rule))
        with original_lookup(runner):
            old = list(valuations(runner, rule, prep, EventState(runner, ctx['gamma']), {'y': 2}, known_head=True))
        new = list(valuations(runner, rule, prep, EventState(runner, ctx['gamma']), {'y': 2}, known_head=True))
        assert old == new == []
    finally:
        runner.close()


def test_rdg_scopes_and_batch_rollback_match_old_random_programs(tmp_path):
    for seed in range(40):
        ctx = context(500 + seed)
        runner = IndexedChaseRunner(ctx, tmp_path / str(seed))
        try:
            rules = (
                Rule((Doc('x'), Doc('y'), AttrEqual('x', 'mtd.a', 'y', 'mtd.b'), Label('y', 'A')), Label('x', 'B')),
                Rule((Doc('x'), Doc('y'), AttrEqual('x', 'mtd.b', 'y', 'mtd.a'), Label('y', 'B')), Label('x', 'A')),
                Rule((Doc('x'), Label('x', 'B')), Label('x', 'A', 'neg')),
            )
            result = runner.run(rules, ctx['gamma'])
            session = SimpleNamespace(runner=runner, rules=rules)
            graph = RuleDependencyGraph(rules)
            before = complete_state_identity(result)
            queries = [(i, j) for i in range(6) for j in range(2)]
            with original_lookup(runner):
                old = [graph.activation_scope(session, result, [('pos', i, j)]) for i, j in queries]
                old_scores = list(graph.estimate_batch(session, result, queries, list(range(6))))
            new = [graph.activation_scope(session, result, [('pos', i, j)]) for i, j in queries]
            assert new == old
            assert list(graph.estimate_batch(session, result, queries, list(range(6)))) == old_scores
            assert complete_state_identity(result) == before
        finally:
            runner.close()
