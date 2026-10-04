import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, LabelCompare, Rule
from loris._engine.typed_session import TypedChaseSession, complete_state_identity
from loris._learning.rdg import RuleDependencyGraph, _HypothesisState
from tests.test_labeling import fixture


@pytest.mark.parametrize('comparison', ['subset', 'strict_subset', 'eq'])
def test_batched_hypotheses_match_fresh_states_and_preserve_original(tmp_path, comparison):
    context, rules, _ = fixture()
    rules = rules + (
        Rule((Doc('x'), Label('x', 'B')), Label('x', 'A')),
        Rule((Doc('x'), Doc('y'), Label('x', 'B'), Label('y', 'A')),
             LabelCompare('x', comparison, 'y')),
        Rule((Doc('x'), Doc('y'), LabelCompare('x', 'subset', 'y'), Label('x', 'B')),
             Label('y', 'B')),
    )
    session = TypedChaseSession(context, rules, tmp_path / 'cache', model_snapshot_id='fixture')
    result = session.chase(context['gamma'])
    identity = complete_state_identity(result)
    graph = RuleDependencyGraph(rules)
    queries = [(row, label) for row in range(4) for label in range(2)]
    expected = [graph.estimate(session, result, row, label, [1, 2, 3]) for row, label in queries]
    assert list(graph.estimate_batch(session, result, queries, [1, 2, 3])) == expected
    assert list(graph.estimate_batch(session, result, reversed(queries), [1, 2, 3])) == expected[::-1]
    assert complete_state_identity(result) == identity
    session.close()


def test_hypothesis_state_rolls_back_partial_mutation_after_error(tmp_path):
    context, rules, _ = fixture()
    session = TypedChaseSession(context, rules, tmp_path / 'cache', model_snapshot_id='fixture')
    result = session.chase(context['gamma'])
    state = _HypothesisState.from_result(session.runner, result)
    before = dict(positive=state.pos.copy(), negative=state.neg.copy(),
                  out=state.out.copy(), inc=state.inc.copy(),
                  proper=state.proper.copy(), proper_inc=state.proper_inc.copy())
    with pytest.raises(RuntimeError):
        try:
            state.set_label('pos', 2, 1, 'hypothesis')
            state.remember('subset', 2, 3, 'hypothesis')
            state.add_edge(2, 3, False)
            state.set_label('neg', 3, 0, 'hypothesis')
            raise RuntimeError('synthetic interruption')
        finally:
            state.rollback()
    assert np.array_equal(state.pos, before['positive'])
    assert np.array_equal(state.neg, before['negative'])
    for name in ['out', 'inc', 'proper', 'proper_inc']:
        assert getattr(state, name) == before[name]
    assert not state.events and not state.proof and not state.provenance.maps[0]
    session.close()
