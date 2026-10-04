from types import SimpleNamespace

import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, LabelCompare, ML, Rule
from loris._engine.typed_runtime import IndexedChaseRunner
from loris._engine.typed_session import digest
from loris.labeling.unknown_incremental import _Candidates
from tests.test_unknown_incremental import context


@pytest.mark.parametrize('dependencies', [(), ('pos:1:0',), ('pos:2:0', 'neg:3:1')])
def test_cached_witness_hash_matches_original_canonical_digest(dependencies):
    rule = Rule((Doc('文档'), ML('文档', 'aux', '标签', .731)), Label('文档', '标签'))
    candidates = _Candidates(None, None, (rule,), None)
    for valuation in [{'文档': 3}, {'文档': np.int64(12)}, {'文档': 0}, {'文档': 3}]:
        expected = digest(dict(source=rule.to_dict(), valuation=valuation, dependencies=dependencies))
        assert candidates.witness_order(0, 'rule_application', valuation, dependencies) == expected
    assert candidates.witness_order(None, 'copy', {}, dependencies) == digest(
        dict(source='copy', valuation={}, dependencies=dependencies))


def test_discarded_witness_does_not_materialize_static_evidence():
    rule = Rule((Doc('x'), ML('x', 'aux', 'A', .5)), Label('x', 'A'))
    state = SimpleNamespace(current={}, present=lambda key: False)
    candidates = _Candidates(state, np.zeros((2, 1), bool), (rule,), None)
    calls = []

    def evidence(selected_rule, valuation):
        calls.append(valuation)
        return [], [{'row': valuation['x']}]

    runner = SimpleNamespace(_evidence=evidence)
    candidates.add('pos:0:0', 'rule_application', rule_id=0, valuation={'x': 0}, static=None, runner=runner)
    candidates.add('pos:0:0', 'rule_application', rule_id=0, valuation={'x': 0}, static=None, runner=runner)
    # Static evidence is materialized once, for the winning witness only.
    assert calls == []
    assert candidates.values['pos:0:0']['static_evidence'] == [{'row': 0}]
    assert candidates.values['pos:0:0']['static_evidence'] == [{'row': 0}]
    assert calls == [{'x': 0}]
    assert candidates.attempts == 2


@pytest.mark.parametrize('direction', ['standard', 'reverse'])
def test_dependency_projection_preserves_relation_and_label_order(tmp_path, direction):
    current = context(2)
    current['subset_direction'] = direction
    rule = Rule((Doc('x'), Doc('y'), Label('x', 'A'), LabelCompare('x', 'eq', 'y'),
                 LabelCompare('x', 'subset', 'y'), LabelCompare('y', 'strict_subset', 'x')),
                Label('y', 'B'))
    runner = IndexedChaseRunner(current, tmp_path / 'cache')
    try:
        candidates = _Candidates(None, None, (rule,), None)
        valuation = dict(x=0, y=1)
        assert candidates.rule_dependencies(runner, 0, valuation) == runner._evidence(rule, valuation)[0]
    finally:
        runner.close()
