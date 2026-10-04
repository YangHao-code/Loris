"""Storage-only change: preserve original groups, joins, proofs and counters."""
import numpy as np
import pytest

from loris.document import Document
from loris._engine.attribute_index import CompactAttributeIndex, row_bits, shared_group_positions
from loris._engine.rdl_reference import Doc, AttrEqual, Label, LabelCompare, RefGamma, Rule
from loris._engine.typed_runtime import IndexedChaseRunner
from loris._storage.selection_execution import make_selection_runner
from loris._storage.discovery_factory import make_discovery_runner


def legacy_groups(column):
    result = {}
    for row, code in enumerate(column.tolist()):
        if code >= 0:
            result[code] = result.get(code, 0) | (1 << row)
    return result


@pytest.mark.parametrize('size', [0, 1, 23, 257, 1025])
def test_group_values_and_insertion_order(size):
    rng = np.random.RandomState(size)
    columns = [rng.randint(-1, 11, size=size) for _ in range(2)]
    groups = [CompactAttributeIndex(x) for x in columns]
    old = [legacy_groups(x) for x in columns]
    for group, mapping in zip(groups, old):
        assert list(group.items()) == list(mapping.items())
        for code in range(-1, 12):
            assert group.get(code, 0) == mapping.get(code, 0)
            assert (code in group) == (code in mapping)
        for mask in [np.ones(size, bool), rng.rand(size) < .25]:
            actual = group.matching_groups(mask)
            for position, code in enumerate(group.codes):
                rows = group.group_rows(position)
                assert actual[position] == any(mask[row] for row in rows)
                assert row_bits(rows[mask[rows]]) == (mapping[code] & row_bits(np.flatnonzero(mask)))
    lp, rp = shared_group_positions(*groups)
    assert [(int(groups[0].codes[i]), groups[0].get(groups[0].codes[i]),
             groups[1].get(groups[1].codes[j])) for i, j in zip(lp, rp)] == [
             (code, rows, old[1][code]) for code, rows in old[0].items() if code in old[1]]


def test_high_row_singleton_storage_is_linear_and_does_not_cache_bitmaps():
    size = 200_000
    index = CompactAttributeIndex(np.arange(size, dtype=np.int64))
    before = index.storage_bytes
    assert before <= 21 * size
    for row in [0, size // 2, size - 1]:
        assert index[row] == 1 << row
    assert index.storage_bytes == before
    assert not hasattr(index, '__dict__')
    assert index.get(-1, 0) == 0


def assert_result_same(left, right):
    assert left.status == right.status
    assert left.n_rounds == right.n_rounds
    assert left.conflicts == right.conflicts
    for field in ('positive_facts', 'negative_facts', 'predictions'):
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
    for field in ('subset_facts', 'strict_subset_facts'):
        assert list(getattr(left, field)) == list(getattr(right, field))
    for field in ('gamma_protection', 'comparison_conflicts', 'provenance', 'proof_dag'):
        if hasattr(left, field):
            assert getattr(left, field) == getattr(right, field), field
    assert left.reference_audit['incremental_relation_statistics'] == right.reference_audit['incremental_relation_statistics']
    assert left.reference_audit['finite_comparison_solver_calls'] == right.reference_audit['finite_comparison_solver_calls']
    assert left.reference_audit['finite_comparison_certificate_statistics'] == right.reference_audit['finite_comparison_certificate_statistics']


@pytest.mark.parametrize('factory', [IndexedChaseRunner, make_discovery_runner, make_selection_runner])
@pytest.mark.parametrize('protected', [False, True])
@pytest.mark.parametrize('direction', ['standard', 'paper_p7_reverse'])
def test_complete_random_programs_match_legacy_storage(tmp_path, factory, protected, direction):
    for seed in range(15):
        rng = np.random.RandomState(seed)
        n, k = 11, 4
        pos = rng.rand(n, k) < .19
        neg = (rng.rand(n, k) < .17) & ~pos
        attrs = {key: [None if value == 0 else str(value) for value in rng.randint(5, size=n)]
                 for key in ['mtd.a', 'mtd.b']}
        context = dict(docs=[Document(str(i)) for i in range(n)], labels=list('ABCD'),
            gamma=RefGamma(pos, neg), attribute_values=attrs)
        runners = [factory(context, tmp_path / ('%s_%s' % (seed, flag)),
                          compact_attribute_index=flag, gamma_protection=protected,
                          subset_direction=direction) for flag in (False, True)]
        rules = []
        try:
            for step in range(4):
                same = seed % 3 == 0
                body = [Doc('x'), Doc('y'), AttrEqual('x', 'mtd.a', 'y', 'mtd.a' if same else 'mtd.b')]
                if seed % 4:
                    body.append(Label('x', chr(65 + rng.randint(k)), 'neg' if seed % 4 == 1 else 'pos'))
                if step % 3 == 0:
                    head = Label('y', chr(65 + rng.randint(k)), 'neg' if seed % 5 == 0 else 'pos')
                else:
                    head = LabelCompare('y', ['subset', 'eq', 'strict_subset'][seed % 3], 'x')
                rules.append(Rule(tuple(body), head))
                results = [runner(rules) for runner in runners]
                assert_result_same(*results)
                assert runners[0].stats == runners[1].stats
        finally:
            for runner in runners:
                runner.close()


def test_cross_attribute_singletons_and_missing_values_are_not_dropped(tmp_path):
    context = dict(docs=[Document(str(i)) for i in range(3)], labels=['A'],
        gamma=RefGamma(np.array([[1], [0], [0]], bool), np.zeros((3, 1), bool)),
        attribute_values={'mtd.a': ['first', None, 'third'], 'mtd.b': [None, 'first', 'other']})
    rule = Rule((Doc('x'), Doc('y'), AttrEqual('x', 'mtd.a', 'y', 'mtd.b'), Label('x', 'A')),
                LabelCompare('x', 'eq', 'y'))
    runner = make_selection_runner(context, tmp_path)
    try:
        result = runner([rule])
        assert result.positive_facts[:, 0].tolist() == [True, True, False]
        assert runner.stats['attribute_blocks_matched'] > 0
    finally:
        runner.close()


def test_policy_changes_bindings_and_rejects_mutation(tmp_path):
    gamma = np.zeros((1, 1), bool)
    context = dict(docs=[Document('a')], labels=['A'], gamma=RefGamma(gamma, gamma))
    a = IndexedChaseRunner(context, tmp_path / 'old', compact_attribute_index=False)
    b = IndexedChaseRunner(context, tmp_path / 'new', compact_attribute_index=True)
    try:
        assert a.context_id != b.context_id
        b.compact_attribute_index = False
        with pytest.raises(RuntimeError, match='attribute index policy'):
            b([])
    finally:
        a.close(); b.close()
