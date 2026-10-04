"""Complete independent-source closures may omit Gamma-refuted derivations."""
import copy
import numpy as np
import pytest
from types import SimpleNamespace

from loris._engine.rdl_reference import RefGamma
from loris._storage.discovery_factory import make_discovery_runner
from loris.labeling.unknown_incremental import retaining_chase
from loris.labeling.reliability import (
    direct_rule_weights, relation_scoring, RETAINED_GAMMA_WINS_SEMANTICS,
)
from tests.test_quarantine_reliability import label_rule
from tests.test_relation_rule_reliability import ctx_for, relation, scored_artifact


def artifact(rules, counts):
    result = scored_artifact(rules, counts)
    result['chase_semantics'] = RETAINED_GAMMA_WINS_SEMANTICS
    result['relation_scoring'] = relation_scoring(RETAINED_GAMMA_WINS_SEMANTICS)
    return result


@pytest.mark.parametrize('sign', ['pos', 'neg'])
def test_refuted_direct_head_is_not_missing_closure_or_a_support(sign, tmp_path):
    ctx = ctx_for(('source',))
    pos = np.zeros((1, 2), bool); neg = pos.copy()
    (neg if sign == 'pos' else pos)[0, 0] = True
    ctx['gamma'] = RefGamma(pos, neg)
    rules = (label_rule('source', 'A', sign), label_rule('source', 'B'))
    runner = make_discovery_runner(ctx, tmp_path/'cache')
    try:
        closure = retaining_chase(runner, rules, ctx['gamma'], gamma_wins=True)
        assert closure.status == 'fixpoint'
        weights, audit = direct_rule_weights(runner, rules, closure, ctx['gamma'], artifact(rules, [(8, 8), (8, 4)]))
        assert np.allclose(weights, [[1., .5]])
        assert [r['distinct_supported_labels'] for r in audit['rules']] == [0, 1]
        missing = copy.copy(closure)
        missing.positive_facts = closure.positive_facts.copy()
        missing.positive_facts[0, 1] = False
        with pytest.raises(ValueError, match='absent'):
            direct_rule_weights(runner, rules, missing, ctx['gamma'], artifact(rules, [(8, 8), (8, 4)]))
    finally:
        runner.close()


def test_valid_negative_gamma_head_keeps_its_direct_support(tmp_path):
    ctx = ctx_for(('source',))
    pos = np.zeros((1, 2), bool); neg = pos.copy(); neg[0, 0] = True
    ctx['gamma'] = RefGamma(pos, neg)
    rules = (label_rule('source', 'A', 'neg'),)
    runner = make_discovery_runner(ctx, tmp_path/'cache')
    try:
        closure = retaining_chase(runner, rules, ctx['gamma'], gamma_wins=True)
        weights, audit = direct_rule_weights(runner, rules, closure, ctx['gamma'], artifact(rules, [(8, 4)]))
        assert weights[0, 0] == 1.
        assert audit['rules'][0]['distinct_supported_labels'] == 1
    finally:
        runner.close()


@pytest.mark.parametrize('trusted_tail', [False, True])
def test_relation_attribution_passes_protected_intermediate_but_never_counts_it(tmp_path, trusted_tail):
    ctx = ctx_for(('source', 'middle', 'target'))
    pos = np.zeros((3, 2), bool); neg = pos.copy()
    pos[0, 0] = True; neg[1, 0] = True
    ctx['gamma'] = RefGamma(pos, neg, ((1, 2),) if trusted_tail else ())
    rules = (relation('source', 'middle'),)
    counts = [(8, 1)]
    if not trusted_tail:
        rules += (relation('middle', 'target'),)
        counts.append((8, 8))
    runner = make_discovery_runner(ctx, tmp_path/'cache')
    try:
        closure = retaining_chase(runner, rules, ctx['gamma'], gamma_wins=True)
        assert closure.positive_facts[:, 0].tolist() == [True, False, True]
        assert closure.negative_facts[:, 0].tolist() == [False, True, False]
        weights, audit = direct_rule_weights(runner, rules, closure, ctx['gamma'], artifact(rules, counts))
        assert np.allclose(weights[:, 0], [1, 1, .2 if trusted_tail else .9])
        assert [r['distinct_supported_labels'] for r in audit['rules']] == ([1] if trusted_tail else [0, 1])
        assert audit['unsupported_nonGamma_labels'] == 0
    finally:
        runner.close()


def test_trusted_only_transport_keeps_source_q_across_protected_cell(tmp_path):
    ctx = ctx_for(('source', 'middle', 'target'))
    pos = np.zeros((3, 2), bool); neg = pos.copy(); neg[1, 0] = True
    ctx['gamma'] = RefGamma(pos, neg, ((0, 1), (1, 2)))
    rules = (label_rule('source', 'A'),)
    runner = make_discovery_runner(ctx, tmp_path/'cache')
    try:
        closure = retaining_chase(runner, rules, ctx['gamma'], gamma_wins=True)
        weights, audit = direct_rule_weights(runner, rules, closure, ctx['gamma'], artifact(rules, [(8, 4)]))
        assert np.allclose(weights[:, 0], [.5, 1, .5])
        assert audit['rules'][0]['distinct_supported_labels'] == 2
    finally:
        runner.close()


def test_protected_relation_sources_do_not_decode_reflexive_document_rows():
    from loris._engine.relation_rows import RelationRows
    from loris.labeling.reliability import _relation_source_bits

    class NoWideRows(RelationRows):
        def __getitem__(self, key):
            raise AssertionError('compact relation access must not decode wide rows')

    n = 100000
    incoming = NoWideRows(n, reflexive=True)
    incoming[99999] = (1 << 99999) | 1
    positive = [0] * n
    positive[0] = 1
    blocked = [3] * n
    expected = positive.copy()
    expected[-1] = 1
    assert _relation_source_bits(SimpleNamespace(inc=incoming), positive, blocked) == expected
