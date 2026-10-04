"""Immutable storage reuse, with the same complete data/model interpretation."""
import numpy as np
import pytest

from loris._engine.array_cache import _readonly
from loris._engine.rdl_reference import Doc, ML, Label, Rule, ReferenceChase
from loris._engine.rlearner_config import ModelSpec, ParameterDomain
from loris._learning.structured_bo import build_fit_witness_prior, _array_id
from loris._storage.selection_execution import make_selection_runner
from loris.discovery.runner import _shared_fit_arrays
from loris.discovery.search.configuration import build_config_space
from tests.test_discovery_runner import inputs


@pytest.mark.parametrize('write_protected', [False, True])
def test_readonly_reuses_only_immutable_backing(write_protected):
    caller = np.arange(12, dtype=np.float64).reshape(3, 4)
    caller.setflags(write=not write_protected)
    frozen = _readonly(caller)
    again = _readonly(frozen)
    assert np.shares_memory(frozen, again)
    assert not np.shares_memory(caller, frozen)
    caller.setflags(write=True)
    caller[:] = -1
    np.testing.assert_array_equal(frozen, np.arange(12).reshape(3, 4))
    with pytest.raises(ValueError):
        frozen.setflags(write=True)
    # A read-only memoryview of mutable storage is not an immutable owner.
    buffer = bytearray(np.arange(3, dtype=np.int64).tobytes())
    view = np.frombuffer(memoryview(buffer).toreadonly(), dtype=np.int64)
    safe = _readonly(view)
    buffer[:] = bytes(len(buffer))
    np.testing.assert_array_equal(safe, np.arange(3))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_runtime_snapshots_caller_and_preserves_explicit_static_interpretation(tmp_path, dtype):
    context = inputs()['context']
    context['model_probabilities']['m'] = context['model_probabilities']['m'].astype(dtype)
    caller = context['model_probabilities']['m']
    before = caller.copy()
    runner = make_selection_runner(context, tmp_path/'initial')
    try:
        assert np.shares_memory(runner.probabilities['m'], runner.static_probabilities['m'])
        assert not np.shares_memory(caller, runner.probabilities['m'])
        caller[:] = .01
        np.testing.assert_array_equal(runner.probabilities['m'], before)
        with pytest.raises(ValueError):
            runner.probabilities['m'].setflags(write=True)
        # A new dynamic version must retain the initial static probabilities.
        updated = dict(context, static_model_probabilities=dict(runner.static_probabilities))
        new = make_selection_runner(updated, tmp_path/'updated')
        try:
            dynamic = Rule((Doc('x'), ML('x','m','A',.5,True)), Label('x','A'))
            static = Rule((Doc('x'), ML('x','m','A',.5,False)), Label('x','A'))
            assert not new([dynamic]).positive_facts[2,0]
            assert new([static]).positive_facts[2,0]
            np.testing.assert_array_equal(new.static_probabilities['m'], before)
            assert np.shares_memory(new.static_probabilities['m'], runner.static_probabilities['m'])
        finally:
            new.close()
    finally:
        runner.close()


def test_reference_dynamic_mapping_replacement_leaves_static_frozen():
    caller = np.array([[.9]], dtype=np.float64)
    reference = ReferenceChase(['A'], model_probabilities={'m':caller})
    assert np.shares_memory(reference.model_probabilities['m'], reference.static_model_probabilities['m'])
    reference.model_probabilities['m'] = np.array([[.1]])
    caller[:] = .2
    assert reference.static_model_probabilities['m'][0,0] == .9


def test_cluster_priors_share_full_fit_values_without_mutable_state():
    context = inputs()['context']
    fit = context['fit_indices']
    shared = _shared_fit_arrays(context, fit, ['m'])
    expected = context['model_probabilities']['m'][fit].copy()
    priors = []
    for cluster in ['0','1']:
        space = build_config_space(cluster_id=cluster, labels=context['labels'], patterns=(),
            models=(ModelSpec('m',True,ParameterDomain(low=0.,high=1.)),), relations=())
        prior = build_fit_witness_prior(space,
            docs=[context['docs'][i] for i in fit], row_ids=[context['row_ids'][i] for i in fit],
            row_roles=['source_fit']*len(fit), labels=context['labels'],
            gamma_pos=shared['positive'], gamma_neg=shared['negative'],
            model_probabilities={'m':shared['probabilities']['m']},
            model_decisions={'m':shared['decisions']['m']}, fit_head_labels=shared['head_labels'])
        priors.append(prior)
    left, right = priors
    assert left.probabilities is not right.probabilities
    assert left.probabilities is not shared['probabilities']
    assert np.shares_memory(left.probabilities['m'], right.probabilities['m'])
    assert np.shares_memory(left.pos, right.pos)
    assert _array_id(left.probabilities['m']) == _array_id(expected)
    context['model_probabilities']['m'][:] = 0
    np.testing.assert_array_equal(right.probabilities['m'], expected)
    with pytest.raises(ValueError):
        left.probabilities['m'][0,0] = 0
    # Per-cluster bookkeeping remains separately owned.
    left.probabilities['m'] = _readonly(np.zeros_like(expected))
    np.testing.assert_array_equal(right.probabilities['m'], expected)
