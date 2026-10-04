"""Real random-joint proposals, closure admission, exact recovery and archives."""
import copy
import json

import numpy as np
import pytest

from loris.discovery import runner
from loris.discovery.__main__ import main
from loris.discovery.archive import archive_index, status
from loris.discovery.manifest import load_inputs
from tests.test_discovery_runner import inputs, semantic_trials
from tests.test_discovery_variants import discovery_manifest


def random_inputs():
    return dict(inputs(), budget_per_cluster=16, initializer_slots=0,
                proposal_strategy='random_joint', admission_order='random_consistent')


def read(path):
    return json.loads(path.read_text())


def test_random_joint_ignores_validation_accuracy_and_fit_initializers(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('random joint must not build/read a fit-conditioned prior or initializer')
    monkeypatch.setattr(runner, 'build_fit_witness_prior', forbidden)
    monkeypatch.setattr(runner, 'build_fit_initialized_space', forbidden)
    a = random_inputs()
    b = copy.deepcopy(a)
    b['context']['truth'] = ~b['context']['truth']
    b['context']['source_gold']['val_ctrl'] = b['context']['truth']
    out, changed = tmp_path/'original', tmp_path/'changed_truth'
    done = runner.run_discovery(output=out, **a)
    runner.run_discovery(output=changed, **b)
    left, right = semantic_trials(out), semantic_trials(changed)
    assert [(p, r) for p, r, score in left] == [(p, r) for p, r, score in right]
    assert read(out/'learning/SELECTED.json')['rules'] == read(changed/'learning/SELECTED.json')['rules']
    assert done['proposal_strategy'] == 'random_joint'
    assert done['admission_order'] == 'random_consistent'
    assert read(out/'initialization/0/COMPLETE.json')['seed_count'] == 0
    assert read(out/'learning/cluster_00/initial_configurations.json') == []
    protocol = read(out/'learning/protocol.json')
    assert protocol['priors'] == [None]
    assert protocol['validation_accuracy_used_for_selection'] is False
    records = [read(p) for p in sorted((out/'learning/cluster_00').glob('trial_*.json'))]
    # Both contextual and ML choices actually vary on the complete joint axis.
    for family in ('pattern', 'model'):
        counts = [r['actual_body_indicators_by_family'][family] for r in records]
        assert max(counts) > min(counts)
    receipts = [read(p) for p in sorted((out/'learning/cluster_00/admission').glob('candidate_*.json'))]
    accepted = [r['decision'] for r in receipts if r['decision']['accepted']]
    assert accepted
    assert any(d['delta_macro_f1'] <= 0 for d in accepted)
    assert all(d['score_used_for_selection'] is False for d in accepted)
    assert status(out)['status'] == 'complete_verified_identity'
    archive = archive_index(out, tmp_path/'archive')
    assert archive['clusters'][0]['all_sequential_admissions_verified']
    assert runner.run_discovery(output=out, resume=True, **a) == done
    with pytest.raises(ValueError):
        runner.run_discovery(output=out, resume=True, **dict(a,
            proposal_strategy='accuracy_guided', admission_order='standalone_score_desc'))


def test_random_joint_checkpoint_recovery_preserves_future_proposals(tmp_path, monkeypatch):
    kw = random_inputs()
    actual = runner.run_rich_cluster_search
    def interrupt(**args):
        callback = args['on_progress']
        def progress(record):
            callback(record)
            if record['complete'] == 3:
                raise RuntimeError('controlled random interruption')
        return actual(**dict(args, on_progress=progress))
    monkeypatch.setattr(runner, 'run_rich_cluster_search', interrupt)
    out = tmp_path/'interrupted'
    with pytest.raises(RuntimeError, match='controlled random interruption'):
        runner.run_discovery(output=out, **kw)
    monkeypatch.setattr(runner, 'run_rich_cluster_search', actual)
    resumed = runner.run_discovery(output=out, resume=True, **kw)
    fresh = runner.run_discovery(output=tmp_path/'fresh', **kw)
    assert semantic_trials(out) == semantic_trials(tmp_path/'fresh')
    assert resumed['final_positive_metrics'] == fresh['final_positive_metrics']
    with np.load(out/'FACTS.npz') as a, np.load(tmp_path/'fresh/FACTS.npz') as b:
        assert all(np.array_equal(a[k], b[k]) for k in ('pos', 'neg'))


def test_random_joint_cli_is_explicit_and_rejects_hybrid_defaults(tmp_path):
    manifest = discovery_manifest(tmp_path)
    with pytest.raises(ValueError, match='initializer_slots=0'):
        load_inputs(manifest, dict(proposal_strategy='random_joint'))
    out = tmp_path/'random'
    assert main(['learn-rules', '--manifest', str(manifest), '--output', str(out),
                 '--proposal-strategy', 'random_joint', '--initializer-slots', '0',
                 '--admission-order', 'random_consistent']) == 0
    assert read(out/'COMPLETE.json')['proposal_strategy'] == 'random_joint'
    with pytest.raises(ValueError, match='requires random_joint'):
        load_inputs(manifest, dict(admission_order='random_consistent'))
