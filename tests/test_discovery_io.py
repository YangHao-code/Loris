"""Persistence changes must preserve real sampler trajectories after interruption."""
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

from loris._engine import discovery_io, rlearner
from loris.discovery import runner
from tests.test_discovery_runner import inputs, semantic_trials


def test_restartable_sqlite_policy_on_every_optuna_connection(tmp_path, monkeypatch):
    monkeypatch.setenv('LORIS_DISCOVERY_IO', 'restartable')
    store = discovery_io.optuna_storage(tmp_path / 'optuna.sqlite')
    for _ in range(2):
        with store.engine.connect() as db:
            assert db.exec_driver_sql('PRAGMA journal_mode').scalar() == 'wal'
            assert db.exec_driver_sql('PRAGMA synchronous').scalar() == 1
        store.engine.dispose()
    calls = []
    monkeypatch.setattr(discovery_io.os, 'fsync', lambda fd: calls.append(fd))
    rlearner.atomic_json(tmp_path / 'hot.json', {'ok': True})
    assert not calls
    monkeypatch.setenv('LORIS_DISCOVERY_IO', 'durable')
    rlearner.atomic_json(tmp_path / 'cold.json', {'ok': True})
    assert len(calls) == 1


@pytest.mark.parametrize('boundary', ['receipt', 'tell', 'sampler'])
def test_restartable_resume_replays_exact_next_candidates(tmp_path, monkeypatch, boundary):
    monkeypatch.setenv('LORIS_DISCOVERY_IO', 'restartable')
    kw = inputs()
    kw['budget_per_cluster'] = 8
    target = tmp_path / 'interrupted'
    original = rlearner.atomic_json
    fired = []

    def interrupted(path, value):
        path = Path(path)
        hit = (not fired and ((boundary == 'receipt' and path.name == 'trial_00001.json')
                             or (boundary != 'receipt' and path.name == 'sampler.json'
                                 and value.get('completed_trial') == 1)))
        if hit and boundary == 'tell':
            fired.append(True)
            raise RuntimeError('injected persistence interruption')
        original(path, value)
        if hit:
            fired.append(True)
            raise RuntimeError('injected persistence interruption')

    monkeypatch.setattr(rlearner, 'atomic_json', interrupted)
    with pytest.raises(RuntimeError, match='injected persistence interruption'):
        runner.run_discovery(output=target, **kw)
    checker_path = Path(__file__).resolve().parents[1] / 'scripts/discovery_checkpoint_state.py'
    checker_spec = importlib.util.spec_from_file_location('checkpoint_state', checker_path)
    checker = importlib.util.module_from_spec(checker_spec)
    checker_spec.loader.exec_module(checker)
    checked = checker.check(target)
    assert checked['safe'] and checked['live_sample_only']
    pending = target / 'learning/cluster_00/pending.json'
    if boundary == 'receipt':
        saved = pending.read_bytes()
        pending.unlink()
        unsafe = checker.check(target)
        assert not unsafe['safe'] and 'unsealed RUNNING' in unsafe['clusters'][0]['error']
        pending.write_bytes(saved)
    monkeypatch.setattr(rlearner, 'atomic_json', original)
    resumed = runner.run_discovery(output=target, resume=True, **kw)
    fresh = tmp_path / 'fresh'
    baseline = runner.run_discovery(output=fresh, **kw)
    assert semantic_trials(target) == semantic_trials(fresh)
    assert resumed['final_positive_metrics'] == baseline['final_positive_metrics']
    assert not (target / 'learning/cluster_00/pending.json').exists()


def test_offline_hot_relocation_preserves_exact_resume(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts/discovery_hot_storage.py'
    spec = importlib.util.spec_from_file_location('hot_storage', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kw = inputs()
    target = tmp_path / 'interrupted'
    original = runner.run_rich_cluster_search

    def interrupt(**args):
        callback = args['on_progress']
        def progress(record):
            callback(record)
            if record['complete'] == 2:
                raise RuntimeError('stop for relocation')
        return original(**dict(args, on_progress=progress))

    monkeypatch.setattr(runner, 'run_rich_cluster_search', interrupt)
    with pytest.raises(RuntimeError, match='stop for relocation'):
        runner.run_discovery(output=target, **kw)
    monkeypatch.setattr(runner, 'run_rich_cluster_search', original)
    before = semantic_trials(target)
    report = module.relocate(target, tmp_path / 'ssd')
    assert json.loads(report.read_text())['status'] == 'relocated_exact_resume_required'
    assert (target / 'learning/cluster_00').is_symlink()
    assert semantic_trials(target) == before
    runner.run_discovery(output=target, resume=True, **kw)
    runner.run_discovery(output=tmp_path / 'fresh', **kw)
    assert semantic_trials(target) == semantic_trials(tmp_path / 'fresh')
