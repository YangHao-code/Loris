"""CUDA launch environment boundaries; the selection algorithm is unchanged."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from loris.discovery import selector


@pytest.fixture(autouse=True)
def restore_workspace_environment():
    previous = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    yield
    if previous is None:
        os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    else:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = previous


def fake_cuda(monkeypatch, initialized):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: initialized)))


def test_missing_cuda_environment_set_before_import(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    monkeypatch.delitem(sys.modules, 'torch', raising=False)
    result = selector.configure_workspace('cuda:0')
    assert result['effective'] == ':4096:8'
    assert result['cuda_initialized_before'] is False
    assert 'torch' not in sys.modules


@pytest.mark.parametrize('workspace', selector.WORKSPACES)
def test_existing_valid_environment_is_preserved(monkeypatch, workspace):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', workspace)
    fake_cuda(monkeypatch, True)
    assert selector.configure_workspace('cuda:0')['action'] == 'preserved'


def test_explicit_early_override(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    fake_cuda(monkeypatch, False)
    assert selector.configure_workspace('cuda:1', ':4096:8')['effective'] == ':4096:8'


@pytest.mark.parametrize('previous', [None, ':16:8'])
def test_late_workspace_changes_rejected(monkeypatch, previous):
    if previous is None:
        monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    else:
        monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', previous)
    fake_cuda(monkeypatch, True)
    with pytest.raises(RuntimeError, match='already initialized'):
        selector.configure_workspace('cuda:0', ':4096:8')
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == previous


@pytest.mark.parametrize('value', ['', 'invalid', ':4096:2'])
def test_invalid_environment_is_not_silently_repaired(monkeypatch, value):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', value)
    fake_cuda(monkeypatch, False)
    with pytest.raises(ValueError, match='existing'):
        selector.configure_workspace('cuda:0', ':4096:8')
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == value


def test_cpu_never_touches_cuda_or_environment(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', 'unused_on_cpu')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    assert selector.configure_workspace('cpu', ':16:8')['effective'] == 'unused_on_cpu'


def test_failure_receipt_outside_output_no_precreation(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'protocol': {'device': 'cuda:0'}}))
    output = tmp_path / 'selector'
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', 'invalid')
    with pytest.raises(ValueError, match='existing'):
        selector.train_selector(manifest, output)
    assert not output.exists()
    launches = list(tmp_path.glob('selector.selector_launches/*'))
    assert len(launches) == 1
    assert json.loads((launches[0] / 'STARTED.json').read_text())['selector_algorithm_modified'] is False
    assert json.loads((launches[0] / 'RESULT.json').read_text())['status'] == 'failed'


def test_delegates_selector_resume_without_precreating_output(tmp_path, monkeypatch):
    from loris.discovery import selector_backend
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'protocol': {'device': 'cpu'}}))
    output = tmp_path / 'selector'
    calls = []
    def delegate(path, directory, *, resume):
        assert not directory.exists()
        calls.append((path, directory, resume))
        raise RuntimeError('original resume rejects nonexistent output')
    monkeypatch.setattr(selector_backend, 'run_selector_manifest', delegate)
    with pytest.raises(RuntimeError, match='original resume'):
        selector.train_selector(manifest, output, resume=True)
    assert calls == [(manifest.resolve(), output.resolve(), True)]
