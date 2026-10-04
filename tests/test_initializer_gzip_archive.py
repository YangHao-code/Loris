"""Compare real initializers; only the archived stream may change representation."""
import gzip
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys
import time

import pytest

from loris.discovery import runner
from loris.discovery.archive import _semantic_files
from tests.test_discovery_initialization_acceleration import fixture, make_options


HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location('loris.discovery.search.' + name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OLD = load('_gzip_probe_reference', 'fixtures/initialization_before_gzip_v2.py')
from loris.discovery.search import initialization as NEW


def candidate_name(mode):
    return ('ALL_FIT_CANDIDATES' if mode == 'all' else 'EVALUATED_FIT_CANDIDATES') + '.jsonl'


@pytest.mark.parametrize('mode', ['all', 'evaluated_with_analytic_skips'])
@pytest.mark.parametrize('slots', [0, 3])
@pytest.mark.parametrize('rotation', ['axis', 'seeded_cluster_rotation'])
def test_lossless_bytes_seeds_axis_order_fallback_and_resume(tmp_path, mode, slots, rotation):
    space, prior = fixture(seed=7)
    before = space.manifest()
    options = dict(initializer_slots=slots, label_rotation=rotation,
                   rotation_seed=19, cluster_position=2, initializer_candidate_archive=mode)
    try:
        old = OLD.build_fit_initialized_space(space, prior, output_dir=tmp_path/'old', **options)
        new = NEW.build_fit_initialized_space(space, prior, output_dir=tmp_path/'new', **options)
        name = candidate_name(mode)
        raw = (tmp_path/'old'/name).read_bytes()
        zipped = tmp_path/'new'/(name + '.gz')
        assert gzip.decompress(zipped.read_bytes()) == raw
        assert not (tmp_path/'new'/name).exists()
        for name in ['SEEDS.json', 'PARAMETER_AXIS.json']:
            assert (tmp_path/'old'/name).read_bytes() == (tmp_path/'new'/name).read_bytes()
        if mode != 'all':
            assert (tmp_path/'old/ANALYTICALLY_SKIPPED.json').read_bytes() == (tmp_path/'new/ANALYTICALLY_SKIPPED.json').read_bytes()
        for seed in (0, 42):
            assert old.initial_configurations(35, seed=seed) == new.initial_configurations(35, seed=seed)
        assert space.manifest() == before
        c = json.loads((tmp_path/'new/COMPLETE.json').read_text())
        original = json.loads((tmp_path/'old/COMPLETE.json').read_text())
        assert c['counts'] == original['counts']
        assert c['seed_count'] == original['seed_count']
        assert c['BO_calls'] == c['Chase_calls'] == 0
        assert c['initialization_options'] == original['initialization_options']
        assert c['space_manifest_id'] == original['space_manifest_id']
        assert c['fit_witness_id'] == original['fit_witness_id']
        record = next(r for r in c['files'] if Path(r['path']) == zipped)
        assert record['sha256'] == hashlib.sha256(zipped.read_bytes()).hexdigest()
        p = json.loads((tmp_path/'new/PROTOCOL.json').read_text())
        assert p['candidate_rows'] == zipped.name
        assert p['candidate_encoding']['compression'] == 'gzip'
        snapshot = {p.name: p.read_bytes() for p in (tmp_path/'new').iterdir()}
        resumed = NEW.build_fit_initialized_space(space, prior, output_dir=tmp_path/'new', **options)
        assert resumed.initial_configurations(35, seed=42) == old.initial_configurations(35, seed=42)
        assert snapshot == {p.name: p.read_bytes() for p in (tmp_path/'new').iterdir()}
    finally:
        prior.close()


@pytest.mark.parametrize('mode', ['all', 'evaluated_with_analytic_skips'])
def test_gzip_output_deterministic_and_tampering_rejected(tmp_path, mode):
    space, prior = fixture()
    try:
        options = dict(initializer_slots=3, initializer_candidate_archive=mode)
        for path in ['one', 'two']:
            NEW.build_fit_initialized_space(space, prior, output_dir=tmp_path/path, **options)
        name = candidate_name(mode) + '.gz'
        a, b = tmp_path/'one'/name, tmp_path/'two'/name
        assert a.read_bytes() == b.read_bytes()
        with a.open('ab') as stream:
            stream.write(b'changed')
        with pytest.raises(ValueError, match='artifact changed'):
            NEW.FitInitializedSpace(space, tmp_path/'one')
    finally:
        prior.close()


def test_utf8_stream_and_exclusive_output(tmp_path):
    path = tmp_path/'utf8.jsonl.gz'
    lines = ['{"label":"中文","pattern":"é\\\\b"}\n', '{"label":"İ ı ſ K 😀"}\n']
    with NEW._candidate_writer(path) as stream:
        for line in lines:
            stream.write(line)
    assert gzip.decompress(path.read_bytes()) == ''.join(lines).encode('utf-8')
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        with NEW._candidate_writer(path):
            pass
    assert path.read_bytes() == original


def test_archive_inventory_keeps_gzip_initialization_files(tmp_path):
    folder = tmp_path/'initialization/0'
    folder.mkdir(parents=True)
    path = folder/'EVALUATED_FIT_CANDIDATES.jsonl.gz'
    with NEW._candidate_writer(path) as stream:
        stream.write('{}\n')
    assert path in list(_semantic_files(tmp_path))


@pytest.mark.skipif('fork' not in multiprocessing.get_all_start_methods(), reason='fork unavailable')
def test_real_parallel_driver_supports_gzip_without_bo(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'build_fit_initialized_space', NEW.build_fit_initialized_space)
    monkeypatch.setattr(runner, 'FitInitializedSpace', NEW.FitInitializedSpace)
    pairs = [fixture(seed=42+i, cluster=str(i), cache_dir=tmp_path/'cache'/str(i)) for i in range(2)]
    spaces, priors = map(list, zip(*pairs))
    try:
        initialized = runner.initialize_spaces(spaces, priors, ['0','1'], tmp_path,
            make_options(2, 'evaluated_with_analytic_skips'), time.monotonic())
        assert [s.cluster_id for s in initialized] == ['0','1']
        for space, prior, new in zip(spaces, priors, initialized):
            old = OLD.build_fit_initialized_space(space, prior, output_dir=tmp_path/('reference_' + space.cluster_id),
                initializer_slots=30, label_rotation='seeded_cluster_rotation', rotation_seed=42,
                cluster_position=int(space.cluster_id), initializer_candidate_archive='evaluated_with_analytic_skips')
            assert old.initial_configurations(35, seed=42) == new.initial_configurations(35, seed=42)
            raw = tmp_path/('reference_' + space.cluster_id)/candidate_name('evaluated_with_analytic_skips')
            zipped = tmp_path/'initialization'/space.cluster_id/(raw.name+'.gz')
            assert gzip.decompress(zipped.read_bytes()) == raw.read_bytes()
    finally:
        for prior in priors:
            prior.close()
