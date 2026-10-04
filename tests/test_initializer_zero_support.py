import gzip
import json

import pytest

from loris.discovery.search import initialization
from tests.test_discovery_initialization_acceleration import fixture
from tests.test_initializer_gzip_archive import OLD


@pytest.mark.parametrize('seed', [0, 7, 42])
@pytest.mark.parametrize('archive', ['all', 'evaluated_with_analytic_skips'])
def test_zero_support_keeps_every_record_and_startup_configuration(tmp_path, seed, archive):
    space, prior = fixture(seed=seed, labels=35)
    options = dict(initializer_slots=30, label_rotation='seeded_cluster_rotation',
                   rotation_seed=19, cluster_position=3, initializer_candidate_archive=archive)
    try:
        expected = OLD.build_fit_initialized_space(space, prior, output_dir=tmp_path / 'old', **options)
        actual = initialization.build_fit_initialized_space(space, prior, output_dir=tmp_path / 'new', **options)
        filename = 'ALL_FIT_CANDIDATES.jsonl' if archive == 'all' else 'EVALUATED_FIT_CANDIDATES.jsonl'
        assert gzip.decompress((tmp_path / 'new' / (filename + '.gz')).read_bytes()) == (tmp_path / 'old' / filename).read_bytes()
        for filename in ['SEEDS.json', 'PARAMETER_AXIS.json']:
            assert (tmp_path / 'new' / filename).read_bytes() == (tmp_path / 'old' / filename).read_bytes()
        expected_complete = json.loads((tmp_path / 'old' / 'COMPLETE.json').read_text())
        actual_complete = json.loads((tmp_path / 'new' / 'COMPLETE.json').read_text())
        assert expected_complete['counts'] == actual_complete['counts']
        assert expected.initial_configurations(36, seed=19) == actual.initial_configurations(36, seed=19)
    finally:
        prior.close()


def test_complete_archive_keeps_empty_scope_thresholds_and_static_conflicts(tmp_path):
    space, prior = fixture(seed=7)
    try:
        output = tmp_path / 'all'
        initialization.build_fit_initialized_space(
            space, prior, output_dir=output, initializer_slots=30,
            initializer_candidate_archive="all"
        )
        with gzip.open(output / 'ALL_FIT_CANDIDATES.jsonl.gz', 'rt') as stream:
            records = [json.loads(line) for line in stream]
        absent = [record for record in records if record['pattern_index'] == 2]
        assert absent and all(record['fit_tp'] == record['fit_fp'] == 0 for record in absent)
        assert all(record['reason'] == 'zero_support' for record in absent)
        assert any(record['thresholds'] == [{'alias': 'limited', 'eta': .1, 'raw_boundary': .1}]
                   for record in absent)
        assert any(record['reason'] == 'boundary_outside_domain' for record in records)
        assert any(record['reason'] == 'direct_root_conflict' and record['model_indices'] == [2]
                   for record in records)
    finally:
        prior.close()
