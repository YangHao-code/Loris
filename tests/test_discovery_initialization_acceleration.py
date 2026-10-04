"""Exact initializer receipts and independent-process regression checks."""
import copy
import gzip
import json
import multiprocessing
import time

import numpy as np
import pytest

from loris._engine.compact_storage import ParameterCodec
from loris._engine.rlearner_config import ModelSpec, ParameterDomain, PatternSpec
from loris.discovery import runner
from loris.discovery.search import initialization
from loris.discovery.search.configuration import build_config_space
from loris.discovery.search.sampler import build_fit_witness_prior
from loris.document import Document
from loris.predicates import MatchPredicate, FreqPredicate


def fixture(seed=42, labels=7, cluster='0', cache_dir=None):
    rng = np.random.RandomState(seed)
    axis = tuple('L%02d' % i for i in range(labels))
    docs = [Document(cnt=s) for s in ('red red', 'blue', 'red blue', '', 'green', 'red')]
    y = rng.rand(len(docs), labels) > .5
    y[:, 0] = True
    if labels > 1:
        y[:, 1] = False
    patterns = [PatternSpec('red', MatchPredicate('cnt', 'red')),
                PatternSpec('blue', MatchPredicate('cnt', 'blue')),
                PatternSpec('absent', MatchPredicate('cnt', 'never_present')),
                PatternSpec('freq', FreqPredicate('cnt', 'red', '>=', 1),
                            frequency_domain=ParameterDomain(low=0, high=3, integer=True),
                            frequency_ops=('>=', '<'))]
    space = build_config_space(cluster_id=cluster, labels=axis, patterns=patterns,
        models=(ModelSpec('dynamic', True, ParameterDomain(low=0, high=1)),
                ModelSpec('limited', True, ParameterDomain(values=(.1, .5, .8))),
                ModelSpec('fixed', False)))
    probabilities = rng.choice([0., .1, .5, .8, 1.], size=y.shape).astype(float)
    fixed = y.copy()
    if labels > 2:
        fixed[:, 2] = ~fixed[:, 2]
    prior = build_fit_witness_prior(space, docs=docs, row_ids=list(range(len(docs))),
        row_roles=['source_fit'] * len(docs), labels=axis, gamma_pos=y, gamma_neg=~y,
        model_probabilities={'dynamic': probabilities, 'limited': np.where(y, .8, 1.)},
        model_decisions={'fixed': fixed}, attribute_values={}, attribute_presence={},
        cache_dir=cache_dir)
    return space, prior


class DenseParameterBuffer:
    """Independent original full-axis encode/allocate contract for comparison."""
    def __init__(self, space, codec, variable):
        self.space, self.codec, self.variable = space, codec, variable

    def reset(self, label):
        self.params = self.space.empty_parameters('add', label=label, variable=self.variable)
        return self.params

    def encode(self):
        return self.codec.encode(self.params)


@pytest.mark.parametrize('seed', [0, 7, 42])
@pytest.mark.parametrize('rotation', ['axis', 'seeded_cluster_rotation'])
def test_full_candidate_and_seed_bytes_match_dense_factory(tmp_path, monkeypatch, seed, rotation):
    space, prior = fixture(seed, labels=35)
    kwargs = dict(initializer_slots=30, label_rotation=rotation, rotation_seed=42,
                  cluster_position=3, initializer_candidate_archive="all")
    try:
        initialization.build_fit_initialized_space(space, prior, output_dir=tmp_path/'fast', **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(initialization, '_ReusableParameterBuffer', DenseParameterBuffer)
            patch.setattr(type(space), '_instantiate_compact',
                          lambda self, value: self.instantiate(self._compact_codec.decode(value)))
            initialization.build_fit_initialized_space(space, prior, output_dir=tmp_path/'dense', **kwargs)
        for filename in ['ALL_FIT_CANDIDATES.jsonl.gz', 'SEEDS.json', 'PARAMETER_AXIS.json']:
            assert (tmp_path/'fast'/filename).read_bytes() == (tmp_path/'dense'/filename).read_bytes()
        complete = json.loads((tmp_path/'fast/COMPLETE.json').read_text())
        assert complete['counts']['total'] == 35 * (3 * (4 + 1) + 3)
        assert complete['counts']['boundary_outside_domain'] > 0
        assert complete['counts']['zero_support'] > 0
        assert complete['BO_calls'] == complete['Chase_calls'] == 0
    finally:
        prior.close()


def test_compact_factory_keeps_exact_rules_or_errors_for_startup_draws():
    space, prior = fixture()
    codec = ParameterCodec.from_space(space)
    try:
        for params in space.iter_initial_configurations(80, seed=19):
            answers = []
            for factory, value in [(space.instantiate, params),
                                   (space._instantiate_compact, codec.encode(params))]:
                try:
                    answers.append(('rule', factory(value).to_dict()))
                except Exception as error:
                    answers.append((type(error).__name__, str(error)))
            assert answers[0] == answers[1]
    finally:
        prior.close()


@pytest.mark.parametrize('domain',['rich','no_text','local'])
def test_compact_codec_uses_final_capability_axis(domain):
    from loris.discovery.search.domain import build_config_space as build_domain
    space,prior=fixture()
    try:
        restricted=build_domain(candidate_domain=domain,cluster_id='0',labels=space.labels,
                                patterns=space.patterns,models=space.models)
        codec=ParameterCodec.from_space(restricted)
        assert restricted._compact_codec is None
        for params in restricted.initial_configurations(12,seed=29):
            answers=[]
            for factory,value in [(restricted.instantiate,params),
                                  (restricted._instantiate_compact,codec.encode(params))]:
                try: answers.append(('rule',factory(value).to_dict()))
                except Exception as error: answers.append((type(error).__name__,str(error)))
            assert answers[0]==answers[1]
        assert restricted._compact_codec.space_id==restricted.space_id
        assert restricted._compact_codec.include_keys==tuple(f.include_key for f in restricted.features)
    finally:
        prior.close()


@pytest.mark.parametrize('damage', ['missing_eta', 'bad_eta', 'extra', 'bad_head'])
def test_compact_factory_preserves_active_domain_failures(damage):
    space, prior = fixture()
    try:
        params = space.empty_parameters('add')
        feature = next(f for f in space.features if f.kind == 'model' and space.models[f.index].dynamic)
        params[feature.include_key] = 1
        params[feature.key+'.eta'] = .5
        if damage == 'missing_eta': del params[feature.key+'.eta']
        if damage == 'bad_eta': params[feature.key+'.eta'] = 2.
        if damage == 'extra': params['not_a_parameter'] = 3
        if damage == 'bad_head': params['head.label'] = 'outside_axis'
        encoded = ParameterCodec.from_space(space).encode(params)
        errors=[]
        for factory, value in [(space.instantiate,params),(space._instantiate_compact,encoded)]:
            with pytest.raises(Exception) as error:
                factory(value)
            errors.append((type(error.value).__name__,str(error.value)))
        assert errors[0] == errors[1]
    finally:
        prior.close()


@pytest.mark.parametrize('damage', ['foreign_axis','duplicate','negative','outside','embedded_bit'])
def test_compact_factory_rejects_uncertified_axis(damage):
    space, prior = fixture()
    try:
        codec=ParameterCodec.from_space(space)
        value=codec.encode(space.empty_parameters('add'))
        if damage=='foreign_axis': value['include_axis_sha256']='incorrect'
        if damage=='duplicate': value['on_indices']=[0,0]
        if damage=='negative': value['on_indices']=[-1]
        if damage=='outside': value['on_indices']=[len(space.features)]
        if damage=='embedded_bit': value['active_parameters'][space.features[0].include_key]=1
        with pytest.raises(ValueError): space._instantiate_compact(value)
    finally:
        prior.close()


def make_options(workers, archive='all'):
    return runner.RunOptions(seed=42,budget_per_cluster=30,workers=1,
        initializer_slots=30,label_rotation='seeded_cluster_rotation',initialization_workers=workers,
        initializer_candidate_archive=archive)


@pytest.mark.skipif('fork' not in multiprocessing.get_all_start_methods(),reason='fork unavailable')
@pytest.mark.parametrize('archive',['all','evaluated_with_analytic_skips'])
def test_parallel_initialization_exact_order_masks_and_startup_rng(tmp_path,archive):
    outputs=[]
    for name, workers in [('serial',1),('parallel',2)]:
        out=tmp_path/name
        pairs=[fixture(seed=42+i,cluster=str(i),cache_dir=out/'lexical'/str(i)) for i in range(3)]
        spaces,priors=map(list,zip(*pairs))
        try:
            initialized=runner.initialize_spaces(spaces,priors,['0','1','2'],out,make_options(workers,archive),time.monotonic())
            outputs.append([list(space.iter_initial_configurations(35,seed=42+1009*i))
                            for i,space in enumerate(initialized)])
            assert [space.cluster_id for space in initialized]==['0','1','2']
        finally:
            for prior in priors: prior.close()
    assert outputs[0]==outputs[1]
    for cluster in ['0','1','2']:
        candidate='ALL_FIT_CANDIDATES.jsonl.gz' if archive=='all' else 'EVALUATED_FIT_CANDIDATES.jsonl.gz'
        for filename in [candidate,'SEEDS.json','PARAMETER_AXIS.json']:
            assert (tmp_path/'serial/initialization'/cluster/filename).read_bytes()==(tmp_path/'parallel/initialization'/cluster/filename).read_bytes()


@pytest.mark.skipif('fork' not in multiprocessing.get_all_start_methods(),reason='fork unavailable')
def test_parallel_failure_is_explicit_and_preserves_partial_evidence(tmp_path):
    pairs=[fixture(cluster=str(i),cache_dir=tmp_path/'lexical'/str(i)) for i in range(2)]
    spaces,priors=map(list,zip(*pairs))
    (tmp_path/'initialization/1').mkdir(parents=True)
    try:
        with pytest.raises(RuntimeError,match='initializer failed'):
            runner.initialize_spaces(spaces,priors,['0','1'],tmp_path,make_options(2),time.monotonic())
        failure=json.loads((tmp_path/'initialization/1/WORKER_FAILURE.json').read_text())
        assert failure['type']=='FileExistsError'
        assert not (tmp_path/'initialization/1/COMPLETE.json').exists()
    finally:
        for prior in priors: prior.close()


def test_parallel_refuses_shared_sqlite_before_fork(tmp_path):
    pairs=[fixture(cluster=str(i),cache_dir=tmp_path/'shared') for i in range(2)]
    spaces,priors=map(list,zip(*pairs))
    try:
        with pytest.raises(ValueError,match='independent lexical cache'):
            runner.initialize_spaces(spaces,priors,['0','1'],tmp_path,make_options(2),time.monotonic())
    finally:
        for prior in priors: prior.close()


@pytest.mark.parametrize('seed',[0,1,7,42])
@pytest.mark.parametrize('rotation',['axis','seeded_cluster_rotation'])
@pytest.mark.parametrize('pattern_count',[0,4])
def test_analytic_archive_only_omits_noneligible_preserves_seeds(tmp_path,seed,rotation,pattern_count):
    space,prior=fixture(seed)
    if pattern_count==0:
        # Preserve the same models/roots while constructing a genuinely empty Pi.
        old_prior=prior
        space=build_config_space(cluster_id='0',labels=space.labels,patterns=(),models=space.models)
        prior=build_fit_witness_prior(space,docs=old_prior.docs,row_ids=list(old_prior.rows),
            row_roles=['source_fit']*len(old_prior.rows),labels=space.labels,
            gamma_pos=old_prior.pos,gamma_neg=old_prior.neg,
            model_probabilities=old_prior.probabilities,model_decisions=old_prior.decisions,
            attribute_values={},attribute_presence={})
        old_prior.close()
    try:
        for name,mode in [('full','all'),('skip','evaluated_with_analytic_skips')]:
            initialization.build_fit_initialized_space(space,prior,output_dir=tmp_path/name,
                initializer_slots=3,label_rotation=rotation,rotation_seed=5,cluster_position=2,
                initializer_candidate_archive=mode)
        for filename in ['SEEDS.json','PARAMETER_AXIS.json']:
            assert (tmp_path/'full'/filename).read_bytes()==(tmp_path/'skip'/filename).read_bytes()
        original={r['candidate_number']:r for r in map(json.loads,gzip.decompress((tmp_path/'full/ALL_FIT_CANDIDATES.jsonl.gz').read_bytes()).decode('utf-8').splitlines())}
        kept={r['candidate_number']:r for r in map(json.loads,gzip.decompress((tmp_path/'skip/EVALUATED_FIT_CANDIDATES.jsonl.gz').read_bytes()).decode('utf-8').splitlines())}
        assert all(row==original[number] for number,row in kept.items())
        missing=set(original)-set(kept)
        assert all(original[number]['fit_tp']==0 and original[number]['reason']!='eligible' for number in missing)
        proof=json.loads((tmp_path/'skip/ANALYTICALLY_SKIPPED.json').read_text())
        M,P=proof['M'],proof['P']
        certified={li*(M*(P+1)+M*(M-1)//2)+mi*(P+1)+1+pattern['pattern_index']
                   for pattern in proof['patterns'] for li in pattern['label_indices']
                   for mi in proof['model_indices']}
        assert certified==missing
        before=json.loads((tmp_path/'full/COMPLETE.json').read_text())
        after=json.loads((tmp_path/'skip/COMPLETE.json').read_text())
        assert before['counts']['total']==after['counts']['total']
        assert before['counts']['eligible']==after['counts']['eligible']
        assert after['counts']['analytically_omitted_zero_text_positive']==len(missing)
        assert not (tmp_path/'skip/ALL_FIT_CANDIDATES.jsonl.gz').exists()
        with pytest.raises(ValueError,match='options differ'):
            initialization.build_fit_initialized_space(space,prior,output_dir=tmp_path/'skip',
                initializer_slots=3,label_rotation=rotation,rotation_seed=5,cluster_position=2,
                initializer_candidate_archive='all')
    finally:
        prior.close()


def test_zero_slots_analytic_archive_does_not_compute_masks(tmp_path,monkeypatch):
    space,prior=fixture()
    def forbidden(*args,**kwargs):
        raise AssertionError('zero slots must not enumerate pattern masks')
    monkeypatch.setattr(initialization,'pattern_mask',forbidden)
    try:
        initialized=initialization.build_fit_initialized_space(space,prior,output_dir=tmp_path/'zero',
            initializer_slots=0,initializer_candidate_archive='evaluated_with_analytic_skips')
        assert gzip.decompress((tmp_path/'zero/EVALUATED_FIT_CANDIDATES.jsonl.gz').read_bytes())==b''
        assert json.loads((tmp_path/'zero/ANALYTICALLY_SKIPPED.json').read_text())['fit_grid_skipped_because_zero_slots']
        assert initialized.initial_configurations(4,seed=7)==space.initial_configurations(4,seed=7)
    finally:
        prior.close()
