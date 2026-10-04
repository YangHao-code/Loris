"""Historical-pool replay has different identity from a fresh BO search."""
import copy
import json

import pytest

from loris.discovery.runner import run_discovery
from loris.discovery.refilter import refilter
from tests.test_discovery_runner import inputs


@pytest.fixture(scope='module')
def frozen_pool(tmp_path_factory):
    directory=tmp_path_factory.mktemp('portable_refilter');kw=inputs()
    kw['admission_order']='proposal_order'  # This fixture is an explicitly historical pool.
    receipt=run_discovery(output=directory/'source',**kw)
    return directory,kw,receipt


def test_real_ordered_refilter_zero_and_strict_floor(frozen_pool):
    directory,kw,original=frozen_pool;source=directory/'source'
    before={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
    zero=refilter(source=source,output=directory/'zero',min_gain=0.,**kw)
    restricted=refilter(source=source,output=directory/'restricted',min_gain=.5,**kw)
    assert zero['new_BO']==restricted['new_BO']==0
    assert zero['source_BO_trials']==zero['candidates_processed']==4
    assert zero['final_positive_metrics']==original['final_positive_metrics']
    assert zero['actual_scoring_Chase_calls']==0 and zero['final_proof_Chase_calls']==1 and zero['cache_hits']>0
    assert zero['selected_rules']==2 and restricted['selected_rules']==0
    assert restricted['final_positive_metrics']['macro_f1']==0.
    decisions=[json.loads(p.read_text()) for p in sorted((directory/'restricted/decisions').glob('candidate_*.json'))]
    assert len(decisions)==4
    assert any(d['decision'].get('delta_macro_f1')==.5 and not d['decision']['accepted'] for d in decisions)
    assert all(p.read_bytes()==value for p,value in before.items())
    assert (directory/'zero/PROVENANCE.jsonl.gz').is_file()


@pytest.mark.parametrize('case',['model_change','budget_change','missing_admission','source_output','negative_gain'])
def test_refilter_rejects_identity_or_archive_mismatch(frozen_pool,case):
    directory,base,_=frozen_pool;source=directory/'source';kw=copy.deepcopy(base)
    out=directory/('invalid_'+case);gain=0.;restore=None
    if case=='model_change':kw['context']['model_probabilities']['m']+=.01
    elif case=='budget_change':kw['budget_per_cluster']=5
    elif case=='missing_admission':
        path=source/'learning/cluster_00/admission/candidate_00000.json'
        restore=(path,path.read_bytes());path.unlink()
    elif case=='source_output':out=source/'child'
    elif case=='negative_gain':gain=-1.
    try:
        with pytest.raises((ValueError,FileNotFoundError)):
            refilter(source=source,output=out,min_gain=gain,**kw)
    finally:
        if restore:restore[0].write_bytes(restore[1])
