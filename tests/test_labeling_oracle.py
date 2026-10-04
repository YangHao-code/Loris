"""One-row simulated provider: actual child process, no external assets."""
import json
import pickle
import subprocess
import sys

import numpy as np
import pytest

from loris.discovery.manifest import file_record
from loris.labeling.oracle import prepare, answer
from loris.labeling.runner import LabelingRun
from tests.test_labeling import fixture, synthetic_same_trainer


def assets(tmp_path):
    ctx,rules,m=fixture();ctx['raw_indices']=np.arange(4)
    c=tmp_path/'context.pkl'
    with c.open('wb')as f:pickle.dump(ctx,f)
    t=tmp_path/'truth.npz';np.savez(t,truth=np.array([[1,0],[1,0],[1,0],[0,1]],bool))
    a=tmp_path/'axes.json';a.write_text(json.dumps({'labels':['A','B']}))
    out=tmp_path/'oracle.json'
    kwargs=dict(context_path=c,truth_path=t,truth_key='truth',axis_path=a,axis_key='labels',
                roles=['train'],source_rows=4,source_offset=0,output=out)
    return ctx,rules,m,kwargs


def test_prepare_never_decodes_targets_and_child_returns_one_answer(tmp_path,monkeypatch):
    _,_,_,kw=assets(tmp_path)
    with monkeypatch.context()as patch:
        patch.setattr(np,'load',lambda *a,**k:pytest.fail('prepare must not decode truth'))
        receipt=prepare(**kw)
    assert receipt['queries']==0 and not receipt['truth_decoded']
    child=subprocess.run([sys.executable,'-m','loris.labeling.oracle','--manifest',str(kw['output'])],
        input=json.dumps(dict(row=2,row_id='r2',labels=['A','B'])),text=True,capture_output=True,check=True)
    result=json.loads(child.stdout)
    assert result['targets']==[1,0] and result['row_id']=='r2'
    assert result['disclosed_answer_rows']==1 and result['unqueried_answers_returned']==0
    assert 'rows'not in result
    with pytest.raises(FileExistsError):prepare(**kw)


def test_forbidden_row_rejected_before_truth_and_source_changes_fail(tmp_path,monkeypatch):
    _,_,_,kw=assets(tmp_path);prepare(**kw)
    with monkeypatch.context()as patch:
        patch.setattr(np,'load',lambda *a,**k:pytest.fail('forbidden request accessed truth'))
        with pytest.raises(ValueError,match='not permitted'):
            answer(kw['output'],dict(row=3,row_id='r3',labels=['A','B']))
    kw['truth_path'].write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity differs'):
        answer(kw['output'],dict(row=2,row_id='r2',labels=['A','B']))


def test_actual_query_then_complete_Chase_preserves_signed_answer(tmp_path):
    ctx,rules,m,kw=assets(tmp_path);prepare(**kw)
    m['strategy']='frozen';m['loop']['label_budget']=1
    m['oracle']=dict(kind='npz_simulated_human',directory=None,allow_known_source_review=False,
                     manifest=file_record(kw['output']))
    run=LabelingRun(context=ctx,rules=rules,manifest=m,output=tmp_path/'run',trainer=synthetic_same_trainer)
    result=run.run()
    assert result['actual_label_requests']==1 and result['training_attempts']==0
    assert result['current_Gamma_has_valid_closure'] and run.gamma.pos[2,0] and run.gamma.neg[2,1]
    assert (tmp_path/'run/ANSWER_00001.json').is_file()
    events=[e for e in run.events if e['kind']=='label_request']
    assert len(events)==1 and events[0]['row']==2
    with pytest.raises(ValueError,match='known complete'):
        run.oracle_answer(2,'would_repeat')


def test_budget_overrides_are_written_into_effective_configuration(tmp_path,monkeypatch):
    from loris.labeling import cli
    _,_,m,kw=assets(tmp_path)
    p=tmp_path/'input.json';p.write_text(json.dumps(m))
    observed=[]
    def stop_after_validate(data):
        observed.append(data);raise RuntimeError('validation seam before all asset loading')
    monkeypatch.setattr(cli,'validate',stop_after_validate)
    with pytest.raises(RuntimeError,match='validation seam'):
        cli.load(p,label_budget=7,source_review_budget=12,query_batch_size=3,query_strategy='random')
    assert observed[0]['loop']['label_budget']==7 and observed[0]['loop']['source_review_budget']==12
    assert observed[0]['loop']['query_batch_size']==3 and observed[0]['query']['strategy']=='random'
