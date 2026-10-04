"""The opt-in approximation affects only the candidate domain, never labels/GT."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from loris.labeling import runner as module
from loris.labeling.cli import validate
from tests.test_batch_query_and_exports import configured
from tests.test_labeling import synthetic_same_trainer

@pytest.mark.parametrize('limit',[None,2,1024])
def test_manifest_optional_cap_accepts(tmp_path,limit):
 _,_,m=configured(tmp_path);m['feedback']['trusted_replay']=True;m['query']['rdg_candidate_limit']=limit;validate(m)

@pytest.mark.parametrize('limit',[0,-1,True,1.5,'1024',1])
def test_invalid_or_batch_smaller_cap_fails(tmp_path,limit):
 _,_,m=configured(tmp_path);m['feedback']['trusted_replay']=True;m['query']['rdg_candidate_limit']=limit
 with pytest.raises(ValueError,match='rdg_candidate_limit'):validate(m)

def rank(tmp_path,monkeypatch,cap='absent',seed=42,policy='rdg',rows=None):
 rows=list(range(20))if rows is None else rows;mask=np.zeros(20,bool);mask[rows]=True;seen=[];events=[]
 class Estimator:
  def __init__(self,rules):assert rules==()
  def estimate_batch(self,session,result,queries,uncovered):
   assert uncovered==list(range(20))
   for row,label in queries:seen.append((row,label));yield row+label/10
 monkeypatch.setattr(module,'RuleDependencyGraph',Estimator);monkeypatch.setattr(module,'pending_rows',lambda *args:np.ones(20,bool))
 query=dict(strategy=policy,seed=seed)
 if cap!='absent':query['rdg_candidate_limit']=cap
 run=SimpleNamespace(eligible_query_rows=lambda result:mask,manifest={'query':query,'loop':{'max_rdg_estimates':10000}},rows=tuple('row'+str(x)for x in range(20)),query_random=np.random.RandomState(seed),output=tmp_path,queries=0,gamma=None,k=3,rdg_estimates=0,event=lambda kind,**kw:events.append(dict(kind=kind,**kw)))
 tmp_path.mkdir(parents=True,exist_ok=True)
 selected=module.LabelingRun.rdg(run,object(),SimpleNamespace(rules=(),runner=SimpleNamespace(probabilities={'aux':np.zeros((20,3))})),limit=2)
 return selected,seen,events,json.loads((tmp_path/'RDG_00000.json').read_text())

def test_fixed_seed_uniform_exact_all_labels_and_original_row_order(tmp_path,monkeypatch):
 selected,seen,events,record=rank(tmp_path/'a',monkeypatch,5)
 sample=sorted(np.random.RandomState(42).choice(np.arange(20,dtype=np.int64),size=5,replace=False).tolist())
 assert seen==[(r,k)for r in sample for k in range(3)]
 assert selected==sorted(sample,reverse=True)[:2]
 audit=record['candidate_domain'];assert audit['sampled_row_indices']==sample and audit['sampled_row_ids']==['row'+str(r)for r in sample]
 assert audit['full_eligible_count']==20 and audit['sampled_count']==5 and audit['approximate_domain']is True and audit['unqueried_truth_used']is False
 assert json.loads((tmp_path/'a/RDG_DOMAIN_00000.json').read_text())==audit
 assert events[0]['kind']=='query_candidate_domain'and events[1]['hypotheses']==15
 other=rank(tmp_path/'b',monkeypatch,5);assert other[0]==selected and other[1]==seen and other[3]==record
 different=rank(tmp_path/'c',monkeypatch,5,seed=43);assert different[1]!=seen

def test_absent_none_nonbinding_limit_preserve_all_scores_and_choices(tmp_path,monkeypatch):
 a=rank(tmp_path/'a',monkeypatch);b=rank(tmp_path/'b',monkeypatch,None);c=rank(tmp_path/'c',monkeypatch,100)
 assert a[:3]==b[:3]and a[3]==b[3]and a[0:2]==c[0:2]
 assert c[3]['candidate_domain']['approximate_domain']is False
 assert not(tmp_path/'a/RDG_DOMAIN_00000.json').exists()and not(tmp_path/'b/RDG_DOMAIN_00000.json').exists()

def test_ineligible_or_previously_answered_rows_never_sampled(tmp_path,monkeypatch):
 _,seen,_,r=rank(tmp_path,monkeypatch,3,rows=[2,5,7,9,11]);assert set(x[0]for x in seen)<=set([2,5,7,9,11]);assert r['candidate_domain']['full_eligible_count']==5

def test_random_existing_seed_sequence_unchanged_by_rdg_cap(tmp_path,monkeypatch):
 a=rank(tmp_path/'a',monkeypatch,policy='random');b=rank(tmp_path/'b',monkeypatch,5,policy='random');assert a==b

def test_runtime_fractional_request_cannot_silently_shorten_batch(tmp_path,monkeypatch):
 with pytest.raises(ValueError,match='requested query batch'):rank(tmp_path,monkeypatch,1)

def test_real_chase_and_complete_report_records_declared_cap(tmp_path):
 context,rules,manifest=configured(tmp_path);manifest['query']['rdg_candidate_limit']=2
 run=module.LabelingRun(context=context,rules=rules,manifest=manifest,output=tmp_path/'run',trainer=synthetic_same_trainer)
 report=run.run();assert report['query_policy']['rdg_candidate_limit']==2
 files=list((tmp_path/'run').glob('RDG_DOMAIN_*.json'));assert files
 for f in files:
  audit=json.loads(f.read_text());assert audit['all_native_labels_evaluated']==2 and audit['sampled_count']<=2 and audit['unqueried_truth_used']is False
