import copy,json,pickle
from pathlib import Path
import numpy as np
import pytest
from loris._engine.rdl_reference import Doc,ML,Label,Rule,RefGamma
from loris._engine.typed_session import digest
from loris._storage.discovery_factory import make_discovery_runner
from loris.discovery.manifest import file_record
from loris.labeling import calibration_inference_cache as cache, reliability, cli, unknown_incremental
from tests.test_fit_reliability import calibration_fixture
from tests.test_current_model_reliability import prediction_record
from tests.test_relation_rule_reliability import ctx_for


def cache_fixture(tmp_path):
    ctx=ctx_for(['train','dev1','dev2'],labels=('A','B'))
    ctx['row_ids']=['t','d1','d2'];ctx['model_probabilities']['fixed']=np.full((3,2),.7,np.float32)
    ctx['static_model_probabilities']={k:v.copy() for k,v in ctx['model_probabilities'].items()}
    rules=(Rule((Doc('x'),ML('x','fixed','A',.5,dynamic=False)),Label('x','A')),)
    p=np.ones((3,2),bool);n=~p;sel=np.array([0,1,1],bool)
    r=make_discovery_runner(ctx,tmp_path/'cache')
    binding=cache.dependency_binding(r,ctx,rules,ctx['gamma'],p,n,sel,reliability.RETAINED_GAMMA_WINS_SEMANTICS)
    closure=unknown_incremental.retaining_chase(r,rules,ctx['gamma'],collect_rule_conclusions=True,
        rule_conclusion_observed=(p|n)&sel[:,None],gamma_wins=True)
    scores=reliability.count_rule_conclusions(r,closure,rules,p,n,sel)
    output=tmp_path/'out';output.mkdir();ref=cache.store(output,binding,closure,scores)
    receipt=output/'RESULTS.json';receipt.write_text(json.dumps(dict(status='complete',rule_scores=scores,rule_inference_cache=ref)))
    return r,ctx,rules,p,n,sel,binding,closure,scores,file_record(receipt)


def test_same_dependencies_reuse_arrays_and_every_score(tmp_path):
    r,ctx,rules,p,n,s,b,old,scores,ref=cache_fixture(tmp_path)
    try:
        new,new_scores,origin=cache.load_parent(ref,b)
        assert new_scores==scores
        for a in ('positive_facts','negative_facts','conflict_unknown'):assert np.array_equal(getattr(new,a),getattr(old,a))
        out=tmp_path/'again';out.mkdir();c=cache.store(out,b,new,new_scores,origin)
        v=json.loads(Path(c['path']).read_text());assert not v['fresh_inference_executed']
        assert v['reused_from']['parent_receipt']==ref
    finally:r.close()


@pytest.mark.parametrize('change',['used_static','gamma','observed','selected','threshold','semantics','text','presence','rows','labels','sources'])
def test_real_dependencies_invalidate_cache(tmp_path,change):
    r,ctx,rules,p,n,s,b,old,scores,ref=cache_fixture(tmp_path);r.close()
    changed=copy.deepcopy(ctx);rr=rules;sem=reliability.RETAINED_GAMMA_WINS_SEMANTICS
    if change=='used_static':changed['static_model_probabilities']['fixed'][0,0]=.9
    if change=='gamma':changed['gamma'].pos[0,0]=True
    if change=='observed':p[1,0]=False;n[1,0]=True
    if change=='selected':s[0]=True
    if change=='threshold':rr=(Rule((Doc('x'),ML('x','fixed','A',.8,dynamic=False)),Label('x','A')),)
    if change=='semantics':sem=reliability.RETAINED_CHASE_SEMANTICS
    if change=='text':changed['docs'][0].cnt+=' edit'
    if change=='presence':changed['attribute_presence']={'mtd.group':np.array([0,1,1],bool)}
    if change=='rows':changed['row_ids']=['other','d1','d2']
    if change=='labels':changed['labels']=['B','A']
    r=make_discovery_runner(changed,tmp_path/'new')
    try:
        bb=cache.dependency_binding(r,changed,rr,changed['gamma'],p,n,s,sem)
        if change=='sources':bb['semantic_sources']['some_new_file.py']='changed'
        assert cache.load_parent(ref,bb) is None
    finally:r.close()


def test_unused_aux_probability_and_opaque_provenance_do_not_invalidate(tmp_path):
    r,ctx,rules,p,n,s,b,old,scores,ref=cache_fixture(tmp_path);r.close()
    ctx['model_probabilities']['aux'][:]=.99;ctx['context_id']='new_unused_aux_model'
    r=make_discovery_runner(ctx,tmp_path/'new')
    try:assert cache.load_parent(ref,cache.dependency_binding(r,ctx,rules,ctx['gamma'],p,n,s,reliability.RETAINED_GAMMA_WINS_SEMANTICS)) is not None
    finally:r.close()


@pytest.mark.parametrize('which',['npz','manifest','receipt'])
def test_corrupt_bound_sidecar_is_rejected(tmp_path,which):
    r,ctx,rules,p,n,s,b,old,scores,ref=cache_fixture(tmp_path);r.close()
    path=tmp_path/'out'/dict(npz='RULE_INFERENCE.npz',manifest='RULE_INFERENCE.json',receipt='RESULTS.json')[which]
    with path.open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError):cache.load_parent(ref,b)


def test_unused_aux_refresh_hits_but_fallback_and_gate_are_recomputed(tmp_path,monkeypatch):
    args,calibration,_=calibration_fixture(tmp_path)
    source=json.loads(Path(args.manifest).read_text())
    source['loop'].update(conflict_policy='unknown_arbitrate',unknown_semantics='retain_consequences',gamma_conflict_policy='validated_gamma_wins',commit_gate='independent_calibration')
    source['arbitration']=dict(alias='aux',default_threshold=.5,thresholds={})
    source['fallback']=dict(enabled=True,alias='aux',calibration='independent_calibration',default_threshold=.5,thresholds={},grid=[.5,1.0],minimum_positive_support=1)
    program=Path(source['program']['path']);program.write_text(json.dumps([Rule((Doc('x'),ML('x','fixed','A',.5,dynamic=False)),Label('x','A')).to_dict()]))
    source['program']=file_record(program);Path(args.manifest).write_text(json.dumps(source))
    assert reliability.main_calibrate(args)==0
    artifact_path=Path(args.output)/'RELIABILITY.json'
    manifest,ctx,rules,artifact,_=cli.load(args.manifest,weighting='provenance',source_reliability=artifact_path)
    baseline=json.loads((Path(args.output)/'FALLBACK_CALIBRATION.json').read_text())
    checkpoint=tmp_path/'new.pt';checkpoint.write_bytes(b'new model')
    manifest=copy.deepcopy(manifest);manifest['dynamic']['checkpoint']=file_record(checkpoint)
    ctx=copy.deepcopy(ctx);ctx['model_probabilities']['aux'][:]=.9
    texts=reliability.calibration_texts(artifact)
    record=prediction_record(tmp_path,texts,ctx['labels'],manifest['dynamic']['checkpoint'])
    # New Aux predicts independent B positives correctly; q has no Aux premise.
    probs=np.full((20,2),.1,np.float32);probs[:15,1]=.9
    path=Path(record['file']['path']);np.savez(path,probabilities=probs);record['file']=file_record(path)
    old_fallback=reliability._fallback_calibration;calls=[]
    def seen(*a,**k):calls.append(True);return old_fallback(*a,**k)
    monkeypatch.setattr(reliability,'_fallback_calibration',seen)
    monkeypatch.setattr(unknown_incremental,'retaining_chase',lambda *a,**k:pytest.fail('unused Aux should hit exact inference cache'))
    refreshed,_=reliability.refresh_current(artifact,ctx,rules,manifest,record,tmp_path/'refresh')
    result=json.loads((tmp_path/'refresh/RESULTS.json').read_text())
    fallback=json.loads((tmp_path/'refresh/FALLBACK_CALIBRATION.json').read_text())
    assert calls==[True] and result['calibration_inference_execution']=='verified_dependency_cache'
    assert refreshed['rule_scores']==artifact['rule_scores']
    assert refreshed['binding']['checkpoint_sha256']==file_record(checkpoint)['sha256']
    assert fallback['gate']!=baseline['gate']
    assert fallback['thresholds']!=baseline['thresholds']
    assert fallback['binding']!=baseline['binding']

@pytest.mark.parametrize('alias',['aux','dependent_ensemble'])
def test_used_dynamic_model_and_dependent_ensemble_change_invalidate(tmp_path,alias):
    ctx=ctx_for(['a','b','c'],labels=('A','B'));ctx['row_ids']=['a','b','c']
    ctx['model_probabilities'][alias]=np.full((3,2),.7,np.float32)
    rules=(Rule((Doc('x'),ML('x',alias,'A',.5,dynamic=True)),Label('x','A')),)
    pos=np.ones((3,2),bool);neg=~pos;selected=np.ones(3,bool)
    r=make_discovery_runner(ctx,tmp_path/'a')
    try:a=cache.dependency_binding(r,ctx,rules,ctx['gamma'],pos,neg,selected,reliability.RETAINED_GAMMA_WINS_SEMANTICS)
    finally:r.close()
    ctx['model_probabilities'][alias][1,0]=.1
    r=make_discovery_runner(ctx,tmp_path/'b')
    try:b=cache.dependency_binding(r,ctx,rules,ctx['gamma'],pos,neg,selected,reliability.RETAINED_GAMMA_WINS_SEMANTICS)
    finally:r.close()
    assert a!=b
    assert a['execution']['probabilities'][alias]!=b['execution']['probabilities'][alias]
