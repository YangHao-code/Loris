import json
from pathlib import Path

import numpy as np
import pytest

from loris.document import Document
from loris._engine.rdl_reference import Doc, ML, Label, Rule, RefGamma
from loris.discovery.staged.pipeline import choose, discover
from loris.discovery.staged.pool import (
    calibrate_expert, calibrated_probabilities, extend_context, fit_pool, pool_probabilities)


def test_diversity_priority_is_bounded_and_does_not_count_fp_only_families():
    def entry(name, macro, micro, families):
        return dict(candidate_id=name, quality_B=dict(macro_f1=macro, micro_f1=micro),
                    effective_families_B={str(i):dict(necessary_TP=v) for i,v in enumerate(families)}, rules=[{}])
    entries = [entry('base', .70, .80, [100]), entry('rich', .695, .797, [90, 2, 1]),
               entry('bad', .66, .80, [70, 1, 1, 1]), entry('decorative', .70, .80, [100, 0, 0, 0])]
    assert choose(entries, macro_tolerance=.01, micro_tolerance=.01) == 'rich'
    assert choose(entries, priority='macro', macro_tolerance=.01, micro_tolerance=.01) == 'base'
    assert choose(entries) == 'base'


def test_margin_ensemble_preserves_calibrated_boundaries_and_axes():
    values = {'a':np.array([[.2,.8],[.7,.2]]), 'b':np.array([[.6,.3],[.9,.1]])}
    cfg = dict(members=['a','b'], labels=['A','B'], expert_cuts={'a':[.2,.8],'b':[.6,.3]})
    p = pool_probabilities(values,cfg)
    assert np.allclose(p[0],.5) and p[1,0]>.5 and p[1,1]<.5
    assert np.allclose(calibrated_probabilities(values['a'],[.2,.8])[0],.5)
    with pytest.raises(ValueError):pool_probabilities({'a':values['a'],'b':values['b'][:1]},cfg)
    with pytest.raises(ValueError):extend_context(dict(labels=['B','A']),cfg)


def test_rare_and_absent_labels_keep_calibrated_rules():
    y=np.zeros((12,3),bool);y[:1,0]=1;y[:5,1]=1
    p=np.full(y.shape,.1);p[y]=.9
    d=calibrate_expert(p,y)
    assert len(d['per_label_thresholds'])==3
    assert d['per_label_thresholds'][2]==d['global_threshold']
    assert d['per_label_positive_support']==[1,5,0]
    assert np.isfinite(d['per_label_thresholds']).all()


def fixture():
    rng=np.random.RandomState(99);n=18;k=2
    pos=np.zeros((n,k),bool);neg=pos.copy();pos[:4]=rng.rand(4,k)<.4;neg[:4]=~pos[:4]
    ctx=dict(docs=[Document(cnt='red '*i+' blue',mtd=str(i%3)) for i in range(n)],
             row_ids=[str(i) for i in range(n)],row_roles=['train']*4+['dev']*14,
             labels=['A','B'],gamma=RefGamma(pos,neg),
             model_probabilities={'m':rng.rand(n,k), 'q':rng.rand(n,k)})
    pool=dict(alias='loris_calibrated_pool',members=['m','q'],labels=['A','B'],expert_cuts={'m':[.5]*k,'q':[.5]*k})
    A=np.arange(4,13);B=np.arange(13,n)
    p=pool_probabilities(ctx['model_probabilities'],pool)
    return ctx,pool,A,B,p[A]>=.5,p[B]>=.5


@pytest.mark.parametrize('mode',['frozen','diverse'])
def test_portable_discovery_verifies_native_closure_and_preserves_inputs(tmp_path,mode):
    ctx,pool,A,B,yA,yB=fixture();original=ctx['gamma'].pos.copy()
    old=[Rule((Doc('x'),Label('x','A'),ML('x','m','B',.5)),Label('x','B'))]
    result=discover(context=ctx,pool=pool,old_rules=old,A=A,truth_A=yA,B=B,truth_B=yB,
                    output=tmp_path/mode,mode=mode,seed=28101)
    assert result['complete_native_closures_verified']
    assert result['settings']['B_used_for_pruning']==(mode=='diverse')
    assert 'loris_calibrated_pool' not in ctx['model_probabilities']
    assert np.array_equal(ctx['gamma'].pos,original)
    selected=json.loads((tmp_path/mode/'SELECTED.json').read_text())
    assert selected['candidate_id']==result['selected_candidate_id']
    assert all(r['head']['polarity']=='pos' for r in selected['rules'])


def test_discovery_rejects_overlap_target_roles_and_trusted_answers(tmp_path):
    ctx,pool,A,B,yA,yB=fixture()
    args=dict(context=ctx,pool=pool,old_rules=[],A=A,truth_A=yA,B=B,truth_B=yB,output=tmp_path/'run')
    with pytest.raises(ValueError,match='Disjoint'):discover(**dict(args,B=A,truth_B=yA))
    ctx['row_roles'][0]='test'
    with pytest.raises(ValueError,match='train/dev-only'):discover(**args)
    ctx['row_roles'][0]='train'
    pos=ctx['gamma'].pos.copy();pos[A[0],0]=True
    ctx['gamma']=RefGamma(pos,ctx['gamma'].neg.copy())
    with pytest.raises(ValueError,match='Gamma'):discover(**args)
    assert not (tmp_path/'run').exists()


def test_calibrated_pool_runs_real_original_neural_selector(tmp_path):
    from loris._learning.selection import ModelSelectionProtocol
    rng=np.random.RandomState(31);n=14;k=2;y=rng.rand(n,k)<.3
    values={a:rng.uniform(.05,.95,(n,k)) for a in ('a','b','c')}
    cfg=ModelSelectionProtocol(k=2,epochs=27,hidden_dim=8,num_samples=8,sigma=.1,
        learning_rate=.001,lambda_task=.1,lambda_entropy=.01,imitation_positive_weight=2.,
        vote_smoothing=.1,dropout=.1,seed=42,device='cpu',
        oracle_policy='global_margin_macro_f1',task_objective='pooled_margin_macro_f1')
    result=fit_pool(features=rng.randn(n,4).astype('float32'),probabilities=values,truth=y,
        row_ids=list(range(n)),labels=['A','B'],input_binding=dict(role='fit',expected_fit_rows=n,
        feature_provenance={'synthetic':True},decision_provenance={'synthetic':True}),
        output=tmp_path/'pool',k=2,protocol=cfg)
    selected=json.loads((tmp_path/'pool/selector/SELECTED.json').read_text())
    assert result['members']==next(iter(selected['selected_aliases_by_cluster'].values()))
    assert selected['completed_optimizer_steps']==27


def test_cli_authenticates_assets_and_runs_portable_discovery(tmp_path):
    import pickle
    from loris.discovery.manifest import file_record
    from loris.discovery.staged.cli import run
    ctx,pool,A,B,yA,yB=fixture()
    context=tmp_path/'context.pkl';context.write_bytes(pickle.dumps(ctx))
    p=tmp_path/'POOL.json';p.write_text(json.dumps(pool))
    rules=tmp_path/'rules.json';rules.write_text(json.dumps({'rules':[]}))
    arrays=tmp_path/'partitions.npz';np.savez(arrays,row_ids=np.asarray(ctx['row_ids']),
        labels=np.asarray(ctx['labels']),A=A,B=B,truth_A=yA,truth_B=yB)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(
        schema='loris_staged_discovery_inputs_v1',context=file_record(context),pool=file_record(p),
        partitions=file_record(arrays),old_rules=file_record(rules),settings=dict(mode='frozen'))))
    result=run(manifest,tmp_path/'output')
    assert result['native_verified']
    assert (tmp_path/'output/INPUTS.json').exists()
    rules.write_text(json.dumps({'rules':[],'changed':True}))
    with pytest.raises(ValueError,match='identity'):run(manifest,tmp_path/'changed')
    assert not (tmp_path/'changed').exists()
