"""Tiny REAL torch/Aux optimization, requiring the explicit optional backend.

This is a synthetic software fixture, never a BGC accuracy experiment. No
pretrained download, production truth or GPU is used. LORIS_SSL_SOURCE points
to the separately versioned real backend included in the review bundle.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from loris.discovery.manifest import file_record
from loris.labeling.feedback import Batch
from loris.labeling.trainer import train_actual,load_backend


class Tokenizer:
    pad_token_id=0
    def __call__(self,texts,**kwargs):
        values=[[3+sum(w.encode())%50 for w in t.split()]for t in texts]
        return {'input_ids':values}
    def num_special_tokens_to_add(self,pair=False):return 2
    def build_inputs_with_special_tokens(self,values):return [1]+list(values)+[2]
    def pad(self,features,padding=True,return_tensors='pt'):
        n=max(len(x['input_ids'])for x in features)
        return {'input_ids':torch.tensor([list(x['input_ids'])+[0]*(n-len(x['input_ids']))for x in features]),
            'attention_mask':torch.tensor([list(x['attention_mask'])+[0]*(n-len(x['input_ids']))for x in features])}


class Encoder(nn.Module):
    def __init__(self,k):
        super().__init__();self.encoder=nn.Embedding(64,8,padding_idx=0)
        self.label_queries=nn.Parameter(torch.randn(k,8)*.01)
        self.classifier_weight=nn.Parameter(torch.randn(k,8)*.01)
        self.classifier_bias=nn.Parameter(torch.full((k,),-2.))
        self.binary_prototypes=nn.Parameter(torch.randn(k,2,4)*.01)
        self.projection=nn.Linear(8,4)
    def forward(self,input_ids,attention_mask,**kwargs):
        z=self.encoder(input_ids);m=attention_mask.to(z.dtype).unsqueeze(-1)
        pooled=(z*m).sum(1)/m.sum(1).clamp_min(1)
        return {'logits':pooled@self.classifier_weight.T+self.classifier_bias}


def assets(tmp_path):
    source=os.environ.get('LORIS_SSL_SOURCE')
    if not source or not (Path(source)/'loris_ssl/chase_backend.py').is_file():
        pytest.skip('actual optional Aux integration requires explicit LORIS_SSL_SOURCE; semantic tests remain portable')
    backend=load_backend(source);torch.set_num_threads(1);torch.manual_seed(43)
    model=Encoder(2);parent=tmp_path/'parent.pt'
    torch.save(dict(state_dict=model.state_dict(),label_names=['A','B'],model_name='cpu-labeling-fixture',head='label_attention_linear'),str(parent))
    config=backend.ChaseBackendConfig(model_name='cpu-labeling-fixture',cache_dir='',offline=True,
        max_length=12,head='label_attention_linear',projection_dim=4,gradient_checkpointing=False,
        loss='masked_weighted_bce',epochs=1,min_epochs=1,patience=0,batch_size=2,prediction_batch_size=2,
        grad_accum_steps=2,learning_rate=.005,head_learning_rate=.01,weight_decay=0.,warmup_ratio=0.,
        prefer_ema=True,amp='none',num_workers=0,seed=43,initialize_class_priors=False)
    return source,parent,config.to_dict()


def test_actual_aux_current_checkpoint_two_fits_and_window_weights(tmp_path):
    source,parent,config=assets(tmp_path)
    targets=np.array([[1,0],[1,0],[0,1],[0,0]],np.float32)
    true=np.array([[1,1],[0,0],[1,1],[0,0]],bool)
    weak=np.array([[0,0],[1,0],[0,0],[0,0]],bool)
    observed=true|weak;weights=true.astype(np.float32)+weak.astype(np.float32)*.25
    batch=Batch(targets,observed,weights,true,weak,{})
    kwargs=dict(texts=['red zero','red one','blue two','nothing'],labels=['A','B'],batch=batch,
        source=source,configuration=config,device='cpu',gpu_lock=None,max_seconds=60,
        tokenizer_factory=lambda _:Tokenizer(),model_factory=lambda _,k:Encoder(k))
    p,first=train_actual(checkpoint=parent,checkpoint_sha256=file_record(parent)['sha256'],output=tmp_path/'fit1',**kwargs)
    q,second=train_actual(checkpoint=first['checkpoint']['path'],checkpoint_sha256=first['checkpoint']['sha256'],output=tmp_path/'fit2',**kwargs)
    assert first['parameter_state_changed'] and second['parameter_state_changed']
    assert second['parent_state_sha256']==first['selected_state_sha256']
    assert p.shape==q.shape==(4,2) and np.isfinite(q).all() and not np.array_equal(p,q)
    assert first['optimizer_updates']==second['optimizer_updates']==1
    trace=json.loads((tmp_path/'fit1/ACTUAL_TRAINING_TRACE.json').read_text())
    assert trace['windows'][0]['true_mass']==4 and trace['windows'][0]['weak_mass']==.25
    assert trace['windows'][0]['weak_fraction']==pytest.approx(.25/4.25)
    assert set(trace['sampled_rows'])=={0,1,2}  # unknown row3 not optimized
    assert first['audit']['validation_supervision'] is None


def test_full_real_aux_complete_chase_inner_loop(tmp_path):
    from tests.test_labeling import fixture
    from loris.labeling.runner import LabelingRun
    source,parent,config=assets(tmp_path);ctx,rules,m=fixture()
    m['dynamic'].update(checkpoint=file_record(parent),backend_source=source,configuration=config)
    def actual(**kwargs):
        return train_actual(**kwargs,tokenizer_factory=lambda _:Tokenizer(),model_factory=lambda _,k:Encoder(k))
    run=LabelingRun(context=ctx,rules=rules,manifest=m,output=tmp_path/'loop',trainer=actual)
    report=run.run()
    assert report['status']=='label_budget_exhausted'
    assert report['actual_optimizer_updates']==1 and report['committed_model_updates']==0
    assert report['current_Gamma_has_valid_closure'] and report['actual_Chase_calls']==2
    assert run.gamma.pos.sum()==1 and run.gamma.neg.sum()==1


def test_actual_aux_independent_prediction_set_never_enters_optimizer(tmp_path):
    from loris._engine.typed_session import digest
    source,parent,config=assets(tmp_path)
    targets=np.array([[1,0],[0,1]],np.float32);observed=np.ones((2,2),bool)
    batch=Batch(targets,observed,np.ones_like(targets),observed,~observed,{})
    independent=['never optimized independent text','another independent document','third independent document']
    _,receipt=train_actual(texts=['actual training first','actual training second'],labels=['A','B'],batch=batch,
        checkpoint=parent,checkpoint_sha256=file_record(parent)['sha256'],source=source,configuration=config,
        device='cpu',output=tmp_path/'fit',gpu_lock=None,max_seconds=60,
        tokenizer_factory=lambda _:Tokenizer(),model_factory=lambda _,k:Encoder(k),
        prediction_sets={'independent_calibration':independent})
    record=receipt['prediction_sets']['independent_calibration']
    assert record['texts_id']==digest(independent) and record['checkpoint_sha256']==receipt['checkpoint']['sha256']
    assert record['supervision_used']is False
    with np.load(record['file']['path'])as archive:
        assert archive['probabilities'].shape==(3,2) and np.isfinite(archive['probabilities']).all()
    trace=json.loads((tmp_path/'fit/ACTUAL_TRAINING_TRACE.json').read_text())
    assert set(trace['sampled_rows'])=={0,1} and receipt['optimizer_updates']==1
