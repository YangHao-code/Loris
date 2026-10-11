import itertools
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from loris._learning.selection import (build_joint_vote_oracle,pooled_macro_task,
    PerturbedTopK,hard_topk,ModelSelectionProtocol,train_model_selection,verify_model_selection_output)

@pytest.mark.parametrize('seed',range(4))
def test_global_F1_target_matches_independent_all_subsets(seed):
 rng=np.random.RandomState(seed);y=rng.rand(19,5)<.2;d=rng.rand(6,19,5)<.25
 for policy in ['global_macro_f1','global_balanced_f1']:
  result=build_joint_vote_oracle(d,y,3,.1,oracle_policy=policy)
  scores=[]
  for subset in itertools.combinations(range(6),3):
   p=d[list(subset)].sum(0)>1;tp=(p&y).sum(0);den=p.sum(0)+y.sum(0)
   macro=np.divide(2*tp,den,out=np.zeros(5),where=den!=0).mean();micro=2*tp.sum()/max(den.sum(),1)
   key=(macro,micro) if policy=='global_macro_f1' else ((macro+micro)/2,)
   scores.append((key,subset))
  best=max(scores,key=lambda e:e[0])[1]
  assert np.all(result['mask'][:,list(best)]==1) and np.all(result['mask'].sum(1)==3)

def test_population_loss_uses_frequency_pool_and_backpropagates():
 torch.manual_seed(5);scores=torch.randn(11,5,requires_grad=True)
 mask=PerturbedTopK.apply(scores,3,32,.1);d=torch.randint(0,2,(5,11,7)).float();y=torch.randint(0,2,(11,7)).float()
 loss=pooled_macro_task(mask,d,y,3,.1,32,.1)
 pool=hard_topk(mask.detach().mean(0),3);p=(torch.einsum('m,mrl->rl',pool,d)+.1)/3.2
 tp=(p*y).sum(0);den=p.sum(0)+y.sum(0)
 expected=1-(.8*(2*tp/den).mean()+.2*2*tp.sum()/den.sum())
 assert torch.allclose(loss,expected);loss.backward()
 assert torch.isfinite(scores.grad).all() and scores.grad.abs().sum()>0

def test_original_trainer_and_verifier_accept_actual_f1_training(tmp_path):
 rng=np.random.RandomState(8);x=rng.randn(12,5).astype('float32');y=rng.rand(12,4)<.3;d=rng.rand(4,12,4)<.3
 cfg=ModelSelectionProtocol(k=2,epochs=3,hidden_dim=8,num_samples=16,sigma=.1,
  learning_rate=.01,lambda_task=1,lambda_entropy=.01,imitation_positive_weight=2,
  vote_smoothing=.1,dropout=0,seed=42,device='cpu',oracle_policy='global_macro_f1',task_objective='pooled_macro_f1')
 train_model_selection(features=x,decisions=d,truth=y,cluster_assignments=np.zeros(12,dtype='int64'),
  row_ids=list(range(12)),aliases=['a','b','c','d'],labels=['w','x','y','z'],
  input_binding=dict(role='fit',expected_fit_rows=12,feature_provenance={'synthetic':True},decision_provenance={'synthetic':True}),
  protocol=cfg,output_dir=tmp_path/'selector')
 result=verify_model_selection_output(tmp_path/'selector')
 assert result['result']['oracle_policy']=='global_macro_f1'

def test_probability_oracle_and_training_match_deployed_margin_utility(tmp_path):
 from loris._learning.selection import interpretation_margins,margin_predictions
 rng=np.random.RandomState(18);x=rng.randn(15,6).astype('float32');y=rng.rand(15,3)<.3
 p=rng.uniform(.02,.98,(4,15,3));d=p>=.5;aliases=['a','b','c','d'];rows=list(range(15))
 interpretation=dict(row_ids=rows,probabilities={a:p[i] for i,a in enumerate(aliases)},decisions={a:d[i] for i,a in enumerate(aliases)})
 margins=interpretation_margins(interpretation,aliases,rows)
 oracle=build_joint_vote_oracle(d,y,2,.1,oracle_policy='global_margin_macro_f1',margins=margins)
 selected=np.flatnonzero(oracle['mask'][0]);best=oracle['best_global_macro_subset']
 assert selected.tolist()==best['indices']
 cfg=ModelSelectionProtocol(k=2,epochs=3,hidden_dim=8,num_samples=16,sigma=.1,learning_rate=.01,
  lambda_task=.1,lambda_entropy=.01,imitation_positive_weight=2,vote_smoothing=.1,dropout=0,seed=42,device='cpu',
  oracle_policy='global_margin_macro_f1',task_objective='pooled_margin_macro_f1')
 train_model_selection(features=x,decisions=d,truth=y,cluster_assignments=np.zeros(15,dtype='int64'),row_ids=rows,aliases=aliases,
  labels=['x','y','z'],input_binding=dict(role='fit',expected_fit_rows=15,feature_provenance={'synthetic':True},decision_provenance={'synthetic':True}),
  protocol=cfg,output_dir=tmp_path/'margin',model_interpretation=interpretation)
 assert verify_model_selection_output(tmp_path/'margin')['result']['completed_optimizer_steps']==3

def test_centered_mc_has_zero_spurious_gradient_for_stable_selection():
 from loris._learning.selection import CenteredPerturbedTopK
 x=torch.tensor([[100.,-100.,-200.]],requires_grad=True)
 mask=CenteredPerturbedTopK.apply(x,1,128,.01);mask.sum().backward()
 assert torch.equal(x.grad,torch.zeros_like(x))
