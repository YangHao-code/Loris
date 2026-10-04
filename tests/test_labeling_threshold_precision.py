"""Threshold comparison follows the actual complete Chase's float64 tables."""
import numpy as np

from loris.labeling.feedback import Batch,calibrate
from loris.labeling.runner import LabelingRun
from loris._engine.rdl_reference import Rule,Doc,ML,Label
from loris._engine.typed_session import digest,array_identity
from tests.test_labeling import fixture,synthetic_same_trainer


def test_float32_stored_boundary_uses_runtime_float64_in_calibration_and_predicates(tmp_path):
    p=np.array([[.95],[.98]],np.float32)
    assert float(p[0,0]) < .95  # Explicit float64 comparison, including NumPy 2.
    y=np.array([[0],[1]],np.float32);mask=np.ones_like(y,bool)
    batch=Batch(y,mask,np.ones_like(y),mask,np.zeros_like(mask),{})
    channels={(0,1):(0,.9)}
    selected,receipt=calibrate(p,batch,channels,grid=[.9,.95],minimum_support_each_sign=1)
    assert selected[(0,1)]==(0,.95)
    assert receipt['occurrences'][0]['scores']==[dict(kappa=.9,score=.5),dict(kappa=.95,score=1.)]
    assert receipt['occurrences'][0]['probability_comparison_dtype']=='float64_like_ReferenceChase'

    ctx,_,m=fixture();ctx['model_probabilities']['aux'][:]=np.float32(.95)
    # .95 stored in float32 lies below the declared real .95 threshold.
    # A .95 vote must also be false; preserve an exact original ensemble.
    ctx['model_probabilities']['vote']=np.zeros((4,2),np.float32)
    m['dynamic']['ensemble_dependencies']=[dict(alias='vote',dependencies=['aux'],operation='binary_vote_mean',vote_threshold=.95)]
    rules=(Rule((Doc('x'),ML('x','aux','B',.95)),Label('x','B')),
           Rule((Doc('x'),ML('x','vote','B',.5)),Label('x','B')))
    run=LabelingRun(context=ctx,rules=rules,manifest=m,output=tmp_path/'run',trainer=synthetic_same_trainer)
    expected=digest([[0,1,array_identity(np.zeros(4,bool))],[1,1,array_identity(np.zeros(4,bool))]])
    assert run.predicate_id(run.version)==expected
    result,session,_=run.chase(run.version,phase='precision_boundary')
    assert result.status=='fixpoint' and not result.positive_facts[:,1].any()
    run.release(session)
