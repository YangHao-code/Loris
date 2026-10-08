"""Real Chase transitions and signed receiver calibration, never mock F1."""
import copy
import numpy as np
import pytest
from loris._engine.rdl_reference import Doc, Label, Rule, Text
from loris.predicates import MatchPredicate
from loris.labeling.aux_extension import attach_context, extend_program
from loris.labeling.feedback import Batch, assignments, calibrate, _channel_tables
from loris.labeling.runner import LabelingRun, Version
from tests.test_labeling import fixture, synthetic_same_trainer


def correction_fixture(policy):
    ctx, _, manifest = fixture()
    ctx['model_probabilities']['aux'][:] = .9
    old = (Rule((Doc('x'), Text('x', MatchPredicate('cnt', 'red'))), Label('x','A')),
           Rule((Doc('x'), Label('x','A')), Label('x','B')))
    manifest['dynamic']['alias'] = 'adaptive'
    manifest['aux_extension'] = dict(source_alias='aux', alias='adaptive', default_threshold=.95,
        thresholds={}, correction=dict(complement_alias='adaptive_low', positive_active=[],
        negative_rule_indices=[0], negative_default_threshold=.1, negative_thresholds={}, min_gap=.05))
    manifest['loop'].update(conflict_policy='unknown_arbitrate', unknown_semantics='retain_consequences',
                            gamma_conflict_policy='validated_gamma_wins',conflict_update_policy=policy)
    manifest['arbitration'] = dict(alias='aux',default_threshold=.5,thresholds={})
    return attach_context(ctx,manifest), extend_program(old,manifest,ctx['labels']), manifest


def test_fresh_update_withdraws_descendant_of_newly_conflicted_ancestor(tmp_path):
    outputs = {}
    for policy in ('retain_history','fresh_from_gamma'):
        ctx,rules,manifest=correction_fixture(policy)
        run=LabelingRun(context=ctx,rules=rules,manifest=manifest,output=tmp_path/policy,
                        trainer=synthetic_same_trainer)
        try:
            before, session, _ = run.chase(run.version,phase='initial')
            assert before.positive_facts[1].all()
            candidate=Version(np.full_like(run.version.probabilities,.01),
                              run.version.checkpoint,run.version.channels,'new')
            after,_,_=run.chase(candidate,phase='model_change',previous=(before,session),update_kind='chaseDelta')
            fresh,_,_=run.chase(candidate,phase='independent_fresh')
            outputs[policy]=after.positive_facts.copy()
            assert after.conflict_unknown[1,0]
            assert np.array_equal(run.tables_for(candidate)['aux'],ctx['model_probabilities']['aux'])
            if policy=='fresh_from_gamma':
                assert not after.positive_facts[1,1]
                for name in ('positive_facts','negative_facts','conflict_unknown'):
                    np.testing.assert_array_equal(getattr(after,name),getattr(fresh,name))
        finally:
            for session in run.sessions: session.close()
    assert outputs['retain_history'][1,1]  # documented V1/V2 retained consequence
    assert not outputs['fresh_from_gamma'][1,1]


def test_complement_view_cache_revalidates_mutable_probability_owner(tmp_path):
    ctx,rules,manifest=correction_fixture('fresh_from_gamma')
    run=LabelingRun(context=ctx,rules=rules,manifest=manifest,output=tmp_path/'cache',trainer=synthetic_same_trainer)
    try:
        values=np.full_like(run.version.probabilities,.2)
        version=Version(values,run.version.checkpoint,run.channels,'mutable_fixture')
        first=run.tables_for(version)['adaptive_low']
        assert run.tables_for(version)['adaptive_low'] is first
        values[:]=.3
        second=run.tables_for(version)['adaptive_low']
        assert second is not first
        np.testing.assert_array_equal(second,1.-values.astype(np.float64))
    finally:
        for session in run.sessions: session.close()


def test_negative_channel_calibrates_against_negative_event_without_inverting_training_targets():
    p=np.array([[.1],[.2],[.8],[.9]],np.float32)
    y=np.array([[0],[0],[1],[1]],np.float32)
    mask=np.ones_like(y,bool)
    batch=Batch(y,mask,np.ones_like(y),mask,np.zeros_like(mask),{})
    channels={(0,1):(0,.5),(1,2):(0,.05)}
    comp=1.-p.astype(np.float64)
    fitted, audit=calibrate(p,batch,channels,grid=[.5,.75],minimum_support_each_sign=1,
        probabilities_by_channel={(0,1):p,(1,2):comp},channel_polarities={(1,2):-1})
    assert fitted[(1,2)] == (0,.75)
    np.testing.assert_array_equal(batch.targets,y)
    assert audit['occurrences'][1]['channel_polarity']==-1
    with pytest.raises(ValueError,match='polarities'):
        calibrate(p,batch,channels,grid=[.5],minimum_support_each_sign=1,
                  channel_polarities={(9,9):-1})


def test_shared_occurrences_promote_one_table_once():
    p=np.array([[.1,.9]],np.float32)
    channels={(0,1):(0,.5),(1,1):(1,.8),(2,1):(0,.7)}
    tables=_channel_tables(p,channels,{key:p for key in channels})
    assert len({id(table) for table in tables.values()})==1
    assert tables[(0,1)].dtype==np.float64


def test_negative_predicate_round_metrics_use_the_correct_sign(tmp_path):
    ctx,rules,manifest=correction_fixture('fresh_from_gamma')
    ctx['truth']=np.array([[1,0]],bool)
    manifest['strategy']='frozen'
    manifest['loop']['label_budget']=0
    run=LabelingRun(context=ctx,rules=rules,manifest=manifest,
                    output=tmp_path/'round_metrics',trainer=synthetic_same_trainer)
    report=run.run()
    # M0=.9: no negative event, hence the associated positive decision is true.
    item=report['round_history'][0]
    assert item['dynamic_predicate_F1']['2_2']['micro_f1']==1.
    assert item['negative_event_F1']['2_2']['micro_f1']==0.


def test_negative_receiver_view_is_validated_and_supervision_stays_positive_class():
    ctx,_,_=fixture()
    p=ctx['model_probabilities']['aux']
    from types import SimpleNamespace
    closure=SimpleNamespace(status='fixpoint',positive_facts=ctx['gamma'].pos.copy(),
                            negative_facts=ctx['gamma'].neg.copy())
    channels={(0,1):(0,.9)}
    batch=assignments(ctx['gamma'],closure,None,p,channels,np.ones(len(p),bool),strategy='paper',
        weighting='uniform',probabilities_by_channel={(0,1):1.-p.astype(np.float64)},
        channel_polarities={(0,1):-1},receiver_occurrences={(0,1)})
    assert batch.targets[0,0]==1 and batch.true_mask[0,0]
    with pytest.raises(ValueError,match='differs'):
        assignments(ctx['gamma'],closure,None,p,channels,np.ones(len(p),bool),strategy='paper',
            weighting='uniform',probabilities_by_channel={(0,1):np.ones_like(p)},
            channel_polarities={(0,1):-1},receiver_occurrences={(0,1)})
