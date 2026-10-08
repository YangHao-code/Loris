"""Query-only extension; real Chase and original answer/budget semantics."""
import json

import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, Rule, Text, RefGamma
from loris.labeling.cli import validate
from loris.labeling.query_policy import batch_size
from loris.labeling.query_selection import configuration, review_candidates, rank_review_candidates
from loris.labeling.runner import LabelingRun, Version
from loris.predicates import MatchPredicate
from tests.test_labeling import fixture


def plan(pos, p, *, mean=3., thresholds=None, gp=None, gn=None, unknown=None, mode='evidence'):
    pos=np.asarray(pos,bool);p=np.asarray(p,float);z=np.zeros_like(pos)
    return review_candidates(positive=pos,negative=z,unknown=z if unknown is None else unknown,
        gamma_positive=z if gp is None else gp,gamma_negative=z if gn is None else gn,
        query_rows=np.ones(len(pos),bool),aux_probabilities=p,
        aux_thresholds=np.full(pos.shape[1],.5) if thresholds is None else thresholds,
        train_mean=mean,eligibility_mode=mode)


def configured(tmp_path,*,name='run',budget=1,audit=0,oracle='directory_simulated_human'):
    ctx,rules,m=fixture()
    ctx.pop('truth');ctx.pop('receiver_indices')
    m['strategy']='frozen'
    m['query']=dict(strategy='review',seed=42,review=dict(eligibility_mode='evidence',audit_fraction=audit))
    m['loop'].update(query_pool='needs_review',query_cadence='each_closure',query_batch_size=1,
                     label_budget=budget,query_answer_mode='complete')
    answers=tmp_path/(name+'_answers');answers.mkdir()
    m['oracle'].update(kind=oracle,directory=str(answers))
    return ctx,rules,m,answers


def forbidden_training(**kwargs):
    raise AssertionError('frozen query fixture must not train')


def make_run(tmp_path,**kwargs):
    ctx,rules,m,answers=configured(tmp_path,**kwargs)
    return LabelingRun(context=ctx,rules=rules,manifest=m,output=tmp_path/kwargs.get('name','run'),
                       trainer=forbidden_training),answers


def close(run):
    for session in run.sessions:
        session.close()


def test_natural_two_far_negative_not_selected_but_near_missing_and_FP_are():
    p=np.array([[.9,.9,.0001],[.9,.9,.0095],[.9,.9,.001]])
    pred=np.array([[1,1,0],[1,1,0],[1,1,1]],bool)
    r=plan(pred,p,thresholds=np.array([.5,.5,.01]))
    assert r['eligible'].tolist()==[False,True,True]
    assert r['reasons']['low_cardinality_near_boundary'].tolist()==[False,True,False]
    assert r['reasons']['possible_false_positive'].tolist()==[False,False,True]
    assert not r['parameters']['cardinality_alone_is_eligibility']
    assert not r['parameters']['deployed_predicate_thresholds_are_precision_certificates']
    assert not r['parameters']['truth_used']


def test_hard_missing_and_uncovered_original_pool_are_retained():
    r=plan([[1,0,0],[0,0,0]],[[.9,.8,.001],[.001,.001,.001]])
    assert r['eligible'].tolist()==[True,True]
    assert r['reasons']['proposed_missing_label'].tolist()==[True,False]
    assert r['reasons']['no_positive_label'].tolist()==[False,True]


def test_fully_trusted_docs_exit_and_known_partial_cells_do_not_create_disagreement():
    gp=np.array([[1,1,0],[1,0,0]],bool);gn=np.array([[0,0,1],[0,0,0]],bool)
    r=plan([[1,1,0],[1,1,0]],[[.1,.9,.001],[.1,.9,.001]],gp=gp,gn=gn)
    assert r['unanswered'].tolist()==[False,True]
    assert not r['eligible'].any()


def test_fully_trusted_natural_one_and_two_label_docs_are_excluded():
    gp=np.array([[1,0,0],[1,1,0]],bool)
    r=plan(gp,[[.01,.99,.99],[.01,.01,.99]],gp=gp,gn=~gp)
    assert not r['unanswered'].any() and not r['eligible'].any()


def test_conflict_is_explicit_not_all_unselected_negative_cells():
    unknown=np.zeros((2,3),bool);unknown[1,2]=True
    r=plan([[1,1,0],[1,1,0]],[[.9,.9,.001],[.9,.9,.001]],unknown=unknown)
    assert r['eligible'].tolist()==[False,True]
    assert r['reasons']['conflict'].tolist()==[False,True]


def test_relative_nearness_does_not_treat_all_low_ASL_probabilities_as_near():
    r=plan([[1,0],[1,0]],[[.9,.0095],[.9,.00001]],mean=2.,thresholds=np.array([.5,.01]))
    assert r['eligible'].tolist()==[True,False]
    assert r['components']['near_boundary_labels'].tolist()==[1,0]
    zero=plan([[1,1]],[[.9,0]],mean=3.,thresholds=np.array([.5,0.]))
    assert not zero['reasons']['low_cardinality_near_boundary'].any()


def test_new_options_really_pass_whitelist_and_wrong_strategy_pool_reject(tmp_path):
    _,_,m,_=configured(tmp_path)
    validate(m)
    m['query']['review']['near_boundary_ratio']=.95;validate(m)
    for key,value in [('near_boundary_ratio',0),('near_boundary_ratio',True),('eligibility_mode','risk_review')]:
        x=json.loads(json.dumps(m));x['query']['review'][key]=value
        with pytest.raises(ValueError):validate(x)
    m['loop']['query_pool']='uncovered'
    with pytest.raises(ValueError,match='needs_review'):validate(m)


def test_batch_budget_and_random_audit_are_original():
    loop=dict(query_batch_size=None,label_budget=42)
    assert batch_size(loop,4096,4096,0)==41
    assert batch_size(loop,968,968,0)==32
    assert batch_size(loop,4096,3,0)==3
    assert batch_size(loop,4096,4096,41)==1
    r=plan(np.tile([1,1,0],(10,1)),np.tile([.9,.9,.001],(10,1)))
    assert not r['eligible'].any()
    rows,a=rank_review_candidates(r,limit=4,seed=42,audit_fraction=.125)
    assert len(rows)==len(set(rows))==4 and rows==a['random_audit_rows']
    assert rank_review_candidates(r,limit=4,seed=42,audit_fraction=0)[0]==[]


def test_actual_loop_queries_already_positive_document_with_missing_evidence(tmp_path):
    ctx,rules,m,answers=configured(tmp_path)
    ctx['model_probabilities']['aux'][1,1]=.9
    run=LabelingRun(context=ctx,rules=rules,manifest=m,output=tmp_path/'run',trainer=forbidden_training)
    (answers/'00000001.json').write_text(json.dumps(dict(row_id='r1',labels=['A','B'],targets=[1,1])))
    report=run.run()
    assert report['actual_label_requests']==1 and run.gamma.pos[1].all()
    assert report['actual_optimizer_updates']==0
    assert 'risk_query_policy' not in report and 'training_pause' not in report
    assert 'query_run_completed' not in report
    assert report['status']=='label_budget_exhausted'  # Original stopping status.


def test_has_query_work_uses_expanded_pool_after_all_query_docs_have_positive_labels(tmp_path):
    ctx,rules,m,answers=configured(tmp_path)
    # Existing rules label both query docs A; Aux strongly proposes missing B.
    ctx['model_probabilities']['aux'][:,0]=.9
    ctx['model_probabilities']['aux'][2,1]=.9
    run=LabelingRun(context=ctx,rules=rules,manifest=m,output=tmp_path/'has_work',trainer=forbidden_training)
    try:
        result,session,_=run.chase(run.version,phase='query_test')
        assert result.positive_facts[run.query_rows].any(1).all()
        assert run.has_query_work(result)
        assert run.eligible_query_rows(result).tolist()==[False,False,True,False]
        assert run.rdg(result,session,limit=1)==2
    finally:close(run)


def test_partial_legacy_answered_membership_does_not_block_query_but_full_Gamma_does(tmp_path):
    run,answers=make_run(tmp_path,oracle='directory_human',budget=2)
    run.manifest['loop']['query_answer_mode']='allow_partial'
    path=answers/'00000001.human.json'
    def provide(values):
        path.write_text(json.dumps(dict(row_id='r1',labels=['A','B'],targets=values,provider='human',
            provenance='explicit synthetic provided human fixture')))
    try:
        result,session,_=run.chase(run.version,phase='partial_test')
        provide([1,None]);assert run.oracle_answer(1,'partial',result=result,session=session,version=run.version)
        assert 1 in run.answered and not (run.gamma.pos[1]|run.gamma.neg[1]).all()
        # B is now strongly proposed by current Aux, which keeps the partial
        # row eligible without asking again about its known A label.
        p=run.version.probabilities.copy();p[1,1]=.9
        current=Version(p,dict(path='fixture2',sha256='synthetic2'),run.version.channels,'synthetic2')
        result,session,_=run.chase(current,phase='current_partial_model')
        assert run.eligible_query_rows(result)[1]
        provide([1,0]);assert run.oracle_answer(1,'complete',result=result,session=session,version=current)
        assert (run.gamma.pos[1]|run.gamma.neg[1]).all()
        assert not run.eligible_query_rows(result)[1]
        with pytest.raises(ValueError,match='complete answers'):
            run.oracle_answer(1,'repeat',result=result,session=session,version=current)
        assert run.queries==2  # Original per-request budget semantics, no new polling protocol.
    finally:close(run)


def test_current_closure_probability_used_even_when_global_version_is_initial(tmp_path):
    ctx,rules,m,_=configured(tmp_path)
    ctx['model_probabilities']['aux'][:,0]=.9
    run=LabelingRun(context=ctx,rules=rules,manifest=m,output=tmp_path/'current',trainer=forbidden_training)
    try:
        old_id=run.version.identity
        p=run.version.probabilities.copy();p[2,1]=.9
        current=Version(p,dict(path='new',sha256='new'),run.version.channels,'new')
        result,session,_=run.chase(current,phase='current_query_model')
        assert run.version.identity==old_id  # Training/global version behavior unchanged.
        assert run.eligible_query_rows(result)[2]
        assert run.rdg(result,session,limit=1)==2
    finally:close(run)


def test_remaining_undecided_cells_are_not_misclassified_as_conflicts(tmp_path):
    ctx,rules,m,_=configured(tmp_path)
    ctx['model_probabilities']['aux'][:,0]=.9
    run=LabelingRun(context=ctx,rules=rules,manifest=m,output=tmp_path/'unknown',trainer=forbidden_training)
    try:
        result,_,_=run.chase(run.version,phase='query_test')
        # Simulate a typed result retaining ordinary unselected negatives.
        result.remaining_unknown=np.ones((run.n,run.k),bool)
        r=run.review_plan(result)
        assert not r['reasons']['conflict'].any()
        assert not r['eligible'][1] and not r['eligible'][2]
    finally:close(run)


def test_early_decision_only_return_exposes_current_query_probability_without_changing_global_version(tmp_path):
    ctx,rules,m,_=configured(tmp_path)
    ctx['model_probabilities']['aux'][:,0]=.9
    m['loop'].update(conflict_policy='unknown_arbitrate',unknown_semantics='retain_consequences',
                     gamma_conflict_policy='validated_gamma_wins')
    m['arbitration']=dict(alias='aux',default_threshold=.5,thresholds={})
    run=LabelingRun(context=ctx,rules=rules,manifest=m,output=tmp_path/'early',trainer=forbidden_training)
    try:
        original=run.version.identity
        p=run.version.probabilities.copy();p[2,1]=.9
        current=Version(p,dict(path='current',sha256='current'),run.version.channels,'current')
        # A same-round check must use a baseline built under the same model.
        # The global self.version remains initial, exercising the stale-query
        # fallback bug without violating the actual engine context contract.
        result,session,_=run.chase(current,phase='early_base_current_model')
        tested,_,_=run.chase(current,phase='LLM_proposal_validation',previous=(result,session),
            update_kind='check',decision_only=True,check_candidates=(np.zeros_like(run.gamma.pos),np.zeros_like(run.gamma.neg)))
        assert tested.query_aux_probabilities[2,1]==pytest.approx(.9)
        assert run.version.identity==original
        assert run.review_plan(tested)['reasons']['proposed_missing_label'][2]
    finally:close(run)


def test_default_random_audits_keep_original_budget_behavior_outside_evidence_pool(tmp_path):
    ctx,rules,m,_=configured(tmp_path,audit=.125)
    ctx['model_probabilities']['aux'][:,0]=.9
    run=LabelingRun(context=ctx,rules=rules,manifest=m,output=tmp_path/'audits',trainer=forbidden_training)
    try:
        result,session,_=run.chase(run.version,phase='audit_pool')
        assert not run.review_plan(result)['eligible'].any()
        assert run.has_query_work(result) and run.eligible_query_rows(result)[run.query_rows].all()
        assert run.rdg(result,session,limit=1) in (1,2)
        run.manifest['query']['review']['audit_fraction']=0
        assert not run.has_query_work(result)
    finally:close(run)


def test_LLM_first_original_consistency_check_and_Gamma_commit_are_preserved(tmp_path):
    run,answers=make_run(tmp_path,oracle='directory_llm_then_human',budget=1)
    (answers/'00000001.llm.json').write_text(json.dumps(dict(row_id='r1',labels=['A','B'],targets=[1,0],
        provider='llm',provenance='explicit synthetic LLM response; original consistency path')))
    try:
        result,session,_=run.chase(run.version,phase='llm_test')
        assert run.oracle_answer(1,'LLM_first',result=result,session=session,version=run.version)
        assert run.gamma.pos[1,0] and run.gamma.neg[1,1] and 1 in run.answered
        assert run.annotation_sources['llm_accepted']==1
        assert any(e['kind']=='Chase' and e.get('phase')=='LLM_proposal_validation' for e in run.events)
        assert not hasattr(run,'risk_queries')
    finally:close(run)
