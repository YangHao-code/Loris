import json
from pathlib import Path
import pytest
from loris._engine.rdl_reference import Doc, Label, Rule, Text
from loris.predicates import MatchPredicate
from loris.rule_forms import parse_author_rule
from loris.discovery.runner import evaluator_pair, run_discovery
from loris.discovery.search.parallel_admission import ParallelScorer, admit_batches, decide
from tests.test_discovery_runner import inputs


def test_real_parallel_prefilter_can_miss_complement_and_resume_without_rescoring(tmp_path):
    kw=inputs(); ctx=kw['context']
    ctx['truth'][0,1]=True
    rules=[Rule((Doc('x'),Text('x',MatchPredicate('cnt','red'))),Label('x','A')),
           Rule((Doc('x'),Label('x','A')),Label('x','B'))]
    # A->B would conflict with trusted A/not-B; make that known B true too.
    from loris.chase import GammaSeed
    gp,gn=ctx['gamma'].pos.copy(),ctx['gamma'].neg.copy()
    gn[0,1]=False; gp[0,1]=True; ctx['gamma']=GammaSeed(gp,gn)
    records=[dict(number=i,rule=r.to_dict(),standalone=dict(score=1-i/2)) for i,r in enumerate(rules)]
    runner,evaluator,admission=evaluator_pair(ctx,tmp_path)
    scorer=ParallelScorer(ctx,tmp_path,workers=2)
    try:
        selected=admit_batches(order='standalone_score_desc',directory=tmp_path/'admission',
            records=records,numbers=[0,1],selected=[],evaluator=admission,rule_parser=parse_author_rule,
            score_batch=scorer,batch_size=2)
        assert len(selected)==1  # second needs first to reach the receiver
        assert decide(admission,selected,rules[1],'standalone_score_desc')['accepted']
        def forbidden(*args):pytest.fail('durably completed batches must not rescore')
        resumed=admit_batches(order='standalone_score_desc',directory=tmp_path/'admission',
            records=records,numbers=[0,1],selected=[],evaluator=admission,rule_parser=parse_author_rule,
            score_batch=forbidden,batch_size=2)
        assert [r.to_dict() for r in resumed]==[r.to_dict() for r in selected]
        assert scorer.calls>0
    finally:scorer.close();evaluator.close();runner.close()


def test_real_discovery_workers_do_not_change_fixed_batch_selection(tmp_path):
    results=[]
    for workers in [1,8]:
        out=tmp_path/str(workers)
        report=run_discovery(output=out,admission_policy='batch_prefilter',
                             admission_workers=workers,admission_batch_size=4,**inputs())
        selected=json.loads((out/'learning/SELECTED.json').read_text())
        assert not selected['all_candidates_sequentially_evaluated']
        assert report['actual_prefilter_scoring_Chase_calls_this_attempt']>0
        assert report['actual_total_Chase_calls_this_attempt']==sum(report[k] for k in (
            'actual_parent_scoring_Chase_calls_this_attempt','actual_worker_scoring_Chase_calls_this_attempt',
            'actual_final_proof_Chase_calls_this_attempt','actual_prefilter_scoring_Chase_calls_this_attempt'))
        assert json.loads((out/'PROOF_REPLAY.json').read_text())['complete_facts_conflicts_rounds_scores_equal']
        from loris.discovery.archive import _verify_cluster
        checked,_=_verify_cluster(out/'learning/cluster_00',[])
        assert checked['approximate_batch_admissions_verified']
        results.append(selected['rules'])
    assert results[0]==results[1]


def test_continue_completed_BO_preserves_prefix_and_rejects_data_changes(tmp_path):
    from loris.discovery.continue_admission import continue_admission
    source=tmp_path/'source';run_discovery(output=source,**inputs())
    # Simulate an old worker stopped midway through admission; the immutable
    # BO remains complete. Remove receipts only in this synthetic test run.
    directory=source/'learning/cluster_00/admission'
    (directory/'after.json').unlink()
    ordering=json.loads((directory/'ORDER.json').read_text())['trial_numbers']
    for n in ordering[1:]:(directory/('candidate_%05d.json'%n)).unlink()
    kw=inputs();kw.update(admission_policy='batch_prefilter',admission_workers=2,admission_batch_size=2)
    out=tmp_path/'continued'
    result=continue_admission(source=source,output=out,**kw)
    prefix=json.loads((out/'PRESERVED_PREFIX.json').read_text())
    assert prefix['consumed_by_cluster']==[1] and result['new_BO']==0
    selected=json.loads((out/'learning/SELECTED.json').read_text())['rules']
    assert selected[:len(prefix['rules'])]==prefix['rules']
    assert continue_admission(source=source,output=out,resume=True,**kw)==result
    from loris.discovery.archive import status
    assert status(out)['status']=='complete_verified_admission_continuation'
    kw=inputs();kw.update(admission_policy='batch_prefilter')
    kw['context']['row_ids'][0]='changed-row'
    with pytest.raises(ValueError,match='scientific input differs'):
        continue_admission(source=source,output=tmp_path/'wrong',**kw)


def test_random_survivor_recheck_rejects_new_combination_conflict(tmp_path):
    ctx=inputs()['context']
    from loris.document import Document
    ctx['docs'][4]=Document(cnt='target-only',ttl='4',mtd='')
    body=(Doc('x'),Text('x',MatchPredicate('cnt','target-only')))
    rules=[Rule(body,Label('x','A')),Rule(body,Label('x','A','neg'))]
    records=[dict(number=i,rule=r.to_dict(),standalone=dict(score=0.)) for i,r in enumerate(rules)]
    engine,evaluator,admission=evaluator_pair(ctx,tmp_path)
    scorer=ParallelScorer(ctx,tmp_path,2)
    try:
        selected=admit_batches(order='random_consistent',directory=tmp_path/'admission',
            records=records,numbers=[0,1],selected=[],evaluator=admission,
            rule_parser=parse_author_rule,score_batch=scorer,batch_size=2)
        receipt=json.loads((tmp_path/'admission/candidate_00001.json').read_text())
        assert len(selected)==1 and receipt['prefilter_decision']['accepted']
        assert receipt['rechecked_after_prefix_change'] and not receipt['decision']['accepted']
        assert receipt['decision']['reason']=='conflict'
    finally:scorer.close();evaluator.close();engine.close()


def test_frozen_input_loader_authenticates_assets_without_rerunning_selector(tmp_path,monkeypatch):
    from tests.test_discovery_variants import discovery_manifest
    from loris.discovery.manifest import load_inputs
    from loris.discovery.continue_admission import load_frozen_inputs,continue_admission
    from loris.discovery import model_selection
    manifest=discovery_manifest(tmp_path)
    original=load_inputs(manifest);source=tmp_path/'old_run'
    run_discovery(output=source,**original)
    monkeypatch.setattr(model_selection,'verify_selection_output',lambda *a:pytest.fail('frozen pool must not rerun selector'))
    frozen=load_frozen_inputs(manifest,source,workers=2,batch_size=8)
    assert frozen['si']==original['si']
    assert frozen['input_binding']['selector_recomputed'] is False
    result=continue_admission(source=source,output=tmp_path/'continued',**frozen)
    assert result['new_BO']==0 and result['status']=='complete'
    manifest.write_text(manifest.read_text()+'\n')
    with pytest.raises(ValueError,match='manifest or original bound inputs changed'):
        load_frozen_inputs(manifest,source)
