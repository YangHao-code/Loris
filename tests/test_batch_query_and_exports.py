"""Real Chase with explicit synthetic training/provider seams; no production metrics."""
import json
import numpy as np
import pytest
from loris.labeling.runner import LabelingRun
from loris.labeling.query_policy import batch_size, parse_budget
from tests.test_labeling import fixture, synthetic_same_trainer


def configured(tmp_path, **loop):
    context, rules, manifest = fixture()
    # Three unanswered train/query rows; no truth is read by RDG.
    context['row_roles'] = ['known', 'train', 'train', 'train']
    manifest['loop'].update(label_budget=None, query_batch_size=2,
                            query_cadence='after_saturation', query_answer_mode='complete', **loop)
    manifest['feedback'].update(trusted_replay=False, new_verified_supervision='full_batch')
    answers = tmp_path/'answers'; answers.mkdir()
    for row in (1, 2, 3):
        (answers/('%08d.json' % row)).write_text(json.dumps(
            dict(row_id='r%d' % row, labels=['A','B'], targets=[0,1])))
    manifest['oracle'].update(kind='directory_simulated_human', directory=str(answers))
    return context, rules, manifest


def test_batch_closure_trains_new_full_labels_and_old_logic_together(tmp_path):
    ctx, rules, m = configured(tmp_path)
    seen=[]
    def train(**kw):
        b=kw['batch']; seen.append((b.targets.copy(), b.true_mask.copy(), b.weak_mask.copy(), b.weights.copy()))
        return synthetic_same_trainer(**kw)
    run=LabelingRun(context=ctx,rules=rules,manifest=m,output=tmp_path/'run',trainer=train)
    report=run.run()
    assert report['actual_label_requests']==2 and report['inference_counts']['chaseInc']==1
    assert report['query_policy']['batches']==1 and report['status']=='all_documents_resolved'
    # Initial logical correction and post-answer correction remain; last full
    # batch is actually trained even though chaseInc resolved the last docs.
    assert len(seen)==2 and seen[0][2][1,0] and seen[1][2][1,0]
    assert seen[1][1][2:4].all() and (seen[1][3][2:4]==1).all()
    assert seen[1][1][0,0] and not seen[1][1][0,1]  # only old Gamma's actual correction
    assert not (seen[1][1]&seen[1][2]).any()
    assert run.gamma.pos[2:,1].all() and run.gamma.neg[2:,0].all()
    assert not run.gamma.pos[1,0]  # logical correction never persisted
    events=[e['kind'] for e in run.events]
    assert events.index('supervision')<events.index('query_batch_selected')
    assert events.count('supervision')==2


def test_finite_budget_clips_batch_and_reports_unresolved(tmp_path):
    ctx, rules, m = configured(tmp_path)
    m['loop']['label_budget']=1
    run=LabelingRun(context=ctx,rules=rules,manifest=m,output=tmp_path/'run',trainer=synthetic_same_trainer)
    report=run.run()
    assert report['actual_label_requests']==1
    assert report['status']=='label_budget_exhausted' and not report['task_complete']
    assert report['unresolved_documents']==1 and report['training_attempts']==2
    assert next(e for e in run.events if e['kind']=='query_batch_selected')['rows']==[2]


def test_missing_second_answer_propagates_first_before_pending(tmp_path):
    ctx,rules,m=configured(tmp_path)
    (tmp_path/'answers/00000003.json').unlink()
    run=LabelingRun(context=ctx,rules=rules,manifest=m,output=tmp_path/'run',trainer=synthetic_same_trainer)
    report=run.run()
    assert report['status']=='pending_external_answer'
    assert report['actual_label_requests']==2 and report['inference_counts']['chaseInc']==1
    assert run.gamma.pos[2,1] and not (run.gamma.pos[3]|run.gamma.neg[3]).any()


def test_llm_human_fallback_uses_one_document_budget_and_full_answer(tmp_path):
    ctx,rules,m=configured(tmp_path)
    m['loop']['label_budget']=1
    m['oracle']['kind']='directory_llm_then_human'
    for kind, target in [('llm',[1,None]),('human',[0,1])]:
        (tmp_path/'answers'/('00000002.%s.json' % kind)).write_text(json.dumps(dict(
            row_id='r2',labels=['A','B'],targets=target,provider=kind,provenance='synthetic test')))
    run=LabelingRun(context=ctx,rules=rules,manifest=m,output=tmp_path/'run',trainer=synthetic_same_trainer)
    report=run.run()
    assert report['actual_label_requests']==1 and report['annotation_sources']['llm_proposals']==1
    assert report['annotation_sources']['human_answers']==1
    assert run.gamma.neg[2,0] and run.gamma.pos[2,1]


@pytest.mark.parametrize('mode',['periodic','final'])
def test_export_modes_preserve_training_facts_and_full_final_history(tmp_path,mode):
    reports=[]
    for export in ['all',mode]:
        ctx,rules,m=fixture();m['loop']['intermediate_exports']=export
        run=LabelingRun(context=ctx,rules=rules,manifest=m,output=tmp_path/export,trainer=synthetic_same_trainer)
        report=run.run();reports.append(report)
        assert (run.output/'final_provenance/PROOF.jsonl.gz').exists()
        for e in run.events:
            if e['kind']=='Chase':
                assert set(e['timings'])=={'setup_seconds','inference_seconds','export_seconds'}
                if e['phase']=='feedback_check':
                    assert e['full_provenance_exported']==(export=='all')
    for key in ['Gamma','state','version','actual_optimizer_updates','consistency_rejected_cells']:
        assert reports[0][key]==reports[1][key]


def test_fraction_size_uses_initial_pool_and_unlimited_default():
    assert batch_size({},100000,10000,0)==1000
    assert batch_size({},20,10,0)==10
    assert batch_size(dict(label_budget=35),100000,10000,32)==3
    assert batch_size(dict(label_budget=0),20,10,0)==0
    assert batch_size(dict(query_batch_size=5),100000,10000,0)==5
    assert parse_budget('unlimited')=='unlimited'
    with pytest.raises(ValueError):parse_budget('-1')
