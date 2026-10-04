"""Local LLM consistency gate, human fallback and explicit unknown labels."""
import json

import pytest

from loris.labeling.external import external_answer, signed_answer
from loris.labeling.cli import validate
from tests.test_labeling import fixture


def answer(provider='llm',targets=None):
    return dict(row_id='row-7',labels=['A','B','C'],targets=[1,None,0] if targets is None else targets,
                provider=provider,provenance='Explicit synthetic mechanism fixture; never a real annotation claim.')


def put(directory,kind,value):
    path=directory/('00000007.'+kind+'.json');path.write_text(json.dumps(value));return path


def test_consistent_llm_partial_proposal_preserves_unknown_and_skips_human(tmp_path):
    put(tmp_path,'llm',answer());put(tmp_path,'human',{'invalid':'must not be read'})
    seen=[]
    def check(value):seen.append(value);return True
    value,receipt=external_answer(7,'row-7',['A','B','C'],tmp_path,check)
    assert value['targets']==[1,None,0] and seen==[value]
    assert receipt['status']=='accepted_llm' and receipt['llm_consistency_checks']==1
    assert receipt['human_answers']==receipt['network_calls']==0


def test_inconsistent_llm_falls_back_to_human_and_keeps_declared_origin(tmp_path):
    put(tmp_path,'llm',answer('simulated_llm'))
    put(tmp_path,'human',answer('human',[None,1,None]))
    value,receipt=external_answer(7,'row-7',['A','B','C'],tmp_path,lambda value:False)
    assert value['targets']==[None,1,None] and value['provider']=='human'
    assert receipt['status']=='accepted_human' and receipt['llm_accepted']==0
    assert receipt['human_answers']==1 and receipt['simulated_human_answers']==0
    assert receipt['simulated_llm_proposals']==1


def test_no_proposal_uses_available_simulated_human_without_llm_call(tmp_path):
    put(tmp_path,'human',answer('simulated_human'))
    value,receipt=external_answer(7,'row-7',['A','B','C'],tmp_path,
        lambda value:pytest.fail('there is no LLM proposal'))
    assert value['provider']=='simulated_human'
    assert receipt['human_answers']==0 and receipt['simulated_human_answers']==1
    assert receipt['llm_proposals']==receipt['llm_consistency_checks']==0


@pytest.mark.parametrize('kind', ['llm', 'human'])
def test_complete_empty_label_set_is_an_answer_and_llm_still_requires_check(tmp_path, kind):
    put(tmp_path, kind, answer(kind, [0, 0, 0]))
    checked = []
    def check(value):
        checked.append(value['targets'])
        return True
    value, receipt = external_answer(7, 'row-7', ['A', 'B', 'C'], tmp_path, check)
    assert value['targets'] == [0, 0, 0]
    assert receipt['status'] == 'accepted_' + kind
    assert checked == ([[0, 0, 0]] if kind == 'llm' else [])


def test_inconsistent_complete_empty_llm_answer_uses_human(tmp_path):
    put(tmp_path, 'llm', answer('llm', [0, 0, 0]))
    put(tmp_path, 'human', answer('human', [0, 1, 0]))
    checked = []
    def check(value):
        checked.append(value['targets'])
        return False
    value, receipt = external_answer(7, 'row-7', ['A', 'B', 'C'], tmp_path, check)
    assert checked == [[0, 0, 0]]
    assert value['targets'] == [0, 1, 0]
    assert receipt['human_answers'] == 1 and receipt['llm_accepted'] == 0


def test_missing_proposal_and_human_are_pending_not_empty_annotation(tmp_path):
    value,receipt=external_answer(7,'row-7',['A','B','C'],tmp_path,lambda value:True)
    assert value is None and receipt['status']=='pending_external_answer'
    assert receipt['llm_proposals']==receipt['human_answers']==0


def test_invalid_llm_proposal_cannot_enter_check_and_can_use_human(tmp_path):
    bad=answer();bad['targets']=[None,None,0]
    put(tmp_path,'llm',bad);put(tmp_path,'human',answer('human'))
    value,receipt=external_answer(7,'row-7',['A','B','C'],tmp_path,
        lambda value:pytest.fail('invalid proposal must not reach Chase'))
    assert value['provider']=='human'
    assert receipt['observations'][0]['status']=='invalid_proposal'


@pytest.mark.parametrize('field,value',[('targets',[0,None,0]),('targets',[1,False,0]),
    ('targets',[1,2,0]),('targets',[1,0]),('row_id','other-row'),('labels',['B','A','C']),
    ('provider','unattributed'),('provenance','')])
def test_invalid_human_answer_cannot_be_promoted_to_Gamma(tmp_path,field,value):
    bad=answer('human');bad[field]=value;put(tmp_path,'human',bad)
    with pytest.raises(ValueError):external_answer(7,'row-7',['A','B','C'],tmp_path,lambda value:True)


def test_budget_or_execution_error_in_check_is_not_an_inconsistency(tmp_path):
    put(tmp_path,'llm',answer());put(tmp_path,'human',answer('human'))
    def exhausted(value):raise RuntimeError('Chase budget exhausted')
    with pytest.raises(RuntimeError,match='budget'):external_answer(7,'row-7',['A','B','C'],tmp_path,exhausted)
    with pytest.raises(TypeError,match='bool'):
        external_answer(7,'row-7',['A','B','C'],tmp_path,lambda value:None)


def test_public_manifest_accepts_only_explicit_directory_adapter():
    _,_,manifest=fixture();manifest['loop']['saturation_rounds']=1
    manifest['oracle'].update(kind='directory_llm_then_human',directory='answers')
    validate(manifest)
    manifest['oracle']['directory']=''
    with pytest.raises(ValueError,match='directory'):validate(manifest)


def test_human_only_ablation_never_reads_llm_file(tmp_path):
    from loris.labeling.external import human_answer
    put(tmp_path,'llm',{'malformed':'NoLLM must not inspect this'})
    put(tmp_path,'human',answer('human'))
    value,receipt=human_answer(7,'row-7',['A','B','C'],tmp_path)
    assert value['provider']=='human' and receipt['human_answers']==1
    assert receipt['llm_stage_skipped'] and receipt['llm_proposals']==receipt['llm_consistency_checks']==0


def test_legacy_simulated_directory_rejects_null_instead_of_marking_empty_answered(tmp_path):
    from loris.labeling.runner import LabelingRun
    from tests.test_labeling import synthetic_same_trainer
    ctx,rules,manifest=fixture();manifest['loop']['label_budget']=1
    manifest['loop']['query_answer_mode']='allow_partial'
    folder=tmp_path/'answers';folder.mkdir()
    manifest['oracle'].update(kind='directory_simulated_human',directory=str(folder))
    (folder/'00000002.json').write_text(json.dumps(dict(row_id='r2',labels=['A','B'],targets=[None,None])))
    run=LabelingRun(context=ctx,rules=rules,manifest=manifest,output=tmp_path/'run',trainer=synthetic_same_trainer)
    with pytest.raises(ValueError):run.oracle_answer(2,'regression_empty_legacy_answer')
    assert 2 not in run.answered and not (run.gamma.pos[2]|run.gamma.neg[2]).any()


@pytest.mark.parametrize('provider',['human','simulated_human'])
def test_public_human_only_partial_annotation_reaches_Gamma_without_llm(tmp_path,provider):
    from loris.labeling.runner import LabelingRun
    from tests.test_labeling import synthetic_same_trainer
    ctx,rules,manifest=fixture();manifest['loop']['label_budget']=1
    manifest['loop']['query_answer_mode']='allow_partial'
    folder=tmp_path/'answers';folder.mkdir()
    manifest['oracle'].update(kind='directory_human',directory=str(folder))
    validate(manifest)
    (folder/'00000002.human.json').write_text(json.dumps(dict(row_id='r2',labels=['A','B'],targets=[1,None],
        provider=provider,provenance='Synthetic workflow test; no real human quality claim.')))
    (folder/'00000002.llm.json').write_text('invalid JSON is deliberately ignored')
    run=LabelingRun(context=ctx,rules=rules,manifest=manifest,output=tmp_path/'run',trainer=synthetic_same_trainer)
    assert run.oracle_answer(2,'human_only_regression')
    assert run.gamma.pos[2,0] and not run.gamma.neg[2].any() and not run.gamma.pos[2,1]
    assert run.annotation_sources[provider+'_answers']==1 and run.annotation_sources['llm_proposals']==0
    assert run.annotation_documents[provider]=={2}
    assert all(not rows for name,rows in run.annotation_documents.items() if name!=provider)
