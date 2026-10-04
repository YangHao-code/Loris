import copy
import json
import random
from pathlib import Path

import pytest
from loris._engine.rdl_reference import Doc, Label, Rule, Text
from loris._engine.rlearner import digest
from loris.predicates import MatchPredicate
from loris.rule_forms import parse_author_rule
from loris.discovery.runner import evaluator_pair, run_discovery
from loris.discovery.search.admission import order_trials
from loris.discovery.search.parallel_admission import ParallelScorer, admit_batches
from loris.discovery.search.standalone_reuse import BOStandaloneScorer, authenticated_scores
from tests.test_discovery_runner import inputs


def records_for(path, rules, evaluator):
    records = [dict(number=i, rule=r.to_dict(), standalone=evaluator.standalone(r))
               for i, r in enumerate(rules)]
    protocol = dict(evaluator=evaluator.binding, whole_program_prefix_used_in_BO=False,
                    historical_candidates_loaded=False)
    path.mkdir(parents=True)
    (path/'protocol.json').write_text(json.dumps(protocol))
    complete = dict(protocol_id=digest(protocol), complete_trials=len(records),
                    all_trial_records_sha256=digest(records))
    (path/'BO_COMPLETE.json').write_text(json.dumps(complete))
    return records


def candidates(seed):
    rng = random.Random(seed)
    rules = [Rule((Doc('x'), Text('x', MatchPredicate('cnt', 'red'))), Label('x','A'))]
    for _ in range(18):
        # Mix fixed fit-precision rejects, valid proposals, signed conflicts,
        # duplicates, Label-premise interactions and provably empty bodies.
        token = rng.choice(['red', 'blue', 'missing'])
        premise = (Text('x', MatchPredicate('cnt',token)) if rng.random() < .75
                   else Label('x',rng.choice(['A','B']),rng.choice(['pos','neg'])))
        rules.append(Rule((Doc('x'), premise),Label('x',rng.choice(['A','B']),rng.choice(['pos','neg']))))
    return rules


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('order', ['standalone_score_desc','random_consistent'])
def test_same_full_decisions_order_and_final_program(seed, order, tmp_path):
    context=inputs()['context']
    engine,evaluator,_=evaluator_pair(context,tmp_path/'bo',gamma_protection=True,fit_precision_floor=.9)
    try:records=records_for(tmp_path/'cluster', candidates(seed), evaluator)
    finally:evaluator.close();engine.close()
    results=[];calls=[]
    for name,cls in [('original',ParallelScorer),('reuse',BOStandaloneScorer)]:
        out=tmp_path/name
        engine,evaluator,admission=evaluator_pair(context,out,gamma_protection=True,fit_precision_floor=.9)
        scorer=cls(context,out,workers=2,gamma_protection=True,fit_precision_floor=.9)
        try:
            if name=='reuse':scorer.register_cluster(tmp_path/'cluster',records,evaluator)
            nums=order_trials(records,order)
            selected=admit_batches(order=order,directory=out/'admission',records=records,numbers=nums,
                selected=[],evaluator=admission,rule_parser=parse_author_rule,score_batch=scorer,batch_size=3)
            receipts=[json.loads((out/'admission'/('candidate_%05d.json'%n)).read_text()) for n in nums]
            results.append(dict(rules=[r.to_dict() for r in selected],receipts=receipts,
                                final=evaluator.evaluate(selected)))
            calls.append(scorer.calls)
        finally:scorer.close();evaluator.close();engine.close()
    def scientific(value):
        if isinstance(value,dict):return {k:scientific(v) for k,v in value.items() if k!='n_rounds'}
        if isinstance(value,list):return [scientific(v) for v in value]
        return value
    # Rounds are an execution statistic of the cache/warm-start schedule, as
    # in semantic_result_projection; all other receipt fields must agree.
    assert scientific(results[0])==scientific(results[1])
    if order=='standalone_score_desc':assert calls[1]<calls[0]


@pytest.mark.parametrize('damage', ['records','binding','order','payload','score'])
def test_authentication_rejects_tampering(tmp_path,damage):
    ctx=inputs()['context'];eng,e,_=evaluator_pair(ctx,tmp_path/'eval',gamma_protection=True,fit_precision_floor=.9)
    try:
        path=tmp_path/'cluster';records=records_for(path,candidates(0),e)
        if damage=='records':records[0]['standalone']['score']+=.01
        elif damage=='binding':
            p=json.loads((path/'protocol.json').read_text());p['evaluator']['labels']=['wrong']
            (path/'protocol.json').write_text(json.dumps(p))
            c=json.loads((path/'BO_COMPLETE.json').read_text());c['protocol_id']=digest(p)
            (path/'BO_COMPLETE.json').write_text(json.dumps(c))
        else:
            if damage=='order':records[0]['number']=42
            if damage=='payload':records[0]['standalone']['evaluation']['program_key']='f'*64
            if damage=='score':records[0]['standalone']['score']+=.01
            c=json.loads((path/'BO_COMPLETE.json').read_text());c['all_trial_records_sha256']=digest(records)
            (path/'BO_COMPLETE.json').write_text(json.dumps(c))
        with pytest.raises(ValueError):authenticated_scores(path,records,e)
    finally:e.close();eng.close()


def test_actual_completed_BO_binding_registration(tmp_path):
    ctx=inputs();ctx['gamma_conflict_policy']='soft_fit_precision'
    out=tmp_path/'real';run_discovery(output=out,**ctx)
    folder=out/'learning/cluster_00'
    records=[json.loads(p.read_text()) for p in sorted(folder.glob('trial_*.json'))]
    protocol=json.loads((folder/'protocol.json').read_text())
    floor=protocol['evaluator']['fit_precision_gate']['floor']
    eng,e,_=evaluator_pair(ctx['context'],tmp_path/'check',gamma_protection=True,fit_precision_floor=floor)
    try:
        values,receipt=authenticated_scores(folder,records,e)
        assert receipt['complete_trials']==4 and values
    finally:e.close();eng.close()


def test_continuation_captures_only_whole_committed_batches(tmp_path):
    from loris.discovery.continue_admission import continue_admission
    from loris.discovery.archive import status
    kw=inputs();kw.update(budget_per_cluster=8,admission_policy='batch_prefilter',
                         admission_workers=2,admission_batch_size=2)
    source=tmp_path/'source';run_discovery(output=source,**kw)
    expected=json.loads((source/'learning/SELECTED.json').read_text())['rules']
    d=source/'learning/cluster_00/admission';(d/'after.json').unlink()
    # Last two receipts exist, but their atomic batch commit was not written.
    (d/'batch_00003.json').unlink()
    out=tmp_path/'continue'
    result=continue_admission(source=source,output=out,reuse_BO_standalone=True,**kw)
    prefix=json.loads((out/'PRESERVED_PREFIX.json').read_text())
    assert prefix['consumed_by_cluster']==[6]
    assert result['new_BO']==0 and result['complete_BO_trials']==8
    assert json.loads((out/'learning/SELECTED.json').read_text())['rules']==expected
    assert json.loads((out/'BO_SOURCE_ARCHIVE.json').read_text())['all_proposed_trials_retained']
    assert status(out)['status']=='complete_verified_admission_continuation'
    assert continue_admission(source=source,output=out,reuse_BO_standalone=True,resume=True,**kw)==result


def test_committed_prefix_rejects_changed_receipts(tmp_path):
    from loris.discovery.search.committed_prefix import capture_committed_prefix
    kw=inputs();kw.update(admission_policy='batch_prefilter',admission_batch_size=2)
    source=tmp_path/'source';run_discovery(output=source,**kw)
    folder=source/'learning/cluster_00';protocol=json.loads((folder/'protocol.json').read_text())
    d=folder/'admission';p=next(d.glob('candidate_*.json'));r=json.loads(p.read_text())
    r['selected_after']+=1;p.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        capture_committed_prefix(source,1,'standalone_score_desc',2,protocol['evaluator'])
