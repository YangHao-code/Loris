"""Independent relational maintenance review against complete recomputation."""
import copy

import numpy as np
import pytest

from loris.document import Document
from loris.labeling.inference import chase_delta, chase_inc
from loris._engine.rdl_reference import Doc, Label, LabelCompare, ML, RefGamma, Rule, Text
from loris._engine.typed_runtime import IndexedChaseRunner
from loris.predicates import MatchPredicate


def context(probabilities):
    p=np.asarray(probabilities,float)
    return dict(docs=[Document(cnt='node'+str(i),ttl='',mtd='') for i in range(len(p))],
        labels=['A','B','C','D'],gamma=RefGamma.empty(len(p),4),
        model_probabilities={'aux':p},model_decisions={})


def match(var,row):return Text(var,MatchPredicate('cnt','^node'+str(row)+'$'))


def assert_same(actual,expected):
    assert actual.status==expected.status
    if expected.status=='fixpoint':
        assert np.array_equal(actual.positive_facts,expected.positive_facts)
        assert np.array_equal(actual.negative_facts,expected.negative_facts)
        assert set(actual.subset_facts)==set(expected.subset_facts)
        assert set(actual.strict_subset_facts)==set(expected.strict_subset_facts)


def check_proof(result,runner,rules):
    visiting=set();seen=set()
    def visit(key):
        if key in seen:return
        assert key not in visiting, 'first-witness proof contains unsupported cycle'
        visiting.add(key);node=result.proof_dag[key]
        for dependency in node['dependencies']:
            assert dependency in result.proof_dag
            visit(dependency)
        if node['origin']=='rule_application':
            rule=rules[node['rule_id']]
            for atom in rule.body:
                if isinstance(atom,ML):
                    assert runner._static(atom) & (1<<node['valuation'][atom.var])
        visiting.remove(key);seen.add(key)
    for key in result.proof_dag:visit(key)


def test_relation_proofs_new_masks_multi_round_deltas_match_full(tmp_path):
    rng=np.random.RandomState(91023);successful=0
    for trial in range(24):
        ctx=context(rng.rand(3,4))
        gp=ctx['gamma'].pos.copy();gp[0,3]=True
        ctx['gamma']=RefGamma(gp,ctx['gamma'].neg)
        rules=[Rule((Doc('x'),ML('x','aux','A',.5)),Label('x','A')),
               Rule((Doc('x'),ML('x','aux','B',.5)),Label('x','B','neg')),
               Rule((Doc('x'),Label('x','A')),Label('x','C')),
               Rule((Doc('x'),Label('x','C')),Label('x','A'))]
        for _ in range(5):
            a,b=int(rng.randint(3)),int(rng.randint(3))
            body=(Doc('x'),Doc('y'),match('x',a),match('y',b))
            if rng.rand()<.7:body+=(ML('x','aux','C',float(rng.choice([.3,.5,.7]))),)
            op=str(rng.choice(['subset','eq']))
            rules.append(Rule(body,LabelCompare('x',op,'y')))
        rules.append(Rule((Doc('x'),Doc('y'),LabelCompare('x','subset','y'),Label('x','D')),
                          Label('y','C')))
        old=IndexedChaseRunner(ctx,tmp_path/('old%d'%trial));before=old(rules)
        try:
            if before.status!='fixpoint':continue
            for step in range(3):
                nxt=context(rng.rand(3,4));nxt['gamma']=ctx['gamma']
                new=IndexedChaseRunner(nxt,tmp_path/('new%d-%d'%(trial,step)))
                new_rules=[Rule(tuple(ML(a.var,a.alias,a.label,float(rng.choice([.3,.5,.7])),a.dynamic)
                                     if isinstance(a,ML) else a for a in rule.body),rule.head)
                           for rule in rules]
                try:
                    after=chase_delta(old,new,rules,new_rules,before,ctx['gamma'])
                    complete=new(new_rules,ctx['gamma']);assert_same(after,complete)
                    if after.status!='fixpoint':break
                    check_proof(after,new,new_rules);successful+=1
                except BaseException:
                    new.close();raise
                old.close();old=new;rules=new_rules;before=after
        finally:old.close()
    assert successful>=30


def test_withdrawn_edge_restored_by_builtin_transitivity_without_rule_refiring(tmp_path):
    old=context([[.9,0,0,0],[0,0,0,0],[0,0,0,0]])
    new=context([[.1,0,0,0],[0,0,0,0],[0,0,0,0]])
    gp=old['gamma'].pos.copy();gp[0,3]=True
    old['gamma']=new['gamma']=RefGamma(gp,old['gamma'].neg)
    rules=[Rule((Doc('x'),Doc('y'),match('x',0),match('y',2),ML('x','aux','A',.5)),LabelCompare('x','subset','y')),
           Rule((Doc('x'),Doc('y'),match('x',0),match('y',1)),LabelCompare('x','subset','y')),
           Rule((Doc('x'),Doc('y'),match('x',1),match('y',2)),LabelCompare('x','subset','y'))]
    a,b=IndexedChaseRunner(old,tmp_path/'a'),IndexedChaseRunner(new,tmp_path/'b')
    try:
        before=a(rules);assert before.proof_dag['subset:0:2']['rule_id']==0
        after=chase_delta(a,b,rules,rules,before,old['gamma'])
        assert_same(after,b(rules));assert after.positive_facts[2,3]
        assert after.proof_dag['subset:0:2']['origin']=='inclusion_transitivity'
        check_proof(after,b,rules)
    finally:a.close();b.close()


def test_promoting_intermediate_root_refreshes_explanatory_rule_ancestry(tmp_path):
    ctx=context([[.9,0,0,0]])
    rules=[Rule((Doc('x'),ML('x','aux','A',.5)),Label('x','A')),
           Rule((Doc('x'),Label('x','A')),Label('x','B')),
           Rule((Doc('x'),Label('x','B')),Label('x','C'))]
    runner=IndexedChaseRunner(ctx,tmp_path/'r')
    try:
        before=runner(rules);gp=ctx['gamma'].pos.copy();gp[0,1]=True
        gamma=RefGamma(gp,ctx['gamma'].neg)
        after=chase_inc(runner,rules,before,ctx['gamma'],gamma)
        assert_same(after,runner(rules,gamma))
        assert list(after.proof_dag['pos:0:1']['support_rule_ids'])==[]
        assert list(after.proof_dag['pos:0:2']['support_rule_ids'])==[2]
    finally:runner.close()


def test_proper_inclusion_withdrawal_preserves_unchanged_subset_and_Gamma(tmp_path):
    old=context([[.9,0,0,0],[0,0,0,0],[0,0,0,0]])
    new=context([[.1,0,0,0],[0,0,0,0],[0,0,0,0]])
    gp=old['gamma'].pos.copy();gp[0,3]=True
    old['gamma']=new['gamma']=RefGamma(gp,old['gamma'].neg,((1,2),),())
    body=(Doc('x'),Doc('y'),match('x',0),match('y',1))
    rules=[Rule(body+(ML('x','aux','A',.5),),LabelCompare('x','strict_subset','y')),
           Rule(body,LabelCompare('x','subset','y'))]
    a,b=IndexedChaseRunner(old,tmp_path/'a'),IndexedChaseRunner(new,tmp_path/'b')
    try:
        before=a(rules);assert (0,2) in before.strict_subset_facts
        after=chase_delta(a,b,rules,rules,before,old['gamma']);assert_same(after,b(rules))
        assert not set(after.strict_subset_facts)
        assert {(0,1),(0,2),(1,2)}<=set(after.subset_facts)
        assert after.positive_facts[:,3].all()
        check_proof(after,b,rules)
    finally:a.close();b.close()


def test_exact_instance_scope_withdraws_and_rederives_only_valid_producers(tmp_path):
    from loris._engine.instance_scope import FrozenInstanceScope,scope_corpus_identity
    from loris._engine.typed_session import digest
    ctx=context([[.9,0,0,0],[.9,0,0,0]])
    rules=[Rule((Doc('x'),ML('x','aux','A',.5)),Label('x','A')),
           Rule((Doc('x'),match('x',0)),Label('x','A')),
           Rule((Doc('x'),Label('x','A')),Label('x','B'))]
    def approval(ri):
        return dict(status='approved',action='disable_rule_instance',reason='Synthetic scope property test',
            approval_receipt_id='test-'+str(ri),reviewer='synthetic',authority_scope='exact_instance_same_corpus_program',
            parent_conflict_id='synthetic',corpus_id=scope_corpus_identity(ctx),
            program_id=digest([r.to_dict()for r in rules]),rule_index=ri,rule_sha256=digest(rules[ri].to_dict()),
            valuation={'x':0},provider='development_truth_review',development_only=True)
    a=IndexedChaseRunner(ctx,tmp_path/'a')
    b=IndexedChaseRunner(ctx,tmp_path/'b',application_scope=FrozenInstanceScope(ctx,rules,[approval(0)]))
    c=IndexedChaseRunner(ctx,tmp_path/'c',application_scope=FrozenInstanceScope(ctx,rules,[approval(0),approval(1)]))
    try:
        before=a(rules);middle=chase_delta(a,b,rules,rules,before,ctx['gamma'])
        assert_same(middle,b(rules));assert middle.proof_dag['pos:0:0']['rule_id']==1
        after=chase_delta(b,c,rules,rules,middle,ctx['gamma']);assert_same(after,c(rules))
        assert not after.positive_facts[0].any() and after.positive_facts[1,:2].all()
        check_proof(after,c,rules)
    finally:a.close();b.close();c.close()


def test_seen_model_decision_cycle_never_installs_returned_old_state(tmp_path):
    from pathlib import Path
    from loris.labeling.runner import LabelingRun
    from tests.test_labeling import fixture
    ctx,rules,manifest=fixture();manifest['loop'].update(max_inner_rounds=4,label_budget=0)
    steps=[]
    def trainer(**kwargs):
        folder=Path(kwargs['output']);folder.mkdir()
        steps.append(folder.name);p=ctx['model_probabilities']['aux'].copy()
        if len(steps)==1:p[1,0]=.9
        return p,dict(optimizer_updates=1,checkpoint={'path':str(folder/'checkpoint'),'sha256':folder.name})
    run=LabelingRun(context=ctx,rules=rules,manifest=manifest,output=tmp_path/'run',trainer=trainer)
    report=run.run()
    assert report['committed_model_updates']==1 and report['actual_optimizer_updates']==2
    assert report['checkpoint']['sha256']=='update_00001'
    assert report['status']=='label_budget_exhausted' and not report['task_complete']
    assert len(report['round_history'])==2
    assert any(e['kind']=='refinement_no_new_decisions' and e.get('reason')=='previously_seen' for e in run.events)


def test_reliability_resource_exhaustion_cannot_report_partial_rho_or_completed_task(tmp_path):
    from loris.labeling import cli
    from loris.labeling.reliability import main_calibrate
    from loris.labeling.runner import LabelingRun
    from tests.test_fit_reliability import calibration_fixture
    args,_,_=calibration_fixture(tmp_path);assert main_calibrate(args)==0
    manifest,ctx,rules,artifact,_=cli.load(args.manifest,weighting='provenance',
        source_reliability=tmp_path/'calibrated/RELIABILITY.json')
    manifest['loop']['max_chase_calls']=2
    def no_train(**kwargs):pytest.fail('incomplete rho must never train')
    run=LabelingRun(context=ctx,rules=rules,manifest=manifest,output=tmp_path/'run',trainer=no_train,reliability=artifact)
    report=run.run()
    assert report['status']=='pending_resource_limit' and not report['task_complete']
    assert report['current_Gamma_has_valid_closure'] and report['committed_model_updates']==0
    assert report['actual_Chase_calls']==2
    assert not list((tmp_path/'run').glob('supervision_*'))
