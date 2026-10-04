"""Opt-in exact unary signed-pattern evaluator with replayable delta receipts.

No result here is presented as an executed typed Chase. On the admitted fragment
all bodies are fixed unary text tests, hence unioning their signed heads over the
fixed signed roots gives the complete closure; P & N is precisely infeasibility.
Other syntax and relation roots use the unchanged real-Chase evaluator.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from loris._engine.rdl_reference import Doc,Text,Label,Rule
from loris._engine.rlearner import array_identity,canonical,digest
from loris._engine.typed_session import seed_identity
from .artifacts import file_record,bound_path
from .candidate_generation import exact_pattern_mask
from .signed_patterns import (SignedChaseEvaluator,SignedChaseEvaluation,_immutable_binary,
    correction_diagnostics,macro_f1)

BACKEND='analytic_unary_v1'
DELTA='paper36_exact_unary_signed_fact_delta_v1'
RECEIPT='paper36_exact_unary_signed_evaluation_v1'


def supported(rules, seed, background_rules=()):
    if background_rules or seed.subset or seed.strict_subset:return False
    for r in rules:
        if not isinstance(r,Rule) or not isinstance(r.head,Label) or r.head.polarity not in ('pos','neg'):return False
        if not r.body or not any(isinstance(a,Doc) for a in r.body):return False
        for a in r.body:
            if not isinstance(a,(Doc,Text)) or a.var!=r.head.var:return False
            if isinstance(a,Text):
                p=a.to_dict()['predicate']
                if p['type'] not in ('MatchPredicate','FreqPredicate','CooccurPredicate','BeforePredicate') or p.get('sim'):return False
    return True


def _save_npz(path, **arrays):
    with Path(path).open('xb') as stream:np.savez_compressed(stream,**arrays)
    return file_record(path)


def _load_mask(record,n,body):
    path=bound_path(record)
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'mask','body'} or str(z['body'].item())!=canonical(body):raise ValueError('unary mask/body binding mismatch')
        mask=z['mask']
        if mask.dtype!=np.bool_ or mask.shape!=(n,):raise ValueError('unary full-row mask required')
        return mask


def read_delta(value,shape,*,expected_program=None,expected_gamma=None,labels=None):
    if not isinstance(value,dict) or set(value)!={'schema','shape','seed','heads'} or value['schema']!=DELTA or value['shape']!=list(shape):raise ValueError('exact unary delta schema/axis required')
    with np.load(bound_path(value['seed']),allow_pickle=False) as z:
        if set(z.files)!={'positive','negative'}:raise ValueError('complete unary signed seed required')
        p,n=z['positive'],z['negative']
    if p.dtype!=np.bool_ or n.dtype!=np.bool_ or p.shape!=tuple(shape) or n.shape!=tuple(shape):raise ValueError('invalid unary root arrays')
    if expected_gamma is not None and (expected_gamma.get('positive')!=array_identity(p) or expected_gamma.get('negative')!=array_identity(n) or expected_gamma.get('subset') or expected_gamma.get('strict_subset')):raise ValueError('unary seed differs from declared Gamma')
    if expected_program is not None and len(value['heads'])!=len(expected_program):raise ValueError('unary head count differs from program')
    for ri,h in enumerate(value['heads']):
        if set(h)!={'rule_index','label_index','polarity','body','mask'} or h['rule_index']!=ri or type(h['label_index']) is not int or not 0<=h['label_index']<shape[1] or h['polarity'] not in ('pos','neg'):raise ValueError('invalid unary head record')
        if expected_program is not None:
            r=Rule.from_dict(expected_program[ri]);body=[a.to_dict() for a in r.body]
            if not supported((r,),type('EmptyRelations',(),{'subset':(),'strict_subset':()})()) or h['body']!=body or h['polarity']!=r.head.polarity or labels[h['label_index']]!=r.head.label:raise ValueError('unary delta changed rule semantics')
        mask=_load_mask(h['mask'],shape[0],h['body'])
        (p if h['polarity']=='pos' else n)[mask,h['label_index']]=True
    return p,n


@dataclass(frozen=True)
class AnalyticEvaluation(SignedChaseEvaluation):
    delta: object=None
    def record(self):
        return {'feasible':self.feasible,'score':self.macro_f1,'diagnostics':self.diagnostics,
            'status':self.status,'state_identity':self.state_identity,'chase_receipt':self.chase_receipt,
            'conflict_certificate':self.conflict_certificate,'signed_facts':self.delta if self.feasible else None}


class AnalyticUnarySignedChaseEvaluator(SignedChaseEvaluator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._binding['execution_backend']=BACKEND
        self._binding['analytic_fragment']='empty background; no root relations; same-variable Doc/static Match|Freq|Cooccur|Before Text -> signed Label'
        self._binding['analytic_source']=file_record(__file__)
        self._dir=self.cache_dir/'analytic_unary';self._dir.mkdir(parents=True,exist_ok=True)
        self._masks={};self._seed_record=None;self.analytic_calls=0;self.real_fallback_calls=0;self.final_reference_checks=[]
        if supported((),self.seed,self.background_rules):
            path=self._dir/('seed_'+digest(seed_identity(self.seed))+'.npz')
            if path.exists():
                with np.load(path,allow_pickle=False) as z:
                    if not np.array_equal(z['positive'],self.seed.pos) or not np.array_equal(z['negative'],self.seed.neg):raise ValueError('existing unary seed mismatch')
                self._seed_record=file_record(path)
            else:self._seed_record=_save_npz(path,positive=self.seed.pos,negative=self.seed.neg)

    def _mask(self,rule):
        body=[a.to_dict() for a in rule.body];key=digest(body)
        if key not in self._masks:
            mask=np.ones(len(self._context['docs']),bool)
            for atom in rule.body:
                if isinstance(atom,Text):mask&=exact_pattern_mask(atom.predicate,self._context['docs'],self._context.get('attribute_presence'))
            path=self._dir/('mask_'+key+'.npz')
            if path.exists():
                rec=file_record(path)
                if not np.array_equal(_load_mask(rec,len(mask),body),mask):raise ValueError('existing unary mask changed')
            else:rec=_save_npz(path,mask=mask,body=np.asarray(canonical(body)))
            self._masks[key]=(mask,rec)
        return self._masks[key]

    def evaluate(self,rules,*,previous_predictions=None):
        rules=tuple(rules)
        if not supported(rules,self.seed,self.background_rules):
            self.real_fallback_calls+=1
            return super().evaluate(rules,previous_predictions=previous_predictions)
        self.analytic_calls+=1;p,n=self.seed.pos.copy(),self.seed.neg.copy();heads=[]
        for ri,r in enumerate(rules):
            mask,rec=self._mask(r);j=self._context['labels'].index(r.head.label)
            (p if r.head.polarity=='pos' else n)[mask,j]=True
            heads.append({'rule_index':ri,'label_index':j,'polarity':r.head.polarity,'body':[a.to_dict() for a in r.body],'mask':rec})
        delta={'schema':DELTA,'shape':list(p.shape),'seed':self._seed_record,'heads':heads}
        both=p&n;feasible=not both.any();status='fixpoint' if feasible else 'conflict'
        state={'schema':RECEIPT+'_state','status':status,'positive':array_identity(p),'negative':array_identity(n),'predictions':array_identity((p&~n).astype(np.float32)),'label_conflicts':np.argwhere(both)[:1].tolist(),'complete_conflict_count':int(both.sum()),'comparison_conflicts':[],'proof_materialized':False,'derivation':'unary simultaneous signed-head union; conflict state is infeasible union, not a partial typed Chase closure'}
        state['sha256']=digest(state)
        receipt={'schema':RECEIPT,'backend':BACKEND,'context_id':self._binding['context_id'],'model_snapshot_id':self.model_snapshot_id,'Gamma':digest(self._binding['gamma']),'Sigma':digest([r.to_dict() for r in rules]),'state_sha256':state['sha256'],'fact_delta':delta,'source':file_record(__file__),'actual_typed_Chase_calls':0}
        cert=None
        if not feasible:
            row,j=map(int,np.argwhere(both)[0]);sides={}
            for sign,roots in [('pos',self.seed.pos),('neg',self.seed.neg)]:
                sources=[]
                if roots[row,j]:sources.append({'origin':'validated_Gamma','row':row,'label_index':j,'polarity':sign})
                for ri,r in enumerate(rules):
                    if r.head.polarity==sign and r.head.label==self._context['labels'][j] and self._mask(r)[0][row]:sources.append({'origin':'unary_rule_application','rule_index':ri,'valuation':{r.head.var:row},'mask':heads[ri]['mask']})
                assert sources;sides[sign]=sources
            cert={'schema':RECEIPT+'_conflict','row':row,'label_index':j,'label':self._context['labels'][j],'sources':sides,'all_direct_supports_in_this_unary_program':True,'actual_typed_Chase_proof':False}
            cert['sha256']=digest(cert)
        old=self.baseline if previous_predictions is None else _immutable_binary(previous_predictions,self.baseline.shape)
        corrected=_immutable_binary((self.baseline&~n)|p) if feasible else None
        score=macro_f1(self.truth,corrected[self.indices]) if feasible else None
        changes=correction_diagnostics(self.truth,old[self.indices],corrected[self.indices]) if feasible else None
        return AnalyticEvaluation(feasible,score,_immutable_binary(p),_immutable_binary(n),corrected,changes,status,receipt,state,cert,delta)

    def verify_final(self,rules,current):
        """One real fixed-point replay for the final selected signed rule group."""
        actual=super().evaluate(tuple(rules))
        if actual.feasible!=current.feasible or actual.macro_f1!=current.macro_f1 or (actual.feasible and (not np.array_equal(actual.positive_facts,current.positive_facts) or not np.array_equal(actual.negative_facts,current.negative_facts) or not np.array_equal(actual.corrected_predictions,current.corrected_predictions))):raise AssertionError('analytic unary final group differs from complete real Chase')
        record={'analytic_calls':self.analytic_calls,'real_fallback_calls':self.real_fallback_calls,'final_real_Chase_calls':1,'facts_and_score_equal':True,'real_chase_receipt':actual.chase_receipt,'real_state_identity':actual.state_identity}
        self.final_reference_checks.append(record);return record


def verify_analytic_receipt(evaluation,protocol,program):
    receipt=evaluation['chase_receipt'];state=evaluation['state_identity'];binding=protocol['context']
    if binding.get('execution_backend')!=BACKEND or receipt.get('schema')!=RECEIPT or receipt.get('backend')!=BACKEND or receipt.get('source')!=file_record(__file__) or receipt.get('actual_typed_Chase_calls')!=0:raise ValueError('unary backend/source is not explicitly bound')
    if receipt.get('context_id')!=binding['context_id'] or receipt.get('model_snapshot_id')!=binding['model_snapshot_id'] or receipt.get('Gamma')!=digest(binding['gamma']) or receipt.get('Sigma')!=digest(program) or receipt.get('state_sha256')!=state.get('sha256') or state.get('sha256')!=digest({k:v for k,v in state.items() if k!='sha256'}):raise ValueError('unary receipt input/state binding mismatch')
    shape=(len(protocol['full_row_ids']),len(protocol['labels']))
    p,n=read_delta(receipt['fact_delta'],shape,expected_program=program,expected_gamma=binding['gamma'],labels=protocol['labels']);both=p&n
    if state.get('positive')!=array_identity(p) or state.get('negative')!=array_identity(n) or state.get('predictions')!=array_identity((p&~n).astype(np.float32)) or state.get('comparison_conflicts')!=[] or state.get('label_conflicts')!=np.argwhere(both)[:1].tolist() or state.get('complete_conflict_count')!=int(both.sum()) or state.get('status')!=('conflict' if both.any() else 'fixpoint'):raise ValueError('unary complete fact union/status differs')
    if both.any():
        c=evaluation.get('conflict_certificate',{});row,j=map(int,np.argwhere(both)[0])
        if c.get('row')!=row or c.get('label_index')!=j or c.get('sha256')!=digest({k:v for k,v in c.items() if k!='sha256'}) or not c.get('sources',{}).get('pos') or not c.get('sources',{}).get('neg'):raise ValueError('unary conflict lacks both bound direct supports')
        # Independently reconstruct the exact reported supports from bound roots,
        # masks and program, not merely a nonempty signed list.
        with np.load(bound_path(receipt['fact_delta']['seed']),allow_pickle=False) as z:roots={'pos':z['positive'],'neg':z['negative']}
        expected={}
        for sign in ('pos','neg'):
            witnesses=[]
            if roots[sign][row,j]:witnesses.append({'origin':'validated_Gamma','row':row,'label_index':j,'polarity':sign})
            for ri,(encoded,h) in enumerate(zip(program,receipt['fact_delta']['heads'])):
                r=Rule.from_dict(encoded)
                if r.head.polarity==sign and r.head.label==protocol['labels'][j] and _load_mask(h['mask'],shape[0],h['body'])[row]:witnesses.append({'origin':'unary_rule_application','rule_index':ri,'valuation':{r.head.var:row},'mask':h['mask']})
            expected[sign]=witnesses
        if c['sources']!=expected:raise ValueError('unary conflict support list differs from actual direct derivations')
    elif evaluation.get('conflict_certificate') is not None:raise ValueError('feasible unary union has a conflict certificate')
