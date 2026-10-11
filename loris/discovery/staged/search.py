"""Macro-first rule replacements; every mutation recomputes closure.

Growth answers propose bundles; separate A-pruning answers admit bundles. B is
not passed to this module. Diversity counts only clauses that affect closure.
No test targets, per-label rule cap, or positive-support-five rejection.
"""
import collections
import json
import itertools
import numpy as np
import optuna
from . import inventory as prior
from . import program as rb
from . import program as fast
from . import helpers as staged_bo

FAMILIES = ('single_model','multi_model','text','model_text','label_propagation','cross_document')


def score(q):
    return .9*q['macro_f1']+.1*q['micro_f1']


def complement(predicate):
    from loris.predicates import predicate_from_dict, predicate_to_dict
    d=predicate_to_dict(predicate)
    if d['type']!='FreqPredicate':return None
    opposite={'>':'<=','>=':'<','<':'>=','<=':'>','==':'!=','=':'!=','!=':'=='}
    if d['op'] not in opposite:return None
    d['op']=opposite[d['op']]
    return predicate_from_dict(d)


def complete_complement(factory,predicate):
    """Missing attributes are unknown, not false; do not split those domains."""
    from loris._engine.rdl_reference import Text
    inverse=complement(predicate)
    if inverse is None:return None
    left=factory.runner._static(Text('x',predicate))
    right=factory.runner._static(Text('x',inverse))
    if (left&right) or (left|right)!=factory.all:return None
    return inverse


def mutate(selected,bundle):
    remove=set(bundle['remove']);add=list(bundle['add'])
    if not remove.issubset(selected):return None
    result=[i for i in selected if i not in remove]
    result.extend(i for i in add if i not in result)
    return result


class Evaluator:
    def __init__(self,program,rows,y):
        self.program=program
        self.metric=fast.Metric(rows,y,program.n)

    def __call__(self,ids):
        # Removing a supporting rule must start from Gamma, never the previous
        # closure: otherwise its obsolete facts and descendants would survive.
        state=self.program.closure(self.program.seed,ids)
        return state,self.metric.quality(self.metric.counts(state))


def effective_families(program,rules,ids,rows):
    full=program.closure(program.seed,ids);scope=rb.bits(np.isin(np.arange(program.n),rows))
    output={}
    for family in FAMILIES:
        kept=[i for i in ids if prior.family(rules[i])!=family]
        n=len(ids)-len(kept)
        if not n:continue
        sub=program.closure(program.seed,kept)
        assert all(not (b&~a) for a,b in zip(full,sub))
        losses=[fast.count((a&~b)&scope) for a,b in zip(full,sub)]
        output[family]=dict(rules=n,necessary_facts=sum(losses),affected_labels=sum(v>0 for v in losses))
    return output


def accept(program,rules,base,bundles,grow_rows,grow_y,prune_rows,prune_y,*,diverse=False,add_only=False):
    ge,ve=Evaluator(program,grow_rows,grow_y),Evaluator(program,prune_rows,prune_y)
    ids=list(base);gs,gq=ge(ids);vs,vq=ve(ids)
    gbase,vbase=gq.copy(),vq.copy();remaining=list(bundles);trace=[]
    while remaining:
        best=None
        active={f for f,v in effective_families(program,rules,ids,prune_rows).items() if v['necessary_facts']}
        for bi,b in enumerate(remaining):
            if add_only and b['remove']:continue
            candidate=mutate(ids,b)
            if candidate is None or set(candidate)==set(ids):continue
            ns,nq=ge(candidate)
            if nq['macro_f1']<gq['macro_f1']-1e-12 or nq['micro_f1']<gbase['micro_f1']-.002-1e-12:continue
            ps,pq=ve(candidate)
            if pq['macro_f1']<vq['macro_f1']-1e-12 or pq['micro_f1']<vbase['micro_f1']-.002-1e-12:continue
            gain=.5*(score(nq)-score(gq)+score(pq)-score(vq))
            if gain<-1e-12:continue
            # Equal A predictions from a syntactic guard/complement split are
            # not a diversity gain. A candidate must alter A facts somewhere.
            changed=any((a^z)&ge.metric.scope for a,z in zip(gs,ns)) or any((a^z)&ve.metric.scope for a,z in zip(vs,ps))
            if not changed:continue
            novelty=0
            if diverse:
                f=effective_families(program,rules,candidate,prune_rows)
                novelty=len({name for name,v in f.items() if v['necessary_facts']}-active)
            value=gain+.00025*novelty
            if value<=1e-12:continue
            key=(value,pq['macro_f1'],nq['macro_f1'],-len(candidate),-bi)
            if best is None or key>best[0]:best=(key,bi,candidate,ns,nq,ps,pq,gain,novelty)
        if best is None:break
        _,bi,ids,gs,gq,vs,vq,gain,novelty=best
        b=remaining.pop(bi)
        trace.append(dict(bundle=b,quality_growth=gq,quality_pruning=vq,gain=gain,new_effective_families=novelty))
    return ids,trace


def proposals(ctx,factory,base,rows,y,members,cuts,patterns,matrix,old_rules,seed=28101,ensemble='loris_calibrated_pool',trial_per_label=16,allow_replacement=True):
    from loris._engine.rdl_reference import Rule,Doc,ML,Text,Label,AttrEqual
    from loris._engine.sparse_bo import JointBinaryTPESampler
    program=factory.program;labels=ctx['labels'];k=len(labels);evaluator=Evaluator(program,rows,y)
    state,before=evaluator(base);pred=program.prediction(state,rows)
    anchors={}
    for i in base:
        r=factory.rules[i];ml=[a for a in r.body if isinstance(a,ML)]
        if len(ml)==1 and all(isinstance(a,(Doc,ML)) for a in r.body):anchors[labels.index(r.head.label)]=(i,ml[0])
    aliases=list(members)
    if ensemble not in aliases:aliases.append(ensemble)
    counts=ctx['gamma'].pos.sum(0);budgets=staged_bo.allocate(y,counts,pred,trial_per_label*k)
    all_bundles=[];records=[];seen=set();screening={}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for kind in ('model','text','label','cross'):
        features=[];by_label=[[] for _ in labels]
        if kind=='text':
            features=list(patterns)
            by_label,screening[kind]=prior.shortlist(matrix,np.zeros_like(pred),y,6)
            # Include regions rich in current FP as well as potential positives.
            error=pred&~y
            extra,_=prior.shortlist(matrix,np.zeros_like(pred),error,3)
            by_label=[sorted(set(a+b)) for a,b in zip(by_label,extra)]
        elif kind=='label':
            features=list(range(k));by_label,screening[kind]=prior.shortlist(pred,np.zeros_like(pred),y,6)
            by_label=[[s for s in xs if s!=j] for j,xs in enumerate(by_label)]
            # Full-train label implications add a source of evidence that does
            # not depend on in-sample model scores.
            train=ctx['gamma'].pos.astype(np.int32);joint=train.T@train
            precision=joint/np.maximum(counts[:,None],1)
            for j in range(k):
                implied=[i for i in range(k) if i!=j and counts[i]>0 and precision[i,j]>=.98]
                by_label[j]=sorted(set(by_label[j]+implied))
            screening[kind]['train_implication_precision']=.98
        elif kind=='cross':
            pairs=sorted({(a.left_attr,a.right_attr) if a.left_var==r.head.var else (a.right_attr,a.left_attr)
                for r in old_rules for a in r.body if isinstance(a,AttrEqual) and a.left_var!=a.right_var})
            columns=[]
            for pair in pairs:
                join=factory.join(*pair)
                for source in range(k):
                    active=np.zeros(join['ng'],bool)
                    active[join['scodes'][rb.unbits(state[source],program.n)[join['sr']]]]=True
                    mask=np.zeros(program.n,bool);mask[join['hr']]=active[join['hcodes']]
                    features.append((pair,source));columns.append(mask[rows])
            if columns:by_label,screening[kind]=prior.shortlist(np.stack(columns,1),np.zeros_like(pred),y,6)
            else:screening[kind]=dict(reason='No declared cross-document metadata relation; no invented relation')
        for j,label in enumerate(labels):
            fs=by_label[j]
            if kind!='model' and not fs:continue
            keys=['m%d'%i for i in range(len(aliases))]
            modes=(['union','replace','rescue']+(['partition'] if kind=='text' else [])) if allow_replacement else ['union']
            sampler=JointBinaryTPESampler(keys,seed=seed+j+1000*list(('model','text','label','cross')).index(kind),
                expected_body_size=1.5,startup_trials=12,acquisition_candidates=24)
            study=optuna.create_study(direction='maximize',sampler=sampler)
            default={key:0 for key in keys};warm=[]
            features_for_warm=[None] if kind=='model' else fs
            for f in features_for_warm:
                # Every shortlisted guard is tested alone; selected experts
                # are all eligible in BO, without manual global top-3 choice.
                choices=list(range(len(aliases))) if kind=='model' else [None,len(aliases)-1]
                for m in choices:
                    for mode in modes:
                        w=dict(default,mode=mode,base_cut=.65)
                        if f is not None:w['feature']=f
                        if m is not None:
                            w[keys[m]]=1;w['cut.'+keys[m]]=float(.5 if aliases[m]==ensemble else cuts[aliases[m]][j])
                        warm.append(w)
            if kind=='model' and allow_replacement:
                for a,b in itertools.combinations(range(len(aliases)),2):
                    w=dict(default,mode='rescue',base_cut=.65)
                    for m in (a,b):w[keys[m]]=1;w['cut.'+keys[m]]=float(.5 if aliases[m]==ensemble else cuts[aliases[m]][j])
                    warm.append(w)
            for w in warm:study.enqueue_trial(w)
            local=[]
            def objective(trial):
                mode=trial.suggest_categorical('mode',modes)
                active=[m for m,key in enumerate(keys) if trial.suggest_categorical(key,[0,1])]
                body=[Doc('x')];feature=None
                if kind=='model' and not active:return -1.
                if kind!='model':
                    feature=features[trial.suggest_categorical('feature',fs)]
                    if kind=='text':body.append(Text('x',feature))
                    if kind=='label':body.append(Label('x',labels[feature]))
                    if kind=='cross':
                        pair,s=feature;body.extend([Doc('y'),AttrEqual('x',pair[0],'y',pair[1]),Label('y',labels[s])])
                for m in active:
                    body.append(ML('x',aliases[m],label,trial.suggest_float('cut.'+keys[m],.001,.999)))
                candidate=Rule(tuple(body),Label('x',label));add=[factory.append(candidate)];remove=[]
                if mode!='union':
                    if j not in anchors:return -1.
                    index,anchor=anchors[j];remove=[index]
                    if mode=='rescue':
                        cut=trial.suggest_float('base_cut',anchor.eta,.999)
                        add.append(factory.append(Rule((Doc('x'),ML('x',anchor.alias,label,cut)),Label('x',label))))
                    elif mode=='partition':
                        inverse=complete_complement(factory,feature)
                        if inverse is None:return -1.
                        # Reject the identity rewrite G&M OR !G&M == M.
                        if len(active)==1:
                            atom=next(a for a in body if isinstance(a,ML))
                            if atom.alias==anchor.alias and atom.eta==anchor.eta:return -1.
                        add.append(factory.append(Rule((Doc('x'),Text('x',inverse),anchor),Label('x',label))))
                bundle=dict(remove=remove,add=sorted(set(add)),head=j,kind=kind,mode=mode)
                ids=mutate(base,bundle)
                after,q=evaluator(ids)
                gain=score(q)-score(before)
                # Keep mild growth losses as proposals too: pruning can reveal
                # useful complementary changes; admission still checks both.
                value=gain-1e-7*sum(len(factory.rules[i].body) for i in bundle['add'])
                changed=any((a^b)&evaluator.metric.scope for a,b in zip(state,after))
                signature=json.dumps([remove,bundle['add']])
                if changed and signature not in seen and q['macro_f1']>=before['macro_f1']-.002 and q['micro_f1']>=before['micro_f1']-.005:
                    seen.add(signature);local.append((value,bundle))
                return float(value if changed else -1.)
            study.optimize(objective,n_trials=len(warm)+int(budgets[j]),show_progress_bar=False)
            # Bound repeated admission cost, not the final number of rules per
            # label: preserve best proposals separately for every mutation mode.
            for mode in modes:
                chosen=sorted((v for v in local if v[1]['mode']==mode),key=lambda x:-x[0])[:8]
                all_bundles.extend(v[1] for v in chosen)
            records.append(dict(kind=kind,label=label,trials=len(study.trials),distinct_viable=len(local)))
            if (j+1)%10==0:print(json.dumps(dict(stage='replacement_BO',family=kind,label=j+1,total_labels=k,proposals=len(all_bundles))),flush=True)
    return all_bundles,dict(records=records,screening=screening,growth_rows=rows.tolist(),adaptive_label_budget=budgets.tolist(),
        final_rule_cap_per_label=None,minimum_positive_support=None)
