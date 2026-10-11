"""Original joint binary TPE and macro-oriented search helpers."""
import collections,json,hashlib
import numpy as np
from . import program as rb

def objective(q):return .8*q['macro_f1']+.2*q['micro_f1']


def allocate(truth,train_counts,prediction,total):
    """Every native label is visited; remaining budget follows residual need.

    No cap on rules per label. Large labels receive a sqrt-frequency share,
    while a macro-oriented uniform residual term keeps rare labels in scope.
    """
    k=truth.shape[1];minimum=8;assert total>=minimum*k
    tp=(truth&prediction).sum(0);fp=(~truth&prediction).sum(0);fn=(truth&~prediction).sum(0)
    residual=1-rb.f1(tp,fp,fn)
    root=np.sqrt(np.asarray(train_counts)+1.)
    w=.6*(.1+residual)/(.1+residual).sum()+.4*root/root.sum()
    shares=w*(total-minimum*k);extra=np.floor(shares).astype(int)
    for j in np.argsort(-(shares-extra),kind='stable')[:total-minimum*k-int(extra.sum())]:extra[j]+=1
    return extra+minimum


def make_rule(label,atoms,guard=None):
    from loris._engine.rdl_reference import Rule,Doc,ML,Label
    if guard is None:body=[Doc('x')];head=Label('x',label)
    else:body=list(guard.body);head=guard.head
    body += [ML(var,alias,tag,float(cut)) for var,alias,tag,cut in atoms]
    return Rule(tuple(body),head)


def sampler(keys,seed):
    from loris._engine.sparse_bo import JointBinaryTPESampler
    return JointBinaryTPESampler(keys,seed=seed,expected_body_size=1.5,startup_trials=12,acquisition_candidates=32)


def guard_templates(old):
    """Reuse learned lexical/relational guards; ML parts are freshly searched."""
    from loris._engine.rdl_reference import Rule,Doc,ML
    result=[];seen=set()
    for r in old:
        family=rb.family(r)
        if family=='legacy_ml':continue
        atoms=[a for a in r.body if not isinstance(a,ML)]
        # Retain all variable declarations even when their only other atom was ML.
        declared={a.var for a in atoms if isinstance(a,Doc)}
        atoms=[Doc(v) for v in sorted(set(r.variables)-declared)]+atoms
        guard=Rule(tuple(atoms),r.head)
        roles=sorted({(a.var,a.label) for a in r.body if isinstance(a,ML)}) or [(r.head.var,r.head.label)]
        key=json.dumps([guard.to_dict(),roles],sort_keys=True)
        if key in seen:continue
        seen.add(key);result.append((guard,roles,family))
    return result


def watchers(program,selected):
    result=collections.defaultdict(list)
    for i in selected:
        for j in {j for cols in program.items[i]['deps'].values() for j in cols}:result[j].append(i)
    return result


def folded_gain(program,current,after,metrics):
    gains=[]
    for m in metrics:
        a=objective(m.quality(m.counts(current)));b=objective(m.quality(m.counts(after)));gains.append(b-a)
    return float(np.mean(gains)-.25*np.std(gains))

def folds(y, row_ids, parts=3, seed=27):
    """Deterministic multilabel stratification, with no minimum support cutoff."""
    if y.ndim != 2 or len(y) != len(row_ids) or len(y) < parts:
        raise ValueError('Invalid fold axes')
    counts=y.sum(0); rarity=(y/np.maximum(counts,1)).sum(1)
    tie=[hashlib.sha256(('%d:%s' % (seed,r)).encode()).hexdigest() for r in row_ids]
    order=sorted(range(len(y)),key=lambda i:(-rarity[i],tie[i]))
    assigned=np.zeros((parts,y.shape[1]),float);sizes=np.zeros(parts,int)
    target=counts/parts;limits=np.full(parts,len(y)//parts,int);limits[:len(y)%parts]+=1
    result=np.empty(len(y),np.int64)
    for i in order:
        eligible=np.flatnonzero(sizes<limits)
        j=min(eligible,key=lambda f:(float((((assigned[f]+y[i])-target)**2/np.maximum(counts,1)).sum()
                                            -(((assigned[f])-target)**2/np.maximum(counts,1)).sum()),sizes[f],int(f)))
        result[i]=j;assigned[j]+=y[i];sizes[j]+=1
    return result
