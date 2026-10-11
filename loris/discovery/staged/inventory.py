"""Reuse primitive masks and declared relation joins for rule proposals."""
import collections,json,time
import numpy as np
from scipy import sparse
from . import program as rb

def family(rule):
    d=rule.to_dict(); types=[a['type'] for a in d['body']]
    if 'AttrEqual' in types:return 'cross_document'
    if 'Label' in types:return 'label_propagation'
    if 'Text' in types:return 'model_text' if 'ML' in types else 'text'
    return 'multi_model' if types.count('ML')>1 else 'single_model'


def projected_scores(features, pred, y):
    """All feature/target residual counts in sparse matrix multiplications."""
    x=sparse.csr_matrix(features,dtype=np.int32)
    tp=np.asarray(x.T @ (y & ~pred).astype(np.int32))
    fp=np.asarray(x.T @ (~y & ~pred).astype(np.int32))
    a=(y&pred).sum(0);b=(~y&pred).sum(0);c=(y&~pred).sum(0)
    dm=(rb.f1(a[None,:]+tp,b[None,:]+fp,c[None,:]-tp)-rb.f1(a,b,c)[None,:])/y.shape[1]
    du=rb.f1(a.sum()+tp,b.sum()+fp,c.sum()-tp)-rb.f1(a.sum(),b.sum(),c.sum())
    return tp,fp,.8*dm+.2*du


def shortlist(features,pred,y,limit=12):
    tp,fp,gain=projected_scores(features,pred,y)
    result=[]
    for j in range(y.shape[1]):
        valid=np.flatnonzero(tp[:,j]>0)
        # Half by actual residual F1, half by breadth of recoverable positives.
        # Conjunctions can make a broad, imprecise guard useful.
        best=sorted(valid,key=lambda i:(-gain[i,j],-tp[i,j],int(i)))[:limit]
        broad=sorted(valid,key=lambda i:(-tp[i,j]/np.sqrt(tp[i,j]+fp[i,j]),int(i)))[:limit//2]
        result.append(sorted(set(map(int,best+broad))))
    return result,dict(features=int(features.shape[1]),targets=int(y.shape[1]),
        pairs_screened=int(features.shape[1]*y.shape[1]),positive_residual_pairs=int((tp>0).sum()),
        targets_without_A_residual_positive=[int(j) for j in range(y.shape[1]) if not len(result[j])])


def pattern_inventory(payload):
    from loris.predicates import predicate_from_dict
    found={}
    for entries in payload.values():
        for entry in entries:
            d=entry['predicate']
            if d.get('sim',False):continue
            candidates=[d]
            # A frequency domain is a search domain, not merely its serialized
            # placeholder (often <= 0). Include bounded positive count guards.
            if d['type']=='FreqPredicate':
                high=(entry.get('frequency_domain') or {}).get('high',5)
                candidates += [dict(d,op='>=',eta=float(v)) for v in (1,2,3,5) if v<=high]
            for value in candidates:
                key=json.dumps(value,sort_keys=True)
                found[key]=predicate_from_dict(value)
    return [found[k] for k in sorted(found)]


def text_features(ctx,A,patterns,cache):
    from loris._engine.primitive_masks import TextPrimitiveCache
    presence=ctx.get('attribute_presence')
    presence=None if presence is None else {k:np.asarray(v)[A] for k,v in presence.items()}
    lex=TextPrimitiveCache([ctx['docs'][int(i)] for i in A],cache,attribute_presence=presence)
    columns=[];retained=[];seen=set();begin=time.monotonic()
    try:
        for i,p in enumerate(patterns):
            mask=np.asarray(lex.mask(p),bool)
            packed=np.packbits(mask,bitorder='little').tobytes()
            # Equal A masks remain different full-context predicates. Do not
            # deduplicate them by A predictions: that could lose useful guards.
            if not mask.any() or mask.all():continue
            retained.append(p);columns.append(mask.copy())
            if (i+1)%500==0:print(json.dumps(dict(stage='text_inventory_masks',done=i+1,total=len(patterns),seconds=time.monotonic()-begin)),flush=True)
        audit=lex.audit()
    finally:lex.close()
    return retained,np.stack(columns,1) if columns else np.zeros((len(A),0),bool),audit


class Factory:
    def __init__(self,ctx,runner,program,rules):
        self.ctx,self.runner,self.program,self.rules=ctx,runner,program,rules
        self.labels={v:i for i,v in enumerate(ctx['labels'])}
        self.all=(1<<program.n)-1
        self.joins={};self.seen={json.dumps(r.to_dict(),sort_keys=True):i for i,r in enumerate(rules)}

    def join(self,left,right):
        key=(left,right)
        if key not in self.joins:
            h,s=self.runner._attribute_codes(left,right)
            hr=np.flatnonzero(h>=0);sr=np.flatnonzero(s>=0)
            self.joins[key]=dict(hr=hr,sr=sr,hcodes=h[hr],scodes=s[sr],ng=int(max(h.max(),s.max()))+1)
        return self.joins[key]

    def append(self,rule):
        from loris._engine.rdl_reference import Doc,ML,Text,Label,AttrEqual
        if not isinstance(rule.head, Label) or rule.head.polarity != 'pos':
            raise ValueError('Positive Label head required')
        sig=json.dumps(rule.to_dict(),sort_keys=True)
        if sig in self.seen:return self.seen[sig]
        head=self.labels[rule.head.label];hv=rule.head.var
        dom={v:self.all for v in rule.variables};deps={v:[] for v in rule.variables};joins=[]
        for a in rule.body:
            if isinstance(a,Doc):continue
            if isinstance(a,(ML,Text)):dom[a.var] &= self.runner._static(a)
            elif isinstance(a,Label):
                j=self.labels[a.label]
                if a.polarity=='pos':deps[a.var].append(j)
                else:dom[a.var] &= self.program.neg[j]
            elif isinstance(a,AttrEqual):
                assert a.left_var!=a.right_var
                joins.append((a.left_attr,a.right_attr) if a.left_var==hv else (a.right_attr,a.left_attr))
            else:raise TypeError(type(a))
        dom[hv] &= ~self.program.neg[head]
        item=dict(head=head,headvar=hv,domains=dom,deps=deps,cache=collections.OrderedDict(),sourcevar=None)
        if len(rule.variables)>1:
            assert len(rule.variables)==2 and len(joins)==1
            item.update(self.join(*joins[0]));item['sourcevar']=next(v for v in rule.variables if v!=hv)
            assert any(deps.values()),'Cross-document Label head requires a label premise'
        idx=len(self.rules);assert idx==len(self.program.items)
        self.rules.append(rule);self.program.items.append(item);self.seen[sig]=idx
        return idx
