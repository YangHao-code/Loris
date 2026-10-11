"""Positive-head rule discovery using exact bitsets and attribute joins.

Restricted deliberately to the positive unary/text/model/attribute-join fragment. Unsupported atoms,
negative heads or nonidentity Gamma relations fail closed. No target answers.
"""
import collections
import numpy as np

POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)


def bits(mask):
    return int.from_bytes(np.packbits(mask, bitorder='little').tobytes(), 'little')


def unbits(value, n):
    return np.unpackbits(np.frombuffer(int(value).to_bytes((n+7)//8, 'little'), np.uint8), bitorder='little')[:n].astype(bool)


def count(value):
    if not value: return 0
    if hasattr(int, 'bit_count'): return int(value).bit_count()
    return int(POP[np.frombuffer(int(value).to_bytes((value.bit_length()+7)//8, 'little'), np.uint8)].sum())


def family(rule):
    types={a['type'] for a in rule.to_dict()['body']}
    if 'AttrEqual' in types: return 'relation'
    if 'Label' in types: return 'label'
    if 'Text' in types: return 'text'
    return 'legacy_ml'


class Program:
    def __init__(self, runner, rules):
        from loris._engine.rdl_reference import Doc, Text, ML, Label, AttrEqual
        self.n,self.k=runner.n,runner.k
        gamma=runner.gamma
        if (gamma.pos & gamma.neg).any():
            raise ValueError('Conflicting Gamma is outside positive discovery')
        if any(a != b for a, b in gamma.subset) or gamma.strict_subset:
            raise ValueError('Nonidentity Gamma relations require the general chase')
        self.seed=tuple(bits(gamma.pos[:,j]) for j in range(self.k))
        self.neg=tuple(bits(gamma.neg[:,j]) for j in range(self.k))
        self.items=[];joins={}
        for rule in rules:
            if not isinstance(rule.head, Label) or rule.head.polarity != 'pos':
                raise ValueError('Positive Label heads required for this discovery fragment')
            if not all(isinstance(a, (Doc, Text, ML, Label, AttrEqual)) for a in rule.body):
                raise ValueError('Unsupported atom in positive discovery fragment')
            if not 1 <= len(rule.variables) <= 2:
                raise ValueError('At most two joined document variables supported')
            head=runner.label_index[rule.head.label];hv=rule.head.var
            normalized,key=runner._rule(rule);p=runner._prepare(normalized,key)
            dom=dict(p['domains']);deps={v:[] for v in rule.variables}
            for v,atoms in p['unary'].items():
                for a in atoms:
                    j=runner.label_index[a.label]
                    if a.polarity=='neg': dom[v] &= self.neg[j]
                    else: deps[v].append(j)
            cross=[]
            for a in p['binary']:
                assert isinstance(a,AttrEqual)
                c=runner._attribute_codes(a.left_attr,a.right_attr)
                if a.left_var==a.right_var:
                    dom[a.left_var] &= bits((c[0]>=0)&(c[0]==c[1]))
                else: cross.append((a,c))
            dom[hv] &= ~self.neg[head]
            record=dict(head=head,headvar=hv,domains=dom,deps=deps,cache=collections.OrderedDict())
            if len(rule.variables)==1:
                assert not cross
                record['sourcevar']=None
            else:
                assert cross, 'Unjoined two-variable rules are outside this fast fragment'
                sv=next(v for v in rule.variables if v!=hv)
                jk=tuple((a.left_attr if a.left_var==hv else a.right_attr,
                          a.right_attr if a.left_var==hv else a.left_attr) for a,c in cross)
                if jk not in joins:
                    hc=np.stack([c[0] if a.left_var==hv else c[1] for a,c in cross],axis=1)
                    sc=np.stack([c[1] if a.left_var==hv else c[0] for a,c in cross],axis=1)
                    joined=np.concatenate([hc,sc]);valid=(joined>=0).all(1)
                    _,inv=np.unique(joined,axis=0,return_inverse=True)
                    inv[~valid]=-1;joins[jk]=(inv[:self.n],inv[self.n:],int(inv.max())+1)
                hc,sc,ng=joins[jk]
                hr=np.flatnonzero(unbits(dom[hv],self.n)&(hc>=0))
                sr=np.flatnonzero(unbits(dom[sv],self.n)&(sc>=0))
                record.update(sourcevar=sv,hr=hr,sr=sr,hcodes=hc[hr],scodes=sc[sr],ng=ng)
            self.items.append(record)

    def firing(self, i, state):
        r=self.items[i];key=tuple(state[j] for v in sorted(r['deps']) for j in r['deps'][v])
        if key in r['cache']: return r['cache'][key]
        domains=dict(r['domains'])
        for v,cols in r['deps'].items():
            for j in cols: domains[v] &= state[j]
        hv,sv=r['headvar'],r['sourcevar']
        if sv is None: value=domains[hv]
        elif not domains[hv] or not domains[sv]: value=0
        else:
            active=np.zeros(r['ng'],bool)
            active[r['scodes'][unbits(domains[sv],self.n)[r['sr']]]]=True
            yes=active[r['hcodes']] & unbits(domains[hv],self.n)[r['hr']]
            mask=np.zeros(self.n,bool);mask[r['hr'][yes]]=True;value=bits(mask)
        if len(r['cache'])>=32:r['cache'].popitem(last=False)
        r['cache'][key]=value
        return value

    def closure(self, state, ids):
        current=list(state)
        while True:
            changed=False
            for i in ids:
                j=self.items[i]['head'];new=self.firing(i,current)&~current[j]
                if new:current[j]|=new;changed=True
            if not changed: return tuple(current)

    def incremental(self, state, selected, candidate, watchers):
        current=list(state);queue=collections.deque([candidate]);queued={candidate}
        while queue:
            i=queue.popleft();queued.remove(i)
            j=self.items[i]['head'];added=self.firing(i,current)&~current[j]
            if not added:continue
            current[j]|=added
            affected=list(watchers.get(j,()))
            if any(j in columns for columns in self.items[candidate]['deps'].values()):affected.append(candidate)
            for target in affected:
                if target not in queued:queue.append(target);queued.add(target)
        return tuple(current)

    def prediction(self, state, rows):
        return np.stack([unbits(v,self.n)[rows] for v in state],axis=1)


def f1(tp, fp, fn):
    den=2*tp+fp+fn
    return np.divide(2*tp,den,out=np.zeros_like(np.asarray(den),dtype=float),where=den!=0)


class Metric:
    def __init__(self, rows, y, n):
        assert y.dtype==np.bool_ and len(rows)==len(y) and len(set(rows.tolist()))==len(rows)
        mask=np.zeros(n,bool);mask[rows]=True;self.scope=bits(mask)
        self.truth=[]
        for j in range(y.shape[1]):
            mask[:]=False;mask[rows]=y[:,j];self.truth.append(bits(mask))
        self.positives=y.sum(0,dtype=np.int64)
        self.origin=int(rows.min()) if len(rows) else 0

    def counts(self, state):
        tp=np.array([count((v&t)>>self.origin) for v,t in zip(state,self.truth)],np.int64)
        size=np.array([count((v&self.scope)>>self.origin) for v in state],np.int64)
        return tp,size-tp,self.positives-tp

    @staticmethod
    def quality(c):
        tp,fp,fn=c
        return dict(macro_f1=float(f1(tp,fp,fn).mean()),micro_f1=float(f1(tp.sum(),fp.sum(),fn.sum())))

    def delta(self, old, new, old_counts):
        tp,fp,fn=[v.copy() for v in old_counts]
        for j,(a,b) in enumerate(zip(old,new)):
            if a==b:continue
            assert not (a&~b),'Positive-head closure must be monotone'
            added=(b&~a)&self.scope
            if added:
                t=count((added&self.truth[j])>>self.origin);f=count(added>>self.origin)-t
                tp[j]+=t;fp[j]+=f;fn[j]-=t
        return tp,fp,fn
