"""
Maximize the LORIS rule DELTA over baseline on BGC (real text + genre hierarchy).
Goal = MAX (model+rules − baseline), NOT max absolute F1. Complementary config:
single-representation base + orthogonal-representation rules.

Config A (recommended): base = embedding (MiniLM+Ridge, semantic, lexically blind),
                         rules = LEXICAL (contrastive-FN add + FP-removal trees).
Config B (contrast):    base = tfidf,
                         rules = embedding multi-hop PROPAGATION (semantic signal tfidf lacks).
Strategy 3 (asymmetric per-label gate) applies to all rule admission.

Honest: rules mined on OOF residual (cross-fit), selected by OOF precision, applied to TEST.
Deterministic seed=42. Checkpointed (embeddings/tfidf/oof cached; --resume reuses).

Usage:
  python bgc_rill/maximize_delta.py --subset 8000      # smoke
  python bgc_rill/maximize_delta.py                    # full
"""
import os, sys, json, argparse, numpy as np, scipy.sparse as sp
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score
import pandas as pd
np.random.seed(42)

ap=argparse.ArgumentParser()
ap.add_argument('--subset',type=int,default=0,help='0=full; else cap train pool / test')
ap.add_argument('--outdir',default='/root/autodl-tmp/Loris/experiments/bgc_maxdelta')
ap.add_argument('--folds',type=int,default=3)
args=ap.parse_args()
OUT=args.outdir; CK=f'{OUT}/cache'; os.makedirs(CK,exist_ok=True)
def hr(t): print("\n"+"="*80+f"\n{t}\n"+"="*80,flush=True)
def log(m): print(m,flush=True)

# ───────────────────────── data ─────────────────────────
hr("load BGC")
tr=pd.read_csv('data/bgc/processed/train.csv'); te=pd.read_csv('data/bgc/processed/test.csv')
labs=[c for c in tr.columns if c!='text']; L=len(labs)
rng=np.random.RandomState(42)
if args.subset:
    tri=rng.permutation(len(tr))[:args.subset]; tei=rng.permutation(len(te))[:max(args.subset//3,2000)]
    tr=tr.iloc[tri].reset_index(drop=True); te=te.iloc[tei].reset_index(drop=True)
TRtxt=tr['text'].fillna('').values; TEtxt=te['text'].fillna('').values
Ypool=tr[labs].values.astype(np.float32); Yte=te[labs].values.astype(np.float32)
ntr,nte=len(TRtxt),len(TEtxt)
tag=f"sub{args.subset}" if args.subset else "full"
log(f"  pool={ntr} test={nte} labels={L} avg/doc={Ypool.sum(1).mean():.2f}")
tn=[i for i in range(L) if Ypool[:,i].sum()>=10]   # trainable labels
log(f"  trainable labels (>=10 pos): {len(tn)}")

# ───────────────────────── representations (cached) ─────────────────────────
def cache_npy(path,fn):
    if os.path.exists(path): log(f"  [cache] {path}"); return np.load(path)
    a=fn(); np.save(path,a); return a
def cache_npz(path,fn):
    if os.path.exists(path): log(f"  [cache] {path}"); return sp.load_npz(path)
    a=fn(); sp.save_npz(path,a); return a

hr("representations")
def build_tfidf():
    vec=TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",ngram_range=(1,2),min_df=5,
                        max_features=120000,sublinear_tf=True,stop_words='english')
    X=vec.fit_transform(np.concatenate([TRtxt,TEtxt]))
    return normalize(X)
Xtf=cache_npz(f'{CK}/tfidf_{tag}.npz',build_tfidf)
Xtf_pool,Xtf_te=Xtf[:ntr],Xtf[ntr:]
def build_emb():
    from sentence_transformers import SentenceTransformer
    m=SentenceTransformer('all-MiniLM-L6-v2',device='cuda')
    e=m.encode(list(np.concatenate([TRtxt,TEtxt])),batch_size=256,show_progress_bar=False,
               normalize_embeddings=True,convert_to_numpy=True).astype(np.float32)
    return e
emb=cache_npy(f'{CK}/emb_{tag}.npy',build_emb)
emb_pool,emb_te=emb[:ntr],emb[ntr:]
log(f"  tfidf {Xtf.shape} | emb {emb.shape}")

# ───────────────────────── base via OOF Ridge multi-output ─────────────────────────
def oof_and_test(fp,ft):
    oof=np.zeros((ntr,L),np.float32); kf=KFold(args.folds,shuffle=True,random_state=42)
    for a,b in kf.split(np.arange(ntr)):
        m=Ridge(alpha=1.0); m.fit(fp[a],Ypool[a]); oof[b]=m.predict(fp[b])
    m=Ridge(alpha=1.0); m.fit(fp,Ypool); ts=m.predict(ft)
    return oof,ts
def tune_thr(oof):
    thr=np.full(L,1e9)
    for i in tn:
        s=oof[:,i]; y=Ypool[:,i]; cs=np.quantile(s,np.linspace(0,1,80)); bf,bt=-1,0
        for t in cs:
            f=f1_score(y,s>=t,zero_division=0)
            if f>bf: bf,bt=f,t
        thr[i]=bt
    return thr
def mm(P): return (f1_score(Yte[:,tn],P[:,tn],average='micro',zero_division=0),
                   f1_score(Yte[:,tn],P[:,tn],average='macro',zero_division=0))

def get_base(name,fp,ft):
    op=f'{CK}/oof_{name}_{tag}.npy'; tp=f'{CK}/ts_{name}_{tag}.npy'
    if os.path.exists(op) and os.path.exists(tp):
        log(f"  [cache] base {name}"); return np.load(op),np.load(tp)
    log(f"  fitting base {name} (OOF {args.folds}-fold + full)…")
    oof,ts=oof_and_test(fp,ft); np.save(op,oof); np.save(tp,ts); return oof,ts

# lexical phrase presence (uni+bigram binary) for rules
def build_counts():
    cv=CountVectorizer(token_pattern=r"(?u)\b\w+\b",ngram_range=(1,2),min_df=10,
                       max_features=80000,binary=True,stop_words='english')
    B=cv.fit_transform(np.concatenate([TRtxt,TEtxt])).tocsc(); return B
Bfile=f'{CK}/counts_{tag}.npz'
if os.path.exists(Bfile): B=sp.load_npz(Bfile).tocsc()
else: B=build_counts(); sp.save_npz(Bfile,B)
Bpool,Bte=B[:ntr],B[ntr:]

# ───────────────────────── rule strategies (honest OOF mining) ─────────────────────────
def asym_gate(i, base_prec, floor=0.5, margin=0.05):
    return max(floor, base_prec[i]+margin)

def strat1_contrastive_fn(Poof,Pt,base_prec,gate_mode='asym'):
    """add-rules: match/cooccur phrase -> add L, mined on OOF pred=0 residual of weak labels."""
    P=Pt.copy(); nr=0
    rec=np.array([ (Poof[:,i].astype(int)&Ypool[:,i].astype(int)).sum()/max(Ypool[:,i].sum(),1) for i in range(L)])
    weak=[i for i in tn if rec[i]<0.6]
    for i in weak:
        sel=(Poof[:,i]==0); cols=Bpool[sel]; yp=Ypool[sel,i]
        pres=np.asarray(cols.sum(0)).ravel(); pos=np.asarray(cols[yp==1].sum(0)).ravel()
        with np.errstate(divide='ignore',invalid='ignore'):
            pr=np.where(pres>=10,pos/np.maximum(pres,1),0)
        g=asym_gate(i,base_prec) if gate_mode=='asym' else 0.6
        feats=np.where((pres>=10)&(pr>=g))[0]
        if len(feats)==0: continue
        nr+=len(feats)
        fire=(np.asarray(Bte[:,feats].sum(1)).ravel()>0)&(P[:,i]==0); P[fire,i]=1
    return P,nr

def strat2_fp_tree(Poof,Pt):
    """remove-rules: depth-3 tree on other-label one-hot to spot FP, mined on OOF pred=1."""
    P=Pt.copy(); nr=0
    fp_by=((Ypool==0)&(Poof==1)).sum(0)
    rngm=np.random.RandomState(0); m80=rngm.rand(ntr)<0.8
    for i in [j for j in np.argsort(-fp_by) if fp_by[j]>=20][:40]:
        seln=(Poof[:,i]==1)&m80; selv=(Poof[:,i]==1)&~m80
        if seln.sum()<40 or selv.sum()<10: continue
        yt=Ypool[seln,i]
        if yt.sum()<5 or (yt==0).sum()<10: continue
        Xtr=np.delete(Poof[seln],i,1); Xv=np.delete(Poof[selv],i,1)
        dt=DecisionTreeClassifier(max_depth=3,class_weight='balanced',random_state=42,min_samples_leaf=15)
        dt.fit(Xtr,yt); cls=list(dt.classes_)
        if 0 not in cls: continue
        c0=cls.index(0); p0v=dt.predict_proba(Xv)[:,c0]; rmv=p0v>=0.85
        if rmv.sum()<3: continue
        if (1-Ypool[selv,i][rmv].mean())<0.85: continue   # val removal precision
        selt=np.where(Pt[:,i]==1)[0]
        if len(selt)==0: continue
        p0t=dt.predict_proba(np.delete(Pt[selt],i,1))[:,c0]; idx=selt[p0t>=0.85]
        P[idx,i]=0; nr+=1
    return P,nr

# embedding multi-hop propagation (for config B); seeds = base CONFIDENT preds (honest, no GT)
def emb_propagation(base_oof,base_ts,thr,Pt,k=10,hops=20,alpha=0.85,conf_q=0.9):
    """Build emb kNN graph on full corpus; seed = base's high-confidence test+pool preds; propagate."""
    import torch
    N=ntr+nte; E=torch.tensor(emb,device='cuda'); rows=[];cols=[];vals=[];bs=2000
    for s0 in range(0,N,bs):
        sim=E[s0:s0+bs]@E.T; sim[torch.arange(sim.shape[0]),torch.arange(s0,s0+sim.shape[0])]=-1
        v,ix=sim.topk(k,1)
        for r in range(sim.shape[0]):
            gi=s0+r; rows+=[gi]*k; cols+=ix[r].tolist(); vals+=v[r].clamp(min=0).tolist()
    W=sp.csr_matrix((vals,(rows,cols)),shape=(N,N)); W=W.maximum(W.T)
    d=np.asarray(W.sum(1)).ravel(); d[d==0]=1; Wn=sp.diags(1.0/d)@W
    # seed labels = confident base predictions (pool: high score; test excluded from seeding to stay honest? use pool only)
    base_pool_pred=(base_oof>=thr).astype(np.float32)
    F0=np.zeros((N,L),np.float32); F0[:ntr]=base_pool_pred          # seed from pool base preds only
    F=F0.copy()
    for _ in range(hops):
        F=alpha*(Wn@F)+(1-alpha)*F0; F[:ntr]=base_pool_pred
    prop_pool=F[:ntr]; prop_te=F[ntr:]
    # per-label add-threshold tuned on POOL OOF for precision>=0.6 among NEW adds (honest, no test peek)
    P=Pt.copy()
    base_pool_pred_b=base_pool_pred.astype(bool)
    for i in tn:
        sc=prop_pool[:,i]; cand=(~base_pool_pred_b[:,i])&(sc>0)
        if cand.sum()<10: continue
        best_t=None
        for t in np.quantile(sc[cand],np.linspace(0.2,0.99,30)):
            fire=cand&(sc>=t); n=int(fire.sum())
            if n<10: continue
            prec=Ypool[fire,i].mean()
            if prec>=0.6: best_t=t; break   # smallest t reaching prec>=0.6 (max recall at that prec)
        if best_t is None: continue
        ft=(prop_te[:,i]>=best_t)&(P[:,i]==0); P[ft,i]=1
    return P

# ───────────────────────── run configs + ablation ─────────────────────────
results={}
def base_prec_vec(Poof):
    return np.array([precision_score(Ypool[:,i],Poof[:,i],zero_division=0) if i in tn else 0 for i in range(L)])

hr("CONFIG A: base=EMBEDDING, rules=LEXICAL (contrastive-FN + FP-tree, asym gate)")
oofA,tsA=get_base('emb',emb_pool,emb_te); thrA=tune_thr(oofA)
PoofA=(oofA>=thrA).astype(np.int8); PtA=(tsA>=thrA).astype(np.int8)
bpA=base_prec_vec(PoofA); bA=mm(PtA); log(f"  baseline(emb): micro={bA[0]:.4f} macro={bA[1]:.4f}")
P=PtA.copy()
P1,n1=strat1_contrastive_fn(PoofA,P,bpA); m1=mm(P1); log(f"  +S1 contrastive-FN ({n1} rules): micro={m1[0]:.4f}(Δ{m1[0]-bA[0]:+.4f}) macro={m1[1]:.4f}(Δ{m1[1]-bA[1]:+.4f})")
P2,n2=strat2_fp_tree(PoofA,P1); m2=mm(P2); log(f"  +S2 FP-tree ({n2} rules): micro={m2[0]:.4f}(Δ{m2[0]-bA[0]:+.4f}) macro={m2[1]:.4f}(Δ{m2[1]-bA[1]:+.4f})")
results['A_emb_base']={'baseline':bA,'+S1':m1,'+S1+S2':m2,'n_rules':[n1,n2]}

hr("CONFIG B: base=TFIDF, rules=EMBEDDING multi-hop PROPAGATION")
oofB,tsB=get_base('tfidf',Xtf_pool,Xtf_te); thrB=tune_thr(oofB)
PoofB=(oofB>=thrB).astype(np.int8); PtB=(tsB>=thrB).astype(np.int8)
bB=mm(PtB); log(f"  baseline(tfidf): micro={bB[0]:.4f} macro={bB[1]:.4f}")
PB=emb_propagation(oofB,tsB,thrB,PtB); mB=mm(PB); log(f"  +emb-propagation: micro={mB[0]:.4f}(Δ{mB[0]-bB[0]:+.4f}) macro={mB[1]:.4f}(Δ{mB[1]-bB[1]:+.4f})")
# also lexical rules on tfidf base (control: should be ~0, redundant)
bpB=base_prec_vec(PoofB); PBl,nBl=strat1_contrastive_fn(PoofB,PtB,bpB); mBl=mm(PBl)
log(f"  +S1 contrastive-FN on tfidf base ({nBl} rules, control): micro={mBl[0]:.4f}(Δ{mBl[0]-bB[0]:+.4f})")
results['B_tfidf_base']={'baseline':bB,'+emb_prop':mB,'+S1_control':mBl}

hr("SUMMARY — Δ over baseline (the target metric)")
for cfg,r in results.items():
    log(f"  [{cfg}] baseline micro={r['baseline'][0]:.4f}")
    for k,v in r.items():
        if k in ('baseline','n_rules'): continue
        log(f"      {k}: micro={v[0]:.4f} (Δ{v[0]-r['baseline'][0]:+.4f}) macro={v[1]:.4f} (Δ{v[1]-r['baseline'][1]:+.4f})")
json.dump({k:{kk:(list(vv) if isinstance(vv,tuple) else vv) for kk,vv in r.items()} for k,r in results.items()},
          open(f'{OUT}/delta_results_{tag}.json','w'),indent=1,default=str)
log(f"\nsaved {OUT}/delta_results_{tag}.json")
print("DONE")
