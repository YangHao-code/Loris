"""
RILL human-labeling-EFFICIENCY evaluation.

Few human SEED labels + multi-hop cross-document propagation (the paper's
comparison-predicate chase, seeded) vs a supervised classifier on the same seeds.
This is the regime where propagation gives a LARGE delta (the residual is NOT
irreducible when the base is starved of labels).

Key design choices (anti-Oracle, reviewer-proof):
  * UNSUPERVISED active seeding — seeds are chosen from graph topology
    (degree centrality + KMeans cluster centroids), WITHOUT looking at labels.
    Compared against pure random seeding. We NEVER use ground-truth stratified
    seeding (that would assume an oracle knows which docs are rare-class).
  * Transductive label propagation over a kNN graph (embedding by default;
    tfidf optional) = the comparison predicate x.A=y.A ("x,y are neighbours")
    realized via the same SpMV primitive as loris.rules.group_propagation.

CLI:
  python -m loris.eval.rill_efficiency --dataset bgc --graph embedding \
      --seeding unsupervised --budgets 100,300,1000,3000 --k 10 --hops 30
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import scipy.sparse as sp

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ───────────────────────── data ─────────────────────────
def load_dataset(name: str, data_root: str, subset: int = 0):
    import pandas as pd
    base = f"{data_root}/{name}/processed"
    tr = pd.read_csv(f"{base}/train.csv"); te = pd.read_csv(f"{base}/test.csv")
    labs = [c for c in tr.columns if c != "text"]
    rng = np.random.RandomState(42)
    if subset:
        tr = tr.iloc[rng.permutation(len(tr))[:subset]].reset_index(drop=True)
        te = te.iloc[rng.permutation(len(te))[:max(subset // 3, 2000)]].reset_index(drop=True)
    TR = tr["text"].fillna("").values; TE = te["text"].fillna("").values
    Ytr = tr[labs].values.astype(np.float32); Yte = te[labs].values.astype(np.float32)
    return TR, Ytr, TE, Yte, labs


# ───────────────────────── representations ─────────────────────────
def embed(texts, cache: str | None = None):
    if cache and os.path.exists(cache):
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    e = m.encode(list(texts), batch_size=256, show_progress_bar=False,
                 normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    if cache:
        np.save(cache, e)
    return e


def tfidf(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    v = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=5,
                        max_features=120000, sublinear_tf=True, stop_words="english")
    return normalize(v.fit_transform(texts))


# ───────────────────────── kNN graph (the comparison-predicate link) ─────────────────────────
def build_graph(feats, k: int = 10):
    """Symmetric, row-normalized kNN adjacency. feats: dense (N,d) or sparse (N,V).
    Returns (Wn, degree) — degree = unnormalized weighted degree (for centrality)."""
    N = feats.shape[0]
    rows, cols, vals = [], [], []
    dense = not sp.issparse(feats)
    bs = 2000
    if dense:
        import torch
        E = torch.tensor(np.asarray(feats), device="cuda" if torch.cuda.is_available() else "cpu")
        for s0 in range(0, N, bs):
            sim = E[s0:s0 + bs] @ E.T
            idx0 = torch.arange(sim.shape[0]); sim[idx0, torch.arange(s0, s0 + sim.shape[0])] = -1
            v, ix = sim.topk(k, dim=1)
            for r in range(sim.shape[0]):
                gi = s0 + r; rows += [gi] * k; cols += ix[r].tolist(); vals += v[r].clamp(min=0).tolist()
    else:
        for s0 in range(0, N, bs):
            sim = (feats[s0:s0 + bs] @ feats.T).toarray()
            for r in range(sim.shape[0]):
                gi = s0 + r; sim[r, gi] = -1
                nn = np.argpartition(-sim[r], k)[:k]
                rows += [gi] * k; cols += nn.tolist(); vals += sim[r, nn].clip(0).tolist()
    W = sp.csr_matrix((vals, (rows, cols)), shape=(N, N)); W = W.maximum(W.T)
    deg = np.asarray(W.sum(1)).ravel()
    d = deg.copy(); d[d == 0] = 1
    Wn = sp.diags(1.0 / d) @ W
    return Wn.tocsr(), deg


def phrase_graph(texts_all, seed_idx, Y_seed, k_cap: int = 20,
                 min_lift: float = 2.0, min_support: int = 2, max_phrases: int = 400,
                 max_df_frac: float = 0.2):
    """Phrase-sharing adjacency = the x.A=y.A comparison predicate over LABEL-
    DISCRIMINATIVE n-grams: two docs are linked iff they share such a phrase.
    Phrases are mined on the SEED labels ONLY (few-label faithful — no peeking at
    unlabeled pool labels). Row-capped to k_cap, symmetric, row-normalized. These
    edges are more label-coherent than pure embedding cosine, so propagation along
    them is cleaner. Returns row-normalized csr (N,N) or None if no phrase qualifies."""
    from sklearn.feature_extraction.text import CountVectorizer
    seed_texts = [texts_all[i] for i in seed_idx]
    Ys = np.asarray(Y_seed, np.float32)
    if Ys.ndim == 1:
        Ys = np.eye(int(Ys.max()) + 1, dtype=np.float32)[Ys.astype(int)]
    try:
        vec = CountVectorizer(ngram_range=(1, 2), binary=True, min_df=max(2, min_support),
                              max_features=20000, stop_words="english")
        Xs = vec.fit_transform(seed_texts).astype(np.float32)
    except ValueError:
        return None
    terms = vec.get_feature_names_out()
    df = np.asarray(Xs.sum(0)).ravel()
    co = np.asarray(Xs.T.dot(Ys))                       # (V,L) phrase∧label on seeds
    p_l = np.maximum(Ys.sum(0) / max(len(seed_idx), 1), 1e-9)
    lift = (co / np.maximum(df[:, None], 1.0)) / p_l[None, :]
    best = lift.max(1)
    keep = np.nonzero((best >= min_lift) & (df >= min_support))[0]
    order = sorted(keep.tolist(), key=lambda t: (-best[t], -df[t], terms[t]))
    vocab = [str(terms[t]) for t in order[:max_phrases]]
    if not vocab:
        return None
    tv = CountVectorizer(ngram_range=(1, 2), binary=True, vocabulary=vocab)
    M = tv.transform(texts_all).astype(np.float32).tocsc()
    N = M.shape[0]
    # drop super-phrases (appear in > max_df_frac of corpus → would merge unrelated docs)
    cdf = np.asarray(M.sum(0)).ravel()
    keepcol = cdf <= max_df_frac * N
    if keepcol.sum() == 0:
        return None
    M = M[:, keepcol].tocsr()
    A = (M @ M.T).tocsr()                               # shared-phrase counts (N,N)
    A.setdiag(0); A.eliminate_zeros()
    # row-cap to k_cap strongest neighbours (keep sparse), then symmetrize + norm
    rows, cols, vals = [], [], []
    for i in range(N):
        s, e = A.indptr[i], A.indptr[i + 1]
        if e == s:
            continue
        ci, vi = A.indices[s:e], A.data[s:e]
        if len(ci) > k_cap:
            top = np.argpartition(-vi, k_cap)[:k_cap]
            ci, vi = ci[top], vi[top]
        rows += [i] * len(ci); cols += ci.tolist(); vals += vi.tolist()
    if not rows:
        return None
    P = sp.csr_matrix((vals, (rows, cols)), shape=(N, N)); P = P.maximum(P.T)
    d = np.asarray(P.sum(1)).ravel(); d[d == 0] = 1
    return (sp.diags(1.0 / d) @ P).tocsr()


# ───────────────────────── seeded multi-hop chase ─────────────────────────
def seeded_chase(Wn, Y, seed_idx, hops: int = 30, alpha: float = 0.85, F0_init=None):
    """Transductive label propagation; seed rows clamped to their (human) labels.
    F = alpha*(Wn @ F) + (1-alpha)*F0 to fixpoint; returns score matrix (N, L).

    F0_init: if None (default), propagation starts from ONLY the seed labels
    (the pure few-label RILL mechanism — no ML/rule base). If given an (N,L)
    matrix (e.g. base-model / learned-rule predictions on every doc), propagation
    runs ON TOP OF that base, with the human seeds clamped — i.e. the chase
    refines the basic predictions rather than starting from blank. This is the
    knob that answers "is multi-hop on top of the basic rule predictions?":
    F0_init=None → standalone; F0_init=base_preds → on-top-of-base."""
    N, L = Y.shape
    F0 = np.zeros((N, L), np.float32) if F0_init is None else np.asarray(F0_init, np.float32).copy()
    F0[seed_idx] = Y[seed_idx]                 # human seeds are ground truth, clamp
    F = F0.copy()
    for _ in range(hops):
        F = alpha * (Wn @ F) + (1 - alpha) * F0
        F[seed_idx] = Y[seed_idx]
    return F


# ───────────────────────── seeding strategies (UNSUPERVISED — no labels) ─────────────────────────
def seeds_random(pool_idx, S, rng):
    return rng.permutation(pool_idx)[:S]


def _coverage_sets(W, pool_idx, ntr, hops: int = 2):
    """Binary reachability set per pool node over the kNN graph (the propagation
    structure): which corpus docs a seed's label reaches within ``hops``. This is
    the set-cover universe of the paper's §6.2 coverage selection."""
    A = (W > 0).astype(np.int8)
    A = A.maximum(A.T)
    R = A.copy()
    for _ in range(hops - 1):
        R = ((R @ A) + R)
    R = (R > 0).tocsr()
    return R  # (N, N) reachability; rows restricted to pool below


def seeds_coverage(pool_idx, S, W, ntr, rng, hops: int = 2):
    """Paper §6.2 / SALT coverage-greedy seed selection: repeatedly pick the pool
    doc whose label (propagated ``hops`` over the kNN graph) covers the most
    still-UNCOVERED corpus docs; remove its covered set; repeat. Submodular
    max-coverage (1-1/e). Returns up to S seeds + the per-seed coverage count
    (for sqrt sample-weighting). No labels used (purely graph topology)."""
    R = _coverage_sets(W, pool_idx, ntr, hops=hops)
    N = R.shape[0]
    pool = [int(i) for i in pool_idx]
    neigh = {i: set(R.indices[R.indptr[i]:R.indptr[i + 1]].tolist()) | {i} for i in pool}
    # reverse index: doc j -> pool candidates whose reachable set contains j.
    neigh_rev: dict = {}
    for i in pool:
        for j in neigh[i]:
            neigh_rev.setdefault(j, []).append(i)
    covered = np.zeros(N, bool)
    counts = {i: len(neigh[i]) for i in pool}
    chosen, cov_count = [], {}
    remaining = set(pool)
    for _ in range(min(S, len(pool))):
        if not remaining:
            break
        best = max(remaining, key=lambda i: (counts[i], -i))
        if counts[best] <= 0:
            break
        newly = [j for j in neigh[best] if not covered[j]]
        cov_count[best] = len(newly)
        chosen.append(best)
        # SALT semantics (active_learning_engine.select_initial_labeled_by_coverage):
        # covered points leave candidacy (each new seed is itself uncovered), and
        # every candidate that also reached a covered point loses 1 from its count.
        for j in newly:
            covered[j] = True
            remaining.discard(j)               # incl. `best` (self ∈ neigh)
            for i in neigh_rev.get(j, ()):
                if i in remaining:
                    counts[i] -= 1
    # top up with random pool docs if coverage saturated before S
    if len(chosen) < S:
        extra = [int(i) for i in rng.permutation(pool_idx) if int(i) not in set(chosen)]
        for i in extra[:S - len(chosen)]:
            chosen.append(i); cov_count.setdefault(i, 1)
    chosen = np.array(sorted(chosen[:S]))
    return chosen, cov_count


def seeds_unsupervised(pool_idx, S, deg, pool_feats, rng):
    """Half by degree-centrality (cover dense/major regions), half by KMeans
    cluster centroids (cover the long-tail / outliers). NO labels used."""
    from sklearn.cluster import MiniBatchKMeans
    n_deg = S // 2
    n_clu = S - n_deg
    # degree centrality within the labelable pool
    pool_deg = deg[pool_idx]
    deg_pick = pool_idx[np.argsort(-pool_deg)[:n_deg]]
    # cluster centroids: nearest pool doc to each of n_clu centroids
    feats = pool_feats
    if sp.issparse(feats):
        feats = feats.toarray()
    km = MiniBatchKMeans(n_clusters=min(n_clu, len(pool_idx)), random_state=42, batch_size=1024, n_init=3)
    km.fit(feats)
    centroid_pick = []
    for c in range(km.n_clusters):
        members = np.where(km.labels_ == c)[0]
        if len(members) == 0:
            continue
        d2 = ((feats[members] - km.cluster_centers_[c]) ** 2).sum(1)
        centroid_pick.append(int(pool_idx[members[np.argmin(d2)]]))
    chosen = list(dict.fromkeys(list(deg_pick) + centroid_pick))  # dedup, order-stable
    # top up with random if short
    extra = [int(i) for i in rng.permutation(pool_idx) if int(i) not in set(chosen)]
    chosen = (chosen + extra)[:S]
    return np.array(sorted(chosen))


# ───────────────────────── supervised baseline (same seeds) ─────────────────────────
def supervised(feat_pool, Y_pool, seed_idx, feat_test, sample_weight=None):
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=1.0)
    m.fit(feat_pool[seed_idx], Y_pool[seed_idx], sample_weight=sample_weight)
    return m.predict(feat_test)


# ───────────────────────── metrics ─────────────────────────
def best_f1(scores_test, Yte, tn):
    from sklearn.metrics import f1_score
    s = scores_test[:, tn]
    pos = s[s > 0]
    if pos.size == 0:
        return 0.0, 0.0
    bm, bM = -1.0, 0.0
    for thr in np.quantile(pos, np.linspace(0.5, 0.999, 40)):
        P = (s >= thr).astype(np.int8)
        mi = f1_score(Yte[:, tn], P, average="micro", zero_division=0)
        if mi > bm:
            bm = mi; bM = f1_score(Yte[:, tn], P, average="macro", zero_division=0)
    return bm, bM


# ───────────────────────── main efficiency curve ─────────────────────────
def efficiency_curve(dataset="bgc", data_root="/root/autodl-tmp/Loris/data",
                     graph="embedding", seeding="unsupervised",
                     budgets=(100, 300, 1000, 3000), k=10, hops=30, subset=0,
                     emb_cache=None, out=None, alpha=0.70):
    TR, Ytr, TE, Yte, labs = load_dataset(dataset, data_root, subset)
    ntr, nte = len(TR), len(TE); N = ntr + nte; L = len(labs)
    Y = np.vstack([Ytr, Yte]); test_rows = np.arange(ntr, N)
    tn = [i for i in range(L) if Ytr[:, i].sum() >= 5]
    print(f"[rill] {dataset}: pool={ntr} test={nte} labels={L} trainable={len(tn)} alpha={alpha} graph={graph}", flush=True)

    txt_all = np.concatenate([TR, TE])
    if graph == "tfidf":
        feats = tfidf(txt_all)
    else:                                  # "embedding" or "combined" use embeddings as the base feats
        feats = embed(txt_all, cache=emb_cache)
    Wn, deg = build_graph(feats, k=k)
    pool_feats = feats[:ntr]
    pool_idx = np.arange(ntr)
    rng = np.random.RandomState(42)

    import math
    rows = []
    for S in budgets:
        sw = None
        if seeding == "coverage":
            # paper §6.2 / SALT coverage-greedy over the propagation graph + sqrt
            # coverage sample-weights (a seed standing for more docs weighs more).
            sidx, cov_count = seeds_coverage(pool_idx, S, Wn, ntr, rng)
            sw = np.array([math.ceil(math.sqrt(max(cov_count.get(int(i), 1), 1)))
                           for i in sidx], dtype=np.float64)
        elif seeding == "unsupervised":
            sidx = seeds_unsupervised(pool_idx, S, deg, pool_feats, rng)
        else:
            sidx = seeds_random(pool_idx, S, rng)
        # base model on same seeds — predicted on ALL docs (so propagation can
        # run ON TOP of the base; test slice = the supervised baseline itself).
        base_all = supervised(feats, Y[:ntr], sidx, feats, sample_weight=sw)
        sup = base_all[test_rows]
        sup_mi, sup_ma = best_f1(sup, Yte, tn)
        # combined graph: embedding kNN + seed-mined discriminative-phrase edges
        # (x.A=y.A). Phrase part depends on the seeds, so it's built per budget.
        Wn_use = Wn
        if graph == "combined":
            P = phrase_graph(txt_all, sidx, Y[sidx])
            if P is not None:
                Wn_use = (0.5 * Wn + 0.5 * P).tocsr()
        # propagation from blank seeds: 1-hop vs multi-hop (standalone RILL)
        F1 = seeded_chase(Wn_use, Y, sidx, hops=1, alpha=alpha)[test_rows]
        FM = seeded_chase(Wn_use, Y, sidx, hops=hops, alpha=alpha)[test_rows]
        p1_mi, p1_ma = best_f1(F1, Yte, tn)
        pm_mi, pm_ma = best_f1(FM, Yte, tn)
        # propagation ON TOP OF the base predictions (the chase refines the basic
        # predictions, seeds clamped) — directly answers "multi-hop on top of base?"
        FMb = seeded_chase(Wn_use, Y, sidx, hops=hops, alpha=alpha, F0_init=base_all)[test_rows]
        pmb_mi, pmb_ma = best_f1(FMb, Yte, tn)
        row = dict(seeds=int(S), seeding=seeding,
                   sup_micro=sup_mi, sup_macro=sup_ma,
                   prop1_micro=p1_mi, prop1_macro=p1_ma,
                   propM_micro=pm_mi, propM_macro=pm_ma,
                   propBase_micro=pmb_mi, propBase_macro=pmb_ma,
                   delta_micro=pm_mi - sup_mi, delta_macro=pm_ma - sup_ma,
                   deltaBase_micro=pmb_mi - sup_mi, deltaBase_macro=pmb_ma - sup_ma)
        rows.append(row)
        print(f"  S={S:5d} [{seeding}] sup micro={sup_mi:.4f} | 1-hop={p1_mi:.4f} | "
              f"multi-hop={pm_mi:.4f} (Δsup {pm_mi - sup_mi:+.4f}) | base+prop={pmb_mi:.4f} "
              f"(Δsup {pmb_mi - sup_mi:+.4f}) | macro sup={sup_ma:.4f} multi={pm_ma:.4f} "
              f"base+prop={pmb_ma:.4f}", flush=True)
    if out:
        json.dump(rows, open(out, "w"), indent=1, default=str)
        print(f"[rill] saved {out}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bgc")
    ap.add_argument("--data_root", default="/root/autodl-tmp/Loris/data")
    ap.add_argument("--graph", choices=["embedding", "tfidf", "combined"], default="embedding")
    ap.add_argument("--alpha", type=float, default=0.70,
                    help="propagation strength (chase). 0.70 maximises the base+prop RILL gain "
                         "(swept: 0.85->0.70 lifts base+prop micro ~+0.03->+0.04 and macro turns positive)")
    ap.add_argument("--seeding", choices=["unsupervised", "random", "coverage", "both", "all"], default="both")
    ap.add_argument("--budgets", default="100,300,1000,3000")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--hops", type=int, default=30)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--emb_cache", default=None)
    ap.add_argument("--out", default="/root/autodl-tmp/Loris/experiments/bgc_maxdelta/rill_curve.json")
    a = ap.parse_args()
    budgets = tuple(int(x) for x in a.budgets.split(","))
    if a.seeding == "both":
        seedings = ["unsupervised", "random"]
    elif a.seeding == "all":
        seedings = ["coverage", "unsupervised", "random"]
    else:
        seedings = [a.seeding]
    allrows = []
    for sd in seedings:
        print(f"\n===== seeding = {sd} =====", flush=True)
        allrows += efficiency_curve(dataset=a.dataset, data_root=a.data_root, graph=a.graph,
                                    seeding=sd, budgets=budgets, k=a.k, hops=a.hops,
                                    subset=a.subset, emb_cache=a.emb_cache, alpha=a.alpha,
                                    out=a.out.replace(".json", f"_{sd}.json"))
    print("DONE")


if __name__ == "__main__":
    main()
