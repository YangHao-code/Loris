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


# ───────────────────────── seeded multi-hop chase ─────────────────────────
def seeded_chase(Wn, Y, seed_idx, hops: int = 30, alpha: float = 0.85):
    """Transductive label propagation; seed rows clamped to their (human) labels.
    F = alpha*(Wn @ F) + (1-alpha)*F0 to fixpoint; returns score matrix (N, L)."""
    N, L = Y.shape
    F0 = np.zeros((N, L), np.float32); F0[seed_idx] = Y[seed_idx]
    F = F0.copy()
    for _ in range(hops):
        F = alpha * (Wn @ F) + (1 - alpha) * F0
        F[seed_idx] = Y[seed_idx]
    return F


# ───────────────────────── seeding strategies (UNSUPERVISED — no labels) ─────────────────────────
def seeds_random(pool_idx, S, rng):
    return rng.permutation(pool_idx)[:S]


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
def supervised(feat_pool, Y_pool, seed_idx, feat_test):
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=1.0); m.fit(feat_pool[seed_idx], Y_pool[seed_idx])
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
                     emb_cache=None, out=None):
    TR, Ytr, TE, Yte, labs = load_dataset(dataset, data_root, subset)
    ntr, nte = len(TR), len(TE); N = ntr + nte; L = len(labs)
    Y = np.vstack([Ytr, Yte]); test_rows = np.arange(ntr, N)
    tn = [i for i in range(L) if Ytr[:, i].sum() >= 5]
    print(f"[rill] {dataset}: pool={ntr} test={nte} labels={L} trainable={len(tn)}", flush=True)

    if graph == "embedding":
        feats = embed(np.concatenate([TR, TE]), cache=emb_cache)
    else:
        feats = tfidf(np.concatenate([TR, TE]))
    Wn, deg = build_graph(feats, k=k)
    pool_feats = feats[:ntr]
    pool_idx = np.arange(ntr)
    rng = np.random.RandomState(42)

    rows = []
    for S in budgets:
        if seeding == "unsupervised":
            sidx = seeds_unsupervised(pool_idx, S, deg, pool_feats, rng)
        else:
            sidx = seeds_random(pool_idx, S, rng)
        # supervised on same seeds
        sup = supervised(feats, Y[:ntr], sidx, feats[ntr:])
        sup_mi, sup_ma = best_f1(sup, Yte, tn)
        # propagation: 1-hop vs multi-hop
        F1 = seeded_chase(Wn, Y, sidx, hops=1)[test_rows]
        FM = seeded_chase(Wn, Y, sidx, hops=hops)[test_rows]
        p1_mi, p1_ma = best_f1(F1, Yte, tn)
        pm_mi, pm_ma = best_f1(FM, Yte, tn)
        row = dict(seeds=int(S), seeding=seeding,
                   sup_micro=sup_mi, sup_macro=sup_ma,
                   prop1_micro=p1_mi, prop1_macro=p1_ma,
                   propM_micro=pm_mi, propM_macro=pm_ma,
                   delta_micro=pm_mi - sup_mi, delta_macro=pm_ma - sup_ma)
        rows.append(row)
        print(f"  S={S:5d} [{seeding}] sup micro={sup_mi:.4f} | 1-hop={p1_mi:.4f} | "
              f"multi-hop={pm_mi:.4f} (Δvs sup {pm_mi - sup_mi:+.4f}, Δvs 1-hop {pm_mi - p1_mi:+.4f}) "
              f"| macro sup={sup_ma:.4f} multi={pm_ma:.4f}", flush=True)
    if out:
        json.dump(rows, open(out, "w"), indent=1, default=str)
        print(f"[rill] saved {out}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bgc")
    ap.add_argument("--data_root", default="/root/autodl-tmp/Loris/data")
    ap.add_argument("--graph", choices=["embedding", "tfidf"], default="embedding")
    ap.add_argument("--seeding", choices=["unsupervised", "random", "both"], default="both")
    ap.add_argument("--budgets", default="100,300,1000,3000")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--hops", type=int, default=30)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--emb_cache", default=None)
    ap.add_argument("--out", default="/root/autodl-tmp/Loris/experiments/bgc_maxdelta/rill_curve.json")
    a = ap.parse_args()
    budgets = tuple(int(x) for x in a.budgets.split(","))
    seedings = ["unsupervised", "random"] if a.seeding == "both" else [a.seeding]
    allrows = []
    for sd in seedings:
        print(f"\n===== seeding = {sd} =====", flush=True)
        allrows += efficiency_curve(dataset=a.dataset, data_root=a.data_root, graph=a.graph,
                                    seeding=sd, budgets=budgets, k=a.k, hops=a.hops,
                                    subset=a.subset, emb_cache=a.emb_cache,
                                    out=a.out.replace(".json", f"_{sd}.json"))
    print("DONE")


if __name__ == "__main__":
    main()
