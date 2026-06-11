"""Propagation join-key CEILING probe — does Phase 1/2 (a real label-coherent
join key) have any hope, before we invest in MIMIC / LLM attribute extraction?

Phase 0 proved: propagating human labels along the EMBEDDING cosine graph adds
exactly 0 F1 over direct labeling (the graph isn't label-coherent). Phase 1
(mtd join keys) and Phase 2 (MIMIC same-patient notes) bet that a BETTER join
key fixes this. This probe measures, with no new data and no LLM:

  PROP-GAIN(G) = macroF1(propagate B seeds along G, on non-seed docs)
               − macroF1(direct: same B seeds, NO propagation, on non-seed docs)

under three graphs:
  embed         — cosine kNN on MiniLM (current; the proved-0 baseline)
  phrase        — shared discriminative-phrase attrs mined from SEED labels only
                  (HONEST Phase-1 proxy: a real attribute join key, no leak)
  label_oracle  — kNN on ground-truth label vectors (CEILING; leaks eval labels)

Read:
  * label_oracle ≈ 0   ⇒ propagation is structurally dead REGARDLESS of join key
                         → do NOT pursue Phase 1/2 propagation; lean on B=0 rules.
  * label_oracle ≫ 0   ⇒ join-key quality IS the lever; phrase vs embed shows how
                         much a realistic honest key already recovers → Phase 1/2
                         worth doing, and MIMIC's patient key should land between.

Both arms get the same weak base (Ridge on the B seeds) and the same oracle
threshold, so the DIFFERENCE isolates neighbour-propagation. Eval on non-seed
docs only (the seeds' own fix cancels — we measure what propagation buys OTHERS).
"""
from __future__ import annotations
import argparse, json
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer

from loris.eval.rill_efficiency import (
    load_dataset, embed, build_graph, phrase_graph, seeded_chase,
    best_f1, seeds_random, seeds_coverage,
)

DATA_ROOT = "/root/autodl-tmp/Loris/data"


def label_oracle_graph(Y, k=10):
    """kNN over GT label vectors (cosine). CEILING — uses eval labels."""
    Yn = normalize(Y.astype(np.float32))
    return build_graph(Yn, k=k)[0]


def weak_base(feats, Y, seed_idx):
    m = Ridge(alpha=1.0)
    m.fit(feats[seed_idx], Y[seed_idx])
    return m.predict(feats).astype(np.float32)


# ───────────────────────── HONEST join-key candidates ─────────────────────────
# (no eval-label leak — each uses ONLY the B seed labels, like a real Phase-1 key)

def pseudo_label_graph(base, k=10):
    """kNN in PREDICTED-label space. `base` = weak Ridge(seed) preds on ALL docs
    (the same base RILL refines). The honest mirror of `label_oracle`: connect
    docs whose seed-trained predicted-label PROFILE is similar, instead of their
    raw text embedding. This is also exactly `learned-metric` kNN — Ridge.predict
    is a linear projection of the embedding into label space. If this recovers a
    chunk of the oracle ceiling, propagation is salvageable with NO new data/LLM."""
    Yhat = normalize(np.asarray(base, dtype=np.float32))
    return build_graph(Yhat, k=k)[0]


def prec_phrase_graph(texts_all, seed_idx, Y_seed, k=10, min_coh=0.60,
                      min_support=3, ngram=(1, 2), max_feats=4000, min_df=2):
    """Honest Phase-1 mtd PROXY done RIGHT — the closed-ontology hypothesis,
    cheaply. Mine n-gram phrases, but KEEP only phrases whose seed carriers
    AGREE on their label vector (mean pairwise label cosine ≥ min_coh), then
    connect docs sharing a high-coherence phrase. Tests whether `phrase`'s
    NEGATIVE gain is just missing precision-gating — i.e. whether an attribute
    join key, restricted to label-coherent values (what a closed ontology buys),
    beats embed. Uses only seed labels → no leak."""
    cv = CountVectorizer(ngram_range=ngram, max_features=max_feats,
                         binary=True, min_df=min_df)
    Xall = cv.fit_transform(texts_all).tocsc()        # (N, V) binary, col-sliceable
    Xs = Xall[seed_idx].tocsc()
    Ys = normalize(np.asarray(Y_seed, dtype=np.float32))  # unit label vecs (cosine)
    keep = []
    for v in range(Xs.shape[1]):
        car = Xs.indices[Xs.indptr[v]:Xs.indptr[v + 1]]   # seed carriers of phrase v
        if len(car) < min_support:
            continue
        Lv = Ys[car]
        Lv = Lv[(Lv != 0).any(axis=1)]                    # carriers with ≥1 label
        m = len(Lv)
        if m < min_support:
            continue
        S = Lv @ Lv.T                                     # pairwise label cosine
        coh = (S.sum() - m) / (m * (m - 1))               # mean off-diagonal
        if coh >= min_coh:
            keep.append(v)
    if not keep:
        return None
    M = Xall[:, keep].tocsr()                             # (N, |keep|) coherent phrases
    if (np.asarray(M.sum(1)).ravel() > 0).sum() < 2 * k:  # too few docs covered
        return None
    return build_graph(M, k=k)[0]                         # cosine-kNN on shared coherent phrases


def llm_attr_graph(texts_all, cache_path, k=10):
    """Lever C acceptance arm — cosine-kNN over the LLM closed-ontology membership
    matrix (the offline-precomputed join key). Connects docs sharing ontology
    values. DECISION RULE: integrate Lever C only if this `llm` PROP-GAIN is
    clearly > `embed` and a meaningful fraction of `label_oracle`; if it doesn't
    beat `prec_phrase`/`embed`, STOP (kills it before any pipeline cost)."""
    from loris.document import Document                       # noqa: PLC0415
    from loris.rules.virtual_attributes import compute_llm_attributes  # noqa: PLC0415
    docs = [Document(cnt=t) for t in texts_all]
    M, _ = compute_llm_attributes(docs, cache_path)
    if M is None or M.shape[1] == 0:
        return None
    if (np.asarray(M.sum(1)).ravel() > 0).sum() < 2 * k:      # too few docs covered
        return None
    return build_graph(M, k=k)[0]


def run(dataset, budgets, subset, hops, alpha, k, seeding, out, llm_cache=None):
    TR, Ytr, TE, Yte, labs = load_dataset(dataset, DATA_ROOT, subset)
    feats = embed(TE, cache=f"/root/autodl-tmp/Loris/experiments/probe_emb_{dataset}.npy")
    Y = Yte.astype(np.float32)
    N, L = Y.shape
    tn = [i for i in range(L) if Yte[:, i].sum() >= 3]
    print(f"[probe] {dataset}: eval_docs={N} labels={L} trainable={len(tn)} "
          f"seeding={seeding} alpha={alpha} k={k}", flush=True)
    W_emb = build_graph(feats, k=k)[0]
    W_lab = label_oracle_graph(Y, k=k)
    rng = np.random.RandomState(42)
    rows = []
    for B in budgets:
        if seeding == "coverage":
            seeds, _ = seeds_coverage(np.arange(N), B, W_emb, N, rng)
        else:
            seeds = seeds_random(np.arange(N), B, rng)
        seeds = np.asarray(sorted(set(int(s) for s in seeds)))
        nonseed = np.setdiff1d(np.arange(N), seeds)
        base = weak_base(feats, Y, seeds)
        # direct control: B seeds clamped to GT, NO propagation
        direct = base.copy(); direct[seeds] = Y[seeds]
        dmi, dma = best_f1(direct[nonseed], Y[nonseed], tn)
        rec = {"B": int(B), "n_seed": int(len(seeds)),
               "direct_micro": dmi, "direct_macro": dma}
        # honest keys first (no leak), embed = proved-0 baseline, oracle = ceiling
        graphs = {"embed": W_emb}
        graphs["pseudo_label"] = pseudo_label_graph(base, k=k)
        W_pp = prec_phrase_graph(list(TE), seeds, Y[seeds], k=k)
        if W_pp is not None:
            graphs["prec_phrase"] = W_pp
        W_ph = phrase_graph(list(TE), seeds, Y[seeds])
        if W_ph is not None:
            graphs["phrase"] = W_ph
        if llm_cache:
            W_llm = llm_attr_graph(list(TE), llm_cache, k=k)
            if W_llm is not None:
                graphs["llm"] = W_llm
        graphs["label_oracle"] = W_lab
        for gname, G in graphs.items():
            prop = seeded_chase(G, Y, seeds, hops=hops, alpha=alpha, F0_init=base)
            pmi, pma = best_f1(prop[nonseed], Y[nonseed], tn)
            rec[f"{gname}_micro"] = pmi
            rec[f"{gname}_macro"] = pma
            rec[f"{gname}_gain_micro"] = pmi - dmi
            rec[f"{gname}_gain_macro"] = pma - dma
        rows.append(rec)
        order = ["embed", "phrase", "prec_phrase", "pseudo_label", "llm", "label_oracle"]
        present = [g for g in order if f"{g}_gain_macro" in rec]
        ma = " | ".join(f"{g} {rec[g + '_gain_macro']:+.4f}" for g in present)
        mi = "  ".join(f"{g} {rec[g + '_gain_micro']:+.4f}" for g in present)
        print(f"  B={B:5d} direct ma={dma:.4f}  Δmacro:: {ma}", flush=True)
        print(f"          Δmicro:: {mi}", flush=True)
    json.dump(rows, open(out, "w"), indent=1, default=str)
    print(f"[probe] saved {out}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bgc")
    ap.add_argument("--budgets", default="50,100,300,1000")
    ap.add_argument("--subset", type=int, default=6000)
    ap.add_argument("--hops", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.70)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seeding", choices=["random", "coverage"], default="coverage")
    ap.add_argument("--llm_cache", default=None,
                    help="Lever C: path to the offline LLM closed-ontology attribute cache "
                         "(from precompute_llm_attributes.py) to add the `llm` join-key arm.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    budgets = tuple(int(x) for x in a.budgets.split(","))
    out = a.out or f"/root/autodl-tmp/Loris/experiments/probe_joinkey_{a.dataset}.json"
    run(a.dataset, budgets, a.subset, a.hops, a.alpha, a.k, a.seeding, out, llm_cache=a.llm_cache)


if __name__ == "__main__":
    main()
