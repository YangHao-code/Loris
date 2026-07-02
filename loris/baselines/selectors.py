"""Group A baselines — model selection (paper §7).

These swap LORIS's dynamic router for a *standalone* model-selection rule, then
score the downstream ENSEMBLE of the K selected models. Each selector picks K
model indices from a shared, fit-once pool (``init_models`` full pool); the
ensemble prediction is a per-label majority vote over those K models (a label is
on iff ``>= ceil(K/2)`` of the K selected models predict it). The result is
scored on test with :func:`loris.baselines.common.score` (hard multihot preds).

Selectors (``BASELINES`` keys):

* ``random_ms``  — ``RandomState(seed).choice(n_models, K, replace=False)``.
* ``indiv_ms``   — top-K models by individual val macro-F1.
* ``hybrid_llm`` — difficulty-routing proxy of Ding et al. Hybrid-LLM: route
  each VAL doc to the model that classifies it best (per-doc F1), then keep the
  K models that win the most docs.
* ``caas``       — LinUCB-style contextual bandit over arms = models; context is
  a low-dim lexical feature of a doc; reward = per-doc correctness. Pull over a
  few hundred val docs, then take the top-K arms by estimated value.

All four share one fit of the pool (memoised on the split object keyed by seed),
so calling every selector in one run only trains the pool once.

CAVEAT — this is the runnable "ensemble-of-selected" PROXY for Group A. The
paper-faithful "LORIS-with-selector" path (feed the K selected models as M into
the chase / rule-discovery pipeline) is a follow-up orchestrator flag, not
implemented here.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from loris.baselines import common


# ──────────────────────────────────────────────────────────────────────────────
# Shared pool fit (once per split+seed)
# ──────────────────────────────────────────────────────────────────────────────

def _fit_pool(split, seed: int) -> dict:
    """Fit the full model pool ONCE and cache per-model val proba + test preds.

    Returns a dict with keys:
      ``names``       List[str]                       — model names (pool order)
      ``val_proba``   List[np.ndarray (n_val, L)]     — per-model val proba
      ``val_pred``    List[np.ndarray (n_val, L)]     — per-model hard val preds
      ``test_pred``   List[np.ndarray (n_test, L)]    — per-model hard test preds
      ``val_macro``   List[float]                     — per-model val macro-F1

    Memoised on ``split`` under ``_loris_selector_pool_<seed>`` so all four
    selectors reuse a single fit within one process.
    """
    cache_key = f"_loris_selector_pool_{seed}"
    cached = getattr(split, cache_key, None)
    if cached is not None:
        return cached

    from loris.pipeline.shared import init_models
    from loris.baselines.common import has_raw_text

    pool = init_models(split.n_labels)
    # On datasets without raw text (e.g. rcv1 = hashed TF-IDF tokens), a
    # pretrained-encoder (RoBERTa) over pseudo-"text" is meaningless AND, with no
    # GPU on that lane, trains on CPU for hours. Drop the encoder there — the
    # paper's rcv1 comparison is the TF-IDF/linear pool anyway (results_auto.tex
    # footnote). Text datasets keep the full pool.
    if not has_raw_text(split.dataset):
        for enc_key in [k for k in pool if "encoder" in k or "lora" in k]:
            pool.pop(enc_key, None)
    names = list(pool.keys())

    val_proba: List[np.ndarray] = []
    val_pred: List[np.ndarray] = []
    test_pred: List[np.ndarray] = []
    val_macro: List[float] = []

    n_val = len(split.val_X)
    n_test = len(split.test_X)
    L = split.n_labels

    for i, name in enumerate(names):
        clf = pool[name]
        try:
            # Deterministic per-model seeding (mirrors train_models convention)
            np.random.seed(42 + seed + i)
            clf.fit(split.train_X, split.train_y, split.val_X, split.val_y)
            vp = np.asarray(clf.predict_proba(split.val_X), dtype=np.float32)
            vpred = np.asarray(clf.predict(split.val_X), dtype=np.int8)
            tpred = np.asarray(clf.predict(split.test_X), dtype=np.int8)
            vmac = common.score(split.val_y, vpred)["macro_f1"]
        except Exception:
            vp = np.zeros((n_val, L), dtype=np.float32)
            vpred = np.zeros((n_val, L), dtype=np.int8)
            tpred = np.zeros((n_test, L), dtype=np.int8)
            vmac = 0.0
        val_proba.append(vp)
        val_pred.append(vpred)
        test_pred.append(tpred)
        val_macro.append(float(vmac))

    out = {
        "names": names,
        "val_proba": val_proba,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "val_macro": val_macro,
    }
    try:
        setattr(split, cache_key, out)
    except Exception:
        pass
    return out


def _ensemble_vote(test_preds: List[np.ndarray], k_indices: List[int]) -> np.ndarray:
    """Per-label majority vote over the K selected models' hard test preds.

    A label is on iff ``>= ceil(K/2)`` of the K selected models predict it.
    """
    sel = [test_preds[i] for i in k_indices]
    stacked = np.stack(sel, axis=0).astype(np.int32)   # (K, n_test, L)
    K = len(k_indices)
    need = math.ceil(K / 2)
    votes = stacked.sum(axis=0)                         # (n_test, L)
    return (votes >= need).astype(np.int8)


def _finish(pool: dict, k_indices: List[int], split, extra: dict) -> dict:
    pred = _ensemble_vote(pool["test_pred"], k_indices)
    metrics = common.score(split.test_y, pred)
    ex = dict(extra)
    ex["selected_models"] = [pool["names"][i] for i in k_indices]
    return {**metrics, "n_annotations": 0, "extra": ex}


def _eff_k(k: int, n_models: int) -> int:
    return max(1, min(int(k), n_models))


# ──────────────────────────────────────────────────────────────────────────────
# Selectors
# ──────────────────────────────────────────────────────────────────────────────

def random_ms(split, seed: int = 0, k: int = 3, **kwargs) -> dict:
    """Random K models (uniform without replacement, seeded)."""
    pool = _fit_pool(split, seed)
    n = len(pool["names"])
    K = _eff_k(k, n)
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(n, K, replace=False).tolist())
    return _finish(pool, idx, split, {"selector": "random_ms", "k": K})


def indiv_ms(split, seed: int = 0, k: int = 3, **kwargs) -> dict:
    """Top-K models by individual val macro-F1."""
    pool = _fit_pool(split, seed)
    n = len(pool["names"])
    K = _eff_k(k, n)
    order = np.argsort(-np.asarray(pool["val_macro"]))[:K]
    idx = sorted(int(i) for i in order)
    return _finish(pool, idx, split,
                   {"selector": "indiv_ms", "k": K,
                    "val_macro": [round(m, 4) for m in pool["val_macro"]]})


def hybrid_llm(split, seed: int = 0, k: int = 3, **kwargs) -> dict:
    """Difficulty-routing proxy (Ding et al. Hybrid-LLM).

    For each VAL doc, route to the model with the highest per-doc F1 (Dice over
    the multihot label set). Keep the K models that win the most docs; break ties
    by total per-doc-F1 mass, then by val macro-F1.
    """
    pool = _fit_pool(split, seed)
    n = len(pool["names"])
    K = _eff_k(k, n)
    val_y = np.asarray(split.val_y, dtype=np.int32)
    n_val = val_y.shape[0]

    # per-doc per-model F1 (Dice): 2*TP / (pred_pos + true_pos)
    f1_dm = np.zeros((n_val, n), dtype=np.float32)
    for j in range(n):
        p = pool["val_pred"][j].astype(np.int32)
        tp = (p & val_y).sum(axis=1).astype(np.float32)
        denom = (p.sum(axis=1) + val_y.sum(axis=1)).astype(np.float32)
        f1_dm[:, j] = np.divide(2.0 * tp, denom, out=np.zeros_like(tp),
                                where=denom > 0)

    winner = f1_dm.argmax(axis=1)                       # best model per doc
    wins = np.bincount(winner, minlength=n).astype(np.float64)
    mass = f1_dm.sum(axis=0).astype(np.float64)
    vmac = np.asarray(pool["val_macro"], dtype=np.float64)
    # composite sort key: primary wins, then F1 mass, then val macro
    key = wins * 1e6 + mass * 1e0 + vmac * 1e-6
    order = np.argsort(-key)[:K]
    idx = sorted(int(i) for i in order)
    return _finish(pool, idx, split,
                   {"selector": "hybrid_llm", "k": K,
                    "wins": wins.astype(int).tolist()})


def caas(split, seed: int = 0, k: int = 3, n_pulls: int = 400,
         alpha: float = 1.0, **kwargs) -> dict:
    """LinUCB-style contextual bandit; arms = models, reward = per-doc F1.

    Context = a small lexical feature vector of the doc (TF-IDF + TruncatedSVD to
    a low dim, plus a bias term). Pull over ``n_pulls`` val docs (sampled with the
    seed); update the chosen arm's LinUCB stats with the doc's per-doc F1 reward.
    Pick top-K arms by estimated value ``theta·x_bar`` over the val contexts.
    """
    pool = _fit_pool(split, seed)
    n = len(pool["names"])
    K = _eff_k(k, n)
    val_y = np.asarray(split.val_y, dtype=np.int32)
    n_val = val_y.shape[0]

    # ── per-doc per-model reward (Dice F1), reused as bandit reward ──
    reward = np.zeros((n_val, n), dtype=np.float32)
    for j in range(n):
        p = pool["val_pred"][j].astype(np.int32)
        tp = (p & val_y).sum(axis=1).astype(np.float32)
        denom = (p.sum(axis=1) + val_y.sum(axis=1)).astype(np.float32)
        reward[:, j] = np.divide(2.0 * tp, denom, out=np.zeros_like(tp),
                                 where=denom > 0)

    # ── low-dim lexical context ──
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    try:
        vec = TfidfVectorizer(max_features=5000, sublinear_tf=True)
        Xv = vec.fit_transform(split.val_X)
        d = max(2, min(8, Xv.shape[0] - 1, Xv.shape[1] - 1))
        svd = TruncatedSVD(n_components=d, random_state=seed)
        ctx = svd.fit_transform(Xv).astype(np.float64)
    except Exception:
        ctx = np.zeros((n_val, 2), dtype=np.float64)
    # append bias term
    ctx = np.hstack([ctx, np.ones((n_val, 1))])
    d = ctx.shape[1]

    # ── LinUCB ──
    A = [np.eye(d) for _ in range(n)]
    b = [np.zeros(d) for _ in range(n)]
    rng = np.random.RandomState(seed)
    pulls = min(int(n_pulls), n_val) if n_val > 0 else 0
    doc_order = rng.choice(n_val, pulls, replace=False) if pulls > 0 else []

    for t in doc_order:
        x = ctx[t]
        # UCB score per arm
        ucb = np.empty(n, dtype=np.float64)
        for j in range(n):
            A_inv = np.linalg.inv(A[j])
            theta = A_inv @ b[j]
            ucb[j] = float(theta @ x + alpha * math.sqrt(x @ A_inv @ x))
        arm = int(ucb.argmax())
        r = float(reward[t, arm])
        A[arm] += np.outer(x, x)
        b[arm] += r * x

    # ── estimated value per arm = mean theta·x over all val contexts ──
    x_bar = ctx.mean(axis=0)
    value = np.empty(n, dtype=np.float64)
    for j in range(n):
        theta = np.linalg.inv(A[j]) @ b[j]
        value[j] = float(theta @ x_bar)
    # untouched arms (b all-zero) → value 0; tie-break by val macro
    vmac = np.asarray(pool["val_macro"], dtype=np.float64)
    key = value * 1e3 + vmac * 1e-3
    order = np.argsort(-key)[:K]
    idx = sorted(int(i) for i in order)
    return _finish(pool, idx, split,
                   {"selector": "caas", "k": K, "n_pulls": int(pulls),
                    "value": [round(float(v), 4) for v in value]})


BASELINES = {
    "random_ms": random_ms,
    "indiv_ms": indiv_ms,
    "hybrid_llm": hybrid_llm,
    "caas": caas,
}
