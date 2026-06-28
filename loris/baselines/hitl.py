"""GROUP B — human-in-the-loop pool-based active learning baselines.

Two acquisition strategies, both simulating the human oracle by revealing the
ground-truth ``train_y`` of queried documents:

* ``besra`` — beta-scoring style **pure uncertainty**: per document, aggregate
  the per-label closeness-to-0.5 of the base model's probabilities into a single
  uncertainty score and query the most-uncertain docs.
* ``ral``   — **uncertainty × diversity**: the same per-doc uncertainty weighted
  by representativeness (mean cosine distance of a doc's TF-IDF vector to the
  already-labeled set), so a batch is both uncertain and spread out.

Protocol (pool-based AL over the TRAIN set):

1. Seed a tiny labeled set (``seed`` docs, drawn deterministically by ``seed``).
2. Fit a base ``TFIDFClassifier(svm, unigram)`` on the labeled set.
3. Score the unlabeled train pool with the acquisition function, query the top
   ``batch`` docs (reveal their ``train_y`` — the simulated human), add them to
   the labeled set, refit.
4. Repeat until the annotation ``budget`` is spent.
5. The final model predicts TEST; report hard-pred metrics +
   ``n_annotations`` = total queried (seed + queried rounds).

CPU-only and fast (linear base model). The acquisition proxies are first-pass
approximations of the BESRA / RAL papers — validate against the originals later.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from loris.baselines.common import Split, score
from loris.models.tfidf_classifier import TFIDFClassifier


# ── Base model factory ────────────────────────────────────────────────────────
def _make_base(n_labels: int) -> TFIDFClassifier:
    """TF-IDF SVM (unigram) base learner used by both AL strategies."""
    return TFIDFClassifier(
        num_labels=n_labels,
        classifier_type="svm",
        ngram_range=(1, 1),
    )


def _fit_on(idx: np.ndarray, split: Split) -> TFIDFClassifier:
    """Fit a fresh base model on the labeled subset given by ``idx``."""
    Xtr = [split.train_X[i] for i in idx]
    ytr = split.train_y[idx]
    clf = _make_base(split.n_labels)
    clf.fit(Xtr, ytr, split.val_X, split.val_y)
    return clf


def _uncertainty(proba: np.ndarray) -> np.ndarray:
    """Per-doc uncertainty: mean over labels of (1 - 2*|p - 0.5|).

    ``|p-0.5|`` is the per-label margin; ``1 - 2*|p-0.5| ∈ [0,1]`` peaks at the
    decision boundary (p=0.5). Averaging over labels yields a single
    multi-label uncertainty per document (beta-scoring style aggregate).
    """
    margin = np.abs(proba - 0.5)              # (n, L), 0 at boundary .. 0.5 confident
    per_label = 1.0 - 2.0 * margin            # (n, L), 1 at boundary .. 0 confident
    return per_label.mean(axis=1)             # (n,)


# ── Acquisition functions ─────────────────────────────────────────────────────
def _acq_besra(
    clf: TFIDFClassifier,
    pool_idx: np.ndarray,
    labeled_idx: np.ndarray,
    split: Split,
    feats: np.ndarray,
) -> np.ndarray:
    """Pure-uncertainty score over the unlabeled pool (higher = query first)."""
    pool_X = [split.train_X[i] for i in pool_idx]
    proba = clf.predict_proba(pool_X)
    return _uncertainty(proba)


def _acq_ral(
    clf: TFIDFClassifier,
    pool_idx: np.ndarray,
    labeled_idx: np.ndarray,
    split: Split,
    feats: np.ndarray,
) -> np.ndarray:
    """Uncertainty × representativeness (mean cosine distance to labeled set)."""
    pool_X = [split.train_X[i] for i in pool_idx]
    proba = clf.predict_proba(pool_X)
    unc = _uncertainty(proba)                 # (n_pool,)

    # Representativeness: mean cosine distance from each pool doc to the labeled
    # set. Vectors are L2-normalized so cosine sim = dot product.
    pool_f = feats[pool_idx]                  # (n_pool, d)
    lab_f = feats[labeled_idx]                # (n_lab, d)
    if lab_f.shape[0] == 0:
        return unc
    sim = pool_f @ lab_f.T                    # (n_pool, n_lab) cosine similarity
    mean_sim = np.asarray(sim.mean(axis=1)).ravel()
    diversity = 1.0 - mean_sim                # higher = farther from labeled
    diversity = np.clip(diversity, 0.0, None)
    return unc * diversity


_ACQ: Dict[str, Callable] = {"besra": _acq_besra, "ral": _acq_ral}


# ── Core AL loop ──────────────────────────────────────────────────────────────
def _run_al(
    split: Split,
    acq_name: str,
    *,
    seed: int = 0,
    budget: int = 400,
    batch: int = 50,
    seed_size: int = 50,
) -> dict:
    if not isinstance(budget, int):
        budget = int(budget)
    n_train = len(split.train_X)
    rng = np.random.RandomState(seed)

    budget = min(budget, n_train)
    seed_size = min(seed_size, budget, n_train)
    acq_fn = _ACQ[acq_name]

    # Precompute normalized TF-IDF features for the diversity term (ral). Cheap
    # and deterministic; computed once over the whole train pool.
    feats = None
    if acq_name == "ral":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 1), min_df=2)
        feats = normalize(vec.fit_transform(split.train_X))  # sparse, L2-normed

    # Seed labeled set
    perm = rng.permutation(n_train)
    labeled = list(perm[:seed_size])
    labeled_mask = np.zeros(n_train, dtype=bool)
    labeled_mask[labeled] = True

    clf = _fit_on(np.asarray(labeled), split)
    rounds = 0

    while len(labeled) < budget:
        pool_idx = np.where(~labeled_mask)[0]
        if pool_idx.size == 0:
            break
        b = min(batch, budget - len(labeled), pool_idx.size)

        scores = acq_fn(clf, pool_idx, np.asarray(labeled), split, feats)
        # Top-b by acquisition score; deterministic tie-break by index order.
        order = np.lexsort((pool_idx, -scores))[:b]
        chosen = pool_idx[order]

        labeled.extend(chosen.tolist())
        labeled_mask[chosen] = True
        clf = _fit_on(np.asarray(labeled), split)
        rounds += 1

    y_pred = clf.predict(split.test_X)
    metrics = score(split.test_y, y_pred)
    metrics["n_annotations"] = int(len(labeled))
    metrics["extra"] = {"budget": int(budget), "rounds": int(rounds),
                        "seed_size": int(seed_size), "batch": int(batch)}
    return metrics


# ── Public baseline fns ───────────────────────────────────────────────────────
def besra(split: Split, seed: int = 0, *, budget: int = 400,
          batch: int = 50, seed_size: int = 50, **kwargs) -> dict:
    """BESRA: pool-based AL with beta-scoring pure-uncertainty acquisition."""
    return _run_al(split, "besra", seed=seed, budget=budget,
                   batch=batch, seed_size=seed_size)


def ral(split: Split, seed: int = 0, *, budget: int = 400,
        batch: int = 50, seed_size: int = 50, **kwargs) -> dict:
    """RAL: pool-based AL with uncertainty × diversity acquisition."""
    return _run_al(split, "ral", seed=seed, budget=budget,
                   batch=batch, seed_size=seed_size)


BASELINES: Dict[str, Callable] = {"besra": besra, "ral": ral}
