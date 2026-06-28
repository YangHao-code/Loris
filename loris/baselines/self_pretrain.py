"""GROUP B — Self-Pretraining (semi-supervised self-training) baseline.

A textbook self-training loop on top of the lightweight TF-IDF + linear-SVM
classifier:

1. Split the labeled train set into a *seed* (kept labeled) portion and an
   *unlabeled pool* via ``seed_frac`` (default 0.5), using ``RandomState(seed)``.
   (The held-out gold labels of the pool are discarded — the pool is treated as
   unlabeled and re-labeled by the model.)
2. Fit the base classifier on the current labeled set.
3. ``predict_proba`` on the still-unlabeled pool; for every (doc, label) cell
   whose probability is ``>= conf`` (default 0.9), assign a positive pseudo-label
   (multi-hot). Docs that receive at least one confident positive are moved out
   of the pool into the labeled set with their pseudo multi-hot row.
4. Retrain and repeat for ``rounds`` iterations (default 3), or until the pool
   is exhausted / no new docs cross the confidence bar.

The final model predicts the test split; we score hard multi-hot predictions
with :func:`loris.baselines.common.score`.

Pure CPU, fast. ``extra={"n_pseudo": ...}`` reports how many pool documents were
pseudo-labeled and folded into training across all rounds.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from loris.baselines import common


def _fit_base(seed_X: List[str], seed_y: np.ndarray, val_X: List[str],
              val_y: np.ndarray, n_labels: int):
    """Fresh TF-IDF SVM (unigram) classifier fitted on the labeled set."""
    from loris.models.tfidf_classifier import TFIDFClassifier

    clf = TFIDFClassifier(
        num_labels=n_labels,
        classifier_type="svm",
        ngram_range=(1, 1),
    )
    clf.fit(seed_X, seed_y, val_X, val_y)
    return clf


def self_pretrain(split, seed: int = 0, **kwargs) -> Dict:
    """Semi-supervised self-training with confidence-thresholded pseudo-labels.

    kwargs
    ------
    seed_frac : float (default 0.5)
        Fraction of train docs kept as the labeled *seed*; the rest form the
        unlabeled pool that is pseudo-labeled.
    conf : float (default 0.9)
        Per-label probability threshold for assigning a positive pseudo-label.
    rounds : int (default 3)
        Number of self-training iterations.
    """
    seed_frac = float(kwargs.get("seed_frac", 0.5))
    conf = float(kwargs.get("conf", 0.9))
    rounds = int(kwargs.get("rounds", 3))

    rng = np.random.RandomState(seed)

    train_X = list(split.train_X)
    train_y = np.asarray(split.train_y, dtype=np.int8)
    n = len(train_X)
    n_labels = split.n_labels

    # ── seed / unlabeled-pool split ────────────────────────────────────────
    perm = rng.permutation(n)
    n_seed = max(1, int(round(seed_frac * n)))
    n_seed = min(n_seed, n - 1) if n > 1 else n  # keep a non-empty pool if we can
    seed_idx = perm[:n_seed]
    pool_idx = perm[n_seed:]

    # Labeled set (grows over rounds); pool docs are treated as unlabeled.
    lab_X: List[str] = [train_X[i] for i in seed_idx]
    lab_y: np.ndarray = train_y[seed_idx].copy()

    pool_X: List[str] = [train_X[i] for i in pool_idx]

    n_pseudo = 0
    clf = None
    for _ in range(max(1, rounds)):
        clf = _fit_base(lab_X, lab_y, split.val_X, split.val_y, n_labels)

        if not pool_X:
            break

        probs = clf.predict_proba(pool_X)            # (n_pool, n_labels) in [0,1]
        pseudo = (probs >= conf).astype(np.int8)     # confident positives
        has_pos = pseudo.sum(axis=1) > 0             # docs with ≥1 confident label

        if not has_pos.any():
            break

        keep = np.where(has_pos)[0]
        new_X = [pool_X[i] for i in keep]
        new_y = pseudo[keep]

        lab_X = lab_X + new_X
        lab_y = np.vstack([lab_y, new_y])
        n_pseudo += int(len(keep))

        # Remove folded-in docs from the pool.
        drop = set(keep.tolist())
        pool_X = [x for i, x in enumerate(pool_X) if i not in drop]

    # Safety: ensure a fitted model exists.
    if clf is None:
        clf = _fit_base(lab_X, lab_y, split.val_X, split.val_y, n_labels)

    # ── final test predictions (hard multi-hot) ────────────────────────────
    with common.Timer() as t:
        test_pred = clf.predict(split.test_X)
    metrics = common.score(split.test_y, test_pred)

    metrics["n_annotations"] = int(n_seed)
    metrics["extra"] = {
        "n_pseudo": int(n_pseudo),
        "seed_frac": seed_frac,
        "conf": conf,
        "rounds": rounds,
        "n_seed": int(n_seed),
        "n_pool_init": int(len(pool_idx)),
        "predict_sec": float(t.sec),
    }
    return metrics


BASELINES = {"self_pretrain": self_pretrain}
