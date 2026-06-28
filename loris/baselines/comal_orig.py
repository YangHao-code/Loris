"""CoMAL active-learning baseline (RAL slot) — adapter over the AUTHORS' code.

RAL (Wertz et al.) released no code; per the experiment plan we substitute the
newer, code-available, role-matched **CoMAL** (Contrastive Active Learning for
Multi-Label Text Classification, KDD 2024; repo ``chengzju/CoMAL``) and report it
under its own name. CoMAL is a pool-based active-learning method for multi-label
text — exactly RAL's slot — with a mechanism distinct from our BESRA baseline.

CoMAL's full pipeline trains a BERT backbone + a contrastive VAE prototype module
with ``apex.amp`` (not installed here) and selects with a VAE-prototype variant.
Per the agreed integration policy (adapter over the AUTHORS' ALGORITHM, not their
training harness) we lift CoMAL's **acquisition criterion verbatim** from
``refs/CoMAL/selection_methods.py`` and apply it over a base classifier's
per-label probabilities. The criterion is the repo's ``adaptive`` query strategy
(``query_samples_other(..., method='adaptive')``, lines 303-338), which is the
VAE-free form of the same score used by CoMAL's main ``query_samples``:

    pred_margin = 1 / (min_positive_prob - max_negative_prob)    # uncertainty
    car_diff    = | #predicted_positives - label_cardinality |   # cardinality gap
    score(b)    = pred_margin**b * car_diff**(1-b)               # b = 0.5

with label_cardinality = mean #positive-labels/doc on the labeled set (as in the
repo). Selection takes the top-scoring candidates. Human = ground truth (the AL
protocol matches our other HITL baseline); a downstream TF-IDF SVM trained on the
CoMAL-acquired labeled set predicts test, so Macro-F1 + #annotations are
comparable across HITL baselines.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from loris.baselines import common
from loris.models.tfidf_classifier import TFIDFClassifier


def _comal_scores(proba: np.ndarray, label_cardinality: float, b: float = 0.5) -> np.ndarray:
    """CoMAL 'adaptive' acquisition score per candidate (higher = query first).

    Verbatim port of refs/CoMAL/selection_methods.py lines 303-315,324:
      pred_pos = proba >= 0.5 ; pred_neg = proba < 0.5
      pred_margin = 1 / (min_{pos} proba - max_{neg} proba)
      car_diff    = | #pred_pos - label_cardinality |
      score       = pred_margin**b * car_diff**(1-b)
    """
    proba = np.asarray(proba, dtype=np.float64)
    n, L = proba.shape
    pos_mask = (proba >= 0.5)
    neg_mask = ~pos_mask
    # max negative prob per row (0 if no negatives)
    neg_vals = np.where(neg_mask, proba, 0.0)
    pred_neg_max = neg_vals.max(axis=1)
    # min positive prob per row (set negatives to 2 so they don't win the min)
    pos_vals = np.where(pos_mask, proba, 2.0)
    pred_pos_min = pos_vals.min(axis=1)
    margin = pred_pos_min - pred_neg_max
    # guard: where margin <= 0 (no positives, or overlap), use a small epsilon so
    # 1/margin is a large-but-finite uncertainty (CoMAL divides directly).
    margin = np.where(np.abs(margin) < 1e-6, 1e-6, margin)
    pred_margin = 1.0 / margin
    pred_margin = np.abs(pred_margin)
    pred_pos_cnt = pos_mask.sum(axis=1).astype(np.float64)
    car_diff = np.abs(pred_pos_cnt - label_cardinality)
    # car_diff can be 0; pow(0, positive) = 0 kills the score -> add tiny eps
    car_diff = car_diff + 1e-10
    score = np.power(pred_margin, b) * np.power(car_diff, 1.0 - b)
    return score


def _proba_matrix(clf, X, n_labels: int) -> np.ndarray:
    """Per-label P(positive) from a TFIDFClassifier, shape (n, n_labels)."""
    p = np.asarray(clf.predict_proba(X), dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    if p.shape[1] != n_labels:
        out = np.zeros((p.shape[0], n_labels), dtype=np.float64)
        out[:, : p.shape[1]] = p
        return out
    return p


def comal(split, seed: int = 0, *, budget: int = 400, batch: int = 50,
          seed_size: int = 50, b: float = 0.5, **kwargs) -> Dict:
    """CoMAL contrastive/cardinality active learning (authors' acquisition; human=GT)."""
    n_train = len(split.train_X)
    rng = np.random.RandomState(seed)
    budget = min(int(budget), n_train)
    seed_size = min(int(seed_size), budget, n_train)

    y_all = np.asarray(split.train_y, dtype=np.int64)
    L = split.n_labels

    perm = rng.permutation(n_train)
    labeled = list(perm[:seed_size])
    labeled_mask = np.zeros(n_train, dtype=bool)
    labeled_mask[labeled] = True

    def _fit(idx):
        clf = TFIDFClassifier(num_labels=L, classifier_type="svm", ngram_range=(1, 1))
        clf.fit([split.train_X[i] for i in idx], y_all[idx],
                split.val_X, split.val_y)
        return clf

    rounds = 0
    with common.Timer() as t:
        clf = _fit(np.asarray(labeled))
        while len(labeled) < budget:
            pool_idx = np.where(~labeled_mask)[0]
            if pool_idx.size == 0:
                break
            bsz = min(batch, budget - len(labeled), pool_idx.size)
            # label cardinality from the CURRENT labeled set (CoMAL uses dataset
            # cardinality; the labeled-set estimate is the available proxy).
            card = float(y_all[labeled].sum(axis=1).mean()) if labeled else 1.0
            pool_X = [split.train_X[i] for i in pool_idx]
            proba = _proba_matrix(clf, pool_X, L)
            scores = _comal_scores(proba, card, b=b)
            order = np.lexsort((pool_idx, -scores))[:bsz]
            chosen = pool_idx[order]
            labeled.extend(chosen.tolist())
            labeled_mask[chosen] = True
            clf = _fit(np.asarray(labeled))
            rounds += 1

        y_pred = clf.predict(split.test_X)

    metrics = common.score(split.test_y, y_pred)
    metrics["n_annotations"] = int(len(labeled))
    metrics["extra"] = {
        "source": "chengzju/CoMAL acquisition (adaptive, b=%.2f)" % b,
        "note": "RAL-slot substitute (KDD'24); reported under own name",
        "budget": int(budget), "rounds": int(rounds), "seed_size": int(seed_size),
        "batch": int(batch), "wall_sec": float(t.sec),
    }
    return metrics


BASELINES = {"comal": comal}
