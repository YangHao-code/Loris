"""Snuba baseline (Varma & Re, VLDB 2019) — programmatic weak-supervision
heuristic synthesis.

This is a **self-contained first-pass** re-implementation of the core Snuba
idea, adapted to LORIS's multi-label text setting and run through the shared
``loris.baselines.common`` infrastructure so that splits/labels/metric match a
real LORIS run. It does NOT call the authors' ``reef`` package; fidelity to the
original implementation should be validated later (see CAVEATS below).

Pipeline (per the SPEC):

1. Vectorize train text with a sklearn ``TfidfVectorizer`` (``max_features``
   ~2000, unigrams). The same fitted vectorizer transforms test text.
2. For EACH label, generate candidate heuristics = single-feature threshold
   rules ``feature present -> label`` plus their negations
   (``feature absent -> label``).
3. Iteratively select heuristics by a combined score (F1 on the labeled train
   for that label) with a diversity/coverage guard: skip a heuristic whose
   firing set is >~0.5 Jaccard-overlapping an already-selected heuristic. Stop
   at ~10 heuristics per label, or earlier when the marginal coverage gain
   becomes tiny (the Snuba stopping statistic).
4. Label model = per-label weighted majority vote of the selected heuristics
   (weight = that heuristic's train F1). Threshold the per-label vote at 0.5 to
   obtain the multihot prediction.

CPU only. Works on any dataset with TF-IDF-able text (including rcv1, which
ships as hashed TF-IDF — Snuba operates purely on bag-of-words features so it is
applicable there, unlike the transformer/LLM baselines).

CAVEATS
-------
* The authors' Snuba uses a *generative* label model (data-programming /
  Snorkel-style) over the heuristics; here we use a simpler weighted majority
  vote, which is the SPEC's first-pass spec. Accuracy may differ from the
  reef implementation.
* Heuristics are single-feature threshold rules only (no shallow decision
  trees over feature pairs as in the full Snuba synthesizer). This keeps the
  candidate space tractable and CPU-cheap.
* The "marginal coverage gain" stopping statistic and the 0.5 Jaccard
  diversity guard are reasonable approximations of Snuba's statistical guards,
  not exact reproductions.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from loris.baselines import common


# ── heuristic synthesis ─────────────────────────────────────────────────────
_MAX_FEATURES = 2000
_MAX_HEUR_PER_LABEL = 10
_JACCARD_OVERLAP_CAP = 0.5
_MIN_MARGINAL_COVERAGE = 0.005   # Snuba-style stop when new coverage is tiny
_MIN_HEUR_F1 = 0.05              # ignore essentially useless candidates


def _binary_f1(fire: np.ndarray, y: np.ndarray) -> float:
    """F1 of a boolean firing mask against a boolean label vector."""
    tp = int(np.sum(fire & y))
    if tp == 0:
        return 0.0
    fp = int(np.sum(fire & ~y))
    fn = int(np.sum(~fire & y))
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom else 0.0


def _select_for_label(
    present: np.ndarray,      # bool (n_train, n_feat): feature present
    y_col: np.ndarray,        # bool (n_train,): label present
) -> List[Dict]:
    """Greedy diversity-guarded selection of single-feature heuristics.

    Returns a list of dicts: {feat, polarity (+1 present / -1 absent),
    weight (train F1), fire_train (bool mask)}.
    """
    n_feat = present.shape[1]

    # Build all candidates (present->label and absent->label) with their F1.
    candidates = []
    for j in range(n_feat):
        col = present[:, j]
        if not col.any():
            continue
        # polarity +1: feature present fires the label
        f1_pos = _binary_f1(col, y_col)
        if f1_pos >= _MIN_HEUR_F1:
            candidates.append((f1_pos, j, +1, col))
        # polarity -1: feature absent fires the label
        ncol = ~col
        f1_neg = _binary_f1(ncol, y_col)
        if f1_neg >= _MIN_HEUR_F1:
            candidates.append((f1_neg, j, -1, ncol))

    # Best first.
    candidates.sort(key=lambda t: t[0], reverse=True)

    selected: List[Dict] = []
    selected_masks: List[np.ndarray] = []
    covered = np.zeros(y_col.shape[0], dtype=bool)

    for f1, j, pol, fire in candidates:
        if len(selected) >= _MAX_HEUR_PER_LABEL:
            break
        # diversity guard: skip if heavily overlapping an existing heuristic
        skip = False
        for m in selected_masks:
            inter = int(np.sum(fire & m))
            union = int(np.sum(fire | m))
            if union and (inter / union) > _JACCARD_OVERLAP_CAP:
                skip = True
                break
        if skip:
            continue
        # marginal coverage gain stopping statistic
        new_covered = covered | fire
        marginal = (int(np.sum(new_covered)) - int(np.sum(covered))) / fire.shape[0]
        if selected and marginal < _MIN_MARGINAL_COVERAGE:
            # tiny marginal coverage — Snuba-style stop for this label
            break
        selected.append({"feat": j, "polarity": pol, "weight": float(f1),
                          "fire_train": fire})
        selected_masks.append(fire)
        covered = new_covered

    return selected


def _vote(
    present: np.ndarray,                 # bool (n_docs, n_feat)
    per_label_heuristics: List[List[Dict]],
) -> np.ndarray:
    """Per-label weighted majority vote -> multihot int8 (threshold 0.5).

    For each label, score = sum(weight) over heuristics that fire / sum(weight)
    over all that label's heuristics. Predict 1 when score >= 0.5.
    """
    n_docs = present.shape[0]
    n_labels = len(per_label_heuristics)
    out = np.zeros((n_docs, n_labels), dtype=np.int8)
    for lab, heurs in enumerate(per_label_heuristics):
        if not heurs:
            continue
        total_w = sum(h["weight"] for h in heurs)
        if total_w <= 0:
            continue
        agg = np.zeros(n_docs, dtype=np.float64)
        for h in heurs:
            col = present[:, h["feat"]]
            fire = col if h["polarity"] == +1 else ~col
            agg += h["weight"] * fire
        out[:, lab] = (agg / total_w >= 0.5).astype(np.int8)
    return out


def snuba(split: common.Split, seed: int = 0, **kwargs) -> dict:
    """Snuba programmatic weak-supervision baseline.

    kwargs: ``max_features`` (default 2000), ``max_heuristics_per_label``
    (default 10), ``min_df`` (default 2).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    global _MAX_HEUR_PER_LABEL
    max_features = int(kwargs.get("max_features", _MAX_FEATURES))
    min_df = int(kwargs.get("min_df", 2))
    # allow override of per-label cap (module-level constant read inside helper)
    if "max_heuristics_per_label" in kwargs:
        _MAX_HEUR_PER_LABEL = int(kwargs["max_heuristics_per_label"])

    rng = np.random.RandomState(seed)  # determinism handle (selection is greedy)
    _ = rng

    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 1),
                          min_df=min_df)
    Xtr = vec.fit_transform(split.train_X)
    Xte = vec.transform(split.test_X)

    # "feature present" = nonzero TF-IDF weight (single-feature threshold > 0).
    present_tr = (Xtr > 0).toarray()
    present_te = (Xte > 0).toarray()

    y_tr = np.asarray(split.train_y, dtype=bool)
    n_labels = split.n_labels

    with common.Timer() as t:
        per_label_heuristics: List[List[Dict]] = []
        for lab in range(n_labels):
            heurs = _select_for_label(present_tr, y_tr[:, lab])
            per_label_heuristics.append(heurs)

        y_pred = _vote(present_te, per_label_heuristics)

    metrics = common.score(split.test_y, y_pred)

    n_heur = sum(len(h) for h in per_label_heuristics)
    metrics["n_annotations"] = int(np.asarray(split.train_y).sum())
    metrics["extra"] = {
        "n_heuristics_total": int(n_heur),
        "avg_heuristics_per_label": float(n_heur / max(n_labels, 1)),
        "n_features": int(present_tr.shape[1]),
        "wall_sec": float(t.sec),
    }
    return metrics


BASELINES = {"snuba": snuba}
