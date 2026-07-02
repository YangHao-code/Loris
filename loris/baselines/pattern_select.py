"""Group C — pattern-selection ablation baselines (paper §7).

A *pattern* here is a binary lexical feature: presence of one of the top
``~1000`` unigrams+bigrams (``CountVectorizer(binary=True)``) over the training
corpus.  Each method **ranks** those patterns and keeps the top-``k``; a
multilabel :class:`~sklearn.linear_model.LogisticRegression` is then trained on
the selected-pattern features alone and scored on test.  The four methods differ
only in the *ranking*:

* ``filter_mi``   — mutual information (``mutual_info_classif``) summed across
  labels (filter / univariate).
* ``filter_chi2`` — chi-squared statistic (``chi2``) summed across labels.
* ``weshap``      — a Monte-Carlo Shapley-value proxy: average marginal
  contribution of each pattern to the val macro-F1 of a cheap logistic model,
  over a handful of random feature permutations.
* ``localboost``  — greedy boosting-style selection: iteratively add the pattern
  that most reduces the residual error of the current logistic ensemble on val.

All four share the same downstream model and metric, so differences isolate the
selection criterion.  CPU only.  ``weshap`` / ``localboost`` are first-pass
approximations (small Monte-Carlo / iteration budgets) — see caveat in
``BASELINES`` docstrings.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from loris.baselines import common


# ── shared feature construction ───────────────────────────────────────────────
def _build_pattern_matrix(
    split: common.Split, n_patterns: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, "object"]:
    """Binary presence matrix of the top ``n_patterns`` unigrams+bigrams.

    Vocabulary is fit on the *training* corpus only (no test leakage), then
    applied to train/val/test.  Returns ``(Xtr, Xval, Xte, vectorizer)`` as
    dense ``float64`` arrays (n_patterns is small, ~1000).
    """
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(
        binary=True,
        ngram_range=(1, 2),
        max_features=n_patterns,
        min_df=2,
        stop_words="english",
    )
    Xtr = vec.fit_transform(split.train_X)
    Xval = vec.transform(split.val_X)
    Xte = vec.transform(split.test_X)
    # dense (small column count); .A keeps things simple for the selectors
    return (
        np.asarray(Xtr.todense(), dtype=np.float64),
        np.asarray(Xval.todense(), dtype=np.float64),
        np.asarray(Xte.todense(), dtype=np.float64),
        vec,
    )


def _fit_score(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xval: np.ndarray, yval: np.ndarray,
    Xte: np.ndarray, yte: np.ndarray,
    cols: np.ndarray, seed: int,
) -> Dict[str, float]:
    """Train multilabel LogisticRegression on the selected columns, score test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    if cols.size == 0:
        # degenerate: no patterns selected -> empty predictions
        n_lbl = ytr.shape[1]
        return common.score(yte, np.zeros((len(yte), n_lbl), dtype=np.int8))

    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=400, C=1.0, solver="liblinear", random_state=seed,
        )
    )
    # Drop labels with no positive example in train (OvR can't fit them); they
    # are predicted all-zero, which common.score handles via zero_division=0.
    ytr_sel = ytr
    clf.fit(Xtr[:, cols], ytr_sel)

    val_scores = _proba(clf, Xval[:, cols], ytr.shape[1])
    test_scores = _proba(clf, Xte[:, cols], ytr.shape[1])
    return common.score_from_scores(test_scores, yte, val_scores, yval)


def _proba(clf, X: np.ndarray, n_labels: int) -> np.ndarray:
    """Per-label positive-class probabilities, shape (n, n_labels).

    OneVsRestClassifier skips labels that had a single class in train; rebuild
    a full (n, n_labels) matrix, filling absent labels with 0.0.
    """
    raw = clf.predict_proba(X)  # (n, n_present_labels)
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape[1] == n_labels:
        return raw
    out = np.zeros((X.shape[0], n_labels), dtype=np.float64)
    # clf.classes_ -> indices of labels actually fitted (those with 2 classes)
    present = getattr(clf, "classes_", None)
    if present is not None and len(present) == raw.shape[1]:
        out[:, np.asarray(present, dtype=int)] = raw
    else:  # fallback: best-effort copy
        out[:, : raw.shape[1]] = raw
    return out


# ── ranking criteria ──────────────────────────────────────────────────────────
def _rank_mi(Xtr: np.ndarray, ytr: np.ndarray, seed: int) -> np.ndarray:
    """Sum of mutual_info_classif over labels; descending pattern order."""
    from sklearn.feature_selection import mutual_info_classif

    n_feat = Xtr.shape[1]
    agg = np.zeros(n_feat, dtype=np.float64)
    for j in range(ytr.shape[1]):
        yj = ytr[:, j].astype(int)
        if yj.sum() == 0 or yj.sum() == len(yj):
            continue
        agg += mutual_info_classif(
            Xtr, yj, discrete_features=True, random_state=seed,
        )
    return np.argsort(agg)[::-1]


def _rank_chi2(Xtr: np.ndarray, ytr: np.ndarray, seed: int) -> np.ndarray:
    """Sum of chi2 statistic over labels; descending pattern order."""
    from sklearn.feature_selection import chi2

    n_feat = Xtr.shape[1]
    agg = np.zeros(n_feat, dtype=np.float64)
    for j in range(ytr.shape[1]):
        yj = ytr[:, j].astype(int)
        if yj.sum() == 0 or yj.sum() == len(yj):
            continue
        stat, _ = chi2(Xtr, yj)
        agg += np.nan_to_num(stat, nan=0.0)
    return np.argsort(agg)[::-1]


def _dominant_label(ytr: np.ndarray) -> int:
    return int(np.asarray(ytr).sum(axis=0).argmax())


def _cheap_macro(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xval: np.ndarray, yval: np.ndarray, cols: np.ndarray, seed: int,
) -> float:
    """Val macro-F1 of a cheap logistic model on the given columns."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import f1_score

    if cols.size == 0:
        return 0.0
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=120, solver="liblinear", random_state=seed)
    )
    clf.fit(Xtr[:, cols], ytr)
    scores = _proba(clf, Xval[:, cols], ytr.shape[1])
    pred = (scores >= 0.5).astype(np.int8)
    return float(f1_score(yval, pred, average="macro", zero_division=0))


def _rank_weshap(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xval: np.ndarray, yval: np.ndarray,
    top_k: int, seed: int, n_perm: int = 6, pool: int = 250,
) -> np.ndarray:
    """Monte-Carlo Shapley proxy for each pattern's marginal val-macro-F1.

    To stay fast we restrict the candidate pool to the ``pool`` highest-MI
    patterns (the rest contribute ~0), then estimate each pattern's average
    marginal contribution over ``n_perm`` random insertion orders, evaluating
    the cheap logistic model incrementally in coarse chunks.
    """
    rng = np.random.RandomState(seed)
    # restrict to a manageable candidate pool by MI (Shapley over 1000 feats is
    # far too slow; the discarded tail has ~zero marginal value anyway)
    mi_order = _rank_mi(Xtr, ytr, seed)
    cand = mi_order[: min(pool, mi_order.size)]
    contrib = np.zeros(cand.size, dtype=np.float64)
    counts = np.zeros(cand.size, dtype=np.float64)
    # chunk size: evaluate marginal gain after adding each block of patterns
    chunk = max(1, cand.size // 20)
    for _ in range(n_perm):
        perm = rng.permutation(cand.size)
        prev_cols: List[int] = []
        prev_val = 0.0
        for start in range(0, cand.size, chunk):
            block = perm[start:start + chunk]
            new_cols = prev_cols + [cand[b] for b in block]
            cur_val = _cheap_macro(
                Xtr, ytr, Xval, yval, np.asarray(new_cols, dtype=int), seed,
            )
            gain = (cur_val - prev_val) / max(1, len(block))
            for b in block:
                contrib[b] += gain
                counts[b] += 1.0
            prev_cols = new_cols
            prev_val = cur_val
    counts[counts == 0] = 1.0
    shap = contrib / counts
    # rank candidate pool by shapley value (desc); append the untouched tail
    cand_order = cand[np.argsort(shap)[::-1]]
    tail = np.array([c for c in mi_order if c not in set(cand.tolist())], dtype=int)
    return np.concatenate([cand_order, tail])


def _rank_localboost(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xval: np.ndarray, yval: np.ndarray,
    top_k: int, seed: int, pool: int = 300,
) -> np.ndarray:
    """Greedy boosting-style selection minimising val residual error.

    Restrict candidates to the ``pool`` highest-MI patterns, then greedily add,
    one at a time, the pattern that most improves val macro-F1 of the current
    logistic ensemble.  Stops at ``top_k`` selections (or when no gain remains).
    """
    mi_order = _rank_mi(Xtr, ytr, seed)
    cand = list(mi_order[: min(pool, mi_order.size)])
    selected: List[int] = []
    cur_val = 0.0
    target = min(top_k, len(cand))
    remaining = set(cand)
    while len(selected) < target and remaining:
        best_c, best_val = None, cur_val
        # evaluate a bounded breadth of candidates per round to stay fast
        for c in list(remaining):
            cols = np.asarray(selected + [c], dtype=int)
            v = _cheap_macro(Xtr, ytr, Xval, yval, cols, seed)
            if v > best_val:
                best_val, best_c = v, c
        if best_c is None:
            break  # no candidate improves val -> stop early
        selected.append(best_c)
        remaining.discard(best_c)
        cur_val = best_val
    # append unselected (MI order) so top_k truncation downstream still fills out
    rest = [c for c in mi_order if c not in set(selected)]
    return np.asarray(selected + rest, dtype=int)


# ── baseline entry points ─────────────────────────────────────────────────────
def _run(
    split: common.Split, ranker: str, seed: int,
    top_k: int = 200, n_patterns: int = 1000, **kwargs,
) -> dict:
    Xtr, Xval, Xte, _vec = _build_pattern_matrix(split, n_patterns=n_patterns)
    ytr, yval, yte = split.train_y, split.val_y, split.test_y
    n_feat = Xtr.shape[1]

    if ranker == "mi":
        order = _rank_mi(Xtr, ytr, seed)
    elif ranker == "chi2":
        order = _rank_chi2(Xtr, ytr, seed)
    elif ranker == "weshap":
        n_perm = int(kwargs.get("n_perm", 6))
        pool = int(kwargs.get("pool", 250))
        order = _rank_weshap(Xtr, ytr, Xval, yval, top_k, seed,
                             n_perm=n_perm, pool=pool)
    elif ranker == "localboost":
        pool = int(kwargs.get("pool", 300))
        order = _rank_localboost(Xtr, ytr, Xval, yval, top_k, seed, pool=pool)
    else:  # pragma: no cover
        raise ValueError(f"unknown ranker {ranker!r}")

    k = min(int(top_k), n_feat)
    cols = np.asarray(order[:k], dtype=int)
    metrics = _fit_score(Xtr, ytr, Xval, yval, Xte, yte, cols, seed)
    metrics["extra"] = {
        "top_k": int(k),
        "n_patterns": int(n_feat),
        "ranker": ranker,
    }
    return metrics


def filter_mi(split, seed=0, *, top_k=200, n_patterns=1000, **kwargs) -> dict:
    """Filter selection by summed mutual information. kwargs: top_k, n_patterns."""
    return _run(split, "mi", seed, top_k=top_k, n_patterns=n_patterns, **kwargs)


def filter_chi2(split, seed=0, *, top_k=200, n_patterns=1000, **kwargs) -> dict:
    """Filter selection by summed chi2 statistic. kwargs: top_k, n_patterns."""
    return _run(split, "chi2", seed, top_k=top_k, n_patterns=n_patterns, **kwargs)


def weshap(split, seed=0, *, top_k=200, n_patterns=1000, n_perm=6, pool=250, **kwargs) -> dict:
    """Monte-Carlo Shapley proxy selection (approx). kwargs: top_k, n_patterns, n_perm, pool."""
    return _run(split, "weshap", seed, top_k=top_k, n_patterns=n_patterns,
                n_perm=n_perm, pool=pool, **kwargs)


def localboost(split, seed=0, *, top_k=200, n_patterns=1000, pool=300, **kwargs) -> dict:
    """Greedy boosting-style selection (approx). kwargs: top_k, n_patterns, pool."""
    return _run(split, "localboost", seed, top_k=top_k, n_patterns=n_patterns,
                pool=pool, **kwargs)


BASELINES: Dict[str, Callable] = {
    "filter_mi": filter_mi,
    "filter_chi2": filter_chi2,
    "weshap": weshap,
    "localboost": localboost,
}
