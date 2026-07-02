"""Snuba baseline — adapter over the AUTHORS' original ``reef`` code.

Unlike the first-pass ``loris/baselines/snuba.py`` (weighted majority vote), this
drives the real HazyResearch ``reef`` Snuba pipeline vendored under ``refs/reef``:

    Synthesizer.generate_heuristics  (LR/DT programs over feature primitives)
      -> HeuristicGenerator.prune_heuristics  (F1 + Jaccard-diversity selection)
      -> Verifier / LabelAggregator           (generative label model, no snorkel)
      -> probabilistic train labels

Snuba is a BINARY weak-supervision method (labels in {-1,+1}, class prior ``b``).
Our task is multi-label, so we run Snuba **one-vs-rest**: for each of the L labels
we synthesize+verify an independent heuristic set, take the label model's train
marginals as that label's score, tune a single global threshold on val (LORIS
convention), and stack into a multihot prediction. Primitives = top TF-IDF
features (the "primitive matrix" Snuba's synthesizer expects).

The reef code is Python-2-origin; ``label_aggregator.py`` was 2to3-converted and
``verifier.py``'s import made package-absolute when vendored. We import it by
putting ``refs/reef`` on sys.path. The generative label model (LabelAggregator)
is the Snorkel-v0.4 NaiveBayes port shipped in reef (``has_snorkel=False``).
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Dict, List

import numpy as np

from loris.baselines import common

# Vendored authors' code (refs/reef). Resolve relative to repo root.
_REEF = os.path.join(os.path.dirname(__file__), "..", "..", "refs", "reef")
_REEF = os.path.abspath(_REEF)
if _REEF not in sys.path:
    sys.path.insert(0, _REEF)


def _primitive_matrix(split, max_features: int, min_df: int):
    """TF-IDF primitive matrices (train/val/test) as dense float arrays.

    Snuba's Synthesizer fits LR/DT programs over a dense primitive matrix and
    indexes columns by integer combos, so we hand it a dense ndarray.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 1),
                          min_df=min_df, sublinear_tf=True)
    Xtr = vec.fit_transform(split.train_X).toarray().astype(np.float64)
    Xva = vec.transform(split.val_X).toarray().astype(np.float64)
    Xte = vec.transform(split.test_X).toarray().astype(np.float64)
    return Xtr, Xva, Xte


def _snuba_one_label(train_pm, val_pm, val_ground, train_ground,
                     cardinality: int, keep: int):
    """Run reef's Snuba loop for ONE binary label.

    ``*_ground`` are in Snuba's {-1,+1} convention. Returns
    ``(train_marginals, val_marginals)`` from the generative label model — both
    produced by a SINGLE run of the synthesizer-verifier loop (the
    HeuristicGenerator computes both in ``run_verifier`` -> ``assign_marginals``).
    """
    from program_synthesis.heuristic_generator import HeuristicGenerator

    hg = HeuristicGenerator(train_pm, val_pm, val_ground, train_ground, b=0.5)
    # iterate the synthesizer-verifier loop, growing the heuristic set
    n_iter = max(1, keep)
    for i in range(n_iter):
        card = min(cardinality, i + 1)
        try:
            hg.run_synthesizer(max_cardinality=card, idx=None, keep=1, model="lr")
            hg.run_verifier()
        except Exception:
            # synthesis can fail when a label has too few positives in val;
            # fall back to whatever marginals we have (or all-negative).
            break
    if getattr(hg, "vf", None) is None or not hasattr(hg.vf, "train_marginals"):
        z_tr = np.zeros(train_pm.shape[0], dtype=np.float64)
        z_va = np.zeros(val_pm.shape[0], dtype=np.float64)
        return z_tr, z_va
    return (np.asarray(hg.vf.train_marginals, dtype=np.float64),
            np.asarray(hg.vf.val_marginals, dtype=np.float64))


def snuba(split: common.Split, seed: int = 0, **kwargs) -> dict:
    """Snuba (Varma & Re) via the authors' reef code, one-vs-rest multi-label.

    kwargs: max_features (default 150 — Snuba's "primitives" are a modest feature
            set; the reef synthesizer fits one program per primitive and
            re-applies all candidates to val each ``keep`` round, so this is the
            main cost knob), min_df (default 2),
            cardinality (max feature combo size, default 1),
            keep (synthesizer-verifier rounds / heuristics, default 3).
    """
    max_features = int(kwargs.get("max_features", 150))
    min_df = int(kwargs.get("min_df", 2))
    cardinality = int(kwargs.get("cardinality", 1))
    keep = int(kwargs.get("keep", 3))

    np.random.seed(42 + seed)
    Xtr, Xva, Xte = _primitive_matrix(split, max_features, min_df)

    y_tr = np.asarray(split.train_y, dtype=np.int64)
    y_va = np.asarray(split.val_y, dtype=np.int64)
    L = split.n_labels

    # Per-label train + val marginals, for global-threshold tuning on val.
    train_scores = np.zeros((Xtr.shape[0], L), dtype=np.float64)
    val_scores = np.zeros((Xva.shape[0], L), dtype=np.float64)

    n_fit = 0
    with common.Timer() as t, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for lab in range(L):
            # Snuba {-1,+1} ground; skip degenerate labels (all one class in val).
            tg = np.where(y_tr[:, lab] > 0, 1.0, -1.0)
            vg = np.where(y_va[:, lab] > 0, 1.0, -1.0)
            if len(np.unique(vg)) < 2 or len(np.unique(tg)) < 2:
                continue
            tr_marg, va_marg = _snuba_one_label(Xtr, Xva, vg, tg,
                                                cardinality, keep)
            train_scores[:, lab] = tr_marg
            val_scores[:, lab] = va_marg
            n_fit += 1

    # Snuba labels the TRAIN (unlabeled) set; for a test-set Macro-F1 comparable
    # to the other end-to-end baselines, train a downstream classifier on the
    # Snuba-labeled train marginals and predict test. Use the same TF-IDF feats.
    metrics = _downstream_eval(Xtr, train_scores, val_scores, y_va, Xte,
                               split, seed)
    metrics["n_annotations"] = 0
    metrics["extra"] = {
        "source": "reef (HazyResearch) one-vs-rest",
        "labels_fit": int(n_fit), "max_features": max_features,
        "cardinality": cardinality, "keep": keep, "wall_sec": float(t.sec),
    }
    return metrics


def _downstream_eval(Xtr, train_scores, val_scores, y_va, Xte, split, seed):
    """Train an end model on Snuba's probabilistic train labels, score on test.

    Mirrors Snuba's protocol (the generative label model produces train labels;
    a downstream discriminative model is trained on them). We use a per-label
    logistic regression on the TF-IDF features with Snuba's soft train labels
    binarised at a val-tuned global threshold.
    """
    from sklearn.linear_model import LogisticRegression

    L = split.n_labels
    # Tune one global threshold on val to convert marginals -> hard train labels.
    thr = common.tune_threshold(val_scores, y_va)
    y_train_hat = (train_scores >= thr).astype(np.int8)

    test_pred = np.zeros((Xte.shape[0], L), dtype=np.int8)
    for lab in range(L):
        col = y_train_hat[:, lab]
        if col.sum() == 0 or col.sum() == len(col):
            # degenerate: predict the majority (all-0 if no positives)
            test_pred[:, lab] = int(col.sum() > len(col) / 2)
            continue
        clf = LogisticRegression(max_iter=200, C=1.0, solver="liblinear",
                                 random_state=seed)
        clf.fit(Xtr, col)
        test_pred[:, lab] = clf.predict(Xte).astype(np.int8)

    return common.score(split.test_y, test_pred)


BASELINES = {"snuba": snuba}
