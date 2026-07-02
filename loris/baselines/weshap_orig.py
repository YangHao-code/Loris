"""WeShap pattern-selection (Group C) — adapter over the AUTHORS' code.

Drives the real ``WeShapAnalysis`` class from ``refs/WeShap/weshap.py`` (Guan et
al., VLDB; repo ``Gnaiqing/WeShap``) to score patterns by their Shapley value as
weak-supervision sources, replacing the first-pass Monte-Carlo proxy in
``loris/baselines/pattern_select.py``.

WeShap's closed-form Shapley (the ``sv_pos``/``sv_neg`` DP over LF vote counts,
neighbour-weighted by a KNN fit on majority-vote pseudo-labels) is inherently
**multi-class single-label** (one ``n_class``-way label per instance). LORIS is
multi-label, so we run WeShap **one-vs-rest**: for each label we build a binary
LF matrix from the candidate patterns, get per-pattern WeShap scores, and sum the
absolute contribution across labels to rank patterns globally. The top-``k`` then
feed the SAME downstream OvR-logistic model + metric as the other Group-C
methods (so only the ranking differs).

Each candidate *pattern* j becomes a labeling function:
  - fires (votes class 1) on docs where the binary pattern is present AND the
    pattern is positively associated with the label (P(label|present) >= prior);
    votes class 0 where present but negatively associated; abstains (-1) where
    absent. This is the standard "pattern -> LF" reduction used for WS pattern
    evaluation.

The authors' code calls ``wrench``'s MajorityVoting only to get KNN target
pseudo-labels; we supply a dependency-free majority-vote shim with the same
semantics (per row, most-frequent non-abstain class; ties -> lowest class).
"""

from __future__ import annotations

import os
import sys
import types
from typing import Tuple

import numpy as np

from loris.baselines import common

_WESHAP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                       "refs", "WeShap"))


def _install_label_model_shim() -> None:
    """Provide ``label_model.get_wrench_label_model`` without the wrench dep.

    weshap.py does ``from label_model import get_wrench_label_model`` and uses
    only ``MajorityVoting`` -> ``.fit(L)`` + ``.predict(L)``. We register a stub
    module implementing exactly that (per-row majority over non-abstain votes).
    """
    if "label_model" in sys.modules:
        return

    class _MajorityVoting:
        def fit(self, L, *a, **k):
            self._n_class = int(np.max(L)) + 1 if L.size else 2
            return self

        def predict(self, L):
            L = np.asarray(L)
            n, m = L.shape
            nc = getattr(self, "_n_class", int(np.max(L)) + 1 if L.size else 2)
            nc = max(nc, 2)
            out = np.zeros(n, dtype=int)
            for i in range(n):
                row = L[i]
                votes = row[row != -1]
                if votes.size == 0:
                    out[i] = 0
                else:
                    out[i] = int(np.bincount(votes, minlength=nc).argmax())
            return out

    mod = types.ModuleType("label_model")
    def get_wrench_label_model(method, **kwargs):
        return _MajorityVoting()
    mod.get_wrench_label_model = get_wrench_label_model
    sys.modules["label_model"] = mod


class _DS:
    """Minimal dataset duck-type matching what WeShapAnalysis reads.

    WeShapAnalysis uses: ``.n_class``, ``.weak_labels`` (n x m), ``.features``
    (n x d), ``.labels`` (n,), and ``len(dataset)``.
    """
    def __init__(self, weak_labels, features, labels, n_class):
        self.weak_labels = weak_labels
        self.features = features
        self.labels = np.asarray(labels, dtype=int)
        self.n_class = int(n_class)

    def __len__(self):
        return len(self.labels)


def _pattern_lfs_for_label(Xtr_bin: np.ndarray, ytr_col: np.ndarray) -> np.ndarray:
    """Build a binary (n x m) LF matrix from patterns for ONE label.

    LF j (pattern j): where present, vote 1 if P(label|present)>=P(label), else
    vote 0; where absent, abstain (-1). Captures both positive and negative
    pattern-label association as a single 2-class LF per pattern.
    """
    n, m = Xtr_bin.shape
    prior = float(ytr_col.mean()) if n else 0.0
    L = -np.ones((n, m), dtype=int)
    present = Xtr_bin > 0
    for j in range(m):
        col = present[:, j]
        if not col.any():
            continue
        p_label_given = float(ytr_col[col].mean()) if col.any() else 0.0
        vote = 1 if p_label_given >= prior else 0
        L[col, j] = vote
    return L


def _weshap_rank(split, seed: int, n_patterns: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, "object"]:
    """Return WeShap global pattern ranking + the Group-C feature matrices."""
    _install_label_model_shim()
    if _WESHAP not in sys.path:
        sys.path.insert(0, _WESHAP)
    from weshap import WeShapAnalysis

    # Reuse the Group-C pattern matrix builder for parity with the other methods.
    from loris.baselines.pattern_select import _build_pattern_matrix
    Xtr, Xval, Xte, vec = _build_pattern_matrix(split, n_patterns=n_patterns)

    ytr = np.asarray(split.train_y, dtype=int)
    yval = np.asarray(split.val_y, dtype=int)
    n_lbl = ytr.shape[1]
    m = Xtr.shape[1]

    # KNN features: the binary pattern vectors themselves (euclidean over them).
    scores = np.zeros(m, dtype=np.float64)
    rng = np.random.RandomState(seed)
    for lab in range(n_lbl):
        ytr_col = ytr[:, lab]
        yval_col = yval[:, lab]
        if ytr_col.sum() == 0 or yval_col.sum() == 0:
            continue
        L_train = _pattern_lfs_for_label(Xtr, ytr_col)
        train_ds = _DS(L_train, Xtr, ytr_col, n_class=2)
        # valid set: features + true binary labels (WeShap scores LFs by how well
        # they help predict val labels via the KNN neighbourhood).
        valid_ds = _DS(_pattern_lfs_for_label(Xval, yval_col), Xval, yval_col, 2)
        try:
            wa = WeShapAnalysis(train_ds, valid_ds, n_neighbors=min(5, len(train_ds) - 1))
            s = wa.calculate_weshap_score()
            scores += np.abs(np.asarray(s, dtype=np.float64))
        except Exception:
            continue

    order = np.argsort(-scores)
    return order, Xtr, Xval, Xte, ytr, yval, vec


def weshap(split, seed: int = 0, *, top_k: int = 200, n_patterns: int = 1000,
           **kwargs) -> dict:
    """WeShap (Shapley-value pattern selection) via the authors' code."""
    order, Xtr, Xval, Xte, ytr, yval, vec = _weshap_rank(split, seed, n_patterns)
    from loris.baselines.pattern_select import _fit_score

    n_feat = Xtr.shape[1]
    k = min(int(top_k), n_feat)
    cols = np.asarray(order[:k], dtype=int)
    with common.Timer() as t:
        metrics = _fit_score(Xtr, ytr, Xval, yval, Xte,
                             np.asarray(split.test_y), cols, seed)
    metrics["extra"] = {
        "source": "Gnaiqing/WeShap one-vs-rest", "top_k": int(k),
        "n_patterns": int(n_feat), "ranker": "weshap", "wall_sec": float(t.sec),
    }
    metrics["n_annotations"] = 0
    return metrics


BASELINES = {"weshap": weshap}
