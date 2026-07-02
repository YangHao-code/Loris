"""BESRA HITL baseline — adapter over the AUTHORS' beta-scoring code.

Drives the real ``B2M`` (generalized Pólya-Gamma Beta-to-Multilabel) model from
``refs/BESRA/gpb2m.py`` (Tan et al., AAAI'24; repo ``davidtw999/BESRA``) as the
acquisition function in a pool-based active-learning loop, replacing the
first-pass pure-uncertainty heuristic in ``loris/baselines/hitl.py``.

BESRA's contribution is the beta-scoring acquisition: a GP-over-logits model with
Beta-distributed per-label parameters whose ``ALsample`` ranks candidates by
expected score (entropy ``sampleType='en'``). The HITL protocol matches the paper
and our other HITL baseline: human = ground truth; seed a small labeled set, fit,
acquire a batch by beta-scoring, reveal GT, refit, until the annotation budget is
spent; the final discriminative model predicts test (Macro-F1 + #annotations).

SCALABILITY NOTE. ``B2M.fit`` builds a ``[K, N, N]`` GP covariance and inverts an
``N x N`` Gram matrix (O(N^2) memory, O(N^3) inverse) where N = labeled-pool size.
That is intrinsic to the method and fine for active learning (small budgets), but
we (a) featurize text to a compact TF-IDF+SVD embedding for the kernel and (b)
cap the candidate pool scored per round (``cand_cap``) so the GP stays tractable;
both are disclosed in the result ``extra``. The downstream classifier that
predicts TEST is the same TF-IDF SVM used by the other HITL baseline, trained on
the BESRA-acquired labeled set, so Macro-F1 is comparable.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Dict

import numpy as np

from loris.baselines import common
from loris.models.tfidf_classifier import TFIDFClassifier

_BESRA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                      "refs", "BESRA"))


def _embed(split, dim: int = 64, seed: int = 0):
    """Compact dense embedding of all train docs for the GP kernel + acquisition.

    TF-IDF -> TruncatedSVD(dim). Fit on train only; deterministic by seed.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(max_features=10000, sublinear_tf=True, min_df=2)
    Xtr = vec.fit_transform(split.train_X)
    d = max(2, min(dim, Xtr.shape[1] - 1, Xtr.shape[0] - 1))
    svd = TruncatedSVD(n_components=d, random_state=seed)
    emb = normalize(svd.fit_transform(Xtr)).astype(np.float64)
    return emb


def _besra_acquire(emb_lab, y_lab, emb_cand, batch, seed):
    """Use the authors' B2M beta-scoring to pick ``batch`` candidate indices.

    Returns the chosen positions into the candidate array. Falls back to random
    on any numerical failure (the GP can be ill-conditioned on tiny pools).
    """
    if _BESRA not in sys.path:
        sys.path.insert(0, _BESRA)
    from gpb2m import B2M
    from sklearn.gaussian_process.kernels import RBF

    try:
        model = B2M()
        K = max(2, min(10, y_lab.shape[1]))
        model.fit(emb_lab, y_lab.astype(float), K=K, seed=seed,
                  kernel_function=RBF(length_scale=1.0), K_threds=0.75)
        model.learn(learnIter=10)
        res = model.ALsample(emb_cand, eta=0, test=False, sampleType="en",
                             testX=None, testY=None)
        idx = np.asarray(res[:batch], dtype=int)
        if idx.size == 0:
            raise ValueError("empty acquisition")
        return idx
    except Exception:
        rng = np.random.RandomState(seed)
        return rng.choice(emb_cand.shape[0], min(batch, emb_cand.shape[0]),
                          replace=False)


def besra(split, seed: int = 0, *, budget: int = 400, batch: int = 50,
          seed_size: int = 50, cand_cap: int = 600, **kwargs) -> Dict:
    """BESRA beta-scoring active learning via the authors' B2M code (human=GT)."""
    n_train = len(split.train_X)
    rng = np.random.RandomState(seed)
    budget = min(int(budget), n_train)
    seed_size = min(int(seed_size), budget, n_train)

    emb = _embed(split, seed=seed)            # (n_train, d) for the GP kernel
    y_all = np.asarray(split.train_y, dtype=np.int64)

    # seed labeled set
    perm = rng.permutation(n_train)
    labeled = list(perm[:seed_size])
    labeled_mask = np.zeros(n_train, dtype=bool)
    labeled_mask[labeled] = True

    rounds = 0
    with common.Timer() as t, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while len(labeled) < budget:
            pool_idx = np.where(~labeled_mask)[0]
            if pool_idx.size == 0:
                break
            b = min(batch, budget - len(labeled), pool_idx.size)
            # cap candidates scored this round (GP is O(cand^2) in ALsample)
            if pool_idx.size > cand_cap:
                cand = rng.choice(pool_idx, cand_cap, replace=False)
            else:
                cand = pool_idx
            pick = _besra_acquire(emb[labeled], y_all[labeled], emb[cand],
                                  b, seed + rounds)
            chosen = np.asarray(cand)[pick]
            labeled.extend(chosen.tolist())
            labeled_mask[chosen] = True
            rounds += 1

        # downstream model trained on the BESRA-acquired labeled set
        clf = TFIDFClassifier(num_labels=split.n_labels,
                              classifier_type="svm", ngram_range=(1, 1))
        Xl = [split.train_X[i] for i in labeled]
        clf.fit(Xl, y_all[labeled], split.val_X, split.val_y)
        y_pred = clf.predict(split.test_X)

    metrics = common.score(split.test_y, y_pred)
    metrics["n_annotations"] = int(len(labeled))
    metrics["extra"] = {
        "source": "davidtw999/BESRA B2M beta-scoring (en)",
        "budget": int(budget), "rounds": int(rounds), "seed_size": int(seed_size),
        "batch": int(batch), "cand_cap": int(cand_cap), "wall_sec": float(t.sec),
    }
    return metrics


BASELINES = {"besra": besra}
