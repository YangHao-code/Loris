"""EnsembleVoteModel — multi-model consensus voting.

Aggregates predictions from multiple already-trained labeler models.
Output = fraction of models that predict a label as positive. This is
extremely stable across splits because model predictions are deterministic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from loris.models.base import BaseDocumentClassifier
from loris.predicates._core import _ML_MODEL_CACHE


class EnsembleVoteModel(BaseDocumentClassifier):
    """Multi-model voting consensus as probability output.

    Parameters
    ----------
    num_labels : int
        Number of output labels.
    source_model_names : List[str]
        Names of registered ML models to aggregate votes from.
    vote_threshold : float
        Threshold for each source model's per-label probability to count
        as a positive vote.
    """

    def __init__(
        self,
        num_labels: int,
        source_model_names: Optional[List[str]] = None,
        vote_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_labels, **kwargs)
        self._source_names = source_model_names or []
        self._vote_threshold = vote_threshold
        self._active_models: List[str] = []

    def fit(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
    ) -> "EnsembleVoteModel":
        self._active_models = [
            name for name in self._source_names
            if name in _ML_MODEL_CACHE
        ]
        if not self._active_models:
            self._active_models = [
                name for name in _ML_MODEL_CACHE.keys()
                if name.startswith("loris_clf_") and "ensemble" not in name
            ]
        self.is_fitted = True
        return self

    def predict_proba(self, X_test: List[str]) -> np.ndarray:
        self._check_fitted()
        n = len(X_test)

        if not self._active_models:
            return np.zeros((n, self.num_labels), dtype=np.float32)

        votes = np.zeros((n, self.num_labels), dtype=np.float32)

        for model_name in self._active_models:
            model = _ML_MODEL_CACHE.get(model_name)
            if model is None:
                continue
            if hasattr(model, "_clf") and hasattr(model._clf, "predict_proba"):
                proba = model._clf.predict_proba(X_test)
            elif hasattr(model, "predict_proba_single"):
                proba = np.array(
                    [model.predict_proba_single(t) for t in X_test],
                    dtype=np.float32,
                )
            else:
                continue
            votes += (proba >= self._vote_threshold).astype(np.float32)

        n_models = len(self._active_models)
        return (votes / max(n_models, 1)).astype(np.float32)

    def predict(self, X_test: List[str]) -> np.ndarray:
        proba = self.predict_proba(X_test)
        return (proba >= 0.5).astype(np.int8)
