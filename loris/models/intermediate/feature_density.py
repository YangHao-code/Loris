"""FeatureDensityModel — chi-square keyword hit rate per label.

For each label, selects top-K most discriminative words via chi-square test,
then outputs the fraction of those keywords present in a document. This is
more robust than exact regex matching because it's a statistical density
measure that tolerates partial keyword absence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

from loris.models.base import BaseDocumentClassifier


class FeatureDensityModel(BaseDocumentClassifier):
    """Per-label keyword density as probability output.

    Parameters
    ----------
    num_labels : int
        Number of output labels.
    top_k_keywords : int
        Number of top chi-square keywords per label.
    min_df : int
        Minimum document frequency for vocabulary.
    max_df : float
        Maximum document frequency ratio for vocabulary.
    """

    def __init__(
        self,
        num_labels: int,
        top_k_keywords: int = 20,
        min_df: int = 3,
        max_df: float = 0.8,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_labels, **kwargs)
        self._top_k = top_k_keywords
        self._min_df = min_df
        self._max_df = max_df
        self._label_keywords: Optional[List[Set[str]]] = None
        self._vocab: Optional[List[str]] = None

    def fit(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
    ) -> "FeatureDensityModel":
        y_train = np.asarray(y_train)

        vectorizer = CountVectorizer(
            min_df=self._min_df,
            max_df=self._max_df,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
            stop_words="english",
        )
        X_counts = vectorizer.fit_transform(X_train)
        self._vocab = vectorizer.get_feature_names_out()

        self._label_keywords = []
        for i in range(self.num_labels):
            labels_i = y_train[:, i].astype(np.float64)
            if labels_i.sum() == 0 or labels_i.sum() == len(labels_i):
                self._label_keywords.append(set())
                continue

            scores, _ = chi2(X_counts, labels_i)
            scores = np.nan_to_num(scores, nan=0.0)
            top_indices = np.argsort(scores)[-self._top_k:]
            keywords = set(self._vocab[idx] for idx in top_indices)
            self._label_keywords.append(keywords)

        self.is_fitted = True
        return self

    def predict_proba(self, X_test: List[str]) -> np.ndarray:
        self._check_fitted()
        n = len(X_test)
        proba = np.zeros((n, self.num_labels), dtype=np.float32)

        for doc_idx, text in enumerate(X_test):
            words = set(text.lower().split())
            for label_idx, keywords in enumerate(self._label_keywords):
                if not keywords:
                    continue
                hits = len(words & keywords)
                proba[doc_idx, label_idx] = hits / len(keywords)

        return proba

    def predict(self, X_test: List[str]) -> np.ndarray:
        proba = self.predict_proba(X_test)
        return (proba >= 0.3).astype(np.int8)
