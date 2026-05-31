"""StructuralFeatureModel — document structure features + LogisticRegression.

Extracts deterministic structural features from text (negation density,
transition words, citation patterns, abbreviations, math symbols, etc.)
and trains a lightweight LR classifier. These features are completely
independent of specific vocabulary, making them highly stable across splits.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from loris.models.base import BaseDocumentClassifier


def _extract_structural_features(text: str) -> np.ndarray:
    """Extract universal structural features from a document."""
    words = text.split()
    n_words = max(len(words), 1)
    n_sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)

    negation = len(re.findall(
        r"\b(not|no|never|without|unlike|neither|nor|cannot|can't|won't|don't|doesn't|isn't|aren't|wasn't|weren't)\b",
        text, re.I
    ))
    transition = len(re.findall(
        r"\b(however|but|although|despite|yet|nevertheless|moreover|furthermore|therefore|thus|hence|consequently)\b",
        text, re.I
    ))
    citations = len(re.findall(r"\[\d+\]|\([A-Z][a-z]+ et al", text))
    abbreviations = len(re.findall(r"\b[A-Z]{2,}\b", text))
    math_symbols = 1.0 if re.search(r"\\[a-z]+|\$.*?\$|[∀∃∈∑∏∫≤≥≠±]", text) else 0.0
    percentages = len(re.findall(r"\d+\.?\d*%", text))
    questions = text.count("?")
    parentheticals = len(re.findall(r"\([^)]+\)", text))
    enumerations = len(re.findall(r"(?:^|\n)\s*(?:\d+[.)]\s|[-•]\s)", text))
    avg_word_len = sum(len(w) for w in words) / n_words
    long_words = sum(1 for w in words if len(w) > 10)

    return np.array([
        np.log1p(n_words),
        np.log1p(n_sentences),
        negation / n_words * 100,
        transition / n_words * 100,
        citations / n_sentences,
        abbreviations / n_words * 100,
        math_symbols,
        percentages / n_sentences,
        questions / n_sentences,
        parentheticals / n_sentences,
        enumerations / n_sentences,
        avg_word_len,
        long_words / n_words * 100,
    ], dtype=np.float32)


N_STRUCTURAL_FEATURES = 13


class StructuralFeatureModel(BaseDocumentClassifier):
    """Structural features + LogisticRegression as probability output.

    Parameters
    ----------
    num_labels : int
        Number of output labels.
    C : float
        Regularization strength for LogisticRegression.
    """

    def __init__(
        self,
        num_labels: int,
        C: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_labels, **kwargs)
        self._C = C
        self._scaler: Optional[StandardScaler] = None
        self._clf: Optional[LogisticRegression] = None

    def _featurize(self, texts: List[str]) -> np.ndarray:
        return np.array(
            [_extract_structural_features(t) for t in texts],
            dtype=np.float32,
        )

    def fit(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
    ) -> "StructuralFeatureModel":
        y_train = np.asarray(y_train)
        X_feat = self._featurize(X_train)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_feat)

        from sklearn.multiclass import OneVsRestClassifier
        base_clf = LogisticRegression(
            C=self._C, max_iter=1000, solver="lbfgs"
        )
        self._clf = OneVsRestClassifier(base_clf, n_jobs=-1)
        self._clf.fit(X_scaled, y_train)

        self.is_fitted = True
        return self

    def predict_proba(self, X_test: List[str]) -> np.ndarray:
        self._check_fitted()
        X_feat = self._featurize(X_test)
        X_scaled = self._scaler.transform(X_feat)
        proba = self._clf.predict_proba(X_scaled)
        if isinstance(proba, list):
            proba = np.column_stack([p[:, 1] for p in proba])
        return proba.astype(np.float32)

    def predict(self, X_test: List[str]) -> np.ndarray:
        proba = self.predict_proba(X_test)
        return (proba >= 0.5).astype(np.int8)
