"""
models/tfidf_classifier.py
--------------------------
Traditional feature-based multi-label classifier.

Supports two classifier backends selectable via ``classifier_type``:
- ``"svm"`` (default): TF-IDF → OneVsRest(LinearSVC)
- ``"logistic_regression"``: TF-IDF → OneVsRest(LogisticRegression)

Logistic Regression natively outputs calibrated probabilities via
``predict_proba``. For LinearSVC, ``predict_proba`` applies a sigmoid to
decision-function scores (fast approximation), or full Platt scaling when
``use_calibration=True``.
"""

from __future__ import annotations

import inspect
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.special import expit  # numerically stable sigmoid
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from loris.models.base import BaseDocumentClassifier


class TFIDFClassifier(BaseDocumentClassifier):
    """
    TF-IDF + linear classifier for multi-label classification.

    Uses scikit-learn's ``OneVsRestClassifier`` to decompose the multi-label
    problem into one binary classifier per label.

    Parameters
    ----------
    num_labels : int
        Number of output labels.
    classifier_type : str
        Backend classifier: ``"svm"`` (LinearSVC, default) or
        ``"logistic_regression"`` (LogisticRegression).
    max_features : int
        Maximum vocabulary size for TF-IDF (default 50 000).
    ngram_range : Tuple[int, int]
        Lower and upper n-gram boundaries for the vectorizer (default (1, 2)).
    min_df : int or float
        Minimum document frequency for a token to be included (default 2).
    C : float
        Regularisation parameter for both LinearSVC and LogisticRegression
        (default 1.0).
    use_calibration : bool
        When ``classifier_type="svm"`` and ``use_calibration=True``, wraps
        LinearSVC with ``CalibratedClassifierCV`` for proper Platt-scaled
        probabilities (slower). Ignored when using Logistic Regression.
        Default ``False``.
    task_type : str
        ``"multilabel"`` (default) preserves independent label predictions.
        Explicit ``"single_label"`` uses direct multiclass LR, or one-vs-rest
        SVM with argmax decoding. One-hot training rows do not select a mode.
    **kwargs
        Forwarded to ``BaseDocumentClassifier.__init__``.

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    classifier : OneVsRestClassifier
        Fitted one-vs-rest classifier.
    """

    def __init__(
        self,
        num_labels: int,
        classifier_type: str = "svm",
        max_features: int = 50_000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        C: float = 1.0,
        use_calibration: bool = False,
        task_type: str = "multilabel",
        **kwargs: Any,
    ) -> None:
        super().__init__(num_labels, task_type=task_type, **kwargs)
        classifier_type = classifier_type.lower()
        if classifier_type not in ("svm", "logistic_regression"):
            raise ValueError(
                f"classifier_type must be 'svm' or 'logistic_regression', "
                f"got '{classifier_type}'"
            )
        self.classifier_type = classifier_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.C = C
        self.use_calibration = use_calibration

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[OneVsRestClassifier] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
    ) -> "TFIDFClassifier":
        """
        Fit TF-IDF vectorizer and linear classifier on training data.

        Validation data are accepted for API consistency but are not used
        (linear models do not require early stopping).

        Parameters
        ----------
        X_train : List[str]
            Training text samples.
        y_train : np.ndarray
            Multi-hot label matrix, shape ``(n_train, num_labels)``.
        X_val : List[str]
            Validation texts (unused).
        y_val : np.ndarray
            Validation labels (unused).

        Returns
        -------
        TFIDFClassifier
            ``self`` for method chaining.
        """
        y_train = self._validated_targets(y_train, len(X_train)).astype(np.int32)
        self._constant_prediction = None

        # Build TF-IDF features
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,  # apply 1 + log(tf) scaling
        )
        X_vec = self.vectorizer.fit_transform(X_train)
        if np.all(y_train == y_train[0]):
            # In particular, sklearn's single-column binarizer can encode an
            # all-positive target as its sole class's zero indicator.
            self._constant_prediction = y_train[0].copy()
            self.classifier = None
            self.is_fitted = True
            return self

        # Build the inner classifier
        if self.classifier_type == "logistic_regression":
            if self._single_label:
                # Single-label: use multinomial LR directly (no OvR wrapper)
                # Preserve the existing option on old sklearn; newer releases
                # removed it and select their supported objective automatically.
                options = (
                    {"multi_class": "multinomial"}
                    if "multi_class" in inspect.signature(LogisticRegression).parameters
                    else {}
                )
                self.classifier = LogisticRegression(
                    C=self.C,
                    max_iter=2000,
                    solver="lbfgs",
                    **options,
                )
                self.classifier.fit(X_vec, y_train.argmax(axis=1))
            else:
                inner = LogisticRegression(
                    C=self.C,
                    max_iter=2000,
                    solver="lbfgs",
                )
                self.classifier = OneVsRestClassifier(inner, n_jobs=self.config.get("n_jobs", -1))
                self.classifier.fit(X_vec, y_train)
        else:  # svm
            base_svc = LinearSVC(C=self.C, max_iter=2000)
            inner = (
                CalibratedClassifierCV(base_svc, cv=3, method="sigmoid")
                if self.use_calibration
                else base_svc
            )
            self.classifier = OneVsRestClassifier(inner, n_jobs=self.config.get("n_jobs", -1))
            self.classifier.fit(X_vec, y_train)

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X_test: List[str]) -> np.ndarray:
        """
        Return multi-hot binary predictions.

        Parameters
        ----------
        X_test : List[str]
            Input text samples.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, num_labels)``, dtype ``int8``.
        """
        self._check_fitted()
        if getattr(self, "_constant_prediction", None) is not None:
            return np.tile(self._constant_prediction, (len(X_test), 1)).astype(np.int8)
        if (self._single_label and self.classifier_type == "svm"
                and getattr(self, "task_type", None) == "single_label"):
            probabilities = self.predict_proba(X_test)
            predictions = np.zeros_like(probabilities, dtype=np.int8)
            predictions[np.arange(len(X_test)), probabilities.argmax(axis=1)] = 1
            return predictions
        X_vec = self.vectorizer.transform(X_test)

        if self._single_label and self.classifier_type == "logistic_regression":
            # Multinomial LR returns 1D class indices; convert to multi-hot
            class_preds = self.classifier.predict(X_vec)
            preds = np.zeros((len(X_test), self.num_labels), dtype=np.int8)
            preds[np.arange(len(X_test)), class_preds] = 1
            return preds

        return np.stack(
            [estimator.predict(X_vec) for estimator in self.classifier.estimators_], axis=1
        ).astype(np.int8)

    def predict_proba(self, X_test: List[str]) -> np.ndarray:
        """
        Return per-label probability estimates.

        - ``"logistic_regression"``: calibrated probabilities from LR.
        - ``"svm"`` + ``use_calibration=True``: Platt-scaled probabilities.
        - ``"svm"`` + ``use_calibration=False`` (default): sigmoid applied to
          SVM decision-function scores (fast, monotonically related to P).

        Parameters
        ----------
        X_test : List[str]
            Input text samples.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, num_labels)``, dtype ``float32``.
        """
        self._check_fitted()
        if getattr(self, "_constant_prediction", None) is not None:
            return np.tile(self._constant_prediction, (len(X_test), 1)).astype(np.float32)
        X_vec = self.vectorizer.transform(X_test)

        if self._single_label and self.classifier_type == "logistic_regression":
            probs = np.zeros((len(X_test), self.num_labels), dtype=np.float32)
            probs[:, self.classifier.classes_] = self.classifier.predict_proba(X_vec)
            return probs

        values = []
        for estimator in self.classifier.estimators_:
            if hasattr(estimator, "y_"):
                # OvR substitutes constant estimators for all-zero/all-one
                # columns. Their decision scores must not be sigmoid-transformed.
                values.append(np.full(len(X_test), float(np.asarray(estimator.y_).item())))
            elif self.classifier_type == "logistic_regression" or self.use_calibration:
                values.append(estimator.predict_proba(X_vec)[:, 1])
            else:
                values.append(expit(estimator.decision_function(X_vec)))
        return np.stack(values, axis=1).astype(np.float32)
