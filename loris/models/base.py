"""
models/base.py
--------------
Abstract base class for all multi-label text classifiers in the Loris framework.

Every concrete subclass must implement ``fit``, ``predict``, and
``predict_proba``. The ``evaluate`` method is fully implemented here so that
metric definitions are identical across all models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class BaseDocumentClassifier(ABC):
    """
    Abstract base class for multi-label document classifiers.

    All subclasses accept raw text inputs (``List[str]``) in
    ``fit`` / ``predict`` / ``predict_proba`` and must return
    multi-hot numpy arrays of shape ``(n_samples, num_labels)``.

    Attributes
    ----------
    num_labels : int
        Total number of distinct labels (output dimension).
    config : Dict[str, Any]
        Arbitrary hyperparameter storage populated from ``**kwargs``.
    is_fitted : bool
        Set to ``True`` after ``fit()`` completes successfully.
    """

    num_labels: int
    config: Dict[str, Any]
    is_fitted: bool

    def __init__(self, num_labels: int, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        num_labels : int
            Number of output labels for multi-label classification.
        **kwargs
            Subclass-specific hyperparameters stored in ``self.config``.
        """
        self.num_labels = num_labels
        self.config = kwargs
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by every subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
    ) -> "BaseDocumentClassifier":
        """
        Train the model on labelled data.

        Parameters
        ----------
        X_train : List[str]
            Training text samples.
        y_train : np.ndarray
            Multi-hot label matrix, shape ``(n_train, num_labels)``,
            dtype ``float32`` or ``int``.
        X_val : List[str]
            Validation text samples (used for early stopping / tuning).
        y_val : np.ndarray
            Multi-hot label matrix, shape ``(n_val, num_labels)``.

        Returns
        -------
        BaseDocumentClassifier
            ``self``, to allow method chaining.
        """
        ...

    @abstractmethod
    def predict(self, X_test: List[str]) -> np.ndarray:
        """
        Predict binary multi-hot labels for the given texts.

        Parameters
        ----------
        X_test : List[str]
            Input text samples.

        Returns
        -------
        np.ndarray
            Multi-hot prediction matrix of shape ``(n_samples, num_labels)``,
            dtype ``int8``.
        """
        ...

    @abstractmethod
    def predict_proba(self, X_test: List[str]) -> np.ndarray:
        """
        Return per-label probability estimates.

        Parameters
        ----------
        X_test : List[str]
            Input text samples.

        Returns
        -------
        np.ndarray
            Probability matrix of shape ``(n_samples, num_labels)``,
            dtype ``float32``, values in ``[0, 1]``.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete evaluate — identical metric definition for all subclasses
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_test: List[str],
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute multi-label classification metrics on a test split.

        Metrics returned:
        - **micro_f1** — F1 computed globally across all label–sample pairs.
        - **macro_f1** — Unweighted average of per-label F1 scores.
        - **subset_accuracy** — Exact-match / Subset Accuracy: fraction of
          samples where the predicted label set exactly matches the true set.

        Parameters
        ----------
        X_test : List[str]
            Test text samples.
        y_test : np.ndarray
            Ground-truth multi-hot matrix, shape ``(n_samples, num_labels)``.

        Returns
        -------
        Dict[str, float]
            ``{"micro_f1": float, "macro_f1": float, "subset_accuracy": float}``

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        self._check_fitted()
        y_pred = self.predict(X_test)
        y_true = np.asarray(y_test, dtype=np.int8)

        micro_f1 = float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        )
        macro_f1 = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        # accuracy_score on 2-D arrays computes exact-match (subset accuracy)
        subset_acc = float(accuracy_score(y_true, y_pred))

        return {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "subset_accuracy": subset_acc,
        }

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise ``RuntimeError`` if the model has not been fitted yet."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted yet. "
                "Call fit() before predict/evaluate."
            )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"num_labels={self.num_labels}, status={status})"
        )
