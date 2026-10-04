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


def fixed_axis_metrics(truth, prediction):
    """Score positive membership on a fixed [documents, labels] binary axis.

    In particular, a single column remains one label, not two classes to
    average over. Undefined positive-label F1 is zero, including empty support.
    """
    y, p = np.asarray(truth), np.asarray(prediction)
    if (y.ndim != 2 or not y.size or y.shape != p.shape
            or not np.isin(y, (0, 1)).all() or not np.isin(p, (0, 1)).all()):
        raise ValueError("matching nonempty binary document-by-label axes required")
    y, p = y.astype(bool), p.astype(bool)
    tp = (y & p).sum(axis=0)
    denominator = 2 * tp + (~y & p).sum(axis=0) + (y & ~p).sum(axis=0)
    per_label = np.divide(
        2. * tp, denominator, out=np.zeros(len(tp)), where=denominator > 0
    )
    return dict(
        micro_f1=float(2 * tp.sum() / max(1, denominator.sum())),
        macro_f1=float(per_label.mean()),
        subset_accuracy=float((y == p).all(axis=1).mean()),
    )


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

    def __init__(self, num_labels: int, *, task_type: str = "multilabel", **kwargs: Any) -> None:
        """
        Parameters
        ----------
        num_labels : int
            Number of output labels for multi-label classification.
        task_type : str
            Explicit ``"multilabel"`` (default) or ``"single_label"``. Training
            rows that happen to be one-hot never infer task exclusivity.
        **kwargs
            Subclass-specific hyperparameters stored in ``self.config``.
        """
        if type(num_labels) is not int or num_labels < 1:
            raise ValueError("num_labels must be a positive integer")
        if task_type not in ("multilabel", "single_label"):
            raise ValueError("task_type must be multilabel or single_label; automatic inference is unsupported")
        self.num_labels = num_labels
        self.task_type = task_type
        self._single_label = task_type == "single_label"
        self.config = dict(kwargs, task_type=task_type)
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
        if np.asarray(y_test).shape != (len(X_test), self.num_labels):
            raise ValueError("evaluation targets differ from the declared label axis")
        return fixed_axis_metrics(y_test, y_pred)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _validated_targets(self, targets, n_rows):
        """Check declared task semantics before allocating/training a model."""
        mode = getattr(self, "task_type", None)
        if mode not in ("multilabel", "single_label"):
            raise ValueError("refitting a historical model requires an explicit task_type")
        y = np.asarray(targets)
        if y.shape != (n_rows, self.num_labels) or not np.isin(y, (0, 1)).all():
            raise ValueError("fully observed binary targets on the declared label axis required")
        self._single_label = mode == "single_label"
        if self._single_label and np.any(y.sum(axis=1) != 1):
            raise ValueError("single_label targets must contain exactly one positive label per document")
        return y.astype(np.float32)

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
