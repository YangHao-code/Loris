"""
chase_inference/oracle.py
--------------------------
Oracle interface for the RILL active-learning loop.

The oracle provides labels for unlabeled documents when the Chase engine
reaches fixpoint.  Two concrete implementations:

- ``GroundTruthOracle``: reads from ground-truth labels (experiment mode).
- ``LLMOracle``: calls an LLM API with evidence-document and stability
  protocols (paper §6.2, deployment mode — stub for now).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from loris.document import Document

logger = logging.getLogger(__name__)


class OracleBase(ABC):
    """Abstract base class for label oracles."""

    @abstractmethod
    def query(
        self, doc_idx: int, doc: Document, label_names: List[str],
    ) -> List[str]:
        """Return **all** predicted labels for *doc* (one human interaction).

        A single query represents one human / LLM interaction — the annotator
        reads the document and provides *all* applicable labels at once.
        """
        ...

    def query_with_evidence(
        self, doc_idx: int, doc: Document, label_names: List[str],
    ) -> Tuple[List[str], Optional[str]]:
        """Return ``(predicted_labels, evidence_text)``.

        Paper §6.2 requires LLM to provide an evidence document E alongside
        the labels.  Default implementation returns no evidence.
        """
        return self.query(doc_idx, doc, label_names), None

    def query_paraphrased(
        self,
        doc_idx: int,
        doc: Document,
        label_names: List[str],
        evidence: Optional[str] = None,
    ) -> List[str]:
        """Re-query with a paraphrased prompt + evidence E (stability check).

        Paper §6.2 mandates that the same question asked differently must
        yield the same labels for the result to be trusted.  Default returns
        the same as ``query()``.
        """
        return self.query(doc_idx, doc, label_names)

    def get_last_evidence(self) -> Optional[str]:
        """Return the evidence text cached from the most recent query."""
        return getattr(self, "_last_evidence", None)

    def fallback(
        self, doc_idx: int, doc: Document, label_names: List[str],
    ) -> Optional[List[str]]:
        """Fallback labels when trust check fails.  ``None`` = skip document."""
        return None


# =========================================================================
# GroundTruthOracle — experiment mode
# =========================================================================


class GroundTruthOracle(OracleBase):
    """Oracle backed by ground-truth labels (for controlled experiments).

    Since GT is always correct, ``TrustChecker`` skips the evidence and
    stability steps (Steps 1–2) and only runs the sandbox mini-Chase
    (Step 3).

    Parameters
    ----------
    ground_truth : np.ndarray
        ``(n_docs, n_labels)`` multi-hot bool/int matrix.
    label_names : List[str]
        Ordered label vocabulary matching columns of *ground_truth*.
    """

    def __init__(
        self, ground_truth: np.ndarray, label_names: List[str],
    ) -> None:
        self.ground_truth = np.asarray(ground_truth, dtype=bool)
        self.label_names = list(label_names)
        self.n_queries = 0

    def query(
        self, doc_idx: int, doc: Document, label_names: List[str],
    ) -> List[str]:
        """Return all true labels for *doc_idx* (one human interaction).

        Each call = one annotator reading the document and providing all
        applicable labels.
        """
        self.n_queries += 1
        pos_cols = np.where(self.ground_truth[doc_idx])[0]
        if len(pos_cols) == 0:
            logger.warning(
                "GroundTruthOracle: doc %d has no GT labels.", doc_idx,
            )
            return []
        return [self.label_names[int(c)] for c in pos_cols]

    def fallback(
        self, doc_idx: int, doc: Document, label_names: List[str],
    ) -> List[str]:
        """GT oracle always has an answer — fallback returns the same."""
        return self.query(doc_idx, doc, label_names)


# =========================================================================
# LLMOracle — deployment mode (stub)
# =========================================================================


class LLMOracle(OracleBase):
    """Oracle backed by an LLM API (deployment mode).

    Implements the full paper §6.2 prompt protocol:
    1. ``query_with_evidence`` — ask LLM for label + evidence document E.
    2. ``query_paraphrased`` — re-ask with paraphrased prompt + evidence.
    3. TrustChecker runs sandbox mini-Chase on the label.

    .. note::
       This is a stub.  Concrete API integration (OpenAI, Anthropic, etc.)
       should be added when moving to deployment.
    """

    def __init__(
        self,
        model_name: str = "gpt-4",
        label_names: Optional[List[str]] = None,
    ) -> None:
        self.model_name = model_name
        self.label_names = label_names or []
        self._last_evidence: Optional[str] = None
        self.n_queries = 0

    def query(
        self, doc_idx: int, doc: Document, label_names: List[str],
    ) -> List[str]:
        raise NotImplementedError(
            "LLMOracle.query() is a stub — implement with actual LLM API."
        )

    def query_with_evidence(
        self, doc_idx: int, doc: Document, label_names: List[str],
    ) -> Tuple[List[str], Optional[str]]:
        """Ask LLM: classify document AND provide evidence document E.

        Returns ``(labels, evidence_text)``.
        """
        self.n_queries += 1
        raise NotImplementedError(
            "LLMOracle.query_with_evidence() is a stub."
        )

    def query_paraphrased(
        self,
        doc_idx: int,
        doc: Document,
        label_names: List[str],
        evidence: Optional[str] = None,
    ) -> List[str]:
        """Re-ask with paraphrased prompt + evidence E (stability check).

        Returns list of labels.
        """
        self.n_queries += 1
        raise NotImplementedError(
            "LLMOracle.query_paraphrased() is a stub."
        )
