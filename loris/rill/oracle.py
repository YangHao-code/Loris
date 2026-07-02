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
    """Oracle backed by an LLM API (the noLLM ablation's *standard* arm).

    Paper §6.2 protocol, faithfully:
      1. ``query_with_evidence`` — the LLM pre-annotates the document.
      2. ``query_paraphrased``  — TrustChecker re-asks with a **paraphrased**
         prompt (stability / confidence check); the label must be unchanged.
      3. TrustChecker also runs a sandbox mini-Chase (local consistency).
      4. ``fallback`` — if the trust check fails, defer to the **human**
         (``GroundTruthOracle``). This is the only path that costs a human
         annotation, so ``n_human_queries`` measures the human budget the LLM
         pre-annotator saves.

    Determinism (no prompt-engineering bias): a small FIXED bank of
    semantically-equivalent prompt templates; ``query`` uses template 0 and
    ``query_paraphrased`` picks a *different* template deterministically by doc
    index. The LLM is called at temperature 0 (see ``LLMClient``).

    Parameters
    ----------
    label_names : List[str]
    ground_truth : np.ndarray, optional
        ``(n_docs, n_labels)`` — the human fallback answers (experiment mode).
    llm_client : LLMClient, optional
        Pre-built client; otherwise one is created from env (OPENAI_API_KEY /
        OPENAI_BASE_URL / LORIS_LLM_MODEL — SiliconFlow-compatible).
    """

    # Fixed, semantically-equivalent templates: (system, user_format). The
    # user_format takes {labels} and {text}. Committed here so paraphrasing is
    # reproducible and unbiased across a sweep (no LLM-generated paraphrase).
    _TEMPLATES = [
        ("You are a precise multi-label text classifier. Given a document and a "
         "fixed set of candidate labels, return ONLY a JSON array (possibly "
         "empty) of the labels that apply, drawn verbatim from the candidate set. "
         "No explanation.",
         "Candidate labels: [{labels}]\n\nDocument:\n{text}\n\n"
         "Return a JSON array of the applicable labels."),
        ("You assign topic labels to documents. From the allowed label list, "
         "output the subset that describes the document as a JSON array only "
         "(use the labels exactly as written; output [] if none apply).",
         "Allowed labels: [{labels}]\n\nText to label:\n{text}\n\n"
         "Which of the allowed labels apply? Respond with a JSON array."),
        ("Act as a document tagging system. Select every candidate tag that is "
         "relevant to the passage and reply with just a JSON array of those tags "
         "(verbatim from the candidates; [] if nothing fits).",
         "Tags to choose from: [{labels}]\n\nPassage:\n{text}\n\n"
         "List the relevant tags as a JSON array."),
    ]

    def __init__(
        self,
        label_names: Optional[List[str]] = None,
        ground_truth: Optional[np.ndarray] = None,
        llm_client: object = None,
        model_name: Optional[str] = None,
        max_chars: int = 6000,
    ) -> None:
        self.label_names = list(label_names or [])
        self.max_chars = max_chars
        self._last_evidence: Optional[str] = None
        self.n_queries = 0          # documents processed (compat)
        self.n_llm_calls = 0        # LLM API calls (query + stability re-query)
        self.n_human_queries = 0    # human fallbacks (trust-check failures)
        if llm_client is None:
            from loris.baselines.llm_client import LLMClient
            llm_client = LLMClient(model=model_name)
        self.client = llm_client
        self.model_name = getattr(llm_client, "model", model_name or "")
        self._human = (
            GroundTruthOracle(ground_truth, label_names)
            if ground_truth is not None else None
        )

    @staticmethod
    def _doc_text(doc: Document) -> str:
        ttl = getattr(doc, "ttl", None) or ""
        cnt = getattr(doc, "cnt", None) or ""
        return (ttl + ". " + cnt) if ttl else cnt

    def _classify(self, text: str, label_names: List[str], tmpl_idx: int) -> List[str]:
        system, user_fmt = self._TEMPLATES[tmpl_idx % len(self._TEMPLATES)]
        user = user_fmt.format(labels=", ".join(label_names), text=text[: self.max_chars])
        reply = self.client.chat(system, user)
        self.n_llm_calls += 1
        return self.client._parse_label_list(reply, label_names)

    def query(
        self, doc_idx: int, doc: Document, label_names: List[str],
    ) -> List[str]:
        return self._classify(self._doc_text(doc), label_names, 0)

    def query_with_evidence(
        self, doc_idx: int, doc: Document, label_names: List[str],
    ) -> Tuple[List[str], Optional[str]]:
        self.n_queries += 1
        labels = self._classify(self._doc_text(doc), label_names, 0)
        # No separate evidence document: trust is enforced by the stability
        # (paraphrase) + sandbox mini-chase checks, so evidence stays None
        # (TrustChecker skips the evidence step when get_last_evidence() is None).
        self._last_evidence = None
        return labels, self._last_evidence

    def query_paraphrased(
        self,
        doc_idx: int,
        doc: Document,
        label_names: List[str],
        evidence: Optional[str] = None,
    ) -> List[str]:
        # Deterministic: a DIFFERENT template than query(), chosen by doc index.
        n = len(self._TEMPLATES)
        tmpl = (1 + (doc_idx % (n - 1))) if n > 1 else 0
        return self._classify(self._doc_text(doc), label_names, tmpl)

    def fallback(
        self, doc_idx: int, doc: Document, label_names: List[str],
    ) -> Optional[List[str]]:
        """Trust check failed → the human annotator (ground truth) answers."""
        if self._human is None:
            return None
        self.n_human_queries += 1
        return self._human.query(doc_idx, doc, label_names)
