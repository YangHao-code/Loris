"""
pattern_extraction/pattern_store.py
-------------------------------------
PatternStore — persistence and predicate application for extracted patterns.

After running ``PatternAbstractor.fit()``, call ``abstractor.to_store()`` to
get a :class:`PatternStore`.  The store separates two concerns that should not
live in the mining pipeline itself:

1. **Predicate application** — apply a fixed set of predicates to *any* new
   corpus and obtain a boolean feature matrix.  This is the entry point for
   downstream rule-discovery components that use predicates as rule bodies.

2. **Persistence** — save the predicate set to a JSON file and reload it in a
   later session (or a different codebase component) without re-running the
   full mining pipeline.

The serialization format is a plain JSON dict; every
:class:`~pattern_extraction.predicates.TextualPredicate` is round-tripped
through :func:`~pattern_extraction.predicates.predicate_to_dict` /
:func:`~pattern_extraction.predicates.predicate_from_dict`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np

from loris.document import Document
from loris.predicates import (
    Predicate,
    TextualPredicate,
    predicate_from_dict,
    predicate_to_dict,
)

logger = logging.getLogger(__name__)

#: Serialization format version — bump when the schema changes.
_STORE_VERSION = "0.1.0"


class PatternStore:
    """
    Immutable container for a fitted set of textual predicates.

    Obtained via :meth:`PatternAbstractor.to_store()` or loaded from disk
    via :meth:`load`.

    Parameters
    ----------
    patterns : List[Predicate]
        The predicates extracted and screened by the abstraction pipeline.
    n_classes : int
        Number of label classes known at extraction time.
    label_names : List[str] or None, optional
        Human-readable class names (from :class:`sklearn.LabelEncoder`
        when string labels were used).  ``None`` when integer labels or
        multi-hot labels were used.

    Examples
    --------
    .. code-block:: python

        from pattern_extraction import PatternAbstractor, PatternStore, Document

        # -- Mine patterns --------------------------------------------------
        abstractor = PatternAbstractor(n_clusters=4)
        abstractor.fit(train_texts, train_labels)
        store = abstractor.to_store()
        store.save("patterns.json")

        # -- Apply to new corpus (different session) -------------------------
        store = PatternStore.load("patterns.json")
        matrix = store.apply(new_texts)   # shape: (n_docs, n_patterns)

        # -- Inspect a single document --------------------------------------
        doc = Document(cnt="The bank raised interest rates.", ttl="Finance")
        matched = store.matching_predicates(doc)
        for pred in matched:
            print(pred)
    """

    def __init__(
        self,
        patterns: List[Predicate],
        n_classes: int,
        label_names: Optional[List[str]] = None,
    ) -> None:
        self._patterns: List[Predicate] = list(patterns)
        self.n_classes: int = n_classes
        self.label_names: Optional[List[str]] = label_names

    # ------------------------------------------------------------------
    # Predicate application
    # ------------------------------------------------------------------

    def apply(
        self,
        docs: Union[List[str], List[Document]],
        batch_size: int = 512,
        preprocessor=None,
    ) -> np.ndarray:
        """
        Apply all stored predicates to each document in *docs*.

        Parameters
        ----------
        docs : List[str] or List[Document]
            The corpus to evaluate.  Plain strings are automatically wrapped
            into :class:`~pattern_extraction.document.Document` objects
            (``cnt=text``), which means predicates targeting ``mtd`` or
            ``ttl`` will not match — pass full ``Document`` objects if those
            attributes are needed.
        batch_size : int, optional
            Number of documents processed per logging checkpoint.
            Does not affect results. Default 512.
        preprocessor : Preprocessor or None, optional
            If provided, each document is preprocessed before evaluation.

        Returns
        -------
        np.ndarray, shape (n_docs, n_patterns), dtype bool
            ``result[i, j]`` is ``True`` iff ``patterns[j]`` matches
            ``docs[i]``.  This boolean matrix can be used directly as a
            sparse feature matrix for downstream rule-based classifiers.
        """
        normalized: List[Document] = [
            Document(cnt=d) if isinstance(d, str) else d
            for d in docs
        ]

        if preprocessor is not None:
            normalized = [preprocessor.process_document(d) for d in normalized]

        n_docs = len(normalized)
        n_pats = len(self._patterns)
        result = np.zeros((n_docs, n_pats), dtype=bool)

        for i, doc in enumerate(normalized):
            for j, pred in enumerate(self._patterns):
                result[i, j] = pred(doc)

            if batch_size > 0 and (i + 1) % batch_size == 0:
                logger.debug(
                    "PatternStore.apply: processed %d / %d documents.",
                    i + 1, n_docs,
                )

        logger.info(
            "PatternStore.apply: %d docs × %d predicates → "
            "%d True entries (%.1f%%).",
            n_docs, n_pats,
            int(result.sum()),
            100.0 * result.mean() if result.size else 0.0,
        )
        return result

    def matching_predicates(
        self,
        doc: Union[str, Document],
    ) -> List[Predicate]:
        """
        Return the subset of stored predicates that match *doc*.

        Parameters
        ----------
        doc : str or Document
            A single document (plain string is auto-wrapped).

        Returns
        -------
        List[Predicate]
        """
        d = Document(cnt=doc) if isinstance(doc, str) else doc
        return [pred for pred in self._patterns if pred(d)]

    def coverage_stats(
        self,
        docs: Union[List[str], List[Document]],
    ) -> List[Dict[str, Any]]:
        """
        Compute per-predicate coverage statistics over *docs*.

        Useful for auditing how well patterns mined on a training set
        transfer to a new domain or test set.

        Parameters
        ----------
        docs : List[str] or List[Document]

        Returns
        -------
        List[dict]
            One dict per predicate with keys:
            ``"predicate"`` (repr), ``"type"``, ``"attr"``,
            ``"n_matching"`` (int), ``"coverage"`` (float 0–1).
            Sorted by ``coverage`` descending.
        """
        matrix = self.apply(docs)
        n_docs = len(docs)
        stats = []
        for j, pred in enumerate(self._patterns):
            n_match = int(matrix[:, j].sum())
            stats.append({
                "predicate": repr(pred),
                "type": type(pred).__name__,
                "attr": pred.attr if hasattr(pred, "attr") else None,
                "n_matching": n_match,
                "coverage": n_match / n_docs if n_docs else 0.0,
            })
        stats.sort(key=lambda d: d["coverage"], reverse=True)
        return stats

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persist the store to a JSON file at *path*.

        Schema::

            {
              "version": "0.1.0",
              "n_classes": int,
              "label_names": [...] | null,
              "n_patterns": int,
              "patterns": [
                {"type": "...", "attr": "cnt", ...},
                ...
              ]
            }

        Parameters
        ----------
        path : str
            Destination file path.  Parent directories are created if they
            do not exist.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "version": _STORE_VERSION,
            "n_classes": self.n_classes,
            "label_names": self.label_names,
            "n_patterns": len(self._patterns),
            "patterns": [predicate_to_dict(p) for p in self._patterns],
        }

        with dest.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        logger.info(
            "PatternStore saved %d predicates to %s.", len(self._patterns), dest
        )

    @classmethod
    def load(cls, path: str) -> "PatternStore":
        """
        Reconstruct a :class:`PatternStore` from a JSON file saved by
        :meth:`save`.

        Parameters
        ----------
        path : str

        Returns
        -------
        PatternStore

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file's version is not recognised.
        """
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"PatternStore file not found: {src}")

        with src.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        version = payload.get("version", "unknown")
        if version != _STORE_VERSION:
            logger.warning(
                "PatternStore file version %r differs from current %r.  "
                "Loading anyway — fields may be missing.",
                version, _STORE_VERSION,
            )

        patterns = [predicate_from_dict(d) for d in payload.get("patterns", [])]
        store = cls(
            patterns=patterns,
            n_classes=int(payload.get("n_classes", 0)),
            label_names=payload.get("label_names"),
        )
        logger.info(
            "PatternStore loaded %d predicates from %s.", len(patterns), src
        )
        return store

    # ------------------------------------------------------------------
    # Container interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._patterns)

    def __iter__(self) -> Iterator[Predicate]:
        return iter(self._patterns)

    def __getitem__(self, idx: int) -> Predicate:
        return self._patterns[idx]

    def __repr__(self) -> str:
        label_info = (
            f", label_names={self.label_names!r}"
            if self.label_names is not None
            else ""
        )
        type_counts: Dict[str, int] = {}
        for p in self._patterns:
            t = type(p).__name__
            type_counts[t] = type_counts.get(t, 0) + 1
        counts_str = ", ".join(
            f"{t}={n}" for t, n in sorted(type_counts.items())
        )
        return (
            f"PatternStore(n_patterns={len(self._patterns)}, "
            f"n_classes={self.n_classes}"
            f"{label_info}, [{counts_str}])"
        )
