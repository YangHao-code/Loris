"""Explicit initial ground-truth state for the paper's Chase semantics.

``GammaSeed`` represents the validated label-equivalence facts ``Gamma``
that initialise ``LBL``.  It is deliberately separate from both
``Document.lbl`` (a mutable output/view field) and classifier predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class GammaSeed:
    """Validated positive and negative label facts used to initialise Chase.

    Both arrays are immutable boolean matrices with shape
    ``(n_documents, n_labels)``.  A valid ``Gamma`` cannot assert and
    exclude the same ``(document, label)`` fact.
    """

    pos: np.ndarray
    neg: np.ndarray

    def __post_init__(self) -> None:
        pos = np.array(self.pos, dtype=bool, copy=True)
        neg = np.array(self.neg, dtype=bool, copy=True)
        if pos.ndim != 2:
            raise ValueError(f"GammaSeed.pos must be 2-D, got shape {pos.shape}")
        if neg.ndim != 2:
            raise ValueError(f"GammaSeed.neg must be 2-D, got shape {neg.shape}")
        if pos.shape != neg.shape:
            raise ValueError(
                f"GammaSeed pos/neg shapes differ: {pos.shape} != {neg.shape}"
            )
        if np.any(pos & neg):
            first = tuple(int(v) for v in np.argwhere(pos & neg)[0])
            raise ValueError(
                "GammaSeed is inconsistent: the same fact is positive and "
                f"negative at index {first}"
            )
        pos.setflags(write=False)
        neg.setflags(write=False)
        object.__setattr__(self, "pos", pos)
        object.__setattr__(self, "neg", neg)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.pos.shape

    def validate_shape(self, n_docs: int, n_labels: int) -> None:
        expected = (n_docs, n_labels)
        if self.shape != expected:
            raise ValueError(
                f"GammaSeed shape {self.shape} != expected {expected}"
            )

    @classmethod
    def empty(cls, n_docs: int, n_labels: int) -> "GammaSeed":
        if n_docs < 0 or n_labels < 0:
            raise ValueError("n_docs and n_labels must be non-negative")
        shape = (n_docs, n_labels)
        return cls(np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool))

    @classmethod
    def from_multihot(
        cls,
        labels: np.ndarray,
        *,
        doc_mask: Optional[np.ndarray] = None,
        complete: bool = False,
    ) -> "GammaSeed":
        """Build ``Gamma`` from validated multi-hot annotations.

        ``doc_mask`` selects which document rows have been validated.  By
        default only positive labels become facts.  Set ``complete=True`` only
        when the annotation is known to be exhaustive; zeros on selected rows
        then become validated negative facts.
        """
        raw = np.asarray(labels)
        if raw.ndim != 2:
            raise ValueError(f"labels must be 2-D, got shape {raw.shape}")
        if not np.isin(raw, (0, 1, False, True)).all():
            raise ValueError("labels must be a binary multi-hot matrix")
        binary = raw.astype(bool, copy=False)
        n_docs, n_labels = binary.shape
        if doc_mask is None:
            selected = np.ones(n_docs, dtype=bool)
        else:
            selected = np.asarray(doc_mask, dtype=bool)
            if selected.shape != (n_docs,):
                raise ValueError(
                    f"doc_mask shape {selected.shape} != expected ({n_docs},)"
                )
        rows = selected[:, None]
        pos = binary & rows
        neg = (~binary & rows) if complete else np.zeros((n_docs, n_labels), bool)
        return cls(pos=pos, neg=neg)


@dataclass(frozen=True)
class PseudoSeed:
    """Certified, revocable ALT facts used to initialise one fresh Chase.

    A ``PseudoSeed`` is intentionally not a :class:`GammaSeed`: its facts were
    proposed by the role-independent auxiliary annotator and admitted by an
    adaptive ``val_ctrl`` gate (then rebuilt after terminal ``val_cert`` family
    testing), not verified by a human. Paper Chase may consume both typed
    inputs, but only Gamma populates the immutable ``validated`` source mask.
    The outer ALT controller replaces this object between fresh Chase runs,
    which makes pseudo-label revocation explicit and deterministic.
    """

    pos: np.ndarray
    neg: np.ndarray
    provenance: Mapping[Tuple[int, int, str], Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        pos = np.array(self.pos, dtype=bool, copy=True)
        neg = np.array(self.neg, dtype=bool, copy=True)
        if pos.ndim != 2 or neg.ndim != 2:
            raise ValueError("PseudoSeed pos/neg must be two-dimensional")
        if pos.shape != neg.shape:
            raise ValueError(
                f"PseudoSeed pos/neg shapes differ: {pos.shape} != {neg.shape}"
            )
        if np.any(pos & neg):
            first = tuple(int(v) for v in np.argwhere(pos & neg)[0])
            raise ValueError(
                "PseudoSeed is inconsistent: the same fact is positive and "
                f"negative at index {first}"
            )
        clean = {}
        for key, metadata in dict(self.provenance or {}).items():
            if (
                not isinstance(key, tuple)
                or len(key) != 3
                or key[2] not in {"pos", "neg"}
            ):
                raise ValueError(
                    "PseudoSeed provenance keys must be (doc, label, 'pos'|'neg')"
                )
            doc_idx, label_idx, polarity = key
            if not (0 <= int(doc_idx) < pos.shape[0]
                    and 0 <= int(label_idx) < pos.shape[1]):
                raise ValueError("PseudoSeed provenance index is out of bounds")
            active = pos if polarity == "pos" else neg
            if not active[int(doc_idx), int(label_idx)]:
                raise ValueError(
                    "PseudoSeed provenance refers to an inactive pseudo fact"
                )
            clean[(int(doc_idx), int(label_idx), polarity)] = MappingProxyType(
                dict(metadata)
            )
        pos.setflags(write=False)
        neg.setflags(write=False)
        object.__setattr__(self, "pos", pos)
        object.__setattr__(self, "neg", neg)
        object.__setattr__(self, "provenance", MappingProxyType(clean))

    @property
    def shape(self) -> Tuple[int, int]:
        return self.pos.shape

    def validate_shape(self, n_docs: int, n_labels: int) -> None:
        expected = (n_docs, n_labels)
        if self.shape != expected:
            raise ValueError(
                f"PseudoSeed shape {self.shape} != expected {expected}"
            )

    @classmethod
    def empty(cls, n_docs: int, n_labels: int) -> "PseudoSeed":
        if n_docs < 0 or n_labels < 0:
            raise ValueError("n_docs and n_labels must be non-negative")
        shape = (n_docs, n_labels)
        return cls(np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool))
