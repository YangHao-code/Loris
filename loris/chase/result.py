"""Complete signed chase results."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class ChaseResult:
    """Result of a multi-label chase run.

    Attributes
    ----------
    status : str
        ``"fixpoint"`` (normal convergence), ``"conflict"`` (halted on
        contradiction), or ``"max_rounds"`` / ``"timeout"``.
    predictions : np.ndarray
        (n_docs, n_labels) float32 multi-hot prediction matrix.
    conflicts : List[Tuple[int, str]]
        (doc_idx, label_name) pairs that were in conflict.
    n_rounds : int
        Number of chase rounds executed.
    provenance : Dict[Tuple[int, str], List[int]]
        Optional mapping ``(doc_idx, label_name) → [rule_indices]``
        that derived that label.  Empty if ``track_provenance=False``.
    derivation_audit : ChaseDepthAudit or None
        Optional scheduler-independent shortest-proof audit, populated only by
        :meth:`MultiChase.audit_derivations`.
    """

    status: str
    predictions: np.ndarray
    conflicts: List[Tuple[int, str]]
    n_rounds: int
    provenance: Dict[Tuple[int, str], List[int]] = field(default_factory=dict)
    derivation_audit: Optional["ChaseDepthAudit"] = None
    positive_facts: Optional[np.ndarray] = None
    negative_facts: Optional[np.ndarray] = None

    @property
    def scheduler_rounds(self) -> int:
        """Explicit name for the procedural round count ``n_rounds``."""
        return int(self.n_rounds)

    @property
    def logical_depth(self) -> Optional[int]:
        """Maximum shortest proof height, or ``None`` before depth audit."""
        if self.derivation_audit is None:
            return None
        return int(self.derivation_audit.logical_depth)

    @property
    def propagation_hops(self) -> Optional[int]:
        """Maximum shortest cross-document hop count, if audited."""
        if self.derivation_audit is None:
            return None
        return int(self.derivation_audit.propagation_hops)

