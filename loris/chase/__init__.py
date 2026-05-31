"""LORIS Chase inference.

The Chase engine applies a discovered :class:`~loris.rules.rdl.RDLSet` to a
dataset until fixpoint, maintaining the dual ``pos``/``neg`` label sets with
sub/sup transitivity (the paper's BulkLBL semantics) and resolving conflicts
(positive_wins). ``RuleDependencyGraph`` orders rule application.

Migrated from ``chase_inference/`` (Phase 4). Legacy modules re-export as shims.
"""

from __future__ import annotations

from loris.chase.multi_chase import BulkLBL, ChaseResult, MultiChase
from loris.chase.rdg import RuleDependencyGraph

__all__ = [
    "BulkLBL",
    "ChaseResult",
    "MultiChase",
    "RuleDependencyGraph",
]
