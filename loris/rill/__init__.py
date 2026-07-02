"""LORIS RILL — Rule-Informed active Learning Loop.

Optional active-learning stage that estimates rule influence, applies a
three-tier trust check, and queries an oracle for uncertain labels. The
default :class:`GroundTruthOracle` answers from held-out labels (no network).

Migrated from ``chase_inference/{rill,oracle}.py`` (Phase 4). Legacy modules
re-export as shims.
"""

from __future__ import annotations

from loris.rill.oracle import OracleBase, GroundTruthOracle, LLMOracle
from loris.rill.rill import (
    InfluenceEstimator,
    TrustChecker,
    RILLResult,
    RILLController,
)

__all__ = [
    "OracleBase",
    "GroundTruthOracle",
    "LLMOracle",
    "InfluenceEstimator",
    "TrustChecker",
    "RILLResult",
    "RILLController",
]
