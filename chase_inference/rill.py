"""chase_inference/rill.py — MIGRATION SHIM.

Implementation moved to :mod:`loris.rill.rill` (Phase 4).
"""

from __future__ import annotations

from loris.rill.rill import (  # noqa: F401
    InfluenceEstimator,
    TrustChecker,
    RILLResult,
    RILLController,
)
