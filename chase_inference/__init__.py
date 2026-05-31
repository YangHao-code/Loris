"""
chase_inference
---------------
Multi-Label Chase algorithm and RILL active-learning loop for the LORIS framework.

Implements the paper's label-equivalence-class-based chase with
incremental queue optimization, conflict detection, and transitivity (§6.1),
plus the RILL outer loop for oracle-assisted labeling (§6.2).
"""

from chase_inference.multi_chase import (
    BulkLBL,
    ChaseResult,
    MultiChase,
)
from chase_inference.rdg import RuleDependencyGraph
from chase_inference.oracle import (
    GroundTruthOracle,
    LLMOracle,
    OracleBase,
)
from chase_inference.rill import (
    InfluenceEstimator,
    RILLController,
    RILLResult,
    TrustChecker,
)

__all__ = [
    "BulkLBL",
    "ChaseResult",
    "GroundTruthOracle",
    "InfluenceEstimator",
    "LLMOracle",
    "MultiChase",
    "OracleBase",
    "RILLController",
    "RILLResult",
    "RuleDependencyGraph",
    "TrustChecker",
]
