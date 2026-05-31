"""LORIS rule discovery and the RDL rule representation.

Contents:
  * ``rdl``         — RDL (rule X -> p0) and RDLSet container, plus _is_redundant
  * ``discovery``   — ChaseRuleLearner: Track1 Bayesian-optimisation rule search
                      and Track2 propagation rule search over the Chase
  * ``sim_graph``   — sentence-embedding similarity graph (SimPredicate support)
  * ``group_propagation`` — virtual-attribute group rules (GroupPredicate)
  * ``virtual_attributes`` — KMeans cluster attributes for group rules

Migrated from ``rule_discovery/`` (Phase 3). The legacy single-track
``RuleLearner`` and the experimental ``HybridRuleLearner`` are intentionally
NOT migrated — they are dead relative to the chase main line and removed with
their pipelines in Phase 6.
"""

from __future__ import annotations

from loris.rules.rdl import RDL, RDLSet, _is_redundant
from loris.rules.discovery import ChaseRuleLearner

__all__ = [
    "RDL",
    "RDLSet",
    "ChaseRuleLearner",
    "_is_redundant",
]
