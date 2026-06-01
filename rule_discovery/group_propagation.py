"""rule_discovery/group_propagation.py — MIGRATION SHIM.

Implementation moved to :mod:`loris.rules.group_propagation` (Phase 3).
"""

from __future__ import annotations

from loris.rules.group_propagation import (  # noqa: F401
    MockTrial,
    compute_group_fire_mask,
    evaluate_group_rule,
    discover_group_rules,
    discover_equal_rules,
    simulate_cross_attr_cascade,
)
