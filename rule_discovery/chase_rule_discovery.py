"""rule_discovery/chase_rule_discovery.py — MIGRATION SHIM.

Implementation moved to :mod:`loris.rules.discovery` (Phase 3). The ``minus``
LabelPredicate candidate pool (already disabled and never populated) follows
the predicate change in :mod:`loris.predicates`.
"""

from __future__ import annotations

from loris.rules.discovery import (  # noqa: F401
    ChaseRuleLearner,
    _save_stage_logs,
    _extract_trial_stats,
    _vectorized_staged_predict,
    precompute_fire_masks,
    precompute_ml_proba,
    MLThresholdPredicate,
    _fast_per_label_f1,
    _error_driven_filter,
    _tree_seeded_rules,
    ERROR_AWARE_WEIGHTS,
)
