"""LORIS predicate hierarchy.

The 9 predicate types from the paper's RDL specification:

  Textual: MatchPredicate, FreqPredicate, CooccurPredicate, BeforePredicate
  ML:      MLPredicate, MLThresholdPredicate
  Label:   LabelPredicate            (pure reads: contains/eq/subset/strict_subset)
  Graph:   SimPredicate, GroupPredicate

plus the serialization helpers ``predicate_to_dict`` / ``predicate_from_dict``
and ``register_ml_model``.

Migrated from ``pattern_extraction/predicates.py`` (Phase 1). The single
behavioral change vs. legacy: the side-effecting ``LabelPredicate`` op
``"minus"`` (which mutated ``doc.lbl``) has been removed — label mutation is a
rule consequence, not a predicate. The legacy module re-exports these names as
a shim during migration.

Implementation currently lives in ``_core`` (a faithful copy of the legacy
module) and will be split into ``base``/``textual``/``ml``/``label``/``graph``/
``serialize`` submodules once equivalence is locked by the test suite.
"""

from __future__ import annotations

from loris.predicates._core import (  # noqa: F401
    VALID_OPS,
    Predicate,
    TextualPredicate,
    MatchPredicate,
    FreqPredicate,
    CooccurPredicate,
    BeforePredicate,
    MLPredicate,
    MLThresholdPredicate,
    LabelPredicate,
    SimPredicate,
    GroupPredicate,
    predicate_to_dict,
    predicate_from_dict,
    register_ml_model,
    flags_from_names,
)

__all__ = [
    "VALID_OPS",
    "Predicate",
    "TextualPredicate",
    "MatchPredicate",
    "FreqPredicate",
    "CooccurPredicate",
    "BeforePredicate",
    "MLPredicate",
    "MLThresholdPredicate",
    "LabelPredicate",
    "SimPredicate",
    "GroupPredicate",
    "predicate_to_dict",
    "predicate_from_dict",
    "register_ml_model",
    "flags_from_names",
]
