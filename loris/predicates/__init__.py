"""LORIS predicate hierarchy.

The predicate types used by the RDL implementation:

  Textual: MatchPredicate, FreqPredicate, CooccurPredicate, BeforePredicate
  ML:      MLPredicate, MLThresholdPredicate, MLScoreBandPredicate
  Label:   LabelPredicate            (positive membership only)
  Graph:   SimPredicate, TopKPredicate, GroupPredicate

plus the serialization helpers ``predicate_to_dict`` / ``predicate_from_dict``
and ``register_ml_model``.

The former side-effecting ``LabelPredicate`` op ``"minus"`` has been removed:
label mutation is a rule consequence, not a predicate. Literal-set comparison
operations are also rejected because this API has no complete labeling state.

RDL label-set comparisons use explicit ``LabelCompare`` relation facts in the
scoped Chase. Partial ``Document.lbl`` sets are not compared as complete sets;
in particular two unknown documents do not establish an equality relation.
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
    MLScoreBandPredicate,
    LabelPredicate,
    SimPredicate,
    TopKPredicate,
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
    "MLScoreBandPredicate",
    "LabelPredicate",
    "SimPredicate",
    "TopKPredicate",
    "GroupPredicate",
    "predicate_to_dict",
    "predicate_from_dict",
    "register_ml_model",
    "flags_from_names",
]
