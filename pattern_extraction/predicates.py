"""pattern_extraction/predicates.py — MIGRATION SHIM.

The predicate hierarchy now lives in :mod:`loris.predicates` (Phase 1 of the
golden-master refactor). This module re-exports every name from the single
authoritative implementation (``loris.predicates._core``) so that legacy
imports — ``from pattern_extraction.predicates import ...`` — keep working
unchanged while the rest of the codebase is migrated.

Importing from ``_core`` (not a re-implementation) guarantees there is exactly
ONE module object backing both import paths, so module-level state such as
``_ML_MODEL_CACHE`` is shared: a model registered via ``register_ml_model`` is
visible regardless of which path imported the cache.

Behavioral note: the legacy ``LabelPredicate`` op ``"minus"`` (which mutated
``doc.lbl`` as a side effect) has been removed — label mutation is a rule
consequence, not a predicate. See ``loris/predicates/__init__.py``.

This shim is deleted once all importers point at ``loris`` directly.
"""

from __future__ import annotations

from loris.predicates import _core as _core

# Public API ---------------------------------------------------------------
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

# Internal symbols still imported by legacy modules -------------------------
from loris.predicates._core import (  # noqa: F401
    _Pattern,
    _as_pattern,
    _get_embedding_model,
    _get_embedding,
    _cosine_similarity,
    _split_sentences,
    _get_ml_model,
    _COMPARE_OPS,
    _FLAG_MAP,
    _VALID_LABEL_OPS,
    _PATTERN_CACHE,
    _EMBEDDING_CACHE,
)

# Shared mutable module state: bind the SAME dict object as _core, so
# register_ml_model() writes are observed through this name too.
_ML_MODEL_CACHE = _core._ML_MODEL_CACHE
