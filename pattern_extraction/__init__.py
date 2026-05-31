"""
pattern_extraction/__init__.py
-------------------------------
Pattern abstraction pipeline for the LORIS (Logic-Reinforced Inference for
Semantic labeling) framework.

Implements the three-step strategy described in the LORIS paper:
  1. Clustering        — embed documents then group with k-means.
  2. Candidate retrieval — extract TF-IDF, chunking, and regex anchors per
     cluster; form MatchPredicate, CooccurPredicate, BeforePredicate instances.
  3. Lightweight screening — filter by coverage and entropy discriminability.

Public API
----------
Core data types::

    Document               — text variable x with (lbl, mtd, ttl, cnt)
    TextualPredicate       — abstract base for all textual predicates
    MatchPredicate         — match(x.A, r)
    FreqPredicate          — freq(x.A, r) ⊕ η
    CooccurPredicate       — cooccur(x.A, r1, r2)
    BeforePredicate        — before(x.A, r1, r2)

Pipeline::

    PatternAbstractor      — three-step mining pipeline (fit + to_store)
    PatternStore           — predicate application (apply) and persistence
    load_regex_config      — load entity-regex patterns from YAML/JSON

Serialization helpers::

    predicate_to_dict      — TextualPredicate → JSON-safe dict
    predicate_from_dict    — dict → TextualPredicate

Convenience function::

    extract_patterns       — one-shot fit + store (+ optional save)

Quick start
-----------
::

    from pattern_extraction import PatternAbstractor, Document, PatternStore

    # Mine predicates from labelled corpus
    abstractor = PatternAbstractor(n_clusters=None)   # auto-select k
    abstractor.fit(train_texts, train_labels)
    store = abstractor.to_store()
    store.save("patterns.json")

    # Apply to new corpus (different session)
    store = PatternStore.load("patterns.json")
    matrix = store.apply(new_texts)   # shape: (n_docs, n_patterns), dtype bool

    # Inspect a single document
    doc = Document(cnt="The bank raised rates.", ttl="Finance")
    for pred in store.matching_predicates(doc):
        print(pred)
"""

from __future__ import annotations

# -- Core data structure ----------------------------------------------------
from pattern_extraction.document import Document

# -- Textual predicate hierarchy -------------------------------------------
from pattern_extraction.predicates import (
    BeforePredicate,
    CooccurPredicate,
    FreqPredicate,
    GroupPredicate,
    LabelPredicate,
    MatchPredicate,
    MLPredicate,
    MLThresholdPredicate,
    Predicate,
    SimPredicate,
    TextualPredicate,
    predicate_from_dict,
    predicate_to_dict,
    register_ml_model,
)

# -- Mining pipeline --------------------------------------------------------
from pattern_extraction.pattern_abstractor import (
    PatternAbstractor,
    load_regex_config,
)

# -- Auto-regex discovery ---------------------------------------------------
from pattern_extraction.auto_regex import AutoRegexExtractor

# -- Predicate store (apply + persistence) ----------------------------------
from pattern_extraction.pattern_store import PatternStore

# -- Preprocessor -----------------------------------------------------------
from pattern_extraction.preprocessor import Preprocessor

# -- Convenience function ---------------------------------------------------
from pattern_extraction.extract import extract_patterns

__all__ = [
    # data structure
    "Document",
    # predicate hierarchy
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
    # serialization helpers
    "predicate_to_dict",
    "predicate_from_dict",
    # ML model registration
    "register_ml_model",
    # pipeline
    "PatternAbstractor",
    "load_regex_config",
    # auto-regex
    "AutoRegexExtractor",
    # store
    "PatternStore",
    # preprocessor
    "Preprocessor",
    # convenience
    "extract_patterns",
]

__version__ = "0.1.0"
