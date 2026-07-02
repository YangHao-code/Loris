"""LORIS pattern abstraction subpackage.

Discovers textual predicates from a corpus (the paper's pattern-abstraction
step): TF-IDF + KMeans clustering -> n-gram / AutoRegex candidate generation ->
coverage/discriminativeness screening, producing Match/Freq/Cooccur/Before
predicates and a :class:`PatternStore` for applying them to new documents.

Migrated from ``pattern_extraction/`` (Phase 2). The legacy MetaPAD pattern
mode (external Java/C++ tooling, absent from the methodology) has been removed.
The legacy modules re-export these names as shims during migration.
"""

from __future__ import annotations

from loris.patterns.auto_regex import AutoRegexExtractor
from loris.patterns.preprocessor import Preprocessor
from loris.patterns.pattern_store import PatternStore
from loris.patterns.pattern_abstractor import PatternAbstractor, load_regex_config
from loris.patterns.extract import extract_patterns

__all__ = [
    "AutoRegexExtractor",
    "Preprocessor",
    "PatternStore",
    "PatternAbstractor",
    "load_regex_config",
    "extract_patterns",
]
