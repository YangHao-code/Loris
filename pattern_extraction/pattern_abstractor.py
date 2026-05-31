"""pattern_extraction/pattern_abstractor.py — MIGRATION SHIM.

Implementation moved to :mod:`loris.patterns.pattern_abstractor` (Phase 2).
The legacy MetaPAD pattern mode and its ``metapad_*`` constructor parameters
have been removed (external Java/C++ tooling, absent from the methodology).
"""

from __future__ import annotations

from loris.patterns.pattern_abstractor import (  # noqa: F401
    PatternAbstractor,
    load_regex_config,
)
