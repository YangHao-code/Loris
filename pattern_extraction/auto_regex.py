"""pattern_extraction/auto_regex.py — MIGRATION SHIM.

Implementation moved to :mod:`loris.patterns.auto_regex` (Phase 2). Re-exports
preserved for legacy imports; deleted once all importers point at ``loris``.
"""

from __future__ import annotations

from loris.patterns.auto_regex import AutoRegexExtractor  # noqa: F401
