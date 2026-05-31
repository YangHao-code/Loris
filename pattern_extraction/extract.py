"""pattern_extraction/extract.py — MIGRATION SHIM.

Implementation moved to :mod:`loris.patterns.extract` (Phase 2).
"""

from __future__ import annotations

from loris.patterns.extract import extract_patterns, main  # noqa: F401

if __name__ == "__main__":
    main()
