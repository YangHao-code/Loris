"""Shared pytest fixtures for LORIS tests.

The predicate / document suites are written to run against EITHER the legacy
``pattern_extraction`` package OR the new ``loris`` package, selected by the
``LORIS_TEST_TARGET`` env var:

    LORIS_TEST_TARGET=legacy  -> pattern_extraction.*   (default, pre-migration)
    LORIS_TEST_TARGET=loris   -> loris.*                (post-migration)

This lets the same characterization tests act as the bit-for-bit equivalence
gate: they must pass identically against both targets while a module is being
migrated and shimmed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _target() -> str:
    return os.environ.get("LORIS_TEST_TARGET", "legacy").lower()


@pytest.fixture(scope="session")
def predicates_mod():
    """The predicates module under test (legacy or loris)."""
    if _target() == "loris":
        import loris.predicates as mod
    else:
        import pattern_extraction.predicates as mod
    return mod


@pytest.fixture(scope="session")
def document_cls():
    """The Document class under test (legacy or loris)."""
    if _target() == "loris":
        from loris.document import Document
    else:
        from pattern_extraction.document import Document
    return Document
