"""LORIS pipeline — end-to-end chase orchestration.

  * ``shared``       — dataset-agnostic steps (model init/training, pattern
                       abstraction, dynamic model selection, predicate
                       selection, results reporting)
  * ``steps``        — chase rule-discovery steps (Track1/Track2 search)
  * ``orchestrator`` — CLI arg parsing and ``main`` (the full run)

Migrated from ``run_loris_chase_pipeline.py`` + ``run_loris_multi_pipeline.py``
SECTION 2 (Phase 6). Run via ``python -m loris`` (see loris/__main__.py).
"""

from __future__ import annotations

from loris.pipeline.orchestrator import main, parse_args

__all__ = ["main", "parse_args"]
