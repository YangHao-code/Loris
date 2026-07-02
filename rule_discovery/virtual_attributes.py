"""rule_discovery/virtual_attributes.py — MIGRATION SHIM.

Implementation moved to :mod:`loris.rules.virtual_attributes` (Phase 3).
"""

from __future__ import annotations

from loris.rules.virtual_attributes import (  # noqa: F401
    compute_cluster_attributes,
    compute_all_virtual_attributes,
    compute_phrase_attributes,
    compute_llm_attributes,
    llm_doc_hash,
    filter_degenerate_groups,
)
