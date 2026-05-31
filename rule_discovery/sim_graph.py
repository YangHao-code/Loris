"""rule_discovery/sim_graph.py — MIGRATION SHIM.

Implementation moved to :mod:`loris.rules.sim_graph` (Phase 3).
"""

from __future__ import annotations

from loris.rules.sim_graph import (  # noqa: F401
    MAX_AVG_DEGREE,
    compute_embeddings,
    auto_threshold_bins,
    build_sim_graph,
    precompute_neighbor_label_masks,
    precompute_neighbor_label_counts,
    save_sim_graphs,
    load_sim_graphs,
)
