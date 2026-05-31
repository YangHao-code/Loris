"""LORIS dynamic model selection.

Learns a per-document model-selection network (the paper's dynamic router) and
its final Top-K selector. Migrated from ``model_selection/`` + the 3 perturbed
optimizers from ``perturbations/`` (Phase 6).

Lazy ``__getattr__`` import keeps ``import loris.selection`` cheap and avoids a
circular import when the router module is executed directly.
"""

from __future__ import annotations

_PUBLIC = {"SelectionNetwork", "HybridLoss", "FinalSelector", "build_oracle_mask"}


def __getattr__(name):
    if name in _PUBLIC:
        from loris.selection.dynamic_router import (
            SelectionNetwork,
            HybridLoss,
            FinalSelector,
            build_oracle_mask,
        )
        return {
            "SelectionNetwork": SelectionNetwork,
            "HybridLoss": HybridLoss,
            "FinalSelector": FinalSelector,
            "build_oracle_mask": build_oracle_mask,
        }[name]
    raise AttributeError(f"module 'loris.selection' has no attribute {name!r}")


__all__ = ["SelectionNetwork", "HybridLoss", "FinalSelector", "build_oracle_mask"]
