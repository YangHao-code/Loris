"""model_selection/dynamic_router.py — MIGRATION SHIM (Phase 6). See loris.selection.dynamic_router."""
from __future__ import annotations
import loris.selection.dynamic_router as _m
for _n in dir(_m):
    if not _n.startswith("__"):
        globals()[_n] = getattr(_m, _n)
