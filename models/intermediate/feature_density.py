"""models/intermediate/feature_density.py — MIGRATION SHIM (Phase 6)."""
from __future__ import annotations
import loris.models.intermediate.feature_density as _m
for _n in dir(_m):
    if not _n.startswith("__"):
        globals()[_n] = getattr(_m, _n)
