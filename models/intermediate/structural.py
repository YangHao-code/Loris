"""models/intermediate/structural.py — MIGRATION SHIM (Phase 6)."""
from __future__ import annotations
import loris.models.intermediate.structural as _m
for _n in dir(_m):
    if not _n.startswith("__"):
        globals()[_n] = getattr(_m, _n)
