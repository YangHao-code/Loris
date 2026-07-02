"""models/intermediate/ensemble_vote.py — MIGRATION SHIM (Phase 6)."""
from __future__ import annotations
import loris.models.intermediate.ensemble_vote as _m
for _n in dir(_m):
    if not _n.startswith("__"):
        globals()[_n] = getattr(_m, _n)
