"""models/base.py — MIGRATION SHIM (Phase 6). See loris.models.base."""
from __future__ import annotations
import loris.models.base as _m
from loris.models.base import *  # noqa: F401,F403
_g = globals()
for _n in dir(_m):
    if not _n.startswith("__"):
        _g[_n] = getattr(_m, _n)
