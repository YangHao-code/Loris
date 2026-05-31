"""models/utils.py — MIGRATION SHIM (Phase 6). See loris.models.utils."""
from __future__ import annotations
import loris.models.utils as _m
from loris.models.utils import *  # noqa: F401,F403
_g = globals()
for _n in dir(_m):
    if not _n.startswith("__"):
        _g[_n] = getattr(_m, _n)
