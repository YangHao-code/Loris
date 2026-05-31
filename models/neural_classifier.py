"""models/neural_classifier.py — MIGRATION SHIM (Phase 6). See loris.models.neural_classifier."""
from __future__ import annotations
import loris.models.neural_classifier as _m
from loris.models.neural_classifier import *  # noqa: F401,F403
_g = globals()
for _n in dir(_m):
    if not _n.startswith("__"):
        _g[_n] = getattr(_m, _n)
