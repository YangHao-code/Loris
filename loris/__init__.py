"""LORIS — Logic-Reinforced Inference for Semantic labeling.

Rule-based multi-label document classification framework. Pipeline:
ML baseline prediction -> logic-rule correction (Chase inference) ->
optional RILL active learning.

This package is the single, clean home for the LORIS system. Modules are
migrated here incrementally from the legacy top-level packages
(``pattern_extraction``, ``rule_discovery``, ``chase_inference``, ...) using a
golden-master (bit-for-bit) verification process; see
``docs/06_methodology.tex`` for the authoritative method specification.
"""

from __future__ import annotations

__version__ = "0.2.0"
