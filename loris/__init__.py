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


def _install_torch_mps_shim() -> None:
    """Stub ``torch.backends.mps`` when the installed torch lacks it.

    This torch build (1.11) has no ``torch.backends.mps`` submodule, but the
    installed ``sentence_transformers.util.get_device_name()`` calls
    ``torch.backends.mps.is_available()`` whenever CUDA is unavailable (e.g. a
    CPU-only worker), raising AttributeError. mps is Apple-Silicon-only and
    irrelevant on this Linux host. Importing :mod:`loris` (in the main process
    and in every joblib/loky worker that imports a loris module) installs a
    stub so the probe returns False instead of crashing. Previously this lived
    only in the golden harness's sitecustomize; making it part of the package
    lets ``python -m loris`` run standalone without that scaffolding.
    """
    try:  # pragma: no cover - environment-dependent defensive shim
        import types
        import torch
        if not hasattr(torch.backends, "mps"):
            torch.backends.mps = types.SimpleNamespace(
                is_available=lambda: False,
                is_built=lambda: False,
            )
    except Exception:
        pass


_install_torch_mps_shim()
