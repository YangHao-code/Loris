"""Auto-imported by Python's site module in EVERY interpreter start.

Placed on PYTHONPATH by the golden harness (run_golden.py) so the compat shim
below also applies inside joblib/loky worker processes, which spawn fresh
interpreters and therefore do NOT inherit monkeypatches made in the parent.

Shim: this torch build (1.11) has no ``torch.backends.mps`` submodule, but the
installed sentence_transformers calls ``torch.backends.mps.is_available()`` in
get_device_name(). mps is Apple-Silicon-only and irrelevant on this Linux host.
We add a stub so the probe returns False instead of raising AttributeError.

This file is golden-verification scaffolding only; it modifies no production
source. (The same fix will be applied properly in loris/ during Phase 1.)
"""

try:  # pragma: no cover - defensive
    import torch
    import types

    if not hasattr(torch.backends, "mps"):
        torch.backends.mps = types.SimpleNamespace(
            is_available=lambda: False,
            is_built=lambda: False,
        )
except Exception:
    pass
