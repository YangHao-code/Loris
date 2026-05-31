"""Deterministic entry wrapper for golden-master runs.

Runs the legacy chase pipeline but makes it fully reproducible so the captured
"golden" artifacts can be verified bit-for-bit after each refactor step:

  * Model pool is restricted to the deterministic CPU classifiers (tfidf_*),
    dropping neural / pretrained models (textcnn, bilstm, encoder_mlp,
    lora_slm). These are GPU-nondeterministic and score ~0 F1 on the tiny
    golden subset, so they never win baseline selection anyway. (This is for
    DETERMINISM, not memory — the host now has ample RAM + a 24 GB GPU.)
  * SentenceTransformer embeddings are pinned to CPU. The golden run is
    captured once and re-checked many times; GPU float ops are not guaranteed
    bit-identical across runs, whereas CPU MiniLM inference is. Encoding the
    small subset on CPU is cheap.
  * Seeds (random / numpy / torch) and thread counts are fixed.

This wrapper modifies NO legacy source — it monkeypatches names in the
pipeline / sentence_transformers namespaces, then delegates to ``main()``.
"""

from __future__ import annotations

import os
import random

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np  # noqa: E402

try:
    import torch  # noqa: E402
    torch.manual_seed(42)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    # Compat shim: this torch build (1.11) lacks torch.backends.mps, which
    # newer sentence_transformers probes unconditionally in get_device_name().
    if not hasattr(torch.backends, "mps"):
        import types as _types
        torch.backends.mps = _types.SimpleNamespace(
            is_available=lambda: False, is_built=lambda: False)
except Exception:
    torch = None

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# CPU-pinned SentenceTransformer proxy (determinism for golden verification).
# Subclasses the real class but forces device="cpu" regardless of what the
# caller passes, so embeddings feeding KMeans clustering + the SimPredicate
# graph are reproducible across capture and check runs.
# ---------------------------------------------------------------------------
import sentence_transformers as _st  # noqa: E402

_RealST = _st.SentenceTransformer


class _CPUSentenceTransformer(_RealST):
    def __init__(self, *args, **kwargs):
        kwargs["device"] = "cpu"
        super().__init__(*args, **kwargs)


_st.SentenceTransformer = _CPUSentenceTransformer

import run_loris_chase_pipeline as _cp  # noqa: E402

_NEURAL_KEYS = ("textcnn", "bilstm", "encoder_mlp", "lora_slm")
_orig_init_models = _cp.init_models


def _deterministic_init_models(*args, **kwargs):
    pool = _orig_init_models(*args, **kwargs)
    for k in _NEURAL_KEYS:
        pool.pop(k, None)
    return pool


_cp.init_models = _deterministic_init_models


if __name__ == "__main__":
    _cp.main()
