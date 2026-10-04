"""Canonical sentence-transformer (MiniLM) loading for the whole package.

Before 2026-07-09 the SentenceTransformer construction was re-implemented at
seven sites in ``loris/`` with four independent ``.npy`` cache layers. This
module is the single loader:

* :func:`get_sentence_encoder` — process-level singleton per
  ``(model_name, device)``; short model names (``all-MiniLM-L6-v2``) are
  canonicalized to the full hub id so they share the singleton.
* :func:`encode_texts` — encode, optionally writing a new ``.npy`` export.
  Existing unbound caches are rejected: row count cannot certify their text,
  model or normalization settings.

Consolidation is deliberately construction-only: every call site keeps its
own ``encode()`` arguments (batch size, normalization, progress bar) verbatim,
because those change numerics — normalization changes KMeans geometry, and on
GPU the batch composition changes padding/reduction order. Sites that must
stay on CPU (predicate-level embedding inside joblib worker processes — see
``loris/predicates/_core.py``) pass ``device="cpu"`` explicitly.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_MODELS: Dict[Tuple[str, Optional[str]], object] = {}
_LOCK = threading.Lock()


def canonical_model_name(model_name: str) -> str:
    """Map bare model names (``all-MiniLM-L6-v2``) to their full hub id."""
    if "/" not in model_name:
        return f"sentence-transformers/{model_name}"
    return model_name


def get_sentence_encoder(model_name: str = DEFAULT_SENTENCE_MODEL,
                         device: Optional[str] = None):
    """Return a process-level SentenceTransformer singleton.

    ``device=None`` keeps sentence-transformers' auto selection (CUDA when
    available). Pass ``device="cpu"`` from code that fans out across joblib
    worker processes (see the rationale in ``loris/predicates/_core.py``).
    """
    key = (canonical_model_name(model_name), device)
    with _LOCK:
        model = _MODELS.get(key)
        if model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformer %s (device=%s)",
                        key[0], device or "auto")
            model = SentenceTransformer(key[0], device=device)
            _MODELS[key] = model
    return model


def encode_texts(
    texts: Iterable[str],
    *,
    normalize: bool,
    batch_size: int = 256,
    model_name: str = DEFAULT_SENTENCE_MODEL,
    device: Optional[str] = None,
    cache_path: Optional[str] = None,
    show_progress_bar: bool = False,
    dtype=np.float32,
) -> np.ndarray:
    """Encode ``texts``; optionally export to a previously nonexistent .npy path.

    ``normalize`` is intentionally required (no default): whether embeddings
    are L2-normalized is the one knob that silently changes downstream
    numerics (cosine vs. Euclidean KMeans geometry), so every caller states it.
    """
    texts = list(texts)
    if cache_path:
        cache_path = os.fspath(cache_path)
        if not cache_path.endswith('.npy'):
            cache_path += '.npy'
    if cache_path and os.path.exists(cache_path):
        raise ValueError(
            "Existing embedding caches have no text/model/normalization binding; "
            "reuse is disabled. Omit cache_path or choose a new export path."
        )

    model = get_sentence_encoder(model_name, device)
    logger.info("Computing embeddings for %d texts with %s ...",
                len(texts), canonical_model_name(model_name))
    emb = model.encode(texts, batch_size=batch_size,
                       show_progress_bar=show_progress_bar,
                       normalize_embeddings=normalize,
                       convert_to_numpy=True)
    emb = np.asarray(emb, dtype=dtype)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, 'xb') as stream:
            np.save(stream, emb)
        logger.info("Saved embeddings cache to %s", cache_path)
    return emb
