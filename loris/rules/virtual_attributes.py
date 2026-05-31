"""
Virtual attribute computation for group-based propagation (Track 2 v7).

Generates discrete group IDs from:
  1. K-Means clustering on sentence embeddings (fit on train, predict on target)
  2. Intermediate model prediction patterns (argmax top-1, top-2 signatures)

All attributes are (n_docs,) int32 arrays where each value is a group ID.
Group ID = -1 means "excluded from propagation" (degenerate group).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_cluster_attributes(
    train_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    k_list: List[int] = [50, 100, 200, 500],
    kmeans_models: Optional[Dict[int, object]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[int, object]]:
    """K-Means fit on train, predict on target.

    Parameters
    ----------
    train_embeddings : (n_train, dim) float32 — used for fitting only
    target_embeddings : (n_target, dim) float32 — predictions computed here
    k_list : cluster counts to try
    kmeans_models : if provided, skip fitting and reuse these models

    Returns
    -------
    (attrs_dict, models) where attrs_dict maps "cluster_{k}" -> (n_target,) int32
    """
    from sklearn.cluster import MiniBatchKMeans

    n_train = len(train_embeddings)
    if kmeans_models is None:
        kmeans_models = {}
        for k in k_list:
            # KMeans requires n_clusters <= n_samples. On small corpora a
            # configured k can exceed the training-set size; clamp it so the
            # fit never raises (B0). On full-size data (k < n_train) this is a
            # no-op, so cluster assignments are unchanged.
            k_eff = min(k, n_train)
            if k_eff < k:
                logger.warning(
                    "  KMeans k=%d > n_train=%d; clamping to %d", k, n_train, k_eff)
            km = MiniBatchKMeans(n_clusters=k_eff, random_state=42, batch_size=1024)
            km.fit(train_embeddings)
            kmeans_models[k] = km
            logger.info("  KMeans k=%d fit on %d train docs", k_eff, n_train)

    attrs: Dict[str, np.ndarray] = {}
    for k, km in kmeans_models.items():
        labels = km.predict(target_embeddings).astype(np.int32)
        attrs[f"cluster_{k}"] = labels
        logger.info("  cluster_%d: %d unique groups on %d target docs",
                    k, len(np.unique(labels)), len(target_embeddings))

    return attrs, kmeans_models


def _cantor_pair(a: int, b: int) -> int:
    """Cantor pairing function for two non-negative integers."""
    return (a + b) * (a + b + 1) // 2 + b


def compute_prediction_pattern_attributes(
    ml_proba_cache: Dict[str, np.ndarray],
    label_names: List[str],
    top_k_list: List[int] = [1, 2],
) -> Dict[str, np.ndarray]:
    """Intermediate model predictions → group IDs.

    For each model:
      - top1: group_id = argmax label index (0..n_labels-1)
      - top2: group_id = cantor_pair(sorted top-2 indices)

    Parameters
    ----------
    ml_proba_cache : {model_name: (n_docs, n_labels) float proba array}
    label_names : label name list (for logging)
    top_k_list : which top-k patterns to compute

    Returns
    -------
    {attr_name: (n_docs,) int32}
    """
    attrs: Dict[str, np.ndarray] = {}
    n_labels = len(label_names)

    for model_name, proba in ml_proba_cache.items():
        if proba.ndim != 2 or proba.shape[1] != n_labels:
            logger.warning("  Skipping %s: shape %s doesn't match %d labels",
                          model_name, proba.shape, n_labels)
            continue

        short_name = model_name.replace("loris_", "")

        if 1 in top_k_list:
            top1 = np.argmax(proba, axis=1).astype(np.int32)
            attr_name = f"{short_name}_top1"
            attrs[attr_name] = top1
            n_groups = len(np.unique(top1))
            logger.info("  %s: %d unique groups", attr_name, n_groups)

        if 2 in top_k_list:
            top2_idx = np.argsort(proba, axis=1)[:, -2:]
            n_docs = len(proba)
            top2_ids = np.empty(n_docs, dtype=np.int32)
            for i in range(n_docs):
                a, b = sorted([int(top2_idx[i, 0]), int(top2_idx[i, 1])])
                top2_ids[i] = _cantor_pair(a, b)
            attr_name = f"{short_name}_top2"
            attrs[attr_name] = top2_ids
            n_groups = len(np.unique(top2_ids))
            logger.info("  %s: %d unique groups", attr_name, n_groups)

    return attrs


def compute_all_virtual_attributes(
    train_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    ml_proba_cache: Dict[str, np.ndarray],
    label_names: List[str],
    k_list: List[int] = [50, 100, 200, 500],
    top_k_list: List[int] = [1, 2],
    kmeans_models: Optional[Dict[int, object]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[int, object]]:
    """Compute all virtual attributes (clusters + prediction patterns).

    Returns
    -------
    (all_attrs, kmeans_models) — all_attrs maps attr_name -> (n_target,) int32
    """
    logger.info("Computing virtual attributes: %d cluster sizes, %d top-k patterns, "
                "%d intermediate models", len(k_list), len(top_k_list), len(ml_proba_cache))

    cluster_attrs, kmeans_models = compute_cluster_attributes(
        train_embeddings, target_embeddings, k_list, kmeans_models
    )

    pred_attrs = compute_prediction_pattern_attributes(
        ml_proba_cache, label_names, top_k_list
    )

    all_attrs = {**cluster_attrs, **pred_attrs}
    logger.info("Total virtual attributes: %d", len(all_attrs))
    return all_attrs, kmeans_models


def filter_degenerate_groups(
    virtual_attrs: Dict[str, np.ndarray],
    min_group_size: int = 3,
    max_group_fraction: float = 0.33,
) -> Dict[str, np.ndarray]:
    """Mark degenerate groups as -1 (excluded from propagation).

    A group is degenerate if:
      - size < min_group_size (too small for meaningful propagation)
      - size > max_group_fraction * n_docs (too large, no discriminative power)

    Parameters
    ----------
    virtual_attrs : {attr_name: (n_docs,) int32}
    min_group_size : minimum docs per group
    max_group_fraction : maximum fraction of total docs per group

    Returns
    -------
    Filtered copy of virtual_attrs (degenerate group members set to -1)
    """
    filtered: Dict[str, np.ndarray] = {}
    for attr_name, group_ids in virtual_attrs.items():
        n_docs = len(group_ids)
        max_size = int(max_group_fraction * n_docs)
        new_ids = group_ids.copy()

        unique_ids, counts = np.unique(group_ids[group_ids >= 0], return_counts=True)
        n_filtered = 0
        for gid, cnt in zip(unique_ids, counts):
            if cnt < min_group_size or cnt > max_size:
                new_ids[new_ids == gid] = -1
                n_filtered += 1

        n_remaining = len(np.unique(new_ids[new_ids >= 0]))
        if n_filtered > 0:
            logger.info("  %s: filtered %d degenerate groups, %d remaining",
                       attr_name, n_filtered, n_remaining)
        filtered[attr_name] = new_ids

    return filtered
