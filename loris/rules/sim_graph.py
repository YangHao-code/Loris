"""
Similarity graph construction and neighbour-label mask precomputation.

Core utilities for Chase-driven rule discovery (SimPredicate support):
  1. compute_embeddings  — sentence-transformer embeddings with .npy cache
  2. build_sim_graph     — threshold-based sparse adjacency matrices (csr_matrix)
  3. precompute_neighbor_label_masks — SpMV masks for BO evaluation
  4. save/load helpers

All heavy lifting uses scipy.sparse SpMV — 10K docs in microseconds.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# Default threshold bins — MiniLM cosine space is compressed,
# 0.70 pulls too many false neighbours.
DEFAULT_THRESHOLD_BINS: List[float] = [0.80, 0.85, 0.88, 0.92, 0.95]

# If avg_degree exceeds this, skip the threshold (snowball risk).
MAX_AVG_DEGREE = 50


# ---------------------------------------------------------------------------
# 1. Embeddings
# ---------------------------------------------------------------------------

def compute_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_path: Optional[str] = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode *texts* with a sentence-transformer; return (n, dim) float32.

    If *cache_path* is given and the .npy file exists, load from cache.
    Otherwise compute and save.
    """
    if cache_path and os.path.exists(cache_path):
        logger.info("Loading cached embeddings from %s", cache_path)
        emb = np.load(cache_path)
        if emb.shape[0] == len(texts):
            return emb.astype(np.float32)
        logger.warning("Cache size mismatch (%d vs %d), recomputing", emb.shape[0], len(texts))

    from sentence_transformers import SentenceTransformer
    logger.info("Computing embeddings for %d texts with %s ...", len(texts), model_name)
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, emb)
        logger.info("Saved embeddings cache to %s", cache_path)
    return emb


# ---------------------------------------------------------------------------
# 1b. Data-adaptive threshold selection
# ---------------------------------------------------------------------------

def auto_threshold_bins(
    embeddings: np.ndarray,
    n_sample: int = 2000,
    target_avg_degrees: Tuple[float, ...] = (5.0, 10.0, 20.0, 40.0),
    max_avg_degree: int = MAX_AVG_DEGREE,
    seed: int = 42,
    min_avg_degree: float = 0.0,
) -> List[float]:
    """Compute similarity thresholds that yield specific average degrees.

    Samples a subset of embeddings, computes pairwise cosine similarities,
    then finds thresholds corresponding to each target avg_degree via
    percentile lookup.  Thresholds producing avg_degree > *max_avg_degree*
    are dropped.

    *min_avg_degree* (>0) enforces a CONNECTIVITY FLOOR: if no surviving
    threshold reaches it, a denser threshold targeting that degree is added even
    if it exceeds *max_avg_degree*. This guarantees RILL has real propagation
    paths for human-seeded labels (snowball is bounded because RILL clamps the
    seeds). 0 (default) = no floor (legacy behaviour).

    Returns sorted thresholds (ascending — strictest first in value).
    """
    n = embeddings.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, min(n_sample, n), replace=False)
    sub = embeddings[idx]
    sims = np.clip(sub @ sub.T, -1.0, 1.0)
    np.fill_diagonal(sims, -1.0)  # exclude self
    flat = sims.ravel()
    flat = flat[flat > -1.0]

    thresholds: List[float] = []
    for deg in target_avg_degrees:
        # fraction of pairs that must exceed threshold = deg / n
        frac = deg / n
        if frac >= 1.0:
            continue
        # percentile = 1 - frac  (top frac of pairs)
        pctile = 100.0 * (1.0 - frac)
        if pctile < 0 or pctile > 100:
            continue
        t = float(np.percentile(flat, pctile))
        # Estimate actual avg_degree on full graph
        est_deg = float((flat >= t).mean()) * n
        if est_deg > max_avg_degree:
            logger.info("  auto_threshold: deg_target=%.0f → thresh=%.4f "
                        "(est_avg_deg=%.1f > %d, skipped)",
                        deg, t, est_deg, max_avg_degree)
            continue
        if est_deg < 1.0:
            logger.info("  auto_threshold: deg_target=%.0f → thresh=%.4f "
                        "(est_avg_deg=%.1f < 1, skipped)",
                        deg, t, est_deg)
            continue
        thresholds.append(round(t, 4))
        logger.info("  auto_threshold: deg_target=%.0f → thresh=%.4f "
                     "(est_avg_deg=%.1f)",
                     deg, t, est_deg)

    # Deduplicate and sort
    thresholds = sorted(set(thresholds))
    if not thresholds:
        logger.warning("auto_threshold_bins: no thresholds survived filtering! "
                        "Falling back to P99/P99.5/P99.9 percentiles.")
        for p in [99.0, 99.5, 99.9]:
            t = float(np.percentile(flat, p))
            thresholds.append(round(t, 4))
        thresholds = sorted(set(thresholds))

    # ── Connectivity floor: guarantee ≥1 threshold with avg_degree ≥ min_avg_degree ──
    if min_avg_degree and min_avg_degree > 0:
        best_deg = max((float((flat >= t).mean()) * n for t in thresholds), default=0.0)
        if best_deg < min_avg_degree:
            frac = min_avg_degree / n
            if frac < 1.0:
                t_floor = round(float(np.percentile(flat, 100.0 * (1.0 - frac))), 4)
                est = float((flat >= t_floor).mean()) * n
                thresholds = sorted(set(thresholds + [t_floor]))
                logger.warning("auto_threshold_bins: connectivity floor — best avg_deg "
                               "%.1f < %.1f; added thresh=%.4f (est_avg_deg=%.1f) for "
                               "RILL propagation", best_deg, min_avg_degree, t_floor, est)
            else:
                logger.warning("auto_threshold_bins: min_avg_degree %.1f >= n=%d; "
                               "cannot enforce floor", min_avg_degree, n)

    logger.info("auto_threshold_bins: %d thresholds selected: %s", len(thresholds), thresholds)
    return thresholds


# ---------------------------------------------------------------------------
# 2. Sparse adjacency graph
# ---------------------------------------------------------------------------

def build_sim_graph(
    embeddings: np.ndarray,
    threshold_bins: Optional[List[float]] = None,
    chunk_size: int = 1000,
    max_avg_degree: int = MAX_AVG_DEGREE,
) -> Dict[float, sp.csr_matrix]:
    """Build threshold-based sparse adjacency matrices.

    For each threshold θ, returns a ``(n, n)`` boolean csr_matrix where
    ``adj[i, j] = True`` iff ``cos(emb_i, emb_j) >= θ`` and ``i != j``.

    Thresholds whose average degree exceeds *max_avg_degree* are skipped
    with a warning (snowball risk).

    Parameters
    ----------
    embeddings : (n, dim) float32, **L2-normalised** (inner product = cosine).
    threshold_bins : list of floats, default ``[0.80, 0.85, 0.88, 0.92, 0.95]``.
    chunk_size : docs per chunk for blocked matmul (controls peak RAM).
    max_avg_degree : thresholds with avg degree above this are dropped.

    Returns
    -------
    dict mapping threshold → csr_matrix(n, n, dtype=bool)
    """
    if threshold_bins is None:
        threshold_bins = list(DEFAULT_THRESHOLD_BINS)

    n = embeddings.shape[0]
    logger.info("Building sim graph: %d docs, thresholds=%s", n, threshold_bins)

    # Accumulate COO entries per threshold
    rows: Dict[float, list] = {t: [] for t in threshold_bins}
    cols: Dict[float, list] = {t: [] for t in threshold_bins}

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # (chunk, n) — cosine similarity (embeddings are L2-normed)
        sims = np.clip(embeddings[start:end] @ embeddings.T, -1.0, 1.0)

        for t in threshold_bins:
            chunk_rows, chunk_cols = np.where(sims >= t)
            # Map local row indices to global and exclude self-loops
            global_rows = chunk_rows + start
            keep = global_rows != chunk_cols
            rows[t].append(global_rows[keep])
            cols[t].append(chunk_cols[keep])

        if end % (chunk_size * 5) == 0 or end == n:
            logger.info("  processed %d / %d docs", end, n)

    result: Dict[float, sp.csr_matrix] = {}
    for t in sorted(threshold_bins, reverse=True):  # strictest first
        r = np.concatenate(rows[t]) if rows[t] else np.array([], dtype=np.intp)
        c = np.concatenate(cols[t]) if cols[t] else np.array([], dtype=np.intp)
        adj = sp.csr_matrix(
            (np.ones(len(r), dtype=bool), (r, c)),
            shape=(n, n),
        )
        avg_deg = adj.nnz / max(n, 1)
        density_pct = 100.0 * adj.nnz / max(n * n, 1)
        logger.info("  thresh=%.2f: nnz=%d, density=%.4f%%, avg_degree=%.1f",
                     t, adj.nnz, density_pct, avg_deg)
        if avg_deg > max_avg_degree:
            logger.warning("  SKIPPING thresh=%.2f — avg_degree %.1f > %d (snowball risk)",
                           t, avg_deg, max_avg_degree)
            continue
        result[t] = adj

    logger.info("Sim graph built: %d thresholds retained", len(result))
    return result


# ---------------------------------------------------------------------------
# 3. Neighbour-label masks (for BO evaluation)
# ---------------------------------------------------------------------------

def precompute_neighbor_label_masks(
    sim_graphs: Dict[float, sp.csr_matrix],
    label_state: np.ndarray,
    label_names: List[str],
) -> Dict[Tuple[str, float], np.ndarray]:
    """For each (label, threshold), compute a bool mask over docs.

    ``mask[i] = True`` iff any neighbour *j* of doc *i* (at threshold θ)
    has *label* active in *label_state*.

    Uses SpMV: ``mask = (adj @ label_state[:, lidx].astype(float32)) > 0``.

    Parameters
    ----------
    sim_graphs : {threshold: csr_matrix(n, n)}
    label_state : (n_docs, n_labels) bool array — can be base predictions,
                  ground truth, or Track-1-updated predictions.
    label_names : list of label name strings (length n_labels).

    Returns
    -------
    dict mapping (label_name, threshold) → (n_docs,) bool array
    """
    masks: Dict[Tuple[str, float], np.ndarray] = {}
    for thresh, adj in sim_graphs.items():
        for lidx, lname in enumerate(label_names):
            col = label_state[:, lidx].astype(np.float32)
            has_neighbor = np.asarray((adj @ col) > 0).ravel()
            masks[(lname, thresh)] = has_neighbor
    logger.info("Precomputed %d neighbor-label masks", len(masks))
    return masks


def precompute_neighbor_label_counts(
    sim_graphs: Dict[float, sp.csr_matrix],
    label_state: np.ndarray,
    label_names: List[str],
) -> Dict[Tuple[str, float], np.ndarray]:
    """For each (label, threshold), compute neighbor count per doc.

    ``counts[i]`` = number of neighbours *j* of doc *i* (at threshold θ)
    that have *label* active in *label_state*.

    Uses SpMV: ``counts = adj @ label_state[:, lidx]``.
    """
    counts: Dict[Tuple[str, float], np.ndarray] = {}
    for thresh, adj in sim_graphs.items():
        for lidx, lname in enumerate(label_names):
            col = label_state[:, lidx].astype(np.float32)
            count = np.asarray(adj @ col).ravel().astype(np.int32)
            counts[(lname, thresh)] = count
    logger.info("Precomputed %d neighbor-label counts", len(counts))
    return counts


# ---------------------------------------------------------------------------
# 4. Persistence
# ---------------------------------------------------------------------------

def save_sim_graphs(sim_graphs: Dict[float, sp.csr_matrix], directory: str) -> None:
    """Save each threshold's adjacency matrix as a separate .npz file."""
    os.makedirs(directory, exist_ok=True)
    for thresh, adj in sim_graphs.items():
        path = os.path.join(directory, f"thresh_{thresh:.2f}.npz")
        sp.save_npz(path, adj)
    logger.info("Saved %d sim graphs to %s", len(sim_graphs), directory)


def load_sim_graphs(directory: str) -> Dict[float, sp.csr_matrix]:
    """Load sim graphs saved by :func:`save_sim_graphs`."""
    result: Dict[float, sp.csr_matrix] = {}
    for fname in sorted(os.listdir(directory)):
        if fname.startswith("thresh_") and fname.endswith(".npz"):
            thresh_str = fname[len("thresh_"):-len(".npz")]
            thresh = float(thresh_str)
            path = os.path.join(directory, fname)
            result[thresh] = sp.load_npz(path)
    logger.info("Loaded %d sim graphs from %s", len(result), directory)
    return result
