"""Shared infrastructure for the LORIS paper baselines.

Every baseline goes through these helpers so that **splits, labels, and the
metric are identical to a LORIS run** (paper §7 requires comparability):

* :func:`load_split`   — wrap ``loris.data.load_data`` with the canonical
  per-dataset caps used by ``run_main_table.sh``.
* :func:`score`        — standalone micro/macro-F1 (matches
  ``BaseDocumentClassifier.evaluate``), for methods that emit hard labels.
* :func:`tune_threshold` / :func:`apply_threshold` — LORIS's global-threshold
  convention (max micro-F1 on val) for methods that emit per-label scores.
* :func:`write_result` — uniform per-run JSON schema the aggregator reads.

Run baselines **from the repo root** so ``DATASET_REGISTRY`` resolves
``data/<name>/processed/``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from loris.data import DATASET_REGISTRY, HParams, load_data


# ── Canonical per-dataset config (mirrors run_main_table.sh) ──────────────────
# subset_size = labeled train cap; max_test_docs = test cap; has_text = real raw
# text available (False ⇒ transformer/LLM baselines are inapplicable, e.g. rcv1
# ships as hashed TF-IDF only — paper footnote).
@dataclass
class DSConfig:
    subset_size: int = 0
    max_test_docs: int = 0
    top_labels: int = 30
    has_text: bool = True


DATASET_DEFAULTS: Dict[str, DSConfig] = {
    "reuters21578": DSConfig(subset_size=0, max_test_docs=0, top_labels=30, has_text=True),
    "rcv1": DSConfig(subset_size=20000, max_test_docs=0, top_labels=30, has_text=False),
    "aapd": DSConfig(subset_size=15000, max_test_docs=0, top_labels=30, has_text=True),
    "bgc": DSConfig(subset_size=12000, max_test_docs=8000, top_labels=30, has_text=True),
    "arxiv": DSConfig(subset_size=20000, max_test_docs=8000, top_labels=30, has_text=True),
    # PubMed MeSH: biomedical abstracts, 14 top-level MeSH categories (coarse,
    # dense multi-label). Real raw text ⇒ text predicates apply.
    "pubmed": DSConfig(subset_size=15000, max_test_docs=8000, top_labels=14, has_text=True),
    # HUPD patents (Jan-2016 sample): title+abstract, IPC subclass labels.
    "hupd": DSConfig(subset_size=15000, max_test_docs=8000, top_labels=30, has_text=True),
    # Goodreads book blurbs → 18 aggregated genres.
    "goodreads": DSConfig(subset_size=0, max_test_docs=0, top_labels=18, has_text=True),
    # Full / million-scale variants. subset_size=0 = no cap (override --subset_size
    # for a tractable single chase); test capped for eval-time tractability.
    "arxiv_full": DSConfig(subset_size=0, max_test_docs=8000, top_labels=100, has_text=True),
    "hupd_full": DSConfig(subset_size=0, max_test_docs=8000, top_labels=100, has_text=True),
    "pubmed_full": DSConfig(subset_size=0, max_test_docs=8000, top_labels=14, has_text=True),
    "goodreads_full": DSConfig(subset_size=0, max_test_docs=8000, top_labels=20, has_text=True),
}


@dataclass
class Split:
    """A loaded dataset split (single-val mode)."""
    dataset: str
    train_X: List[str]
    val_X: List[str]
    test_X: List[str]
    train_y: np.ndarray
    val_y: np.ndarray
    test_y: np.ndarray
    label_names: List[str]
    train_docs: list = field(default_factory=list)
    val_docs: list = field(default_factory=list)
    test_titles: list = field(default_factory=list)

    @property
    def n_labels(self) -> int:
        return len(self.label_names)


def build_hparams(
    dataset: str,
    top_labels: Optional[int] = None,
    subset_size: Optional[int] = None,
    max_test_docs: Optional[int] = None,
    val_ratio: float = 0.40,
) -> HParams:
    """HParams with the canonical caps for *dataset* (overridable)."""
    d = DATASET_DEFAULTS.get(dataset, DSConfig())
    hp = HParams(
        top_labels=top_labels if top_labels is not None else d.top_labels,
        subset_size=subset_size if subset_size is not None else d.subset_size,
        val_ratio=val_ratio,
        two_val=False,
    )
    hp.max_test_docs = max_test_docs if max_test_docs is not None else d.max_test_docs
    return hp


def load_split(
    dataset: str,
    top_labels: Optional[int] = None,
    subset_size: Optional[int] = None,
    max_test_docs: Optional[int] = None,
    val_ratio: float = 0.40,
) -> Split:
    """Load *dataset* through the LORIS data layer (single-val mode)."""
    if dataset not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{dataset}'. Known: {sorted(DATASET_REGISTRY)}"
        )
    cfg = DATASET_REGISTRY[dataset]
    hp = build_hparams(dataset, top_labels, subset_size, max_test_docs, val_ratio)
    out = load_data(cfg, hp)
    # single-val return: train_X, val_X, test_X, train_y, val_y, test_y,
    #                    label_names, train_docs, val_docs, test_titles
    (train_X, val_X, test_X, train_y, val_y, test_y,
     label_names, train_docs, val_docs, test_titles) = out
    return Split(
        dataset=dataset,
        train_X=train_X, val_X=val_X, test_X=test_X,
        train_y=np.asarray(train_y), val_y=np.asarray(val_y), test_y=np.asarray(test_y),
        label_names=list(label_names),
        train_docs=train_docs, val_docs=val_docs, test_titles=test_titles,
    )


def has_raw_text(dataset: str) -> bool:
    """False ⇒ transformer/LLM baselines are inapplicable (e.g. rcv1)."""
    return DATASET_DEFAULTS.get(dataset, DSConfig()).has_text


# ── Metrics ───────────────────────────────────────────────────────────────────
def score(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Standalone micro/macro-F1 + subset-accuracy (matches base.evaluate)."""
    from sklearn.metrics import f1_score, accuracy_score
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
    }


def tune_threshold(val_scores: np.ndarray, val_y: np.ndarray) -> float:
    """Global threshold maximising micro-F1 on val (LORIS convention)."""
    from sklearn.metrics import f1_score
    val_y = np.asarray(val_y, dtype=np.int8)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        p = (val_scores >= t).astype(np.int8)
        f1 = f1_score(val_y, p, average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def apply_threshold(scores: np.ndarray, thr: float) -> np.ndarray:
    return (np.asarray(scores) >= thr).astype(np.int8)


def score_from_scores(
    test_scores: np.ndarray, test_y: np.ndarray,
    val_scores: np.ndarray, val_y: np.ndarray,
) -> Dict[str, float]:
    """Score per-label scores under a val-tuned global threshold."""
    thr = tune_threshold(val_scores, val_y)
    return {**score(test_y, apply_threshold(test_scores, thr)), "threshold": thr}


# ── Result IO ─────────────────────────────────────────────────────────────────
def write_result(
    out_dir: Path,
    baseline: str,
    dataset: str,
    metrics: Dict[str, float],
    *,
    seed: int = 0,
    regime: str = "full",
    n_annotations: int = 0,
    n_test: int = 0,
    wall_sec: float = 0.0,
    extra: Optional[dict] = None,
) -> Path:
    """Write the uniform per-run JSON the aggregator consumes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": baseline,
        "dataset": dataset,
        "seed": seed,
        "regime": regime,
        "macro_f1": float(metrics.get("macro_f1", float("nan"))),
        "micro_f1": float(metrics.get("micro_f1", float("nan"))),
        "subset_accuracy": float(metrics.get("subset_accuracy", float("nan"))),
        "n_annotations": int(n_annotations),
        "n_test": int(n_test),
        "wall_sec": float(wall_sec),
        "extra": extra or {},
    }
    path = out_dir / f"{baseline}__{dataset}__seed{seed}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


class Timer:
    """``with Timer() as t: ...`` then ``t.sec``."""
    def __enter__(self):
        self._t0 = time.time()
        self.sec = 0.0
        return self

    def __exit__(self, *exc):
        self.sec = time.time() - self._t0
        return False
