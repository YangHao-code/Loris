from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .config import ExperimentConfig
from .utils import LOG, file_sha256, normalize_text, text_digest


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    relative_dir: str
    native_labels: int
    real_text: bool = True
    recommended_model: str = "microsoft/deberta-v3-base"
    recommended_max_length: int = 384


DATASETS: Dict[str, DatasetSpec] = {
    "reuters21578": DatasetSpec(
        "reuters21578", "data/reuters21578/processed", 119,
        recommended_max_length=512,
    ),
    "aapd": DatasetSpec(
        "aapd", "data/aapd/processed", 54,
        recommended_model="allenai/scibert_scivocab_uncased",
        recommended_max_length=384,
    ),
    "bgc": DatasetSpec("bgc", "data/bgc/processed", 146, recommended_max_length=384),
    "arxiv": DatasetSpec(
        "arxiv", "data/arxiv/processed", 40,
        recommended_model="allenai/scibert_scivocab_uncased",
        recommended_max_length=512,
    ),
    "pubmed": DatasetSpec(
        "pubmed", "data/pubmed/processed", 14,
        recommended_model=(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        ),
        recommended_max_length=512,
    ),
    "hupd": DatasetSpec(
        "hupd", "data/hupd/processed", 60,
        recommended_model="anferico/bert-for-patents",
        recommended_max_length=256,
    ),
    # RCV1's "text" is a sequence of anonymised sparse-feature IDs (w123 ...).
    "rcv1": DatasetSpec(
        "rcv1", "data/rcv1/processed_subset20k", 103,
        real_text=False, recommended_model="hash_transformer",
        recommended_max_length=128,
    ),
}


# Historical Exp-1/main_7ds caps.  These are audit metadata, not implicit
# defaults: ``max_train_docs=0`` and ``max_test_docs=0`` always remain uncapped.
# A caller requesting the exact paper cells must pass the non-zero cap shown
# here (AAPD/BGC/PubMed train and BGC target).
LORIS_MAIN_7DS_CAPS = {
    "reuters21578": {"max_train_docs": 0, "max_test_docs": 0},
    "aapd": {"max_train_docs": 15000, "max_test_docs": 0},
    "rcv1": {"max_train_docs": 20000, "max_test_docs": 0},
    "bgc": {"max_train_docs": 12000, "max_test_docs": 8000},
    "arxiv": {"max_train_docs": 20000, "max_test_docs": 0},
    "pubmed": {"max_train_docs": 15000, "max_test_docs": 0},
    "hupd": {"max_train_docs": 15000, "max_test_docs": 0},
}


@dataclass
class SplitBundle:
    dataset: str
    label_names: List[str]
    fit_texts: List[str]
    fit_y: np.ndarray
    val_texts: List[str]
    val_y: np.ndarray
    select_texts: List[str]
    select_y: np.ndarray
    unlabeled_texts: List[str]
    test_texts: List[str]
    test_y: np.ndarray
    fit_indices: np.ndarray
    val_indices: np.ndarray
    select_indices: np.ndarray
    unlabeled_indices: np.ndarray
    test_indices: np.ndarray
    manifest: dict

    @property
    def n_labels(self) -> int:
        return len(self.label_names)


def _dominant(y: np.ndarray) -> np.ndarray:
    dominant = y.argmax(axis=1)
    return np.where(y.sum(axis=1) > 0, dominant, y.shape[1])


def _random_indices(total: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(total, size=count, replace=False)).astype(np.int64)


def _dominant_indices(y: np.ndarray, count: int, seed: int) -> np.ndarray:
    all_indices = np.arange(len(y), dtype=np.int64)
    if count >= len(y):
        return all_indices
    try:
        chosen, _ = train_test_split(
            all_indices,
            train_size=count,
            random_state=seed,
            stratify=_dominant(y),
        )
        return np.sort(chosen.astype(np.int64))
    except ValueError:
        return _random_indices(len(y), count, seed)


def _coverage_indices(y: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Greedy batched multi-label coverage selection.

    This is deliberately labelled as an oracle acquisition simulation: it uses
    labels to create a low-variance benchmark subset.  ``random`` remains
    available for a deployable, label-blind acquisition comparison.
    """
    total, n_labels = y.shape
    if count >= total:
        return np.arange(total, dtype=np.int64)
    if count <= 0:
        return np.empty(0, dtype=np.int64)

    y_float = np.asarray(y, dtype=np.float32)
    frequencies = y_float.sum(axis=0)
    target = frequencies * (float(count) / float(total))
    target[(frequencies > 0) & (target < 1.0)] = 1.0
    selected = np.zeros(total, dtype=bool)
    current = np.zeros(n_labels, dtype=np.float32)
    rng = np.random.RandomState(seed)
    jitter = rng.uniform(0.0, 1.0e-7, size=total).astype(np.float32)
    cardinality = np.maximum(1.0, y_float.sum(axis=1))
    batch = max(8, min(128, int(math.ceil(count / 50.0))))

    while int(selected.sum()) < count:
        deficit = np.maximum(target - current, 0.0)
        if not np.any(deficit > 1.0e-6):
            break
        # Relative deficit preserves marginals; inverse frequency gives tail
        # labels enough influence to appear in a small annotation budget.
        weights = deficit / np.maximum(target, 1.0)
        weights /= np.sqrt(np.maximum(frequencies, 1.0))
        scores = (y_float @ weights) / np.sqrt(cardinality) + jitter
        scores[selected] = -np.inf
        take = min(batch, count - int(selected.sum()))
        candidates = np.argpartition(scores, -take)[-take:]
        candidates = candidates[np.argsort(scores[candidates])[::-1]]
        selected[candidates] = True
        current += y_float[candidates].sum(axis=0)

    remaining = count - int(selected.sum())
    if remaining:
        pool = np.flatnonzero(~selected)
        selected[rng.choice(pool, size=remaining, replace=False)] = True
    return np.flatnonzero(selected).astype(np.int64)


def select_indices(y: np.ndarray, count: int, method: str, seed: int) -> np.ndarray:
    if method == "coverage":
        return _coverage_indices(y, count, seed)
    if method == "dominant":
        return _dominant_indices(y, count, seed)
    if method == "random":
        return _random_indices(len(y), count, seed)
    raise ValueError("unknown sampling method: %s" % method)


def _split_indices(
    indices: np.ndarray,
    y: np.ndarray,
    heldout_count: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if heldout_count <= 0:
        return np.sort(indices), np.empty(0, dtype=np.int64)
    if heldout_count >= len(indices):
        raise ValueError("heldout split would leave no fit examples")
    local = np.arange(len(indices), dtype=np.int64)
    local_y = y[indices]
    try:
        kept, heldout = train_test_split(
            local,
            test_size=heldout_count,
            random_state=seed,
            stratify=_dominant(local_y),
        )
    except ValueError:
        kept, heldout = train_test_split(
            local, test_size=heldout_count, random_state=seed,
        )
    return np.sort(indices[kept]), np.sort(indices[heldout])


def _stable_cap(indices: np.ndarray, texts: Sequence[str], cap: int, namespace: str) -> np.ndarray:
    if not cap or len(indices) <= cap:
        return np.asarray(indices, dtype=np.int64)
    keyed = sorted(
        ((text_digest(namespace + "\0" + str(int(i)) + "\0" + texts[int(i)]), int(i))
         for i in indices),
        key=lambda item: (item[0], item[1]),
    )
    return np.asarray(sorted(index for _, index in keyed[:cap]), dtype=np.int64)


def _support(y: np.ndarray, names: Sequence[str]) -> Dict[str, int]:
    if len(y) == 0:
        return {name: 0 for name in names}
    counts = np.asarray(y).sum(axis=0)
    return {name: int(counts[i]) for i, name in enumerate(names)}


def _source_index_digest(indices: np.ndarray) -> str:
    """Stable digest for an ordered role split in original-CSV index space."""
    values = np.asarray(indices, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(b"loris_ssl/source_csv_row_index/v1\0")
    digest.update(str(len(values)).encode("ascii"))
    digest.update(b"\0")
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _loris_dominant(frame: pd.DataFrame, labels: Sequence[str]) -> np.ndarray:
    values = frame[list(labels)].to_numpy()
    dominant = values.argmax(axis=1)
    return np.where(values.sum(axis=1) > 0, dominant, len(labels))


def _loris_split_frame(
    frame: pd.DataFrame,
    *,
    train_size: Optional[int] = None,
    test_size: Optional[float] = None,
    labels: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Replay Loris's dominant-label split, including its fallback."""
    kwargs = {
        "random_state": 42,
        "stratify": _loris_dominant(frame, labels),
    }
    if train_size is not None:
        kwargs["train_size"] = int(train_size)
    if test_size is not None:
        kwargs["test_size"] = float(test_size)
    try:
        left, right = train_test_split(frame, **kwargs)
    except ValueError:
        kwargs.pop("stratify")
        left, right = train_test_split(frame, **kwargs)
    return left, right


def _loris_exact_test_cap(
    frame: pd.DataFrame,
    cap: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Replay ``loris.data.loader``'s label-blind ``test-cap-v1`` order."""
    if not cap or len(frame) <= int(cap):
        return frame.reset_index(drop=True), frame.iloc[0:0].copy()
    keys = [
        hashlib.sha256(
            (
                "test-cap-v1\0test:%d\0%s"
                % (int(source_index), str(text))
            ).encode("utf-8")
        ).hexdigest()
        for source_index, text in zip(
            frame["__source_index__"].tolist(), frame["text"].tolist()
        )
    ]
    ordered = (
        frame.assign(
            __loris_cap_key__=keys,
            __loris_row_id__=frame["__source_index__"].map(
                lambda value: "test:%d" % int(value)
            ),
        )
        .sort_values(["__loris_cap_key__", "__loris_row_id__"])
    )
    kept = ordered.head(int(cap)).drop(
        columns=["__loris_cap_key__", "__loris_row_id__"]
    )
    removed = ordered.iloc[int(cap):].drop(
        columns=["__loris_cap_key__", "__loris_row_id__"]
    )
    return kept.reset_index(drop=True), removed.reset_index(drop=True)


def _load_loris_exact(
    cfg: ExperimentConfig,
    source_train: pd.DataFrame,
    source_test: pd.DataFrame,
    train_path: Path,
    test_path: Path,
) -> SplitBundle:
    """Replay the Loris main-table row membership exactly.

    It first reproduces task-family construction, conditional all-zero
    filtering, caps, and the fixed seed-42 60/40 + validation-half splits used
    by ``loris.data.loader``.  A smaller annotation fraction is acquired only
    inside the already-frozen full fit role; its complement is the sole SSL
    unlabeled pool.  Model RNG may use ``cfg.seed`` without changing any role.
    """
    if not math.isclose(float(cfg.val_fraction), 0.40):
        raise ValueError("loris_exact requires val_fraction=0.40")
    if not math.isclose(float(cfg.select_fraction), 0.50):
        raise ValueError("loris_exact requires select_fraction=0.50")
    if int(cfg.max_unlabeled_docs or 0):
        raise ValueError(
            "loris_exact requires max_unlabeled_docs=0 so every unrevealed "
            "full-fit row remains in the unlabeled pool"
        )

    ignored = {"text", "title", "__source_index__"}
    label_columns = [column for column in source_train.columns if column not in ignored]
    missing_test = sorted(set(label_columns) - set(source_test.columns))
    if missing_test:
        raise ValueError("test split is missing label columns: %s" % missing_test[:5])

    # Loris freezes top-N before missing-text and all-zero filtering.
    frequency = source_train[label_columns].sum(axis=0).sort_values(ascending=False)
    label_names = frequency.index[: min(cfg.top_labels, len(frequency))].tolist()
    keep_columns = ["text", "__source_index__"]
    source_train = source_train[keep_columns + label_names].dropna(subset=["text"])
    source_test = source_test[keep_columns + label_names].dropna(subset=["text"])
    source_train = source_train[
        source_train[label_names].sum(axis=1) > 0
    ].reset_index(drop=True)
    source_test = source_test[
        source_test[label_names].sum(axis=1) > 0
    ].reset_index(drop=True)
    conditional_train_count = len(source_train)
    conditional_test_count = len(source_test)

    test_frame, test_cap_removed = _loris_exact_test_cap(
        source_test, int(cfg.max_test_docs or 0)
    )

    train_cap = int(cfg.max_train_docs or 0)
    if train_cap and len(source_train) > train_cap:
        # Loris's cap fallback is DataFrame.sample (its later validation-split
        # fallback is train_test_split without stratification).  PubMed reaches
        # this branch because one dominant-label stratum has a single row.
        try:
            train_pool, train_cap_removed = train_test_split(
                source_train,
                train_size=train_cap,
                stratify=_loris_dominant(source_train, label_names),
                random_state=42,
            )
        except ValueError:
            train_pool = source_train.sample(train_cap, random_state=42)
            train_cap_removed = source_train.drop(train_pool.index)
        train_pool = train_pool.reset_index(drop=True)
        train_cap_removed = train_cap_removed.reset_index(drop=True)
    else:
        train_pool = source_train.reset_index(drop=True)
        train_cap_removed = source_train.iloc[0:0].copy()

    full_fit_frame, validation_frame = _loris_split_frame(
        train_pool, test_size=0.40, labels=label_names
    )
    val_ctrl_frame, val_select_frame = _loris_split_frame(
        validation_frame, test_size=0.50, labels=label_names
    )

    # Freeze all outer roles first.  Acquisition is confined to full_fit_frame:
    # neither validation nor cap-removed rows can enter the unlabeled pool.
    full_fit_count = len(full_fit_frame)
    annotation_count = max(
        1, int(round(full_fit_count * float(cfg.labeled_fraction)))
    )
    annotation_count = min(annotation_count, full_fit_count)
    if annotation_count == full_fit_count:
        acquired_local = np.arange(full_fit_count, dtype=np.int64)
    else:
        acquired_local = select_indices(
            full_fit_frame[label_names].to_numpy(dtype=np.float32),
            annotation_count,
            cfg.sampling,
            int(cfg.acquisition_seed),
        )
    unrevealed_local = np.setdiff1d(
        np.arange(full_fit_count, dtype=np.int64),
        acquired_local,
        assume_unique=True,
    )
    fit_frame = full_fit_frame.iloc[acquired_local]
    unlabeled_frame = full_fit_frame.iloc[unrevealed_local]

    def indices(frame: pd.DataFrame) -> np.ndarray:
        return frame["__source_index__"].to_numpy(dtype=np.int64)

    full_fit_indices = indices(full_fit_frame)
    fit_indices = indices(fit_frame)
    val_indices = indices(val_ctrl_frame)
    select_indices_exact = indices(val_select_frame)
    test_indices = indices(test_frame)
    unlabeled_indices = indices(unlabeled_frame)
    source_role_indices = {
        "fit": fit_indices,
        "val_ctrl": val_indices,
        "val_select": select_indices_exact,
        "unlabeled": unlabeled_indices,
        "test": test_indices,
    }
    source_train_sets = {
        role: set(map(int, source_role_indices[role]))
        for role in ("fit", "val_ctrl", "val_select", "unlabeled")
    }
    for left_position, left in enumerate(source_train_sets):
        for right in list(source_train_sets)[left_position + 1:]:
            if source_train_sets[left] & source_train_sets[right]:
                raise RuntimeError(
                    "loris_exact source-train roles overlap: %s::%s"
                    % (left, right)
                )
    canonical_caps = LORIS_MAIN_7DS_CAPS.get(
        cfg.dataset, {"max_train_docs": 0, "max_test_docs": 0}
    )
    manifest = {
        "schema_version": 2,
        "dataset": cfg.dataset,
        "protocol": "loris_exact",
        "source": {
            "train_path": str(train_path.resolve()),
            "test_path": str(test_path.resolve()),
            "train_sha256": file_sha256(train_path),
            "test_sha256": file_sha256(test_path),
        },
        "label_names": label_names,
        "top_labels_defined_before_dropna_and_filtering": True,
        "target_truth_used_for_top_n_zero_filter": True,
        "sampling": cfg.sampling,
        "sampling_is_label_aware": cfg.sampling in {"coverage", "dominant"},
        "annotation_fraction": float(cfg.labeled_fraction),
        "annotation_fraction_scope": "frozen_full_fit_only_validation_fixed",
        "annotation_count_including_validation": int(
            len(fit_frame) + len(val_ctrl_frame) + len(val_select_frame)
        ),
        "counts": {
            "fit": int(len(fit_frame)),
            "val_ctrl": int(len(val_ctrl_frame)),
            "val_select": int(len(val_select_frame)),
            "unlabeled": int(len(unlabeled_frame)),
            "test": int(len(test_frame)),
        },
        "loris_exact_protocol": {
            "task_order": [
                "top_n_from_raw_train_frequency",
                "drop_missing_text",
                "drop_train_and_test_rows_zero_after_top_n",
                "label_blind_test_cap_test_cap_v1",
                "dominant_stratified_train_cap_seed_42",
                "dominant_stratified_train_validation_60_40_seed_42",
                "dominant_stratified_validation_halves_seed_42",
                "acquire_only_within_frozen_full_fit",
            ],
            "split_random_state": 42,
            "full_fit_count": int(full_fit_count),
            "full_fit_ordered_source_index_sha256": _source_index_digest(
                full_fit_indices
            ),
            "acquisition": {
                "seed": int(cfg.acquisition_seed),
                "sampling": cfg.sampling,
                "sampling_is_label_aware": cfg.sampling
                in {"coverage", "dominant"},
                "fraction_of_full_fit": float(cfg.labeled_fraction),
                "labeled_count": int(len(fit_frame)),
                "unlabeled_count": int(len(unlabeled_frame)),
                "cfg_seed_affects_membership": False,
            },
            "requested_caps": {
                "max_train_docs": train_cap,
                "max_test_docs": int(cfg.max_test_docs or 0),
            },
            "canonical_main_7ds_caps": canonical_caps,
            "matches_canonical_main_7ds_caps": {
                "train": train_cap == int(canonical_caps["max_train_docs"]),
                "test": int(cfg.max_test_docs or 0)
                == int(canonical_caps["max_test_docs"]),
            },
            "conditional_top_n_counts_before_caps": {
                "train": int(conditional_train_count),
                "test": int(conditional_test_count),
            },
            "cap_removed_counts": {
                "train": int(len(train_cap_removed)),
                "test": int(len(test_cap_removed)),
            },
            "excluded_train_rows_are_not_an_unlabeled_ssl_pool": True,
            "validation_rows_are_not_in_unlabeled_pool": True,
        },
        "source_train_role_indices_disjoint": True,
        "target_positive_support_withheld_until_final_evaluation": False,
        "target_selection_is_label_blind_after_conditional_task_filter": True,
        "split_index_provenance": {
            "index_space": "zero_based_source_csv_row_index",
            "defined_before_dropna_and_protocol_filtering": True,
            "ordered_role_sha256": {
                role: _source_index_digest(role_indices)
                for role, role_indices in source_role_indices.items()
            },
            "digest_encoding": (
                "domain-tag + decimal-count + little-endian-int64-bytes"
            ),
        },
        "cleaning": {
            "train_target_overlaps_removed": False,
            "train_duplicates_removed": False,
            "test_rows_preserved": False,
            "zero_target_rows_removed": True,
        },
    }

    def texts(frame: pd.DataFrame) -> List[str]:
        return frame["text"].astype(str).tolist()

    def labels(frame: pd.DataFrame) -> np.ndarray:
        return frame[label_names].to_numpy(dtype=np.float32)

    return SplitBundle(
        dataset=cfg.dataset,
        label_names=label_names,
        fit_texts=texts(fit_frame),
        fit_y=labels(fit_frame),
        val_texts=texts(val_ctrl_frame),
        val_y=labels(val_ctrl_frame),
        select_texts=texts(val_select_frame),
        select_y=labels(val_select_frame),
        unlabeled_texts=texts(unlabeled_frame),
        test_texts=texts(test_frame),
        test_y=labels(test_frame),
        fit_indices=fit_indices,
        val_indices=val_indices,
        select_indices=select_indices_exact,
        unlabeled_indices=unlabeled_indices,
        test_indices=test_indices,
        manifest=manifest,
    )


def load_splits(cfg: ExperimentConfig) -> SplitBundle:
    cfg.validate()
    if cfg.dataset not in DATASETS:
        raise KeyError("unknown dataset %r; choose from %s" % (cfg.dataset, sorted(DATASETS)))
    spec = DATASETS[cfg.dataset]
    relative = spec.relative_dir
    if cfg.dataset == "rcv1" and cfg.rcv1_full:
        relative = "data/rcv1/processed"
    base = Path(cfg.loris_root).resolve() / relative
    train_path, test_path = base / "train.csv", base / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("missing processed dataset under %s" % base)

    LOG.info("Reading %s from %s", cfg.dataset, base)
    source_train = pd.read_csv(train_path)
    source_test = pd.read_csv(test_path)
    if "text" not in source_train or "text" not in source_test:
        raise ValueError("processed CSV must contain a text column")
    source_train["__source_index__"] = source_train.index.astype(np.int64)
    source_test["__source_index__"] = source_test.index.astype(np.int64)
    if cfg.protocol == "loris_exact":
        return _load_loris_exact(
            cfg, source_train, source_test, train_path, test_path
        )
    source_train = source_train.dropna(subset=["text"]).copy()
    source_test = source_test.dropna(subset=["text"]).copy()
    source_train["text"] = source_train["text"].astype(str)
    source_test["text"] = source_test["text"].astype(str)

    ignored = {"text", "title", "__source_index__"}
    label_columns = [column for column in source_train.columns if column not in ignored]
    missing_test = sorted(set(label_columns) - set(source_test.columns))
    if missing_test:
        raise ValueError("test split is missing label columns: %s" % missing_test[:5])
    frequency = source_train[label_columns].sum(axis=0).sort_values(ascending=False)
    label_names = frequency.index[: min(cfg.top_labels, len(frequency))].tolist()

    train_hashes = source_train["text"].map(text_digest)
    test_hashes = source_test["text"].map(text_digest)
    overlap_hashes = set(train_hashes) & set(test_hashes)
    audit = {
        "train_rows_raw": int(len(source_train)),
        "test_rows_raw": int(len(source_test)),
        "train_duplicate_rows_raw": int(train_hashes.duplicated(keep="first").sum()),
        "test_duplicate_rows_raw": int(test_hashes.duplicated(keep="first").sum()),
        "cross_split_overlap_groups_raw": int(len(overlap_hashes)),
        "cross_split_train_rows_raw": int(train_hashes.isin(overlap_hashes).sum()),
        "cross_split_test_rows_raw": int(test_hashes.isin(overlap_hashes).sum()),
    }

    if cfg.protocol == "clean_ssl":
        # Target membership wins: remove duplicates and target overlaps only
        # from the source-train pool.  The official target rows stay unchanged.
        keep = ~train_hashes.isin(overlap_hashes)
        keep &= ~train_hashes.duplicated(keep="first")
        source_train = source_train.loc[keep].copy()
    train_columns = ["text", "__source_index__"] + label_names
    test_columns = ["text", "__source_index__"] + label_names
    source_train = source_train[train_columns].reset_index(drop=True)
    source_test = source_test[test_columns].reset_index(drop=True)

    if cfg.protocol == "loris_compat":
        # This reproduces Loris's conditional top-N task.  It is intentionally
        # separate because consulting target truth to drop all-zero rows is not
        # a clean deployment protocol.
        source_train = source_train[source_train[label_names].sum(axis=1) > 0].reset_index(drop=True)
        source_test = source_test[source_test[label_names].sum(axis=1) > 0].reset_index(drop=True)

    train_texts = source_train["text"].tolist()
    test_texts_all = source_test["text"].tolist()
    # Internal positions index the post-cleaning frames used below.  Public
    # split indices are mapped back to the immutable, zero-based source CSV
    # row positions so CSV SHA + split_indices.npz can reconstruct every row.
    train_source_indices = source_train["__source_index__"].to_numpy(dtype=np.int64)
    test_source_indices = source_test["__source_index__"].to_numpy(dtype=np.int64)
    train_y = source_train[label_names].to_numpy(dtype=np.float32)
    test_y_all = source_test[label_names].to_numpy(dtype=np.float32)
    universe_indices = np.arange(len(source_train), dtype=np.int64)
    if cfg.max_train_docs and len(universe_indices) > cfg.max_train_docs:
        universe_indices = _stable_cap(
            universe_indices, train_texts, cfg.max_train_docs,
            "%s-train-cap" % cfg.dataset,
        )

    annotation_count = max(1, int(round(len(universe_indices) * cfg.labeled_fraction)))
    annotation_count = min(annotation_count, len(universe_indices))
    # Sampling sees only the fixed source-train universe, never target labels.
    local_selected = select_indices(
        train_y[universe_indices], annotation_count, cfg.sampling, cfg.seed,
    )
    revealed = np.sort(universe_indices[local_selected])
    unlabeled = np.setdiff1d(universe_indices, revealed, assume_unique=True)

    heldout_total = int(round(len(revealed) * cfg.val_fraction))
    if len(revealed) >= 10 and cfg.val_fraction > 0:
        heldout_total = max(2, heldout_total)
    heldout_total = min(max(0, heldout_total), max(0, len(revealed) - 1))
    fit_indices, validation = _split_indices(
        revealed, train_y, heldout_total, cfg.seed + 101,
    )
    select_count = int(round(len(validation) * cfg.select_fraction))
    if len(validation) >= 4 and cfg.select_fraction > 0:
        select_count = max(1, min(len(validation) - 1, select_count))
    else:
        select_count = 0
    val_indices, select_split = _split_indices(
        validation, train_y, select_count, cfg.seed + 202,
    ) if len(validation) else (validation, np.empty(0, dtype=np.int64))

    unlabeled = _stable_cap(
        unlabeled, train_texts, cfg.max_unlabeled_docs,
        "%s-unlabeled-cap-%d" % (cfg.dataset, cfg.seed),
    )
    test_indices = _stable_cap(
        np.arange(len(source_test), dtype=np.int64),
        test_texts_all,
        cfg.max_test_docs,
        "%s-test-cap" % cfg.dataset,
    )

    split_sets = {
        "fit": set(map(int, fit_indices)),
        "val_ctrl": set(map(int, val_indices)),
        "val_select": set(map(int, select_split)),
        "unlabeled": set(map(int, unlabeled)),
    }
    names = list(split_sets)
    intersections = {}
    for left_pos, left in enumerate(names):
        for right in names[left_pos + 1:]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                intersections[left + "::" + right] = len(overlap)
    if intersections:
        raise RuntimeError("source-train roles overlap: %r" % intersections)

    source_role_indices = {
        "fit": train_source_indices[fit_indices],
        "val_ctrl": train_source_indices[val_indices],
        "val_select": train_source_indices[select_split],
        "unlabeled": train_source_indices[unlabeled],
        "test": test_source_indices[test_indices],
    }

    manifest = {
        "schema_version": 2,
        "dataset": cfg.dataset,
        "protocol": cfg.protocol,
        "source": {
            "train_path": str(train_path.resolve()),
            "test_path": str(test_path.resolve()),
            "train_sha256": file_sha256(train_path),
            "test_sha256": file_sha256(test_path),
        },
        "label_names": label_names,
        "top_labels_defined_from_source_train_only": True,
        "sampling": cfg.sampling,
        "sampling_is_label_aware": cfg.sampling in {"coverage", "dominant"},
        "annotation_fraction": float(cfg.labeled_fraction),
        "annotation_count_including_validation": int(len(revealed)),
        "counts": {
            "fit": int(len(fit_indices)),
            "val_ctrl": int(len(val_indices)),
            "val_select": int(len(select_split)),
            "unlabeled": int(len(unlabeled)),
            "test": int(len(test_indices)),
        },
        "positive_support_observed": {
            "fit": _support(train_y[fit_indices], label_names),
            "val_ctrl": _support(train_y[val_indices], label_names),
            "val_select": _support(train_y[select_split], label_names),
        },
        "target_positive_support_withheld_until_final_evaluation": True,
        "target_selection_is_label_blind": True,
        "source_train_role_indices_disjoint": True,
        "split_index_provenance": {
            "index_space": "zero_based_source_csv_row_index",
            "defined_before_dropna_and_protocol_filtering": True,
            "ordered_role_sha256": {
                role: _source_index_digest(indices)
                for role, indices in source_role_indices.items()
            },
            "digest_encoding": "domain-tag + decimal-count + little-endian-int64-bytes",
        },
        "contamination_audit": audit,
        "cleaning": {
            "normalization": "lowercase_and_collapse_whitespace_sha256",
            "train_target_overlaps_removed": cfg.protocol == "clean_ssl",
            "train_duplicates_removed": cfg.protocol == "clean_ssl",
            "test_rows_preserved": cfg.protocol == "clean_ssl",
            "zero_target_rows_removed": cfg.protocol == "loris_compat",
        },
    }

    LOG.info(
        "Task %s: fit=%d val_ctrl=%d val_select=%d unlabeled=%d test=%d labels=%d",
        cfg.dataset, len(fit_indices), len(val_indices), len(select_split),
        len(unlabeled), len(test_indices), len(label_names),
    )
    return SplitBundle(
        dataset=cfg.dataset,
        label_names=label_names,
        fit_texts=[train_texts[i] for i in fit_indices],
        fit_y=train_y[fit_indices],
        val_texts=[train_texts[i] for i in val_indices],
        val_y=train_y[val_indices],
        select_texts=[train_texts[i] for i in select_split],
        select_y=train_y[select_split],
        unlabeled_texts=[train_texts[i] for i in unlabeled],
        test_texts=[test_texts_all[i] for i in test_indices],
        test_y=test_y_all[test_indices],
        fit_indices=source_role_indices["fit"],
        val_indices=source_role_indices["val_ctrl"],
        select_indices=source_role_indices["val_select"],
        unlabeled_indices=source_role_indices["unlabeled"],
        test_indices=source_role_indices["test"],
        manifest=manifest,
    )


class TextDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Optional[np.ndarray] = None):
        self.texts = list(texts)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        if self.labels is not None and len(self.labels) != len(self.texts):
            raise ValueError("texts and labels have different lengths")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        item = {"text": self.texts[index]}
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


class TextBatchCollator:
    def __init__(self, tokenizer, max_length: int, truncation: str = "head_tail"):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.truncation = truncation

    def _head_tail(self, texts: Sequence[str]) -> dict:
        raw = self.tokenizer(
            list(texts), add_special_tokens=False, truncation=False,
            padding=False, return_attention_mask=False, verbose=False,
        )["input_ids"]
        budget = self.max_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        if budget < 0:
            raise ValueError("max_length is smaller than the tokenizer's required special tokens")
        head_count = int(round(budget * 0.75))
        tail_count = budget - head_count
        features = []
        for ids in raw:
            if len(ids) > budget:
                ids = ids[:head_count] + (ids[-tail_count:] if tail_count else [])
            built = self.tokenizer.build_inputs_with_special_tokens(ids)
            features.append({"input_ids": built, "attention_mask": [1] * len(built)})
        return self.tokenizer.pad(
            features, padding=True,
            return_tensors="pt",
        )

    def __call__(self, examples: Sequence[dict]) -> dict:
        texts = [example["text"] for example in examples]
        if self.truncation == "head_tail":
            batch = self._head_tail(texts)
        elif self.truncation == "head":
            batch = self.tokenizer(
                texts, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
        else:
            raise ValueError("truncation must be head or head_tail")
        if "labels" in examples[0]:
            labels = np.stack([example["labels"] for example in examples])
            batch["labels"] = torch.from_numpy(labels).float()
        return batch
