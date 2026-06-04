"""Logging setup and dataset loading.

``load_data`` reads a dataset's prepared CSVs, restricts to the top-N labels,
applies the optional subset / dual-validation split, and wraps rows as
:class:`~loris.document.Document` objects. Migrated from
``run_loris_multi_pipeline.py`` (Phase 5); the legacy module re-exports as shim.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from loris.document import Document
from loris.data.config import HParams
from loris.data.prepare import DatasetConfig

log = logging.getLogger("loris_pipeline")


def configure_logging(exp_dir: Path, debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    handlers: List[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(exp_dir / "pipeline.log", encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    if not debug:
        for noisy in ("optuna", "transformers", "sentence_transformers",
                      "torch", "sklearn"):
            logging.getLogger(noisy).setLevel(logging.WARNING)



def load_data(
    dataset_cfg: DatasetConfig,
    hp: HParams,
) -> Tuple:
    t0 = time.time()
    log.info("Loading %s …", dataset_cfg.display_name)

    processed_dir = dataset_cfg.data_dir / "processed"
    train_csv = processed_dir / "train.csv"
    test_csv = processed_dir / "test.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_dir}.\n"
            f"Run first: python run_loris_multi_pipeline.py "
            f"--dataset {dataset_cfg.name} --prepare"
        )

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # ── identify text/title/label columns ──────────────────────────────────────
    text_col = "text"
    has_title = "title" in train_df.columns
    title_col = "title" if has_title else None
    label_cols = [c for c in train_df.columns
                  if c not in (text_col, title_col) and c != "title"]

    # ── restrict to top-N labels by frequency ─────────────────────────────────
    label_freq = train_df[label_cols].sum().sort_values(ascending=False)
    top_label_cols = label_freq.index[: hp.top_labels].tolist()
    log.info("Top-%d labels: %s", hp.top_labels, top_label_cols[:10])

    keep_cols = [text_col] + ([title_col] if has_title else []) + top_label_cols
    train_df = train_df[keep_cols].dropna(subset=[text_col])
    test_df = test_df[keep_cols].dropna(subset=[text_col])

    # keep only rows with at least one active label
    train_df = train_df[train_df[top_label_cols].sum(axis=1) > 0].reset_index(drop=True)
    test_df = test_df[test_df[top_label_cols].sum(axis=1) > 0].reset_index(drop=True)

    # ── optional test subsample (cap eval size for a tractable test-time chase) ──
    _max_test = getattr(hp, "max_test_docs", 0)
    if _max_test and len(test_df) > _max_test:
        _dom_te = test_df[top_label_cols].values.argmax(axis=1)
        try:
            test_df, _ = train_test_split(
                test_df, train_size=_max_test, stratify=_dom_te, random_state=42)
        except ValueError:
            test_df = test_df.sample(_max_test, random_state=42)
        test_df = test_df.reset_index(drop=True)
        log.info("Capped test set to %d docs (--max_test_docs).", len(test_df))

    # ── optional stratified subsample ────────────────────────────────────────
    if hp.subset_size and len(train_df) > hp.subset_size:
        dominant = train_df[top_label_cols].values.argmax(axis=1)
        try:
            train_df, _ = train_test_split(
                train_df, train_size=hp.subset_size,
                stratify=dominant, random_state=42,
            )
        except ValueError:
            train_df = train_df.sample(hp.subset_size, random_state=42)
        train_df = train_df.reset_index(drop=True)
        log.info("Sampled %d training documents.", len(train_df))

    # ── train / val split ────────────────────────────────────────────────────
    dominant_train = train_df[top_label_cols].values.argmax(axis=1)
    try:
        tr_df, val_df = train_test_split(
            train_df, test_size=hp.val_ratio,
            stratify=dominant_train, random_state=42,
        )
    except ValueError:
        tr_df, val_df = train_test_split(
            train_df, test_size=hp.val_ratio, random_state=42
        )

    train_X = tr_df[text_col].tolist()
    test_X = test_df[text_col].tolist()
    train_y = tr_df[top_label_cols].values.astype(np.float32)
    test_y = test_df[top_label_cols].values.astype(np.float32)
    label_names = top_label_cols

    if has_title:
        log.info("Title column present: using explicit titles for ttl predicates")
    elif not getattr(hp, "no_title_predicates", False):
        log.info("No title column: deriving ttl from first sentence (heuristic)")
    else:
        log.info("Title predicates disabled (--no_title_predicates)")

    # ── title extraction (per-dataset heuristic when no explicit title) ──────
    def _derive_title(text: str) -> str:
        """First-sentence (or first 200 chars) heuristic for datasets
        without an explicit title column. Aim: capture the topic-stating
        opener of an abstract / blurb."""
        if not text:
            return ""
        # Split on first sentence boundary; cap to 200 chars
        for sep in (". ", "? ", "! ", "\n"):
            idx = text.find(sep)
            if 0 < idx < 200:
                return text[:idx].strip()
        return text[:200].strip()

    use_title_heuristic = (
        not has_title and not getattr(hp, "no_title_predicates", False)
    )

    def _df_titles(df) -> List[str]:
        if has_title:
            return df[title_col].fillna("").astype(str).tolist()
        if use_title_heuristic:
            return [_derive_title(t) for t in df[text_col].tolist()]
        return ["" for _ in range(len(df))]

    train_titles = _df_titles(tr_df)
    test_titles = _df_titles(test_df)

    # ── wrap as Document objects ──────────────────────────────────────────────
    def _to_docs(texts: List[str], labels_mat: np.ndarray,
                 titles: Optional[List[str]] = None) -> List[Document]:
        docs = []
        titles = titles if titles is not None else [""] * len(texts)
        for txt, lbl_row, ttl in zip(texts, labels_mat, titles):
            active = {label_names[i] for i, v in enumerate(lbl_row) if v > 0}
            docs.append(Document(cnt=txt, ttl=(ttl or ""), lbl=active))
        return docs

    train_docs = _to_docs(train_X, train_y, train_titles)

    # ── optional two-val split: val → val_bo + val_select ────────────────────
    if hp.two_val:
        dominant_val = val_df[top_label_cols].values.argmax(axis=1)
        try:
            val_bo_df, val_select_df = train_test_split(
                val_df, test_size=0.5,
                stratify=dominant_val, random_state=42,
            )
        except ValueError:
            val_bo_df, val_select_df = train_test_split(
                val_df, test_size=0.5, random_state=42
            )

        val_X = val_bo_df[text_col].tolist()
        val_y = val_bo_df[top_label_cols].values.astype(np.float32)
        val_docs = _to_docs(val_X, val_y, _df_titles(val_bo_df))

        val_select_X = val_select_df[text_col].tolist()
        val_select_y = val_select_df[top_label_cols].values.astype(np.float32)
        val_select_docs = _to_docs(val_select_X, val_select_y, _df_titles(val_select_df))

        log.info(
            "Data loaded in %.1fs — train=%d  val_bo=%d  val_select=%d  test=%d  labels=%d",
            time.time() - t0, len(train_X), len(val_X), len(val_select_X),
            len(test_X), len(label_names),
        )
        return (
            train_X, val_X, test_X,
            train_y, val_y, test_y,
            label_names,
            train_docs, val_docs,
            val_select_X, val_select_y, val_select_docs,
            test_titles,  # index 12
        )
    else:
        val_X = val_df[text_col].tolist()
        val_y = val_df[top_label_cols].values.astype(np.float32)
        val_docs = _to_docs(val_X, val_y, _df_titles(val_df))

        log.info(
            "Data loaded in %.1fs — train=%d  val=%d  test=%d  labels=%d",
            time.time() - t0, len(train_X), len(val_X), len(test_X), len(label_names),
        )
        return (
            train_X, val_X, test_X,
            train_y, val_y, test_y,
            label_names,
            train_docs, val_docs,
            test_titles,  # index 9 in single-val mode
        )


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — model pool initialisation
# ──────────────────────────────────────────────────────────────────────────────

