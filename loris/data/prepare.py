"""Dataset preparation, domain stopwords, and the dataset registry.

Migrated verbatim from ``run_loris_multi_pipeline.py`` SECTION 1 (Phase 5).
Each ``prepare_<dataset>`` downloads/normalises a corpus into the multi-hot CSV
the pipeline consumes; :data:`DATASET_REGISTRY` maps a dataset name to its
:class:`DatasetConfig` (paths, default label count, domain stopwords).

The legacy ``run_loris_multi_pipeline`` re-exports these names as a shim.
"""

from __future__ import annotations

import csv
import json
import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

log = logging.getLogger("loris_pipeline")

# Project root (repo dir). prepare.py lives at <root>/loris/data/prepare.py,
# so the root is three parents up. Dataset CSVs live under <root>/data/<name>/,
# matching the legacy layout that DATASET_REGISTRY points at.
_ROOT = Path(__file__).resolve().parents[2]

# ── Domain stopwords ──────────────────────────────────────────────────────────

AAPD_STOP_WORDS = {
    # academic boilerplate
    "abstract", "paper", "propose", "proposed", "approach", "method",
    "results", "show", "experimental", "model", "based", "using",
    "problem", "algorithm", "et", "al", "present", "study",
    "performance", "achieve", "evaluate", "introduce", "demonstrate",
}

EURLEX_STOP_WORDS = {
    # legal boilerplate
    "article", "regulation", "directive", "commission", "council",
    "member", "states", "european", "community", "parliament",
    "pursuant", "annex", "whereas", "shall", "thereof",
    "accordance", "referred", "paragraph", "section", "decision",
}

RCV1_STOP_WORDS = {
    # wire service abbreviations (same as Reuters-21578)
    "mln", "bln", "cts", "dlrs", "shr", "shrs", "pct", "stg",
    "revs", "avg", "prev", "oper", "qtr", "yr", "mths",
    "vs", "000", "net", "prior",
    "inc", "corp", "ltd", "co",
    "loss", "profit", "revenue", "earnings",
    "share", "shares", "stock", "price",
    "reuter", "reuters",
}

BGC_STOP_WORDS = {
    # publishing boilerplate
    "book", "novel", "story", "author", "edition", "published",
    "reader", "readers", "chapter", "page", "pages", "volume",
    "bestselling", "bestseller", "series", "sequel", "debut",
    "paperback", "hardcover", "publisher",
}

REUTERS21578_STOP_WORDS = {
    # wire service boilerplate (same as RCV1)
    "mln", "bln", "cts", "dlrs", "shr", "shrs", "pct", "stg",
    "revs", "avg", "prev", "oper", "qtr", "yr", "mths",
    "vs", "000", "net", "prior",
    "inc", "corp", "ltd", "co",
    "reuter", "reuters", "said",
}


# ── Download helper ───────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Download a file with retry. Skip if already exists."""
    if dest.exists() and dest.stat().st_size > 0:
        log.info("Already downloaded: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = desc or dest.name
    log.info("Downloading %s from %s …", label, url)
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (LORIS pipeline)")
    try:
        urllib.request.urlretrieve(url, str(dest))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {label} from {url}: {exc}\n"
            f"You may need to download manually and place at {dest}"
        ) from exc
    log.info("Downloaded %s (%.1f MB)", label, dest.stat().st_size / 1e6)
    return dest


def _save_multi_hot_csv(
    texts: List[str],
    labels_list: List[List[str]],
    all_labels: List[str],
    out_path: Path,
    titles: Optional[List[str]] = None,
) -> None:
    """Write multi-hot encoded CSV: text[,title],label0,label1,...

    If ``titles`` is provided and non-empty, a ``title`` column is added
    after the ``text`` column. Loaders should detect and consume it.
    """
    label2idx = {l: i for i, l in enumerate(all_labels)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    has_titles = titles is not None and any(bool(t) for t in titles)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["text"] + (["title"] if has_titles else []) + all_labels
        writer.writerow(header)
        for i, (text, doc_labels) in enumerate(zip(texts, labels_list)):
            vec = [0] * len(all_labels)
            for l in doc_labels:
                idx = label2idx.get(l)
                if idx is not None:
                    vec[idx] = 1
            row = [text]
            if has_titles:
                row.append(titles[i] if i < len(titles) else "")
            writer.writerow(row + vec)
    log.info("Saved %d docs × %d labels to %s%s",
             len(texts), len(all_labels), out_path,
             " (with titles)" if has_titles else "")


# ── Prepare: AAPD ────────────────────────────────────────────────────────────

def prepare_aapd(data_dir: Path) -> Tuple[Path, Path]:
    """Download AAPD dataset and convert to multi-hot CSV.

    AAPD (Arxiv Academic Paper Dataset): 55,840 CS paper abstracts → 54 arXiv topics.
    Source: Yang et al. (2018) "SGM: Sequence Generation Model for Multi-label Classification"

    The dataset is available from multiple sources. We try Zenodo first,
    then fallback to the SGM GitHub repository.
    """
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = data_dir / "processed"

    train_csv = processed_dir / "train.csv"
    test_csv = processed_dir / "test.csv"
    if train_csv.exists() and test_csv.exists():
        log.info("AAPD already prepared at %s", processed_dir)
        return train_csv, test_csv

    # ── Try downloading from Zenodo ───────────────────────────────────────────
    # Zenodo record 6344750 contains AAPD in the format used by
    # "Adapting Transformers for Multi-Label Text Classification"
    zenodo_url = "https://zenodo.org/records/6344750/files/AAPD.zip?download=1"
    zip_path = raw_dir / "AAPD.zip"

    downloaded = False
    try:
        _download_file(zenodo_url, zip_path, "AAPD from Zenodo")
        downloaded = True
    except RuntimeError:
        log.warning("Zenodo download failed. Trying alternative sources…")

    if downloaded:
        # Extract zip
        log.info("Extracting %s …", zip_path)
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(raw_dir))
        # Find extracted files — Zenodo version typically has
        # AAPD/aapd_train.tsv and AAPD/aapd_test.tsv
        # or text_train, text_test, label_train, label_test
        _convert_aapd_raw(raw_dir, processed_dir)
        return train_csv, test_csv

    # ── Fallback: try SGM GitHub (lancopku) ───────────────────────────────────
    sgm_base = "https://raw.githubusercontent.com/lancopku/SGM/master/data/aapd"
    for fname in ["text_train", "text_test", "label_train", "label_test"]:
        url = f"{sgm_base}/{fname}"
        try:
            _download_file(url, raw_dir / fname, f"AAPD {fname}")
        except RuntimeError:
            pass

    if (raw_dir / "text_train").exists():
        _convert_aapd_raw(raw_dir, processed_dir)
        return train_csv, test_csv

    raise RuntimeError(
        "Could not download AAPD dataset from any source.\n"
        "Please download manually:\n"
        "  Option 1: https://zenodo.org/records/6344750 (download AAPD.zip)\n"
        "  Option 2: https://github.com/lancopku/SGM/tree/master/data/aapd\n"
        f"Place files in {raw_dir} and re-run --prepare"
    )


def _convert_aapd_raw(raw_dir: Path, processed_dir: Path) -> None:
    """Convert AAPD raw files to standard multi-hot CSV."""
    # Try TSV format first (Zenodo / some mirrors)
    tsv_train = None
    for candidate in [
        raw_dir / "AAPD" / "aapd_train.tsv",
        raw_dir / "aapd_train.tsv",
        raw_dir / "AAPD" / "train.tsv",
    ]:
        if candidate.exists():
            tsv_train = candidate
            break

    if tsv_train is not None:
        tsv_test = tsv_train.parent / tsv_train.name.replace("train", "test")
        _convert_aapd_tsv(tsv_train, tsv_test, processed_dir)
        return

    # Try separate text/label files (SGM format)
    text_train = raw_dir / "text_train"
    label_train = raw_dir / "label_train"
    text_test = raw_dir / "text_test"
    label_test = raw_dir / "label_test"

    if text_train.exists() and label_train.exists():
        _convert_aapd_separate(
            text_train, label_train, text_test, label_test, processed_dir
        )
        return

    # Try looking inside subdirectories
    for subdir in raw_dir.iterdir():
        if subdir.is_dir():
            for fname in ["text_train", "aapd_train.tsv"]:
                if (subdir / fname).exists():
                    log.info("Found AAPD data in %s", subdir)
                    _convert_aapd_raw(subdir, processed_dir)
                    return

    raise FileNotFoundError(
        f"Could not find AAPD data files in {raw_dir}. "
        f"Expected: aapd_train.tsv or text_train+label_train"
    )


def _convert_aapd_tsv(
    train_tsv: Path, test_tsv: Path, processed_dir: Path
) -> None:
    """Convert AAPD TSV (text\\tlabel1 label2) to multi-hot CSV."""
    def _read_tsv(path: Path) -> Tuple[List[str], List[List[str]]]:
        texts, labels = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    texts.append(parts[0])
                    labels.append(parts[1].split())
                elif len(parts) == 1:
                    texts.append(parts[0])
                    labels.append([])
        return texts, labels

    train_texts, train_labels = _read_tsv(train_tsv)
    test_texts, test_labels = _read_tsv(test_tsv)

    all_labels = sorted(
        set(l for doc_labels in train_labels + test_labels for l in doc_labels)
    )
    log.info("AAPD: %d train, %d test, %d unique labels",
             len(train_texts), len(test_texts), len(all_labels))

    _save_multi_hot_csv(train_texts, train_labels, all_labels,
                        processed_dir / "train.csv")
    _save_multi_hot_csv(test_texts, test_labels, all_labels,
                        processed_dir / "test.csv")


def _convert_aapd_separate(
    text_train: Path, label_train: Path,
    text_test: Path, label_test: Path,
    processed_dir: Path,
) -> None:
    """Convert AAPD separate text/label files to multi-hot CSV."""
    def _read_lines(path: Path) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    train_texts = _read_lines(text_train)
    train_label_strs = _read_lines(label_train)
    train_labels = [s.split() for s in train_label_strs]

    if text_test.exists() and label_test.exists():
        test_texts = _read_lines(text_test)
        test_label_strs = _read_lines(label_test)
        test_labels = [s.split() for s in test_label_strs]
    else:
        # If no test split, create one from train
        log.warning("No test files found; splitting 80/20 from train.")
        from sklearn.model_selection import train_test_split as tts
        train_texts, test_texts, train_labels, test_labels = tts(
            train_texts, train_labels, test_size=0.2, random_state=42
        )

    all_labels = sorted(
        set(l for doc_labels in train_labels + test_labels for l in doc_labels)
    )
    log.info("AAPD: %d train, %d test, %d unique labels",
             len(train_texts), len(test_texts), len(all_labels))

    _save_multi_hot_csv(train_texts, train_labels, all_labels,
                        processed_dir / "train.csv")
    _save_multi_hot_csv(test_texts, test_labels, all_labels,
                        processed_dir / "test.csv")


# ── Prepare: EUR-Lex ──────────────────────────────────────────────────────────

def prepare_eurlex(data_dir: Path) -> Tuple[Path, Path]:
    """Download EUR-Lex dataset and convert to multi-hot CSV.

    EUR-Lex: EU legal documents → EuroVoc concept classification.
    We use the EUR-Lex 57K variant (Chalkidis et al., 2019).

    Tries HuggingFace `datasets` first, then falls back to direct download.
    """
    processed_dir = data_dir / "processed"
    train_csv = processed_dir / "train.csv"
    test_csv = processed_dir / "test.csv"
    if train_csv.exists() and test_csv.exists():
        log.info("EUR-Lex already prepared at %s", processed_dir)
        return train_csv, test_csv

    # ── Try HuggingFace datasets ──────────────────────────────────────────────
    try:
        from datasets import load_dataset  # type: ignore

        log.info("Loading EUR-Lex via HuggingFace datasets …")
        # Try multiple dataset names
        ds = None
        for ds_name in [
            "eurlex",
            "joelito/eurlex_resources",
            "multi_eurlex",
        ]:
            try:
                ds = load_dataset(ds_name, split=None, trust_remote_code=True)
                log.info("Loaded dataset: %s", ds_name)
                break
            except Exception:
                continue

        if ds is not None:
            _convert_eurlex_hf(ds, processed_dir)
            return train_csv, test_csv
        log.warning("No EUR-Lex dataset found on HuggingFace.")
    except ImportError:
        log.warning("HuggingFace `datasets` not installed.")

    # ── Fallback: direct download ─────────────────────────────────────────────
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # EUR-Lex 57K from AUEB NLP group
    eurlex_url = "http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets/EURLEX57K.json.zip"
    zip_path = raw_dir / "EURLEX57K.json.zip"

    try:
        _download_file(eurlex_url, zip_path, "EUR-Lex 57K")
        log.info("Extracting %s …", zip_path)
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(raw_dir))
        _convert_eurlex_json(raw_dir, processed_dir)
        return train_csv, test_csv
    except Exception as exc:
        log.warning("Direct EUR-Lex download failed: %s", exc)

    raise RuntimeError(
        "Could not download EUR-Lex dataset.\n"
        "Options:\n"
        "  1. pip install datasets && re-run --prepare\n"
        "  2. Download from http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/\n"
        f"     and place JSON files in {raw_dir}"
    )


def _convert_eurlex_hf(ds, processed_dir: Path) -> None:
    """Convert HuggingFace EUR-Lex dataset to multi-hot CSV."""
    # HuggingFace eurlex datasets have varying schemas.
    # Common fields: "text" (or "celex_id" + "text"), "labels" (list of ints or strings)
    train_split = ds.get("train", ds.get("training"))
    test_split = ds.get("test", ds.get("testing"))

    if train_split is None:
        raise ValueError(f"EUR-Lex HF dataset has unexpected splits: {list(ds.keys())}")

    # Determine text and label fields
    columns = train_split.column_names
    text_field = "text" if "text" in columns else "celex_id"
    label_field = None
    for candidate in ["labels", "eurovoc_concepts", "concepts", "label"]:
        if candidate in columns:
            label_field = candidate
            break
    if label_field is None:
        raise ValueError(f"Cannot find label field in EUR-Lex. Columns: {columns}")

    log.info("EUR-Lex HF: text=%s, labels=%s, train=%d, test=%d",
             text_field, label_field, len(train_split),
             len(test_split) if test_split else 0)

    def _extract(split):
        texts, labels = [], []
        for row in split:
            t = row.get(text_field, "")
            if isinstance(t, list):
                t = " ".join(str(x) for x in t)
            t = str(t).strip()
            if not t:
                continue
            lbls = row.get(label_field, [])
            if not isinstance(lbls, list):
                lbls = [lbls]
            labels.append([str(l) for l in lbls])
            texts.append(t)
        return texts, labels

    train_texts, train_labels = _extract(train_split)
    if test_split is not None:
        test_texts, test_labels = _extract(test_split)
    else:
        log.warning("No test split; creating 80/20 from train.")
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            train_texts, train_labels, test_size=0.2, random_state=42
        )

    all_labels = sorted(
        set(l for doc_labels in train_labels + test_labels for l in doc_labels)
    )
    log.info("EUR-Lex: %d train, %d test, %d unique labels",
             len(train_texts), len(test_texts), len(all_labels))

    _save_multi_hot_csv(train_texts, train_labels, all_labels,
                        processed_dir / "train.csv")
    _save_multi_hot_csv(test_texts, test_labels, all_labels,
                        processed_dir / "test.csv")


def _convert_eurlex_json(raw_dir: Path, processed_dir: Path) -> None:
    """Convert EUR-Lex 57K JSON files to multi-hot CSV."""
    # EURLEX57K format: each split is a JSON file with list of dicts
    # Each dict has "text", "concepts" (list of EuroVoc IDs)
    for split_name, out_name in [("train", "train"), ("test", "test")]:
        json_path = None
        for candidate in [
            raw_dir / f"EURLEX57K" / f"{split_name}.json",
            raw_dir / f"{split_name}.json",
            raw_dir / f"EURLEX57K_{split_name}.json",
        ]:
            if candidate.exists():
                json_path = candidate
                break

    if json_path is None:
        # Try to find any JSON files
        json_files = list(raw_dir.rglob("*.json"))
        raise FileNotFoundError(
            f"Cannot find EUR-Lex JSON splits. Found files: {json_files}"
        )

    # Load all splits
    all_texts = {"train": [], "test": []}
    all_labels_list = {"train": [], "test": []}

    for split_name in ["train", "test"]:
        json_path = None
        for candidate in [
            raw_dir / "EURLEX57K" / f"{split_name}.json",
            raw_dir / f"{split_name}.json",
        ]:
            if candidate.exists():
                json_path = candidate
                break
        if json_path is None:
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                text = item.get("text", item.get("header", ""))
                if isinstance(text, list):
                    text = " ".join(text)
                concepts = item.get("concepts", item.get("labels", []))
                if text.strip():
                    all_texts[split_name].append(text.strip())
                    all_labels_list[split_name].append(
                        [str(c) for c in concepts]
                    )

    if not all_texts["train"]:
        raise ValueError("No training data found in EUR-Lex JSON files.")

    if not all_texts["test"]:
        log.warning("No test split; creating 80/20 from train.")
        all_texts["train"], all_texts["test"], \
            all_labels_list["train"], all_labels_list["test"] = train_test_split(
                all_texts["train"], all_labels_list["train"],
                test_size=0.2, random_state=42,
            )

    unique_labels = sorted(set(
        l for split in all_labels_list.values()
        for doc_labels in split for l in doc_labels
    ))
    log.info("EUR-Lex: %d train, %d test, %d unique labels",
             len(all_texts["train"]), len(all_texts["test"]), len(unique_labels))

    _save_multi_hot_csv(all_texts["train"], all_labels_list["train"],
                        unique_labels, processed_dir / "train.csv")
    _save_multi_hot_csv(all_texts["test"], all_labels_list["test"],
                        unique_labels, processed_dir / "test.csv")


# ── Prepare: RCV1-v2 ──────────────────────────────────────────────────────────

def prepare_rcv1(data_dir: Path) -> Tuple[Path, Path]:
    """Prepare RCV1-v2 dataset from sklearn (pseudo-text from TF-IDF features).

    NOTE: sklearn's fetch_rcv1() provides pre-computed TF-IDF features, NOT raw text.
    We reconstruct "pseudo-text" by extracting the top-K feature names per document.
    This is a lossy approximation but sufficient for the LORIS pipeline.
    """
    processed_dir = data_dir / "processed"
    train_csv = processed_dir / "train.csv"
    test_csv = processed_dir / "test.csv"
    if train_csv.exists() and test_csv.exists():
        log.info("RCV1 already prepared at %s", processed_dir)
        return train_csv, test_csv

    from sklearn.datasets import fetch_rcv1

    log.info("Fetching RCV1-v2 via sklearn (this may take a while) …")
    rcv1 = fetch_rcv1()

    # rcv1.data: sparse (804414, 47236) — cosine-normalised TF-IDF
    # rcv1.target: sparse (804414, 103) — multi-label binary
    # rcv1.target_names: array of 103 topic names
    # rcv1.sample_id: document IDs

    n_total = rcv1.data.shape[0]
    n_features = rcv1.data.shape[1]
    label_names_all = list(rcv1.target_names)
    log.info("RCV1: %d docs, %d features, %d labels",
             n_total, n_features, len(label_names_all))

    # The first 23,149 samples (sample_id < 26150) form the "training set" in
    # the original Lewis et al. split. The rest is test.
    # sklearn orders by sample_id.
    train_mask = rcv1.sample_id < 26150
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(~train_mask)[0]

    log.info("RCV1 split: train=%d, test=%d", len(train_idx), len(test_idx))

    # Subsample test set (it's huge: ~780k). Keep 10k for efficiency.
    if len(test_idx) > 10000:
        rng = np.random.RandomState(42)
        test_idx = rng.choice(test_idx, size=10000, replace=False)
        test_idx.sort()
        log.info("Subsampled test set to %d docs", len(test_idx))

    # Also subsample train if huge
    if len(train_idx) > 20000:
        rng = np.random.RandomState(42)
        train_idx = rng.choice(train_idx, size=20000, replace=False)
        train_idx.sort()
        log.info("Subsampled train set to %d docs", len(train_idx))

    # Reconstruct pseudo-text: for each document, get the top-30 features by
    # TF-IDF weight and join them as space-separated words.
    log.info("Reconstructing pseudo-text from TF-IDF features …")

    # We need feature names — fetch_rcv1 doesn't provide them directly,
    # but they're hashed token IDs. We'll use the feature indices as
    # "word_XXXXX" tokens since actual vocabulary is not available.
    # This still allows TF-IDF and pattern matching to work.
    TOP_K_FEATURES = 50

    def _pseudo_texts(indices):
        texts = []
        data_subset = rcv1.data[indices]
        for i in range(len(indices)):
            row = data_subset[i].toarray().ravel()
            top_feat_idx = np.argsort(row)[-TOP_K_FEATURES:][::-1]
            top_feat_idx = top_feat_idx[row[top_feat_idx] > 0]
            tokens = [f"w{fi}" for fi in top_feat_idx]
            texts.append(" ".join(tokens) if tokens else "empty")
        return texts

    train_texts = _pseudo_texts(train_idx)
    test_texts = _pseudo_texts(test_idx)

    # Labels
    train_labels_mat = rcv1.target[train_idx].toarray()
    test_labels_mat = rcv1.target[test_idx].toarray()

    train_labels = [
        [label_names_all[j] for j in range(len(label_names_all))
         if train_labels_mat[i, j] > 0]
        for i in range(len(train_idx))
    ]
    test_labels = [
        [label_names_all[j] for j in range(len(label_names_all))
         if test_labels_mat[i, j] > 0]
        for i in range(len(test_idx))
    ]

    log.warning(
        "RCV1 pseudo-text mode: documents are reconstructed from TF-IDF features "
        "(not raw text). Pattern quality may differ from raw-text datasets."
    )

    _save_multi_hot_csv(train_texts, train_labels, label_names_all,
                        processed_dir / "train.csv")
    _save_multi_hot_csv(test_texts, test_labels, label_names_all,
                        processed_dir / "test.csv")
    return train_csv, test_csv


# ── Prepare: BGC ──────────────────────────────────────────────────────────────

def prepare_bgc(data_dir: Path) -> Tuple[Path, Path]:
    """Download Blurb Genre Collection and convert to multi-hot CSV.

    BGC: Book blurbs → hierarchical genre classification (~32 labels at level 1).
    Source: University of Hamburg Language Technology Group.
    License: CC BY-NC (copyright: Penguin Random House).
    """
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = data_dir / "processed"
    train_csv = processed_dir / "train.csv"
    test_csv = processed_dir / "test.csv"
    if train_csv.exists() and test_csv.exists():
        log.info("BGC already prepared at %s", processed_dir)
        return train_csv, test_csv

    # Download from University of Hamburg
    bgc_url = "https://fiona.uni-hamburg.de/ca89b3cf/blurbgenrecollectionen.zip"
    zip_path = raw_dir / "blurbgenrecollectionen.zip"

    try:
        _download_file(bgc_url, zip_path, "BGC")
    except RuntimeError:
        raise RuntimeError(
            "Could not download BGC dataset.\n"
            "Please download manually from:\n"
            "  https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html\n"
            f"Place the zip at {zip_path} and re-run --prepare"
        )

    log.info("Extracting %s …", zip_path)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(raw_dir))

    _convert_bgc_xml(raw_dir, processed_dir)
    return train_csv, test_csv


def _convert_bgc_xml(raw_dir: Path, processed_dir: Path) -> None:
    """Convert BGC XML files to multi-hot CSV."""
    import xml.etree.ElementTree as ET

    # BGC contains XML files: BlurbGenreCollection_EN_train.txt,
    # BlurbGenreCollection_EN_test.txt, BlurbGenreCollection_EN_dev.txt
    # Format: <book><title>...</title><body>blurb text</body><topics><d>genre</d>...</topics></book>

    all_texts = {"train": [], "test": []}
    all_titles = {"train": [], "test": []}
    all_labels_list = {"train": [], "test": []}

    for split_name, file_patterns in [
        ("train", ["*train*", "*Train*"]),
        ("test", ["*test*", "*Test*", "*dev*", "*Dev*"]),
    ]:
        found_files = []
        for pattern in file_patterns:
            found_files.extend(raw_dir.rglob(pattern))
        # Filter to only txt/xml files, skip __MACOSX and hidden files
        found_files = [
            f for f in found_files
            if f.suffix in (".txt", ".xml") and f.is_file()
            and "__MACOSX" not in str(f) and not f.name.startswith("._")
        ]

        for fpath in found_files:
            log.info("Parsing BGC file: %s", fpath)
            try:
                # BGC files may not be well-formed XML at the top level.
                # Wrap in a root element.
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()

                # Add root wrapper if not present
                if not content.strip().startswith("<?xml") and not content.strip().startswith("<collection"):
                    content = f"<collection>{content}</collection>"

                root = ET.fromstring(content)
                for book in root.iter("book"):
                    body = book.find("body")
                    title = book.find("title")
                    topics = book.find(".//topics")
                    if body is not None and body.text and topics is not None:
                        text = body.text.strip()
                        title_text = title.text.strip() if (title is not None and title.text) else ""
                        # BGC uses <d0>, <d1>, <d2> for hierarchy levels
                        genres = []
                        for child in topics:
                            if child.text and child.text.strip():
                                genres.append(child.text.strip())
                        if text and genres:
                            all_texts[split_name].append(text)
                            all_titles[split_name].append(title_text)
                            all_labels_list[split_name].append(genres)
            except ET.ParseError as exc:
                log.warning("XML parse error in %s: %s (trying line-by-line)", fpath, exc)
                _parse_bgc_fallback(fpath, all_texts[split_name],
                                    all_labels_list[split_name],
                                    all_titles[split_name])

    if not all_texts["train"]:
        # If train is empty, check if data is in a subdirectory
        for subdir in raw_dir.iterdir():
            if subdir.is_dir():
                _convert_bgc_xml(subdir, processed_dir)
                return
        raise FileNotFoundError(
            f"No BGC training data found in {raw_dir}. "
            f"Files present: {list(raw_dir.rglob('*'))[:20]}"
        )

    if not all_texts["test"]:
        log.warning("No test split; splitting 80/20 from train.")
        # Split keeping titles aligned
        if all_titles["train"] and len(all_titles["train"]) == len(all_texts["train"]):
            (
                all_texts["train"], all_texts["test"],
                all_titles["train"], all_titles["test"],
                all_labels_list["train"], all_labels_list["test"],
            ) = train_test_split(
                all_texts["train"], all_titles["train"], all_labels_list["train"],
                test_size=0.2, random_state=42,
            )
        else:
            all_texts["train"], all_texts["test"], \
                all_labels_list["train"], all_labels_list["test"] = train_test_split(
                    all_texts["train"], all_labels_list["train"],
                    test_size=0.2, random_state=42,
                )

    unique_labels = sorted(set(
        l for split in all_labels_list.values()
        for doc_labels in split for l in doc_labels
    ))
    log.info("BGC: %d train, %d test, %d unique labels",
             len(all_texts["train"]), len(all_texts["test"]), len(unique_labels))

    _save_multi_hot_csv(all_texts["train"], all_labels_list["train"],
                        unique_labels, processed_dir / "train.csv",
                        titles=all_titles.get("train"))
    _save_multi_hot_csv(all_texts["test"], all_labels_list["test"],
                        unique_labels, processed_dir / "test.csv",
                        titles=all_titles.get("test"))


def _parse_bgc_fallback(
    fpath: Path, texts: List[str], labels: List[List[str]],
    titles: Optional[List[str]] = None,
) -> None:
    """Fallback regex parser for BGC files with XML issues."""
    import re
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Extract <book ...>...</book> blocks
    book_pattern = re.compile(r"<book[^>]*>(.*?)</book>", re.DOTALL)
    body_pattern = re.compile(r"<body>(.*?)</body>", re.DOTALL)
    title_pattern = re.compile(r"<title>(.*?)</title>", re.DOTALL)
    # BGC uses <d0>, <d1>, <d2> for hierarchy levels
    topic_pattern = re.compile(r"<d\d*>(.*?)</d\d*>")

    for book_match in book_pattern.finditer(content):
        book_xml = book_match.group(1)
        body_match = body_pattern.search(book_xml)
        if body_match:
            text = body_match.group(1).strip()
            title_match = title_pattern.search(book_xml)
            title_text = title_match.group(1).strip() if title_match else ""
            genres = [m.group(1).strip() for m in topic_pattern.finditer(book_xml)]
            if text and genres:
                texts.append(text)
                labels.append(genres)
                if titles is not None:
                    titles.append(title_text)


def prepare_reuters21578(data_dir: Path) -> Tuple[Path, Path]:
    """Reuters-21578 — already pre-processed into multi-hot CSV."""
    processed_dir = data_dir / "processed"
    train_csv = processed_dir / "train.csv"
    test_csv = processed_dir / "test.csv"
    if train_csv.exists() and test_csv.exists():
        log.info("Reuters-21578 already prepared at %s", processed_dir)
        return train_csv, test_csv

    # Data was prepared by prepare_reuters21578.py into processed_reuters/
    alt_dir = data_dir / "processed_reuters"
    if (alt_dir / "train.csv").exists():
        processed_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(alt_dir / "train.csv", train_csv)
        shutil.copy2(alt_dir / "test.csv", test_csv)
        log.info("Reuters-21578 copied from %s → %s", alt_dir, processed_dir)
        return train_csv, test_csv

    raise FileNotFoundError(
        f"Reuters-21578 processed data not found at {alt_dir} or {processed_dir}.\n"
        "Run data/reuters21578/prepare_reuters21578.py first."
    )


# ── Prepare: PubMed MeSH ──────────────────────────────────────────────────────

PUBMED_STOP_WORDS = {
    # biomedical abstract boilerplate
    "patients", "patient", "study", "studies", "results", "result",
    "clinical", "using", "based", "analysis", "used", "cases", "case",
    "group", "groups", "significant", "associated", "treatment", "methods",
    "method", "conclusion", "conclusions", "background", "objective", "aim",
    "showed", "found", "observed", "compared", "including", "may",
}

# 14 top-level MeSH category roots present in the processed CSV (A..N, Z).
_PUBMED_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "L", "M", "N", "Z"]
_PUBMED_HF_REPO = "owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH"
_PUBMED_HF_FILE = "PubMed Multi Label Text Classification Dataset Processed.csv"


def prepare_pubmed(data_dir: Path) -> Tuple[Path, Path]:
    """PubMed MeSH multi-label — abstracts → 14 top-level MeSH categories.

    Source: HF dataset ``owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH``
    (columns ``Title`` + ``abstractText`` + 14 MeSH-root 0/1 columns A..N, Z).
    Text = title + abstract; labels = the MeSH roots. Fetched via
    ``huggingface_hub`` (honours ``HF_ENDPOINT`` for the mirror).
    """
    processed_dir = data_dir / "processed"
    train_csv = processed_dir / "train.csv"
    test_csv = processed_dir / "test.csv"
    if train_csv.exists() and test_csv.exists():
        log.info("PubMed already prepared at %s", processed_dir)
        return train_csv, test_csv

    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = raw_dir / _PUBMED_HF_FILE
    if not (raw_csv.exists() and raw_csv.stat().st_size > 0):
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "huggingface_hub is required to fetch the PubMed MeSH dataset"
            ) from exc
        got = hf_hub_download(_PUBMED_HF_REPO, _PUBMED_HF_FILE,
                              repo_type="dataset", local_dir=str(raw_dir))
        raw_csv = Path(got)

    df = pd.read_csv(raw_csv)
    labels = [c for c in _PUBMED_LABELS if c in df.columns]
    if not labels:
        raise RuntimeError(f"PubMed CSV missing MeSH label columns; got {list(df.columns)}")

    title = (df["Title"].fillna("").astype(str) if "Title" in df.columns
             else pd.Series([""] * len(df)))
    abstract = df["abstractText"].fillna("").astype(str)
    texts = [((t.strip() + ". " + a.strip()).strip() if t.strip() else a.strip())
             for t, a in zip(title.tolist(), abstract.tolist())]

    Y = (df[labels].apply(pd.to_numeric, errors="coerce").fillna(0) > 0).astype(int).values
    labels_list = [[labels[j] for j in range(len(labels)) if row[j]] for row in Y]

    # keep only docs with non-empty text AND at least one active label
    keep = [i for i, (t, ls) in enumerate(zip(texts, labels_list)) if t and ls]
    texts = [texts[i] for i in keep]
    labels_list = [labels_list[i] for i in keep]

    tr_txt, te_txt, tr_lab, te_lab = train_test_split(
        texts, labels_list, test_size=0.15, random_state=42)
    _save_multi_hot_csv(tr_txt, tr_lab, labels, train_csv)
    _save_multi_hot_csv(te_txt, te_lab, labels, test_csv)
    log.info("PubMed prepared: %d train / %d test × %d labels",
             len(tr_txt), len(te_txt), len(labels))
    return train_csv, test_csv


# ── Prepare: HUPD (patents) ───────────────────────────────────────────────────

HUPD_STOP_WORDS = {
    # patent boilerplate
    "device", "method", "system", "apparatus", "present", "invention",
    "embodiment", "embodiments", "comprising", "including", "wherein",
    "plurality", "configured", "first", "second", "least", "one",
    "provided", "may", "said", "according", "portion", "member",
}

_HUPD_HF_REPO = "HUPD/hupd"
_HUPD_HF_FILE = "data/sample-jan-2016.tar.gz"  # small slice (Jan 2016), disk-friendly
_HUPD_TOP_LABELS = 60  # keep the 60 most frequent IPC subclasses as columns


def prepare_hupd(data_dir: Path) -> Tuple[Path, Path]:
    """HUPD (Harvard USPTO Patent Dataset) — patent title+abstract → IPC subclasses.

    Uses the small ``sample-jan-2016.tar.gz`` slice (disk-friendly). Each patent
    JSON carries ``title``, ``abstract`` and ``ipcr_labels`` (a list of IPC codes);
    labels are truncated to the 4-char **subclass** level (e.g. ``A61B``) — a
    standard HUPD multi-label setup. Streamed straight from the tar (no per-file
    extraction). Fetched via ``huggingface_hub`` (honours ``HF_ENDPOINT``).
    """
    import tarfile
    from collections import Counter

    processed_dir = data_dir / "processed"
    train_csv = processed_dir / "train.csv"
    test_csv = processed_dir / "test.csv"
    if train_csv.exists() and test_csv.exists():
        log.info("HUPD already prepared at %s", processed_dir)
        return train_csv, test_csv

    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tar_path = raw_dir / "data" / "sample-jan-2016.tar.gz"
    if not (tar_path.exists() and tar_path.stat().st_size > 0):
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("huggingface_hub is required to fetch HUPD") from exc
        got = hf_hub_download(_HUPD_HF_REPO, _HUPD_HF_FILE,
                              repo_type="dataset", local_dir=str(raw_dir))
        tar_path = Path(got)

    texts: List[str] = []
    labels_list: List[List[str]] = []
    log.info("Streaming HUPD patents from %s …", tar_path)
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if not m.name.endswith(".json"):
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            try:
                d = json.load(fh)
            except Exception:
                continue
            title = (d.get("title") or "").strip()
            abstract = (d.get("abstract") or "").strip()
            text = (title + ". " + abstract).strip() if title else abstract
            ipc = d.get("ipcr_labels") or []
            subs = sorted({c[:4] for c in ipc if c and len(c) >= 4})
            if not text or not subs:
                continue
            texts.append(text)
            labels_list.append(subs)

    if not texts:
        raise RuntimeError(f"No usable HUPD patents parsed from {tar_path}")

    # keep the most frequent subclasses as label columns; restrict docs to them
    freq = Counter(l for ls in labels_list for l in ls)
    keep_labels = [l for l, _ in freq.most_common(_HUPD_TOP_LABELS)]
    keepset = set(keep_labels)
    f_txt: List[str] = []
    f_lab: List[List[str]] = []
    for t, ls in zip(texts, labels_list):
        ls2 = [l for l in ls if l in keepset]
        if ls2:
            f_txt.append(t)
            f_lab.append(ls2)

    tr_txt, te_txt, tr_lab, te_lab = train_test_split(
        f_txt, f_lab, test_size=0.15, random_state=42)
    _save_multi_hot_csv(tr_txt, tr_lab, keep_labels, train_csv)
    _save_multi_hot_csv(te_txt, te_lab, keep_labels, test_csv)
    log.info("HUPD prepared: %d train / %d test × %d IPC-subclass labels",
             len(tr_txt), len(te_txt), len(keep_labels))
    return train_csv, test_csv


# ── Prepare: Goodreads book genres ────────────────────────────────────────────

GOODREADS_STOP_WORDS = {
    # book-blurb boilerplate
    "book", "story", "novel", "life", "world", "new", "york", "times",
    "bestseller", "author", "reader", "readers", "series", "edition",
    "one", "two", "will", "must", "young", "old", "years", "year",
}

_GOODREADS_HF_REPO = "pszemraj/goodreads-bookgenres"
_GOODREADS_GENRES = [
    "History & Politics", "Health & Medicine", "Mystery & Thriller",
    "Arts & Design", "Self-Help & Wellness", "Sports & Recreation",
    "Non-Fiction", "Science Fiction & Fantasy", "Countries & Geography",
    "Other", "Nature & Environment", "Business & Finance", "Romance",
    "Philosophy & Religion", "Literature & Fiction", "Science & Technology",
    "Children & Young Adult", "Food & Cooking",
]


def prepare_goodreads(data_dir: Path) -> Tuple[Path, Path]:
    """Goodreads book genres — book title+description → 18 aggregated genres.

    Source: HF dataset ``pszemraj/goodreads-bookgenres`` (``data/`` config:
    ``Book`` + ``Description`` + an 18-dim multi-hot ``Genres`` vector). Train and
    validation splits are merged into train. Needs ``pyarrow`` at prep time
    (parquet); the produced CSVs are plain and need no extra deps.
    """
    processed_dir = data_dir / "processed"
    train_csv = processed_dir / "train.csv"
    test_csv = processed_dir / "test.csv"
    if train_csv.exists() and test_csv.exists():
        log.info("Goodreads already prepared at %s", processed_dir)
        return train_csv, test_csv

    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required to fetch Goodreads") from exc

    api = HfApi()
    files = [s.rfilename for s in (api.dataset_info(_GOODREADS_HF_REPO).siblings or [])]

    def _pick(split: str) -> str:
        for f in files:
            if f.startswith(f"data/{split}") and f.endswith(".parquet"):
                return f
        raise RuntimeError(f"Goodreads: no data/{split} parquet found")

    parts: Dict[str, "pd.DataFrame"] = {}
    for split in ("train", "validation", "test"):
        got = hf_hub_download(_GOODREADS_HF_REPO, _pick(split),
                              repo_type="dataset", local_dir=str(raw_dir))
        parts[split] = pd.read_parquet(got)

    def _s(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and np.isnan(v):
            return ""
        return str(v).strip()

    def _to_rows(df):
        texts, labels_list = [], []
        for _, row in df.iterrows():
            title = _s(row.get("Book"))
            desc = _s(row.get("Description"))
            text = (title + ". " + desc).strip() if title else desc
            g = row.get("Genres")
            if g is None:
                continue
            g = list(g)
            labs = [_GOODREADS_GENRES[i] for i in range(min(len(g), len(_GOODREADS_GENRES)))
                    if g[i]]
            if text and labs:
                texts.append(text)
                labels_list.append(labs)
        return texts, labels_list

    tr_txt, tr_lab = _to_rows(pd.concat([parts["train"], parts["validation"]], ignore_index=True))
    te_txt, te_lab = _to_rows(parts["test"])
    _save_multi_hot_csv(tr_txt, tr_lab, _GOODREADS_GENRES, train_csv)
    _save_multi_hot_csv(te_txt, te_lab, _GOODREADS_GENRES, test_csv)
    log.info("Goodreads prepared: %d train / %d test × %d genres",
             len(tr_txt), len(te_txt), len(_GOODREADS_GENRES))
    return train_csv, test_csv


# ── Dataset config registry ───────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    name: str
    display_name: str
    data_dir: Path
    default_top_labels: int
    stop_words: set
    prepare_fn: Callable


def _full_prepare_stub(ds_name: str) -> Callable:
    """Registry prepare_fn for the *_full datasets — these are prepared out-of-band
    by ``python -m loris.data.prepare_full`` (large downloads), not via load_data."""
    def _fn(data_dir: Path):
        raise RuntimeError(
            f"Full dataset '{ds_name}' is prepared separately (large download):\n"
            f"  python -m loris.data.prepare_full --dataset {ds_name}")
    return _fn


def _build_dataset_registry() -> Dict[str, DatasetConfig]:
    return {
        "aapd": DatasetConfig(
            name="aapd",
            display_name="AAPD (ArXiv Academic Papers)",
            data_dir=_ROOT / "data" / "aapd",
            default_top_labels=54,
            stop_words=AAPD_STOP_WORDS,
            prepare_fn=prepare_aapd,
        ),
        "eurlex": DatasetConfig(
            name="eurlex",
            display_name="EUR-Lex (EU Legal Documents)",
            data_dir=_ROOT / "data" / "eurlex",
            default_top_labels=50,
            stop_words=EURLEX_STOP_WORDS,
            prepare_fn=prepare_eurlex,
        ),
        "rcv1": DatasetConfig(
            name="rcv1",
            display_name="RCV1-v2 (Reuters Newswire, pseudo-text)",
            data_dir=_ROOT / "data" / "rcv1",
            default_top_labels=50,
            stop_words=RCV1_STOP_WORDS,
            prepare_fn=prepare_rcv1,
        ),
        "bgc": DatasetConfig(
            name="bgc",
            display_name="BGC (Blurb Genre Collection)",
            data_dir=_ROOT / "data" / "bgc",
            default_top_labels=32,
            stop_words=BGC_STOP_WORDS,
            prepare_fn=prepare_bgc,
        ),
        "reuters21578": DatasetConfig(
            name="reuters21578",
            display_name="Reuters-21578 (Newswire Topic Classification)",
            data_dir=_ROOT / "data" / "reuters21578",
            default_top_labels=30,
            stop_words=REUTERS21578_STOP_WORDS,
            prepare_fn=prepare_reuters21578,
        ),
        # arXiv cs-papers sample — processed CSVs ship out-of-band under
        # data/arxiv/processed/. No prepare_fn (data is pre-built); reuse AAPD
        # stop words (both are CS-paper abstracts). Added for the baseline suite
        # (the upstream registry omitted it though data/arxiv exists).
        "arxiv": DatasetConfig(
            name="arxiv",
            display_name="arXiv (CS Papers, multi-label)",
            data_dir=_ROOT / "data" / "arxiv",
            default_top_labels=40,
            stop_words=AAPD_STOP_WORDS,
            prepare_fn=prepare_aapd,
        ),
        # PubMed MeSH — biomedical abstracts → 14 top-level MeSH categories.
        # Fetched from HuggingFace (owaiskha9654/…MeSH) via prepare_pubmed.
        "pubmed": DatasetConfig(
            name="pubmed",
            display_name="PubMed MeSH (Biomedical Abstracts, multi-label)",
            data_dir=_ROOT / "data" / "pubmed",
            default_top_labels=14,
            stop_words=PUBMED_STOP_WORDS,
            prepare_fn=prepare_pubmed,
        ),
        # HUPD patents (Jan-2016 sample) → IPC subclass multi-label.
        "hupd": DatasetConfig(
            name="hupd",
            display_name="HUPD (USPTO Patents, IPC subclasses)",
            data_dir=_ROOT / "data" / "hupd",
            default_top_labels=30,
            stop_words=HUPD_STOP_WORDS,
            prepare_fn=prepare_hupd,
        ),
        # Goodreads book blurbs → 18 aggregated genres.
        "goodreads": DatasetConfig(
            name="goodreads",
            display_name="Goodreads (Book Blurbs, genres)",
            data_dir=_ROOT / "data" / "goodreads",
            default_top_labels=18,
            stop_words=GOODREADS_STOP_WORDS,
            prepare_fn=prepare_goodreads,
        ),
        # ── Full / million-scale variants (prepared via loris.data.prepare_full) ──
        "arxiv_full": DatasetConfig(
            name="arxiv_full", display_name="arXiv (FULL metadata, ~2.7M)",
            data_dir=_ROOT / "data" / "arxiv_full", default_top_labels=100,
            stop_words=AAPD_STOP_WORDS, prepare_fn=_full_prepare_stub("arxiv_full")),
        "hupd_full": DatasetConfig(
            name="hupd_full", display_name="HUPD (FULL, all years, ~4.5M patents)",
            data_dir=_ROOT / "data" / "hupd_full", default_top_labels=100,
            stop_words=HUPD_STOP_WORDS, prepare_fn=_full_prepare_stub("hupd_full")),
        "pubmed_full": DatasetConfig(
            name="pubmed_full", display_name="PubMed MeSH (FULL, Tellurio)",
            data_dir=_ROOT / "data" / "pubmed_full", default_top_labels=14,
            stop_words=PUBMED_STOP_WORDS, prepare_fn=_full_prepare_stub("pubmed_full")),
        "goodreads_full": DatasetConfig(
            name="goodreads_full", display_name="Goodreads (FULL, ~2.3M books)",
            data_dir=_ROOT / "data" / "goodreads_full", default_top_labels=20,
            stop_words=GOODREADS_STOP_WORDS, prepare_fn=_full_prepare_stub("goodreads_full")),
    }


DATASET_REGISTRY = _build_dataset_registry()
