"""
pattern_extraction/extract.py
-------------------------------
Convenience function and CLI for running the LORIS pattern abstraction
pipeline on a real dataset file.

Function interface
------------------
::

    from pattern_extraction import extract_patterns, PatternStore

    store = extract_patterns(
        texts=train_texts,
        labels=train_labels,
        output_path="results/patterns.json",
        n_clusters=8,
    )

CLI interface
-------------
::

    python pattern_extraction/extract.py \\
        --input  data/train.csv \\
        --text_col  text \\
        --label_col label \\
        --label_format single \\
        --output results/patterns.json \\
        --n_clusters 8

Run ``python pattern_extraction/extract.py --help`` for full usage.

Supported input formats
-----------------------
* **CSV / TSV** — detected by extension (``.csv`` / ``.tsv``).
* **JSON** — a list of objects, one per document.
* **JSONL** — one JSON object per line.

Label formats
-------------
* ``single``   — one string (or integer) label per document in a single column.
* ``multihot`` — the column contains a JSON-encoded list or a
  comma/space-separated string of class indices or names, which is parsed
  into a multi-hot matrix via ``sklearn.preprocessing.MultiLabelBinarizer``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from pattern_extraction.document import Document
from pattern_extraction.pattern_abstractor import PatternAbstractor
from pattern_extraction.pattern_store import PatternStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public function interface
# ---------------------------------------------------------------------------

def extract_patterns(
    texts: Union[List[str], List[Document]],
    labels: Union[List[str], np.ndarray],
    output_path: Optional[str] = None,
    regex_config_path: Optional[str] = None,
    **abstractor_kwargs: Any,
) -> PatternStore:
    """
    Fit the LORIS pattern abstraction pipeline and return a
    :class:`~pattern_extraction.pattern_store.PatternStore`.

    This is a one-shot convenience wrapper around
    :class:`~pattern_extraction.pattern_abstractor.PatternAbstractor`.

    Parameters
    ----------
    texts : List[str] or List[Document]
        Training corpus.  Plain strings are auto-wrapped.
    labels : List[str] or np.ndarray
        Single-label (1-D) or multi-hot (2-D) labels.
    output_path : str or None, optional
        If given, the fitted :class:`PatternStore` is saved to this path
        (JSON).
    regex_config_path : str or None, optional
        Path to a custom entity-regex YAML/JSON config.  ``None`` uses the
        bundled default.
    **abstractor_kwargs
        Extra keyword arguments forwarded to
        :class:`PatternAbstractor.__init__`, e.g.
        ``n_clusters=8``, ``max_entropy_threshold=0.8``.

    Returns
    -------
    PatternStore
    """
    abstractor = PatternAbstractor(
        regex_config_path=regex_config_path,
        **abstractor_kwargs,
    )
    abstractor.fit(texts, labels)
    store = abstractor.to_store()

    if output_path is not None:
        store.save(output_path)
        logger.info("PatternStore saved to %s.", output_path)

    return store


# ---------------------------------------------------------------------------
# Dataset loading helpers (CLI only)
# ---------------------------------------------------------------------------

def _load_dataset(
    input_path: str,
    text_col: str,
    label_col: str,
    label_format: str,
) -> tuple:
    """
    Load a dataset from file and return ``(texts, labels)``.

    Parameters
    ----------
    input_path : str
    text_col : str
    label_col : str
    label_format : str  — ``"single"`` or ``"multihot"``

    Returns
    -------
    (List[str], Union[List[str], np.ndarray])
    """
    path = Path(input_path)
    suffix = path.suffix.lower()

    # -- Parse file into a list of row dicts --------------------------------
    rows: List[Dict[str, Any]] = []

    if suffix in (".csv", ".tsv"):
        try:
            import pandas as pd  # type: ignore[import]
        except ImportError as err:
            raise ImportError(
                "pandas is required to load CSV/TSV files.\n"
                "Install with:  pip install pandas"
            ) from err
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        rows = df.to_dict(orient="records")

    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            rows = data
        else:
            raise ValueError(
                f"JSON file must contain a list of objects, got {type(data).__name__}."
            )

    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]

    else:
        raise ValueError(
            f"Unsupported file extension {suffix!r}. "
            "Expected .csv, .tsv, .json, or .jsonl."
        )

    if not rows:
        raise ValueError(f"Input file {input_path!r} contains no rows.")

    # -- Extract text and label columns ------------------------------------
    texts: List[str] = []
    raw_labels: List[Any] = []

    for i, row in enumerate(rows):
        if text_col not in row:
            raise KeyError(
                f"Row {i} missing column {text_col!r}.  "
                f"Available columns: {list(row.keys())}"
            )
        if label_col not in row:
            raise KeyError(
                f"Row {i} missing column {label_col!r}.  "
                f"Available columns: {list(row.keys())}"
            )
        texts.append(str(row[text_col]))
        raw_labels.append(row[label_col])

    # -- Label parsing -----------------------------------------------------
    if label_format == "single":
        # Plain string or integer labels — return as list
        labels_out: Union[List[str], np.ndarray] = raw_labels
        return texts, labels_out

    # multihot: each entry is a JSON list, or comma/space-separated string
    label_lists: List[List[str]] = []
    for entry in raw_labels:
        if isinstance(entry, list):
            label_lists.append([str(x) for x in entry])
        elif isinstance(entry, str):
            entry = entry.strip()
            if entry.startswith("["):
                label_lists.append([str(x) for x in json.loads(entry)])
            else:
                # comma or space separated
                import re
                parts = re.split(r"[,\s]+", entry)
                label_lists.append([p for p in parts if p])
        else:
            label_lists.append([str(entry)])

    try:
        from sklearn.preprocessing import MultiLabelBinarizer
    except ImportError as err:
        raise ImportError(
            "scikit-learn is required for multi-hot label parsing.\n"
            "Install with:  pip install scikit-learn"
        ) from err

    mlb = MultiLabelBinarizer()
    matrix = mlb.fit_transform(label_lists).astype(np.int32)
    return texts, matrix


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python pattern_extraction/extract.py",
        description=(
            "LORIS pattern abstraction — extract TextualPredicates from a "
            "labelled document dataset and save them as a PatternStore JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Input / output -----------------------------------------------------
    p.add_argument(
        "--input", required=True,
        help="Path to input file (.csv, .tsv, .json, .jsonl).",
    )
    p.add_argument(
        "--text_col", required=True,
        help="Column name containing document text.",
    )
    p.add_argument(
        "--label_col", required=True,
        help="Column name containing document labels.",
    )
    p.add_argument(
        "--label_format", default="single",
        choices=["single", "multihot"],
        help=(
            "Label format.  'single': one label per doc.  "
            "'multihot': JSON list or comma-separated class names/indices."
        ),
    )
    p.add_argument(
        "--output", default=None,
        help="Path to save the PatternStore JSON.  Omit to skip saving.",
    )
    p.add_argument(
        "--regex_config", default=None,
        dest="regex_config_path",
        help=(
            "Path to a custom entity-regex YAML/JSON config file.  "
            "Omit to use the bundled default (regex_patterns.yaml)."
        ),
    )

    # -- PatternAbstractor params -------------------------------------------
    p.add_argument(
        "--n_clusters", type=int, default=None,
        help="Number of k-means clusters.  None = auto-select via silhouette.",
    )
    p.add_argument(
        "--max_auto_k", type=int, default=15,
        help="Upper bound for automatic k search.",
    )
    p.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformers model identifier.",
    )
    p.add_argument(
        "--tfidf_top_k", type=int, default=20,
        help="Number of top TF-IDF anchors per cluster.",
    )
    p.add_argument(
        "--spacy_model", default="en_core_web_sm",
        help="spaCy model for noun chunks / verb lemmas.",
    )
    p.add_argument(
        "--min_coverage", type=float, default=0.02,
        help="Minimum fraction of docs a predicate must match.",
    )
    p.add_argument(
        "--max_coverage", type=float, default=0.90,
        help="Maximum fraction of docs a predicate may match.",
    )
    p.add_argument(
        "--max_entropy", type=float, default=1.0,
        dest="max_entropy_threshold",
        help="Max label-distribution entropy (nats) for a predicate to survive.",
    )
    p.add_argument(
        "--max_pairs", type=int, default=500,
        dest="max_pairs_per_cluster",
        help="Max anchor pairs per cluster.",
    )
    p.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed for k-means.",
    )

    # -- Auto-regex params --------------------------------------------------
    p.add_argument(
        "--auto_regex", action="store_true", default=False,
        help=(
            "Enable automatic discovery of high-frequency structural "
            "token patterns from the corpus."
        ),
    )
    p.add_argument(
        "--auto_regex_min_freq", type=int, default=3,
        help=(
            "Minimum corpus-wide frequency for a token shape to become "
            "an auto-discovered regex (only used with --auto_regex)."
        ),
    )

    return p


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Load dataset
    logger.info("Loading dataset from %s …", args.input)
    texts, labels = _load_dataset(
        input_path=args.input,
        text_col=args.text_col,
        label_col=args.label_col,
        label_format=args.label_format,
    )
    logger.info("Loaded %d documents.", len(texts))

    # Build kwargs for PatternAbstractor (exclude non-abstractor fields)
    abstractor_kwargs = {
        "n_clusters":            args.n_clusters,
        "max_auto_k":            args.max_auto_k,
        "embedding_model":       args.embedding_model,
        "tfidf_top_k":           args.tfidf_top_k,
        "spacy_model":           args.spacy_model,
        "min_coverage":          args.min_coverage,
        "max_coverage":          args.max_coverage,
        "max_entropy_threshold": args.max_entropy_threshold,
        "max_pairs_per_cluster": args.max_pairs_per_cluster,
        "random_state":          args.random_state,
        "auto_regex":            args.auto_regex,
        "auto_regex_min_freq":   args.auto_regex_min_freq,
    }

    # Run pipeline
    store = extract_patterns(
        texts=texts,
        labels=labels,
        output_path=args.output,
        regex_config_path=args.regex_config_path,
        **abstractor_kwargs,
    )

    # Summary
    print(f"\n{store}")
    print(f"\nTop-10 predicates by type:")
    from collections import Counter
    type_counts: Counter = Counter(type(p).__name__ for p in store)
    for ptype, cnt in type_counts.most_common():
        print(f"  {ptype}: {cnt}")

    sample = list(store)[:5]
    if sample:
        print("\nSample predicates:")
        for pred in sample:
            print(f"  {pred}")

    if args.output:
        print(f"\nPatternStore saved to: {args.output}")
    else:
        print(
            "\nTip: pass --output <path> to save the PatternStore for reuse."
        )


if __name__ == "__main__":
    main()
