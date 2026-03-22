"""
run.py
------
Universal training and evaluation runner for the Loris MLTC model pool.

Accepts real data files in common formats, parses labels in common encoding
styles, trains any of the four model paradigms via hyperparameter flags, and
saves predictions / probabilities / metrics to disk.

Data file formats
~~~~~~~~~~~~~~~~~
* ``csv`` / ``tsv``  — structured table; text column + label column(s)
* ``json``           — JSON array of dicts, e.g. ``[{"text": ..., "labels": ...}]``
* ``jsonl``          — one JSON dict per line
* ``txt``            — plain text (one doc per line) + separate ``--label_file``

Format auto-detected from file extension when ``--data_format`` is omitted.

Label formats (``--label_format``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``multihot_cols``  — one binary column per label (e.g. Reuters processed CSV)
* ``onehot_str``     — single column: ``"0 1 0 0 1 0"`` or ``"[0,1,0,0,1,0]"``
* ``names_list``     — single column: ``"['sports','science']"`` or ``"sports,science"``
* ``names_str``      — single column: ``"sports science"`` or ``"sports,science"``
* ``indices_list``   — single column: ``"[0, 2]"`` or ``"0 2"``
* ``single_name``    — single-class label name  (converted to multi-hot)
* ``single_index``   — single-class integer index (requires ``--num_labels``)

Usage examples
~~~~~~~~~~~~~~
    # Reuters 21578 (multi-hot CSV, 114 labels)
    python run.py \\
        --train data/reuters21578/processed_reuters/train.csv \\
        --test  data/reuters21578/processed_reuters/test.csv \\
        --text_col text --label_format multihot_cols \\
        --model tfidf --clf_type svm \\
        --output_dir results/reuters_tfidf_svm

    # JSONL with label name list, neural model, save probabilities
    python run.py \\
        --train data/train.jsonl --test data/test.jsonl \\
        --text_col text --label_col labels \\
        --label_format names_list \\
        --label_names sports science politics technology health finance \\
        --model neural --variant bilstm --head_type cosine \\
        --output_dir results/bilstm_cosine --save_proba

    # Config file (YAML or JSON), CLI args override
    python run.py --config configs/encoder.yaml --output_dir results/override/

Output files (in ``--output_dir``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``metrics.json``      — Micro-F1, Macro-F1, Subset Accuracy
* ``predictions.csv``   — binary multi-hot predictions, columns = label names
* ``probas.npy``        — float32 (n_test, num_labels) — only with ``--save_proba``
* ``config.json``       — full resolved hyperparameter config
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("run")


# ---------------------------------------------------------------------------
# Config file loading
# ---------------------------------------------------------------------------

def _load_config_file(path: str) -> dict:
    """Load a YAML or JSON config file into a plain dict."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(text) or {}
        except ImportError:
            raise ImportError(
                "PyYAML is required to load .yaml config files. "
                "Install with: pip install pyyaml"
            )
    # JSON fallback
    return json.loads(text)


# ---------------------------------------------------------------------------
# Data loading — file format parsers
# ---------------------------------------------------------------------------

def _detect_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".csv": "csv",
        ".tsv": "tsv",
        ".json": "json",
        ".jsonl": "jsonl",
        ".txt": "txt",
    }.get(ext, "csv")


def _load_csv_tsv(
    path: str,
    sep: str,
    text_col: str,
    label_format: str,
    label_col: Optional[str],
) -> Tuple[List[str], List[dict]]:
    """Return (texts, raw_label_rows) for CSV/TSV.

    raw_label_rows: for multihot_cols → full row dicts; else list of single values.
    """
    import csv

    texts: List[str] = []
    raw: List[dict] = []

    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=sep)
        for row in reader:
            if text_col not in row:
                raise KeyError(
                    f"text column '{text_col}' not found. "
                    f"Available columns: {list(row.keys())}"
                )
            texts.append(row[text_col])
            raw.append(row)

    return texts, raw


def _load_json(path: str, text_col: str) -> Tuple[List[str], List[dict]]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a top-level array of objects.")
    texts = [str(d[text_col]) for d in data]
    return texts, data


def _load_jsonl(path: str, text_col: str) -> Tuple[List[str], List[dict]]:
    texts: List[str] = []
    rows: List[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}")
            texts.append(str(d[text_col]))
            rows.append(d)
    return texts, rows


def _load_txt(
    path: str,
    label_file: str,
) -> Tuple[List[str], List[dict]]:
    """Load parallel text + label files (one entry per line)."""
    with open(path, encoding="utf-8") as fh:
        texts = [l.rstrip("\n") for l in fh if l.strip()]
    with open(label_file, encoding="utf-8") as fh:
        label_lines = [l.rstrip("\n") for l in fh if l.strip()]
    if len(texts) != len(label_lines):
        raise ValueError(
            f"Text file has {len(texts)} lines but label file has "
            f"{len(label_lines)} lines."
        )
    rows = [{"_label": v} for v in label_lines]
    return texts, rows


def load_file(
    path: str,
    fmt: Optional[str],
    text_col: str,
    label_format: str,
    label_col: Optional[str],
    label_file: Optional[str] = None,
) -> Tuple[List[str], List[dict]]:
    """Load a data file and return (texts, raw_rows)."""
    fmt = fmt or _detect_format(path)
    if fmt == "csv":
        return _load_csv_tsv(path, ",", text_col, label_format, label_col)
    if fmt == "tsv":
        return _load_csv_tsv(path, "\t", text_col, label_format, label_col)
    if fmt == "json":
        return _load_json(path, text_col)
    if fmt == "jsonl":
        return _load_jsonl(path, text_col)
    if fmt == "txt":
        if not label_file:
            raise ValueError("--label_file is required when --data_format txt")
        return _load_txt(path, label_file)
    raise ValueError(f"Unknown data format: '{fmt}'")


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def _parse_onehot_str(value: str) -> List[int]:
    """Parse '0 1 0 1' or '[0,1,0,1]' → list of ints."""
    value = value.strip().strip("[]")
    return [int(x) for x in re.split(r"[\s,]+", value) if x]


def _parse_name_sequence(value: str) -> List[str]:
    """Parse list-like strings: "['a','b']", "a,b", "a b"."""
    value = value.strip()
    # Try Python literal first (handles "['a', 'b']" style)
    if value.startswith("["):
        try:
            result = ast.literal_eval(value)
            if isinstance(result, list):
                return [str(s).strip() for s in result]
        except (ValueError, SyntaxError):
            pass
    # Comma-separated
    if "," in value:
        return [s.strip().strip("'\"") for s in value.split(",") if s.strip()]
    # Space-separated
    return [s.strip().strip("'\"") for s in value.split() if s.strip()]


def _parse_index_sequence(value: str) -> List[int]:
    """Parse '[0, 2]' or '0 2' → list of ints."""
    value = value.strip().strip("[]")
    return [int(x) for x in re.split(r"[\s,]+", value) if x]


def _names_to_multihot(names: List[str], label2idx: Dict[str, int], num_labels: int) -> np.ndarray:
    vec = np.zeros(num_labels, dtype=np.float32)
    for n in names:
        if n in label2idx:
            vec[label2idx[n]] = 1.0
    return vec


def extract_labels(
    rows: List[dict],
    label_format: str,
    label_col: Optional[str],
    label_names: List[str],
    text_col: str,
) -> np.ndarray:
    """
    Convert raw_rows → float32 multi-hot matrix (n, num_labels).

    Parameters
    ----------
    rows : List[dict]
        Raw row dicts as returned by load_file().
    label_format : str
        One of the seven supported label formats.
    label_col : Optional[str]
        Column name for the label value (not used for multihot_cols).
    label_names : List[str]
        Ordered list of label names.  Required for name-based formats.
    text_col : str
        Text column name (excluded from multihot_cols scan).

    Returns
    -------
    np.ndarray
        Shape ``(n, len(label_names))``, dtype ``float32``.
    """
    num_labels = len(label_names)
    label2idx = {n: i for i, n in enumerate(label_names)}
    n = len(rows)
    Y = np.zeros((n, num_labels), dtype=np.float32)

    if label_format == "multihot_cols":
        for i, row in enumerate(rows):
            for j, name in enumerate(label_names):
                val = str(row.get(name, "0")).strip()
                Y[i, j] = 1.0 if val in ("1", "1.0", "True", "true") else 0.0
        return Y

    # All remaining formats use a single label column
    col = label_col or "_label"

    for i, row in enumerate(rows):
        raw_val = str(row.get(col, "")).strip()

        if label_format == "onehot_str":
            bits = _parse_onehot_str(raw_val)
            if len(bits) != num_labels:
                raise ValueError(
                    f"Row {i}: onehot_str has {len(bits)} bits but "
                    f"num_labels={num_labels}"
                )
            Y[i] = np.array(bits, dtype=np.float32)

        elif label_format == "names_list":
            names = _parse_name_sequence(raw_val)
            Y[i] = _names_to_multihot(names, label2idx, num_labels)

        elif label_format == "names_str":
            names = _parse_name_sequence(raw_val)
            Y[i] = _names_to_multihot(names, label2idx, num_labels)

        elif label_format == "indices_list":
            idxs = _parse_index_sequence(raw_val)
            for idx in idxs:
                if 0 <= idx < num_labels:
                    Y[i, idx] = 1.0

        elif label_format == "single_name":
            name = raw_val.strip().strip("'\"")
            if name in label2idx:
                Y[i, label2idx[name]] = 1.0

        elif label_format == "single_index":
            idx = int(raw_val)
            if 0 <= idx < num_labels:
                Y[i, idx] = 1.0

        else:
            raise ValueError(f"Unknown label_format: '{label_format}'")

    return Y


def infer_label_names(
    rows: List[dict],
    label_format: str,
    label_col: Optional[str],
    text_col: str,
    provided_names: Optional[List[str]],
    num_labels_hint: Optional[int],
) -> List[str]:
    """
    Determine the ordered list of label names.

    For ``multihot_cols``: derived from column headers (all except text_col).
    For name-based formats: ``provided_names`` is required.
    For index/onehot formats: ``provided_names`` if given, else auto-generated.
    """
    if provided_names:
        return provided_names

    if label_format == "multihot_cols":
        # Infer from the first row's keys, excluding the text column
        if not rows:
            raise ValueError("No rows found in data file.")
        return [k for k in rows[0].keys() if k != text_col]

    if label_format in ("names_list", "names_str", "single_name"):
        raise ValueError(
            f"--label_names is required for label_format='{label_format}'"
        )

    if label_format in ("onehot_str", "indices_list", "single_index"):
        if num_labels_hint:
            return [f"label_{i}" for i in range(num_labels_hint)]
        # Try to infer from first row
        col = label_col or "_label"
        if rows:
            raw = str(rows[0].get(col, "")).strip()
            if label_format == "onehot_str":
                n = len(_parse_onehot_str(raw))
                return [f"label_{i}" for i in range(n)]
        raise ValueError(
            "--label_names or --num_labels required when "
            f"label_format='{label_format}'"
        )

    raise ValueError(f"Unknown label_format: '{label_format}'")


# ---------------------------------------------------------------------------
# Val split auto-creation
# ---------------------------------------------------------------------------

def auto_val_split(
    texts: List[str],
    Y: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """Split texts/Y into train+val using a random stratified-ish split."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(texts))
    # Use first label column as stratification proxy (if imbalanced enough)
    try:
        idx_train, idx_val = train_test_split(
            idx, test_size=val_ratio, random_state=seed, stratify=Y[:, 0]
        )
    except ValueError:
        idx_train, idx_val = train_test_split(
            idx, test_size=val_ratio, random_state=seed
        )

    X_tr = [texts[i] for i in idx_train]
    X_val = [texts[i] for i in idx_val]
    return X_tr, Y[idx_train], X_val, Y[idx_val]


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_classifier(args: argparse.Namespace, num_labels: int):
    """Instantiate and return the selected classifier."""
    model = args.model.lower()

    if model == "tfidf":
        from models import TFIDFClassifier
        return TFIDFClassifier(
            num_labels=num_labels,
            classifier_type=args.clf_type,
            max_features=args.max_features,
            ngram_range=tuple(args.ngram_range),
            min_df=args.min_df,
            C=args.C,
            use_calibration=args.use_calibration,
        )

    if model == "neural":
        from models import NeuralClassifier
        return NeuralClassifier(
            num_labels=num_labels,
            variant=args.variant,
            head_type=args.head_type,
            max_vocab_size=args.max_vocab_size,
            min_freq=args.min_freq,
            max_len=args.max_len,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_filters=args.num_filters,
            filter_sizes=args.filter_sizes,
            num_layers=args.num_layers,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dropout=args.dropout,
            threshold=args.threshold,
        )

    if model == "encoder":
        from models import PretrainedEncoderClassifier
        return PretrainedEncoderClassifier(
            num_labels=num_labels,
            model_name=args.model_name,
            classifier_head=args.classifier_head,
            max_length=args.max_length,
            batch_size=args.batch_size,
            pred_batch_size=args.pred_batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            threshold=args.threshold,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            use_amp=args.use_amp,
            patience=args.patience,
            gradient_checkpointing=args.gradient_checkpointing,
            xgb_n_estimators=args.xgb_n_estimators,
            xgb_max_depth=args.xgb_max_depth,
        )

    if model == "lora":
        from models import LoRASLMClassifier
        return LoRASLMClassifier(
            num_labels=num_labels,
            model_name=args.model_name,
            peft_method=args.peft_method,
            max_length=args.max_length,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            num_epochs=args.num_epochs,
            lr=args.lr,
            threshold=args.threshold,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
            use_4bit=args.use_4bit,
        )

    raise ValueError(f"Unknown model: '{model}'. Choose tfidf/neural/encoder/lora.")


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_outputs(
    output_dir: str,
    label_names: List[str],
    preds: np.ndarray,
    probas: Optional[np.ndarray],
    metrics: Dict[str, float],
    resolved_config: dict,
    save_proba: bool,
) -> None:
    """Write all output files to output_dir."""
    import csv

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Metrics
    metrics_path = out / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved → %s", metrics_path)

    # 2. Binary predictions CSV
    preds_path = out / "predictions.csv"
    with open(preds_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(label_names)
        for row in preds.astype(int).tolist():
            writer.writerow(row)
    logger.info("Predictions saved → %s", preds_path)

    # 3. Probabilities (optional)
    if save_proba and probas is not None:
        proba_path = out / "probas.npy"
        np.save(proba_path, probas.astype(np.float32))
        logger.info("Probabilities saved → %s", proba_path)

    # 4. Config snapshot
    cfg_path = out / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(resolved_config, fh, indent=2, default=str)
    logger.info("Config saved → %s", cfg_path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Loris ML predicates — universal training and evaluation runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (lowest priority; CLI args override)
    p.add_argument("--config", default=None,
                   help="Path to a YAML or JSON config file.")

    # ── Data ──────────────────────────────────────────────────────────────
    grp = p.add_argument_group("Data")
    grp.add_argument("--train", default=None, help="Path to training data file.")
    grp.add_argument("--val",   default=None,
                     help="Path to validation data file. "
                          "If omitted, a fraction of --train is used.")
    grp.add_argument("--test",  default=None, help="Path to test data file.")
    grp.add_argument("--label_file", default=None,
                     help="Separate label file for --data_format txt "
                          "(one label per line).")
    grp.add_argument("--data_format",
                     choices=["csv", "tsv", "json", "jsonl", "txt"],
                     default=None,
                     help="Data file format (auto-detected from extension if omitted).")
    grp.add_argument("--text_col", default="text",
                     help="Column / key name for the text field.")
    grp.add_argument("--label_col", default=None,
                     help="Column / key name for labels "
                          "(not needed for multihot_cols).")
    grp.add_argument("--label_format",
                     choices=["multihot_cols", "onehot_str", "names_list",
                               "names_str", "indices_list", "single_name",
                               "single_index"],
                     default="multihot_cols",
                     help="Label encoding scheme.")
    grp.add_argument("--label_names", nargs="*", default=None,
                     help="Ordered list of label names. "
                          "Inferred from column headers for multihot_cols.")
    grp.add_argument("--num_labels", type=int, default=None,
                     help="Number of labels (used when label_names are unknown).")
    grp.add_argument("--val_split_ratio", type=float, default=0.1,
                     help="Fraction of train data used for auto val split.")

    # ── Model selection ────────────────────────────────────────────────────
    grp = p.add_argument_group("Model")
    grp.add_argument("--model",
                     choices=["tfidf", "neural", "encoder", "lora"],
                     default="tfidf",
                     help="Model paradigm.")

    # TFIDFClassifier
    grp = p.add_argument_group("TFIDFClassifier")
    grp.add_argument("--clf_type",
                     choices=["svm", "logistic_regression"], default="svm")
    grp.add_argument("--max_features", type=int, default=50_000)
    grp.add_argument("--ngram_range", type=int, nargs=2, default=[1, 2],
                     metavar=("MIN", "MAX"))
    grp.add_argument("--min_df", type=int, default=2)
    grp.add_argument("--C", type=float, default=1.0,
                     help="Regularisation strength for SVM / LR.")
    grp.add_argument("--use_calibration", action="store_true",
                     help="Platt-scale LinearSVC probabilities (slow).")

    # NeuralClassifier
    grp = p.add_argument_group("NeuralClassifier")
    grp.add_argument("--variant", choices=["textcnn", "bilstm"], default="textcnn")
    grp.add_argument("--head_type", choices=["linear", "cosine"], default="linear")
    grp.add_argument("--max_vocab_size", type=int, default=30_000)
    grp.add_argument("--min_freq", type=int, default=2)
    grp.add_argument("--max_len", type=int, default=256)
    grp.add_argument("--embed_dim", type=int, default=128)
    grp.add_argument("--hidden_dim", type=int, default=256)
    grp.add_argument("--num_filters", type=int, default=128)
    grp.add_argument("--filter_sizes", type=int, nargs="+", default=[3, 4, 5])
    grp.add_argument("--num_layers", type=int, default=2)

    # Shared training hyperparameters (neural / encoder / lora)
    grp = p.add_argument_group("Training (neural / encoder / lora)")
    grp.add_argument("--num_epochs", type=int, default=5)
    grp.add_argument("--batch_size", type=int, default=32)
    grp.add_argument("--lr", type=float, default=1e-3)
    grp.add_argument("--dropout", type=float, default=0.3)
    grp.add_argument("--threshold", type=float, default=0.5)

    # PretrainedEncoderClassifier
    grp = p.add_argument_group("PretrainedEncoderClassifier")
    grp.add_argument("--model_name", default="roberta-base",
                     help="HuggingFace model identifier.")
    grp.add_argument("--classifier_head", choices=["mlp", "xgboost"], default="mlp")
    grp.add_argument("--max_length", type=int, default=512)
    grp.add_argument("--pred_batch_size", type=int, default=32)
    grp.add_argument("--warmup_ratio", type=float, default=0.1)
    grp.add_argument("--weight_decay", type=float, default=0.01)
    grp.add_argument("--use_amp", action="store_true", default=True,
                     help="Enable AMP on CUDA (default True).")
    grp.add_argument("--no_amp", dest="use_amp", action="store_false")
    grp.add_argument("--patience", type=int, default=3,
                     help="Early-stopping patience (encoder / lora).")
    grp.add_argument("--gradient_checkpointing", action="store_true")
    grp.add_argument("--xgb_n_estimators", type=int, default=300)
    grp.add_argument("--xgb_max_depth", type=int, default=6)

    # LoRASLMClassifier
    grp = p.add_argument_group("LoRASLMClassifier")
    grp.add_argument("--peft_method", choices=["lora", "ia3"], default="lora")
    grp.add_argument("--accumulation_steps", type=int, default=8)
    grp.add_argument("--lora_r", type=int, default=16)
    grp.add_argument("--lora_alpha", type=int, default=32)
    grp.add_argument("--lora_dropout", type=float, default=0.05)
    grp.add_argument("--lora_target_modules", nargs="+",
                     default=["q_proj", "v_proj"])
    grp.add_argument("--use_4bit", action="store_true", default=True)
    grp.add_argument("--no_4bit", dest="use_4bit", action="store_false")

    # ── Output ────────────────────────────────────────────────────────────
    grp = p.add_argument_group("Output")
    grp.add_argument("--output_dir", default="results",
                     help="Directory where outputs are written.")
    grp.add_argument("--save_proba", action="store_true",
                     help="Save predict_proba() output as probas.npy.")

    # ── Misc ──────────────────────────────────────────────────────────────
    grp = p.add_argument_group("Misc")
    grp.add_argument("--seed", type=int, default=42)

    return p


def merge_config(args: argparse.Namespace, config_path: Optional[str]) -> argparse.Namespace:
    """Load config file and update args with its values (CLI args take priority)."""
    if not config_path:
        return args
    cfg = _load_config_file(config_path)
    # Only set values that were NOT explicitly provided on the CLI
    # We can't detect this perfectly with argparse, so we apply config first,
    # then re-parse CLI on top.
    parser = build_parser()
    defaults = vars(parser.parse_args([]))
    cli_args = vars(args)
    merged = dict(defaults)
    merged.update(cfg)
    # CLI overrides config: wherever CLI value differs from default, use CLI
    for k, cli_val in cli_args.items():
        if k == "config":
            continue
        if cli_val != defaults.get(k):
            merged[k] = cli_val
    return argparse.Namespace(**merged)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Config file merging
    if args.config:
        args = merge_config(args, args.config)

    # Seed
    np.random.seed(args.seed)

    # ── Validate required paths ────────────────────────────────────────────
    if not args.train:
        parser.error("--train is required.")
    if not args.test:
        parser.error("--test is required.")

    # ── Load train data ────────────────────────────────────────────────────
    logger.info("Loading train data: %s", args.train)
    X_train_raw, rows_train = load_file(
        args.train, args.data_format, args.text_col,
        args.label_format, args.label_col, args.label_file,
    )

    # ── Infer label names ──────────────────────────────────────────────────
    label_names = infer_label_names(
        rows_train, args.label_format, args.label_col,
        args.text_col, args.label_names, args.num_labels,
    )
    num_labels = len(label_names)
    logger.info("Labels (%d): %s", num_labels, label_names[:10])

    # ── Parse train labels ─────────────────────────────────────────────────
    y_train_full = extract_labels(
        rows_train, args.label_format, args.label_col,
        label_names, args.text_col,
    )

    # ── Val split ─────────────────────────────────────────────────────────
    if args.val:
        logger.info("Loading val data: %s", args.val)
        X_val, rows_val = load_file(
            args.val, args.data_format, args.text_col,
            args.label_format, args.label_col,
        )
        y_val = extract_labels(
            rows_val, args.label_format, args.label_col,
            label_names, args.text_col,
        )
        X_train = X_train_raw
        y_train = y_train_full
    else:
        logger.info(
            "No --val provided. Auto-splitting %.0f%% of train as val.",
            args.val_split_ratio * 100,
        )
        X_train, y_train, X_val, y_val = auto_val_split(
            X_train_raw, y_train_full, args.val_split_ratio, args.seed
        )

    logger.info("Train: %d  Val: %d", len(X_train), len(X_val))

    # ── Load test data ─────────────────────────────────────────────────────
    logger.info("Loading test data: %s", args.test)
    X_test, rows_test = load_file(
        args.test, args.data_format, args.text_col,
        args.label_format, args.label_col,
    )
    y_test = extract_labels(
        rows_test, args.label_format, args.label_col,
        label_names, args.text_col,
    )
    logger.info("Test:  %d", len(X_test))

    # ── Build and train ────────────────────────────────────────────────────
    logger.info("Building model: %s", args.model)
    clf = build_classifier(args, num_labels)

    logger.info("Training …")
    clf.fit(X_train, y_train, X_val, y_val)

    # ── Evaluate ───────────────────────────────────────────────────────────
    logger.info("Evaluating on test set …")
    metrics = clf.evaluate(X_test, y_test)
    logger.info(
        "Test results — Micro-F1: %.4f | Macro-F1: %.4f | Subset-Acc: %.4f",
        metrics["micro_f1"], metrics["macro_f1"], metrics["subset_accuracy"],
    )

    # ── Predict (for saving) ───────────────────────────────────────────────
    preds = clf.predict(X_test)
    probas = clf.predict_proba(X_test) if args.save_proba else None

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  Model  : {args.model}")
    print(f"  Test n : {len(X_test)}")
    print(f"  Labels : {num_labels}")
    print(f"{'=' * 55}")
    print(f"  Micro-F1        : {metrics['micro_f1']:.4f}")
    print(f"  Macro-F1        : {metrics['macro_f1']:.4f}")
    print(f"  Subset Accuracy : {metrics['subset_accuracy']:.4f}")
    print(f"{'=' * 55}\n")

    # ── Save outputs ───────────────────────────────────────────────────────
    resolved_config = {k: v for k, v in vars(args).items()}
    resolved_config["label_names"] = label_names
    resolved_config["num_labels"] = num_labels
    resolved_config["train_size"] = len(X_train)
    resolved_config["val_size"] = len(X_val)
    resolved_config["test_size"] = len(X_test)
    resolved_config.update(metrics)

    save_outputs(
        args.output_dir,
        label_names,
        preds,
        probas,
        metrics,
        resolved_config,
        args.save_proba,
    )

    logger.info("All outputs written to: %s", args.output_dir)


if __name__ == "__main__":
    main()
