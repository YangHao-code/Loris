"""
run_loris_reuters_pipeline.py
=============================
End-to-end LORIS pipeline on Reuters-21578:
  Step 1 · Pattern Abstraction  (PatternAbstractor)
  Step 2 · Dynamic Router       (SelectionNetwork, stochastic Top-K)
  Step 3 · Rule Discovery       (RuleLearner, Bayesian optimisation)

Features
--------
* Diverse model pool: TFIDFClassifier × 3 + NeuralClassifier × 2
  + PretrainedEncoderClassifier (if local HF cache found)
  + LoRASLMClassifier (if --lora_model specified and VRAM ≥ 20 GB)
* Auto-tuning loop: up to MAX_RETRIES=3 retries on quality failure.
* Diagnostic report on permanent failure → sys.exit(1).

Usage
-----
  cd /root/autodl-tmp/Loris
  python run_loris_reuters_pipeline.py [options]

  --subset_size 2000   # 0 = all data
  --top_labels  20     # restrict to N most-frequent labels
  --debug              # verbose Optuna + full tracebacks
  --no_router          # skip neural router, use val-F1 ranking instead
  --lora_model <name>  # HF model name for LoRASLMClassifier (optional)
  --exp_dir <path>     # override experiment output directory
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── make project root importable ──────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

# ── offline mode (use locally cached HF models; no network required) ──────────
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── sklearn helpers ────────────────────────────────────────────────────────────
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ── LORIS pattern extraction ───────────────────────────────────────────────────
from pattern_extraction import (
    Document,
    PatternAbstractor,
    PatternStore,
    register_ml_model,
)

# ── LORIS classifiers (direct import – models/ has no __init__.py) ────────────
from models.tfidf_classifier import TFIDFClassifier
from models.neural_classifier import NeuralClassifier

# ── LORIS model selection ──────────────────────────────────────────────────────
from model_selection.dynamic_router import (
    FinalSelector,
    HybridLoss,
    SelectionNetwork,
)

# ── LORIS rule discovery ───────────────────────────────────────────────────────
from rule_discovery import RDLSet, RuleLearner

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

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
        # suppress noisy libraries
        for noisy in ("optuna", "transformers", "sentence_transformers",
                      "torch", "sklearn"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HParams:
    # ── Data ──────────────────────────────────────────────────────────────────
    subset_size: int = 2000          # 0 = use full training set
    val_ratio: float = 0.20
    top_labels: int = 20             # keep N most-frequent labels only

    # ── PatternAbstractor ─────────────────────────────────────────────────────
    n_clusters: Optional[int] = None
    min_coverage: float = 0.02
    max_entropy_threshold: float = 1.0
    tfidf_top_k: int = 20

    # ── Dynamic Router ────────────────────────────────────────────────────────
    router_feat_dim: int = 128
    router_hidden_dim: int = 256
    k_models: int = 3
    router_sigma: float = 0.1
    router_num_samples: int = 500
    router_epochs: int = 30
    router_lr: float = 1e-3

    # ── RuleLearner ───────────────────────────────────────────────────────────
    max_trials: int = 100
    top_n_rules: int = 10
    min_coverage_rule: float = 0.01   # 4 docs out of 400 val; was 0.03

    # ── Quality gates ─────────────────────────────────────────────────────────
    min_rules: int = 1
    min_avg_coverage: float = 0.03
    min_f1_gain: float = 0.005
    max_avg_body_len: int = 8

    # ── Retry control ─────────────────────────────────────────────────────────
    max_retries: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _setup_experiment(args: argparse.Namespace) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = Path(args.exp_dir) if args.exp_dir else _ROOT / "experiments"
    exp_dir = base / f"reuters_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def _hf_model_available(model_name: str) -> bool:
    """Return True iff the HF model is already cached locally."""
    try:
        from huggingface_hub import snapshot_download  # type: ignore
        snapshot_download(model_name, local_files_only=True)
        return True
    except Exception:
        return False


def _gpu_vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 0.0


class _PredictWrapper:
    """Wrap a multi-label BaseDocumentClassifier → predict(text)->str protocol."""

    def __init__(self, clf, label_names: List[str]) -> None:
        self._clf = clf
        self._label_names = label_names

    def predict(self, text: str) -> str:
        proba = self._clf.predict_proba([text])[0]   # (n_labels,)
        return self._label_names[int(np.argmax(proba))]


# ──────────────────────────────────────────────────────────────────────────────
# Step 0 — data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data(
    data_dir: Path,
    hp: HParams,
) -> Tuple[
    List[str], List[str], List[str],          # train_X, val_X, test_X
    np.ndarray, np.ndarray, np.ndarray,       # train_y, val_y, test_y
    List[str],                                # label_names
    List[Document], List[Document],           # train_docs, val_docs
]:
    t0 = time.time()
    log.info("Loading Reuters-21578 …")

    train_csv = data_dir / "processed_reuters" / "train.csv"
    test_csv  = data_dir / "processed_reuters" / "test.csv"

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    # ── identify text column and label columns ────────────────────────────────
    text_col = "text"
    label_cols = [c for c in train_df.columns if c != text_col]

    # ── restrict to top-N labels by frequency ─────────────────────────────────
    label_freq = train_df[label_cols].sum().sort_values(ascending=False)
    top_label_cols = label_freq.index[: hp.top_labels].tolist()
    log.info("Top-%d labels: %s", hp.top_labels, top_label_cols[:10])

    train_df = train_df[[text_col] + top_label_cols].dropna(subset=[text_col])
    test_df  = test_df[[text_col] + top_label_cols].dropna(subset=[text_col])

    # keep only rows with at least one active label
    train_df = train_df[train_df[top_label_cols].sum(axis=1) > 0].reset_index(drop=True)
    test_df  = test_df[test_df[top_label_cols].sum(axis=1) > 0].reset_index(drop=True)

    # ── optional stratified subsample ────────────────────────────────────────
    if hp.subset_size and len(train_df) > hp.subset_size:
        # Use dominant-label as stratum
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
        tr_df, val_df = train_test_split(train_df, test_size=hp.val_ratio, random_state=42)

    train_X   = tr_df[text_col].tolist()
    val_X     = val_df[text_col].tolist()
    test_X    = test_df[text_col].tolist()
    train_y   = tr_df[top_label_cols].values.astype(np.float32)
    val_y     = val_df[top_label_cols].values.astype(np.float32)
    test_y    = test_df[top_label_cols].values.astype(np.float32)
    label_names = top_label_cols

    # ── wrap as Document objects ──────────────────────────────────────────────
    def _to_docs(texts: List[str], labels_mat: np.ndarray) -> List[Document]:
        docs = []
        for txt, lbl_row in zip(texts, labels_mat):
            active = {label_names[i] for i, v in enumerate(lbl_row) if v > 0}
            docs.append(Document(cnt=txt, lbl=active))
        return docs

    train_docs = _to_docs(train_X, train_y)
    val_docs   = _to_docs(val_X, val_y)

    log.info(
        "Data loaded in %.1fs — train=%d  val=%d  test=%d  labels=%d",
        time.time() - t0, len(train_X), len(val_X), len(test_X), len(label_names),
    )
    return (
        train_X, val_X, test_X,
        train_y, val_y, test_y,
        label_names,
        train_docs, val_docs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — model pool initialisation
# ──────────────────────────────────────────────────────────────────────────────

def init_models(
    n_labels: int,
    lora_model_name: Optional[str] = None,
) -> Dict[str, object]:
    """Return OrderedDict name → unfitted classifier."""
    pool: Dict[str, object] = {}

    # ── TF-IDF variants (always included) ─────────────────────────────────────
    pool["tfidf_svm_unigram"] = TFIDFClassifier(
        num_labels=n_labels, classifier_type="svm", C=1.0, ngram_range=(1, 1),
    )
    pool["tfidf_svm_bigram"] = TFIDFClassifier(
        num_labels=n_labels, classifier_type="svm", C=1.0, ngram_range=(1, 2),
    )
    pool["tfidf_lr_bigram"] = TFIDFClassifier(
        num_labels=n_labels, classifier_type="logistic_regression",
        C=1.0, ngram_range=(1, 2),
    )

    # ── Neural variants (always included — fully offline) ─────────────────────
    pool["textcnn"] = NeuralClassifier(
        num_labels=n_labels, variant="textcnn",
        embed_dim=64, num_filters=64, hidden_dim=128, num_epochs=5,
        batch_size=32, max_vocab_size=15_000,
    )
    pool["bilstm"] = NeuralClassifier(
        num_labels=n_labels, variant="bilstm",
        embed_dim=64, hidden_dim=128, num_epochs=5,
        batch_size=32, max_vocab_size=15_000,
    )

    # ── Pretrained Encoder (conditional) ──────────────────────────────────────
    encoder_candidates = [
        "roberta-base",
        "distilbert-base-uncased",
        "bert-base-uncased",
    ]
    for enc_name in encoder_candidates:
        if _hf_model_available(enc_name):
            try:
                from models.pretrained_encoder_classifier import (  # noqa: PLC0415
                    PretrainedEncoderClassifier,
                )
                pool["encoder_mlp"] = PretrainedEncoderClassifier(
                    num_labels=n_labels,
                    model_name=enc_name,
                    classifier_head="mlp",
                    num_epochs=3,
                    batch_size=8,
                    gradient_checkpointing=True,
                    patience=2,
                )
                log.info("Added PretrainedEncoderClassifier (%s).", enc_name)
            except Exception as exc:
                log.warning("Could not add encoder model %s: %s", enc_name, exc)
            break

    # ── LoRA SLM (conditional) ────────────────────────────────────────────────
    vram = _gpu_vram_gb()
    if lora_model_name and vram >= 20:
        try:
            from models.lora_slm_classifier import LoRASLMClassifier  # noqa: PLC0415
            pool["lora_slm"] = LoRASLMClassifier(
                num_labels=n_labels,
                model_name=lora_model_name,
                use_4bit=True,
                lora_r=8,
                lora_alpha=16,
                num_epochs=2,
                batch_size=1,
                accumulation_steps=4,
            )
            log.info("Added LoRASLMClassifier (%s, VRAM=%.1f GB).", lora_model_name, vram)
        except Exception as exc:
            log.warning("Could not add LoRA model: %s", exc)
    elif lora_model_name:
        log.warning(
            "LoRA model requested but VRAM=%.1f GB < 20 GB — skipping.", vram
        )

    log.info("Model pool: %s", list(pool.keys()))
    return pool


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — train all models
# ──────────────────────────────────────────────────────────────────────────────

def train_models(
    pool: Dict[str, object],
    train_X: List[str], train_y: np.ndarray,
    val_X: List[str],   val_y: np.ndarray,
) -> Dict[str, float]:
    """Fit every model and return val micro-F1 per model."""
    val_f1: Dict[str, float] = {}
    for name, clf in pool.items():
        t0 = time.time()
        log.info("Training  %s …", name)
        try:
            clf.fit(train_X, train_y, val_X, val_y)
            metrics = clf.evaluate(val_X, val_y)
            val_f1[name] = float(metrics["micro_f1"])
            log.info(
                "  %-22s  micro-F1=%.4f  macro-F1=%.4f  (%.1fs)",
                name, metrics["micro_f1"], metrics["macro_f1"], time.time() - t0,
            )
        except Exception as exc:
            log.error("  %-22s  FAILED: %s", name, exc, exc_info=True)
            val_f1[name] = 0.0
    return val_f1


# ──────────────────────────────────────────────────────────────────────────────
# Step 3.1 — pattern abstraction
# ──────────────────────────────────────────────────────────────────────────────

def run_pattern_abstraction(
    train_X: List[str],
    train_y: np.ndarray,
    hp: HParams,
    exp_dir: Path,
) -> PatternStore:
    t0 = time.time()
    log.info("=== Step 3.1  Pattern Abstraction ===")

    abstractor = PatternAbstractor(
        n_clusters=hp.n_clusters,
        min_coverage=hp.min_coverage,
        max_entropy_threshold=hp.max_entropy_threshold,
        tfidf_top_k=hp.tfidf_top_k,
        random_state=42,
    )
    abstractor.fit(train_X, train_y)
    store = abstractor.to_store()

    out_path = str(exp_dir / "patterns.json")
    store.save(out_path)

    log.info(
        "Pattern Abstraction done in %.1fs — %d predicates extracted, %d clusters, saved to %s",
        time.time() - t0,
        len(store),
        abstractor.n_clusters_,
        out_path,
    )
    return store


# ──────────────────────────────────────────────────────────────────────────────
# Step 3.2 — dynamic router
# ──────────────────────────────────────────────────────────────────────────────

def _build_multi_label_oracle_mask(
    pool: Dict[str, object],
    texts: List[str],
    y_true: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    For each document, mark the top-k models (by any-label overlap) as oracle.
    Returns float32 array of shape (N, n_models).
    """
    n_docs = len(texts)
    names  = list(pool.keys())
    n_models = len(names)
    scores = np.zeros((n_docs, n_models), dtype=np.float32)

    for j, name in enumerate(names):
        clf = pool[name]
        try:
            proba = clf.predict_proba(texts)          # (N, L)
            preds = (proba >= 0.5).astype(int)        # (N, L)
            # any correct label ↔ row-wise AND then sum
            overlap = (preds & y_true.astype(int)).sum(axis=1).astype(np.float32)
            scores[:, j] = overlap
        except Exception as exc:
            log.debug("oracle mask: model %s failed — %s", name, exc)

    # top-k per row
    oracle = np.zeros((n_docs, n_models), dtype=np.float32)
    eff_k  = min(k, n_models)
    for i in range(n_docs):
        top_idx = np.argpartition(scores[i], -eff_k)[-eff_k:]
        oracle[i, top_idx] = 1.0
    return oracle


def run_dynamic_router(
    pool: Dict[str, object],
    train_X: List[str], train_y: np.ndarray,
    val_X: List[str],   val_y: np.ndarray,
    hp: HParams,
    exp_dir: Path,
    skip_router: bool = False,
) -> List[int]:
    """
    Train the SelectionNetwork and return the indices of the k selected models.
    Falls back to val-F1 ranking when skip_router=True.
    """
    t0 = time.time()
    log.info("=== Step 3.2  Dynamic Router ===")
    model_names = list(pool.keys())
    n_models    = len(model_names)
    k           = min(hp.k_models, n_models)

    # ── fallback: rank by validation F1 ──────────────────────────────────────
    if skip_router or n_models <= k:
        val_f1 = {}
        for name, clf in pool.items():
            try:
                m = clf.evaluate(val_X, val_y)
                val_f1[name] = m["micro_f1"]
            except Exception:
                val_f1[name] = 0.0
        sorted_names = sorted(val_f1, key=val_f1.get, reverse=True)
        selected = [model_names.index(n) for n in sorted_names[:k]]
        log.info("Router skipped — selected by val-F1: %s", [model_names[i] for i in selected])
        return selected

    # ── build TF-IDF + SVD document features ─────────────────────────────────
    log.info("Building document features (TF-IDF + TruncatedSVD) …")
    tfidf_vec = TfidfVectorizer(max_features=20_000, sublinear_tf=True)
    X_sp      = tfidf_vec.fit_transform(train_X)
    svd       = TruncatedSVD(n_components=hp.router_feat_dim, random_state=42)
    X_dense   = svd.fit_transform(X_sp).astype(np.float32)        # (N, 128)
    Xval_dense = svd.transform(
        tfidf_vec.transform(val_X)
    ).astype(np.float32)                                           # (M, 128)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t       = torch.from_numpy(X_dense).to(device)
    Xval_t    = torch.from_numpy(Xval_dense).to(device)

    # ── build oracle masks ────────────────────────────────────────────────────
    log.info("Building oracle masks for %d documents …", len(train_X))
    oracle_np = _build_multi_label_oracle_mask(pool, train_X, train_y, k)
    oracle_t  = torch.from_numpy(oracle_np).to(device)

    # ── instantiate SelectionNetwork ──────────────────────────────────────────
    net = SelectionNetwork(
        input_dim=hp.router_feat_dim,
        hidden_dim=hp.router_hidden_dim,
        n_models=n_models,
        k=k,
        num_samples=hp.router_num_samples,
        sigma=hp.router_sigma,
        device=device,
        backend="custom",
    ).to(device)

    loss_fn   = HybridLoss(lambda_task=0.0, lambda_ent=0.1)
    optimiser = torch.optim.Adam(net.parameters(), lr=hp.router_lr)

    # ── training loop ─────────────────────────────────────────────────────────
    log.info("Training SelectionNetwork for %d epochs …", hp.router_epochs)
    net.train()
    for epoch in range(1, hp.router_epochs + 1):
        _, scores = net(X_t, return_scores=True)     # scores: (N, n_models)
        l_imit = loss_fn.imitation_loss(scores, oracle_t)
        l_ent  = loss_fn.entropy_loss(scores)
        loss   = l_imit - 0.1 * l_ent

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if epoch % 5 == 0 or epoch == hp.router_epochs:
            log.info(
                "  [Router epoch %3d/%d]  L_imit=%.4f  L_ent=%.4f  L_total=%.4f",
                epoch, hp.router_epochs, l_imit.item(), l_ent.item(), loss.item(),
            )

    # ── inference: global frequency-based selection ───────────────────────────
    selected_indices, frequency = FinalSelector.select(net, Xval_t, k)
    selected_names = [model_names[i] for i in selected_indices]

    log.info(
        "Router done in %.1fs — selected models: %s  (frequency: %s)",
        time.time() - t0,
        selected_names,
        {model_names[i]: int(frequency[i].item()) for i in range(n_models)},
    )
    return selected_indices


# ──────────────────────────────────────────────────────────────────────────────
# Register selected models as ML predicates
# ──────────────────────────────────────────────────────────────────────────────

def register_selected_models(
    pool: Dict[str, object],
    selected_indices: List[int],
    label_names: List[str],
) -> List[str]:
    """Wrap & register selected classifiers; return their predicate names."""
    model_names_list = list(pool.keys())
    registered: List[str] = []
    for idx in selected_indices:
        clf        = pool[model_names_list[idx]]
        pred_name  = f"loris_clf_{model_names_list[idx]}"
        wrapper    = _PredictWrapper(clf, label_names)
        register_ml_model(pred_name, wrapper)
        registered.append(pred_name)
        log.info("Registered ML predicate: %s", pred_name)
    return registered


# ──────────────────────────────────────────────────────────────────────────────
# Step 3.3 — rule discovery
# ──────────────────────────────────────────────────────────────────────────────

def _select_top_predicates(
    store: PatternStore,
    val_docs: List[Document],
    val_y: np.ndarray,
    label_names: List[str],
    top_k: int = 100,
) -> list:
    """
    Pre-filter PatternStore to the top-k most discriminative predicates
    based on their best per-label precision on the validation set.

    This keeps the Optuna search space manageable and avoids passing hundreds
    of noisy predicates that individually never fire on the val set.
    """
    candidates = list(store)
    if len(candidates) <= top_k:
        return candidates

    n_labels = len(label_names)
    scores: List[float] = []
    for pred in candidates:
        fires_v = np.array([bool(pred(doc)) for doc in val_docs], dtype=bool)
        n_fires = int(fires_v.sum())
        if n_fires == 0:
            scores.append(0.0)
            continue
        # Best precision for any label (coverage-weighted)
        best_prec = max(
            float((fires_v & (val_y[:, l] > 0)).sum()) / n_fires
            for l in range(n_labels)
        )
        scores.append(best_prec)

    sorted_pairs = sorted(zip(scores, candidates), key=lambda x: -x[0])
    return [pred for _, pred in sorted_pairs[:top_k]]


def run_rule_discovery(
    store: PatternStore,
    registered_model_names: List[str],
    label_names: List[str],
    val_docs: List[Document],
    val_y: np.ndarray,
    hp: HParams,
    exp_dir: Path,
    attempt: int = 0,
) -> RDLSet:
    import optuna  # noqa: PLC0415

    t0 = time.time()
    log.info("=== Step 3.3  Rule Discovery ===")

    # Pre-filter to top-100 most discriminative predicates before search
    filtered_preds = _select_top_predicates(
        store, val_docs, val_y, label_names, top_k=100
    )
    log.info(
        "Candidate predicates=%d→%d (top-100 by val precision)  "
        "ML models=%d  labels=%d  max_trials=%d  top_n=%d",
        len(store), len(filtered_preds), len(registered_model_names),
        len(label_names), hp.max_trials, hp.top_n_rules,
    )

    # suppress Optuna output unless in debug mode
    if not log.isEnabledFor(logging.DEBUG):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    learner = RuleLearner(
        candidate_predicates=filtered_preds,
        candidate_ml_models=registered_model_names,
        label_list=label_names,
        val_docs=val_docs,
        val_labels=val_y,
        max_trials=hp.max_trials,
        top_n=hp.top_n_rules,
        min_coverage=hp.min_coverage_rule,
        seed=42,
        storage_path=str(exp_dir / f"optuna_journal_attempt{attempt}"),
        verbose=log.isEnabledFor(logging.DEBUG),
    )
    rdl_set = learner.discover()

    out_path = str(exp_dir / "rules.json")
    rdl_set.save(out_path)

    log.info(
        "Rule Discovery done in %.1fs — %d rules found, saved to %s",
        time.time() - t0, len(rdl_set.rules), out_path,
    )
    if rdl_set.rules:
        log.info("Top-5 rules by F1-gain:")
        top5 = sorted(rdl_set.rules, key=lambda r: r.score, reverse=True)[:5]
        for i, r in enumerate(top5, 1):
            log.info("  #%d  [%s]  gain=%.4f  cov=%.3f  body=%s",
                     i, r.consequence, r.score, r.coverage, repr(r))
    return rdl_set


# ──────────────────────────────────────────────────────────────────────────────
# Quality assessment & auto-tuning
# ──────────────────────────────────────────────────────────────────────────────

def assess_quality(rdl_set: RDLSet, hp: HParams) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    rules = rdl_set.rules

    if len(rules) < hp.min_rules:
        issues.append("empty_ruleset")

    if rules:
        avg_cov = float(np.mean([r.coverage for r in rules]))
        if avg_cov < hp.min_avg_coverage:
            issues.append("low_coverage")

        avg_body = float(np.mean([len(r.body) for r in rules]))
        if avg_body > hp.max_avg_body_len:
            issues.append("over_complex")

        avg_gain = float(np.mean([r.score for r in rules]))
        if avg_gain < hp.min_f1_gain:
            issues.append("no_f1_gain")

    return (len(issues) == 0), issues


def adjust_hparams(
    hp: HParams,
    issues: List[str],
    attempt: int,
    n_models: int,
) -> None:
    log.warning("Auto-tuning hyperparams (attempt %d) — issues: %s", attempt, issues)
    if "empty_ruleset" in issues:
        hp.min_coverage_rule = max(hp.min_coverage_rule * 0.5, 0.001)
        hp.max_trials        = min(hp.max_trials * 2, 400)
        hp.top_n_rules       = min(hp.top_n_rules + 5, 20)
        log.warning(
            "  ↳ empty_ruleset → min_coverage_rule=%.4f  max_trials=%d  top_n=%d",
            hp.min_coverage_rule, hp.max_trials, hp.top_n_rules,
        )
    if "low_coverage" in issues:
        hp.min_coverage      = max(hp.min_coverage * 0.5, 0.001)
        hp.min_coverage_rule = max(hp.min_coverage_rule * 0.5, 0.001)
        log.warning(
            "  ↳ low_coverage → min_coverage=%.4f  min_coverage_rule=%.4f",
            hp.min_coverage, hp.min_coverage_rule,
        )
    if "over_complex" in issues:
        hp.max_entropy_threshold = max(hp.max_entropy_threshold * 0.7, 0.3)
        hp.tfidf_top_k           = max(hp.tfidf_top_k - 5, 5)
        log.warning(
            "  ↳ over_complex → max_entropy=%.2f  tfidf_top_k=%d",
            hp.max_entropy_threshold, hp.tfidf_top_k,
        )
    if "no_f1_gain" in issues:
        hp.max_trials  = min(hp.max_trials * 2, 400)
        hp.router_sigma *= 1.5
        hp.k_models    = min(hp.k_models + 1, n_models)
        log.warning(
            "  ↳ no_f1_gain → max_trials=%d  router_sigma=%.3f  k_models=%d",
            hp.max_trials, hp.router_sigma, hp.k_models,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

def _rule_to_readable(rule, idx: int) -> str:
    body_str = " ∧ ".join(str(p) for p in rule.body) if rule.body else "(empty body)"
    return (
        f"Rule #{idx:02d}  [{rule.consequence}]  "
        f"F1-gain={rule.score:+.4f}  Coverage={rule.coverage:.2%}\n"
        f"  BODY: {body_str}"
    )


def save_and_print_results(
    rdl_set: RDLSet,
    baseline_micro_f1: float,
    final_micro_f1: float,
    exp_dir: Path,
    hp: HParams,
) -> None:
    # ── metrics.json ──────────────────────────────────────────────────────────
    metrics = {
        "baseline_micro_f1":  baseline_micro_f1,
        "final_micro_f1":     final_micro_f1,
        "f1_delta":           final_micro_f1 - baseline_micro_f1,
        "n_rules":            len(rdl_set.rules),
        "hparams":            hp.to_dict(),
    }
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # ── readable rules.txt ────────────────────────────────────────────────────
    sorted_rules = sorted(rdl_set.rules, key=lambda r: r.score, reverse=True)
    width = 70
    border = "═" * width
    lines = [
        f"╔{border}╗",
        f"║{'  LORIS Rule Discovery — Reuters-21578':^{width}}║",
        f"║{'  Baseline micro-F1: ' + f'{baseline_micro_f1:.4f}':^{width}}║",
        f"║{'  Post-rule micro-F1: ' + f'{final_micro_f1:.4f}':^{width}}║",
        f"╠{border}╣",
    ]
    for i, rule in enumerate(sorted_rules, 1):
        lines.append(f"║{_rule_to_readable(rule, i):<{width}}║")
        lines.append(f"╠{border}╣")
    lines[-1] = f"╚{border}╝"
    report = "\n".join(lines)

    print("\n" + report)
    with open(exp_dir / "rules_readable.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")

    log.info(
        "Results saved to %s  (baseline F1=%.4f → final F1=%.4f  Δ=%+.4f)",
        exp_dir, baseline_micro_f1, final_micro_f1,
        final_micro_f1 - baseline_micro_f1,
    )


def dump_diagnostic_report(
    hp: HParams,
    label_names: List[str],
    val_y: np.ndarray,
    issues_history: List[Tuple[int, List[str]]],
    rdl_set: Optional[RDLSet],
    exp_dir: Path,
) -> None:
    lines = [
        "=" * 72,
        "  LORIS PIPELINE DIAGNOSTIC REPORT",
        "  All retries exhausted — human intervention required.",
        "=" * 72,
        "",
        "### Data stats",
        f"  Labels: {label_names}",
        f"  Label counts per doc (val): min={val_y.sum(axis=1).min():.0f}  "
        f"max={val_y.sum(axis=1).max():.0f}  mean={val_y.sum(axis=1).mean():.2f}",
        f"  Label freq (val): {val_y.sum(axis=0).tolist()}",
        "",
        "### Final hyperparameters",
    ]
    for k, v in hp.to_dict().items():
        lines.append(f"  {k}: {v}")
    lines += [
        "",
        "### Issues per attempt",
    ]
    for attempt, issues in issues_history:
        lines.append(f"  Attempt {attempt}: {issues}")
    lines += [
        "",
        "### Last rule set",
        f"  Rules found: {len(rdl_set.rules) if rdl_set else 'N/A'}",
    ]
    if rdl_set and rdl_set.rules:
        for r in sorted(rdl_set.rules, key=lambda x: x.score, reverse=True)[:5]:
            lines.append(f"  {repr(r)}")
    lines += [
        "",
        "### Recommendations",
        "  1. Increase --subset_size for more training data.",
        "  2. Try fewer --top_labels (e.g. 10) for a simpler labelling task.",
        "  3. Manually lower min_coverage_rule below current value.",
        "  4. Check pattern coverage stats in patterns.json.",
        "  5. Add stronger pre-trained models (encoder/LoRA) to the pool.",
        "=" * 72,
    ]
    report = "\n".join(lines)
    report_path = exp_dir / "DIAGNOSTIC_REPORT.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logging.error(
        "\n%s\nDiagnostic report written to %s", report, report_path
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LORIS end-to-end pipeline on Reuters-21578",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",    default=str(_ROOT / "data" / "reuters21578"))
    p.add_argument("--exp_dir",     default=None, help="Override experiment output dir")
    p.add_argument("--subset_size", type=int, default=2000,
                   help="Max training docs (0=all)")
    p.add_argument("--top_labels",  type=int, default=20,
                   help="Restrict to N most-frequent labels")
    p.add_argument("--no_router",   action="store_true",
                   help="Skip neural router; rank by val-F1 instead")
    p.add_argument("--lora_model",  default=None,
                   help="HF model name for LoRASLMClassifier (requires 20 GB VRAM)")
    p.add_argument("--debug",       action="store_true",
                   help="Enable verbose logging (Optuna, model training, etc.)")
    # fine-grained overrides
    p.add_argument("--max_trials",  type=int, default=100)
    p.add_argument("--top_n_rules", type=int, default=10)
    p.add_argument("--min_f1_gain", type=float, default=0.005,
                   help="Force failure for testing: set to large value e.g. 99.0")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    exp_dir = _setup_experiment(args)
    configure_logging(exp_dir, debug=args.debug)

    log.info("Experiment directory: %s", exp_dir)

    # ── load data once ────────────────────────────────────────────────────────
    hp = HParams(
        subset_size=args.subset_size,
        top_labels=args.top_labels,
        max_trials=args.max_trials,
        top_n_rules=args.top_n_rules,
        min_f1_gain=args.min_f1_gain,
    )

    (
        train_X, val_X, test_X,
        train_y, val_y, test_y,
        label_names,
        train_docs, val_docs,
    ) = load_data(Path(args.data_dir), hp)

    # ── initialise model pool once ────────────────────────────────────────────
    pool = init_models(len(label_names), lora_model_name=args.lora_model)

    # ── train all models once (independent of hyper-param loop) ──────────────
    val_f1_per_model = train_models(pool, train_X, train_y, val_X, val_y)
    baseline_micro_f1 = max(val_f1_per_model.values()) if val_f1_per_model else 0.0
    log.info("Best single-model val micro-F1 (baseline): %.4f", baseline_micro_f1)

    # ── save hyperparams checkpoint ───────────────────────────────────────────
    with open(exp_dir / "hparams_initial.json", "w") as f:
        json.dump(hp.to_dict(), f, indent=2)

    # ─────────────────────────────────────────────────────────────────────────
    # Auto-tuning retry loop
    # ─────────────────────────────────────────────────────────────────────────
    issues_history: List[Tuple[int, List[str]]] = []
    final_rdl_set: Optional[RDLSet] = None

    for attempt in range(hp.max_retries + 1):
        log.info("")
        log.info("=" * 60)
        log.info("PIPELINE ATTEMPT %d / %d", attempt + 1, hp.max_retries + 1)
        log.info("=" * 60)

        try:
            # ── 3.1 pattern abstraction ───────────────────────────────────────
            store = run_pattern_abstraction(train_X, train_y, hp, exp_dir)

            if len(store) == 0:
                log.warning("No patterns extracted! Loosening coverage constraints.")
                hp.min_coverage      = max(hp.min_coverage * 0.3, 0.001)
                hp.max_entropy_threshold = min(hp.max_entropy_threshold * 1.5, 5.0)
                issues_history.append((attempt + 1, ["no_patterns"]))
                continue

            # ── 3.2 dynamic router ────────────────────────────────────────────
            selected_idx = run_dynamic_router(
                pool, train_X, train_y, val_X, val_y, hp, exp_dir,
                skip_router=args.no_router,
            )

            # ── register selected models ──────────────────────────────────────
            registered_names = register_selected_models(pool, selected_idx, label_names)

            # ── 3.3 rule discovery ────────────────────────────────────────────
            rdl_set = run_rule_discovery(
                store, registered_names, label_names,
                val_docs, val_y, hp, exp_dir,
                attempt=attempt,
            )

        except Exception as exc:
            log.error("Pipeline error on attempt %d: %s", attempt + 1, exc, exc_info=True)
            issues_history.append((attempt + 1, [f"exception: {exc}"]))
            if attempt < hp.max_retries:
                hp.min_coverage      = max(hp.min_coverage * 0.5, 0.001)
                hp.max_trials        = min(hp.max_trials * 2, 400)
            continue

        # ── quality check ─────────────────────────────────────────────────────
        is_good, issues = assess_quality(rdl_set, hp)
        final_rdl_set   = rdl_set

        if is_good:
            log.info("✓ Rule quality check PASSED on attempt %d.", attempt + 1)
            break

        log.warning("✗ Quality issues on attempt %d: %s", attempt + 1, issues)
        issues_history.append((attempt + 1, issues))

        if attempt == hp.max_retries:
            log.error("All %d retries exhausted.", hp.max_retries + 1)
            dump_diagnostic_report(
                hp, label_names, val_y, issues_history, final_rdl_set, exp_dir
            )
            sys.exit(1)

        adjust_hparams(hp, issues, attempt + 1, n_models=len(pool))

    # ── evaluate final rule set on test set ───────────────────────────────────
    if final_rdl_set is not None and len(final_rdl_set.rules) > 0:
        test_docs = [Document(cnt=t) for t in test_X]
        try:
            test_metrics = final_rdl_set.evaluate(test_docs, test_y)
            final_micro_f1 = float(test_metrics.get("micro_f1", 0.0))
            log.info(
                "Test set — micro-F1=%.4f  macro-F1=%.4f",
                test_metrics.get("micro_f1", 0), test_metrics.get("macro_f1", 0),
            )
        except Exception as exc:
            log.warning("Could not evaluate rule set on test set: %s", exc)
            final_micro_f1 = 0.0
    else:
        final_micro_f1 = 0.0

    save_and_print_results(
        final_rdl_set or RDLSet([], label_names),
        baseline_micro_f1,
        final_micro_f1,
        exp_dir,
        hp,
    )

    log.info("Pipeline complete. Results in %s", exp_dir)


if __name__ == "__main__":
    main()
