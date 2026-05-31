"""
run_loris_multi_pipeline.py
===========================
Unified LORIS pipeline supporting multiple datasets:
  --dataset aapd     Academic paper abstracts → arXiv topics (54 labels)
  --dataset eurlex   EU legal texts → EuroVoc concepts (top-50)
  --dataset rcv1     Reuters news (large-scale) → topics (top-50, pseudo-text from TF-IDF)
  --dataset bgc      Book blurbs → genre classification (~32 labels)

Steps:
  Step 1 · Pattern Abstraction  (PatternAbstractor)
  Step 2 · Dynamic Router       (SelectionNetwork, stochastic Top-K)
  Step 3 · Rule Discovery       (RuleLearner, Bayesian optimisation)

Usage
-----
  # 1. Prepare dataset (download + convert to CSV):
  python run_loris_multi_pipeline.py --dataset aapd --prepare

  # 2. Run pipeline:
  python run_loris_multi_pipeline.py --dataset aapd [options]

  Options (same as reuters pipeline):
    --subset_size 2000   # 0 = all data
    --top_labels  20     # restrict to N most-frequent labels (default: dataset-specific)
    --debug              # verbose Optuna + full tracebacks
    --no_router          # skip neural router, use val-F1 ranking
    --rule_strategy batch
    --baseline_mode mean_top_k
"""

from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import csv
import json
import logging
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# ── LORIS pattern extraction ───────────────────────────────────────────────────
from pattern_extraction import (
    Document,
    PatternAbstractor,
    PatternStore,
    register_ml_model,
)

# ── LORIS classifiers ─────────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Dataset-specific: stopwords, prepare functions, config registry
# ══════════════════════════════════════════════════════════════════════════════
#
# Migrated to loris.data (Phase 5). Re-exported here so this module (and the
# chase pipeline that imports these names from it) keep working unchanged.
from loris.data import (  # noqa: E402,F401
    HParams,
    DatasetConfig,
    DATASET_REGISTRY,
    AAPD_STOP_WORDS,
    EURLEX_STOP_WORDS,
    RCV1_STOP_WORDS,
    BGC_STOP_WORDS,
    REUTERS21578_STOP_WORDS,
    prepare_aapd,
    prepare_eurlex,
    prepare_rcv1,
    prepare_bgc,
    prepare_reuters21578,
    configure_logging,
    load_data,
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Shared pipeline logic (dataset-agnostic)
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter dataclass
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _setup_experiment(args: argparse.Namespace, dataset_name: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = Path(args.exp_dir) if args.exp_dir else _ROOT / "experiments"
    exp_dir = base / f"{dataset_name}_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def _hf_model_available(model_name: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
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
        self.label_list = label_names
        self._label2idx = {l: i for i, l in enumerate(label_names)}

    def predict(self, text: str) -> str:
        proba = self._clf.predict_proba([text])[0]
        return self._label_names[int(np.argmax(proba))]

    def predict_proba_single(self, text: str) -> np.ndarray:
        """Per-label probability vector for a single document."""
        return self._clf.predict_proba([text])[0]

    def label_index(self, label: str) -> int:
        return self._label2idx[label]


# ──────────────────────────────────────────────────────────────────────────────
# Step 0 — data loading (unified)
# ──────────────────────────────────────────────────────────────────────────────


def init_models(
    n_labels: int,
    lora_model_name: Optional[str] = None,
) -> Dict[str, object]:
    """Return OrderedDict name → unfitted classifier."""
    pool: Dict[str, object] = {}

    # ── TF-IDF variants ───────────────────────────────────────────────────────
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

    # ── Neural variants ───────────────────────────────────────────────────────
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
                from models.pretrained_encoder_classifier import (
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
            from models.lora_slm_classifier import LoRASLMClassifier
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
    val_X: List[str], val_y: np.ndarray,
    subsample_ratio: float = 1.0,
) -> Dict[str, float]:
    """Fit every model and return val micro-F1."""
    val_f1: Dict[str, float] = {}
    n_train = len(train_X)
    n_sub = int(n_train * subsample_ratio)
    for i, (name, clf) in enumerate(pool.items()):
        t0 = time.time()
        rng = np.random.RandomState(42 + i)
        sub_idx = rng.choice(n_train, n_sub, replace=False)
        sub_X = [train_X[j] for j in sub_idx]
        sub_y = train_y[sub_idx]
        log.info("Training  %s (subsample %d/%d, seed=%d) …",
                 name, n_sub, n_train, 42 + i)
        try:
            clf.fit(sub_X, sub_y, val_X, val_y)
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


def evaluate_models_on_test(
    pool: Dict[str, object],
    test_X: List[str],
    test_y: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Evaluate every trained model on the held-out test set."""
    results: Dict[str, Dict[str, float]] = {}
    for name, clf in pool.items():
        try:
            metrics = clf.evaluate(test_X, test_y)
            results[name] = {
                "micro_f1": float(metrics["micro_f1"]),
                "macro_f1": float(metrics["macro_f1"]),
            }
            log.info(
                "Test  %-22s  micro-F1=%.4f  macro-F1=%.4f",
                name, metrics["micro_f1"], metrics["macro_f1"],
            )
        except Exception as exc:
            log.error("Test  %-22s  FAILED: %s", name, exc)
            results[name] = {"micro_f1": 0.0, "macro_f1": 0.0}
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Step 3.1 — pattern abstraction
# ──────────────────────────────────────────────────────────────────────────────

def run_pattern_abstraction(
    train_X: List[str],
    train_y: np.ndarray,
    hp: HParams,
    exp_dir: Path,
) -> Tuple[PatternStore, np.ndarray]:
    """Returns (store, cluster_labels) where cluster_labels[i] is the cluster
    assignment of train_X[i].  Downstream batch BO reuses this to avoid
    clustering drift."""
    t0 = time.time()
    log.info("=== Step 3.1  Pattern Abstraction ===")

    abstractor = PatternAbstractor(
        n_clusters=hp.n_clusters,
        min_coverage=hp.min_coverage,
        max_entropy_threshold=hp.max_entropy_threshold,
        tfidf_top_k=hp.tfidf_top_k,
        random_state=42,
        pattern_mode=hp.pattern_mode,
        extra_stop_words=hp.extra_stop_words,
        anchor_min_df=hp.anchor_min_df,
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
    return store, abstractor.cluster_labels_


# ──────────────────────────────────────────────────────────────────────────────
# Step 3.2 — dynamic router
# ──────────────────────────────────────────────────────────────────────────────

def _build_multi_label_oracle_mask(
    pool: Dict[str, object],
    texts: List[str],
    y_true: np.ndarray,
    k: int,
) -> np.ndarray:
    n_docs = len(texts)
    names = list(pool.keys())
    n_models = len(names)
    scores = np.zeros((n_docs, n_models), dtype=np.float32)

    for j, name in enumerate(names):
        clf = pool[name]
        try:
            proba = clf.predict_proba(texts)
            preds = (proba >= 0.5).astype(int)
            overlap = (preds & y_true.astype(int)).sum(axis=1).astype(np.float32)
            scores[:, j] = overlap
        except Exception as exc:
            log.debug("oracle mask: model %s failed — %s", name, exc)

    oracle = np.zeros((n_docs, n_models), dtype=np.float32)
    eff_k = min(k, n_models)
    for i in range(n_docs):
        top_idx = np.argpartition(scores[i], -eff_k)[-eff_k:]
        oracle[i, top_idx] = 1.0
    return oracle


def run_dynamic_router(
    pool: Dict[str, object],
    train_X: List[str], train_y: np.ndarray,
    val_X: List[str], val_y: np.ndarray,
    hp: HParams,
    exp_dir: Path,
    skip_router: bool = False,
) -> List[int]:
    t0 = time.time()
    log.info("=== Step 3.2  Dynamic Router ===")
    model_names = list(pool.keys())
    n_models = len(model_names)
    k = min(hp.k_models, n_models)

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
        log.info("Router skipped — selected by val-F1: %s",
                 [model_names[i] for i in selected])
        return selected

    log.info("Building document features (TF-IDF + TruncatedSVD) …")
    tfidf_vec = TfidfVectorizer(max_features=20_000, sublinear_tf=True)
    X_sp = tfidf_vec.fit_transform(train_X)
    svd = TruncatedSVD(n_components=hp.router_feat_dim, random_state=42)
    X_dense = svd.fit_transform(X_sp).astype(np.float32)
    Xval_dense = svd.transform(
        tfidf_vec.transform(val_X)
    ).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.from_numpy(X_dense).to(device)
    Xval_t = torch.from_numpy(Xval_dense).to(device)

    log.info("Building oracle masks for %d documents …", len(train_X))
    oracle_np = _build_multi_label_oracle_mask(pool, train_X, train_y, k)
    oracle_t = torch.from_numpy(oracle_np).to(device)

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

    loss_fn = HybridLoss(lambda_task=0.1, lambda_ent=0.1)
    optimiser = torch.optim.Adam(net.parameters(), lr=hp.router_lr)

    log.info("Training SelectionNetwork for %d epochs …", hp.router_epochs)
    net.train()
    for epoch in range(1, hp.router_epochs + 1):
        _, scores = net(X_t, return_scores=True)
        l_imit = loss_fn.imitation_loss(scores, oracle_t)
        l_ent = loss_fn.entropy_loss(scores)
        loss = l_imit - 0.1 * l_ent

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if epoch % 5 == 0 or epoch == hp.router_epochs:
            log.info(
                "  [Router epoch %3d/%d]  L_imit=%.4f  L_ent=%.4f  L_total=%.4f",
                epoch, hp.router_epochs, l_imit.item(), l_ent.item(), loss.item(),
            )

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
    suffix: str = "",
) -> List[str]:
    model_names_list = list(pool.keys())
    registered: List[str] = []
    for idx in selected_indices:
        clf = pool[model_names_list[idx]]
        pred_name = f"loris_clf_{model_names_list[idx]}{suffix}"
        wrapper = _PredictWrapper(clf, label_names)
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
    per_type_top_k: Optional[int] = None,
) -> Tuple[list, Optional[np.ndarray]]:
    """Select top predicates using hybrid scoring: phi * log1p(coverage * 100).

    Returns ``(selected_preds, fire_masks)`` where *fire_masks* is a bool
    array of shape ``(n_selected, n_val_docs)`` that can be passed downstream
    to avoid recomputing ``pred(doc)`` in ``precompute_fire_masks()``.

    When *per_type_top_k* is set, predicates are grouped by class name
    (MatchPredicate, CooccurPredicate, …) and the top *per_type_top_k* are
    kept **per type**, ensuring diversity across predicate kinds.
    When *per_type_top_k* is ``None`` (default), the global *top_k* is used.

    Optimised with joblib parallel mask generation and NumPy matrix
    multiplication for batch phi (Matthews correlation) computation.
    """
    from joblib import Parallel, delayed

    candidates = list(store)
    if per_type_top_k is None and top_k > 0 and len(candidates) <= top_k:
        return candidates, None  # 无筛选，masks 未计算

    n_labels = len(label_names)
    n_docs = len(val_docs)

    log.info("_select_top_predicates: %d candidates × %d docs × %d labels — "
             "generating fire masks (parallel) …", len(candidates), n_docs, n_labels)
    t0 = time.time()

    # ── 1. joblib 并行生成布尔掩码矩阵 ──────────────────────────────────────
    def _eval_pred(p):
        return np.array([bool(p(doc)) for doc in val_docs], dtype=bool)

    raw_masks = Parallel(n_jobs=-1, batch_size="auto")(
        delayed(_eval_pred)(p) for p in candidates
    )
    fire_masks = np.stack(raw_masks)  # shape: (n_candidates, n_docs)

    t_masks = time.time()
    log.info("  fire masks done in %.1fs", t_masks - t0)

    # ── 2. 向量化过滤：去掉全 0 / 覆盖率过高（>85%）的谓词 ─────────────────
    n_fires = fire_masks.sum(axis=1)
    max_cov_fires = int(n_docs * 0.85)
    valid = (n_fires > 0) & (n_fires < max_cov_fires)
    valid_indices = np.where(valid)[0]

    if len(valid_indices) == 0:
        log.info("  no valid predicates after coverage filter")
        return [], None

    # ── 3. NumPy 矩阵乘法批量计算所有谓词×所有标签的混淆矩阵 ────────────────
    F = fire_masks[valid_indices].astype(np.float32)   # (n_valid, n_docs)
    L = val_y.astype(np.float32)                        # (n_docs, n_labels)
    coverages = n_fires[valid_indices].astype(np.float64) / n_docs

    TP = F @ L                    # (n_valid, n_labels)
    FP = F @ (1 - L)
    FN = (1 - F) @ L
    TN = (1 - F) @ (1 - L)

    den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    phi_matrix = np.divide(
        (TP * TN - FP * FN),
        den,
        out=np.zeros_like(den),
        where=(den > 0),
    )
    best_phis = np.max(np.abs(phi_matrix), axis=1)  # (n_valid,)

    t_phi = time.time()
    log.info("  phi matrix done in %.1fs", t_phi - t_masks)

    # ── 4. 混合打分 ─────────────────────────────────────────────────────────
    # scored: (hybrid, phi, cov, pred, original_candidate_index)
    scored: List[Tuple[float, float, float, object, int]] = []
    for idx_in_valid, orig_idx in enumerate(valid_indices):
        bp = float(best_phis[idx_in_valid])
        cov = float(coverages[idx_in_valid])
        if bp >= 0.02:
            cov_factor = np.sqrt(cov * (1 - cov)) * 2
            hybrid = bp * cov_factor
            scored.append((hybrid, bp, cov, candidates[orig_idx], int(orig_idx)))

    # ── 5. 收集结果（保留 fire_masks 供下游复用）────────────────────────────
    def _collect(items):
        preds = [it[3] for it in items]
        masks = np.stack([fire_masks[it[4]] for it in items]) if items else None
        return preds, masks

    # ── 6. per_type_top_k 分组排序（保留原有多样性逻辑）─────────────────────
    if per_type_top_k is not None:
        from collections import defaultdict
        type_groups: Dict[str, list] = defaultdict(list)
        for item in scored:
            type_groups[type(item[3]).__name__].append(item)
        selected = []
        for type_name, group in type_groups.items():
            group.sort(key=lambda x: -x[0])
            selected.extend(group[:per_type_top_k])
        result_preds, result_masks = _collect(selected)
        log.info("  selected %d predicates (%d types × top-%d) in %.1fs total",
                 len(selected), len(type_groups), per_type_top_k,
                 time.time() - t0)
        return result_preds, result_masks

    scored.sort(key=lambda x: -x[0])
    final = scored if top_k <= 0 else scored[:top_k]
    result_preds, result_masks = _collect(final)
    log.info("  selected %d predicates (top_k=%s) in %.1fs total",
             len(final), "all" if top_k <= 0 else top_k, time.time() - t0)
    return result_preds, result_masks


def run_rule_discovery(
    store: PatternStore,
    registered_model_names: List[str],
    label_names: List[str],
    val_docs: List[Document],
    val_y: np.ndarray,
    hp: HParams,
    exp_dir: Path,
    attempt: int = 0,
    base_val_predictions: Optional[np.ndarray] = None,
    base_val_f1: Optional[float] = None,
    accept_docs: Optional[List[Document]] = None,
    accept_y: Optional[np.ndarray] = None,
    accept_predictions: Optional[np.ndarray] = None,
    accept_f1: Optional[float] = None,
) -> RDLSet:
    import optuna

    t0 = time.time()
    log.info("=== Step 3.3  Rule Discovery ===")

    filtered_preds, _ = _select_top_predicates(
        store, val_docs, val_y, label_names, top_k=hp.predicate_top_k
    )
    log.info(
        "Candidate predicates=%d→%d (top-100 by val precision)  "
        "ML models=%d  labels=%d  max_trials=%d  top_n=%d",
        len(store), len(filtered_preds), len(registered_model_names),
        len(label_names), hp.max_trials, hp.top_n_rules,
    )

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
        base_predictions=base_val_predictions,
        base_f1=base_val_f1,
        top_per_type=hp.top_per_type,
        rule_min_precision=hp.rule_min_precision,
        accept_docs=accept_docs,
        accept_labels=accept_y,
        accept_predictions=accept_predictions,
        accept_f1=accept_f1,
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
# Step 3.3b — batch rule discovery
# ──────────────────────────────────────────────────────────────────────────────

def _select_cluster_models(
    cid: int,
    pool: Dict[str, object],
    label_names: List[str],
    val_y: np.ndarray,
    val_X: List[str],
    val_cluster_labels: Optional[np.ndarray],
    cluster_train_X: List[str],
    cluster_train_y: np.ndarray,
    hp: HParams,
    exp_dir: Path,
    skip_router: bool,
    global_registered_names: List[str],
) -> List[str]:
    """
    We have 3 options for cluster-specific model selection: global, f1_rank, and router. 
    """
    if hp.cluster_model_selection == "global":
        return global_registered_names

    assert val_cluster_labels is not None
    cluster_val_indices = [
        i for i, c in enumerate(val_cluster_labels) if c == cid
    ]

    if len(cluster_val_indices) < 10:
        log.info("  Cluster %d: only %d val docs → fallback to global models",
                 cid, len(cluster_val_indices))
        return global_registered_names

    cluster_val_X = [val_X[i] for i in cluster_val_indices]
    cluster_val_y = val_y[cluster_val_indices]

    if hp.cluster_model_selection == "f1_rank":
        model_f1: Dict[str, float] = {}
        model_names_list = list(pool.keys())
        for name, clf in pool.items():
            try:
                preds = clf.predict(cluster_val_X).astype(np.float32)
                model_f1[name] = float(
                    f1_score(cluster_val_y, preds,
                             average="macro", zero_division=0)
                )
            except Exception:
                model_f1[name] = 0.0
        sorted_names = sorted(model_f1, key=model_f1.get, reverse=True)
        top_k_names = sorted_names[:hp.k_models]
        selected_idx = [model_names_list.index(n) for n in top_k_names]
        cluster_ml_names = register_selected_models(
            pool, selected_idx, label_names, suffix=f"_c{cid}"
        )
        log.info("  Cluster %d model selection (f1_rank): %s",
                 cid, cluster_ml_names)
        return cluster_ml_names

    elif hp.cluster_model_selection == "router":
        cluster_selected_idx = run_dynamic_router(
            pool, cluster_train_X, cluster_train_y,
            cluster_val_X, cluster_val_y,
            hp, exp_dir, skip_router=False,
        )
        cluster_ml_names = register_selected_models(
            pool, cluster_selected_idx, label_names, suffix=f"_c{cid}"
        )
        log.info("  Cluster %d model selection (router): %s",
                 cid, cluster_ml_names)
        return cluster_ml_names

    log.warning("  Unknown cluster_model_selection=%r, using global",
                hp.cluster_model_selection)
    return global_registered_names


def run_rule_discovery_batch(
    train_X: List[str],
    train_y: np.ndarray,
    pool: Dict[str, object],
    label_names: List[str],
    val_docs: List[Document],
    val_X: List[str],
    val_y: np.ndarray,
    hp: HParams,
    exp_dir: Path,
    skip_router: bool = False,
    global_registered_names: Optional[List[str]] = None,
    base_val_predictions: Optional[np.ndarray] = None,
    base_val_f1: Optional[float] = None,
    accept_docs: Optional[List[Document]] = None,
    accept_y: Optional[np.ndarray] = None,
    accept_predictions: Optional[np.ndarray] = None,
    accept_f1: Optional[float] = None,
    attempt: int = 0,
    precomputed_cluster_labels: Optional[np.ndarray] = None,
) -> RDLSet:
    import optuna
    from sklearn.cluster import KMeans

    t0 = time.time()
    log.info("=== Step 3.3b  Batch Rule Discovery (per-cluster BO) ===")

    if global_registered_names is None:
        global_registered_names = []

    if not log.isEnabledFor(logging.DEBUG):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── 1. Cluster training documents ─────────────────────────────────────────
    from sentence_transformers import SentenceTransformer

    if precomputed_cluster_labels is not None:
        # Reuse cluster assignments from Step 3.1 to avoid clustering drift
        cluster_labels = precomputed_cluster_labels
        n_clusters = int(cluster_labels.max()) + 1
        log.info("Batch mode: reusing Step 3.1 clusters for %d training docs "
                 "(%d clusters, model_selection=%s)",
                 len(train_X), n_clusters, hp.cluster_model_selection)
    else:
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = st_model.encode(train_X, show_progress_bar=False)

        if hp.n_clusters is not None:
            n_clusters = hp.n_clusters
        else:
            from sklearn.metrics import silhouette_score

            best_k, best_sil = 2, -1.0
            for k in range(2, min(16, len(train_X))):
                km_tmp = KMeans(n_clusters=k, random_state=42, n_init=5)
                km_labels = km_tmp.fit_predict(embeddings)
                sil = silhouette_score(embeddings, km_labels)
                if sil > best_sil:
                    best_k, best_sil = k, sil
            n_clusters = best_k

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(embeddings)
        log.info("Batch mode: clustered %d training docs into %d clusters "
                 "(model_selection=%s)",
                 len(train_X), n_clusters, hp.cluster_model_selection)

    val_cluster_labels = None
    if hp.cluster_model_selection != "global":
        st_model_val = SentenceTransformer("all-MiniLM-L6-v2")
        val_embeddings = st_model_val.encode(
            [d.cnt for d in val_docs], show_progress_bar=False
        )
        # Fit a KMeans matching the existing cluster labels for prediction
        km_val = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        train_embeddings = st_model_val.encode(train_X, show_progress_bar=False)
        km_val.fit(train_embeddings)
        val_cluster_labels = km_val.predict(val_embeddings)

    # ── 2. Per-cluster: pattern abstraction + model selection + BO ─────────────
    all_trials = []

    for cid in range(n_clusters):
        cluster_indices = [i for i, c in enumerate(cluster_labels) if c == cid]
        if not cluster_indices:
            log.warning("Cluster %d is empty, skipping.", cid)
            continue

        cluster_train_X = [train_X[i] for i in cluster_indices]
        cluster_train_y = train_y[cluster_indices]

        log.info("  Cluster %d: %d training docs", cid, len(cluster_train_X))

        abstractor = PatternAbstractor(
            n_clusters=1,
            min_coverage=hp.min_coverage,
            max_entropy_threshold=hp.max_entropy_threshold,
            tfidf_top_k=hp.tfidf_top_k,
            random_state=42,
            pattern_mode=hp.pattern_mode,
            extra_stop_words=hp.extra_stop_words,
            anchor_min_df=hp.anchor_min_df,
        )
        try:
            abstractor.fit(cluster_train_X, cluster_train_y)
        except Exception as exc:
            log.warning("  Cluster %d pattern abstraction failed: %s", cid, exc)
            continue

        store = abstractor.to_store()
        if len(store) == 0:
            log.warning("  Cluster %d: 0 patterns extracted, skipping.", cid)
            continue

        filtered_preds, _ = _select_top_predicates(
            store, val_docs, val_y, label_names, top_k=hp.predicate_top_k
        )
        if not filtered_preds:
            log.warning("  Cluster %d: 0 filtered predicates, skipping.", cid)
            continue

        cluster_ml_names = _select_cluster_models(
            cid=cid,
            pool=pool,
            label_names=label_names,
            val_y=val_y,
            val_X=val_X,
            val_cluster_labels=val_cluster_labels,
            cluster_train_X=cluster_train_X,
            cluster_train_y=cluster_train_y,
            hp=hp,
            exp_dir=exp_dir,
            skip_router=skip_router,
            global_registered_names=global_registered_names,
        )

        log.info("  Cluster %d: %d→%d predicates, %d ML models, "
                 "running BO with %d trials",
                 cid, len(store), len(filtered_preds),
                 len(cluster_ml_names), hp.max_trials)

        cluster_label_indices = np.where(cluster_train_y.sum(axis=0) > 0)[0].tolist()

        learner = RuleLearner(
            candidate_predicates=filtered_preds,
            candidate_ml_models=cluster_ml_names,
            label_list=label_names,
            val_docs=val_docs,
            val_labels=val_y,
            max_trials=hp.max_trials,
            min_coverage=hp.min_coverage_rule,
            seed=42 + cid,
            storage_path=str(exp_dir / f"optuna_batch_cluster{cid}_attempt{attempt}"),
            verbose=log.isEnabledFor(logging.DEBUG),
            base_predictions=base_val_predictions,
            base_f1=base_val_f1,
            top_per_type=hp.top_per_type,
            rule_min_precision=hp.rule_min_precision,
            metric_mode=hp.batch_metric_mode,
            cluster_label_indices=cluster_label_indices,
        )
        trials = learner.run_bo()
        trials = [(t, tg) for t, tg in trials if t.value >= hp.min_f1_gain]
        for t, tg in trials:
            all_trials.append((t, tg, filtered_preds, cluster_ml_names))

        log.info("  Cluster %d: %d trials surviving min_f1_gain filter",
                 cid, len(trials))

    log.info("Total trials collected across all clusters: %d", len(all_trials))

    rdl_set = RuleLearner.batch_select(
        all_trials=all_trials,
        label_list=label_names,
        val_docs=val_docs,
        val_labels=val_y,
        base_predictions=base_val_predictions,
        base_f1=base_val_f1,
        sort_by_gain=hp.rule_batch_sort,
        verbose=log.isEnabledFor(logging.DEBUG),
        rule_min_precision=hp.rule_min_precision,
        accept_docs=accept_docs,
        accept_labels=accept_y,
        accept_predictions=accept_predictions,
        accept_f1=accept_f1,
    )

    out_path = str(exp_dir / "rules.json")
    rdl_set.save(out_path)

    log.info(
        "Batch Rule Discovery done in %.1fs — %d rules selected, saved to %s",
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
        hp.max_trials = min(hp.max_trials * 2, 400)
        hp.top_n_rules = min(hp.top_n_rules + 5, 20)
        log.warning(
            "  → empty_ruleset → min_coverage_rule=%.4f  max_trials=%d  top_n=%d",
            hp.min_coverage_rule, hp.max_trials, hp.top_n_rules,
        )
    if "low_coverage" in issues:
        hp.min_coverage = max(hp.min_coverage * 0.5, 0.001)
        hp.min_coverage_rule = max(hp.min_coverage_rule * 0.5, 0.001)
        log.warning(
            "  → low_coverage → min_coverage=%.4f  min_coverage_rule=%.4f",
            hp.min_coverage, hp.min_coverage_rule,
        )
    if "over_complex" in issues:
        hp.max_entropy_threshold = max(hp.max_entropy_threshold * 0.7, 0.3)
        hp.tfidf_top_k = max(hp.tfidf_top_k - 5, 5)
        log.warning(
            "  → over_complex → max_entropy=%.2f  tfidf_top_k=%d",
            hp.max_entropy_threshold, hp.tfidf_top_k,
        )
    if "no_f1_gain" in issues:
        hp.max_trials = min(hp.max_trials * 2, 400)
        hp.router_sigma *= 1.5
        hp.k_models = min(hp.k_models + 1, n_models)
        log.warning(
            "  → no_f1_gain → max_trials=%d  router_sigma=%.3f  k_models=%d",
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
    dataset_display_name: str = "Dataset",
    test_metrics_per_model: Optional[Dict[str, Dict[str, float]]] = None,
    baseline_test_macro_f1: float = 0.0,
    final_macro_f1: float = 0.0,
) -> None:
    metrics = {
        "baseline_test_micro_f1": baseline_micro_f1,
        "baseline_test_macro_f1": baseline_test_macro_f1,
        "final_micro_f1": final_micro_f1,
        "final_macro_f1": final_macro_f1,
        "micro_f1_delta": final_micro_f1 - baseline_micro_f1,
        "macro_f1_delta": final_macro_f1 - baseline_test_macro_f1,
        "n_rules": len(rdl_set.rules),
        "hparams": hp.to_dict(),
    }
    if test_metrics_per_model:
        metrics["per_model_test"] = test_metrics_per_model
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    sorted_rules = sorted(rdl_set.rules, key=lambda r: r.score, reverse=True)
    width = 72
    border = "=" * width
    lines = [
        border,
        f"{'LORIS Rule Discovery — ' + dataset_display_name:^{width}}",
        border,
        "",
    ]
    if test_metrics_per_model:
        lines.append("  Per-model test results:")
        for name, m in sorted(
            test_metrics_per_model.items(),
            key=lambda x: -x[1]["micro_f1"],
        ):
            lines.append(
                f"    {name:30s}  micro-F1={m['micro_f1']:.4f}  macro-F1={m['macro_f1']:.4f}"
            )
        lines.append("")

    lines.append(f"  Best model test micro-F1 (baseline): {baseline_micro_f1:.4f}")
    lines.append(f"  Best model test macro-F1 (baseline): {baseline_test_macro_f1:.4f}")
    lines.append(f"  Model+Rules  test micro-F1:          {final_micro_f1:.4f}  (delta={final_micro_f1 - baseline_micro_f1:+.4f})")
    lines.append(f"  Model+Rules  test macro-F1:          {final_macro_f1:.4f}  (delta={final_macro_f1 - baseline_test_macro_f1:+.4f})")
    lines.append("")
    lines.append(f"  Rules discovered: {len(rdl_set.rules)}")
    lines.append(border)

    for i, rule in enumerate(sorted_rules, 1):
        lines.append(_rule_to_readable(rule, i))
        lines.append("-" * width)

    lines.append(border)
    report = "\n".join(lines)

    print("\n" + report)
    with open(exp_dir / "rules_readable.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")

    log.info(
        "Results saved to %s  (baseline micro-F1=%.4f -> model+rules micro-F1=%.4f  delta=%+.4f)",
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
    # lines += [
    #     "",
    #     "### Recommendations",
    #     "  1. Increase --subset_size for more training data.",
    #     "  2. Try fewer --top_labels (e.g. 10) for a simpler labelling task.",
    #     "  3. Manually lower min_coverage_rule below current value.",
    #     "  4. Check pattern coverage stats in patterns.json.",
    #     "  5. Add stronger pre-trained models (encoder/LoRA) to the pool.",
    #     "=" * 72,
    # ]
    report = "\n".join(lines)
    report_path = exp_dir / "DIAGNOSTIC_REPORT.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logging.error(
        "\n%s\nDiagnostic report written to %s", report, report_path
    )


# ──────────────────────────────────────────────────────────────────────────────
# Per-rule correction analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_rule_corrections(
    rdl_set: RDLSet,
    docs: List[Document],
    y_true: np.ndarray,
    base_predictions: np.ndarray,
    label_names: List[str],
    exp_dir: Path,
) -> None:
    from pattern_extraction.predicates import (
        SimPredicate as _SimPred,
        LabelPredicate as _LP,
    )

    n_docs = len(docs)
    n_labels = len(label_names)
    label2idx = {name: i for i, name in enumerate(label_names)}

    # ── Pre-build sim graph for sim rules (SpMV evaluation) ──
    sim_rule_ids = {
        id(r) for r in rdl_set.rules
        if any(isinstance(p, _SimPred) for p in r.body)
    }
    sim_graphs = None
    label_state = None
    if sim_rule_ids and n_docs > 0:
        from rule_discovery.sim_graph import compute_embeddings, build_sim_graph
        thresholds = sorted(set(
            p.threshold for r in rdl_set.rules
            for p in r.body if isinstance(p, _SimPred)
        ))
        texts = [d.cnt for d in docs]
        emb = compute_embeddings(
            texts, cache_path=str(exp_dir / "test_embeddings.npy"))
        sim_graphs = build_sim_graph(
            emb, threshold_bins=thresholds, max_avg_degree=999999)
        label_state = (base_predictions > 0).astype(np.float32)

    # baseline macro-F1 (model only, no rules)
    base_macro_f1 = float(
        f1_score(y_true, base_predictions, average="macro", zero_division=0)
    )

    analysis = []
    lines = [
        "=" * 72,
        "  PER-RULE CORRECTION ANALYSIS  (test set)",
        "  baseline model macro-F1 = {:.4f}".format(base_macro_f1),
        "=" * 72,
        "",
    ]

    for rule_idx, rule in enumerate(rdl_set.rules, 1):
        label_idx = label2idx.get(rule.consequence)
        if label_idx is None:
            continue

        # ── fires: sim rule → SpMV; otherwise → per-doc rule.fires ──
        if id(rule) in sim_rule_ids and sim_graphs is not None:
            sim_pred = next(p for p in rule.body if isinstance(p, _SimPred))
            adj = sim_graphs.get(sim_pred.threshold)
            if adj is None:
                fires = np.zeros(n_docs, dtype=bool)
            else:
                sim_mask = np.ones(n_docs, dtype=bool)
                for lp in (p for p in rule.body if isinstance(p, _LP)):
                    lidx = label2idx.get(lp.label)
                    if lidx is not None:
                        sim_mask &= np.asarray(
                            (adj @ label_state[:, lidx]) > 0).ravel()
                    else:
                        sim_mask[:] = False
                text_preds = [p for p in rule.body
                              if not isinstance(p, (_SimPred, _LP))]
                if text_preds:
                    for i, doc in enumerate(docs):
                        if sim_mask[i] and not all(p(doc) for p in text_preds):
                            sim_mask[i] = False
                fires = sim_mask
        else:
            fires = np.zeros(n_docs, dtype=bool)
            for i, doc in enumerate(docs):
                fires[i] = rule.fires(doc)

        n_fires = int(fires.sum())

        # ── 独立模拟：单条规则在 test set 上的 F1 增益 ──────────────────
        test_preds = base_predictions.copy()
        if n_fires > 0:
            op = getattr(rule, "consequence_op", "add")
            if op == "add":
                test_preds[fires, label_idx] = 1.0
            elif op == "remove":
                test_preds[fires, label_idx] = 0.0
            elif op == "replace":
                test_preds[fires, :] = 0.0
                test_preds[fires, label_idx] = 1.0
            else:
                test_preds[fires, label_idx] = 1.0  # fallback: treat as add

        test_macro_f1 = float(
            f1_score(y_true, test_preds, average="macro", zero_division=0)
        )
        test_f1_gain = test_macro_f1 - base_macro_f1

        # ── op-aware 三分类：No-op / Improved / Worsened ─────────────
        if n_fires > 0:
            old_pred = base_predictions[fires, label_idx]
            new_pred = test_preds[fires, label_idx]
            gt = y_true[fires, label_idx]
            changed = (old_pred != new_pred)
            n_no_op = int((~changed).sum())
            n_improved = int((changed & (new_pred == gt)).sum())
            n_worsened = int((changed & (new_pred != gt)).sum())
        else:
            n_no_op = 0
            n_improved = 0
            n_worsened = 0
        n_changes = n_improved + n_worsened
        corr_prec = n_improved / n_changes if n_changes > 0 else 0.0

        rule_info = {
            "rule_idx": rule_idx,
            "consequence": rule.consequence,
            "consequence_op": getattr(rule, "consequence_op", "add"),
            "body": str(rule),
            "f1_gain_val": rule.score,
            "coverage_val": rule.coverage,
            "test_fires": n_fires,
            "test_fires_pct": n_fires / n_docs if n_docs > 0 else 0.0,
            "n_no_op": n_no_op,
            "n_improved": n_improved,
            "n_worsened": n_worsened,
            "correction_precision": corr_prec,
            "test_macro_f1": test_macro_f1,
            "test_f1_gain": test_f1_gain,
        }
        analysis.append(rule_info)

        body_str = " ^ ".join(str(p) for p in rule.body) if rule.body else "(empty)"
        lines.append(f"Rule #{rule_idx:02d}  [{rule.consequence}] op={getattr(rule, 'consequence_op', 'add')}  "
                      f"val-F1-gain={rule.score:+.4f}")
        lines.append(f"  Body: {body_str}")
        lines.append(f"  Test fires: {n_fires} / {n_docs} ({n_fires/n_docs:.1%})")
        lines.append(f"  No-op (prediction unchanged): {n_no_op}")
        lines.append(f"  Improved (wrong→correct):     {n_improved}")
        lines.append(f"  Worsened (correct→wrong):     {n_worsened}")
        lines.append(f"  Correction precision: {corr_prec:.1%}")
        lines.append(f"  Test F1 (this rule alone): {test_macro_f1:.4f}  "
                      f"(gain={test_f1_gain:+.4f})")
        lines.append("")

    lines.append("=" * 72)
    report = "\n".join(lines)
    print("\n" + report)

    with open(exp_dir / "rule_analysis.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(exp_dir / "rule_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    log.info("Rule correction analysis saved to %s", exp_dir / "rule_analysis.txt")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LORIS end-to-end pipeline — multi-dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", required=True,
                   choices=list(DATASET_REGISTRY.keys()),
                   help="Dataset to use")
    p.add_argument("--prepare", action="store_true",
                   help="Download and prepare dataset (run once before training)")
    p.add_argument("--exp_dir", default=None,
                   help="Override experiment output directory")
    p.add_argument("--subset_size", type=int, default=0,
                   help="Max training docs (0=all)")
    p.add_argument("--top_labels", type=int, default=None,
                   help="Restrict to N most-frequent labels (default: dataset-specific)")
    p.add_argument("--no_router", action="store_true",
                   help="Skip neural router; rank by val-F1 instead")
    p.add_argument("--pattern_mode", default="full",
                   choices=["full", "fast", "metapad", "sim"],
                   help="Pattern extraction mode: full=all types, fast=Match/Freq only, sim=semantic similarity")
    p.add_argument("--lora_model", default=None,
                   help="HF model name for LoRASLMClassifier (requires 20 GB VRAM)")
    p.add_argument("--debug", action="store_true",
                   help="Enable verbose logging")
    p.add_argument("--max_trials", type=int, default=200)
    p.add_argument("--top_n_rules", type=int, default=10)
    p.add_argument("--min_f1_gain", type=float, default=0.001)
    p.add_argument("--top_per_type", type=int, default=30)
    p.add_argument("--rule_strategy", default="greedy",
                   choices=["greedy", "batch"])
    p.add_argument("--rule_batch_sort", action="store_true", default=True)
    p.add_argument("--no_rule_batch_sort", dest="rule_batch_sort",
                   action="store_false")
    p.add_argument("--rule_min_precision", type=float, default=0.50)
    p.add_argument("--anchor_min_df", type=int, default=3)
    p.add_argument("--rule_eval_data", default="val",
                   choices=["val", "train"])
    p.add_argument("--cluster_model_selection", default="global",
                   choices=["global", "f1_rank", "router"])
    p.add_argument("--baseline_mode", default="best",
                   choices=["best", "mean_top_k"])
    p.add_argument("--no_encoder", action="store_true",
                   help="Exclude pretrained encoder model (encoder_mlp) from pool")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    dataset_cfg = DATASET_REGISTRY[args.dataset]

    # ── Handle --prepare mode ─────────────────────────────────────────────────
    if args.prepare:
        # Minimal logging for prepare mode
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True,
        )
        log.info("Preparing dataset: %s", dataset_cfg.display_name)
        # Temporarily allow network for downloads
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        dataset_cfg.prepare_fn(dataset_cfg.data_dir)
        log.info("Dataset prepared. CSV files in %s",
                 dataset_cfg.data_dir / "processed")
        return

    # ── Normal pipeline mode ──────────────────────────────────────────────────
    exp_dir = _setup_experiment(args, dataset_cfg.name)
    configure_logging(exp_dir, debug=args.debug)

    log.info("Experiment directory: %s", exp_dir)
    log.info("Dataset: %s", dataset_cfg.display_name)

    # Resolve top_labels
    top_labels = args.top_labels if args.top_labels is not None else dataset_cfg.default_top_labels

    hp = HParams(
        subset_size=args.subset_size,
        top_labels=top_labels,
        max_trials=args.max_trials,
        top_n_rules=args.top_n_rules,
        min_f1_gain=args.min_f1_gain,
        top_per_type=args.top_per_type,
        pattern_mode=args.pattern_mode,
        rule_strategy=args.rule_strategy,
        rule_batch_sort=args.rule_batch_sort,
        cluster_model_selection=args.cluster_model_selection,
        rule_min_precision=args.rule_min_precision,
        rule_eval_data=args.rule_eval_data,
        anchor_min_df=args.anchor_min_df,
        baseline_mode=args.baseline_mode,
        extra_stop_words=dataset_cfg.stop_words,
    )

    (
        train_X, val_X, test_X,
        train_y, val_y, test_y,
        label_names,
        train_docs, val_docs,
    ) = load_data(dataset_cfg, hp)

    # ── initialise model pool ────────────────────────────────────────────────
    pool = init_models(len(label_names), lora_model_name=args.lora_model)
    if args.no_encoder and "encoder_mlp" in pool:
        del pool["encoder_mlp"]
        log.info("Excluded encoder_mlp from model pool (--no_encoder)")

    # ── train all models ─────────────────────────────────────────────────────
    val_f1_per_model = train_models(pool, train_X, train_y, val_X, val_y)
    baseline_micro_f1 = max(val_f1_per_model.values()) if val_f1_per_model else 0.0
    best_model_name = max(val_f1_per_model, key=val_f1_per_model.get) if val_f1_per_model else None
    log.info("Best single-model val micro-F1 (baseline): %.4f (%s)",
             baseline_micro_f1, best_model_name)

    # ── evaluate all models on test set ──────────────────────────────────────
    test_metrics_per_model = evaluate_models_on_test(pool, test_X, test_y)
    baseline_test_micro_f1 = max(
        (m["micro_f1"] for m in test_metrics_per_model.values()), default=0.0
    )
    baseline_test_macro_f1 = max(
        (m["macro_f1"] for m in test_metrics_per_model.values()), default=0.0
    )
    log.info(
        "Best single-model test micro-F1: %.4f  macro-F1: %.4f",
        baseline_test_micro_f1, baseline_test_macro_f1,
    )

    # ── get best model's val predictions ─────────────────────────────────────
    best_clf = pool[best_model_name] if best_model_name else None
    if best_clf is not None:
        base_val_preds = best_clf.predict(val_X).astype(np.float32)
        base_val_macro_f1 = float(
            f1_score(val_y, base_val_preds, average="macro", zero_division=0)
        )
        log.info("Best model val macro-F1 (rule discovery baseline): %.4f",
                 base_val_macro_f1)
    else:
        base_val_preds = None
        base_val_macro_f1 = None

    # ── 消除 data leakage: doc.lbl 从真实标签改为模型预测 ──────────────────
    n_labels = len(label_names)
    if best_clf is not None:
        log.info("Replacing doc.lbl with model predictions "
                 "(match test-time semantics, eliminate label leakage)")
        for i, doc in enumerate(val_docs):
            doc.lbl.clear()
            doc.lbl.update(
                label_names[j] for j in range(n_labels)
                if base_val_preds[i, j] > 0
            )
        _train_preds_for_lbl = best_clf.predict(train_X).astype(np.float32)
        for i, doc in enumerate(train_docs):
            doc.lbl.clear()
            doc.lbl.update(
                label_names[j] for j in range(n_labels)
                if _train_preds_for_lbl[i, j] > 0
            )
    else:
        for doc in val_docs:
            doc.lbl.clear()
        for doc in train_docs:
            doc.lbl.clear()

    # ── select which data Step 3 evaluates on ────────────────────────────────
    if hp.rule_eval_data == "train":
        base_rule_preds = (
            best_clf.predict(train_X).astype(np.float32) if best_clf else None
        )
        base_rule_f1 = (
            float(f1_score(train_y, base_rule_preds, average="macro", zero_division=0))
            if base_rule_preds is not None else None
        )
        rule_eval_docs, rule_eval_y, rule_eval_X = train_docs, train_y, train_X
        log.info("rule_eval_data=train — Step 3 uses training set (%d docs, "
                 "base macro-F1=%.4f)", len(train_X),
                 base_rule_f1 if base_rule_f1 is not None else 0.0)
    else:
        base_rule_preds = base_val_preds
        base_rule_f1 = base_val_macro_f1
        rule_eval_docs, rule_eval_y, rule_eval_X = val_docs, val_y, val_X
        log.info("rule_eval_data=val — Step 3 uses validation set (%d docs)",
                 len(val_X))

    # ── acceptance data (cross-validation: BO on one split, accept on the other)
    accept_kwargs: Dict = {}
    if hp.rule_eval_data == "train":
        accept_kwargs = dict(
            accept_docs=val_docs,
            accept_y=val_y,
            accept_predictions=base_val_preds,
            accept_f1=base_val_macro_f1,
        )
    else:
        # BO discovers on val → acceptance check on train (anti-overfit)
        _accept_train_preds = (
            best_clf.predict(train_X).astype(np.float32) if best_clf else None
        )
        _accept_train_f1 = (
            float(f1_score(train_y, _accept_train_preds, average="macro", zero_division=0))
            if _accept_train_preds is not None else None
        )
        accept_kwargs = dict(
            accept_docs=train_docs,
            accept_y=train_y,
            accept_predictions=_accept_train_preds,
            accept_f1=_accept_train_f1,
        )
        log.info("Acceptance cross-check on training set (%d docs, base macro-F1=%.4f)",
                 len(train_X), _accept_train_f1 if _accept_train_f1 is not None else 0.0)

    # ── save hyperparams ─────────────────────────────────────────────────────
    with open(exp_dir / "hparams_initial.json", "w") as f:
        json.dump(hp.to_dict(), f, indent=2)

    # ─────────────────────────────────────────────────────────────────────────
    # Auto-tuning retry loop
    # ─────────────────────────────────────────────────────────────────────────
    issues_history: List[Tuple[int, List[str]]] = []
    final_rdl_set: Optional[RDLSet] = None

    # Cache pattern abstraction across retry attempts when only BO params change
    _cached_store: Optional[PatternStore] = None
    _cached_pa_key: Optional[tuple] = None
    _cached_cluster_labels: Optional[np.ndarray] = None

    for attempt in range(hp.max_retries + 1):
        log.info("")
        log.info("=" * 60)
        log.info("PIPELINE ATTEMPT %d / %d", attempt + 1, hp.max_retries + 1)
        log.info("=" * 60)

        try:
            # ── 3.1 pattern abstraction (cached if params unchanged) ─────────
            pa_key = (hp.min_coverage, hp.max_entropy_threshold, hp.tfidf_top_k,
                      hp.anchor_min_df, hp.pattern_mode)
            if _cached_store is not None and _cached_pa_key == pa_key:
                store = _cached_store
                log.info("=== Step 3.1  Pattern Abstraction (cached, %d predicates) ===",
                         len(store))
            else:
                store, _cached_cluster_labels = run_pattern_abstraction(
                    train_X, train_y, hp, exp_dir)
                _cached_store = store
                _cached_pa_key = pa_key

            if len(store) == 0:
                log.warning("No patterns extracted! Loosening coverage constraints.")
                hp.min_coverage = max(hp.min_coverage * 0.3, 0.001)
                hp.max_entropy_threshold = min(hp.max_entropy_threshold * 1.5, 5.0)
                issues_history.append((attempt + 1, ["no_patterns"]))
                continue

            # ── 3.2 dynamic router ───────────────────────────────────────────
            selected_idx = run_dynamic_router(
                pool, train_X, train_y, rule_eval_X, rule_eval_y, hp, exp_dir,
                skip_router=args.no_router,
            )

            # ── register selected models ─────────────────────────────────────
            registered_names = register_selected_models(pool, selected_idx, label_names)

            # ── recompute baseline if mean_top_k ────────────────────────────
            if hp.baseline_mode == "mean_top_k" and len(selected_idx) > 1:
                model_names_list = list(pool.keys())
                selected_clfs = [pool[model_names_list[idx]] for idx in selected_idx]
                _val_preds_list = [c.predict(val_X).astype(np.float32) for c in selected_clfs]
                base_val_preds = (np.mean(_val_preds_list, axis=0) >= 0.5).astype(np.float32)
                base_val_macro_f1 = float(
                    f1_score(val_y, base_val_preds, average="macro", zero_division=0))
                log.info("mean_top_k baseline val macro-F1: %.4f (from %d models)",
                         base_val_macro_f1, len(selected_idx))
                if hp.rule_eval_data == "train":
                    _rule_preds_list = [c.predict(train_X).astype(np.float32)
                                        for c in selected_clfs]
                    base_rule_preds = (np.mean(_rule_preds_list, axis=0) >= 0.5).astype(np.float32)
                    base_rule_f1 = float(
                        f1_score(train_y, base_rule_preds, average="macro", zero_division=0))
                else:
                    base_rule_preds = base_val_preds
                    base_rule_f1 = base_val_macro_f1
                if hp.rule_eval_data == "train":
                    accept_kwargs = dict(
                        accept_docs=val_docs, accept_y=val_y,
                        accept_predictions=base_val_preds, accept_f1=base_val_macro_f1,
                    )
                else:
                    _accept_train_preds_mtk = (
                        np.mean(_rule_preds_list, axis=0) >= 0.5
                    ).astype(np.float32) if hp.baseline_mode == "mean_top_k" else (
                        best_clf.predict(train_X).astype(np.float32) if best_clf else None
                    )
                    _accept_train_f1_mtk = (
                        float(f1_score(train_y, _accept_train_preds_mtk,
                                       average="macro", zero_division=0))
                        if _accept_train_preds_mtk is not None else None
                    )
                    accept_kwargs = dict(
                        accept_docs=train_docs, accept_y=train_y,
                        accept_predictions=_accept_train_preds_mtk,
                        accept_f1=_accept_train_f1_mtk,
                    )

                # ── mean_top_k: 刷新 doc.lbl 以匹配新的集成预测 ────────────
                log.info("mean_top_k: refreshing doc.lbl with ensemble predictions")
                for i, doc in enumerate(val_docs):
                    doc.lbl.clear()
                    doc.lbl.update(
                        label_names[j] for j in range(n_labels)
                        if base_val_preds[i, j] > 0
                    )
                _mtk_train_preds = (
                    np.mean([c.predict(train_X).astype(np.float32)
                             for c in selected_clfs], axis=0) >= 0.5
                ).astype(np.float32)
                for i, doc in enumerate(train_docs):
                    doc.lbl.clear()
                    doc.lbl.update(
                        label_names[j] for j in range(n_labels)
                        if _mtk_train_preds[i, j] > 0
                    )

            # ── 3.3 rule discovery ───────────────────────────────────────────
            if hp.rule_strategy == "batch":
                rdl_set = run_rule_discovery_batch(
                    train_X, train_y,
                    pool, label_names,
                    rule_eval_docs, rule_eval_X, rule_eval_y,
                    hp, exp_dir,
                    skip_router=args.no_router,
                    global_registered_names=registered_names,
                    base_val_predictions=base_rule_preds,
                    base_val_f1=base_rule_f1,
                    attempt=attempt,
                    precomputed_cluster_labels=_cached_cluster_labels,
                    **accept_kwargs,
                )
            else:
                rdl_set = run_rule_discovery(
                    store, registered_names, label_names,
                    rule_eval_docs, rule_eval_y, hp, exp_dir,
                    attempt=attempt,
                    base_val_predictions=base_rule_preds,
                    base_val_f1=base_rule_f1,
                    **accept_kwargs,
                )

        except Exception as exc:
            log.error("Pipeline error on attempt %d: %s", attempt + 1, exc, exc_info=True)
            issues_history.append((attempt + 1, [f"exception: {exc}"]))
            if attempt < hp.max_retries:
                hp.min_coverage = max(hp.min_coverage * 0.5, 0.001)
                hp.max_trials = min(hp.max_trials * 2, 400)
            continue

        # ── quality check ────────────────────────────────────────────────────
        is_good, issues = assess_quality(rdl_set, hp)
        final_rdl_set = rdl_set

        if is_good:
            log.info("Rule quality check PASSED on attempt %d.", attempt + 1)
            break

        log.warning("Quality issues on attempt %d: %s", attempt + 1, issues)
        issues_history.append((attempt + 1, issues))

        if attempt == hp.max_retries:
            log.error("All %d retries exhausted.", hp.max_retries + 1)
            dump_diagnostic_report(
                hp, label_names, val_y, issues_history, final_rdl_set, exp_dir
            )
            sys.exit(1)

        adjust_hparams(hp, issues, attempt + 1, n_models=len(pool))

    # ── evaluate final rule set on test set ───────────────────────────────────
    n_labels = len(label_names)
    final_micro_f1 = 0.0
    final_macro_f1 = 0.0
    base_test_preds = None

    if best_clf is not None:
        base_test_preds = best_clf.predict(test_X).astype(np.float32)

    if final_rdl_set is not None and len(final_rdl_set.rules) > 0:
        test_docs = []
        for i, t in enumerate(test_X):
            pred_labels = set()
            if base_test_preds is not None:
                pred_labels = {
                    label_names[j]
                    for j in range(n_labels)
                    if base_test_preds[i, j] > 0
                }
            test_docs.append(Document(cnt=t, lbl=pred_labels))

        try:
            if base_test_preds is not None:
                test_metrics = final_rdl_set.evaluate_on_base(
                    test_docs, test_y, base_test_preds,
                    propagate_labels=True,
                )
            else:
                test_metrics = final_rdl_set.evaluate(test_docs, test_y)
            final_micro_f1 = float(test_metrics.get("micro_f1", 0.0))
            final_macro_f1 = float(test_metrics.get("macro_f1", 0.0))
            log.info(
                "Test set (model+rules) — micro-F1=%.4f  macro-F1=%.4f",
                final_micro_f1, final_macro_f1,
            )
        except Exception as exc:
            log.warning("Could not evaluate rule set on test set: %s", exc)
    else:
        final_micro_f1 = baseline_test_micro_f1
        final_macro_f1 = baseline_test_macro_f1

    save_and_print_results(
        final_rdl_set or RDLSet([], label_names),
        baseline_test_micro_f1,
        final_micro_f1,
        exp_dir,
        hp,
        dataset_display_name=dataset_cfg.display_name,
        test_metrics_per_model=test_metrics_per_model,
        baseline_test_macro_f1=baseline_test_macro_f1,
        final_macro_f1=final_macro_f1,
    )

    # ── per-rule correction analysis ──────────────────────────────────────
    if (final_rdl_set is not None and len(final_rdl_set.rules) > 0
            and base_test_preds is not None):
        test_docs_for_analysis = []
        for i, t in enumerate(test_X):
            pred_labels = {
                label_names[j]
                for j in range(n_labels)
                if base_test_preds[i, j] > 0
            }
            test_docs_for_analysis.append(Document(cnt=t, lbl=pred_labels))
        analyze_rule_corrections(
            final_rdl_set, test_docs_for_analysis, test_y,
            base_test_preds, label_names, exp_dir,
        )

    log.info("Pipeline complete. Results in %s", exp_dir)


if __name__ == "__main__":
    main()
