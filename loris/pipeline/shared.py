"""Shared pipeline steps (model init/training, pattern abstraction, model
selection, predicate selection, results reporting).

These dataset-agnostic steps are used by the chase pipeline. Migrated verbatim
from ``run_loris_multi_pipeline.py`` SECTION 2 (Phase 6), imports retargeted to
loris.*. This is the single authoritative home; the legacy multi pipeline now
re-exports these names from here.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from loris.document import Document
from loris.patterns import PatternAbstractor, PatternStore
from loris.predicates import register_ml_model
from loris.models.tfidf_classifier import TFIDFClassifier
from loris.models.neural_classifier import NeuralClassifier
from loris.selection.dynamic_router import FinalSelector, HybridLoss, SelectionNetwork
from loris.rules import RDLSet
from loris.data import HParams, DATASET_REGISTRY, load_data, configure_logging

log = logging.getLogger("loris_pipeline")

# Project root (repo dir). shared.py lives at <root>/loris/pipeline/shared.py,
# so the root is three parents up. Default experiment output goes under
# <root>/experiments/ (matching the legacy pipeline layout).
_ROOT = Path(__file__).resolve().parents[2]


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
    drop_tfidf: bool = False,
) -> Dict[str, object]:
    """Return OrderedDict name → unfitted classifier.

    ``drop_tfidf=True`` builds an EMBEDDING/neural-only pool (textcnn / bilstm /
    pretrained-encoder), excluding the bag-of-words tfidf models. Used to make
    the baseline lexically blind so lexical/text RULES contribute orthogonal,
    non-redundant signal (the rule-delta is measured against this base). Default
    False ⇒ full pool (unchanged / golden-neutral).
    """
    pool: Dict[str, object] = {}

    # ── TF-IDF variants ───────────────────────────────────────────────────────
    if not drop_tfidf:
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
                from loris.models.pretrained_encoder_classifier import (
                    PretrainedEncoderClassifier,
                )
                pool["encoder_mlp"] = PretrainedEncoderClassifier(
                    num_labels=n_labels,
                    model_name=enc_name,
                    classifier_head="mlp",
                    num_epochs=3,
                    batch_size=64,
                    pred_batch_size=128,
                    gradient_checkpointing=False,
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
            from loris.models.lora_slm_classifier import LoRASLMClassifier
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


def _baseline_select_models(
    selector: str,
    pool: Dict[str, object],
    model_names: List[str],
    train_X: List[str], train_y: np.ndarray,
    val_X: List[str], val_y: np.ndarray,
    k: int,
    seed: int = 0,
) -> List[int]:
    """Group-A model selection: return K model indices via a baseline rule.

    The selection LOGIC mirrors loris/baselines/selectors.py but returns indices
    (no ensemble vote) so LORIS's pipeline runs on the K selected models and
    reports LORIS's downstream F1 (paper protocol). Models are fit-as-needed on
    train; predictions evaluated on val.

      random_ms  — K uniformly at random (seeded).
      indiv_ms   — top-K by individual val macro-F1.
      hybrid_llm — difficulty routing: keep the K models that "win" the most val
                   docs (highest per-doc Dice-F1).
      caas       — LinUCB contextual bandit over arms=models; top-K by value.
    """
    import numpy as _np
    from sklearn.metrics import f1_score as _f1

    n = len(model_names)
    k = max(1, min(k, n))

    # random_ms needs no fitting.
    if selector == "random_ms":
        rng = _np.random.RandomState(seed)
        return sorted(rng.choice(n, k, replace=False).tolist())

    # Fit each model (train) and collect val hard preds + per-model val macro.
    val_pred: List[_np.ndarray] = []
    val_macro: List[float] = []
    yval = _np.asarray(val_y, dtype=_np.int32)
    for i, name in enumerate(model_names):
        clf = pool[name]
        try:
            _np.random.seed(42 + seed + i)
            clf.fit(train_X, train_y, val_X, val_y)
            vp = _np.asarray(clf.predict(val_X), dtype=_np.int32)
            vm = float(_f1(yval, vp, average="macro", zero_division=0))
        except Exception:
            vp = _np.zeros_like(yval)
            vm = 0.0
        val_pred.append(vp)
        val_macro.append(vm)

    if selector == "indiv_ms":
        order = _np.argsort(-_np.asarray(val_macro))[:k]
        return sorted(int(i) for i in order)

    # per-doc per-model Dice-F1 (shared by hybrid_llm and caas)
    nval = yval.shape[0]
    f1_dm = _np.zeros((nval, n), dtype=_np.float64)
    for j in range(n):
        p = val_pred[j]
        tp = (p & yval).sum(axis=1).astype(_np.float64)
        denom = (p.sum(axis=1) + yval.sum(axis=1)).astype(_np.float64)
        f1_dm[:, j] = _np.divide(2.0 * tp, denom, out=_np.zeros(nval),
                                 where=denom > 0)

    if selector == "hybrid_llm":
        wins = _np.bincount(f1_dm.argmax(axis=1), minlength=n).astype(_np.float64)
        mass = f1_dm.sum(axis=0)
        key = wins * 1e6 + mass + _np.asarray(val_macro) * 1e-6
        order = _np.argsort(-key)[:k]
        return sorted(int(i) for i in order)

    if selector == "caas":
        # LinUCB over arms=models; context = low-dim lexical features of val docs.
        import math
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        try:
            vec = TfidfVectorizer(max_features=5000, sublinear_tf=True)
            Xv = vec.fit_transform(val_X)
            d = max(2, min(8, Xv.shape[0] - 1, Xv.shape[1] - 1))
            ctx = TruncatedSVD(n_components=d, random_state=seed).fit_transform(Xv)
        except Exception:
            ctx = _np.zeros((nval, 2))
        ctx = _np.hstack([ctx, _np.ones((nval, 1))]).astype(_np.float64)
        d = ctx.shape[1]
        A = [_np.eye(d) for _ in range(n)]
        b = [_np.zeros(d) for _ in range(n)]
        rng = _np.random.RandomState(seed)
        pulls = min(400, nval)
        for t in rng.choice(nval, pulls, replace=False) if pulls else []:
            x = ctx[t]
            ucb = _np.array([
                float((_np.linalg.inv(A[j]) @ b[j]) @ x
                      + math.sqrt(max(x @ _np.linalg.inv(A[j]) @ x, 0)))
                for j in range(n)])
            arm = int(ucb.argmax())
            A[arm] += _np.outer(x, x)
            b[arm] += float(f1_dm[t, arm]) * x
        xbar = ctx.mean(axis=0)
        value = _np.array([float((_np.linalg.inv(A[j]) @ b[j]) @ xbar)
                           for j in range(n)])
        key = value * 1e3 + _np.asarray(val_macro) * 1e-3
        order = _np.argsort(-key)[:k]
        return sorted(int(i) for i in order)

    # unknown selector → fall back to top-K by val macro
    order = _np.argsort(-_np.asarray(val_macro))[:k]
    return sorted(int(i) for i in order)


def run_dynamic_router(
    pool: Dict[str, object],
    train_X: List[str], train_y: np.ndarray,
    val_X: List[str], val_y: np.ndarray,
    hp: HParams,
    exp_dir: Path,
    skip_router: bool = False,
    selector: str = "router",
) -> List[int]:
    t0 = time.time()
    log.info("=== Step 3.2  Dynamic Router ===")
    model_names = list(pool.keys())
    n_models = len(model_names)
    k = min(hp.k_models, n_models)

    # ── Group A: baseline model-selection (replace the router's selected_idx) ──
    # The paper's "model selection" experiment keeps the full LORIS pipeline and
    # only swaps THIS component, then reports LORIS's downstream labeling F1.
    sel = selector if selector != "router" else getattr(hp, "selector", "router")
    if sel and sel != "router":
        idx = _baseline_select_models(
            sel, pool, model_names, train_X, train_y, val_X, val_y, k,
            seed=int(getattr(hp, "seed", 0) or 0))
        log.info("Model selection via '%s' (K=%d): %s",
                 sel, k, [model_names[i] for i in idx])
        return idx

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

    # D-10: precompute each candidate model's per-label TRAIN probabilities once
    # (constants w.r.t. the router). This lets us train on the paper's full hybrid
    # objective L = L_imit + λ_task·L_task − λ_ent·L_ent, not imitation alone. The
    # task loss back-propagates through the DifferentiableTopK mask into the
    # SelectionNetwork, so the router learns to pick the models that maximise
    # downstream multi-label accuracy (previously it only mimicked an oracle and
    # could drop the strongest model — e.g. the encoder).
    log.info("Precomputing per-model train probabilities for router task loss …")
    _model_probs = []
    for name in model_names:
        try:
            p = np.asarray(pool[name].predict_proba(train_X), dtype=np.float32)
        except Exception as exc:
            log.debug("router task-loss proba: model %s failed — %s", name, exc)
            p = np.zeros((len(train_X), train_y.shape[1]), dtype=np.float32)
        _model_probs.append(torch.from_numpy(p))
    model_probs_t = torch.stack(_model_probs).to(device)                  # (n, B, L)
    labels_t = torch.from_numpy(np.asarray(train_y, dtype=np.float32)).to(device)  # (B, L)

    log.info("Training SelectionNetwork for %d epochs …", hp.router_epochs)
    net.train()
    for epoch in range(1, hp.router_epochs + 1):
        mask, scores = net(X_t, return_scores=True)
        l_imit = loss_fn.imitation_loss(scores, oracle_t)
        l_task = loss_fn.task_loss_multilabel(mask, model_probs_t, labels_t)
        l_ent = loss_fn.entropy_loss(scores)
        loss = l_imit + loss_fn.lambda_task * l_task - loss_fn.lambda_ent * l_ent

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if epoch % 5 == 0 or epoch == hp.router_epochs:
            log.info(
                "  [Router epoch %3d/%d]  L_imit=%.4f  L_task=%.4f  L_ent=%.4f  L_total=%.4f",
                epoch, hp.router_epochs, l_imit.item(), l_task.item(),
                l_ent.item(), loss.item(),
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

def _alt_pattern_scores(method: str, F, L, coverages, val_y):
    """Group-C predicate ranking criteria over the fire-mask × label matrix.

    F: (n_pred, n_docs) bool/float predicate firings on val.
    L: (n_docs, n_labels) float multi-hot labels.
    Returns a length-n_pred score array (higher = keep).

      filter_mi    — summed mutual information between each predicate's firing
                     and each label (binary MI), over labels.
      filter_chi2  — summed chi-square statistic, over labels.
      weshap       — Monte-Carlo Shapley: marginal gain of each predicate to the
                     val macro-F1 of a cheap logistic ensemble over random
                     permutations (approximate; cost noted in plan).
      localboost   — greedy boosting: iteratively add the predicate that most
                     improves the current logistic ensemble's val macro-F1; rank
                     = negative add-order (earlier picks score higher).
    """
    import numpy as _np
    n_pred, n_docs = F.shape
    Fb = (F > 0).astype(_np.int8)            # (n_pred, n_docs)
    Y = (val_y > 0).astype(_np.int8)         # (n_docs, n_labels)
    n_lbl = Y.shape[1]

    if method in ("filter_mi", "filter_chi2"):
        from sklearn.feature_selection import mutual_info_classif, chi2
        Xp = Fb.T                            # (n_docs, n_pred) features = predicates
        agg = _np.zeros(n_pred, dtype=_np.float64)
        for j in range(n_lbl):
            yj = Y[:, j]
            if yj.sum() == 0 or yj.sum() == n_docs:
                continue
            if method == "filter_mi":
                agg += mutual_info_classif(Xp, yj, discrete_features=True,
                                           random_state=0)
            else:
                sc, _ = chi2(Xp, yj)
                agg += _np.nan_to_num(sc)
        return agg

    # weshap / localboost use a cheap per-label logistic ensemble val macro-F1.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score as _f1
    # split val in half: fit on one, score Shapley/boost gain on the other
    rng = _np.random.RandomState(0)
    perm = rng.permutation(n_docs)
    half = n_docs // 2
    fit_idx, ev_idx = perm[:half], perm[half:]
    Xp = Fb.T

    def _macro_with(cols):
        if not cols:
            return 0.0
        Xf, Xe = Xp[fit_idx][:, cols], Xp[ev_idx][:, cols]
        preds = _np.zeros((len(ev_idx), n_lbl), dtype=_np.int8)
        for j in range(n_lbl):
            yj = Y[fit_idx, j]
            if yj.sum() == 0 or yj.sum() == len(yj):
                continue
            try:
                lr = LogisticRegression(max_iter=120, solver="liblinear")
                lr.fit(Xf, yj)
                preds[:, j] = lr.predict(Xe)
            except Exception:
                pass
        return float(_f1(Y[ev_idx], preds, average="macro", zero_division=0))

    if method == "weshap":
        # Monte-Carlo Shapley over predicates (small budget for tractability).
        n_perm = 4
        cap = min(n_pred, 250)               # cap candidate pool for cost
        cand = list(_np.argsort(-coverages)[:cap])
        contrib = _np.zeros(n_pred, dtype=_np.float64)
        for _ in range(n_perm):
            order = list(cand)
            rng.shuffle(order)
            cur, prev = [], 0.0
            for p in order:
                cur.append(p)
                f = _macro_with(cur)
                contrib[p] += (f - prev)
                prev = f
        return contrib

    if method == "localboost":
        # greedy forward selection by val-macro gain; earlier picks rank higher.
        cap = min(n_pred, 250)
        cand = set(_np.argsort(-coverages)[:cap].tolist())
        chosen, scores = [], _np.zeros(n_pred, dtype=_np.float64)
        prev = 0.0
        budget = min(60, len(cand))
        for step in range(budget):
            best_p, best_f = None, prev
            for p in list(cand):
                f = _macro_with(chosen + [p])
                if f > best_f:
                    best_f, best_p = f, p
            if best_p is None:
                break
            chosen.append(best_p)
            cand.discard(best_p)
            scores[best_p] = float(budget - step)  # earlier = higher
            prev = best_f
        # un-chosen but candidate predicates get a small coverage-based tail
        return scores

    return _np.asarray(coverages, dtype=_np.float64)


def _select_top_predicates(
    store: PatternStore,
    val_docs: List[Document],
    val_y: np.ndarray,
    label_names: List[str],
    top_k: int = 100,
    per_type_top_k: Optional[int] = None,
    pattern_select: str = "loris",
) -> Tuple[list, Optional[np.ndarray]]:
    """Select top predicates using hybrid scoring: phi * log1p(coverage * 100).

    Returns ``(selected_preds, fire_masks)`` where *fire_masks* is a bool
    array of shape ``(n_selected, n_val_docs)`` that can be passed downstream
    to avoid recomputing ``pred(doc)`` in ``precompute_fire_masks()``.

    ``pattern_select`` (Group C, paper "varying pattern selection"): ranks the
    candidate **predicates** by an alternative criterion instead of LORIS's
    hybrid-phi, then keeps top-k the same way. Options: 'loris' (default,
    hybrid-phi), 'filter_mi' (summed mutual information over labels), 'filter_chi2'
    (summed chi-square), 'weshap' (Monte-Carlo Shapley over the predicate set),
    'localboost' (greedy boosting: iteratively add the predicate that most
    improves val macro-F1 of the current logistic ensemble). All operate on the
    same predicate fire-mask × label matrix, so only the SELECTION changes.

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
    # scored: (score, phi, cov, pred, original_candidate_index)
    # Group C: replace the hybrid-phi score with an alternative predicate ranking
    # criterion when pattern_select != "loris" (paper "varying pattern selection").
    alt = None
    if pattern_select and pattern_select != "loris":
        alt = _alt_pattern_scores(pattern_select, F, L, coverages, val_y)
        log.info("  pattern selection via '%s' over %d predicates",
                 pattern_select, F.shape[0])

    scored: List[Tuple[float, float, float, object, int]] = []
    for idx_in_valid, orig_idx in enumerate(valid_indices):
        bp = float(best_phis[idx_in_valid])
        cov = float(coverages[idx_in_valid])
        if alt is not None:
            # alternative criterion drives ranking; keep a tiny phi floor so
            # pure-noise predicates (no label association at all) are dropped.
            if bp >= 0.0:
                scored.append((float(alt[idx_in_valid]), bp, cov,
                               candidates[orig_idx], int(orig_idx)))
        elif bp >= 0.02:
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
            selector=getattr(hp, "selector", "router"),
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
    from loris.predicates import (
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
        from loris.rules.sim_graph import compute_embeddings, build_sim_graph
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
    _loo_rows = []  # (fire_mask, label_idx, op, stage) for leave-one-out marginal
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
            "stage": (rule.val_stats or {}).get("stage", "unknown")
                     if getattr(rule, "val_stats", None) else "unknown",
        }
        analysis.append(rule_info)
        _loo_rows.append((fires.copy(), label_idx,
                          getattr(rule, "consequence_op", "add"),
                          rule_info["stage"]))

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

    # ── Leave-one-out MARGINAL contribution (combined prediction, not isolated) ──
    # Build the full cumulative prediction from fixed per-rule fire masks
    # (add→pos, remove→neg, final pos & ~neg), then drop each rule and measure
    # the macro-F1 delta. This shows a REMOVE that fixes an ADD's FP as positive,
    # which the per-rule isolation simulation above cannot.
    stage_rollup: Dict[str, Dict[str, float]] = {}
    if _loo_rows:
        add_count = np.zeros((n_docs, n_labels), dtype=np.int32)
        rem_count = np.zeros((n_docs, n_labels), dtype=np.int32)
        for fmask, lidx, op, _stage in _loo_rows:
            if op == "remove":
                rem_count[fmask, lidx] += 1
            else:
                add_count[fmask, lidx] += 1
        base_pos = (np.asarray(base_predictions) > 0)
        full_pos = base_pos | (add_count > 0)
        full_neg = (rem_count > 0)
        full_pred = (full_pos & ~full_neg).astype(np.float32)
        full_macro = float(f1_score(y_true, full_pred, average="macro", zero_division=0))

        for i, (fmask, lidx, op, stage) in enumerate(_loo_rows):
            col = full_pred[:, lidx].copy()
            if op == "remove":
                neg_wo = (rem_count[:, lidx] - fmask.astype(np.int32)) > 0
                pos_l = base_pos[:, lidx] | (add_count[:, lidx] > 0)
                new_col = (pos_l & ~neg_wo)
            else:
                pos_wo = base_pos[:, lidx] | ((add_count[:, lidx] - fmask.astype(np.int32)) > 0)
                neg_l = rem_count[:, lidx] > 0
                new_col = (pos_wo & ~neg_l)
            if np.array_equal(col > 0, new_col):
                marginal = 0.0
            else:
                pred_wo = full_pred.copy()
                pred_wo[:, lidx] = new_col.astype(np.float32)
                wo_macro = float(f1_score(y_true, pred_wo, average="macro", zero_division=0))
                marginal = full_macro - wo_macro
            analysis[i]["loo_marginal_f1"] = marginal
            r = stage_rollup.setdefault(stage, {"n_rules": 0, "loo_marginal_f1_sum": 0.0})
            r["n_rules"] += 1
            r["loo_marginal_f1_sum"] += marginal

        lines.append("")
        lines.append("  PER-STAGE leave-one-out marginal macro-F1 (combined):")
        lines.append(f"  full model+rules macro-F1 = {full_macro:.4f}")
        for stage, r in sorted(stage_rollup.items(), key=lambda kv: -kv[1]["loo_marginal_f1_sum"]):
            lines.append(f"    {stage:18s}  n={r['n_rules']:3d}  "
                         f"Σ marginal-F1 = {r['loo_marginal_f1_sum']:+.4f}")

    lines.append("=" * 72)
    report = "\n".join(lines)
    print("\n" + report)

    with open(exp_dir / "rule_analysis.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(exp_dir / "rule_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    with open(exp_dir / "rule_analysis_stage_rollup.json", "w", encoding="utf-8") as f:
        json.dump(stage_rollup, f, indent=2, ensure_ascii=False)

    log.info("Rule correction analysis saved to %s", exp_dir / "rule_analysis.txt")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

