"""
run_loris_chase_pipeline.py
============================
Chase-driven LORIS pipeline with dual-track Bayesian Optimisation.

Uses ChaseRuleLearner with:
  - Track 1 (seed rules): ML + text predicates only, add-only consequence
  - Track 2 (propagation): SimPredicate + LabelPredicate, add-only
  - Sparse adjacency graph for document similarity (SpMV-based evaluation)

Usage
-----
  python run_loris_chase_pipeline.py --dataset aapd --rule_strategy batch [options]

  Chase-specific options:
    --track2_label_source {track1,gt,model}
    --sim_threshold_bins "0.80,0.85,0.88,0.92,0.95"
"""

from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    if os.environ.get(_k, "0") in ("", "0"):
        os.environ[_k] = "4"

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score

# ── make project root importable ──────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── 从原版 multi pipeline 复用全部共享逻辑 ──────────────────────────────────
from run_loris_multi_pipeline import (
    # Dataset
    DATASET_REGISTRY,
    DatasetConfig,
    HParams,
    # Pipeline steps
    configure_logging,
    load_data,
    init_models,
    train_models,
    evaluate_models_on_test,
    run_pattern_abstraction,
    run_dynamic_router,
    register_selected_models,
    _select_top_predicates,
    _select_cluster_models,
    # Output
    save_and_print_results,
    analyze_rule_corrections,
    # Helpers
    _setup_experiment,
    _PredictWrapper,
)

from pattern_extraction import Document, PatternAbstractor, PatternStore
from pattern_extraction.predicates import register_ml_model, LabelPredicate
from rule_discovery import RDLSet
from rule_discovery.chase_rule_discovery import (
    ChaseRuleLearner, _save_stage_logs, _extract_trial_stats,
    _vectorized_staged_predict, precompute_fire_masks, precompute_ml_proba,
    MLThresholdPredicate, _fast_per_label_f1,
    _error_driven_filter, _tree_seeded_rules, ERROR_AWARE_WEIGHTS,
)
from rule_discovery.sim_graph import (
    auto_threshold_bins,
    compute_embeddings,
    build_sim_graph,
    save_sim_graphs,
    load_sim_graphs,
    precompute_neighbor_label_masks,
    precompute_neighbor_label_counts,
)

log = logging.getLogger("loris_chase_pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# 重写 Step 3.3 — 使用 ChaseRuleLearner 替代 RuleLearner
# ══════════════════════════════════════════════════════════════════════════════

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
    # ── split mode ──
    fit_docs: Optional[List[Document]] = None,
    fit_labels: Optional[np.ndarray] = None,
    fit_base_predictions: Optional[np.ndarray] = None,
) -> RDLSet:
    """Greedy 模式规则发现 — 使用 ChaseRuleLearner。"""
    import optuna

    t0 = time.time()
    log.info("=== Step 3.3  Chase Rule Discovery (greedy) ===")

    filtered_preds, _val_fire_masks = _select_top_predicates(
        store, val_docs, val_y, label_names,
        top_k=hp.predicate_top_k,
        per_type_top_k=getattr(hp, "per_type_top_k", None),
    )
    # ── 自适应 max_trials：按标签数扩展搜索预算 ──
    effective_max_trials = max(hp.max_trials, 15 * len(label_names))
    log.info(
        "Candidate predicates=%d→%d (top-k by hybrid scoring)  "
        "ML models=%d  labels=%d  max_trials=%d (adaptive from %d)  top_n=%d",
        len(store), len(filtered_preds), len(registered_model_names),
        len(label_names), effective_max_trials, hp.max_trials, hp.top_n_rules,
    )

    if not log.isEnabledFor(logging.DEBUG):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── 自适应 min_rule_fires ──
    _eval_n = len(val_docs)
    effective_min_fires = max(3, min(hp.min_rule_fires, int(_eval_n * 0.005)))
    log.info("Adaptive min_rule_fires: hp=%d, val_size=%d → effective=%d",
             hp.min_rule_fires, _eval_n, effective_min_fires)

    _fit_kw = {}
    if fit_docs is not None:
        _fit_kw = dict(
            fit_docs=fit_docs,
            fit_labels=fit_labels,
            fit_base_predictions=fit_base_predictions,
        )
    learner = ChaseRuleLearner(
        candidate_predicates=filtered_preds,
        candidate_ml_models=registered_model_names,
        label_list=label_names,
        val_docs=val_docs,
        val_labels=val_y,
        max_trials=effective_max_trials,
        top_n=hp.top_n_rules,
        min_coverage=hp.min_coverage_rule,
        seed=42,
        storage_path=str(exp_dir / f"optuna_hybrid_attempt{attempt}"),
        verbose=log.isEnabledFor(logging.DEBUG),
        base_predictions=base_val_predictions,
        base_f1=base_val_f1,
        top_per_type=hp.top_per_type,
        rule_min_precision=hp.rule_min_precision,
        accept_docs=accept_docs,
        accept_labels=accept_y,
        accept_predictions=accept_predictions,
        accept_f1=accept_f1,
        min_rule_fires=effective_min_fires,
        min_n_changes=hp.min_n_changes,
        min_corr_prec=hp.min_corr_prec,
        precomputed_val_fire_masks=_val_fire_masks,
        **_fit_kw,
    )
    rdl_set = learner.discover()

    out_path = str(exp_dir / "rules.json")
    rdl_set.save(out_path)

    log.info(
        "Chase Rule Discovery done in %.1fs — %d rules found, saved to %s",
        time.time() - t0, len(rdl_set.rules), out_path,
    )
    if rdl_set.rules:
        log.info("Top-5 rules by F1-gain:")
        top5 = sorted(rdl_set.rules, key=lambda r: r.score, reverse=True)[:5]
        for i, r in enumerate(top5, 1):
            log.info("  #%d  [%s]  gain=%.4f  cov=%.3f  body=%s",
                     i, r.consequence, r.score, r.coverage, repr(r))
    return rdl_set


def _generate_label_cooccurrence_rules(
    val_labels: np.ndarray,
    label_names: List[str],
    base_predictions: np.ndarray,
    min_cooccur: int = 10,
    min_precision: float = 0.60,
) -> List["RDL"]:
    """Generate LabelPredicate rules from label co-occurrence patterns.

    For each pair (A, B) where A is frequently predicted correctly and B is
    frequently missed in documents that have A, create rule: label(A) → +B.
    Evaluate precision on the provided data and filter.
    """
    from rule_discovery.loris_rule_discovery import RDL
    n_docs, n_labels = val_labels.shape
    label2idx = {name: i for i, name in enumerate(label_names)}
    pred_binary = (np.asarray(base_predictions) > 0).astype(int)
    gt = np.asarray(val_labels, dtype=int)

    rules = []
    for a_idx, a_label in enumerate(label_names):
        a_predicted = pred_binary[:, a_idx] == 1
        if a_predicted.sum() < min_cooccur:
            continue
        for b_idx, b_label in enumerate(label_names):
            if a_idx == b_idx:
                continue
            fn_b = (gt[:, b_idx] == 1) & (pred_binary[:, b_idx] == 0)
            cooccur_fn = int((a_predicted & fn_b).sum())
            if cooccur_fn < min_cooccur:
                continue
            fires_mask = a_predicted & (pred_binary[:, b_idx] == 0)
            n_fires = int(fires_mask.sum())
            if n_fires == 0:
                continue
            tp = int((fires_mask & (gt[:, b_idx] == 1)).sum())
            prec = tp / n_fires
            if prec < min_precision:
                continue
            rule = RDL(
                body=(LabelPredicate(label=a_label, op="contains"),),
                consequence=b_label,
                consequence_op="add",
                score=prec,
                coverage=n_fires / n_docs,
            )
            rules.append(rule)

    rules.sort(key=lambda r: -r.score)
    return rules


def _extract_error_driven_predicates(
    docs: list,
    labels: np.ndarray,
    predictions: np.ndarray,
    label_names: list,
    top_k_per_label: int = 8,
    mode: str = "fp",
):
    """Extract text predicates that discriminate FP (or FN) from correct predictions."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from pattern_extraction.predicates import MatchPredicate

    new_preds = []
    for lidx, lname in enumerate(label_names):
        if mode == "fp":
            error_mask = (predictions[:, lidx] == 1) & (labels[:, lidx] == 0)
            correct_mask = (predictions[:, lidx] == 1) & (labels[:, lidx] == 1)
        else:
            error_mask = (predictions[:, lidx] == 0) & (labels[:, lidx] == 1)
            correct_mask = (predictions[:, lidx] == 0) & (labels[:, lidx] == 0)

        n_err = int(error_mask.sum())
        n_corr = int(correct_mask.sum())
        if n_err < 3 or n_corr < 3:
            continue

        all_mask = error_mask | correct_mask
        indices = np.where(all_mask)[0]
        texts = [docs[i].cnt if hasattr(docs[i], 'cnt') else docs[i]
                 for i in indices]

        vec = TfidfVectorizer(max_features=500, stop_words='english',
                              ngram_range=(1, 2), min_df=2)
        try:
            X = vec.fit_transform(texts)
        except ValueError:
            continue

        err_idx = np.where(error_mask[all_mask])[0]
        corr_idx = np.where(correct_mask[all_mask])[0]
        if len(err_idx) == 0 or len(corr_idx) == 0:
            continue

        err_mean = X[err_idx].mean(axis=0).A1
        corr_mean = X[corr_idx].mean(axis=0).A1
        diff = err_mean - corr_mean

        top_indices = np.argsort(-diff)[:top_k_per_label]
        feature_names = vec.get_feature_names_out()

        for idx in top_indices:
            if diff[idx] < 0.01:
                break
            word = feature_names[idx]
            if len(word) < 3:
                continue
            new_preds.append(MatchPredicate("cnt", word))

    seen = set()
    unique = []
    for p in new_preds:
        key = (p.r.raw, p.attr)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ══════════════════════════════════════════════════════════════════════════════
# 重写 Step 3.3b — batch 模式，使用 ChaseRuleLearner
# ══════════════════════════════════════════════════════════════════════════════

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
    # ── split mode ──
    fit_docs: Optional[List[Document]] = None,
    fit_labels: Optional[np.ndarray] = None,
    fit_base_predictions: Optional[np.ndarray] = None,
    # ── two_val mode: independent data for batch_select ──
    select_docs: Optional[List[Document]] = None,
    select_y: Optional[np.ndarray] = None,
    select_base_predictions: Optional[np.ndarray] = None,
    select_f1: Optional[float] = None,
    # ── resume support ──
    resume_dir: Optional[str] = None,
    resume_mode: str = "all",  # "all" = skip BO too; "prep" = only model+patterns
    # ── Track 2 uses original model baseline even when Track 1 is blank ──
    model_val_predictions: Optional[np.ndarray] = None,  # BO val model preds
    model_val_f1: Optional[float] = None,
    model_select_predictions: Optional[np.ndarray] = None,  # select val model preds
    model_select_f1: Optional[float] = None,
    # ── error-driven / tree warmup flags ──
    no_error_driven_filter: bool = False,
    no_tree_warmup: bool = False,
    train_docs: Optional[List[Document]] = None,
    group_min_corr_prec: float = 0.50,
) -> RDLSet:
    """Batch 模式规则发现 — per-cluster ChaseRuleLearner BO + 全局贪心筛选。"""
    import optuna
    from sklearn.cluster import KMeans

    t0 = time.time()
    log.info("=== Step 3.3b  Chase Batch Rule Discovery (per-cluster BO) ===")
    label2idx = {name: i for i, name in enumerate(label_names)}

    if global_registered_names is None:
        global_registered_names = []

    if not log.isEnabledFor(logging.DEBUG):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── 1. 聚类训练文档 ─────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer

    if precomputed_cluster_labels is not None:
        cluster_labels = precomputed_cluster_labels
        n_clusters = int(cluster_labels.max()) + 1
        log.info("Chase batch: reusing Step 3.1 clusters for %d training docs "
                 "(%d clusters, model_selection=%s)",
                 len(train_X), n_clusters, hp.cluster_model_selection)
    else:
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = st_model.encode(train_X, show_progress_bar=False)

        best_km = None  # set by auto-k loop below; stays None if n_clusters is fixed
        if hp.n_clusters is not None:
            n_clusters = hp.n_clusters
        else:
            from sklearn.metrics import silhouette_score

            best_k, best_sil, best_km = 2, -1.0, None
            # 采样防 OOM：silhouette_score 计算 O(N^2) 距离矩阵
            _SIL_MAX = 5000
            if len(embeddings) > _SIL_MAX:
                _sil_idx = np.random.default_rng(42).choice(
                    len(embeddings), size=_SIL_MAX, replace=False)
                _sil_emb = embeddings[_sil_idx]
            else:
                _sil_idx, _sil_emb = None, embeddings
            for k in range(2, min(16, len(train_X))):
                km_tmp = KMeans(n_clusters=k, random_state=42, n_init=5)
                km_labels = km_tmp.fit_predict(embeddings)
                _sil_labels = km_labels[_sil_idx] if _sil_idx is not None else km_labels
                sil = silhouette_score(_sil_emb, _sil_labels)
                if sil > best_sil:
                    best_k, best_sil, best_km = k, sil, km_tmp
            n_clusters = best_k

        if best_km is not None:
            # 16H: 复用 auto-k 最优 KMeans，无需重新 fit
            km = best_km
            cluster_labels = km.labels_
        else:
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = km.fit_predict(embeddings)
        log.info("Chase batch: clustered %d training docs into %d clusters "
                 "(model_selection=%s)",
                 len(train_X), n_clusters, hp.cluster_model_selection)

    # Always compute val_cluster_labels for per-cluster val filtering
    if precomputed_cluster_labels is not None:
        # 没有 st_model/km（用了预计算 labels），需要加载
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = st_model.encode(train_X, show_progress_bar=False)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        km.fit(embeddings)
    val_embeddings = st_model.encode(
        [d.cnt for d in val_docs], show_progress_bar=False
    )
    val_cluster_labels = km.predict(val_embeddings)

    # ── 2. Per-cluster: pattern abstraction + model selection + ChaseRuleLearner BO
    # 分为两阶段：
    #   Phase A (sequential): 提取谓词 + 选择模型 + 创建 learner（涉及共享资源）
    #   Phase B (parallel):   run_bo()（最耗时，仅使用预计算掩码，CPU-bound）
    all_trials = []
    import json as _json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # ── 自适应 min_rule_fires ──
    # 根据 BO 评估数据集 (val_bo) 大小自适应，避免超参与数据量不匹配
    _eval_n_bo = len(val_docs)  # BO 评估在 val_bo (或 single-val 模式下的 val) 上
    effective_min_fires = max(3, min(hp.min_rule_fires, int(_eval_n_bo * 0.005)))
    log.info("Adaptive min_rule_fires: hp.min_rule_fires=%d, val_bo_size=%d → effective=%d",
             hp.min_rule_fires, _eval_n_bo, effective_min_fires)

    # ── Phase A: sequential setup ──
    _learners: dict[int, ChaseRuleLearner] = {}  # cid → learner
    for cid in range(n_clusters):
        cluster_indices = [i for i, c in enumerate(cluster_labels) if c == cid]
        if not cluster_indices:
            log.warning("Cluster %d is empty, skipping.", cid)
            continue

        cluster_train_X = [train_X[i] for i in cluster_indices]
        cluster_train_y = train_y[cluster_indices]

        log.info("  Cluster %d: %d training docs", cid, len(cluster_train_X))

        # ── Check for cached PatternStore ──
        _store_cache = exp_dir / f"cluster_{cid}_store.json"
        _resume_store = (Path(resume_dir) / f"cluster_{cid}_store.json"
                         if resume_dir else None)
        _rerun_patterns = getattr(hp, '_rerun_patterns', False)
        if _resume_store and _resume_store.exists() and not _rerun_patterns:
            store = PatternStore.load(str(_resume_store))
            log.info("  Cluster %d: loaded cached PatternStore (%d predicates)", cid, len(store))
        else:
            if _rerun_patterns and _resume_store and _resume_store.exists():
                log.info("  Cluster %d: --rerun_patterns set; rebuilding PatternStore", cid)
            abstractor = PatternAbstractor(
                n_clusters=1,
                min_coverage=hp.min_coverage,
                max_entropy_threshold=hp.max_entropy_threshold,
                tfidf_top_k=hp.tfidf_top_k,
                random_state=42,
                pattern_mode=hp.pattern_mode,
                extra_stop_words=hp.extra_stop_words,
                anchor_min_df=hp.anchor_min_df,
                sim_threshold=getattr(hp, "sim_threshold", None),
                encoder=st_model if precomputed_cluster_labels is None else None,
                glove_path=getattr(hp, "glove_path", None),
            )
            try:
                # Pass full Documents (with ttl/mtd) when available so
                # PatternAbstractor mines title/metadata anchors as well.
                if train_docs is not None:
                    cluster_train_docs = [train_docs[i] for i in cluster_indices]
                    abstractor.fit(cluster_train_docs, cluster_train_y)
                else:
                    abstractor.fit(cluster_train_X, cluster_train_y)
            except Exception as exc:
                log.warning("  Cluster %d pattern abstraction failed: %s", cid, exc)
                continue
            store = abstractor.to_store()
            # Save for future resume
            try:
                store.save(str(_store_cache))
            except Exception:
                pass
        if len(store) == 0:
            log.warning("  Cluster %d: 0 patterns extracted, skipping.", cid)
            continue

        # ── Filter validation docs to this cluster ──
        cluster_val_mask = (val_cluster_labels == cid)
        cluster_val_indices = np.where(cluster_val_mask)[0]
        cluster_val_docs = [val_docs[i] for i in cluster_val_indices]
        cluster_val_y = val_y[cluster_val_indices]

        if len(cluster_val_docs) < 5:
            log.warning("  Cluster %d: only %d val docs, skipping.", cid, len(cluster_val_docs))
            continue

        cluster_base_preds = base_val_predictions[cluster_val_indices]
        cluster_base_f1 = float(f1_score(
            cluster_val_y, cluster_base_preds, average="macro", zero_division=0))

        log.info("  Cluster %d: %d/%d val docs, cluster base macro-F1=%.4f",
                 cid, len(cluster_val_docs), len(val_docs), cluster_base_f1)

        filtered_preds, _val_fm = _select_top_predicates(
            store, cluster_val_docs, cluster_val_y, label_names,
            top_k=hp.predicate_top_k,
            per_type_top_k=getattr(hp, "per_type_top_k", None),
        )
        if not filtered_preds:
            log.warning("  Cluster %d: 0 filtered predicates, skipping.", cid)
            continue

        # ── 谓词类型分布诊断 ──
        _type_counts: dict[str, int] = {}
        for _p in filtered_preds:
            _tname = type(_p).__name__
            _type_counts[_tname] = _type_counts.get(_tname, 0) + 1
        log.info("  Cluster %d predicate type breakdown: %s", cid, _type_counts)

        # 文本谓词覆盖率统计
        _TEXTUAL_TYPES = {"MatchPredicate", "CooccurPredicate", "BeforePredicate", "FreqPredicate"}
        _textual_covs = []
        _ml_covs = []
        for _pi, _p in enumerate(filtered_preds):
            _cov = float(_val_fm[_pi].sum()) / max(len(cluster_val_docs), 1)
            if type(_p).__name__ in _TEXTUAL_TYPES:
                _textual_covs.append(_cov)
            else:
                _ml_covs.append(_cov)
        if _textual_covs:
            _sorted_tc = sorted(_textual_covs)
            log.info("  Cluster %d textual pred coverage (n=%d): "
                     "min=%.4f median=%.4f max=%.4f mean=%.4f",
                     cid, len(_textual_covs),
                     _sorted_tc[0], _sorted_tc[len(_sorted_tc) // 2],
                     _sorted_tc[-1], sum(_sorted_tc) / len(_sorted_tc))
        if _ml_covs:
            _sorted_mc = sorted(_ml_covs)
            log.info("  Cluster %d ML pred coverage (n=%d): "
                     "min=%.4f median=%.4f max=%.4f mean=%.4f",
                     cid, len(_ml_covs),
                     _sorted_mc[0], _sorted_mc[len(_sorted_mc) // 2],
                     _sorted_mc[-1], sum(_sorted_mc) / len(_sorted_mc))

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

        # ── 自适应 max_trials ──
        effective_max_trials = max(hp.max_trials, 15 * len(label_names))
        log.info("  Cluster %d: %d→%d predicates, %d ML models, "
                 "running Chase BO with %d trials (adaptive from %d)",
                 cid, len(store), len(filtered_preds),
                 len(cluster_ml_names), effective_max_trials, hp.max_trials)

        cluster_label_indices = np.where(cluster_train_y.sum(axis=0) > 0)[0].tolist()

        _fit_kw = {}
        if fit_docs is not None:
            cluster_fit_indices = np.where(cluster_labels == cid)[0]
            cluster_fit_docs = [fit_docs[i] for i in cluster_fit_indices]
            cluster_fit_labels = fit_labels[cluster_fit_indices]
            cluster_fit_base_preds = (fit_base_predictions[cluster_fit_indices]
                                      if fit_base_predictions is not None else None)
            _fit_kw = dict(
                fit_docs=cluster_fit_docs,
                fit_labels=cluster_fit_labels,
                fit_base_predictions=cluster_fit_base_preds,
            )
        # BO 评估使用全局 val_bo（不分 cluster），获得更大的 fires 空间
        learner = ChaseRuleLearner(
            candidate_predicates=filtered_preds,
            candidate_ml_models=cluster_ml_names,
            label_list=label_names,
            val_docs=val_docs,
            val_labels=val_y,
            max_trials=effective_max_trials,
            min_coverage=hp.min_coverage_rule,
            seed=42 + cid,
            storage_path=str(exp_dir / f"optuna_chase_batch_c{cid}_a{attempt}"),
            verbose=log.isEnabledFor(logging.DEBUG),
            base_predictions=base_val_predictions,
            base_f1=float(f1_score(val_y, base_val_predictions,
                                   average="macro", zero_division=0)),
            top_per_type=hp.top_per_type,
            rule_min_precision=hp.rule_min_precision,
            metric_mode=hp.batch_metric_mode,
            cluster_label_indices=cluster_label_indices,
            min_rule_fires=effective_min_fires,
            min_n_changes=hp.min_n_changes,
            min_corr_prec=hp.min_corr_prec,
            precomputed_val_fire_masks=None,
            track="base",  # Track 1: no sim/label predicates
            **_fit_kw,
        )
        _learners[cid] = learner

    # ── Per-cluster model selection summary ──
    if hp.cluster_model_selection != "global":
        log.info("Per-cluster model selection summary (mode=%s, k_models=%d):",
                 hp.cluster_model_selection, hp.k_models)
        for cid, learner in _learners.items():
            _ml_short = [n.split("loris_clf_")[-1] if "loris_clf_" in n else n
                         for n in learner.candidate_ml_models]
            log.info("  Cluster %d (%d models): %s", cid, len(_ml_short), _ml_short)

    # ── Compute staged flag early (needed to skip Phase B BO) ──
    _is_blank_early = getattr(hp, 'track1_baseline', 'model') == 'blank'
    _staged_early = _is_blank_early and getattr(hp, 'staged_pipeline', True)

    # ── Phase B: parallel BO execution (or reload from cache) ──
    import pickle as _pkl

    _resume_trials_pkl = (Path(resume_dir) / "all_trials.pkl"
                          if resume_dir and resume_mode == "all" else None)
    _n_workers = min(len(_learners), max(4, os.cpu_count() // 2))

    def _run_bo_for_cluster(cid_learner):
        _cid, _learner = cid_learner
        return _cid, _learner, _learner.run_bo()

    if _staged_early:
        # Staged mode: skip Phase B BO entirely, just precompute caches
        log.info("Staged mode: skipping Phase B BO, precomputing caches only")
        for cid, learner in _learners.items():
            learner._precompute()
        all_trials = []
    elif _resume_trials_pkl and _resume_trials_pkl.exists():
        log.info("[RESUME] Loading cached Track 1 BO trials from %s",
                 _resume_trials_pkl)
        with open(_resume_trials_pkl, "rb") as _pf:
            _cached_ctx = _pkl.load(_pf)
        all_trials = _cached_ctx["all_trials"]
        log.info("[RESUME] Loaded %d trials — skipping Track 1 BO entirely",
                 len(all_trials))
    else:
        log.info("Running BO for %d clusters with %d parallel workers",
                 len(_learners), _n_workers)

        if _n_workers <= 1:
            # Single cluster — no threading overhead
            _bo_results = []
            for cid, learner in _learners.items():
                _bo_results.append((cid, learner, learner.run_bo()))
        else:
            _bo_results = []
            with ThreadPoolExecutor(max_workers=_n_workers) as executor:
                futures = {
                    executor.submit(_run_bo_for_cluster, (cid, learner)): cid
                    for cid, learner in _learners.items()
                }
                for future in as_completed(futures):
                    _bo_results.append(future.result())

        # ── Phase C: collect results + save checkpoints ──
        for cid, learner, raw_trials in sorted(_bo_results, key=lambda x: x[0]):
            trials = [(t, rdl) for t, rdl in raw_trials
                      if t.value >= hp.min_f1_gain]
            all_trials.extend(trials)

            log.info("  Cluster %d: %d trials surviving min_f1_gain filter",
                     cid, len(trials))

            # ── 逐 cluster 保存 trial checkpoint ──
            _ckpt = {
                "cluster_id": int(cid),
                "n_trials_total": len(raw_trials),
                "n_trials_positive": len(trials),
                "best_value": max((t.value for t, _ in raw_trials), default=0.0),
                "trials": [
                    {"number": t.number, "value": t.value,
                     "params": t.params, "state": t.state.name}
                    for t, _ in raw_trials
                ],
            }
            _ckpt_path = exp_dir / f"cluster_{cid}_trials.json"
            with open(_ckpt_path, "w") as _f:
                _json.dump(_ckpt, _f, indent=2, ensure_ascii=False)
            log.info("  Cluster %d: checkpoint saved to %s", cid, _ckpt_path)

            # ── 保存 BO 诊断数据 ──
            if hasattr(learner, '_bo_diagnostics'):
                _diag_path = exp_dir / f"bo_diagnostics_c{cid}.json"
                with open(_diag_path, "w") as _f:
                    _json.dump(learner._bo_diagnostics, _f, indent=2, ensure_ascii=False)
                log.info("  Cluster %d: BO diagnostics saved to %s", cid, _diag_path)

    log.info("Total trials collected across all clusters: %d", len(all_trials))

    # ── 保存 all_trials pickle 以支持 batch_select replay ──
    import pickle as _pkl
    _trials_pkl_path = exp_dir / "all_trials.pkl"
    # 同时保存 batch_select 所需的 val 数据
    _batch_ctx = {
        "all_trials": all_trials,
        "label_names": label_names,
        "val_docs": select_docs if select_docs is not None else val_docs,
        "val_labels": select_y if select_y is not None else val_y,
        "base_predictions": (select_base_predictions if select_base_predictions is not None
                             else base_val_predictions),
        "base_f1": select_f1 if select_f1 is not None else base_val_f1,
        "effective_min_fires": effective_min_fires,
    }
    with open(_trials_pkl_path, "wb") as _pf:
        _pkl.dump(_batch_ctx, _pf)
    log.info("Saved batch_select context to %s (%d trials)", _trials_pkl_path, len(all_trials))

    # ── Common val data for batch_select / staged pipeline ──
    _select_val_docs = select_docs if select_docs is not None else val_docs
    _select_val_y = select_y if select_y is not None else val_y
    _select_base_preds = (select_base_predictions if select_base_predictions is not None
                          else base_val_predictions)
    _select_base_f1 = select_f1 if select_f1 is not None else base_val_f1
    # Model predictions for combined batch_select (always original model, never zeros)
    if model_select_predictions is not None:
        _model_select_preds = model_select_predictions
        _model_select_f1 = model_select_f1
    elif model_val_predictions is not None:
        _model_select_preds = model_val_predictions
        _model_select_f1 = model_val_f1
    else:
        _model_select_preds = _select_base_preds
        _model_select_f1 = _select_base_f1

    _is_blank = _is_blank_early
    _staged = _staged_early

    if _staged:
        # ═══════════════════════════════════════════════════════════════════
        # Staged Pipeline (blank mode only)
        # ═══════════════════════════════════════════════════════════════════
        log.info("=== Staged Pipeline: blank mode, 2-stage rule learning ===")
        _zero_base_select = np.zeros_like(_select_val_y, dtype=np.float32)
        _zero_base_bo = np.zeros((len(val_docs), len(label_names)), dtype=np.float32)
        _is_two_val_staged = (select_docs is not None)

        # ── K3: 一次性预计算所有 candidate 谓词的 fire masks ──
        # BO data: learner._fire_masks 和 _ml_proba_cache 已在 Phase B precompute 中缓存
        _first_learner = next(iter(_learners.values()))
        _bo_ml_proba = _first_learner._ml_proba_cache
        _ml_names = list(_bo_ml_proba.keys())

        # Select data: 收集所有 cluster 的 candidate predicates union
        _all_candidates = []
        _cand_id_set: set = set()
        for _cid, _lrn in _learners.items():
            for _p in _lrn.candidate_predicates:
                if id(_p) not in _cand_id_set:
                    _cand_id_set.add(id(_p))
                    _all_candidates.append(_p)
        _sel_non_ml = [p for p in _all_candidates if not isinstance(p, MLThresholdPredicate)]

        log.info("Precomputing select-data masks: %d non-ML preds on %d docs ...",
                 len(_sel_non_ml), len(_select_val_docs))
        _sel_text_masks = precompute_fire_masks(_sel_non_ml, _select_val_docs)
        _sel_ml_proba = precompute_ml_proba(_ml_names, _select_val_docs)
        _sel_pred_to_idx = {id(p): i for i, p in enumerate(_sel_non_ml)}

        # BO data text masks (from learner caches — merge all clusters)
        _bo_text_masks_list = []
        _bo_pred_to_idx: Dict[int, int] = {}
        _bo_seen: set = set()
        for _cid, _lrn in _learners.items():
            if _lrn._fire_masks is None:
                continue
            for _pi, _p in enumerate(_lrn.candidate_predicates):
                if isinstance(_p, MLThresholdPredicate):
                    continue
                if id(_p) not in _bo_seen:
                    _bo_seen.add(id(_p))
                    _bo_pred_to_idx[id(_p)] = len(_bo_text_masks_list)
                    _bo_text_masks_list.append(_lrn._fire_masks[_pi])
        if _bo_text_masks_list:
            _bo_text_masks = np.stack(_bo_text_masks_list)
        else:
            _bo_text_masks = np.zeros((0, len(val_docs)), dtype=bool)
        log.info("Precomputed masks: select=%d text + %d ML, bo=%d text + %d ML",
                 len(_sel_non_ml), len(_sel_ml_proba),
                 len(_bo_text_masks_list), len(_bo_ml_proba))

        # --- Stage 1: Inject ML baseline rules ---
        _wl_rescue = not getattr(hp, 'no_weak_label_rescue', False)
        _wl_f1_thr = getattr(hp, 'weak_label_f1_threshold', 0.65) if _wl_rescue else 0.0
        _wl_max_k = getattr(hp, 'weak_label_max_rules', 3) if _wl_rescue else 1
        stage1_rules, stage1_log = ChaseRuleLearner.inject_ml_baseline_rules(
            ml_model_names=_ml_names,
            ml_proba_cache=_sel_ml_proba,
            label_names=label_names,
            val_labels=_select_val_y,
            allow_multi_model=getattr(hp, 'inject_multi_model', True),
            weak_label_f1_threshold=_wl_f1_thr,
            weak_label_max_rules=_wl_max_k,
            fine_search=True,
        )
        import json as _json_stage
        with open(str(exp_dir / "stage1_injection_log.json"), "w") as _f:
            _json_stage.dump(stage1_log, _f, indent=2)
        log.info("Stage 1 injection log saved: %d entries", len(stage1_log))

        stage1_preds = _vectorized_staged_predict(
            stage1_rules, label_names, len(_select_val_docs),
            _sel_ml_proba, _sel_text_masks, _sel_pred_to_idx)
        stage1_f1 = float(f1_score(_select_val_y, stage1_preds,
                                    average="macro", zero_division=0))
        if _is_two_val_staged:
            stage1_preds_bo = _vectorized_staged_predict(
                stage1_rules, label_names, len(val_docs),
                _bo_ml_proba, _bo_text_masks, _bo_pred_to_idx)
            stage1_f1_bo = float(f1_score(val_y, stage1_preds_bo,
                                          average="macro", zero_division=0))
        else:
            stage1_preds_bo = stage1_preds
            stage1_f1_bo = stage1_f1
        log.info("Stage 1 (ML baseline): %d rules, macro-F1=%.4f (select), %.4f (bo)",
                 len(stage1_rules), stage1_f1, stage1_f1_bo)

        # --- Weak-label identification (drives BO weighted sampling + relaxed penalties) ---
        _weak_labels: set = set()
        _label_weights: dict = {}
        if _wl_rescue:
            _stage1_plf = _fast_per_label_f1(_select_val_y, stage1_preds)
            for _li, _ln in enumerate(label_names):
                _f1_i = float(_stage1_plf[_li])
                if _f1_i < 0.70:
                    _label_weights[_ln] = (0.70 / max(_f1_i, 0.1)) ** 2
                else:
                    _label_weights[_ln] = 1.0
                if _f1_i < _wl_f1_thr:
                    _weak_labels.add(_ln)
            log.info("Weak labels (%d/%d, threshold=%.2f): %s",
                     len(_weak_labels), len(label_names), _wl_f1_thr,
                     {ln: round(float(_stage1_plf[label_names.index(ln)]), 3)
                      for ln in sorted(_weak_labels)})

        # --- Error-driven FP anchors for REMOVE ---
        _fp_preds = _extract_error_driven_predicates(
            val_docs, val_y, stage1_preds_bo, label_names,
            top_k_per_label=8, mode="fp")
        if _fp_preds:
            log.info("Error-driven FP anchors: %d new predicates for REMOVE", len(_fp_preds))
            for learner in _learners.values():
                learner.candidate_predicates = list(learner.candidate_predicates) + _fp_preds
                learner._precomputed = False

        # --- Stage 2: REMOVE rules ---
        _s2_min_n_changes = max(getattr(hp, 'min_n_changes', 2), 3)
        stage2_trials = []
        for cid, learner in _learners.items():
            learner.update_base_predictions(stage1_preds_bo, stage1_f1_bo)
            learner.consequence_op_mode = "remove"
            learner.track = "remove"
            learner.label_weights = _label_weights if _wl_rescue else None
            learner.weak_labels = _weak_labels if _wl_rescue else None
            learner.storage_path = str(exp_dir / f"optuna_chase_s2_c{cid}_a{attempt}")
            if not no_error_driven_filter:
                learner.error_driven_filter = True
                learner.greedy_weights = ERROR_AWARE_WEIGHTS
            stage2_trials.extend(learner.run_bo())

        if not no_tree_warmup:
            _first_learner = next(iter(_learners.values()))
            _s2_tree_rules = _tree_seeded_rules(
                fire_masks=_first_learner._fire_masks,
                candidate_predicates=_first_learner.candidate_predicates,
                ml_proba_cache=_first_learner._ml_proba_cache,
                existing_predictions=stage1_preds_bo,
                val_labels=_first_learner.val_labels,
                label_list=label_names,
                consequence_op="remove",
            )
            for rule in _s2_tree_rules:
                stage2_trials.append((None, rule))
            log.info("Stage 2 tree warm-up: %d rules injected", len(_s2_tree_rules))

        stage2_rdl = ChaseRuleLearner.batch_select(
            all_trials=stage2_trials,
            label_list=label_names,
            val_docs=_select_val_docs,
            val_labels=_select_val_y,
            base_predictions=stage1_preds,
            base_f1=stage1_f1,
            sort_by_gain=hp.rule_batch_sort,
            verbose=log.isEnabledFor(logging.DEBUG),
            min_rule_fires=effective_min_fires,
            min_n_changes=_s2_min_n_changes,
            min_corr_prec=max(hp.min_corr_prec, 0.75),
            inject_ml_baseline=False,
            accept_docs=val_docs if _is_two_val_staged else None,
            accept_labels=val_y if _is_two_val_staged else None,
            accept_base_predictions=stage1_preds_bo if _is_two_val_staged else None,
            precomputed_pred_masks=_sel_text_masks,
            precomputed_pred_to_idx=_sel_pred_to_idx,
            precomputed_ml_proba=_sel_ml_proba,
        )
        stage12_rules = list(stage1_rules) + list(stage2_rdl.rules)
        stage12_preds = _vectorized_staged_predict(
            stage12_rules, label_names, len(_select_val_docs),
            _sel_ml_proba, _sel_text_masks, _sel_pred_to_idx)
        stage12_f1 = float(f1_score(_select_val_y, stage12_preds,
                                     average="macro", zero_division=0))
        if _is_two_val_staged:
            stage12_preds_bo = _vectorized_staged_predict(
                stage12_rules, label_names, len(val_docs),
                _bo_ml_proba, _bo_text_masks, _bo_pred_to_idx)
            stage12_f1_bo = float(f1_score(val_y, stage12_preds_bo,
                                           average="macro", zero_division=0))
        else:
            stage12_preds_bo = stage12_preds
            stage12_f1_bo = stage12_f1
        log.info("Stage 2 (REMOVE): %d rules, cumulative F1=%.4f",
                 len(stage2_rdl.rules), stage12_f1)

        all_seed_rules = stage12_rules
        _seed_preds = stage12_preds
        _seed_f1 = stage12_f1

        # Save staged pipeline logs
        _staged_log = {
            "stage1": stage1_log,
            "stage2_selected": _extract_trial_stats(stage2_rdl.rules),
            "stage2_all_count": len(stage2_trials),
            "cumulative_f1": {
                "stage1": stage1_f1, "stage12": stage12_f1,
                "seed": _seed_f1,
            },
        }
        _save_stage_logs(str(exp_dir), _staged_log)

        track1_rdl_set = RDLSet(all_seed_rules, label_names)
        log.info("Staged Track 1 complete: %d rules, F1=%.4f",
                 len(all_seed_rules), _seed_f1)

    else:
        # ═══════════════════════════════════════════════════════════════════
        # Original Track 1 flow (non-staged)
        # ═══════════════════════════════════════════════════════════════════

        _add_min_prec = max(hp.min_corr_prec, 0.72)
        track1_rdl_set = ChaseRuleLearner.batch_select(
            all_trials=all_trials,
            label_list=label_names,
            val_docs=_select_val_docs,
            val_labels=_select_val_y,
            base_predictions=_select_base_preds,
            base_f1=_select_base_f1,
            sort_by_gain=hp.rule_batch_sort,
            verbose=log.isEnabledFor(logging.DEBUG),
            min_rule_fires=effective_min_fires,
            min_n_changes=hp.min_n_changes,
            min_corr_prec=_add_min_prec,
            accept_docs=accept_docs if select_docs is None else val_docs,
            accept_labels=accept_y if select_docs is None else val_y,
            accept_base_predictions=accept_predictions if select_docs is None else base_val_predictions,
            inject_ml_baseline=getattr(hp, 'inject_ml_baseline', True),
        )
        log.info("Track 1 (ADD): %d rules selected (min_corr_prec=%.2f)",
                 len(track1_rdl_set.rules), _add_min_prec)

        # ── REMOVE stage (non-staged path) ──
        # Compute Track 1 ADD predictions, then search for REMOVE rules
        _t1_add_preds_bo = track1_rdl_set.predict(val_docs)
        _t1_add_f1_bo = float(f1_score(
            val_y, (_t1_add_preds_bo > 0).astype(int),
            average="macro", zero_division=0))
        log.info("Track 1 ADD predictions on val_bo: F1=%.4f", _t1_add_f1_bo)

        _t1_add_preds_sel = track1_rdl_set.predict(_select_val_docs)
        _t1_add_f1_sel = float(f1_score(
            _select_val_y, (_t1_add_preds_sel > 0).astype(int),
            average="macro", zero_division=0))

        # Run REMOVE BO
        log.info("=== Non-staged REMOVE: searching for FP-correction rules ===")
        _remove_trials = []
        for cid, learner in _learners.items():
            learner.update_base_predictions(_t1_add_preds_bo, _t1_add_f1_bo)
            learner.consequence_op_mode = "remove"
            learner.track = "remove"
            learner.storage_path = str(exp_dir / f"optuna_chase_remove_c{cid}")
            _remove_trials.extend(learner.run_bo())
        log.info("REMOVE BO: %d candidate trials", len(_remove_trials))

        if _remove_trials:
            _remove_min_prec = max(hp.min_corr_prec, 0.70)
            _remove_rdl = ChaseRuleLearner.batch_select(
                all_trials=_remove_trials,
                label_list=label_names,
                val_docs=_select_val_docs,
                val_labels=_select_val_y,
                base_predictions=_t1_add_preds_sel,
                base_f1=_t1_add_f1_sel,
                sort_by_gain=hp.rule_batch_sort,
                verbose=log.isEnabledFor(logging.DEBUG),
                min_rule_fires=effective_min_fires,
                min_n_changes=max(hp.min_n_changes, 3),
                min_corr_prec=_remove_min_prec,
                accept_docs=accept_docs if select_docs is None else val_docs,
                accept_labels=accept_y if select_docs is None else val_y,
                accept_base_predictions=(accept_predictions if select_docs is None
                                         else _t1_add_preds_bo),
                inject_ml_baseline=False,
            )
            if _remove_rdl.rules:
                track1_rdl_set = RDLSet(
                    list(track1_rdl_set.rules) + list(_remove_rdl.rules),
                    label_names,
                )
                log.info("REMOVE stage: %d rules added (min_corr_prec=%.2f), "
                         "total Track 1 rules: %d",
                         len(_remove_rdl.rules), _remove_min_prec,
                         len(track1_rdl_set.rules))
            else:
                log.info("REMOVE stage: no rules passed filters")
        else:
            log.info("REMOVE stage: no candidate trials generated")

    # ══════════════════════════════════════════════════════════════════════════
    # Track 2: Propagation rules (SimPredicate + LabelPredicate)
    # ══════════════════════════════════════════════════════════════════════════
    _sim_threshold_bins_raw = getattr(hp, "sim_threshold_bins", None)
    _track2_label_source = getattr(hp, "track2_label_source", "gt")
    # blank mode: default to track1 unless user explicitly set --track2_label_source
    if getattr(hp, 'track1_baseline', 'model') == 'blank' and _track2_label_source == "gt":
        _track2_label_source = "track1"

    log.info("=== Track 2: Building similarity graphs ===")

    # ── Copy cached embeddings from resume_dir if available ──
    if resume_dir:
        import shutil as _shutil
        for _emb_name in ("embeddings_bo.npy", "embeddings_select.npy"):
            _src = Path(resume_dir) / _emb_name
            _dst = exp_dir / _emb_name
            if _src.exists() and not _dst.exists():
                _shutil.copy2(str(_src), str(_dst))
                log.info("[RESUME] Copied %s from resume dir", _emb_name)

    # ── Embeddings for val_bo (BO) ──
    _bo_texts = [d.cnt for d in val_docs]
    _bo_emb_cache = str(exp_dir / "embeddings_bo.npy")
    bo_embeddings = compute_embeddings(_bo_texts, cache_path=_bo_emb_cache)

    # ── Auto-detect thresholds if not explicitly set ──
    if _sim_threshold_bins_raw and _sim_threshold_bins_raw != [0.80, 0.85, 0.88, 0.92, 0.95]:
        _sim_bins = _sim_threshold_bins_raw
        log.info("Using user-specified sim_threshold_bins: %s", _sim_bins)
    else:
        _sim_bins = auto_threshold_bins(bo_embeddings)
        log.info("Auto-detected sim_threshold_bins: %s", _sim_bins)

    # ── Build sim graph on val_bo (for BO learners) ──
    _bo_sim_dir = exp_dir / "sim_graphs_bo"
    bo_sim_graphs = build_sim_graph(bo_embeddings, _sim_bins)
    save_sim_graphs(bo_sim_graphs, str(_bo_sim_dir))

    # ── Build sim graph on val_select (for combined batch_select) ──
    # In two_val mode, _select_val_docs != val_docs, so need separate graphs
    _is_two_val = (select_docs is not None)
    if _is_two_val:
        _sel_texts = [d.cnt for d in _select_val_docs]
        _sel_emb_cache = str(exp_dir / "embeddings_select.npy")
        sel_embeddings = compute_embeddings(_sel_texts, cache_path=_sel_emb_cache)
        _sel_sim_dir = exp_dir / "sim_graphs_select"
        select_sim_graphs = build_sim_graph(sel_embeddings, _sim_bins)
        save_sim_graphs(select_sim_graphs, str(_sel_sim_dir))
    else:
        select_sim_graphs = bo_sim_graphs

    if False:  # Group propagation always runs; sim graphs are optional
        log.warning("No sim graphs survived threshold filtering! Skipping Track 2.")
        rdl_set = track1_rdl_set
    else:
        # ── Track 2 setup: cache predictions ──
        _t1_preds_bo = None  # cache for reuse below
        if _is_blank and _track2_label_source == "track1":
            if _staged:
                _t1_preds_bo = _vectorized_staged_predict(
                    all_seed_rules, label_names, len(val_docs),
                    _bo_ml_proba, _bo_text_masks, _bo_pred_to_idx)
            else:
                _t1_preds_bo = track1_rdl_set.predict(val_docs)

        # BO discovery 阶段：根据 track2_label_source 决定 label_state
        if _track2_label_source == "gt":
            bo_label_state = val_y > 0
        elif _track2_label_source == "model":
            bo_label_state = base_val_predictions > 0
        else:  # "track1" (default)
            if _is_blank:
                bo_label_state = _t1_preds_bo > 0
            else:
                bo_label_state = base_val_predictions > 0

        neighbor_label_masks = precompute_neighbor_label_masks(
            bo_sim_graphs, bo_label_state, label_names,
        )
        neighbor_label_counts = precompute_neighbor_label_counts(
            bo_sim_graphs, bo_label_state, label_names,
        )
        log.info("Track 2: %d neighbor-label masks computed", len(neighbor_label_masks))

        # ── Label state for batch_select (on val_select) ──
        if _is_two_val:
            if _track2_label_source == "gt":
                select_label_state = _select_val_y > 0
            elif _track2_label_source == "model":
                select_label_state = _select_base_preds > 0
            else:
                if _is_blank:
                    if _staged:
                        select_label_state = _vectorized_staged_predict(
                            all_seed_rules, label_names, len(_select_val_docs),
                            _sel_ml_proba, _sel_text_masks, _sel_pred_to_idx) > 0
                    else:
                        select_label_state = track1_rdl_set.predict(_select_val_docs) > 0
                else:
                    select_label_state = _select_base_preds.copy()
                    select_label_state = select_label_state > 0
        else:
            select_label_state = bo_label_state

        # Track 2 BO: reuse the same cluster structure, create new learners
        track2_trials = []
        _learners_t2: dict = {}

        # Track 2 baseline: blank mode uses Track 1 rule predictions,
        # model mode uses original model predictions
        if _is_blank:
            if _staged:
                _t2_base = _vectorized_staged_predict(
                    all_seed_rules, label_names, len(val_docs),
                    _bo_ml_proba, _bo_text_masks, _bo_pred_to_idx)
                _t2_base_f1 = float(f1_score(val_y, _t2_base,
                                              average="macro", zero_division=0))
                log.info("Track 2 baseline = Staged rules (%d rules, F1=%.4f)",
                         len(all_seed_rules), _t2_base_f1)
            else:
                if _t1_preds_bo is None:
                    _t1_preds_bo = track1_rdl_set.predict(val_docs)
                _t2_base = _t1_preds_bo
                _t2_base_f1 = float(f1_score(val_y, _t1_preds_bo,
                                              average="macro", zero_division=0))
                log.info("Track 2 baseline = Track 1 rules (%d rules, F1=%.4f)",
                         len(track1_rdl_set.rules), _t2_base_f1)
        else:
            _t2_base = (model_val_predictions if model_val_predictions is not None
                         else base_val_predictions)
            _t2_base_f1 = (model_val_f1 if model_val_f1 is not None
                            else float(f1_score(val_y, _t2_base,
                                                average="macro", zero_division=0)))

        # ══════════════════════════════════════════════════════════════════════
        # Track 2 Group Propagation: x.attr == y.attr ∧ label(A∈y.lbl) → add A
        # ══════════════════════════════════════════════════════════════════════
        from rule_discovery.virtual_attributes import (
            compute_all_virtual_attributes, filter_degenerate_groups,
        )
        from rule_discovery.group_propagation import discover_group_rules

        log.info("=== Track 2 Group Propagation: computing virtual attributes ===")

        # Compute train embeddings for KMeans fitting
        _train_texts = [d.cnt for d in train_docs]
        _train_emb_cache = str(exp_dir / "embeddings_train.npy")
        _train_emb = compute_embeddings(_train_texts, cache_path=_train_emb_cache)

        # Virtual attributes on val_bo
        _group_ml_proba = locals().get('_bo_ml_proba') or {}
        bo_virtual_attrs, _kmeans_models = compute_all_virtual_attributes(
            train_embeddings=_train_emb,
            target_embeddings=bo_embeddings,
            ml_proba_cache=_group_ml_proba,
            label_names=label_names,
        )
        bo_virtual_attrs = filter_degenerate_groups(bo_virtual_attrs)
        log.info("Track 2 Group: %d virtual attributes on val_bo", len(bo_virtual_attrs))

        # Discover group rules (exhaustive enumeration)
        _group_min_prec = group_min_corr_prec
        _group_trials = discover_group_rules(
            virtual_attrs=bo_virtual_attrs,
            label_state=bo_label_state.astype(np.float32),
            label_names=label_names,
            val_labels=val_y,
            existing_predictions=_t2_base,
            val_docs=val_docs,
            min_fires=max(3, effective_min_fires // 2),
            min_corr_prec=_group_min_prec,
        )
        log.info("Track 2 Group: %d rules discovered on val_bo", len(_group_trials))

        # Virtual attributes on val_select (for batch_select)
        if _is_two_val:
            sel_virtual_attrs, _ = compute_all_virtual_attributes(
                train_embeddings=_train_emb,
                target_embeddings=sel_embeddings,
                ml_proba_cache=_sel_ml_proba if _sel_ml_proba else {},
                label_names=label_names,
                kmeans_models=_kmeans_models,
            )
            sel_virtual_attrs = filter_degenerate_groups(sel_virtual_attrs)
        else:
            sel_virtual_attrs = bo_virtual_attrs

        # --- Error-driven FN anchors for Track 2 ---
        _fn_preds = _extract_error_driven_predicates(
            val_docs, val_y, _t2_base, label_names,
            top_k_per_label=8, mode="fn")
        if _fn_preds:
            log.info("Error-driven FN anchors: %d new predicates for Track 2", len(_fn_preds))

        # --- Track 2 label_weights: focus BO on high-FN labels ---
        _t2_plf = _fast_per_label_f1(val_y, _t2_base)
        _t2_label_weights: dict = {}
        _t2_weak_labels: set = set()
        for _li, _ln in enumerate(label_names):
            _f1_i = float(_t2_plf[_li])
            if _f1_i < 0.80:
                _t2_label_weights[_ln] = (0.80 / max(_f1_i, 0.1)) ** 2
            else:
                _t2_label_weights[_ln] = 1.0
            if _f1_i < 0.50:
                _t2_weak_labels.add(_ln)
        log.info("Track 2 label_weights: %d labels with weight>1, %d weak labels",
                 sum(1 for v in _t2_label_weights.values() if v > 1.0),
                 len(_t2_weak_labels))

        for cid, learner_t1 in _learners.items():
            t2_learner = ChaseRuleLearner(
                candidate_predicates=learner_t1.candidate_predicates,
                candidate_ml_models=learner_t1.candidate_ml_models,
                label_list=label_names,
                val_docs=val_docs,
                val_labels=val_y,
                max_trials=learner_t1.max_trials,
                min_coverage=learner_t1.min_coverage,
                seed=142 + cid,
                storage_path=str(exp_dir / f"optuna_chase_t2_c{cid}_a{attempt}"),
                verbose=log.isEnabledFor(logging.DEBUG),
                base_predictions=_t2_base,
                base_f1=_t2_base_f1,
                top_per_type=learner_t1.top_per_type,
                rule_min_precision=learner_t1.rule_min_precision,
                metric_mode=learner_t1.metric_mode,
                cluster_label_indices=learner_t1.cluster_label_indices,
                min_rule_fires=effective_min_fires,
                min_n_changes=hp.min_n_changes,
                min_corr_prec=hp.min_corr_prec,
                track="propagation",
                sim_graphs=bo_sim_graphs,
                neighbor_label_masks=neighbor_label_masks,
                sim_cascade_rounds=getattr(hp, 'sim_cascade_rounds', 3),
                sim_min_precision=getattr(hp, 'sim_min_precision', 0.60),
                sim_self_loop_min_prec=getattr(hp, 'sim_self_loop_min_prec', -1.0),
                label_prec_objective=getattr(hp, 'label_prec_objective', False),
            )
            t2_learner.neighbor_label_counts = neighbor_label_counts
            t2_learner.label_weights = _t2_label_weights
            t2_learner.weak_labels = _t2_weak_labels
            if _fn_preds:
                t2_learner.candidate_predicates = list(t2_learner.candidate_predicates) + _fn_preds
                t2_learner._precomputed = False
            _learners_t2[cid] = t2_learner

        # Parallel Track 2 BO
        log.info("Running Track 2 BO for %d clusters", len(_learners_t2))
        if len(_learners_t2) <= 1:
            _bo_results_t2 = []
            for cid, l2 in _learners_t2.items():
                _bo_results_t2.append((cid, l2, l2.run_bo()))
        else:
            _bo_results_t2 = []
            with ThreadPoolExecutor(max_workers=_n_workers) as executor:
                futures = {
                    executor.submit(_run_bo_for_cluster, (cid, l2)): cid
                    for cid, l2 in _learners_t2.items()
                }
                for future in as_completed(futures):
                    _bo_results_t2.append(future.result())

        for cid, _, raw_trials in sorted(_bo_results_t2, key=lambda x: x[0]):
            trials = [(t, rdl) for t, rdl in raw_trials if t.value >= hp.min_f1_gain]
            track2_trials.extend(trials)
            log.info("  Track 2 Cluster %d: %d/%d trials surviving filter",
                     cid, len(trials), len(raw_trials))

        log.info("Track 2 total trials (sim BO): %d", len(track2_trials))

        # Merge group propagation rules into track2_trials
        track2_trials.extend(_group_trials)
        log.info("Track 2 total trials (sim BO + group): %d", len(track2_trials))

        if _staged:
            # Staged: Track 2 batch_select uses seed predictions as base,
            # then concatenate seed rules + Track 2 rules
            _t2_select_base = _vectorized_staged_predict(
                all_seed_rules, label_names, len(_select_val_docs),
                _sel_ml_proba, _sel_text_masks, _sel_pred_to_idx)
            _t2_select_f1 = float(f1_score(_select_val_y, _t2_select_base,
                                            average="macro", zero_division=0))
            _t2_min_fires = max(3, effective_min_fires // 2)
            track2_rdl = ChaseRuleLearner.batch_select(
                all_trials=track2_trials,
                label_list=label_names,
                val_docs=_select_val_docs,
                val_labels=_select_val_y,
                base_predictions=_t2_select_base,
                base_f1=_t2_select_f1,
                sort_by_gain=hp.rule_batch_sort,
                verbose=log.isEnabledFor(logging.DEBUG),
                min_rule_fires=_t2_min_fires,
                min_n_changes=hp.min_n_changes,
                min_corr_prec=hp.min_corr_prec,
                sim_graphs=select_sim_graphs,
                label_state=select_label_state.astype(np.float32),
                virtual_attrs=sel_virtual_attrs,
                inject_ml_baseline=False,
                precomputed_pred_masks=_sel_text_masks,
                precomputed_pred_to_idx=_sel_pred_to_idx,
                precomputed_ml_proba=_sel_ml_proba,
            )
            final_rules = list(all_seed_rules) + list(track2_rdl.rules)
            rdl_set = RDLSet(final_rules, label_names)
            log.info("Staged final: %d seed + %d Track 2 = %d total rules",
                     len(all_seed_rules), len(track2_rdl.rules), len(final_rules))
        else:
            # Original: combined batch_select (Track 1 + Track 2)
            combined_trials = all_trials + track2_trials
            if _is_blank:
                _combined_base = np.zeros_like(_select_val_y, dtype=np.float32)
                _combined_f1 = 0.0
            else:
                _combined_base = _model_select_preds
                _combined_f1 = _model_select_f1
            rdl_set = ChaseRuleLearner.batch_select(
                all_trials=combined_trials,
                label_list=label_names,
                val_docs=_select_val_docs,
                val_labels=_select_val_y,
                base_predictions=_combined_base,
                base_f1=_combined_f1,
                sort_by_gain=hp.rule_batch_sort,
                verbose=log.isEnabledFor(logging.DEBUG),
                min_rule_fires=effective_min_fires,
                min_n_changes=hp.min_n_changes,
                min_corr_prec=hp.min_corr_prec,
                accept_docs=accept_docs if select_docs is None else val_docs,
                accept_labels=accept_y if select_docs is None else val_y,
                accept_base_predictions=accept_predictions if select_docs is None else base_val_predictions,
                sim_graphs=select_sim_graphs,
                label_state=select_label_state.astype(np.float32),
                virtual_attrs=sel_virtual_attrs,
                inject_ml_baseline=getattr(hp, 'inject_ml_baseline', True),
            )

    # ── LabelPredicate co-occurrence rules ──
    # Generate rules from label co-occurrence patterns and add validated ones
    _cooccur_base = rdl_set.predict(_select_val_docs) if rdl_set.rules else _select_base_preds
    _cooccur_rules = _generate_label_cooccurrence_rules(
        val_labels=_select_val_y,
        label_names=label_names,
        base_predictions=_cooccur_base,
        min_cooccur=8,
        min_precision=0.55,
    )
    if _cooccur_rules:
        # Cross-validate on accept set
        _accept_base = rdl_set.predict(accept_docs if select_docs is None else val_docs)
        _accept_y = accept_y if select_docs is None else val_y
        _validated_cooccur = []
        for rule in _cooccur_rules:
            cidx = label2idx.get(rule.consequence)
            if cidx is None:
                continue
            fires = 0
            tp = 0
            for doc_idx, doc in enumerate(accept_docs if select_docs is None else val_docs):
                proxy = Document(
                    cnt=doc.cnt,
                    lbl=set(label_names[j] for j in range(len(label_names))
                            if _accept_base[doc_idx, j] > 0),
                    mtd=doc.mtd, ttl=doc.ttl,
                )
                if rule.fires(proxy) and _accept_base[doc_idx, cidx] == 0:
                    fires += 1
                    if _accept_y[doc_idx, cidx] == 1:
                        tp += 1
            if fires >= 5 and tp / fires >= 0.50:
                _validated_cooccur.append(rule)
        if _validated_cooccur:
            rdl_set = RDLSet(
                list(rdl_set.rules) + _validated_cooccur,
                label_names,
            )
            log.info("LabelPredicate co-occurrence: %d rules added (from %d candidates)",
                     len(_validated_cooccur), len(_cooccur_rules))
        else:
            log.info("LabelPredicate co-occurrence: 0 rules passed cross-validation "
                     "(from %d candidates)", len(_cooccur_rules))
    else:
        log.info("LabelPredicate co-occurrence: no candidate rules generated")

    out_path = str(exp_dir / "rules.json")
    rdl_set.save(out_path)

    log.info(
        "Chase Batch Rule Discovery done in %.1fs — %d rules selected "
        "(Track1=%d + Track2=%d), saved to %s",
        time.time() - t0, len(rdl_set.rules),
        len(track1_rdl_set.rules), len(rdl_set.rules) - len(track1_rdl_set.rules),
        out_path,
    )
    if rdl_set.rules:
        log.info("Top-5 rules by F1-gain:")
        top5 = sorted(rdl_set.rules, key=lambda r: r.score, reverse=True)[:5]
        for i, r in enumerate(top5, 1):
            log.info("  #%d  [%s]  gain=%.4f  cov=%.3f  body=%s",
                     i, r.consequence, r.score, r.coverage, repr(r))
    return rdl_set


# ══════════════════════════════════════════════════════════════════════════════
# CLI — 完全复用原版参数
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LORIS Chase pipeline — multi-dataset (ChaseRuleLearner)",
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
                   help="Pattern extraction mode")
    p.add_argument("--sim_threshold", type=float, default=None,
                   help="Similarity threshold for sim mode (default: 0.45)")
    p.add_argument("--lora_model", default=None,
                   help="HF model name for LoRASLMClassifier (requires 20 GB VRAM)")
    p.add_argument("--debug", action="store_true",
                   help="Enable verbose logging")
    p.add_argument("--max_trials", type=int, default=200)
    p.add_argument("--top_n_rules", type=int, default=10)
    p.add_argument("--min_f1_gain", type=float, default=0.0,
                   help="Minimum F1 gain for a trial to be kept (0 = any positive gain)")
    p.add_argument("--min_corr_prec", type=float, default=0.65,
                   help="Minimum correction precision in evaluate and batch_select")
    p.add_argument("--group_min_corr_prec", type=float, default=0.50,
                   help="Minimum corr_prec for Track 2 group propagation rules (lower than general min_corr_prec because rescue predicates refine precision)")
    p.add_argument("--top_per_type", type=int, default=30)
    p.add_argument("--rule_strategy", default="batch",
                   choices=["greedy", "batch"])
    p.add_argument("--rule_batch_sort", action="store_true", default=True)
    p.add_argument("--no_rule_batch_sort", dest="rule_batch_sort",
                   action="store_false")
    p.add_argument("--rule_min_precision", type=float, default=0.0)
    p.add_argument("--anchor_min_df", type=int, default=None,
                   help="Min document frequency for spaCy anchors (default: 3)")
    p.add_argument("--tfidf_top_k", type=int, default=None,
                   help="Top TF-IDF anchors per cluster (default: 200)")
    p.add_argument("--min_coverage", type=float, default=None,
                   help="Min predicate coverage fraction (default: 0.005)")
    p.add_argument("--max_entropy_threshold", type=float, default=None,
                   help="Max entropy for predicate screening (default: 2.0)")
    p.add_argument("--big", action="store_true", default=False,
                   help="Large-corpus preset: anchor_min_df=10, tfidf_top_k=100, "
                        "min_coverage=0.01, max_entropy_threshold=1.5")
    p.add_argument("--predicate_top_k", type=int, default=0,
                   help="Max candidate predicates per cluster (0=no limit)")
    p.add_argument("--glove_path", type=str, default=None,
                   help="Path to GloVe text file for synonym expansion "
                        "(default: /root/autodl-tmp/glove.6B.100d.txt if exists)")
    p.add_argument("--per_type_top_k", type=int, default=0,
                   help="Top-k predicates per type (0=no limit). "
                        "Set >0 to cap per type.")
    p.add_argument("--batch_metric_mode", default="global_macro",
                   choices=["global_macro", "cluster_local"],
                   help="Metric mode for batch BO (hybrid default: global_macro)")
    p.add_argument("--rule_eval_data", default="val",
                   choices=["val", "train", "split"],
                   help="val: all on val; train: all on train; "
                        "split: greedy_instantiate on train, F1 eval on val")
    p.add_argument("--min_rule_fires", type=int, default=10,
                   help="Minimum number of val documents a rule must fire on "
                        "(higher = more generalizable rules)")
    p.add_argument("--min_n_changes", type=int, default=3,
                   help="Minimum number of val documents whose prediction "
                        "actually changes (improved+worsened >= N)")
    p.add_argument("--cluster_model_selection", default="global",
                   choices=["global", "f1_rank", "router"],
                   help="Per-cluster top-K model selection. "
                        "'global' = same K models for all clusters; "
                        "'f1_rank' = each cluster picks top-K by macro-F1 on its val subset; "
                        "'router' = neural router per cluster")
    p.add_argument("--baseline_mode", default="best",
                   choices=["best", "mean_top_k"])
    p.add_argument("--track1_baseline", default="model",
                   choices=["model", "blank"],
                   help="Track 1 baseline: 'model' uses best model predictions, "
                        "'blank' starts from all-zero predictions so rules build "
                        "labels from scratch (including high-confidence ML rules)")
    p.add_argument("--baseline_model", default=None,
                   help="Force a specific model as baseline (e.g. 'bilstm') "
                        "instead of auto-selecting the best model")
    p.add_argument("--no_encoder", action="store_true",
                   help="Exclude pretrained encoder model from pool")
    p.add_argument("--two_val", action="store_true",
                   help="Split val into val_bo (BO discovery) + val_select "
                        "(batch_select filtering) to prevent overfitting")
    # ── Chase-specific arguments ──
    p.add_argument("--track2_label_source", type=str, default="gt",
                   choices=["track1", "gt", "model"],
                   help="Label state for Track 2 neighbor masks: "
                        "track1=T1-updated, gt=ground truth, model=base predictions")
    p.add_argument("--sim_threshold_bins", type=str,
                   default="auto",
                   help="Comma-separated similarity thresholds, or 'auto' for data-adaptive")
    p.add_argument("--resume_exp_dir", type=str, default=None,
                   help="Resume from a previous experiment dir (reuse model pool + per-cluster stores)")
    p.add_argument("--resume_mode", type=str, default="all",
                   choices=["all", "prep"],
                   help="What to restore from resume_exp_dir: "
                        "'all' = model+patterns+trials (skip BO), "
                        "'prep' = model+patterns only (re-run BO with new params)")
    p.add_argument("--skip_chase_test", action="store_true",
                   help="Skip full chase inference at test time; use fast two-pass evaluation instead")
    p.add_argument("--sim_cascade_rounds", type=int, default=3,
                   help="Number of cascade rounds to simulate during BO evaluation of sim rules (1=current behavior)")
    p.add_argument("--sim_decay", type=float, default=1.0,
                   help="Sim confidence decay factor per hop (1.0=no decay)")
    p.add_argument("--sim_conf_threshold", type=float, default=0.0,
                   help="Minimum confidence for sim labels to trigger further propagation")
    p.add_argument("--sim_min_precision", type=float, default=0.75,
                   help="Minimum raw precision for sim rules in BO (lower → rejected)")
    p.add_argument("--sim_self_loop_min_prec", type=float, default=-1.0,
                   help="Min precision for self-loop sim rules (A→A). -1 = max(sim_min_precision, 0.85)")
    p.add_argument("--no_inject_ml_baseline", action="store_true",
                   help="Disable injection of per-label optimal baseline ML rules in batch_select")
    p.add_argument("--label_prec_objective", action="store_true",
                   help="Use precision-oriented objective for label rules too (not just sim)")
    p.add_argument("--subsample_ratio", type=float, default=1.0,
                   help="Training data subsample ratio per model (default 1.0, no subsampling)")
    p.add_argument("--no_staged_pipeline", action="store_true",
                   help="Disable staged pipeline, use original Track 1+2 flow")
    p.add_argument("--enable_stage4", action="store_true",
                   help="Enable optional second REMOVE pass (Stage 4)")
    p.add_argument("--no_inject_multi_model", action="store_true",
                   help="Disable multi-model combination in Stage 1 injection")
    p.add_argument("--no_title_predicates", action="store_true", default=True,
                   help="Disable title-field (x.ttl) predicates; fallback to content-only")
    p.add_argument("--use_title_predicates", action="store_true",
                   help="Enable title-field predicates (extract anchors from x.ttl too)")
    p.add_argument("--use_rill", action="store_true",
                   help="Apply RILL recursive chase at test time (paper §6)")
    p.add_argument("--rill_mode", default="propagate",
                   choices=["propagate", "active"],
                   help="RILL mode: 'propagate'=run chase to fixpoint without queries; "
                        "'active'=enable oracle queries (max_iterations budget)")
    p.add_argument("--rill_max_iterations", type=int, default=3,
                   help="Max RILL iterations: for propagate mode = chase rounds; "
                        "for active mode = oracle query budget")
    p.add_argument("--rerun_patterns", action="store_true",
                   help="Force pattern abstraction even if cached cluster_X_store.json exists")
    p.add_argument("--no_tree_warmup", action="store_true", default=True,
                   help="Disable decision-tree warm-up rule seeding in Stage 2/3")
    p.add_argument("--no_error_driven_filter", action="store_true", default=False,
                   help="Disable error-driven candidate predicate filtering in Stage 2/3")
    p.add_argument("--use_tree_warmup", action="store_true",
                   help="Enable decision-tree warm-up rule seeding")
    p.add_argument("--use_error_driven_filter", action="store_true", default=True,
                   help="Enable error-driven candidate predicate filtering")
    p.add_argument("--weak_label_f1_threshold", type=float, default=0.65,
                   help="Stage 1 F1 below this => weak label (multi-rule inject + weighted BO)")
    p.add_argument("--weak_label_max_rules", type=int, default=3,
                   help="Max Stage 1 rules per weak label (OR ensemble)")
    p.add_argument("--no_weak_label_rescue", action="store_true",
                   help="Disable weak-label adaptive BO sampling/thresholds")
    p.add_argument("--intermediate_models", action="store_true",
                   help="Enable intermediate feature models (EmbeddingRegion, FeatureDensity, EnsembleVote, Structural)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main — 与 run_loris_multi_pipeline.main() 结构相同，仅 Step 3 调用不同
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── --big preset: 大数据集推荐值 ─────────────────────────────────────────
    _BIG_PRESET = {
        "anchor_min_df": 10,
        "tfidf_top_k": 100,
        "min_coverage": 0.01,
        "max_entropy_threshold": 1.5,
    }
    if args.big:
        for _k, _v in _BIG_PRESET.items():
            if getattr(args, _k, None) is None:
                setattr(args, _k, _v)
    # fill remaining None with HParams defaults
    if args.anchor_min_df is None:
        args.anchor_min_df = 3
    if args.tfidf_top_k is None:
        args.tfidf_top_k = 200
    if args.min_coverage is None:
        args.min_coverage = 0.005
    if args.max_entropy_threshold is None:
        args.max_entropy_threshold = 2.0
    # Auto-detect GloVe path
    if args.glove_path is None:
        _default_glove = "/root/autodl-tmp/glove.6B.100d.txt"
        if os.path.isfile(_default_glove):
            args.glove_path = _default_glove
            log.info("Auto-detected GloVe: %s", _default_glove)

    dataset_cfg = DATASET_REGISTRY[args.dataset]

    # ── --prepare 模式 ───────────────────────────────────────────────────────
    if args.prepare:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True,
        )
        log.info("Preparing dataset: %s", dataset_cfg.display_name)
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        dataset_cfg.prepare_fn(dataset_cfg.data_dir)
        log.info("Dataset prepared. CSV files in %s",
                 dataset_cfg.data_dir / "processed")
        return

    # ── 正常 pipeline 模式 ───────────────────────────────────────────────────
    exp_dir = _setup_experiment(args, f"{dataset_cfg.name}_hybrid")
    configure_logging(exp_dir, debug=args.debug)

    log.info("Experiment directory: %s", exp_dir)
    log.info("Dataset: %s  (Chase Rule Discovery)", dataset_cfg.display_name)

    top_labels = (args.top_labels if args.top_labels is not None
                  else dataset_cfg.default_top_labels)

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
        tfidf_top_k=args.tfidf_top_k,
        min_coverage=args.min_coverage,
        max_entropy_threshold=args.max_entropy_threshold,
        baseline_mode=args.baseline_mode,
        extra_stop_words=dataset_cfg.stop_words,
        predicate_top_k=args.predicate_top_k,
        per_type_top_k=args.per_type_top_k if args.per_type_top_k > 0 else None,
        batch_metric_mode=args.batch_metric_mode,
        min_rule_fires=args.min_rule_fires,
        min_n_changes=args.min_n_changes,
        min_corr_prec=args.min_corr_prec,
        sim_threshold=args.sim_threshold,
        two_val=args.two_val,
        glove_path=args.glove_path,
    )
    # Chase-specific attrs (not in HParams dataclass)
    hp.track1_baseline = args.track1_baseline
    hp.track2_label_source = args.track2_label_source
    hp.skip_chase_test = args.skip_chase_test
    hp.sim_cascade_rounds = args.sim_cascade_rounds
    hp.sim_decay = args.sim_decay
    hp.sim_conf_threshold = args.sim_conf_threshold
    hp.sim_min_precision = args.sim_min_precision
    hp.inject_ml_baseline = not args.no_inject_ml_baseline
    hp.staged_pipeline = not args.no_staged_pipeline
    hp.enable_stage4 = args.enable_stage4
    hp.inject_multi_model = not args.no_inject_multi_model
    hp.no_title_predicates = not getattr(args, 'use_title_predicates', False)
    hp._rerun_patterns = args.rerun_patterns
    hp.no_weak_label_rescue = args.no_weak_label_rescue
    hp.weak_label_f1_threshold = args.weak_label_f1_threshold
    hp.weak_label_max_rules = args.weak_label_max_rules
    if args.sim_threshold_bins == "auto":
        hp.sim_threshold_bins = None  # will be auto-detected from embeddings
    else:
        hp.sim_threshold_bins = [float(x) for x in args.sim_threshold_bins.split(",")]

    _data = load_data(dataset_cfg, hp)
    train_X, val_X, test_X = _data[0], _data[1], _data[2]
    train_y, val_y, test_y = _data[3], _data[4], _data[5]
    label_names = _data[6]
    train_docs, val_docs = _data[7], _data[8]
    # two_val mode: val is val_bo; extra fields are val_select
    if hp.two_val:
        val_select_X = _data[9]
        val_select_y = _data[10]
        val_select_docs = _data[11]
        test_titles = _data[12] if len(_data) > 12 else None
    else:
        val_select_X = None
        val_select_y = None
        val_select_docs = None
        test_titles = _data[9] if len(_data) > 9 else None

    # ── 初始化模型池（支持从缓存恢复） ─────────────────────────────────────
    import pickle as _pkl_model
    _model_cache_path = exp_dir / "model_pool.pkl"
    _resume_dir = getattr(args, 'resume_exp_dir', None)
    _resume_model_path = Path(_resume_dir) / "model_pool.pkl" if _resume_dir else None

    if _resume_model_path and _resume_model_path.exists():
        log.info("Loading cached model pool from %s", _resume_model_path)
        with open(_resume_model_path, "rb") as _f:
            _cached = _pkl_model.load(_f)
        pool = _cached["pool"]
        val_f1_per_model = _cached["val_f1"]
        log.info("Restored %d models, best val micro-F1=%.4f",
                 len(pool), max(val_f1_per_model.values()) if val_f1_per_model else 0)
    else:
        pool = init_models(len(label_names), lora_model_name=args.lora_model)
        if args.no_encoder and "encoder_mlp" in pool:
            del pool["encoder_mlp"]
            log.info("Excluded encoder_mlp from model pool (--no_encoder)")
        val_f1_per_model = train_models(pool, train_X, train_y, val_X, val_y,
                                        subsample_ratio=args.subsample_ratio)
        # Save model pool for future resume
        try:
            with open(_model_cache_path, "wb") as _f:
                _pkl_model.dump({"pool": pool, "val_f1": val_f1_per_model}, _f)
            log.info("Saved model pool to %s", _model_cache_path)
        except Exception as _e:
            log.warning("Could not save model pool: %s", _e)
    baseline_micro_f1 = max(val_f1_per_model.values()) if val_f1_per_model else 0.0
    best_model_name = (max(val_f1_per_model, key=val_f1_per_model.get)
                       if val_f1_per_model else None)
    log.info("Best single-model val micro-F1 (baseline): %.4f (%s)",
             baseline_micro_f1, best_model_name)

    # ── --baseline_model: 强制指定基线（方案三：弱基线）─────────────────────
    if args.baseline_model is not None:
        if args.baseline_model not in pool:
            log.error("--baseline_model '%s' not in pool %s",
                      args.baseline_model, list(pool.keys()))
            sys.exit(1)
        best_model_name = args.baseline_model
        baseline_micro_f1 = val_f1_per_model.get(best_model_name, 0.0)
        log.info("Forced baseline model: %s (val micro-F1: %.4f)",
                 best_model_name, baseline_micro_f1)

    # ── 测试集评估 ──────────────────────────────────────────────────────────
    test_metrics_per_model = evaluate_models_on_test(pool, test_X, test_y)
    baseline_test_micro_f1 = max(
        (m["micro_f1"] for m in test_metrics_per_model.values()), default=0.0
    )
    baseline_test_macro_f1 = max(
        (m["macro_f1"] for m in test_metrics_per_model.values()), default=0.0
    )
    log.info("Best single-model test micro-F1: %.4f  macro-F1: %.4f",
             baseline_test_micro_f1, baseline_test_macro_f1)

    # ── 最佳模型验证集预测 ──────────────────────────────────────────────────
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

    # ── val_select predictions (two_val mode) ────────────────────────────────
    base_select_preds = None
    base_select_macro_f1 = None
    if val_select_X is not None and best_clf is not None:
        base_select_preds = best_clf.predict(val_select_X).astype(np.float32)
        base_select_macro_f1 = float(
            f1_score(val_select_y, base_select_preds, average="macro", zero_division=0)
        )
        log.info("Best model val_select macro-F1: %.4f", base_select_macro_f1)

    # ── 消除 data leakage: doc.lbl 从真实标签改为模型预测 ──────────────────
    # load_data() 用 ground truth 初始化 doc.lbl，但测试时 doc.lbl 是模型预测。
    # 规则应该学习"纠正模型预测的错误"，而非"利用已知真实标签做推理"。
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
        # val_select docs 也需要替换 lbl
        if val_select_docs is not None and base_select_preds is not None:
            for i, doc in enumerate(val_select_docs):
                doc.lbl.clear()
                doc.lbl.update(
                    label_names[j] for j in range(n_labels)
                    if base_select_preds[i, j] > 0
                )
    else:
        # 无模型预测时，清空 doc.lbl（和测试时无预测一致）
        for doc in val_docs:
            doc.lbl.clear()
        for doc in train_docs:
            doc.lbl.clear()
        if val_select_docs:
            for doc in val_select_docs:
                doc.lbl.clear()

    # ── 选择 Step 3 评估数据 ────────────────────────────────────────────────
    # fit_kwargs: split mode 专用 (谓词实例化用 train, 评估用 val)
    fit_kwargs: Dict = {}

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
    elif hp.rule_eval_data == "split":
        # 评估用 val，谓词实例化用 train
        base_rule_preds = base_val_preds
        base_rule_f1 = base_val_macro_f1
        rule_eval_docs, rule_eval_y, rule_eval_X = val_docs, val_y, val_X
        _fit_train_preds = (
            best_clf.predict(train_X).astype(np.float32) if best_clf else None
        )
        fit_kwargs = dict(
            fit_docs=train_docs,
            fit_labels=train_y,
            fit_base_predictions=_fit_train_preds,
        )
        log.info("rule_eval_data=split — greedy_instantiate on train (%d docs), "
                 "F1 eval on val (%d docs, base macro-F1=%.4f)",
                 len(train_X), len(val_X),
                 base_rule_f1 if base_rule_f1 is not None else 0.0)
    else:  # val
        base_rule_preds = base_val_preds
        base_rule_f1 = base_val_macro_f1
        rule_eval_docs, rule_eval_y, rule_eval_X = val_docs, val_y, val_X
        log.info("rule_eval_data=val — Step 3 uses validation set (%d docs)",
                 len(val_X))

    # ── track1_baseline=blank: 从零开始，让规则自己建立标注 ───────────────
    if getattr(hp, 'track1_baseline', 'model') == 'blank':
        log.info("track1_baseline=blank — Track 1 starts from zero predictions "
                 "(rules build labels from scratch)")
        base_rule_preds = np.zeros_like(rule_eval_y, dtype=np.float32)
        base_rule_f1 = 0.0
        # split mode 的 fit 基线也清零
        if 'fit_base_predictions' in fit_kwargs:
            fit_kwargs['fit_base_predictions'] = np.zeros_like(
                fit_kwargs['fit_base_predictions'], dtype=np.float32)

    # ── acceptance data (交叉验证) ──────────────────────────────────────────
    accept_kwargs: Dict = {}
    if hp.rule_eval_data == "train":
        # train 模式：用 val 做 cross-check（合理：模型没在 val 上训练）
        _blank_t1 = getattr(hp, 'track1_baseline', 'model') == 'blank'
        accept_kwargs = dict(
            accept_docs=val_docs,
            accept_y=val_y,
            accept_predictions=(np.zeros_like(val_y, dtype=np.float32) if _blank_t1
                                else base_val_preds),
            accept_f1=0.0 if _blank_t1 else base_val_macro_f1,
        )
        log.info("Acceptance cross-check on validation set (%d docs, base macro-F1=%.4f)",
                 len(val_X), base_val_macro_f1 if base_val_macro_f1 is not None else 0.0)
    else:
        # val / split 模式：禁用 accept cross-check
        # 原因：模型在 training 上 F1≈0.98（过拟合），用 training 做 cross-check 无意义
        # two_val 模式下 batch_select 已通过 val_bo/val_select 独立验证
        log.info("Acceptance cross-check disabled (rule_eval_data=%s — "
                 "training predictions too accurate due to overfitting)",
                 hp.rule_eval_data)

    # ── 保存超参数 ──────────────────────────────────────────────────────────
    _hp_dict = hp.to_dict()
    # Include Chase-specific attrs not in dataclass
    for _extra_attr in ("track1_baseline", "track2_label_source",
                        "sim_threshold_bins"):
        if hasattr(hp, _extra_attr):
            _hp_dict[_extra_attr] = getattr(hp, _extra_attr)
    with open(exp_dir / "hparams_initial.json", "w") as f:
        json.dump(_hp_dict, f, indent=2)

    # ─────────────────────────────────────────────────────────────────────────
    # 单轮规则发现（无重试）
    # ─────────────────────────────────────────────────────────────────────────
    final_rdl_set = None

    # ── 3.1 pattern abstraction ───────────────────────────────────────
    # batch 模式下 per-cluster 会自己做 pattern abstraction，跳过全局提取（省 ~2h）
    if hp.rule_strategy == "batch":
        log.info("=== Step 3.1  Pattern Abstraction "
                 "(skipped — batch mode does per-cluster) ===")
        store = PatternStore([], n_classes=len(label_names), label_names=label_names)
        _cluster_labels = None  # batch 会自己做 KMeans
    else:
        store, _cluster_labels = run_pattern_abstraction(
            train_X, train_y, hp, exp_dir)

    if len(store) == 0 and hp.rule_strategy != "batch":
        log.warning("No patterns extracted — proceeding with 0 rules.")
        final_rdl_set = RDLSet(rules=[], label_names=label_names)
    else:
        # ── 3.2 dynamic router ──────────────────────────────────────────
        selected_idx = run_dynamic_router(
            pool, train_X, train_y, rule_eval_X, rule_eval_y, hp, exp_dir,
            skip_router=args.no_router,
        )
        registered_names = register_selected_models(pool, selected_idx, label_names)

        # ── Register intermediate feature models (optional) ─────────────
        if getattr(args, 'intermediate_models', False):
            from models.intermediate import (
                EmbeddingRegionModel, FeatureDensityModel,
                EnsembleVoteModel, StructuralFeatureModel,
            )
            _im_t0 = time.time()
            log.info("=== Intermediate Feature Models: fitting 4 models ===")

            # 1. EmbeddingRegionModel
            _emb_model = EmbeddingRegionModel(num_labels=n_labels)
            _emb_model.fit(train_X, train_y, val_X, val_y)
            _emb_wrapper = _PredictWrapper(_emb_model, label_names)
            register_ml_model("loris_emb_region", _emb_wrapper)
            registered_names.append("loris_emb_region")
            log.info("  Registered: loris_emb_region")

            # 2. FeatureDensityModel
            _fd_model = FeatureDensityModel(num_labels=n_labels, top_k_keywords=20)
            _fd_model.fit(train_X, train_y, val_X, val_y)
            _fd_wrapper = _PredictWrapper(_fd_model, label_names)
            register_ml_model("loris_feat_density", _fd_wrapper)
            registered_names.append("loris_feat_density")
            log.info("  Registered: loris_feat_density")

            # 3. EnsembleVoteModel
            _ens_model = EnsembleVoteModel(
                num_labels=n_labels,
                source_model_names=registered_names[:],
            )
            _ens_model.fit(train_X, train_y, val_X, val_y)
            _ens_wrapper = _PredictWrapper(_ens_model, label_names)
            register_ml_model("loris_ensemble_vote", _ens_wrapper)
            registered_names.append("loris_ensemble_vote")
            log.info("  Registered: loris_ensemble_vote")

            # 4. StructuralFeatureModel
            _struct_model = StructuralFeatureModel(num_labels=n_labels)
            _struct_model.fit(train_X, train_y, val_X, val_y)
            _struct_wrapper = _PredictWrapper(_struct_model, label_names)
            register_ml_model("loris_structural", _struct_wrapper)
            registered_names.append("loris_structural")
            log.info("  Registered: loris_structural")

            log.info("=== Intermediate Feature Models: done in %.1fs, "
                     "%d total registered models ===",
                     time.time() - _im_t0, len(registered_names))

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
            # split mode: update fit_kwargs with mean_top_k train predictions
            if hp.rule_eval_data == "split":
                _rule_preds_list_mtk = [c.predict(train_X).astype(np.float32)
                                         for c in selected_clfs]
                _fit_train_preds_mtk = (np.mean(_rule_preds_list_mtk, axis=0) >= 0.5).astype(np.float32)
                fit_kwargs = dict(
                    fit_docs=train_docs,
                    fit_labels=train_y,
                    fit_base_predictions=_fit_train_preds_mtk,
                )
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
            # train_docs: 使用集成预测
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

        # ── 3.3 Chase Rule Discovery ───────────────────────────────────
        if hp.rule_strategy == "batch":
            # ── two_val: pass select data for independent batch_select ──
            _select_kwargs: Dict = {}
            if val_select_docs is not None:
                _blank = getattr(hp, 'track1_baseline', 'model') == 'blank'
                _select_kwargs = dict(
                    select_docs=val_select_docs,
                    select_y=val_select_y,
                    select_base_predictions=(
                        np.zeros_like(val_select_y, dtype=np.float32) if _blank
                        else base_select_preds),
                    select_f1=0.0 if _blank else base_select_macro_f1,
                )
            # When track1_baseline=blank, pass original model preds for Track 2
            _model_kwargs: Dict = {}
            if getattr(hp, 'track1_baseline', 'model') == 'blank':
                _model_kwargs = dict(
                    model_val_predictions=base_val_preds,
                    model_val_f1=base_val_macro_f1,
                )
                if val_select_docs is not None:
                    _model_kwargs['model_select_predictions'] = base_select_preds
                    _model_kwargs['model_select_f1'] = base_select_macro_f1
            final_rdl_set = run_rule_discovery_batch(
                train_X, train_y,
                pool, label_names,
                rule_eval_docs, rule_eval_X, rule_eval_y,
                hp, exp_dir,
                skip_router=args.no_router,
                global_registered_names=registered_names,
                base_val_predictions=base_rule_preds,
                base_val_f1=base_rule_f1,
                attempt=0,
                precomputed_cluster_labels=_cluster_labels,
                resume_dir=_resume_dir,
                resume_mode=getattr(args, 'resume_mode', 'all'),
                **accept_kwargs,
                **fit_kwargs,
                **_select_kwargs,
                **_model_kwargs,
                no_error_driven_filter=getattr(args, 'no_error_driven_filter', False) or not getattr(args, 'use_error_driven_filter', True),
                no_tree_warmup=not getattr(args, 'use_tree_warmup', False),
                train_docs=train_docs,
                group_min_corr_prec=getattr(args, 'group_min_corr_prec', 0.50),
            )
        else:
            final_rdl_set = run_rule_discovery(
                store, registered_names, label_names,
                rule_eval_docs, rule_eval_y, hp, exp_dir,
                attempt=0,
                base_val_predictions=base_rule_preds,
                base_val_f1=base_rule_f1,
                **accept_kwargs,
                **fit_kwargs,
            )

    log.info("Rule discovery complete: %d rules found.",
             len(final_rdl_set.rules) if final_rdl_set else 0)

    # ── 测试集最终评估 ──────────────────────────────────────────────────────
    n_labels = len(label_names)
    final_micro_f1 = 0.0
    final_macro_f1 = 0.0
    _test_is_blank = getattr(hp, 'track1_baseline', 'model') == 'blank'

    if _test_is_blank:
        # Blank mode: start from zero predictions — rules build labels from scratch
        base_test_preds = np.zeros((len(test_X), n_labels), dtype=np.float32)
        log.info("Blank mode: test evaluation starts from zero predictions")
    elif best_clf is not None:
        base_test_preds = best_clf.predict(test_X).astype(np.float32)
    else:
        base_test_preds = None

    if final_rdl_set is not None and len(final_rdl_set.rules) > 0:
        test_docs = []
        for i, t in enumerate(test_X):
            if _test_is_blank:
                pred_labels = set()  # blank: no initial labels
            elif base_test_preds is not None:
                pred_labels = {
                    label_names[j]
                    for j in range(n_labels)
                    if base_test_preds[i, j] > 0
                }
            else:
                pred_labels = set()
            ttl_text = test_titles[i] if test_titles and i < len(test_titles) else ""
            test_docs.append(Document(cnt=t, ttl=ttl_text, lbl=pred_labels))

        from pattern_extraction.predicates import SimPredicate as _SimPred
        from pattern_extraction.predicates import LabelPredicate as _LP
        from pattern_extraction.predicates import GroupPredicate as _GroupPred
        from sklearn.metrics import f1_score as _f1
        _nosim_rules = [r for r in final_rdl_set.rules
                        if not any(isinstance(p, _SimPred) for p in r.body)]
        _sim_rules = [r for r in final_rdl_set.rules
                      if any(isinstance(p, _SimPred) for p in r.body)]
        log.info("Rules: %d total (%d non-sim, %d sim)",
                 len(final_rdl_set.rules), len(_nosim_rules), len(_sim_rules))

        _skip_chase = getattr(hp, 'skip_chase_test', False)

        _has_group_rules = any(
            any(isinstance(p, _GroupPred) for p in r.body)
            for r in final_rdl_set.rules
        )
        _test_virtual_attrs = None
        if _has_group_rules:
            from rule_discovery.virtual_attributes import (
                compute_all_virtual_attributes, filter_degenerate_groups,
            )
            _test_texts = [d.cnt for d in test_docs]
            _test_emb = compute_embeddings(
                _test_texts, cache_path=str(exp_dir / "test_embeddings.npy"))
            _test_ml_proba = {}
            if hasattr(hp, '_test_ml_proba'):
                _test_ml_proba = hp._test_ml_proba
            _test_virtual_attrs, _ = compute_all_virtual_attributes(
                train_embeddings=compute_embeddings(
                    train_X, cache_path=str(exp_dir / "embeddings_train.npy")),
                target_embeddings=_test_emb,
                ml_proba_cache=_test_ml_proba,
                label_names=label_names,
            )
            _test_virtual_attrs = filter_degenerate_groups(_test_virtual_attrs)
            log.info("Built test virtual_attrs: %d attributes", len(_test_virtual_attrs))

        try:
            if _skip_chase or not _sim_rules:
                # ── Fast two-pass evaluation (no full chase) ──
                _result = base_test_preds.copy() if base_test_preds is not None else np.zeros(
                    (len(test_docs), n_labels), dtype=np.float32)

                # Pass 1: non-sim rules (text/ML + label(x))
                if _nosim_rules:
                    _t_start = time.time()
                    # Collect unique text predicates across all rules
                    _seen_ids = set()
                    _all_text_preds = []
                    for r in _nosim_rules:
                        for p in r.body:
                            if not isinstance(p, (MLThresholdPredicate, _SimPred, _LP)):
                                if id(p) not in _seen_ids:
                                    _seen_ids.add(id(p))
                                    _all_text_preds.append(p)
                    _test_text_masks = precompute_fire_masks(
                        _all_text_preds, test_docs)
                    _test_pred2idx = {id(p): i for i, p in enumerate(_all_text_preds)}
                    _test_ml_names = list({
                        p.model_name for r in _nosim_rules for p in r.body
                        if isinstance(p, MLThresholdPredicate)
                    })
                    _test_ml_proba = precompute_ml_proba(_test_ml_names, test_docs)
                    _result = _vectorized_staged_predict(
                        rules=_nosim_rules,
                        label_names=label_names,
                        n_docs=len(test_docs),
                        ml_proba_cache=_test_ml_proba,
                        text_fire_masks=_test_text_masks,
                        text_pred_to_idx=_test_pred2idx,
                        base=_result,
                    )
                    log.info("Test vectorized predict: %d rules on %d docs in %.1fs",
                             len(_nosim_rules), len(test_docs), time.time() - _t_start)
                    _ns_micro = float(_f1(test_y, _result, average="micro", zero_division=0))
                    _ns_macro = float(_f1(test_y, _result, average="macro", zero_division=0))
                    log.info("Test (non-sim rules) — micro-F1=%.4f  macro-F1=%.4f",
                             _ns_micro, _ns_macro)

                # Optional: RILL recursive chase to fixpoint (paper §6).
                # Amplifies rule effects via incremental chase propagation.
                if getattr(args, 'use_rill', False) and len(final_rdl_set.rules) > 0:
                    try:
                        from chase_inference.rill import RILLController
                        from chase_inference.oracle import GroundTruthOracle
                        _rill_oracle = GroundTruthOracle(
                            ground_truth=test_y, label_names=label_names,
                        )
                        _rill = RILLController(
                            rules=final_rdl_set.rules,
                            label_names=label_names,
                            oracle=_rill_oracle,
                            max_iterations=getattr(args, 'rill_max_iterations', 3),
                            trust_check=False,
                            conflict_mode="negative_wins",
                            sim_graphs=None,  # set below if Pass 2 builds graphs
                            verbose=False,
                        )
                        _n_changed_before = int((_result > 0).sum())
                        _rill_result = _rill.run(test_docs, base_predictions=_result)
                        _result = _rill_result.predictions
                        _n_changed_after = int((_result > 0).sum())
                        _rill_micro = float(_f1(test_y, _result, average="micro", zero_division=0))
                        _rill_macro = float(_f1(test_y, _result, average="macro", zero_division=0))
                        log.info(
                            "RILL: status=%s, %d iter, %d queries, "
                            "labels %d → %d (Δ=%+d), micro-F1=%.4f macro-F1=%.4f",
                            _rill_result.status,
                            _rill_result.n_iterations,
                            _rill_result.n_queries,
                            _n_changed_before, _n_changed_after,
                            _n_changed_after - _n_changed_before,
                            _rill_micro, _rill_macro,
                        )
                    except Exception as _rill_exc:
                        log.warning("RILL skipped: %s", _rill_exc, exc_info=True)

                # Pass 2: sim rules — single-pass SpMV evaluation
                if _sim_rules:
                    _rule_thresholds = sorted(set(
                        p.threshold for r in _sim_rules
                        for p in r.body if isinstance(p, _SimPred)
                    ))
                    _test_texts = [d.cnt for d in test_docs]
                    _test_emb = compute_embeddings(
                        _test_texts,
                        cache_path=str(exp_dir / "test_embeddings.npy"),
                    )
                    _test_sim_graphs = build_sim_graph(
                        _test_emb, threshold_bins=_rule_thresholds,
                        max_avg_degree=999999)  # test: apply all thresholds, no skip
                    log.info("Built test sim_graphs for %d docs, %d thresholds",
                             len(test_docs), len(_test_sim_graphs))

                    _label2idx = {name: i for i, name in enumerate(label_names)}
                    _label_state = (_result > 0).astype(np.float32)
                    _n_sim_fired = 0
                    for rule in _sim_rules:
                        cidx = _label2idx.get(rule.consequence)
                        if cidx is None:
                            continue
                        sim_pred = next((p for p in rule.body if isinstance(p, _SimPred)), None)
                        if sim_pred is None:
                            continue
                        adj = _test_sim_graphs.get(sim_pred.threshold)
                        if adj is None:
                            continue
                        # SpMV: check neighbors' labels
                        lps = [p for p in rule.body if isinstance(p, _LP)]
                        sim_mask = np.ones(len(test_docs), dtype=bool)
                        for lp in lps:
                            lidx = _label2idx.get(lp.label)
                            if lidx is not None:
                                sim_mask &= np.asarray(
                                    (adj @ _label_state[:, lidx]) > 0).ravel()
                            else:
                                sim_mask[:] = False
                        # Text/ML predicates on x
                        text_preds = [p for p in rule.body
                                      if not isinstance(p, (_SimPred, _LP))]
                        if text_preds:
                            for i, doc in enumerate(test_docs):
                                if sim_mask[i] and not all(p(doc) for p in text_preds):
                                    sim_mask[i] = False
                        # Apply consequence (add only)
                        fires = sim_mask & (_result[:, cidx] == 0)
                        if fires.any():
                            _result[fires, cidx] = 1.0
                            _n_sim_fired += 1
                    log.info("Sim rules: %d/%d rules fired at least once",
                             _n_sim_fired, len(_sim_rules))

                # Pass 3: group propagation rules
                if _has_group_rules and _test_virtual_attrs:
                    from rule_discovery.group_propagation import compute_group_fire_mask
                    _group_rules = [r for r in final_rdl_set.rules
                                    if any(isinstance(p, _GroupPred) for p in r.body)]
                    _label2idx = {name: i for i, name in enumerate(label_names)}
                    _label_state = (_result > 0).astype(np.float32)
                    _n_group_fired = 0
                    for _round in range(5):
                        _round_fired = 0
                        for rule in _group_rules:
                            cidx = _label2idx.get(rule.consequence)
                            if cidx is None:
                                continue
                            gp = next((p for p in rule.body if isinstance(p, _GroupPred)), None)
                            if gp is None:
                                continue
                            group_ids = _test_virtual_attrs.get(gp.attr_name)
                            if group_ids is None:
                                continue
                            lps = [p for p in rule.body if isinstance(p, _LP)]
                            if not lps:
                                continue
                            lp_label_idx = _label2idx.get(lps[0].label)
                            if lp_label_idx is None:
                                continue
                            fire_mask = compute_group_fire_mask(
                                group_ids, lp_label_idx, _label_state)
                            fires = fire_mask & (_result[:, cidx] == 0)
                            if fires.any():
                                _result[fires, cidx] = 1.0
                                _label_state[fires, cidx] = 1.0
                                _round_fired += int(fires.sum())
                                _n_group_fired += 1
                        if _round_fired == 0:
                            break
                    log.info("Group rules: %d rules fired, converged in %d rounds",
                             _n_group_fired, _round + 1)

                final_micro_f1 = float(_f1(test_y, _result, average="micro", zero_division=0))
                final_macro_f1 = float(_f1(test_y, _result, average="macro", zero_division=0))
                log.info("Test set (fast eval) — micro-F1=%.4f  macro-F1=%.4f",
                         final_micro_f1, final_macro_f1)
            else:
                # ── Full chase inference ──
                _rule_thresholds = sorted(set(
                    p.threshold for r in final_rdl_set.rules
                    for p in r.body if isinstance(p, _SimPred)
                ))
                _test_texts = [d.cnt for d in test_docs]
                _test_emb = compute_embeddings(
                    _test_texts,
                    cache_path=str(exp_dir / "test_embeddings.npy"),
                )
                _test_sim_graphs = build_sim_graph(
                    _test_emb, threshold_bins=_rule_thresholds,
                    max_avg_degree=999999)  # test: apply all thresholds, no skip
                log.info("Built test sim_graphs for %d docs, %d thresholds",
                         len(test_docs), len(_test_sim_graphs))

                chase_result = final_rdl_set.chase_predict(
                    test_docs,
                    base_predictions=base_test_preds,
                    sim_graphs=_test_sim_graphs,
                    sim_decay=getattr(hp, 'sim_decay', 1.0),
                    sim_conf_threshold=getattr(hp, 'sim_conf_threshold', 0.0),
                    virtual_attrs=_test_virtual_attrs,
                )
                combined = chase_result.predictions
                final_micro_f1 = float(_f1(test_y, combined, average="micro", zero_division=0))
                final_macro_f1 = float(_f1(test_y, combined, average="macro", zero_division=0))
                log.info("Test set (chase) — micro-F1=%.4f  macro-F1=%.4f",
                         final_micro_f1, final_macro_f1)
            log.info("Test set (model+rules) — micro-F1=%.4f  macro-F1=%.4f",
                     final_micro_f1, final_macro_f1)
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
        dataset_display_name=f"{dataset_cfg.display_name} (Chase)",
        test_metrics_per_model=test_metrics_per_model,
        baseline_test_macro_f1=baseline_test_macro_f1,
        final_macro_f1=final_macro_f1,
    )

    # ── per-rule correction analysis ─────────────────────────────────────────
    if (final_rdl_set is not None and len(final_rdl_set.rules) > 0
            and base_test_preds is not None):
        test_docs_for_analysis = []
        for i, t in enumerate(test_X):
            pred_labels = {
                label_names[j]
                for j in range(n_labels)
                if base_test_preds[i, j] > 0
            }
            ttl_text = test_titles[i] if test_titles and i < len(test_titles) else ""
            test_docs_for_analysis.append(Document(cnt=t, ttl=ttl_text, lbl=pred_labels))
        analyze_rule_corrections(
            final_rdl_set, test_docs_for_analysis, test_y,
            base_test_preds, label_names, exp_dir,
        )

    log.info("Chase pipeline complete. Results in %s", exp_dir)


if __name__ == "__main__":
    main()
