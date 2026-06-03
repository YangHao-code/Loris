"""Chase rule-discovery pipeline steps.

The rule-discovery stage of the LORIS chase pipeline: per-cluster pattern
abstraction, candidate predicate selection, Track1 (Bayesian-optimisation) and
Track2 (propagation) rule search, and label-cooccurrence / error-driven helpers.

Migrated verbatim from ``run_loris_chase_pipeline.py`` (Phase 6), imports
retargeted to the loris.* packages. Orchestration (arg parsing, main) lives in
loris.pipeline.orchestrator; shared model/data steps come from
loris.pipeline.shared (re-exported from the legacy multi pipeline during
migration).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score

from loris.pipeline.shared import (
    HParams, DATASET_REGISTRY, load_data, configure_logging,
    init_models, train_models, evaluate_models_on_test,
    run_pattern_abstraction, run_dynamic_router, register_selected_models,
    _select_top_predicates, _select_cluster_models, save_and_print_results,
    analyze_rule_corrections, _setup_experiment, _PredictWrapper,
)
from loris.document import Document
from loris.patterns import PatternAbstractor, PatternStore
from loris.predicates import register_ml_model, LabelPredicate
from loris.rules import RDLSet
from loris.rules.discovery import (
    ChaseRuleLearner, _save_stage_logs, _extract_trial_stats,
    _vectorized_staged_predict, precompute_fire_masks, precompute_ml_proba,
    MLThresholdPredicate, _fast_per_label_f1,
    _error_driven_filter, _tree_seeded_rules, ERROR_AWARE_WEIGHTS,
)
from loris.rules.sim_graph import (
    auto_threshold_bins, compute_embeddings, build_sim_graph,
    save_sim_graphs, load_sim_graphs,
    precompute_neighbor_label_masks, precompute_neighbor_label_counts,
)

log = logging.getLogger("loris_chase_pipeline")


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
    effective_max_trials = (hp.max_trials if getattr(hp, 'no_adaptive_trials', False)
                            else max(hp.max_trials, 15 * len(label_names)))
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

        # ── per-predicate-family ablation filter ──
        # Keep ML/Label/Group predicates (base/structural signal) always; restrict
        # only the TEXTUAL families to the allowed set. Empty ⇒ all (golden-neutral).
        # Short keys: match/freq/before/cooccur.
        _fam = (getattr(hp, "predicate_families", "") or "").strip()
        if _fam:
            _MAP = {"match": "MatchPredicate", "freq": "FreqPredicate",
                    "before": "BeforePredicate", "cooccur": "CooccurPredicate"}
            _allowed = {_MAP.get(s.strip(), s.strip()) for s in _fam.split(",") if s.strip()}
            _TEXTUAL = {"MatchPredicate", "CooccurPredicate", "BeforePredicate", "FreqPredicate"}
            _before_n = len(filtered_preds)
            filtered_preds = [p for p in filtered_preds
                              if type(p).__name__ not in _TEXTUAL
                              or type(p).__name__ in _allowed]
            log.info("  Cluster %d predicate-family filter [%s]: %d → %d",
                     cid, _fam, _before_n, len(filtered_preds))

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
        effective_max_trials = (hp.max_trials if getattr(hp, 'no_adaptive_trials', False)
                                else max(hp.max_trials, 15 * len(label_names)))
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
        from rule_discovery.group_propagation import (
            discover_group_rules, discover_equal_rules,
        )

        log.info("=== Track 2 Group Propagation: computing virtual attributes ===")

        # Compute train embeddings for KMeans fitting
        _train_texts = [d.cnt for d in train_docs]
        _train_emb_cache = str(exp_dir / "embeddings_train.npy")
        _train_emb = compute_embeddings(_train_texts, cache_path=_train_emb_cache)

        # Virtual attributes on val_bo — FIT kmeans + text-attr vocabularies on
        # train_docs, transform val_bo (mirrors KMeans fit-on-train/predict).
        # _text_vocabs is returned for reuse on val_select / test so attribute
        # value-columns stay consistent across BO / select / test.
        bo_virtual_attrs, _kmeans_models, _text_vocabs = compute_all_virtual_attributes(
            train_embeddings=_train_emb,
            target_embeddings=bo_embeddings,
            train_docs=train_docs,
            target_docs=val_docs,
            label_names=label_names,
        )
        bo_virtual_attrs = filter_degenerate_groups(bo_virtual_attrs)
        log.info("Track 2 Group: %d virtual attributes on val_bo", len(bo_virtual_attrs))

        # Discover group rules (exhaustive enumeration)
        _group_min_prec = group_min_corr_prec
        # Asymmetric per-label gate (strategy 3): tighten precise head labels,
        # let weak tail labels through. Opt-in (default off ⇒ golden-neutral:
        # base_label_prec=None reproduces the flat-gate behaviour bit-for-bit).
        _base_lbl_prec = None
        if getattr(hp, "asym_group_gate", False):
            _bp = (_t2_base > 0)
            _tp = (_bp & (val_y == 1)).sum(0).astype(np.float64)
            _fp = (_bp & (val_y == 0)).sum(0).astype(np.float64)
            _base_lbl_prec = np.where(_tp + _fp > 0, _tp / np.maximum(_tp + _fp, 1.0), 0.0)
        _group_trials = discover_group_rules(
            virtual_attrs=bo_virtual_attrs,
            label_state=bo_label_state.astype(np.float32),
            label_names=label_names,
            val_labels=val_y,
            existing_predictions=_t2_base,
            val_docs=val_docs,
            min_fires=max(3, effective_min_fires // 2),
            min_corr_prec=_group_min_prec,
            base_label_prec=_base_lbl_prec,
            narrow_prec_floor=(0.30 if getattr(hp, "asym_group_gate", False) else None),
        )
        log.info("Track 2 Group: %d rules discovered on val_bo", len(_group_trials))

        # Comparison-consequence (x.lbl=y.lbl) rules — one per attribute, admitted
        # accuracy-guided (paper §5.2/§6.1). These cannot pass through batch_select
        # (consequence=""), so they are gated here on val_bo f1_gain and appended
        # to the final rule set directly (see final_rules assembly below).
        _equal_trials = discover_equal_rules(
            virtual_attrs=bo_virtual_attrs,
            val_labels=val_y,
            existing_predictions=_t2_base,
            min_fires=max(3, effective_min_fires // 2),
            min_corr_prec=_group_min_prec,
            min_f1_gain=hp.min_f1_gain,
        )
        _equal_rules = [rdl for _t, rdl in _equal_trials]
        log.info("Track 2 Equal: %d comparison-consequence rules admitted on val_bo",
                 len(_equal_rules))

        # Virtual attributes on val_select (for batch_select)
        if _is_two_val:
            # Transform val_select with the SAME fitted kmeans + text vocab.
            sel_virtual_attrs, _, _ = compute_all_virtual_attributes(
                train_embeddings=_train_emb,
                target_embeddings=sel_embeddings,
                train_docs=train_docs,
                target_docs=select_docs,
                label_names=label_names,
                kmeans_models=_kmeans_models,
                text_vocabs=_text_vocabs,
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
            final_rules = (list(all_seed_rules) + list(track2_rdl.rules)
                           + _equal_rules)
            rdl_set = RDLSet(final_rules, label_names)
            log.info("Staged final: %d seed + %d Track 2 + %d equal = %d total rules",
                     len(all_seed_rules), len(track2_rdl.rules), len(_equal_rules),
                     len(final_rules))
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
            if _equal_rules:
                rdl_set = RDLSet(list(rdl_set.rules) + _equal_rules, label_names)
                log.info("Track 2 Equal: appended %d comparison-consequence rules "
                         "(total %d)", len(_equal_rules), len(rdl_set.rules))

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

