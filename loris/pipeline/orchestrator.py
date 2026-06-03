"""LORIS chase pipeline orchestration: CLI argument parsing and ``main``.

Wires together data loading, model init/training, pattern abstraction, dynamic
model selection, chase rule discovery, and evaluation/reporting into the
end-to-end run. Migrated verbatim from ``run_loris_chase_pipeline.py`` (Phase 6),
imports retargeted to the loris.* packages.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score

from loris.document import Document
from loris.patterns import PatternAbstractor, PatternStore
from loris.predicates import register_ml_model, LabelPredicate, MLThresholdPredicate
from loris.rules import RDLSet
from loris.rules.sim_graph import (
    auto_threshold_bins,
    compute_embeddings,
    build_sim_graph,
    save_sim_graphs,
    load_sim_graphs,
    precompute_neighbor_label_masks,
    precompute_neighbor_label_counts,
)
from loris.rules.discovery import (
    _vectorized_staged_predict,
    precompute_fire_masks,
    precompute_ml_proba,
)
from loris.pipeline.shared import (
    HParams, DATASET_REGISTRY, load_data, configure_logging,
    init_models, train_models, evaluate_models_on_test,
    run_pattern_abstraction, run_dynamic_router, register_selected_models,
    _select_top_predicates, _select_cluster_models, save_and_print_results,
    analyze_rule_corrections, _setup_experiment, _PredictWrapper,
)
from loris.pipeline.steps import (
    run_rule_discovery, run_rule_discovery_batch,
    _generate_label_cooccurrence_rules, _extract_error_driven_predicates,
)

log = logging.getLogger("loris_chase_pipeline")


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
                   choices=["full", "fast", "sim"],
                   help="Pattern extraction mode")
    p.add_argument("--sim_threshold", type=float, default=None,
                   help="Similarity threshold for sim mode (default: 0.45)")
    p.add_argument("--predicate_families", default="",
                   help="Per-family ablation: comma-sep textual families to KEEP "
                        "(match,freq,before,cooccur). Empty = all. ML/label/group preds "
                        "always kept. Lets you isolate each family's delta over a fixed base.")
    p.add_argument("--no_adaptive_trials", action="store_true", default=False,
                   help="Respect --max_trials exactly (skip the 15×n_labels floor) — for fast ablation runs.")
    p.add_argument("--model_pool", choices=["all", "embedding"], default="all",
                   help="'all' = full pool (tfidf + neural + encoder). 'embedding' = "
                        "drop tfidf bag-of-words models (textcnn/bilstm/encoder only) so "
                        "lexical RULES add orthogonal signal vs a semantic base.")
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
    p.add_argument("--asym_group_gate", action="store_true", default=False,
                   help="Strategy 3: asymmetric per-label gate for Track 2 group rules "
                        "(max(group_min_corr_prec, base_prec_L+0.05)) + wider text-narrowing band. "
                        "Default off ⇒ golden-neutral.")
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
    hp.predicate_families = args.predicate_families
    hp.no_adaptive_trials = args.no_adaptive_trials
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
        pool = init_models(len(label_names), lora_model_name=args.lora_model,
                           drop_tfidf=(args.model_pool == "embedding"))
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
    # B2 fix (paper §3.3): baseline = the SINGLE val-selected model's test score,
    # NOT max over all models on test. Reporting max-over-models inflates the
    # baseline (it cherry-picks the best model *on the test set*, which the
    # pipeline never actually selects), making rule deltas look flat/negative.
    # The rules stack on `best_clf` (= pool[best_model_name], chosen on val), so
    # the honest baseline is that same model's test score. Per-model table is
    # still kept (test_metrics_per_model) for diagnostics.
    test_metrics_per_model = evaluate_models_on_test(pool, test_X, test_y)
    _baseline_test_metrics = test_metrics_per_model.get(best_model_name, {})
    baseline_test_micro_f1 = float(_baseline_test_metrics.get("micro_f1", 0.0))
    baseline_test_macro_f1 = float(_baseline_test_metrics.get("macro_f1", 0.0))
    log.info("Baseline (val-selected model '%s') test micro-F1: %.4f  macro-F1: %.4f",
             best_model_name, baseline_test_micro_f1, baseline_test_macro_f1)

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
            from loris.models.intermediate import (
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

        from loris.predicates import SimPredicate as _SimPred
        from loris.predicates import LabelPredicate as _LP
        from loris.predicates import GroupPredicate as _GroupPred
        from sklearn.metrics import f1_score as _f1
        _nosim_rules = [r for r in final_rdl_set.rules
                        if not any(isinstance(p, _SimPred) for p in r.body)]
        _sim_rules = [r for r in final_rdl_set.rules
                      if any(isinstance(p, _SimPred) for p in r.body)]
        # C-8 Part 2: rules whose body has a LabelPredicate but NO Sim/Group
        # predicate are PURE label-dependent (multi-round) rules, e.g.
        # `label(x,τ1) ∧ match(x,phrase) → τ2`. The fast evaluator
        # (_vectorized_staged_predict) SKIPS any LabelPredicate rule, so these
        # would silently never fire at test. They must go through the full chase
        # (which iterates label state to fixpoint). Sim rules go through the full
        # chase too; group/equal rules are handled by the fast-path Pass-3.
        _label_dep_rules = [
            r for r in final_rdl_set.rules
            if any(isinstance(p, _LP) for p in r.body)
            and not any(isinstance(p, (_SimPred, _GroupPred)) for p in r.body)
        ]
        log.info("Rules: %d total (%d non-sim, %d sim, %d pure-label-dependent)",
                 len(final_rdl_set.rules), len(_nosim_rules), len(_sim_rules),
                 len(_label_dep_rules))

        _skip_chase = getattr(hp, 'skip_chase_test', False)

        _has_group_rules = any(
            any(isinstance(p, _GroupPred) for p in r.body)
            for r in final_rdl_set.rules
        )
        _test_virtual_attrs = None
        if _has_group_rules:
            from loris.rules.virtual_attributes import (
                compute_all_virtual_attributes, filter_degenerate_groups,
            )
            _test_texts = [d.cnt for d in test_docs]
            _test_emb = compute_embeddings(
                _test_texts, cache_path=str(exp_dir / "test_embeddings.npy"))
            # Test path is a separate scope from BO/select: re-FIT kmeans + text
            # vocab on train_docs (deterministic → same columns as BO) and
            # transform test_docs. text_vocabs/kmeans discarded (re-derived).
            _test_virtual_attrs, _, _ = compute_all_virtual_attributes(
                train_embeddings=compute_embeddings(
                    train_X, cache_path=str(exp_dir / "embeddings_train.npy")),
                target_embeddings=_test_emb,
                train_docs=train_docs,
                target_docs=test_docs,
                label_names=label_names,
            )
            _test_virtual_attrs = filter_degenerate_groups(_test_virtual_attrs)
            log.info("Built test virtual_attrs: %d attributes", len(_test_virtual_attrs))

        try:
            if _skip_chase or (not _sim_rules and not _label_dep_rules):
                # ── Fast two-pass evaluation (no full chase) ──
                # Used only when nothing needs the iterative chase: no sim rules
                # and no pure label-dependent rules. (--skip_chase_test forces
                # this even if label-dependent rules exist — caller's choice.)
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

                # Pass 3: group propagation rules (add) + comparison consequence
                # (x.lbl=y.lbl, equal). Both cascade together to fixpoint; this
                # is the fast-path mirror of MultiChase._eval_group_rules /
                # _eval_equal_rules (used when there are 0 sim rules).
                if _has_group_rules and _test_virtual_attrs:
                    from loris.rules.group_propagation import compute_group_fire_mask
                    _group_rules = [r for r in final_rdl_set.rules
                                    if r.consequence_op != "equal"
                                    and any(isinstance(p, _GroupPred) for p in r.body)]
                    _equal_rules = [r for r in final_rdl_set.rules
                                    if r.consequence_op == "equal"]
                    _label2idx = {name: i for i, name in enumerate(label_names)}
                    _label_state = (_result > 0).astype(np.float32)
                    _n_group_fired = 0
                    _n_equal_fired = 0
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
                        # Comparison consequence x.lbl=y.lbl: batched SpMV label-set
                        # copy among co-members ( new = (M @ (Mᵀ @ state)) > 0 ).
                        for rule in _equal_rules:
                            gp = next((p for p in rule.body if isinstance(p, _GroupPred)), None)
                            if gp is None:
                                continue
                            M = _test_virtual_attrs.get(gp.attr_name)
                            if M is None:
                                continue
                            new = np.asarray(M.dot(M.T.dot(_label_state))) > 0
                            add = new & (_result == 0)
                            if add.any():
                                _result[add] = 1.0
                                _label_state[add] = 1.0
                                _round_fired += int(add.sum())
                                _n_equal_fired += 1
                        if _round_fired == 0:
                            break
                    log.info("Group rules: %d add-fires, %d equal-fires, converged in %d rounds",
                             _n_group_fired, _n_equal_fired, _round + 1)

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
                # Only build similarity graphs if sim rules exist; for a
                # label-dependent-only rule set (C-8 Part 2) there are no sim
                # thresholds, so skip the (expensive) embedding pass and run the
                # chase with empty sim_graphs — label rules fire via label state.
                if _rule_thresholds:
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
                else:
                    _test_sim_graphs = {}
                    log.info("Full chase with 0 sim rules (label-dependent path) "
                             "— no sim_graphs built")

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
