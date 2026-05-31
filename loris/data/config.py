"""Pipeline hyperparameters (HParams).

The single configuration dataclass threaded through the LORIS pipeline — data
loading, pattern abstraction, model selection, rule discovery, and quality
gates. Migrated from ``run_loris_multi_pipeline.py`` (Phase 5); the legacy
module re-exports it as a shim.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class HParams:
    # ── Data ──────────────────────────────────────────────────────────────────
    subset_size: int = 0
    val_ratio: float = 0.40
    top_labels: int = 20
    two_val: bool = False  # split val into val_bo (BO) + val_select (batch_select)

    # ── PatternAbstractor ─────────────────────────────────────────────────────
    n_clusters: Optional[int] = None
    min_coverage: float = 0.005
    max_entropy_threshold: float = 2.0
    tfidf_top_k: int = 200
    pattern_mode: str = "full"
    extra_stop_words: Optional[set] = None
    anchor_min_df: int = 3
    glove_path: Optional[str] = None

    # ── Dynamic Router ────────────────────────────────────────────────────────
    router_feat_dim: int = 128
    router_hidden_dim: int = 256
    k_models: int = 3
    router_sigma: float = 0.1
    router_num_samples: int = 500
    router_epochs: int = 30
    router_lr: float = 1e-3

    # ── RuleLearner ───────────────────────────────────────────────────────────
    max_trials: int = 300
    top_n_rules: int = 10
    min_coverage_rule: float = 0.005
    top_per_type: int = 30
    rule_strategy: str = "greedy"
    rule_batch_sort: bool = True
    cluster_model_selection: str = "global"
    rule_min_precision: float = 0.50
    rule_eval_data: str = "val"
    min_rule_fires: int = 1
    min_n_changes: int = 1
    min_corr_prec: float = 0.5
    sim_threshold: Optional[float] = None  # similarity threshold for sim mode
    sim_cascade_rounds: int = 1  # BO cascade simulation rounds for sim rules
    predicate_top_k: int = 300
    per_type_top_k: Optional[int] = 300  # top-k per predicate type; None = global top_k
    batch_metric_mode: str = "cluster_local"

    # ── Baseline mode ──────────────────────────────────────────────────────────
    baseline_mode: str = "best"

    # ── Quality gates ─────────────────────────────────────────────────────────
    min_rules: int = 1
    min_avg_coverage: float = 0.01
    min_f1_gain: float = 0.001
    max_avg_body_len: int = 8

    # ── Retry control ─────────────────────────────────────────────────────────
    max_retries: int = 1

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("extra_stop_words") is not None:
            d["extra_stop_words"] = sorted(d["extra_stop_words"])
        return d
