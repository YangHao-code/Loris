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

    # ── Staged error-driven rule learning (ML → FN-add → FP-remove → propagation) ─
    stage0_ml: bool = True            # Stage 0: ML-ensemble ADD baseline rules
    stage1_fn_add: bool = True        # Stage 1: FN-driven text ADD rules (NEW)
    stage2_fp_remove: bool = True     # Stage 2: FP-driven REMOVE rules
    stage3_propagation: bool = True   # Stage 3: composite label/sim propagation
    fn_add_min_val_prec: float = 0.70   # precision floor for FN→ADD rules
    fn_add_top_k_per_label: int = 8     # # FN-discriminative predicates mined per label
    fn_add_ml_guard: bool = True        # AND an ml_thresh(L) guard onto the text body
    fn_add_min_fires: int = 0           # 0 ⇒ use effective_min_fires
    prop_require_text: bool = True      # propagation rules must carry a text predicate
    prop_label_source: str = "track1"   # "gt" | "track1" | "model" seed for propagation BO
    # ^ B2 fix: mine comparison/group rules against the SAME label state seen at
    #   inference (track1 predictions), NOT ground truth. "gt" leaks val labels into
    #   the fire-mask so rules that look good on GT neighbours don't fire at test.
    #   Scoring (F1 gain) still uses val_y; only the neighbour fire-mask changes.
    rill_budget_sweep: str = ""         # e.g. "0,10,25,50,100,200"; empty ⇒ no sweep
    # ── Count allocation (per stage / per type / per label) ──────────────────────
    stage1_max_rules_per_label: int = 0   # 0 = uncapped (redundancy elimination only)
    stage2_max_rules_per_label: int = 0
    stage3_max_rules_per_label: int = 0
    prop_trial_frac: float = 0.5        # fraction of max_trials given to propagation BO
    sim_min_avg_degree: float = 20.0    # sim-graph connectivity floor for RILL propagation
    sim_target_degrees: str = ""        # "" = default (5,10,20,40); else CSV finer/denser avg-degree bins (enrich similarity discovery)
    rill_sweep_max_docs: int = 3000     # cap test docs for the RILL budget sweep (sim graph is O(n^2); 0 ⇒ 3000)

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
