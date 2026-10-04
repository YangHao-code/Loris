from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


_DISTRIBUTION_BALANCED_DEFAULTS = {
    "db_focal_alpha": 0.5,
    "db_focal_gamma": 2.0,
    "db_map_alpha": 0.1,
    "db_map_beta": 10.0,
    "db_map_mu": 0.9,
    "db_init_bias": 0.05,
    "db_negative_scale": 2.0,
}


@dataclass
class ExperimentConfig:
    # Data/protocol.  The annotation fraction normally includes fit + validation;
    # under loris_exact it is explicitly scoped to the frozen full-fit role.
    dataset: str = "reuters21578"
    loris_root: str = "."
    protocol: str = "clean_ssl"  # clean_ssl | loris_compat | loris_exact
    top_labels: int = 30
    labeled_fraction: float = 0.05
    val_fraction: float = 0.20
    select_fraction: float = 0.50
    # Acquisition membership is intentionally independent of ``seed`` so
    # model-RNG replications can share one immutable labeled/unlabeled split.
    acquisition_seed: int = 42
    seed: int = 0
    sampling: str = "coverage"  # coverage | dominant | random
    max_train_docs: int = 0
    max_unlabeled_docs: int = 0
    max_test_docs: int = 0
    rcv1_full: bool = False

    # Encoder/head.
    model_name: str = "microsoft/deberta-v3-base"
    # Explicitly recorded so offline runs do not depend on a shell-specific
    # HF_HOME.  Set to an empty string to use the transformers default.
    cache_dir: str = ""
    # ``t5enc`` selects the full non-autoregressive T5 encoder-decoder from
    # Kementchedjhieva & Chalkidis (Findings ACL 2023), not merely another
    # prediction head on an encoder-only backbone.  Keeping the route on this
    # existing field preserves every pre-T5Enc config/checkpoint by default.
    head: str = "label_attention"  # label_attention | pooled | *_linear | t5enc
    # The paper's T5Enc decoder consumes exactly one fixed token per label.
    # ``pseudo`` uses T5 sentinel tokens and is the faithful primary for Loris,
    # whose labels do not come with manually curated one-token descriptors.
    # ``label_name`` is exploratory and is accepted only when every complete
    # label name tokenizes to one unique, non-special token (never truncated).
    t5enc_descriptor_mode: str = "pseudo"  # pseudo | label_name
    # Optional preregistered digest over the four required local snapshot
    # assets.  Every T5Enc checkpoint still records the individual paths,
    # sizes, and SHA-256 values even when this preflight value is empty.
    t5enc_snapshot_sha256: str = ""
    # Optional sealed provenance manifest for a local encoder-only snapshot
    # (currently the fit-only Reuters TAPT pilot).  When set, training and the
    # one-shot final evaluator re-hash the snapshot around every load and bind
    # the resulting bytes into the classifier checkpoint.  Empty preserves
    # every legacy/remote encoder config.
    encoder_snapshot_manifest: str = ""
    max_length: int = 384
    truncation: str = "head_tail"  # head_tail | head
    dropout: float = 0.10
    projection_dim: int = 128
    init_label_queries: bool = True
    gradient_checkpointing: bool = False

    # Optimisation.
    method: str = "ssl"  # supervised | ssl
    epochs: int = 12
    min_epochs: int = 3
    patience: int = 3
    batch_size: int = 8
    unlabeled_ratio: int = 4
    eval_batch_size: int = 32
    grad_accum_steps: int = 1
    learning_rate: float = 2.0e-5
    head_learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    optimizer: str = "adamw"  # adamw | adafactor
    scheduler: str = "cosine"  # cosine | constant_with_warmup
    max_grad_norm: float = 1.0
    amp: str = "bf16"  # bf16 | fp16 | none; fp16 remains an explicit compatibility choice
    num_workers: int = 2
    # ``distribution_balanced`` is the Reuters ResampleLoss configuration from
    # Huang et al. (EMNLP 2021).  Its opt-in fields are omitted from every
    # non-DB serialization to preserve historical exact config key sets.
    supervised_loss: str = "weighted_bce"  # bce | weighted_bce | asl | two_way | two_way_official | distribution_balanced
    max_pos_weight: float = 10.0
    asl_gamma_negative: float = 4.0
    asl_gamma_positive: float = 1.0
    asl_negative_clip: float = 0.05
    db_focal_alpha: float = 0.5
    db_focal_gamma: float = 2.0
    db_map_alpha: float = 0.1
    db_map_beta: float = 10.0
    db_map_mu: float = 0.9
    db_init_bias: float = 0.05
    db_negative_scale: float = 2.0
    # CVPR 2023 Two-Way variants use TN=1 and recommend TP=T=4.  ``two_way``
    # is the paper-equation variant; ``two_way_official`` follows the released
    # code's positive-axis masking and uses the same TP value.
    two_way_temperature: float = 4.0
    # Multi-label R-Drop adapter.  A positive weight makes two labeled dropout
    # forwards, supervises both equally, and aligns their independent
    # Bernoulli logits with symmetric KL.  Zero is omitted from serialized
    # configs to preserve pre-existing exact-run key sets.
    rdrop_weight: float = 0.0

    # Semi-supervision.
    ema_decay: float = 0.995
    unsup_weight: float = 1.0
    unsup_ramp_epochs: int = 4
    ssl_warmup_epochs: int = 1
    consistency_weight: float = 0.10
    prototype_weight: float = 0.05
    contrastive_weight: float = 0.05
    # Clean DENN training-objective adapter.  A positive weight enables two
    # independent labeled forwards, applies the same supervised criterion once
    # to their concatenated logits/duplicated targets, and applies DENN's DCL
    # equation to the two encoder-CLS views.  The field names are retained for
    # existing run/config compatibility.  This does not reproduce DENN's
    # test-time evaluation protocol or the rest of its model stack.
    denn_dcl_weight: float = 0.0
    denn_dcl_temperature: float = 0.05
    strong_mask_prob: float = 0.15
    pseudo_pos_threshold: float = 0.95
    pseudo_neg_threshold: float = 0.98
    pseudo_min_threshold: float = 0.75
    pseudo_target_precision: float = 0.90
    pseudo_target_npv: float = 0.98
    pseudo_min_calibration_support: int = 3
    pseudo_strategy: str = "cap"  # cap | adaptive | fixed
    cap_refresh_interval: int = 1
    cap_prior_smoothing: float = 2.0
    cap_min_positive_rate: float = 0.001
    cap_max_positive_rate: float = 0.50
    distribution_alignment: bool = True
    softmatch: bool = True
    # A tail label must expose at least one positive pseudo-label in the
    # current unlabeled batch before its negative pseudo-labels are consumed.
    # This prevents an immature teacher from applying an all-negative update
    # to every rare label.  The switch is recorded in each run and can be
    # disabled for an exact ungated ablation.
    require_batch_positive_for_negative: bool = True

    # Balanced binary angular-margin auxiliary (NeurIPS 2024 S2ML2-BBAM).
    bbam_weight: float = 0.10
    bbam_margin: float = 0.20
    bbam_scale: float = 10.0
    bbam_stats_momentum: float = 0.90
    hard_negative_ratio: int = 5

    # Validation/checkpointing.  Threshold mode is selected only from val_ctrl.
    selection_metric: str = "mean_f1"  # macro_f1 | micro_f1 | mean_f1
    # val_ctrl always fits the threshold.  A distinct val_select role may
    # select the inference model and epoch without refitting that threshold.
    checkpoint_role: str = "val_ctrl"  # val_ctrl | val_select
    threshold_shrinkage: float = 10.0
    ensure_one_label: bool = True
    prefer_ema: bool = True

    # Output.
    output_dir: str = "runs"
    run_name: str = ""
    save_checkpoint: bool = False
    save_predictions: bool = False
    skip_test: bool = False
    offline: bool = False

    def validate(self) -> None:
        if not 0.0 < self.labeled_fraction <= 1.0:
            raise ValueError("labeled_fraction must be in (0, 1]")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        if not 0.0 <= self.select_fraction < 1.0:
            raise ValueError("select_fraction must be in [0, 1)")
        if self.protocol not in {"clean_ssl", "loris_compat", "loris_exact"}:
            raise ValueError(
                "protocol must be clean_ssl, loris_compat, or loris_exact"
            )
        if self.method not in {"supervised", "ssl"}:
            raise ValueError("method must be supervised or ssl")
        if self.head not in {
            "pooled", "label_attention", "pooled_linear", "label_attention_linear",
            "t5enc",
        }:
            raise ValueError(
                "head must be pooled, label_attention, pooled_linear, "
                "label_attention_linear, or t5enc"
            )
        if self.t5enc_descriptor_mode not in {"pseudo", "label_name"}:
            raise ValueError(
                "t5enc_descriptor_mode must be pseudo or label_name"
            )
        if self.t5enc_snapshot_sha256 and (
            len(self.t5enc_snapshot_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.t5enc_snapshot_sha256
            )
        ):
            raise ValueError(
                "t5enc_snapshot_sha256 must be an empty string or 64 lowercase hex characters"
            )
        if self.truncation not in {"head", "head_tail"}:
            raise ValueError("truncation must be head or head_tail")
        if self.sampling not in {"coverage", "dominant", "random"}:
            raise ValueError("sampling must be coverage, dominant, or random")
        if self.amp not in {"fp16", "bf16", "none"}:
            raise ValueError("amp must be fp16, bf16, or none")
        if self.optimizer not in {"adamw", "adafactor"}:
            raise ValueError("optimizer must be adamw or adafactor")
        if self.scheduler not in {"cosine", "constant_with_warmup"}:
            raise ValueError(
                "scheduler must be cosine or constant_with_warmup"
            )
        if self.selection_metric not in {"macro_f1", "micro_f1", "mean_f1"}:
            raise ValueError("unsupported selection_metric")
        if self.checkpoint_role not in {"val_ctrl", "val_select"}:
            raise ValueError("checkpoint_role must be val_ctrl or val_select")
        if self.pseudo_strategy not in {"cap", "adaptive", "fixed"}:
            raise ValueError("pseudo_strategy must be cap, adaptive, or fixed")
        if self.supervised_loss not in {
            "bce", "weighted_bce", "asl", "two_way", "two_way_official",
            "distribution_balanced",
        }:
            raise ValueError(
                "supervised_loss must be bce, weighted_bce, asl, two_way, "
                "two_way_official, or distribution_balanced"
            )
        if self.asl_gamma_negative < 0.0 or self.asl_gamma_positive < 0.0:
            raise ValueError("ASL gamma values must be non-negative")
        if not 0.0 <= self.asl_negative_clip < 1.0:
            raise ValueError("asl_negative_clip must be in [0, 1)")
        db_values = {
            name: getattr(self, name)
            for name in _DISTRIBUTION_BALANCED_DEFAULTS
        }
        if self.supervised_loss != "distribution_balanced":
            changed = sorted(
                name for name, default in _DISTRIBUTION_BALANCED_DEFAULTS.items()
                if db_values[name] != default
            )
            if changed:
                raise ValueError(
                    "distribution-balanced parameters require "
                    "supervised_loss=distribution_balanced: %s"
                    % ", ".join(changed)
                )
        else:
            for name, value in db_values.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError("%s must be finite numeric" % name)
            if not 0.0 <= self.db_focal_alpha <= 1.0:
                raise ValueError("db_focal_alpha must be in [0, 1]")
            if self.db_focal_gamma < 0.0:
                raise ValueError("db_focal_gamma must be non-negative")
            if self.db_map_alpha < 0.0:
                raise ValueError("db_map_alpha must be non-negative")
            if self.db_map_beta <= 0.0:
                raise ValueError("db_map_beta must be positive")
            if self.db_init_bias < 0.0:
                raise ValueError("db_init_bias must be non-negative")
            if self.db_negative_scale <= 0.0:
                raise ValueError("db_negative_scale must be positive")
            if self.dataset == "rcv1" or self.model_name.startswith("hash_transformer"):
                raise ValueError(
                    "distribution-balanced loss is not wired to the RCV1 hash trainer"
                )
        if (
            not math.isfinite(self.two_way_temperature)
            or self.two_way_temperature <= 0.0
        ):
            raise ValueError("two_way_temperature must be finite and positive")
        if not math.isfinite(self.rdrop_weight) or self.rdrop_weight < 0.0:
            raise ValueError("rdrop_weight must be finite and non-negative")
        if self.rdrop_weight > 0.0:
            if self.method != "supervised":
                raise ValueError("R-Drop adapter is registered only for supervised runs")
            if self.dataset == "rcv1" or self.model_name.startswith("hash_transformer"):
                raise ValueError("R-Drop adapter is not wired to the RCV1 hash trainer")
        if not math.isfinite(self.denn_dcl_weight) or self.denn_dcl_weight < 0.0:
            raise ValueError("denn_dcl_weight must be finite and non-negative")
        if (
            not math.isfinite(self.denn_dcl_temperature)
            or self.denn_dcl_temperature <= 0.0
        ):
            raise ValueError("denn_dcl_temperature must be finite and positive")
        if self.batch_size < 1 or self.unlabeled_ratio < 1:
            raise ValueError("batch sizes must be positive")
        if self.head == "t5enc":
            if self.method != "supervised":
                raise ValueError("t5enc search runs must use method=supervised")
            if not self.skip_test:
                raise ValueError(
                    "t5enc search runs are validation-only and require skip_test=true"
                )
            if self.supervised_loss not in {"bce", "asl"}:
                raise ValueError(
                    "t5enc supports the faithful bce baseline or optional asl"
                )
            incompatible = {
                "prototype_weight": self.prototype_weight,
                "contrastive_weight": self.contrastive_weight,
                "consistency_weight": self.consistency_weight,
                "bbam_weight": self.bbam_weight,
                "denn_dcl_weight": self.denn_dcl_weight,
                "rdrop_weight": self.rdrop_weight,
            }
            enabled = sorted(
                name for name, value in incompatible.items()
                if not math.isclose(float(value), 0.0, abs_tol=0.0)
            )
            if enabled:
                raise ValueError(
                    "t5enc clean branch requires zero auxiliary weights: %s"
                    % ", ".join(enabled)
                )
        elif self.t5enc_snapshot_sha256:
            raise ValueError(
                "t5enc_snapshot_sha256 is only valid when head=t5enc"
            )
        if self.encoder_snapshot_manifest:
            if self.head == "t5enc":
                raise ValueError(
                    "encoder_snapshot_manifest is only for encoder-only heads"
                )
            if not Path(self.encoder_snapshot_manifest).expanduser().is_absolute():
                raise ValueError("encoder_snapshot_manifest must be an absolute path")
            if not Path(self.model_name).expanduser().is_absolute():
                raise ValueError(
                    "encoder_snapshot_manifest requires an absolute local model_name"
                )
            if self.protocol != "loris_exact" or not self.skip_test:
                raise ValueError(
                    "bound encoder snapshot search requires loris_exact and skip_test=true"
                )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # This opt-in field was added after the original exact/seed plans were
        # frozen.  Omitting its empty default preserves their byte-level key
        # sets and prevents fresh replay audits from seeing a spurious config
        # change.  A bound TAPT classifier persists the non-empty value.
        if not self.encoder_snapshot_manifest:
            payload.pop("encoder_snapshot_manifest", None)
        if self.rdrop_weight == 0.0:
            payload.pop("rdrop_weight", None)
        if self.supervised_loss != "distribution_balanced":
            for name in _DISTRIBUTION_BALANCED_DEFAULTS:
                payload.pop(name, None)
        return payload
