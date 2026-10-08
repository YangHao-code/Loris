"""Strong Transformer backend for Loris's injectable auxiliary annotator.

The module intentionally has no dependency on the Loris package.  Loris's
``AuxiliaryAnnotator`` accepts a ``backend_factory(config, num_labels)`` and
then calls only ``fit``, ``predict_proba`` and ``save_checkpoint``.  The
factory returned by :func:`make_chase_backend_factory` implements exactly that
boundary while keeping this project's model and tokenisation stack reusable.

Only train and optional validation supervision enter the API.  There is no
target/test argument or loader in this module.  Pair-wise unknown labels are a
hard correctness boundary: their weights must already be exactly zero, and
they therefore cannot silently become negative training examples.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data import TextBatchCollator, TextDataset
from .model import MultiLabelEncoder, ema_update


BACKEND_NAME = "loris_ssl_chase_transformer"
BACKEND_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# These are the complete label-axis tensors of ``MultiLabelEncoder``.  A
# checkpoint label permutation must be applied to every one of them while the
# encoder and shared head tensors are copied byte-for-byte.  Custom test or
# integration models can add names through a ``label_axis_state_names``
# attribute; ambiguous, undeclared label-shaped tensors fail closed below.
_MULTILABEL_ENCODER_LABEL_AXIS_STATE_NAMES = frozenset({
    "label_queries",
    "classifier_weight",
    "classifier_bias",
    "binary_prototypes",
})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ChaseBackendConfig:
    """Frozen architecture and optimiser settings for one chase round.

    The defaults mirror the strongest single BGC arm available in this
    project.  Smaller encoders, pooled heads, shorter runs, or CPU settings are
    explicit configuration changes and are recorded in every backend audit.
    """

    model_name: str = "microsoft/deberta-v3-large"
    cache_dir: str = ""
    offline: bool = True
    head: str = "label_attention_linear"
    max_length: int = 384
    truncation: str = "head_tail"
    dropout: float = 0.1
    projection_dim: int = 128
    gradient_checkpointing: bool = True

    loss: str = "asl"  # asl | masked_weighted_bce
    asl_gamma_negative: float = 4.0
    asl_gamma_positive: float = 0.0
    asl_negative_clip: float = 0.05

    epochs: int = 16
    min_epochs: int = 10
    patience: int = 5
    batch_size: int = 2
    prediction_batch_size: int = 8
    # Explicit inference-only shortcut; training forward and RNG are unchanged.
    prediction_logits_only: bool = False
    # None preserves historical training/validation batching and fingerprints.
    # Validation has no backward pass and can use a larger batch independently.
    validation_batch_size: Optional[int] = None
    grad_accum_steps: int = 8
    learning_rate: float = 1.0e-5
    head_learning_rate: float = 5.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    ema_decay: float = 0.995
    # Student and EMA are both evaluated when validation is available; this
    # flag breaks exact ties toward EMA and selects EMA when validation is not.
    prefer_ema: bool = True
    amp: str = "bf16"  # bf16 | fp16 | none; fp16 remains an explicit compatibility choice
    num_workers: int = 0
    seed: int = 0
    initialize_class_priors: bool = True

    # Optional, immutable supervised-M0 bootstrap.  Supplying a path switches
    # each independently constructed chase backend from raw HF initialisation
    # to this exact model-only checkpoint.  All four fields are required as a
    # unit so label-axis remapping is explicit and auditable rather than
    # inferred from ambient Loris state.
    initial_checkpoint_path: str = ""
    initial_checkpoint_sha256: str = ""
    initial_checkpoint_source_labels: Tuple[str, ...] = ()
    initial_checkpoint_target_labels: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.prediction_logits_only) is not bool:
            raise ValueError("prediction_logits_only must be an explicit bool")
        # JSON checkpoint metadata naturally reconstructs tuples as lists.
        # Canonicalise them here so an in-memory and restored config have the
        # same public type and fingerprint.
        if isinstance(self.initial_checkpoint_source_labels, (str, bytes)) or isinstance(
            self.initial_checkpoint_target_labels, (str, bytes)
        ):
            raise ValueError("initial checkpoint label orders must be sequences")
        try:
            source_labels = tuple(
                str(value) for value in self.initial_checkpoint_source_labels
            )
            target_labels = tuple(
                str(value) for value in self.initial_checkpoint_target_labels
            )
        except TypeError as exc:
            raise ValueError("initial checkpoint label orders must be sequences") from exc
        object.__setattr__(self, "initial_checkpoint_source_labels", source_labels)
        object.__setattr__(self, "initial_checkpoint_target_labels", target_labels)
        object.__setattr__(
            self,
            "initial_checkpoint_sha256",
            str(self.initial_checkpoint_sha256).strip().lower(),
        )
        object.__setattr__(
            self,
            "initial_checkpoint_path",
            str(self.initial_checkpoint_path).strip(),
        )
        if not str(self.model_name).strip():
            raise ValueError("model_name must be non-empty")
        if self.head not in {
            "label_attention", "label_attention_linear", "pooled", "pooled_linear",
        }:
            raise ValueError(
                "head must be label_attention, label_attention_linear, pooled, "
                "or pooled_linear"
            )
        if self.truncation not in {"head", "head_tail"}:
            raise ValueError("truncation must be head or head_tail")
        if self.loss not in {"asl", "masked_weighted_bce"}:
            raise ValueError("loss must be asl or masked_weighted_bce")
        if self.amp not in {"fp16", "bf16", "none"}:
            raise ValueError("amp must be fp16, bf16, or none")
        for name in (
            "max_length", "projection_dim", "epochs", "min_epochs", "batch_size",
            "prediction_batch_size", "grad_accum_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError("%s must be positive" % name)
        if self.min_epochs > self.epochs:
            raise ValueError("min_epochs cannot exceed epochs")
        if self.validation_batch_size is not None and (
            type(self.validation_batch_size) is not int or self.validation_batch_size <= 0
        ):
            raise ValueError("validation_batch_size must be a positive integer or None")
        if isinstance(self.patience, bool) or int(self.patience) < 0:
            raise ValueError("patience must be non-negative")
        if isinstance(self.num_workers, bool) or int(self.num_workers) < 0:
            raise ValueError("num_workers must be non-negative")
        for name in (
            "dropout", "asl_gamma_negative", "asl_gamma_positive",
            "asl_negative_clip", "learning_rate", "head_learning_rate",
            "weight_decay", "warmup_ratio", "max_grad_norm", "ema_decay",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError("%s must be finite" % name)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.asl_gamma_negative < 0 or self.asl_gamma_positive < 0:
            raise ValueError("ASL focusing parameters must be non-negative")
        if not 0.0 <= self.asl_negative_clip < 1.0:
            raise ValueError("asl_negative_clip must lie in [0, 1)")
        if self.learning_rate <= 0 or self.head_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must lie in [0, 1)")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1)")

        checkpoint_fields_present = (
            bool(self.initial_checkpoint_path),
            bool(self.initial_checkpoint_sha256),
            bool(source_labels),
            bool(target_labels),
        )
        if any(checkpoint_fields_present) and not all(checkpoint_fields_present):
            raise ValueError(
                "initial checkpoint path, sha256, source labels, and target labels "
                "must be supplied together"
            )
        if all(checkpoint_fields_present):
            if not Path(self.initial_checkpoint_path).is_absolute():
                raise ValueError("initial_checkpoint_path must be absolute")
            if _SHA256_RE.fullmatch(self.initial_checkpoint_sha256) is None:
                raise ValueError("initial_checkpoint_sha256 must be 64 lowercase hex digits")
            for role, labels in (
                ("source", source_labels), ("target", target_labels),
            ):
                if any(not label.strip() for label in labels):
                    raise ValueError("initial checkpoint %s labels must be non-empty" % role)
                if len(set(labels)) != len(labels):
                    raise ValueError("initial checkpoint %s labels must be unique" % role)
            if (
                len(source_labels) != len(target_labels)
                or set(source_labels) != set(target_labels)
            ):
                raise ValueError(
                    "initial checkpoint source and target labels must be a permutation"
                )

    @classmethod
    def from_auxiliary_config(cls, config: Any) -> "ChaseBackendConfig":
        """Bridge Loris's narrower config without importing its class.

        Fields absent from the caller retain this backend's strong defaults.
        Loris currently exposes ``use_amp`` rather than an AMP dtype and has no
        gradient-accumulation field, so those two cases are mapped explicitly.
        """

        defaults = cls()
        values = asdict(defaults)
        shared = (
            "model_name", "max_length", "batch_size", "prediction_batch_size", "validation_batch_size",
            "epochs", "learning_rate", "weight_decay", "warmup_ratio", "patience",
            "max_grad_norm", "seed", "gradient_checkpointing",
        )
        for name in shared:
            if hasattr(config, name):
                values[name] = getattr(config, name)
        if not hasattr(config, "min_epochs"):
            values["min_epochs"] = min(
                int(values["min_epochs"]), int(values["epochs"]),
            )
        else:
            values["min_epochs"] = getattr(config, "min_epochs")
        if hasattr(config, "use_amp"):
            values["amp"] = "bf16" if bool(getattr(config, "use_amp")) else "none"
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        # The opt-in inference feature must not change the serialized schema
        # of historical default configurations or their saved checkpoints.
        if not self.prediction_logits_only:
            values.pop("prediction_logits_only")
        return values

    @property
    def fingerprint(self) -> str:
        values = self.to_dict()
        if not self.prediction_logits_only:
            values.pop("prediction_logits_only", None)
        if self.validation_batch_size is None:
            values.pop("validation_batch_size", None)
        # Preserve fingerprints of pre-bootstrap (schema-v1) checkpoints when
        # the new feature is disabled.  This keeps old inference checkpoints
        # restorable while binding all bootstrap lineage when it is enabled.
        if not self.initial_checkpoint_path:
            for name in (
                "initial_checkpoint_path",
                "initial_checkpoint_sha256",
                "initial_checkpoint_source_labels",
                "initial_checkpoint_target_labels",
            ):
                values.pop(name, None)
        payload = json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class _SupervisionView:
    targets: np.ndarray
    observed: np.ndarray
    weights: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        return self.targets.shape

    def summary(self) -> Dict[str, Any]:
        positive = self.observed & (self.targets == 1)
        negative = self.observed & (self.targets == 0)
        return {
            "documents": int(self.shape[0]),
            "labels": int(self.shape[1]),
            "observed_positive_pairs": int(positive.sum()),
            "observed_negative_pairs": int(negative.sum()),
            "unknown_pairs": int((~self.observed).sum()),
            "positive_weight_sum": float(self.weights[positive].sum()),
            "negative_weight_sum": float(self.weights[negative].sum()),
        }


def _coerce_supervision(
    supervision: Any,
    *,
    role: str,
    expected_rows: Optional[int] = None,
    expected_labels: Optional[int] = None,
) -> _SupervisionView:
    """Validate a duck-typed ``targets/observed/weights`` object.

    Unknown targets may contain arbitrary placeholders, but unknown weights
    are rejected unless they are exactly zero.  Rejecting instead of silently
    masking catches upstream supervision bugs at the model boundary.
    """

    missing = [
        name for name in ("targets", "observed", "weights")
        if not hasattr(supervision, name)
    ]
    if missing:
        raise TypeError(
            "%s supervision is missing %s" % (role, ", ".join(missing))
        )
    raw_targets = np.asarray(getattr(supervision, "targets"))
    observed = np.asarray(getattr(supervision, "observed"), dtype=bool)
    try:
        weights = np.asarray(getattr(supervision, "weights"), dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s supervision weights must be numeric" % role) from exc
    if raw_targets.ndim != 2:
        raise ValueError("%s supervision targets must be two-dimensional" % role)
    if observed.shape != raw_targets.shape or weights.shape != raw_targets.shape:
        raise ValueError(
            "%s targets, observed, and weights must have identical shape" % role
        )
    if expected_rows is not None and raw_targets.shape[0] != int(expected_rows):
        raise ValueError("%s text/supervision row mismatch" % role)
    if expected_labels is not None and raw_targets.shape[1] != int(expected_labels):
        raise ValueError("%s supervision label width mismatch" % role)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("%s supervision weights must be finite and non-negative" % role)
    if np.any(weights[~observed] != 0):
        raise ValueError("%s unknown supervision weights must be zero" % role)
    if np.any(weights[observed] <= 0):
        raise ValueError("%s observed supervision weights must be positive" % role)

    try:
        observed_targets = np.asarray(raw_targets[observed], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s observed targets must be numeric" % role) from exc
    if not np.all(np.isfinite(observed_targets)):
        raise ValueError("%s observed targets must be finite" % role)
    if observed_targets.size and not np.all(
        (observed_targets == 0) | (observed_targets == 1)
    ):
        raise ValueError("%s observed targets must be binary" % role)
    if not np.any(observed):
        raise ValueError("%s supervision contains no observed pair" % role)

    # Do not attempt to cast arbitrary unknown placeholders.  Fill only the
    # observed cells, then freeze private copies for the duration of training.
    targets = np.zeros(raw_targets.shape, dtype=np.float32)
    targets[observed] = observed_targets
    arrays = (
        targets,
        np.array(observed, dtype=bool, copy=True),
        np.array(weights, dtype=np.float32, copy=True),
    )
    for array in arrays:
        array.setflags(write=False)
    return _SupervisionView(*arrays)


def _asl_elements(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_negative: float,
    gamma_positive: float,
    negative_clip: float,
) -> torch.Tensor:
    # Match loris_ssl.losses.asymmetric_loss, but retain its element-wise form
    # so pair confidence and unknown masks can be applied before reduction.
    working_logits = logits.float()
    working_targets = targets.float()
    positive = torch.sigmoid(working_logits)
    negative = 1.0 - positive
    if negative_clip > 0:
        negative = (negative + float(negative_clip)).clamp(max=1.0)
    log_loss = working_targets * torch.log(positive.clamp_min(1.0e-8))
    log_loss = log_loss + (1.0 - working_targets) * torch.log(
        negative.clamp_min(1.0e-8)
    )
    true_probability = (
        positive * working_targets + negative * (1.0 - working_targets)
    )
    exponent = (
        float(gamma_positive) * working_targets
        + float(gamma_negative) * (1.0 - working_targets)
    )
    return -log_loss * torch.pow((1.0 - true_probability).clamp_min(0.0), exponent)


def _weighted_objective_parts(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    loss_kind: str,
    asl_gamma_negative: float,
    asl_gamma_positive: float,
    asl_negative_clip: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or targets.shape != logits.shape or weights.shape != logits.shape:
        raise ValueError("logits, targets, and weights must share shape [batch, labels]")
    working_targets = targets.to(device=logits.device, dtype=torch.float32)
    working_weights = weights.to(device=logits.device, dtype=torch.float32)
    active = working_weights > 0
    # Unknown logits can be arbitrary too (including non-finite values from a
    # custom model).  Replace them *before* BCE/ASL so their forward values and
    # backward derivatives are both guaranteed to be zero-masked.
    working_logits = torch.where(
        active,
        logits.float(),
        torch.zeros_like(logits, dtype=torch.float32),
    )
    if loss_kind == "asl":
        elements = _asl_elements(
            working_logits, working_targets, asl_gamma_negative, asl_gamma_positive,
            asl_negative_clip,
        )
    elif loss_kind == "masked_weighted_bce":
        elements = F.binary_cross_entropy_with_logits(
            working_logits, working_targets, reduction="none",
        )
    else:  # protected by config validation; useful for public primitive callers
        raise ValueError("loss_kind must be asl or masked_weighted_bce")
    denominator = working_weights.sum()
    if not bool((denominator > 0).item()):
        raise ValueError("loss batch contains no observed positive weight")
    weighted_elements = torch.where(
        working_weights > 0,
        elements * working_weights,
        torch.zeros_like(elements),
    )
    numerator = weighted_elements.sum()
    return numerator / denominator, numerator, denominator.detach()


def masked_weighted_loss(
    logits: torch.Tensor,
    supervision: Any,
    *,
    loss_kind: str = "masked_weighted_bce",
    asl_gamma_negative: float = 4.0,
    asl_gamma_positive: float = 0.0,
    asl_negative_clip: float = 0.05,
) -> torch.Tensor:
    """Public, independently testable partial-label objective.

    ``supervision`` is deliberately duck typed.  Unknown cells must carry zero
    weight and consequently contribute neither a loss value nor a gradient.
    """

    view = _coerce_supervision(
        supervision,
        role="objective",
        expected_rows=int(logits.shape[0]) if logits.ndim >= 1 else None,
        expected_labels=int(logits.shape[1]) if logits.ndim == 2 else None,
    )
    targets = torch.from_numpy(np.array(view.targets, copy=True)).to(logits.device)
    weights = torch.from_numpy(np.array(view.weights, copy=True)).to(logits.device)
    loss, _, _ = _weighted_objective_parts(
        logits,
        targets,
        weights,
        loss_kind=loss_kind,
        asl_gamma_negative=asl_gamma_negative,
        asl_gamma_positive=asl_gamma_positive,
        asl_negative_clip=asl_negative_clip,
    )
    return loss


class _WeightedTextDataset(TextDataset):
    def __init__(
        self,
        texts: Sequence[str],
        targets: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        super().__init__(texts, targets)
        self.loss_weights = np.asarray(weights, dtype=np.float32)
        if self.loss_weights.shape != self.labels.shape:
            raise ValueError("loss weights and labels have different shapes")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = super().__getitem__(index)
        item["loss_weights"] = self.loss_weights[index]
        return item


class _WeightedCollator:
    def __init__(self, text_collator: TextBatchCollator) -> None:
        self.text_collator = text_collator

    def __call__(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.text_collator(examples)
        weights = np.stack([example["loss_weights"] for example in examples])
        batch["loss_weights"] = torch.from_numpy(weights).float()
        return batch


TokenizerFactory = Callable[[ChaseBackendConfig], Any]
ModelFactory = Callable[[ChaseBackendConfig, int], nn.Module]


def _default_tokenizer_factory(config: ChaseBackendConfig) -> Any:
    kwargs: Dict[str, Any] = {
        "use_fast": True,
        "local_files_only": bool(config.offline),
    }
    if config.cache_dir:
        kwargs["cache_dir"] = config.cache_dir
    return AutoTokenizer.from_pretrained(config.model_name, **kwargs)


def _default_model_factory(config: ChaseBackendConfig, num_labels: int) -> nn.Module:
    return MultiLabelEncoder(
        config.model_name,
        num_labels,
        head=config.head,
        dropout=config.dropout,
        projection_dim=config.projection_dim,
        gradient_checkpointing=config.gradient_checkpointing,
        offline=config.offline,
        cache_dir=config.cache_dir,
    )


def _seed_fresh_initialization(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _head_state_sha256(model: nn.Module) -> str:
    """Hash freshly initialised non-backbone state without copying the encoder."""

    digest = hashlib.sha256()
    found = False
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith("encoder.") or name.startswith("backbone."):
            continue
        found = True
        value = tensor.detach().cpu().contiguous()
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "little"))
        digest.update(encoded_name)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    if not found:
        # A custom test/integration model may not distinguish its backbone.
        # Bind that fact rather than pretending a full encoder hash was made.
        digest.update(b"no-non-backbone-state")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _label_axis_state_names(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    num_labels: int,
) -> Tuple[str, ...]:
    """Return explicitly declared label-axis state, rejecting ambiguity.

    ``MultiLabelEncoder`` keeps its label parameters at the module root.  A
    custom compatible model may expose an iterable ``label_axis_state_names``
    to add exact state-dict keys.  We deliberately do not guess and permute an
    arbitrary tensor merely because one dimension happens to equal the label
    count: that could silently scramble a shared feature axis.
    """

    declared = {
        name for name in _MULTILABEL_ENCODER_LABEL_AXIS_STATE_NAMES
        if name in state
    }
    custom = getattr(model, "label_axis_state_names", ())
    if custom is None:
        custom = ()
    if isinstance(custom, str):
        raise RuntimeError("label_axis_state_names must be an iterable of exact keys")
    try:
        declared.update(str(name) for name in custom)
    except TypeError as exc:
        raise RuntimeError(
            "label_axis_state_names must be an iterable of exact keys"
        ) from exc
    missing = sorted(name for name in declared if name not in state)
    if missing:
        raise RuntimeError(
            "declared label-axis checkpoint keys are absent from model state: %s"
            % ", ".join(missing)
        )
    for name in sorted(declared):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1:
            raise RuntimeError("label-axis state %s must be a tensor with rank >= 1" % name)
        if int(tensor.shape[0]) != int(num_labels):
            raise RuntimeError(
                "label-axis state %s has width %d, expected %d"
                % (name, int(tensor.shape[0]), int(num_labels))
            )

    # The projection module is explicitly shared across labels even when its
    # configured feature dimension happens to equal num_labels.  Encoder and
    # backbone parameters are likewise never label-axis state.  Any other
    # undeclared label-shaped head tensor is ambiguous and therefore rejected.
    shared_prefixes = ("encoder.", "backbone.", "projection.")
    ambiguous = []
    for name, tensor in state.items():
        if name in declared or name.startswith(shared_prefixes):
            continue
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
            if int(tensor.shape[0]) == int(num_labels):
                ambiguous.append(name)
    if ambiguous:
        raise RuntimeError(
            "undeclared label-shaped checkpoint state; declare its exact label axis: %s"
            % ", ".join(sorted(ambiguous))
        )
    return tuple(sorted(declared))


def _load_supervised_checkpoint_initialization(
    model: nn.Module,
    config: ChaseBackendConfig,
    num_labels: int,
) -> Dict[str, Any]:
    """Load and label-remap one immutable model-only M0 checkpoint.

    The source checkpoint is authenticated before deserialisation.  Only
    explicitly declared label-axis state is indexed by ``target_from_source``;
    encoder and shared head tensors retain the exact checkpoint values.
    ``load_state_dict(strict=True)`` and explicit metadata/shape checks make
    every architecture or schema mismatch fail closed.
    """

    if not config.initial_checkpoint_path:
        raise ValueError("no supervised initial checkpoint is configured")
    source_path = Path(config.initial_checkpoint_path)
    try:
        resolved_path = source_path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("initial supervised checkpoint does not exist") from exc
    if not resolved_path.is_file():
        raise RuntimeError("initial supervised checkpoint is not a regular file")
    actual_sha256 = _file_sha256(resolved_path)
    if actual_sha256 != config.initial_checkpoint_sha256:
        raise RuntimeError("initial supervised checkpoint sha256 mismatch")
    try:
        payload = torch.load(str(resolved_path), map_location="cpu")
    except Exception as exc:
        raise RuntimeError("failed to deserialize initial supervised checkpoint") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("initial supervised checkpoint payload must be a mapping")
    required = {"state_dict", "label_names", "model_name", "head"}
    missing_payload = sorted(required.difference(payload))
    if missing_payload:
        raise RuntimeError(
            "initial supervised checkpoint is missing: %s"
            % ", ".join(missing_payload)
        )
    if str(payload["model_name"]) != config.model_name:
        raise RuntimeError("initial supervised checkpoint model_name mismatch")
    if str(payload["head"]) != config.head:
        raise RuntimeError("initial supervised checkpoint head mismatch")

    source_labels = config.initial_checkpoint_source_labels
    target_labels = config.initial_checkpoint_target_labels
    embedded_labels_raw = payload["label_names"]
    if isinstance(embedded_labels_raw, (str, bytes)):
        raise RuntimeError("initial supervised checkpoint label_names is invalid")
    try:
        embedded_labels = tuple(str(label) for label in embedded_labels_raw)
    except TypeError as exc:
        raise RuntimeError(
            "initial supervised checkpoint label_names is invalid"
        ) from exc
    if embedded_labels != source_labels:
        raise RuntimeError("initial supervised checkpoint source label order mismatch")
    if len(target_labels) != int(num_labels):
        raise RuntimeError("initial supervised checkpoint target label width mismatch")

    model_num_labels = getattr(model, "n_labels", None)
    if model_num_labels is not None and int(model_num_labels) != int(num_labels):
        raise RuntimeError("initial supervised checkpoint model label width mismatch")
    model_name = getattr(model, "model_name", None)
    if model_name is not None and str(model_name) != config.model_name:
        raise RuntimeError("model_factory model_name does not match initial checkpoint")
    model_head_type = getattr(model, "head_type", None)
    if model_head_type is not None:
        expected_head_type = config.head.replace("_linear", "")
        if str(model_head_type) != expected_head_type:
            raise RuntimeError("model_factory head type does not match initial checkpoint")
    linear_classifier = getattr(model, "linear_classifier", None)
    if linear_classifier is not None:
        if bool(linear_classifier) != config.head.endswith("_linear"):
            raise RuntimeError(
                "model_factory classifier mode does not match initial checkpoint"
            )

    checkpoint_state = payload["state_dict"]
    if not isinstance(checkpoint_state, Mapping):
        raise RuntimeError("initial supervised checkpoint state_dict must be a mapping")
    if not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in checkpoint_state.items()
    ):
        raise RuntimeError("initial supervised checkpoint state_dict is invalid")
    target_state = model.state_dict()
    checkpoint_keys = set(checkpoint_state)
    target_keys = set(target_state)
    if checkpoint_keys != target_keys:
        missing = sorted(target_keys.difference(checkpoint_keys))
        unexpected = sorted(checkpoint_keys.difference(target_keys))
        raise RuntimeError(
            "initial supervised checkpoint state keys mismatch "
            "(missing=%s, unexpected=%s)"
            % (missing, unexpected)
        )
    label_axis_names = _label_axis_state_names(model, target_state, num_labels)
    source_position = {label: index for index, label in enumerate(source_labels)}
    target_from_source = tuple(source_position[label] for label in target_labels)
    permutation = torch.tensor(target_from_source, dtype=torch.long)

    remapped_state: Dict[str, torch.Tensor] = {}
    for name, target_tensor in target_state.items():
        source_tensor = checkpoint_state[name]
        if source_tensor.dtype != target_tensor.dtype:
            raise RuntimeError(
                "initial supervised checkpoint dtype mismatch for %s" % name
            )
        if name in label_axis_names:
            if source_tensor.ndim < 1 or int(source_tensor.shape[0]) != len(source_labels):
                raise RuntimeError(
                    "initial supervised checkpoint label-axis shape mismatch for %s"
                    % name
                )
            expected_shape = (int(num_labels),) + tuple(source_tensor.shape[1:])
            if tuple(target_tensor.shape) != expected_shape:
                raise RuntimeError(
                    "model_factory label-axis shape mismatch for %s" % name
                )
            remapped_state[name] = source_tensor.index_select(0, permutation)
        else:
            if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                raise RuntimeError(
                    "initial supervised checkpoint shared-state shape mismatch for %s"
                    % name
                )
            remapped_state[name] = source_tensor
    try:
        model.load_state_dict(remapped_state, strict=True)
    except Exception as exc:
        raise RuntimeError(
            "failed to load initial supervised checkpoint state"
        ) from exc

    encoder_names = tuple(sorted(
        name for name in target_state
        if name.startswith("encoder.") or name.startswith("backbone.")
    ))
    return {
        "kind": "frozen_model_only_supervised_m0",
        "checkpoint_path": str(resolved_path),
        "checkpoint_sha256": actual_sha256,
        "checkpoint_model_name": str(payload["model_name"]),
        "checkpoint_head": str(payload["head"]),
        "source_label_order": list(source_labels),
        "target_label_order": list(target_labels),
        "target_from_source": list(target_from_source),
        "label_axis_state_keys": list(label_axis_names),
        "encoder_state_keys": len(encoder_names),
        "encoder_loaded_without_permutation": True,
        "shared_state_loaded_without_permutation": True,
        "threshold_policy_loaded": False,
        "previous_chase_round_state_loaded": False,
    }


def _supervision_fingerprint(
    texts: Sequence[str],
    supervision: _SupervisionView,
    validation_texts: Optional[Sequence[str]],
    validation_supervision: Optional[_SupervisionView],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"loris_ssl/chase_backend/supervision/v1\0")

    def update(role: bytes, role_texts: Sequence[str], view: _SupervisionView) -> None:
        digest.update(role)
        digest.update(np.asarray(view.shape, dtype="<i8").tobytes())
        for text in role_texts:
            payload = str(text).encode("utf-8")
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
        digest.update(view.observed.astype(np.uint8).tobytes(order="C"))
        digest.update(view.targets.astype("<f4", copy=False).tobytes(order="C"))
        digest.update(view.weights.astype("<f4", copy=False).tobytes(order="C"))

    update(b"train\0", texts, supervision)
    if validation_texts is not None and validation_supervision is not None:
        update(b"validation\0", validation_texts, validation_supervision)
    return digest.hexdigest()


def _model_inputs(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=(device.type == "cuda"))
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }


def _copy_state_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


class ChaseTransformerBackend:
    """Independent MultiLabelEncoder training backend for one chase round.

    Every instance is single-use.  Its initial state is either a fresh HF
    pretrained encoder/head (legacy behaviour) or the same immutable,
    authenticated supervised-M0 checkpoint configured for every round.  It
    never consumes the preceding chase round's fitted state.
    """

    backend_name = BACKEND_NAME

    def __init__(
        self,
        config: ChaseBackendConfig,
        num_labels: int,
        *,
        tokenizer_factory: Optional[TokenizerFactory] = None,
        model_factory: Optional[ModelFactory] = None,
        device: Optional[str] = None,
    ) -> None:
        if int(num_labels) <= 0:
            raise ValueError("num_labels must be positive")
        self.config = config
        self.num_labels = int(num_labels)
        self._tokenizer_factory = tokenizer_factory or _default_tokenizer_factory
        self._model_factory = model_factory or _default_model_factory
        self._requested_device = device
        self._device: Optional[torch.device] = None
        self._tokenizer: Any = None
        self._model: Optional[nn.Module] = None
        self._started = False
        self._fitted = False
        self._history: List[Dict[str, Any]] = []
        self._fit_audit: Optional[Dict[str, Any]] = None

    @property
    def history(self) -> List[Dict[str, Any]]:
        return json.loads(json.dumps(self._history))

    @property
    def fit_audit(self) -> Dict[str, Any]:
        if self._fit_audit is None:
            raise RuntimeError("backend has not been fitted")
        return json.loads(json.dumps(self._fit_audit))

    def _resolve_device(self) -> torch.device:
        if self._requested_device is None or self._requested_device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self._requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "cuda" and self.config.amp == "bf16":
            capability = torch.cuda.get_device_capability(device)
            if int(capability[0]) < 8:
                raise RuntimeError(
                    "bf16 AMP requires an Ampere-or-newer CUDA device"
                )
        return device

    def _autocast(self):
        if self._device is None or self._device.type != "cuda" or self.config.amp == "none":
            return nullcontext()
        dtype = torch.float16 if self.config.amp == "fp16" else torch.bfloat16
        return torch.cuda.amp.autocast(dtype=dtype)

    def _loader(
        self,
        texts: Sequence[str],
        supervision: Optional[_SupervisionView],
        *,
        shuffle: bool,
        prediction: bool = False,
        validation: bool = False,
        seed_offset: int = 0,
    ) -> DataLoader:
        if self._tokenizer is None:
            raise RuntimeError("internal error: tokenizer is not initialised")
        if prediction and validation:
            raise ValueError("a loader cannot be both prediction and validation")
        text_collator = TextBatchCollator(
            self._tokenizer, self.config.max_length, self.config.truncation,
        )
        if supervision is None:
            dataset = TextDataset(texts)
            collator: Any = text_collator
        else:
            active = np.flatnonzero(supervision.observed.any(axis=1))
            if active.size == 0:
                raise ValueError("supervision contains no observed row")
            active_texts = [str(texts[int(index)]) for index in active]
            dataset = _WeightedTextDataset(
                active_texts,
                supervision.targets[active],
                supervision.weights[active],
            )
            collator = _WeightedCollator(text_collator)
        generator = torch.Generator()
        generator.manual_seed(int(self.config.seed) + int(seed_offset))
        return DataLoader(
            dataset,
            batch_size=(
                self.config.prediction_batch_size if prediction
                else (self.config.validation_batch_size or self.config.batch_size) if validation
                else self.config.batch_size
            ),
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collator,
            pin_memory=self._device is not None and self._device.type == "cuda",
            generator=generator,
            persistent_workers=self.config.num_workers > 0,
            drop_last=False,
        )

    def _optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        backbone = getattr(model, "encoder", None)
        backbone_ids = (
            {id(parameter) for parameter in backbone.parameters()}
            if backbone is not None else set()
        )
        groups: Dict[str, List[nn.Parameter]] = {
            "backbone_decay": [],
            "backbone_nodecay": [],
            "head_decay": [],
            "head_nodecay": [],
        }
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            prefix = "backbone" if id(parameter) in backbone_ids else "head"
            no_decay = (
                parameter.ndim < 2 or name.endswith("bias") or "norm" in name.lower()
            )
            groups[prefix + ("_nodecay" if no_decay else "_decay")].append(parameter)
        parameter_groups: List[Dict[str, Any]] = []
        for name, parameters in groups.items():
            if not parameters:
                continue
            parameter_groups.append({
                "params": parameters,
                "lr": (
                    self.config.learning_rate
                    if name.startswith("backbone")
                    else self.config.head_learning_rate
                ),
                "weight_decay": 0.0 if name.endswith("nodecay") else self.config.weight_decay,
            })
        if not parameter_groups:
            raise ValueError("model has no trainable parameters")
        return torch.optim.AdamW(parameter_groups)

    def _scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        total_updates: int,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        warmup_updates = int(round(total_updates * self.config.warmup_ratio))

        def factor(step: int) -> float:
            if warmup_updates > 0 and step < warmup_updates:
                return float(step + 1) / float(warmup_updates)
            progress = float(step - warmup_updates) / float(
                max(1, total_updates - warmup_updates)
            )
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    def _forward_logits(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if self._device is None:
            raise RuntimeError("internal error: device is not initialised")
        output = model(**_model_inputs(batch, self._device))
        if not isinstance(output, Mapping) or "logits" not in output:
            raise RuntimeError("MultiLabelEncoder-compatible model must return logits")
        logits = output["logits"]
        if logits.ndim != 2 or logits.shape[1] != self.num_labels:
            raise RuntimeError("model returned an invalid logit shape")
        return logits

    def _loss_parts(
        self,
        logits: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._device is None:
            raise RuntimeError("internal error: device is not initialised")
        return _weighted_objective_parts(
            logits,
            batch["labels"].to(self._device, non_blocking=self._device.type == "cuda"),
            batch["loss_weights"].to(
                self._device, non_blocking=self._device.type == "cuda"
            ),
            loss_kind=self.config.loss,
            asl_gamma_negative=self.config.asl_gamma_negative,
            asl_gamma_positive=self.config.asl_gamma_positive,
            asl_negative_clip=self.config.asl_negative_clip,
        )

    @torch.no_grad()
    def _validation_loss(self, model: nn.Module, loader: DataLoader) -> float:
        model.eval()
        numerator = 0.0
        denominator = 0.0
        for batch in loader:
            with self._autocast():
                logits = self._forward_logits(model, batch)
                _, batch_numerator, batch_denominator = self._loss_parts(logits, batch)
            numerator += float(batch_numerator.cpu().item())
            denominator += float(batch_denominator.cpu().item())
        if denominator <= 0:
            raise ValueError("validation supervision contains no positive weight")
        return numerator / denominator

    def fit(
        self,
        texts: Sequence[str],
        supervision: Any,
        validation_texts: Optional[Sequence[str]],
        validation_supervision: Optional[Any],
    ) -> Mapping[str, Any]:
        """Train one independently initialised model on train/validation only."""

        if self._started:
            raise RuntimeError(
                "ChaseTransformerBackend is single-use; create a fresh backend "
                "for every chase round"
            )
        self._started = True
        train_texts = [str(text) for text in texts]
        train_view = _coerce_supervision(
            supervision,
            role="training",
            expected_rows=len(train_texts),
            expected_labels=self.num_labels,
        )
        if (validation_texts is None) != (validation_supervision is None):
            raise ValueError(
                "validation_texts and validation_supervision must be supplied together"
            )
        validation_text_list: Optional[List[str]] = None
        validation_view: Optional[_SupervisionView] = None
        if validation_texts is not None and validation_supervision is not None:
            validation_text_list = [str(text) for text in validation_texts]
            validation_view = _coerce_supervision(
                validation_supervision,
                role="validation",
                expected_rows=len(validation_text_list),
                expected_labels=self.num_labels,
            )

        _seed_fresh_initialization(self.config.seed)
        self._device = self._resolve_device()
        self._tokenizer = self._tokenizer_factory(self.config)
        student = self._model_factory(self.config, self.num_labels)
        if not isinstance(student, nn.Module):
            raise TypeError("model_factory must return torch.nn.Module")
        fresh_random_head_sha256 = _head_state_sha256(student)

        initialization_lineage: Optional[Dict[str, Any]] = None
        if self.config.initial_checkpoint_path:
            initialization_lineage = _load_supervised_checkpoint_initialization(
                student, self.config, self.num_labels,
            )

        prior_rates: Optional[List[float]] = None
        class_prior_initialization_applied = False
        if (
            initialization_lineage is None
            and self.config.initialize_class_priors
            and hasattr(
                student, "initialize_class_priors"
            )
        ):
            positive_mass = (train_view.weights * train_view.targets).sum(axis=0)
            negative_mass = (
                train_view.weights * (1.0 - train_view.targets)
            ).sum(axis=0)
            observed_mass = positive_mass + negative_mass
            rates = np.full(self.num_labels, 0.5, dtype=np.float32)
            # Partial rule supervision is often selection-biased.  A label
            # observed with only one polarity must not receive a saturated
            # prior; initialise it neutrally until both polarities are known.
            supported = (positive_mass > 0) & (negative_mass > 0)
            rates[supported] = positive_mass[supported] / observed_mass[supported]
            student.initialize_class_priors(torch.from_numpy(rates))
            prior_rates = rates.astype(float).tolist()
            class_prior_initialization_applied = True
        optimizer_initial_head_sha256 = _head_state_sha256(student)

        student.to(self._device)
        teacher = copy.deepcopy(student).to(self._device)
        teacher.requires_grad_(False)
        teacher.eval()

        train_loader = self._loader(
            train_texts, train_view, shuffle=True, seed_offset=17,
        )
        validation_loader = None
        if validation_text_list is not None and validation_view is not None:
            validation_loader = self._loader(
                validation_text_list,
                validation_view,
                shuffle=False,
                validation=True,
                seed_offset=29,
            )
        optimizer = self._optimizer(student)
        updates_per_epoch = int(math.ceil(
            len(train_loader) / float(self.config.grad_accum_steps)
        ))
        total_updates = max(1, updates_per_epoch * self.config.epochs)
        scheduler = self._scheduler(optimizer, total_updates)
        fp16_scaler_enabled = self._device.type == "cuda" and self.config.amp == "fp16"
        # ``amp_init_scale`` is an optional instance attribute set by the
        # caller; the default is the PyTorch GradScaler default (2**16).
        init_scale = float(getattr(self, "amp_init_scale", None) or 2.0 ** 16)
        if not math.isfinite(init_scale) or init_scale <= 0:
            raise ValueError("amp_init_scale must be a positive finite number")
        scaler = torch.cuda.amp.GradScaler(enabled=fp16_scaler_enabled, init_scale=init_scale)
        # Dynamic loss scaling skips updates while it halves an initial scale
        # that overflows; at most log2(init_scale)+1 consecutive skips before
        # the first successful update are this startup calibration.
        startup_limit = int(math.ceil(math.log2(max(init_scale, 1.0)))) + 1
        startup_skipped_optimizer_updates = 0

        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_validation_loss = math.inf
        best_epoch: Optional[int] = None
        best_source: Optional[str] = None
        stale_epochs = 0
        optimizer_updates = 0
        skipped_optimizer_updates = 0

        for epoch in range(1, self.config.epochs + 1):
            epoch_started = time.perf_counter()
            student.train()
            optimizer.zero_grad(set_to_none=True)
            train_numerator = 0.0
            train_denominator = 0.0
            accumulated_weight = 0.0
            for batch_index, batch in enumerate(train_loader):
                with self._autocast():
                    logits = self._forward_logits(student, batch)
                    _, batch_numerator, batch_denominator = self._loss_parts(
                        logits, batch,
                    )
                # Accumulate weighted numerators, then divide gradients by the
                # exact accumulated observed mass.  This makes an update
                # invariant to how sparse label pairs happen to be batched and
                # also handles a short final accumulation group correctly.
                scaler.scale(batch_numerator).backward()
                accumulated_weight += float(batch_denominator.cpu().item())
                should_update = (
                    (batch_index + 1) % self.config.grad_accum_steps == 0
                    or batch_index + 1 == len(train_loader)
                )
                if should_update:
                    scaler.unscale_(optimizer)
                    if accumulated_weight <= 0:
                        raise RuntimeError(
                            "internal error: gradient accumulation has zero weight"
                        )
                    inverse_weight = 1.0 / accumulated_weight
                    for parameter in student.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(inverse_weight)
                    torch.nn.utils.clip_grad_norm_(
                        student.parameters(), self.config.max_grad_norm,
                    )
                    scale_before = float(scaler.get_scale())
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = float(scaler.get_scale())
                    step_succeeded = (
                        not fp16_scaler_enabled or scale_after >= scale_before
                    )
                    optimizer.zero_grad(set_to_none=True)
                    if step_succeeded:
                        scheduler.step()
                        optimizer_updates += 1
                        warm_decay = min(
                            self.config.ema_decay,
                            1.0 - 1.0 / float(max(1, optimizer_updates)),
                        )
                        ema_update(teacher, student, warm_decay)
                    else:
                        skipped_optimizer_updates += 1
                        if (optimizer_updates == 0
                                and startup_skipped_optimizer_updates < startup_limit):
                            startup_skipped_optimizer_updates += 1
                    accumulated_weight = 0.0
                train_numerator += float(batch_numerator.detach().cpu().item())
                train_denominator += float(batch_denominator.cpu().item())

            record: Dict[str, Any] = {
                "epoch": epoch,
                "training_seconds": time.perf_counter() - epoch_started,
                "train_loss": train_numerator / max(train_denominator, 1.0e-12),
                "train_weight_sum": train_denominator,
                "optimizer_updates": optimizer_updates,
                "skipped_optimizer_updates": skipped_optimizer_updates,
                "learning_rates": [
                    float(group["lr"]) for group in optimizer.param_groups
                ],
            }
            if validation_loader is not None:
                validation_started = time.perf_counter()
                student_validation = self._validation_loss(student, validation_loader)
                student_finished = time.perf_counter()
                ema_validation = self._validation_loss(teacher, validation_loader)
                record["validation_student_seconds"] = student_finished - validation_started
                record["validation_ema_seconds"] = time.perf_counter() - student_finished
                record["validation_batch_size"] = validation_loader.batch_size
                choose_ema = (
                    ema_validation < student_validation
                    or (
                        self.config.prefer_ema
                        and abs(ema_validation - student_validation) <= 1.0e-12
                    )
                )
                selected_model = teacher if choose_ema else student
                selected_source = "ema" if choose_ema else "student"
                selected_validation = (
                    ema_validation if choose_ema else student_validation
                )
                record["validation_loss"] = {
                    "student": student_validation,
                    "ema": ema_validation,
                    "selected": selected_validation,
                    "selected_source": selected_source,
                }
                if selected_validation < best_validation_loss:
                    best_validation_loss = selected_validation
                    best_state = _copy_state_to_cpu(selected_model)
                    best_epoch = epoch
                    best_source = selected_source
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            record["epoch_seconds"] = time.perf_counter() - epoch_started
            self._history.append(record)
            if (
                validation_loader is not None
                and self.config.patience > 0
                and epoch >= self.config.min_epochs
                and stale_epochs >= self.config.patience
            ):
                break

        if best_state is not None:
            student.load_state_dict(best_state)
            selected = student
        elif self.config.prefer_ema:
            selected = teacher
            best_epoch = len(self._history)
            best_source = "ema"
        else:
            selected = student
            best_epoch = len(self._history)
            best_source = "student"
        selected.eval()
        selected.requires_grad_(False)
        self._model = selected
        self._fitted = True

        effective_amp = self.config.amp if self._device.type == "cuda" else "none"
        audit: Dict[str, Any] = {
            "schema_version": BACKEND_SCHEMA_VERSION,
            "backend": BACKEND_NAME,
            "num_labels": self.num_labels,
            "config": self.config.to_dict(),
            "config_sha256": self.config.fingerprint,
            "fresh_pretrained_initialization": initialization_lineage is None,
            "supervised_checkpoint_initialization": initialization_lineage is not None,
            "initialization_lineage": initialization_lineage,
            "previous_chase_round_state_loaded": False,
            "initialization_seed": int(self.config.seed),
            "fresh_random_head_sha256": fresh_random_head_sha256,
            "initial_head_sha256": optimizer_initial_head_sha256,
            "supervision_sha256": _supervision_fingerprint(
                train_texts,
                train_view,
                validation_text_list,
                validation_view,
            ),
            "train_supervision": train_view.summary(),
            "validation_supervision": (
                None if validation_view is None else validation_view.summary()
            ),
            "truth_roles_consumed": (
                ["train"] if validation_view is None else ["train", "validation"]
            ),
            "target_or_test_truth_api": False,
            "effective_device": str(self._device),
            "effective_amp": effective_amp,
            "class_prior_initialization_requested": bool(
                self.config.initialize_class_priors
            ),
            "class_prior_initialization_applied": class_prior_initialization_applied,
            "class_prior_rates": prior_rates,
            "epochs_completed": len(self._history),
            "optimizer_updates": optimizer_updates,
            "skipped_optimizer_updates": skipped_optimizer_updates,
            "startup_skipped_optimizer_updates": startup_skipped_optimizer_updates,
            "amp_init_scale": init_scale if fp16_scaler_enabled else None,
            "best_epoch": best_epoch,
            "selected_source": best_source,
            "best_validation_loss": (
                None
                if not math.isfinite(best_validation_loss)
                else float(best_validation_loss)
            ),
            "history": self.history,
        }
        self._fit_audit = audit
        return self.fit_audit

    @classmethod
    def from_checkpoint(
        cls,
        directory: Path,
        *,
        config: Optional[ChaseBackendConfig] = None,
        num_labels: Optional[int] = None,
        tokenizer_factory: Optional[TokenizerFactory] = None,
        model_factory: Optional[ModelFactory] = None,
        device: Optional[str] = None,
    ) -> "ChaseTransformerBackend":
        """Restore a completed backend without carrying state into a new fit.

        The restored object is inference-only/single-use.  Loris will still
        invoke the separately supplied backend factory for a future round,
        preserving the fresh-initialisation contract.
        """

        source = Path(directory)
        metadata_path = source / "backend.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                "invalid chase backend metadata at %s" % metadata_path
            ) from exc
        if (
            metadata.get("schema_version") != BACKEND_SCHEMA_VERSION
            or metadata.get("backend") != BACKEND_NAME
        ):
            raise RuntimeError("chase backend checkpoint type mismatch")
        try:
            saved_config = ChaseBackendConfig(**metadata["config"])
            saved_num_labels = int(metadata["num_labels"])
        except Exception as exc:
            raise RuntimeError("invalid chase backend checkpoint configuration") from exc
        if metadata.get("config_sha256") != saved_config.fingerprint:
            raise RuntimeError("chase backend checkpoint config fingerprint mismatch")
        if config is not None and config.fingerprint != saved_config.fingerprint:
            raise RuntimeError("chase backend checkpoint does not match expected config")
        if num_labels is not None and int(num_labels) != saved_num_labels:
            raise RuntimeError("chase backend checkpoint label width mismatch")
        if saved_num_labels <= 0:
            raise RuntimeError("invalid chase backend checkpoint label width")

        backend = cls(
            saved_config,
            saved_num_labels,
            tokenizer_factory=tokenizer_factory,
            model_factory=model_factory,
            device=device,
        )
        _seed_fresh_initialization(saved_config.seed)
        backend._device = backend._resolve_device()
        if tokenizer_factory is None:
            tokenizer_directory = source / "tokenizer"
            if metadata.get("tokenizer_saved") is not True or not tokenizer_directory.is_dir():
                raise RuntimeError("chase backend checkpoint has no restorable tokenizer")
            try:
                backend._tokenizer = AutoTokenizer.from_pretrained(
                    str(tokenizer_directory),
                    use_fast=True,
                    local_files_only=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    "failed to restore chase backend tokenizer"
                ) from exc
        else:
            backend._tokenizer = tokenizer_factory(saved_config)

        model = backend._model_factory(saved_config, saved_num_labels)
        if not isinstance(model, nn.Module):
            raise TypeError("model_factory must return torch.nn.Module")
        state_name = metadata.get("model_state_file")
        if state_name != "model_state.pt":
            raise RuntimeError("invalid chase backend model state filename")
        state_path = source / state_name
        declared_state_sha256 = metadata.get("model_state_sha256")
        if declared_state_sha256 is not None:
            if _SHA256_RE.fullmatch(str(declared_state_sha256)) is None:
                raise RuntimeError("invalid chase backend model state sha256")
            if _sha256_file(state_path) != declared_state_sha256:
                raise RuntimeError("chase backend model state sha256 mismatch")
        try:
            state = torch.load(str(state_path), map_location="cpu")
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise RuntimeError(
                "failed to restore chase backend model state"
            ) from exc
        model.to(backend._device)
        model.eval()
        model.requires_grad_(False)
        backend._model = model
        backend._history = list(metadata.get("history", []))
        backend._fit_audit = {
            key: value
            for key, value in metadata.items()
            if key not in {"model_state_file", "tokenizer_saved"}
        }
        backend._fit_audit["restored_from_checkpoint"] = True
        backend._started = True
        backend._fitted = True
        return backend

    @torch.no_grad()
    def _prediction_autocast(self):
        """Inference precision: the training AMP unless prediction_amp overrides it.

        ``prediction_amp`` is an instance attribute set by the caller (fp16 is
        the fast inference mode on pre-Ampere GPUs such as V100).
        """
        amp = getattr(self, "prediction_amp", None) or self.config.amp
        if self._device is None or self._device.type != "cuda" or amp == "none":
            return nullcontext()
        if amp == "bf16" and int(torch.cuda.get_device_capability(self._device)[0]) < 8:
            raise RuntimeError("bf16 inference requires an Ampere-or-newer CUDA device")
        return torch.cuda.amp.autocast(dtype=torch.float16 if amp == "fp16" else torch.bfloat16)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted or self._model is None or self._device is None:
            raise RuntimeError("chase backend is not fitted")
        text_list = [str(text) for text in texts]
        if not text_list:
            return np.empty((0, self.num_labels), dtype=np.float32)
        # Optional length bucketing: rows are predicted in order of character
        # length (stable) and scattered back, so padding per batch is small.
        # Attention masks make each row's logits independent of its batch.
        order = (np.argsort(np.fromiter((len(t) for t in text_list), np.int64, len(text_list)),
                            kind="stable")
                 if getattr(self, "prediction_length_sort", False) else None)
        ordered = text_list if order is None else [text_list[int(i)] for i in order]
        loader = self._loader(
            ordered, None, shuffle=False, prediction=True, seed_offset=41,
        )
        self._model.eval()
        rows: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                with self._prediction_autocast():
                    if self.config.prediction_logits_only and isinstance(self._model, MultiLabelEncoder):
                        # This branch is eval/no-grad only. Custom model factories
                        # keep their original forward contract.
                        output = self._model.prediction_outputs(**_model_inputs(batch, self._device))
                        logits = output["logits"]
                        if logits.ndim != 2 or logits.shape[1] != self.num_labels:
                            raise RuntimeError("model returned an invalid fast prediction logit shape")
                    else:
                        logits = self._forward_logits(self._model, batch)
                rows.append(torch.sigmoid(logits.float()).cpu().numpy())
        probabilities = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
        if order is not None:
            restored = np.empty_like(probabilities)
            restored[order] = probabilities
            probabilities = restored
        if probabilities.shape != (len(text_list), self.num_labels):
            raise RuntimeError("internal error: invalid probability shape")
        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError("model produced non-finite probabilities")
        return probabilities

    def save_checkpoint(self, directory: Path) -> None:
        """Save the selected inference state and its complete training audit."""

        if not self._fitted or self._model is None or self._fit_audit is None:
            raise RuntimeError("cannot checkpoint an unfitted chase backend")
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=False)
        model_path = target / "model_state.pt"
        torch.save(_copy_state_to_cpu(self._model), str(model_path))
        tokenizer_saved = False
        if hasattr(self._tokenizer, "save_pretrained"):
            self._tokenizer.save_pretrained(str(target / "tokenizer"))
            tokenizer_saved = True
        metadata = {
            **self._fit_audit,
            "model_state_file": model_path.name,
            "model_state_sha256": _sha256_file(model_path),
            "tokenizer_saved": tokenizer_saved,
        }
        (target / "backend.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )


def make_chase_backend_factory(
    config: Optional[ChaseBackendConfig] = None,
    *,
    tokenizer_factory: Optional[TokenizerFactory] = None,
    model_factory: Optional[ModelFactory] = None,
    device: Optional[str] = None,
) -> Callable[[Any, int], ChaseTransformerBackend]:
    """Return a factory directly accepted by Loris ``AuxiliaryAnnotator``.

    Passing an explicit frozen config is recommended for preregistered runs.
    With ``config=None``, compatible fields are bridged from Loris's narrower
    ``AuxiliaryAnnotatorConfig`` and all remaining fields use the strong
    defaults above.
    """

    def factory(auxiliary_config: Any, num_labels: int) -> ChaseTransformerBackend:
        effective = (
            config
            if config is not None
            else ChaseBackendConfig.from_auxiliary_config(auxiliary_config)
        )
        return ChaseTransformerBackend(
            effective,
            num_labels,
            tokenizer_factory=tokenizer_factory,
            model_factory=model_factory,
            device=device,
        )

    return factory


def make_chase_backend_loader(
    config: Optional[ChaseBackendConfig] = None,
    *,
    tokenizer_factory: Optional[TokenizerFactory] = None,
    model_factory: Optional[ModelFactory] = None,
    device: Optional[str] = None,
) -> Callable[[Any, int, Path], ChaseTransformerBackend]:
    """Return a loader accepted by ``AuxiliaryAnnotator.load_checkpoint``."""

    def loader(
        auxiliary_config: Any,
        num_labels: int,
        directory: Path,
    ) -> ChaseTransformerBackend:
        del auxiliary_config  # the exact inner config is bound in backend.json
        return ChaseTransformerBackend.from_checkpoint(
            directory,
            config=config,
            num_labels=num_labels,
            tokenizer_factory=tokenizer_factory,
            model_factory=model_factory,
            device=device,
        )

    return loader


__all__ = [
    "BACKEND_NAME",
    "ChaseBackendConfig",
    "ChaseTransformerBackend",
    "make_chase_backend_factory",
    "make_chase_backend_loader",
    "masked_weighted_loss",
]
