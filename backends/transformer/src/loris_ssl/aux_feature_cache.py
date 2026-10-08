"""Prediction-only frozen parent probability/CLS cache for isolated Aux trials.

No answers, trainer supervision or label frequencies enter this API. The cache
binds the exact checkpoint and text order, including truncation and precision.
"""
import hashlib
import json
from pathlib import Path
import time
from dataclasses import replace

import numpy as np
import torch

from .chase_backend import (ChaseTransformerBackend, _model_inputs, _sha256_file,
                            _load_supervised_checkpoint_initialization)
from .model import MultiLabelEncoder
from . import model as model_implementation


SCHEMA = "loris_frozen_selected_aux_probability_cls_cache_v1"


def load_frozen_parent(config, labels, *, checkpoint_path, checkpoint_sha256, device,
                       tokenizer_factory=None, model_factory=None):
    """Strict prediction-only loading; no fit, prior reset or optimiser exists.

    The caller owns the shared GPU lock for the complete cache extraction.
    """
    labels = tuple(labels)
    effective = replace(config, initial_checkpoint_path=str(Path(checkpoint_path).resolve(strict=True)),
                        initial_checkpoint_sha256=checkpoint_sha256,
                        initial_checkpoint_source_labels=labels,
                        initial_checkpoint_target_labels=labels,
                        initialize_class_priors=False)
    backend = ChaseTransformerBackend(effective, len(labels), device=device,
                                     tokenizer_factory=tokenizer_factory, model_factory=model_factory)
    backend._device = backend._resolve_device()
    backend._tokenizer = backend._tokenizer_factory(effective)
    model = backend._model_factory(effective, len(labels))
    lineage = _load_supervised_checkpoint_initialization(model, effective, len(labels))
    model.to(backend._device)
    model.eval()
    model.requires_grad_(False)
    backend._model = model
    backend._started = backend._fitted = True
    backend._fit_audit = dict(prediction_only_frozen_parent=True,
                             optimizer_updates=0, initialization_lineage=lineage)
    return backend


def _array_sha(value):
    value = np.ascontiguousarray(value)
    meta = json.dumps([str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
    return hashlib.sha256(meta + value.tobytes()).hexdigest()


def _list_sha(values):
    return hashlib.sha256(json.dumps(list(values), ensure_ascii=True,
                                  separators=(",", ":")).encode()).hexdigest()


def predict_cache(backend, texts, row_ids, labels, *, checkpoint_path,
                  checkpoint_sha256, feature_dtype="float32"):
    """One encoder pass obtains original logits and its first-token features.

    Caller must have loaded the checkpoint strictly into this fitted backend.
    We verify every loaded tensor against the checkpoint, not merely its path.
    No model parameter or configured training setting is changed by the API.
    """
    if not backend._fitted or backend._model is None or backend._device is None:
        raise RuntimeError("strictly loaded fitted backend required")
    if not isinstance(backend._model, MultiLabelEncoder):
        raise TypeError("cache supports the declared MultiLabelEncoder semantics only")
    if feature_dtype not in ("float16", "float32"):
        raise ValueError("feature_dtype must explicitly be float16 or float32")
    texts, row_ids, labels = list(texts), list(row_ids), list(labels)
    if not texts or len(row_ids) != len(texts) or len(set(row_ids)) != len(row_ids):
        raise ValueError("nonempty complete unique text row axis required")
    if len(labels) != backend.num_labels or len(set(labels)) != len(labels):
        raise ValueError("complete unique label axis required")
    tokenizer = getattr(backend, "_tokenizer", None)
    if tokenizer is None or not callable(getattr(tokenizer, "get_vocab", None)):
        raise ValueError("loaded tokenizer with an explicit vocabulary required")
    vocabulary = tokenizer.get_vocab()
    tokenizer_binding = dict(
        class_name=type(tokenizer).__module__ + "." + type(tokenizer).__qualname__,
        vocabulary_sha256=hashlib.sha256(json.dumps(vocabulary, sort_keys=True,
            ensure_ascii=True, separators=(",", ":")).encode()).hexdigest(),
        special_tokens_sha256=hashlib.sha256(json.dumps(
            getattr(tokenizer, "special_tokens_map", {}), sort_keys=True,
            ensure_ascii=True, separators=(",", ":"), default=str).encode()).hexdigest(),
    )
    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    if _sha256_file(checkpoint_path) != checkpoint_sha256:
        raise ValueError("frozen parent checkpoint SHA differs")
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if payload.get("label_names") != labels or payload.get("model_name") != backend.config.model_name or payload.get("head") != backend.config.head:
        raise ValueError("cache checkpoint architecture/label order differs")
    actual, expected = backend._model.state_dict(), payload.get("state_dict", {})
    if set(actual) != set(expected) or any(
            actual[k].shape != expected[k].shape or actual[k].dtype != expected[k].dtype
            or not torch.equal(actual[k].detach().cpu(), expected[k]) for k in actual):
        raise ValueError("cache backend parameters do not match the frozen checkpoint")
    del payload, expected, actual
    order = (np.argsort(np.fromiter((len(t) for t in texts), np.int64, len(texts)), kind="stable")
             if getattr(backend, "prediction_length_sort", False) else np.arange(len(texts)))
    ordered = [texts[int(i)] for i in order]
    loader = backend._loader(ordered, None, shuffle=False, prediction=True, seed_offset=41)
    model = backend._model
    model.eval()
    probabilities, features = [], []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            with backend._prediction_autocast():
                outputs = model.prediction_outputs(**_model_inputs(batch, backend._device), document_features=True)
            probabilities.append(torch.sigmoid(outputs["logits"].float()).cpu().numpy())
            features.append(outputs["document_features"].float().cpu().numpy())
    p, h = np.concatenate(probabilities).astype(np.float32), np.concatenate(features).astype(feature_dtype)
    restored_p, restored_h = np.empty_like(p), np.empty_like(h)
    restored_p[order], restored_h[order] = p, h
    if restored_p.shape != (len(texts), len(labels)) or not np.isfinite(restored_p).all() or not np.isfinite(restored_h).all():
        raise RuntimeError("cache has invalid full text axes or non-finite values")
    if _sha256_file(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("frozen parent checkpoint changed during cache prediction")
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if any(not torch.equal(v.detach().cpu(), payload["state_dict"][k]) for k,v in model.state_dict().items()):
        raise RuntimeError("frozen parent parameters changed during cache prediction")
    del payload
    receipt = dict(schema=SCHEMA, parent_checkpoint_sha256=checkpoint_sha256,
                   feature_checkpoint_sha256=checkpoint_sha256,
                   row_ids_sha256=_list_sha(row_ids), labels_sha256=_list_sha(labels),
                   texts_sha256=_list_sha(texts), probabilities_sha256=_array_sha(restored_p),
                   features_sha256=_array_sha(restored_h),
                   feature_definition="frozen_parent_normalized_first_encoder_token",
                   feature_supervision_used=False, feature_storage_dtype=feature_dtype,
                   feature_storage_quantized=feature_dtype == "float16",
                   prediction_seconds=time.perf_counter()-started,
                   inference_configuration=dict(model_name=backend.config.model_name,
                       head=backend.config.head, max_length=backend.config.max_length,
                       truncation=backend.config.truncation,
                       prediction_amp=getattr(backend, "prediction_amp", None) or backend.config.amp,
                       prediction_batch_size=backend.config.prediction_batch_size,
                       prediction_length_sort=bool(getattr(backend, "prediction_length_sort", False)),
                       tokenizer=tokenizer_binding,
                       source_files={str(Path(__file__).resolve()): _sha256_file(Path(__file__)),
                           str(Path(model_implementation.__file__).resolve()): _sha256_file(Path(model_implementation.__file__))},
                       logits_only_evaluation=True),
                   old_encoder_and_old_head_frozen=True,
                   mixed_checkpoint_rows=False, complete_requested_row_order_restored=True)
    return restored_p, restored_h, receipt
