"""Real current-checkpoint Aux continuation; no training stub or truth loader."""

from dataclasses import replace
import contextlib
import gc
import fcntl
import importlib
import json
import math
import re
from pathlib import Path
import sys
import time
import types

import numpy as np

from loris.discovery.manifest import file_record


def load_backend(source):
    source = Path(source).resolve(strict=True)
    expected = source / "loris_ssl/chase_backend.py"
    if not expected.is_file():
        raise ValueError("real loris_ssl backend source required")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    backend = importlib.import_module("loris_ssl.chase_backend")
    if Path(backend.__file__).resolve() != expected:
        raise ValueError("already imported backend differs from declared source")
    return backend


AMP_MODES = ("fp16", "bf16", "none", "auto")


def additional_predictions(backend, application_texts, probabilities, values, *, reuse):
    """Predict an extra text axis, optionally reusing this candidate's own rows.

    Reuse never crosses checkpoints or uses supervision. Different batching
    can change mixed-precision rounding; the opted-in receipt declares that.
    """
    if not reuse:
        return np.asarray(backend.predict_proba(list(values)), np.float32), None
    by_text = {}
    for index, text in enumerate(application_texts):
        by_text.setdefault(text, index)
    result = np.empty((len(values), probabilities.shape[1]), np.float32)
    missing, missing_index, requested = [], {}, []
    reused = 0
    for row, text in enumerate(values):
        index = by_text.get(text)
        if index is not None:
            result[row] = probabilities[index]
            reused += 1
        else:
            if text not in missing_index:
                missing_index[text] = len(missing)
                missing.append(text)
            requested.append((row, missing_index[text]))
    if missing:
        predicted = np.asarray(backend.predict_proba(missing), np.float32)
        if predicted.shape != (len(missing), probabilities.shape[1]):
            raise RuntimeError("additional unique-text probability axis differs")
        rows, indices = np.asarray(requested, dtype=int).T
        result[rows] = predicted[indices]
    return result, dict(policy="same_text_same_current_checkpoint",
        application_rows_reused=reused, unique_new_texts=len(missing),
        duplicate_additional_rows_reused=len(requested)-len(missing),
        supervision_used=False, full_requested_row_order_restored=True,
        mixed_precision_batch_rounding_may_differ=True)


def resolve_amp(configuration, device):
    """Configuration with ``amp="auto"`` resolved for the actual device.

    auto selects bf16 on CUDA devices of compute capability 8.0 or newer, fp16
    (dynamic loss scaling) on older CUDA devices such as V100, and none on CPU.
    Explicit fp16/bf16/none values are returned unchanged.
    """
    configuration = dict(configuration)
    amp = configuration.get("amp")
    if amp != "auto":
        return configuration
    if device != "cuda":
        configuration["amp"] = "none"
        return configuration
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but no CUDA device is visible")
    major = int(torch.cuda.get_device_capability()[0])
    configuration["amp"] = "bf16" if major >= 8 else "fp16"
    return configuration


def train_actual(
    *,
    texts,
    labels,
    batch,
    checkpoint,
    checkpoint_sha256,
    source,
    configuration,
    device,
    output,
    gpu_lock,
    max_seconds,
    tokenizer_factory=None,
    model_factory=None,
    prediction_sets=None,
    amp_skip_tolerance=0.0,
    prediction_amp=None,
    prediction_length_sort=False,
    amp_init_scale=None,
    reuse_application_predictions=False,
):
    """Existing actual backend with passive sampler/mass observation.

    No dev loss selects a checkpoint: full declared epochs select the configured
    final student/EMA. Optimizer/scheduler/EMA reset; parent parameters do not.
    A wall-time overrun is checked at loss, completed training, checkpoint and
    prediction boundaries, and before publishing a success receipt. An overrun
    aborts without committing a candidate. These are cooperative checks, not
    hard interruption: model construction and an individual backend operation
    are included in elapsed time but cannot be preempted safely.
    """
    prediction_sets = {} if prediction_sets is None else dict(prediction_sets)
    if type(reuse_application_predictions) is not bool:
        raise ValueError("reuse_application_predictions must be a bool")
    for name, values in prediction_sets.items():
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None
            or not isinstance(values, (list, tuple))
            or not values
            or any(not isinstance(text, str) for text in values)
        ):
            raise ValueError(
                "prediction sets require safe names and nonempty text lists; no supervision accepted"
            )
    implementation = load_backend(source)
    # Import only AFTER authenticating the optional package's source.
    from loris._engine.model import _ContinuationBackend, state_dict_sha256
    import torch
    from types import SimpleNamespace

    requested_amp = dict(configuration).get("amp")
    configuration = resolve_amp(configuration, device)
    config = implementation.ChaseBackendConfig(**configuration)
    if (
        config.loss != "masked_weighted_bce"
        or config.patience != 0
        or config.min_epochs != config.epochs
        or config.initialize_class_priors
        or config.num_workers != 0
    ):
        raise ValueError(
            "continuation requires explicit BCE/full epochs/no prior reset/num_workers=0"
        )
    if type(max_seconds) is not int or max_seconds < 1:
        raise ValueError("explicit positive training seconds cap required")
    if (type(amp_skip_tolerance) not in (int, float) or not 0 <= amp_skip_tolerance < 0.5
            or prediction_amp not in (None, "none", "fp16", "bf16")
            or type(prediction_length_sort) is not bool
            or (amp_init_scale is not None and (type(amp_init_scale) not in (int, float)
                                                or not math.isfinite(amp_init_scale)
                                                or amp_init_scale <= 0))):
        raise ValueError("invalid AMP skip tolerance / init scale / prediction precision options")
    if device not in ("cpu", "cuda"):
        raise ValueError("explicit cpu/cuda device required")
    if device == "cuda" and not gpu_lock:
        raise ValueError("CUDA training requires an explicit shared GPU lock")
    parent = Path(checkpoint).resolve(strict=True)
    if file_record(parent)["sha256"] != checkpoint_sha256:
        raise ValueError("current checkpoint SHA mismatch")
    payload = torch.load(str(parent), map_location="cpu")
    if (
        tuple(payload.get("label_names", ())) != tuple(labels)
        or payload.get("model_name") != config.model_name
        or payload.get("head") != config.head
    ):
        raise ValueError("parent checkpoint architecture/label axis differs")
    parent_state = state_dict_sha256(payload["state_dict"])
    generation = int(payload.get("paper36_continuation", {}).get("generation", 0))
    del payload
    effective = replace(
        config,
        initial_checkpoint_path=str(parent),
        initial_checkpoint_sha256=checkpoint_sha256,
        initial_checkpoint_source_labels=tuple(labels),
        initial_checkpoint_target_labels=tuple(labels),
    )
    active = np.flatnonzero(batch.observed.any(1))
    if not len(active):
        raise ValueError("no effective training observations")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    trace = {
        "sampled_rows": [],
        "microbatches": [],
        "windows": [],
        "successful_optimizer_updates": 0,
    }
    observed_rows = 0
    last_progress = -30.0
    source_records = [file_record(p) for p in sorted((Path(source) / "loris_ssl").rglob("*.py"))]

    def write(name, value):
        with (output / name).open("x") as f:
            json.dump(value, f, indent=2, allow_nan=False)

    def check_time(phase):
        elapsed = time.monotonic() - started
        if elapsed > max_seconds:
            raise TimeoutError("declared Aux update time cap exceeded " + phase)
        return elapsed

    write(
        "STARTED.json",
        dict(
            checkpoint=file_record(parent),
            config=effective.to_dict(),
            backend_sources=source_records,
            training_rows=len(active),
            validation=None,
            objective="sum(m*w*BCE)/sum(m*w) within each actual accumulation window",
            true_and_weak_separate_unit_normalization=False,
            max_seconds=max_seconds,
        ),
    )

    class Sampler:
        def __init__(self, original):
            self.original = original

        def __len__(self):
            return len(self.original)

        def __iter__(self):
            for i in self.original:
                trace["sampled_rows"].append(int(active[int(i)]))
                yield i

    class Recorded(_ContinuationBackend):
        def _optimizer(self, model):
            optimizer = super()._optimizer(model)
            original = optimizer.step

            def step(optimizer_self, *args, **kwargs):
                value = original(*args, **kwargs)
                trace["successful_optimizer_updates"] += 1
                return value

            optimizer.step = types.MethodType(step, optimizer)
            return optimizer

        def _loader(self, *args, **kwargs):
            loader = super()._loader(*args, **kwargs)
            if kwargs.get("shuffle"):
                loader.batch_sampler.sampler = Sampler(loader.batch_sampler.sampler)
            return loader

        def _loss_parts(self, logits, values):
            nonlocal observed_rows, last_progress
            check_time("before loss")
            result = super()._loss_parts(logits, values)
            if logits.requires_grad:
                count = len(logits)
                offset = observed_rows
                rows = trace["sampled_rows"][offset : offset + count]
                if len(rows) != count:
                    raise RuntimeError("actual sampler trace incomplete")
                real = float(batch.weights[rows][batch.true_mask[rows]].sum(dtype=np.float64))
                weak = float(batch.weights[rows][batch.weak_mask[rows]].sum(dtype=np.float64))
                total = float(result[2].detach().cpu())
                if not np.isclose(real + weak, total, rtol=1e-5, atol=1e-5):
                    raise RuntimeError(
                        "observed true/weak mass differs from actual backend denominator"
                    )
                trace["microbatches"].append(
                    dict(
                        rows=count,
                        true_mass=real,
                        weak_mass=weak,
                        total_mass=total,
                        numerator=float(result[1].detach().cpu()),
                    )
                )
                observed_rows += count
                elapsed = time.monotonic() - started
                if elapsed - last_progress >= 30:
                    from loris._engine.rlearner import atomic_json

                    per_epoch = math.ceil(len(active) / config.batch_size)
                    atomic_json(output / "PROGRESS.json", dict(
                        phase="training_forward_observed",
                        epoch=(len(trace["microbatches"]) - 1) // per_epoch + 1,
                        planned_epochs=config.epochs,
                        observed_microbatches=len(trace["microbatches"]),
                        planned_microbatches=per_epoch * config.epochs,
                        optimizer_updates=trace["successful_optimizer_updates"],
                        observed_rows=observed_rows, elapsed_seconds=elapsed,
                    ))
                    last_progress = elapsed
            return result

    backend = None
    lock = None
    try:
        if device == "cuda":
            lock = Path(gpu_lock).open("a+")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        backend = Recorded(
            effective,
            len(labels),
            expected_initial_state_sha256=parent_state,
            parent_generation=generation,
            device=device,
            tokenizer_factory=tokenizer_factory,
            model_factory=model_factory,
        )
        backend.prediction_amp = prediction_amp
        backend.prediction_length_sort = prediction_length_sort
        backend.amp_init_scale = amp_init_scale
        supervision = SimpleNamespace(
            targets=batch.targets, observed=batch.observed, weights=batch.weights
        )
        fit_started = time.monotonic()
        audit = backend.fit(list(texts), supervision, None, None)
        fit_seconds = time.monotonic() - fit_started
        check_time("after training")
        micro_per_epoch = math.ceil(len(active) / config.batch_size)
        for epoch in range(config.epochs):
            chunks = trace["microbatches"][epoch * micro_per_epoch : (epoch + 1) * micro_per_epoch]
            for offset in range(0, len(chunks), config.grad_accum_steps):
                part = chunks[offset : offset + config.grad_accum_steps]
                real, weak = (sum(x[k] for x in part) for k in ("true_mass", "weak_mass"))
                trace["windows"].append(
                    dict(
                        epoch=epoch,
                        true_mass=real,
                        weak_mass=weak,
                        weak_fraction=weak / (real + weak),
                        true_fraction=real / (real + weak),
                    )
                )
        expected = config.epochs * math.ceil(micro_per_epoch / config.grad_accum_steps)
        if (
            audit["epochs_completed"] != config.epochs
            or audit["optimizer_updates"] + audit["skipped_optimizer_updates"] != expected
            or audit["initial_full_state_sha256"] != parent_state
        ):
            raise RuntimeError("real backend failed full-budget/current-parameter contract")
        if audit["optimizer_updates"] != trace["successful_optimizer_updates"]:
            raise RuntimeError("actual optimizer step observation differs from backend receipt")
        # fp16 GradScaler skips an update when gradients overflow (standard
        # dynamic loss scaling). Skips before the first successful update while
        # the initial scale is halved are its startup calibration; the
        # declared tolerance bounds the fraction of the remaining skips. 0
        # keeps the original strict no-skip contract after startup.
        skipped = audit["skipped_optimizer_updates"]
        startup = audit.get("startup_skipped_optimizer_updates", 0)
        if skipped - startup > amp_skip_tolerance * expected:
            raise RuntimeError("AMP skipped updates beyond the declared tolerance; candidate not certified")
        state = {k: v.detach().cpu().clone() for k, v in backend._model.state_dict().items()}
        state_id = state_dict_sha256(state)
        target = output / "continuation.pt"
        torch.save(
            dict(
                state_dict=state,
                label_names=list(labels),
                model_name=config.model_name,
                head=config.head,
                paper36_continuation=dict(
                    schema="paper36_continuation_checkpoint_v1",
                    generation=generation + 1,
                    parent_checkpoint_sha256=checkpoint_sha256,
                    state_dict_sha256=state_id,
                ),
            ),
            str(target),
        )
        check_time("after checkpoint save")
        if file_record(parent)["sha256"] != checkpoint_sha256:
            raise RuntimeError("parent checkpoint changed")
        for item in source_records:
            if file_record(item["path"]) != item:
                raise RuntimeError("backend source changed during training")
        if audit["optimizer_updates"] < 1:
            raise RuntimeError("no actual optimizer update")
        check_time("before full-axis prediction")
        prediction_started = time.monotonic()
        probabilities = np.asarray(backend.predict_proba(list(texts)), np.float32)
        prediction_seconds = time.monotonic() - prediction_started
        check_time("after full-axis prediction")
        if probabilities.shape != batch.targets.shape or not np.isfinite(probabilities).all():
            raise RuntimeError("actual full-axis probability prediction failed")
        np.savez_compressed(output / "PROBABILITIES.npz", probabilities=probabilities)
        from loris._engine.typed_session import digest

        extra_predictions = {}
        for name, values in prediction_sets.items():
            check_time("before independent prediction")
            tick = time.monotonic()
            predictions, reuse_audit = additional_predictions(
                backend, texts, probabilities, values, reuse=reuse_application_predictions,
            )
            if (
                predictions.shape != (len(values), len(labels))
                or not np.isfinite(predictions).all()
                or np.any((predictions < 0) | (predictions > 1))
            ):
                raise RuntimeError("additional full-axis probability prediction failed: " + name)
            path = output / ("PREDICTIONS_" + name + ".npz")
            np.savez_compressed(path, probabilities=predictions)
            extra_predictions[name] = dict(
                file=file_record(path),
                rows=len(values),
                labels=list(labels),
                texts_id=digest(list(values)),
                checkpoint_sha256=file_record(target)["sha256"],
                prediction_seconds=time.monotonic() - tick,
                supervision_used=False,
            )
            if reuse_audit is not None:
                extra_predictions[name]["current_checkpoint_prediction_reuse"] = reuse_audit
            check_time("after independent prediction")
        receipt = dict(
            status="complete_uncommitted",
            checkpoint=file_record(target),
            parent_checkpoint_sha256=checkpoint_sha256,
            parent_state_sha256=parent_state,
            selected_state_sha256=state_id,
            parameter_state_changed=state_id != parent_state,
            audit=audit,
            optimizer_updates=audit["optimizer_updates"],
            model_prediction_seconds=prediction_seconds,
            model_fit_seconds=fit_seconds,
            fit_timing_scope="backend.fit including loaders and optimizer setup; excludes checkpoint save and predictions",
            checkpoint_selection="configured_final_EMA_or_student_no_dev_selection",
            amp_skip_tolerance=amp_skip_tolerance,
            skipped_optimizer_updates=audit["skipped_optimizer_updates"],
            startup_skipped_optimizer_updates=audit.get("startup_skipped_optimizer_updates", 0),
            amp=config.amp,
            amp_requested=requested_amp,
            amp_init_scale=audit.get("amp_init_scale"),
            prediction_amp=prediction_amp or effective.amp,
            prediction_length_sort=prediction_length_sort,
            window_count=len(trace["windows"]),
            source_records=source_records,
            probability_role="dynamic_alias_only_not_trusted_Gamma",
            prediction_sets=extra_predictions,
        )
        receipt["elapsed_seconds"] = check_time("before success receipt")
        write("COMPLETE.json", receipt)
        return probabilities, receipt
    except BaseException as exc:
        write(
            "FAILED.json",
            dict(
                status="failed_not_committed",
                error_type=type(exc).__name__,
                error=str(exc),
                elapsed_seconds=time.monotonic() - started,
                parent_checkpoint_preserved=file_record(parent)["sha256"] == checkpoint_sha256,
                actual_optimizer_updates=trace["successful_optimizer_updates"],
                backend_audit=locals().get("audit"),
                step_count_scope="actual optimizer.step returns observed including failed fits",
            ),
        )
        raise
    finally:
        write("ACTUAL_TRAINING_TRACE.json", trace)
        if backend is not None:
            backend._model = None
            backend._tokenizer = None
        # Optimizer step instrumentation can create reference cycles retaining
        # CUDA tensors after fit(). Release them while the shared GPU lock is
        # still held, including on failed candidates.
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        if lock is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
