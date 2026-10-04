"""Real tiny CPU Aux work with a controlled trainer deadline, not timing benchmarks."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from loris.discovery.manifest import file_record
from loris.labeling.feedback import Batch
from loris.labeling.runner import LabelingRun
from loris.labeling import trainer
from tests.test_labeling import fixture
from tests.test_labeling_actual_aux import Encoder, Tokenizer, assets


@pytest.fixture
def update(tmp_path):
    source, parent, config = assets(tmp_path)
    observed = np.ones((2, 2), bool)
    batch = Batch(
        np.array([[1, 0], [0, 1]], np.float32), observed,
        np.ones((2, 2), np.float32), observed, ~observed, {},
    )
    return dict(
        texts=["actual small train first", "actual small train second"],
        labels=["A", "B"], batch=batch, checkpoint=parent,
        checkpoint_sha256=file_record(parent)["sha256"], source=source,
        configuration=config, device="cpu", gpu_lock=None, max_seconds=1,
        tokenizer_factory=lambda _: Tokenizer(), model_factory=lambda _, k: Encoder(k),
    )


def controlled_deadline(monkeypatch, phase, *, extra=False, elapsed=2.):
    """Run real operations; advance only trainer's clock after the chosen one."""
    from loris._engine.model import _ContinuationBackend

    clock = SimpleNamespace(now=0., optimizer_steps=0, predictions=0, target_records=0)
    monkeypatch.setattr(trainer, "time", SimpleNamespace(monotonic=lambda: clock.now))
    step = torch.optim.AdamW.step
    predict = _ContinuationBackend.predict_proba
    save = torch.save
    record = trainer.file_record

    def actual_step(self, *args, **kwargs):
        value = step(self, *args, **kwargs)
        clock.optimizer_steps += 1
        if phase == "step":
            clock.now = elapsed
        return value

    def actual_prediction(self, texts):
        value = predict(self, texts)
        clock.predictions += 1
        if phase == "prediction" or (phase == "independent" and clock.predictions == 2):
            clock.now = elapsed
        return value

    def actual_save(value, path, *args, **kwargs):
        result = save(value, path, *args, **kwargs)
        if phase == "save" and Path(path).name == "continuation.pt":
            clock.now = elapsed
        return result

    def actual_record(path):
        value = record(path)
        if phase == "receipt" and Path(path).name == "continuation.pt":
            clock.target_records += 1
            # An extra prediction records its checkpoint first; the final
            # receipt then hashes it again. Expire only at that last hash.
            if clock.target_records == 1 + int(extra):
                clock.now = elapsed
        return value

    monkeypatch.setattr(torch.optim.AdamW, "step", actual_step)
    monkeypatch.setattr(_ContinuationBackend, "predict_proba", actual_prediction)
    monkeypatch.setattr(torch, "save", actual_save)
    monkeypatch.setattr(trainer, "file_record", actual_record)
    return clock


def prediction_options(extra):
    return {"prediction_sets": {"independent_calibration": ["never optimized"]}} if extra else {}


@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize("phase,message", [
    ("step", "after training"),
    ("save", "after checkpoint save"),
    ("prediction", "after full-axis prediction"),
    ("receipt", "before success receipt"),
])
def test_overrun_never_publishes_success_and_preserves_real_step_count(
    update, tmp_path, monkeypatch, extra, phase, message
):
    clock = controlled_deadline(monkeypatch, phase, extra=extra)
    output = tmp_path / "update"
    with pytest.raises(TimeoutError, match=message):
        trainer.train_actual(output=output, **update, **prediction_options(extra))
    failure = json.loads((output / "FAILED.json").read_text())
    trace = json.loads((output / "ACTUAL_TRAINING_TRACE.json").read_text())
    assert failure["status"] == "failed_not_committed"
    assert failure["elapsed_seconds"] == 2. > update["max_seconds"]
    assert failure["parent_checkpoint_preserved"] is True
    assert failure["actual_optimizer_updates"] == trace["successful_optimizer_updates"] == 1
    assert failure["backend_audit"]["optimizer_updates"] == clock.optimizer_steps == 1
    assert not (output / "COMPLETE.json").exists()
    assert file_record(update["checkpoint"])["sha256"] == update["checkpoint_sha256"]
    assert clock.predictions == (0 if phase in ("step", "save") else 1 + int(extra and phase == "receipt"))
    if phase == "step":
        assert not (output / "continuation.pt").exists()


def test_independent_prediction_overrun_keeps_the_same_failure_contract(update, tmp_path, monkeypatch):
    clock = controlled_deadline(monkeypatch, "independent", extra=True)
    output = tmp_path / "update"
    with pytest.raises(TimeoutError, match="after independent prediction"):
        trainer.train_actual(output=output, **update, **prediction_options(True))
    failure = json.loads((output / "FAILED.json").read_text())
    assert failure["actual_optimizer_updates"] == 1 and clock.predictions == 2
    assert failure["parent_checkpoint_preserved"] is True
    assert not (output / "COMPLETE.json").exists()


@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize("elapsed", [.75, 1.])
def test_success_before_or_exactly_at_budget_is_unchanged(update, tmp_path, monkeypatch, extra, elapsed):
    clock = controlled_deadline(monkeypatch, "prediction", extra=extra, elapsed=elapsed)
    output = tmp_path / "update"
    p, receipt = trainer.train_actual(output=output, **update, **prediction_options(extra))
    assert receipt["status"] == "complete_uncommitted"
    assert receipt["elapsed_seconds"] == elapsed <= update["max_seconds"]
    assert receipt["optimizer_updates"] == clock.optimizer_steps == 1
    assert clock.predictions == 1 + int(extra)
    assert p.shape == update["batch"].targets.shape and np.isfinite(p).all()
    assert (output / "COMPLETE.json").exists() and not (output / "FAILED.json").exists()
    assert bool(receipt["prediction_sets"]) is extra


def test_real_supervised_runner_rejects_overdue_candidate_and_retains_checkpoint(update, tmp_path, monkeypatch):
    clock = controlled_deadline(monkeypatch, "prediction")
    context, rules, manifest = fixture()
    manifest["strategy"] = "supervised"
    manifest["loop"]["max_update_seconds"] = 1
    manifest["dynamic"].update(
        checkpoint=file_record(update["checkpoint"]), backend_source=update["source"],
        configuration=update["configuration"],
    )
    options = []

    def actual(**kwargs):
        options.append("prediction_sets" in kwargs)
        return trainer.train_actual(**kwargs, tokenizer_factory=update["tokenizer_factory"],
                                    model_factory=update["model_factory"])

    run = LabelingRun(context=context, rules=rules, manifest=manifest,
                      output=tmp_path / "run", trainer=actual)
    report = run.run()
    assert options == [False] and clock.optimizer_steps == 1
    assert report["status"] == "update_failed_retaining_zero_update_state"
    assert report["training_attempts"] == report["actual_optimizer_updates"] == 1
    assert report["committed_model_updates"] == 0
    assert report["actual_optimizer_update_count_complete"] is True
    assert report["failed_training_attempts_without_step_receipt"] == 0
    assert report["checkpoint"]["sha256"] == update["checkpoint_sha256"]
    assert report["current_Gamma_has_valid_closure"] is True
    np.testing.assert_array_equal(run.gamma.pos, context["gamma"].pos)
    np.testing.assert_array_equal(run.gamma.neg, context["gamma"].neg)
    assert any(e["kind"] == "update_not_committed" and e["error_type"] == "TimeoutError" for e in run.events)
