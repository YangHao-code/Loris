"""Opted-in optimizer budgets preserve honest feedback and prediction lineage."""
import copy
import json
import numpy as np
import pytest

from loris.labeling.feedback import Batch, incremental_batch, limit_training_rows
from loris.labeling.trainer import additional_predictions, train_actual
from loris.discovery.manifest import file_record


def batch_fixture():
    weak = np.zeros((12, 2), bool)
    weak[:, 0] = True
    true = np.zeros_like(weak)
    true[-2:, 1] = True
    targets = np.zeros((12, 2), np.float32)
    targets[:, 0] = 1
    weights = weak.astype(np.float32) * .4 + true
    return Batch(targets, weak | true, weights, true, weak,
                 {}, np.where(weak, 1, 0).astype(np.int8))


def test_row_budget_defaults_unchanged_and_validation():
    batch = batch_fixture()
    before = copy.deepcopy(batch)
    assert limit_training_rows(batch, None, [], None, seed=42) is batch
    assert batch.audit == before.audit == {}
    assert np.array_equal(batch.feedback_state, before.feedback_state)
    for maximum in (True, 0, -2, 1.5):
        with pytest.raises(ValueError):
            limit_training_rows(copy.deepcopy(batch), maximum, [], None, seed=42)
    with pytest.raises(ValueError, match="newly verified"):
        limit_training_rows(batch, 1, [10, 11], None, seed=42)


def test_row_budget_keeps_fresh_and_has_actual_masks_mass_and_determinism():
    first = limit_training_rows(batch_fixture(), 4, [10, 11], None, seed=42)
    second = limit_training_rows(batch_fixture(), 4, [10, 11], None, seed=42)
    active = first.observed.any(1)
    assert active.sum() == 4 and active[-2:].all()
    assert np.array_equal(first.observed, second.observed)
    assert first.audit['true_cells'] == first.true_mask.sum() == 2
    assert first.audit['weak_cells'] == first.weak_mask.sum() == 4
    assert first.audit['weak_mass'] == pytest.approx(1.6)
    for array in (first.targets, first.weights, first.observed, first.true_mask, first.weak_mask):
        assert not array[~active].any()
    assert not first.feedback_state[~active].any()


def test_omitted_changed_weak_fact_remains_eligible_on_next_feedback():
    from types import SimpleNamespace
    previous = np.full((12, 2), -1, np.int8)
    first = limit_training_rows(batch_fixture(), 4, [10, 11], previous, seed=42)
    omitted = ~first.observed.any(1)
    assert np.array_equal(first.feedback_state[omitted, 0], previous[omitted, 0])
    next_batch = batch_fixture()
    gamma = SimpleNamespace(pos=np.zeros((12, 2), bool), neg=np.zeros((12, 2), bool))
    incremental_batch(next_batch, gamma, np.ones(12, bool), [], 0., first.feedback_state)
    assert next_batch.weak_mask[omitted, 0].all()
    assert not next_batch.weak_mask[~omitted, 0].any()


def test_same_checkpoint_reuse_restores_duplicate_axis_and_avoids_only_known_texts():
    class Backend:
        calls = []
        def predict_proba(self, values):
            self.calls.append(values)
            return np.array([[.2, .3] for _ in values], np.float32)
    backend = Backend()
    p = np.array([[.8, .7], [.4, .6]], np.float32)
    values = ['b', 'unseen', 'a', 'unseen', 'a']
    actual, audit = additional_predictions(backend, ['a', 'b'], p, values, reuse=True)
    assert backend.calls == [['unseen']]
    assert np.array_equal(actual, np.array([p[1], [.2,.3], p[0], [.2,.3], p[0]], np.float32))
    assert audit['application_rows_reused'] == 3
    assert audit['duplicate_additional_rows_reused'] == 1
    assert audit['mixed_precision_batch_rounding_may_differ']
    backend.calls.clear()
    _, audit = additional_predictions(backend, ['a', 'b'], p, values, reuse=False)
    assert backend.calls == [values] and audit is None


def test_actual_reused_prediction_binding_and_optimizer_scope(tmp_path):
    from tests.test_labeling_actual_aux import assets, Tokenizer, Encoder
    from loris._engine.typed_session import digest
    source, parent, config = assets(tmp_path)
    texts = ['train first', 'train second', 'untouched application']
    values = [texts[2], 'calibration only', texts[0], 'calibration only']
    targets = np.array([[1, 0], [0, 1], [0, 0]], np.float32)
    observed = np.array([[1, 1], [1, 1], [0, 0]], bool)
    batch = Batch(targets, observed, observed.astype(np.float32), observed, np.zeros_like(observed), {})
    p, receipt = train_actual(texts=texts, labels=['A', 'B'], batch=batch,
        checkpoint=parent, checkpoint_sha256=file_record(parent)['sha256'], source=source,
        configuration=config, device='cpu', output=tmp_path/'fit', gpu_lock=None, max_seconds=60,
        tokenizer_factory=lambda _: Tokenizer(), model_factory=lambda _, k: Encoder(k),
        prediction_sets={'independent_calibration': values}, reuse_application_predictions=True)
    record = receipt['prediction_sets']['independent_calibration']
    assert record['texts_id'] == digest(values)
    assert record['checkpoint_sha256'] == receipt['checkpoint']['sha256']
    with np.load(record['file']['path']) as archive:
        predicted = archive['probabilities']
        assert np.array_equal(predicted[[0, 2]], p[[2, 0]])
        assert np.array_equal(predicted[1], predicted[3])
    assert record['current_checkpoint_prediction_reuse']['application_rows_reused'] == 2
    trace = json.loads((tmp_path/'fit/ACTUAL_TRAINING_TRACE.json').read_text())
    assert set(trace['sampled_rows']) == {0, 1} and receipt['optimizer_updates'] == 1


def test_actual_nonfinite_additional_predictions_fail_without_success_receipt(tmp_path, monkeypatch):
    from tests.test_labeling_actual_aux import assets, Tokenizer, Encoder
    import loris.labeling.trainer as trainer
    source, parent, config = assets(tmp_path)
    observed = np.ones((2,2), bool)
    batch = Batch(np.zeros((2,2), np.float32), observed, observed.astype(np.float32), observed, ~observed, {})
    monkeypatch.setattr(trainer, 'additional_predictions',
                        lambda *a, **kw: (np.full((1,2), np.nan), {}))
    with pytest.raises(RuntimeError, match='additional full-axis'):
        train_actual(texts=['first', 'second'], labels=['A','B'], batch=batch,
            checkpoint=parent, checkpoint_sha256=file_record(parent)['sha256'], source=source,
            configuration=config, device='cpu', output=tmp_path/'fit', gpu_lock=None, max_seconds=60,
            tokenizer_factory=lambda _: Tokenizer(), model_factory=lambda _, k: Encoder(k),
            prediction_sets={'independent_calibration':['extra']}, reuse_application_predictions=True)
    assert (tmp_path/'fit/FAILED.json').exists()
    assert not (tmp_path/'fit/COMPLETE.json').exists()
