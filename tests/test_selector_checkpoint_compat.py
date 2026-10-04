"""Actual selector resume plus narrow, explicitly simulated loader API boundaries."""
import inspect
import pickle

import numpy as np
import pytest
import torch

from loris._engine.model_selection import _atomic_checkpoint, _load_selector_checkpoint, _sha_file
from loris._learning.selection import ModelSelectionProtocol, train_model_selection
from loris._learning.selection_validation import verify_model_selection_output


def no_numpy_pickle_objects(value):
    assert not isinstance(value, (np.ndarray, np.generic))
    if isinstance(value, dict):
        for child in value.values():
            no_numpy_pickle_objects(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            no_numpy_pickle_objects(child)


def raw_current_loader(path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    return torch.load(path, **kwargs)


def test_real_selector_training_resume_and_full_artifact_validation(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        truth = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], bool)
        protocol = ModelSelectionProtocol(k=1, epochs=2, hidden_dim=4, num_samples=8,
            sigma=.2, learning_rate=.01, lambda_task=1., lambda_entropy=.01,
            imitation_positive_weight=1., vote_smoothing=.5, dropout=0., seed=7, device="cpu")
        args = dict(features=np.eye(4, dtype=np.float32), decisions=np.stack([truth, ~truth]),
            truth=truth, cluster_assignments=np.zeros(4, np.int64), row_ids=["a", "b", "c", "d"],
            aliases=["left", "right"], labels=["A", "B"],
            input_binding=dict(role="fit", expected_fit_rows=4,
                feature_provenance={"kind": "synthetic"}, decision_provenance={"kind": "synthetic"}),
            protocol=protocol, output_dir=tmp_path / "selector")
        first = train_model_selection(**args)
        checkpoints = sorted((tmp_path / "selector/cluster_00").glob("checkpoint_*.pt"))
        before = [_sha_file(path) for path in checkpoints]
        assert len(checkpoints) == 3 and first["completed_optimizer_steps"] == 2
        for path in checkpoints:
            state = raw_current_loader(path)
            no_numpy_pickle_objects(state)
            assert isinstance(state["scores"], torch.Tensor)
            assert isinstance(state["masks"], torch.Tensor)
        assert train_model_selection(**args) == first
        assert [_sha_file(path) for path in checkpoints] == before
        checked = verify_model_selection_output(tmp_path / "selector")
        assert checked["result"] == first
    finally:
        torch.set_num_threads(previous_threads)


def test_nested_snapshot_dtype_and_optimizer_tensors_round_trip(tmp_path):
    path = tmp_path / "selector.pt"
    net = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(net.parameters())
    net(torch.ones(1, 2)).sum().backward()
    opt.step()
    state = dict(model=net.state_dict(), optimizer=opt.state_dict(),
        rng_state={"cpu": torch.get_rng_state(), "cuda": None},
        selection=dict(scores=np.array([[.2, .8]], np.float32),
                       masks=np.array([[0, 1]], np.float32), frequency=np.array([0, 1], np.int64)))
    _atomic_checkpoint(path, state)
    raw = raw_current_loader(path)
    no_numpy_pickle_objects(raw)
    restored = _load_selector_checkpoint(path, map_location="cpu")
    for key, value in state["selection"].items():
        assert restored["selection"][key].dtype == value.dtype
        np.testing.assert_array_equal(restored["selection"][key], value)
    assert isinstance(restored["rng_state"]["cpu"], torch.Tensor)
    restored_net = torch.nn.Linear(2, 2)
    restored_net.load_state_dict(restored["model"])
    for key, value in net.state_dict().items():
        assert torch.equal(restored_net.state_dict()[key], value)


def test_new_loader_api_is_explicit_weights_only_without_unrestricted_retry(tmp_path, monkeypatch):
    calls = []

    def new_loader(path, *, map_location, weights_only=False):
        calls.append(dict(path=path, map_location=map_location, weights_only=weights_only))
        raise pickle.UnpicklingError("unsupported NumPy global in historical checkpoint")

    monkeypatch.setattr(torch, "load", new_loader)
    with pytest.raises(ValueError, match="regenerate legacy NumPy-pickle"):
        _load_selector_checkpoint(tmp_path / "legacy.pt", map_location="cpu")
    assert len(calls) == 1 and calls[0]["weights_only"] is True


def test_old_loader_api_receives_only_supported_arguments(tmp_path, monkeypatch):
    calls = []

    def old_loader(path, map_location=None, pickle_module=pickle, **pickle_load_args):
        calls.append(pickle_load_args)
        return {"scores": torch.tensor([[.2, .8]]), "masks": torch.tensor([[0., 1.]])}

    monkeypatch.setattr(torch, "load", old_loader)
    state = _load_selector_checkpoint(tmp_path / "internal.pt", map_location="cpu")
    assert calls == [{}]
    assert isinstance(state["scores"], np.ndarray)


def test_new_loader_api_accepts_tensor_snapshots_explicitly(tmp_path, monkeypatch):
    calls = []

    def new_loader(path, *, map_location, weights_only=False):
        calls.append(weights_only)
        return {"scores": torch.tensor([[.2, .8]]), "masks": torch.tensor([[0., 1.]])}

    monkeypatch.setattr(torch, "load", new_loader)
    state = _load_selector_checkpoint(tmp_path / "new.pt", map_location="cpu")
    assert calls == [True]
    np.testing.assert_array_equal(state["masks"], [[0, 1]])
