"""Public perturbation contracts on real CPU and available CUDA devices."""

import json

import pytest
import torch

from loris.selection import perturbations
from loris.selection.dynamic_router import SelectionNetwork
from loris._learning.selection import PerturbedTopK


WRAPPERS = ("perturbed", "perturbed_special", "perturbed_extended")
DEVICES = ["cpu"] + ["cuda:%d" % index for index in range(torch.cuda.device_count())]


def top_one(scores, scale=1., axis=-1):
    indices = scores.argmax(dim=axis, keepdim=True)
    return torch.zeros_like(scores).scatter_(axis, indices, scale)


def options(name, **extra):
    opts = dict(num_samples=32, sigma=.2, noise="normal", hard_fwd=True)
    if name == "perturbed_extended":
        opts["noise_chunks"] = 3
    return dict(opts, **extra)


def run_gradient(layer, scores, *args):
    value = scores.detach().clone().requires_grad_(True)
    output = layer(value, *args)
    coefficients = torch.arange(output.numel(), device=output.device, dtype=output.dtype)
    (output * coefficients.reshape(output.shape)).sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    return output.detach(), value.grad.detach()


@pytest.mark.parametrize("name", WRAPPERS)
@pytest.mark.parametrize("device", DEVICES)
def test_decorator_preserves_constructor_options_and_exact_seeded_gradient(name, device):
    constructor = getattr(perturbations, name)
    opts = options(name, batched=True, device=device)
    decorator = constructor(**opts)
    assert decorator.func is constructor
    assert decorator.keywords == opts
    scores = torch.tensor([[.1, .3, .2], [.8, .2, .1]], device=device)
    torch.manual_seed(91)
    direct, direct_grad = run_gradient(constructor(top_one, **opts), scores, 2., -1)
    torch.manual_seed(91)
    decorated, decorated_grad = run_gradient(decorator(top_one), scores, 2., -1)
    torch.testing.assert_close(direct, top_one(scores, 2., -1), rtol=0, atol=0)
    torch.testing.assert_close(decorated, direct, rtol=0, atol=0)
    torch.testing.assert_close(decorated_grad, direct_grad, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["perturbed", "perturbed_special"])
def test_soft_forward_decorator_stays_soft_and_matches_direct(name):
    constructor = getattr(perturbations, name)
    opts = options(name, hard_fwd=False, device="cpu", num_samples=100)
    scores = torch.zeros(1, 3)
    torch.manual_seed(123)
    direct, direct_grad = run_gradient(constructor(top_one, **opts), scores)
    torch.manual_seed(123)
    decorated, decorated_grad = run_gradient(constructor(**opts)(top_one), scores)
    assert ((direct > 0) & (direct < 1)).all()
    torch.testing.assert_close(decorated, direct, rtol=0, atol=0)
    torch.testing.assert_close(decorated_grad, direct_grad, rtol=0, atol=0)


@pytest.mark.parametrize("name", WRAPPERS)
def test_required_constant_arguments_reach_hard_forward_and_backward(name):
    def required(scores, scale, axis):
        return top_one(scores, scale, axis)

    scores = torch.tensor([[.8, .2, .1], [.1, .2, .9]])
    layer = getattr(perturbations, name)(required, **options(name))
    result, gradient = run_gradient(layer, scores, 2., -1)
    torch.testing.assert_close(result, required(scores, 2., -1), rtol=0, atol=0)
    assert gradient.shape == scores.shape


@pytest.mark.parametrize("name", WRAPPERS)
@pytest.mark.parametrize("requires_grad", [False, True])
def test_extra_tensor_arguments_are_explicitly_unsupported(name, requires_grad):
    extra = torch.tensor(2., requires_grad=requires_grad)
    layer = getattr(perturbations, name)(top_one, **options(name))
    with pytest.raises(TypeError, match="extra Tensor arguments are unsupported"):
        layer(torch.zeros(1, 3), extra)
    with pytest.raises(TypeError, match="extra Tensor arguments are unsupported"):
        layer(torch.zeros(1, 3), {"nested": [extra]})


@pytest.mark.parametrize("name", WRAPPERS)
@pytest.mark.parametrize("reduce", [False, True])
def test_nonbatched_hard_forward_preserves_vector_or_scalar_shape(name, reduce):
    solver = (lambda scores: scores.max(dim=-1).values) if reduce else top_one
    scores = torch.tensor([.2, .9, .1], requires_grad=True)
    layer = getattr(perturbations, name)(solver, **options(name, batched=False))
    actual = layer(scores)
    expected = solver(scores)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert scores.grad.shape == scores.shape and torch.isfinite(scores.grad).all()


@pytest.mark.parametrize("name", ["perturbed", "perturbed_special"])
@pytest.mark.parametrize("reduce", [False, True])
def test_nonbatched_soft_forward_preserves_vector_or_scalar_shape(name, reduce):
    solver = (lambda scores: scores.max(dim=-1).values) if reduce else top_one
    scores = torch.tensor([.2, .9, .1], requires_grad=True)
    layer = getattr(perturbations, name)(solver, **options(name, batched=False, hard_fwd=False))
    result = layer(scores)
    assert result.shape == solver(scores).shape
    result.sum().backward()
    assert scores.grad.shape == scores.shape and torch.isfinite(scores.grad).all()


def test_extended_contract_is_hard_only_and_preserves_actual_chunk_count(monkeypatch):
    with pytest.raises(ValueError, match="only hard_fwd=True"):
        perturbations.perturbed_extended(top_one, hard_fwd=False)
    with pytest.raises(ValueError, match="only hard_fwd=True"):
        perturbations.perturbed_extended(hard_fwd=False)
    for invalid in (0, -1, 1.5):
        with pytest.raises(ValueError, match="noise_chunks"):
            perturbations.perturbed_extended(noise_chunks=invalid)
    calls = []
    original = perturbations.sample_noise_with_gradients_special

    def record(*args, **kwargs):
        calls.append(kwargs["device"])
        return original(*args, **kwargs)

    monkeypatch.setattr(perturbations, "sample_noise_with_gradients_special", record)
    # The default now describes the implementation's existing hard forward.
    layer = perturbations.perturbed_extended(num_samples=8, noise_chunks=4)(top_one)
    scores = torch.zeros(1, 3, requires_grad=True)
    result = layer(scores)
    assert not calls  # Extended still delays all Monte Carlo work to backward.
    torch.testing.assert_close(result, top_one(scores), rtol=0, atol=0)
    result.sum().backward()
    assert calls == [scores.device] * 4


@pytest.mark.parametrize("name", WRAPPERS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("explicit", [False, True])
def test_sampling_uses_actual_device_and_dtype_including_cpu_on_gpu_host(
    tmp_path, name, device, explicit,
):
    scores = torch.tensor([[.1, .4, .2]], dtype=torch.float64, device=device)
    layer = getattr(perturbations, name)(top_one, **options(
        name, device=device if explicit else None))
    result, gradient = run_gradient(layer, scores)
    assert result.device == gradient.device == scores.device
    assert result.dtype == gradient.dtype == scores.dtype
    (tmp_path / "device_evidence.json").write_text(json.dumps(dict(
        wrapper=name, device=str(scores.device), explicit=explicit,
        cuda_available=torch.cuda.is_available(), cuda_count=torch.cuda.device_count(),
        dtype=str(scores.dtype), output=result.tolist(), gradient=gradient.tolist(),
        backward_finite=bool(torch.isfinite(gradient).all()),
    ), indent=2))


@pytest.mark.parametrize("name", WRAPPERS)
def test_inconsistent_explicit_device_fails_before_noise(name, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("noise must not be sampled after a device mismatch")

    monkeypatch.setattr(perturbations, "sample_noise_with_gradients", forbidden)
    monkeypatch.setattr(perturbations, "sample_noise_with_gradients_special", forbidden)
    layer = getattr(perturbations, name)(top_one, **options(name, device="cuda:0"))
    with pytest.raises(ValueError, match="differs from input device"):
        layer(torch.zeros(1, 3))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_public_perturbed_router_tracks_actual_module_device(device, dtype):
    # device=None must not assume CUDA while nn.Linear parameters are on CPU.
    net = SelectionNetwork(3, 4, 3, 1, num_samples=32, dropout=0., backend="perturbed")
    net.to(device=device, dtype=dtype)
    x = torch.tensor([[.4, .7, .1], [.1, .2, .8]], device=device, dtype=dtype)
    mask = net(x)
    assert mask.device == x.device and mask.dtype == x.dtype and (mask.sum(-1) == 1).all()
    (mask * torch.tensor([.2, .7, -.5], device=device)).sum().backward()
    gradients = [p.grad for p in net.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


@pytest.mark.parametrize("device", DEVICES)
def test_production_selector_remains_independent_of_compatibility_wrappers(device, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("production selector must not call a compatibility wrapper")

    for name in WRAPPERS:
        monkeypatch.setattr(perturbations, name, forbidden)
    scores = torch.tensor([[.4, .1, .2], [.2, .8, .3]], device=device, requires_grad=True)
    mask = PerturbedTopK.apply(scores, 1, 32, .2)
    (mask * torch.tensor([.2, .7, -.5], device=device)).sum().backward()
    assert (mask.sum(-1) == 1).all()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
