"""Synthetic CPU loss-boundary tests, not a training run or quality benchmark."""
from __future__ import annotations

import copy

import pytest
import torch
from torch.nn import functional as F

from model_candidates import networks

KEYS = ("router", "invariant-risk", "yarqa-causal")
ERROR = "loss must be a finite floating-point scalar"


@pytest.fixture(autouse=True)
def isolated_cpu():
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(417)
            yield
    finally:
        torch.set_num_threads(threads)


def ordinary_case(key, dtype=torch.float32):
    model = networks.build_candidate(key).to(dtype=dtype)
    if key == "router":
        features = torch.rand(2, 3, 8, dtype=dtype)
        eligible = torch.tensor([[True, False, True], [False, True, True]])
        winners = torch.tensor([0, 2])
        proposal = model(features, eligible)
        target = proposal.logits.masked_fill(~eligible, float("-inf"))
        reference = F.cross_entropy(target, winners)
        return model, lambda: networks.route_loss(proposal, winners), reference
    if key == "invariant-risk":
        logits = model(torch.rand(2, 8, dtype=dtype))
        labels = torch.tensor([[0., 1., 0., 1.], [1., 0., 1., 0.]], dtype=dtype)
        reference = F.binary_cross_entropy_with_logits(logits, labels)
        return model, lambda: networks.risk_loss(logits, labels), reference
    ids = torch.randint(0, 257, (2, 35))
    reference_model = copy.deepcopy(model)
    logits = reference_model(ids)
    reference = F.cross_entropy(logits[:, :-1].reshape(-1, 257), ids[:, 1:].reshape(-1))
    # The reference's parameters are separate, allowing an independent backward.
    return (model, reference_model), lambda: model.loss(ids), reference


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_finite_loss_and_gradients_match_unchanged_backend(key, dtype):
    models, evaluate, reference = ordinary_case(key, dtype)
    actual = evaluate()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert actual.dtype == dtype and actual.requires_grad
    if isinstance(models, tuple):
        model, reference_model = models
        actual.backward()
        reference.backward()
        expected = tuple(p.grad for p in reference_model.parameters())
        observed = tuple(p.grad for p in model.parameters())
    else:
        parameters = tuple(models.parameters())
        expected = torch.autograd.grad(reference, parameters, retain_graph=True)
        observed = torch.autograd.grad(actual, parameters)
    for a, b in zip(observed, expected, strict=True):
        assert a is not None and b is not None and torch.isfinite(a).all()
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_every_loss_boundary_rejects_nonfinite_backend_result(key, value, monkeypatch):
    _, evaluate, _ = ordinary_case(key)
    operation = "binary_cross_entropy_with_logits" if key == "invariant-risk" else "cross_entropy"
    monkeypatch.setattr(networks.F, operation, lambda *args, **kwargs: torch.tensor(value))
    with pytest.raises(ValueError, match=ERROR):
        evaluate()


@pytest.mark.parametrize("key", KEYS)
def test_every_loss_boundary_preserves_the_original_tensor(key, monkeypatch):
    _, evaluate, _ = ordinary_case(key)
    leaf = torch.tensor(3., requires_grad=True)
    value = leaf.square()
    operation = "binary_cross_entropy_with_logits" if key == "invariant-risk" else "cross_entropy"
    monkeypatch.setattr(networks.F, operation, lambda *args, **kwargs: value)
    actual = evaluate()
    assert actual is value
    actual.backward()
    assert leaf.grad.item() == 6.


def extreme_case(key):
    """All inputs and real model parameters stay finite and within input bounds."""
    model = networks.build_candidate(key)
    maximum = torch.finfo(torch.float32).max
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        if key == "router":
            model.scorer[0].weight[0, 0] = 1.
            model.scorer[2].weight[0, 0] = 1.
            model.scorer[4].weight[0, 0] = maximum / 2
        elif key == "invariant-risk":
            model.scorer[4].bias.fill_(maximum / 2)
        else:
            model.norm.bias.fill_(1.)
            model.lm_head.weight.fill_(-maximum / 256)
            model.lm_head.weight[0].fill_(maximum / 256)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    if key == "router":
        features = torch.zeros(32, 2, 8)
        features[:, 1, 0] = 1.
        proposal = model(features, torch.ones(32, 2, dtype=torch.bool))
        assert torch.isfinite(proposal.logits).all()
        labels = torch.zeros(32, dtype=torch.long)
        return lambda: networks.route_loss(proposal, labels), F.cross_entropy(proposal.logits, labels)
    if key == "invariant-risk":
        logits = model(torch.zeros(32, 8))
        assert torch.isfinite(logits).all()
        labels = torch.zeros_like(logits)
        return lambda: networks.risk_loss(logits, labels), F.binary_cross_entropy_with_logits(logits, labels)
    ids = torch.ones(2, 16, dtype=torch.long)
    logits = model(ids)
    assert torch.isfinite(logits).all()
    return lambda: model.loss(ids), F.cross_entropy(logits[:, :-1].reshape(-1, 257), ids[:, 1:].reshape(-1))


@pytest.mark.parametrize("key", KEYS)
def test_real_extreme_forward_cannot_return_nonfinite_reduced_loss(key):
    evaluate, unguarded = extreme_case(key)
    # Backend reduction order may differ by platform/version. Neither path skips
    # the test: valid finite results must remain exact; nonfinite results deny.
    if torch.isfinite(unguarded).item():
        torch.testing.assert_close(evaluate(), unguarded, rtol=0, atol=0)
    else:
        with pytest.raises(ValueError, match=ERROR):
            evaluate()


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("value", (torch.ones(2), torch.tensor(1), None))
def test_non_scalar_or_non_float_backend_result_is_not_a_loss(key, value, monkeypatch):
    _, evaluate, _ = ordinary_case(key)
    operation = "binary_cross_entropy_with_logits" if key == "invariant-risk" else "cross_entropy"
    monkeypatch.setattr(networks.F, operation, lambda *args, **kwargs: value)
    with pytest.raises(ValueError, match=ERROR):
        evaluate()
