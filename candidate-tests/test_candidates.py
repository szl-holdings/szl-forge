"""Synthetic structural checks ONLY: not training or model-quality evidence."""
import copy

import pytest
import torch
from safetensors.torch import load, save

from model_candidates.networks import (
    CanalLanguageModel, InvariantRiskModel, RouteUtilityModel, build_candidate,
    canal_mask, risk_loss, route_loss,
)
from model_candidates.specs import candidate_spec, candidate_specs, mesh_declaration


@pytest.fixture(autouse=True)
def reproducible_cpu_scope():
    previous = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(previous)


def test_specs_remain_source_only_and_detached():
    for row in candidate_specs():
        assert row["state"] == "SOURCE_ONLY_NOT_TRAINED"
        assert row["training_run"] is None and row["benchmark"] is None
        assert row["publication_eligible"] is False and row["runtime_eligible"] is False
    row = candidate_spec("router")
    row["config"]["hidden_dim"] = 1000
    assert candidate_spec("router")["config"]["hidden_dim"] == 32
    assert mesh_declaration()["observed_at"] is None
    with pytest.raises(ValueError):
        candidate_spec("../secrets")


def test_router_mask_abstention_and_finite_backward():
    model = RouteUtilityModel()
    features = torch.rand(2, 3, 8)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    proposal = model(features, mask)
    assert torch.isfinite(proposal.distribution).all()
    assert proposal.distribution[0, 1].item() == 0
    assert proposal.distribution[1].sum().item() == 0
    assert proposal.selected_index[1].item() == -1
    assert proposal.selected_index[0].item() in (0, 2)
    with pytest.raises(ValueError):
        route_loss(proposal, torch.tensor([0, 0]))
    mask[1, 2] = True
    loss = route_loss(model(features, mask), torch.tensor([0, 2]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError):
        route_loss(model(features, mask), torch.tensor([1, 2]))


def test_router_is_permutation_equivariant():
    model = RouteUtilityModel().eval()
    features = torch.rand(2, 4, 8)
    mask = torch.tensor([[True, False, True, True], [True, True, False, True]])
    permutation = torch.tensor([2, 0, 3, 1])
    original = model(features, mask)
    permuted = model(features[:, permutation], mask[:, permutation])
    torch.testing.assert_close(permuted.distribution, original.distribution[:, permutation])


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.01, 1.01])
def test_features_reject_nonfinite_and_out_of_range(invalid):
    values = torch.zeros(1, 2, 8)
    values[0, 0, 0] = invalid
    with pytest.raises(ValueError):
        RouteUtilityModel()(values, torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError):
        InvariantRiskModel()(values[:, 0])


@pytest.mark.parametrize("mask", [torch.ones(1, 2), torch.ones(1, 3, dtype=torch.bool)])
def test_router_rejects_ambiguous_eligibility(mask):
    with pytest.raises(ValueError):
        RouteUtilityModel()(torch.rand(1, 2, 8), mask)


def test_empty_and_wrong_features():
    for shape in [(0, 2, 8), (1, 0, 8), (33, 2, 8), (1, 65, 8), (1, 2, 7)]:
        with pytest.raises(ValueError):
            RouteUtilityModel()(torch.zeros(shape), torch.ones(shape[:2], dtype=torch.bool))


def test_risk_backward_and_strict_outcome_labels():
    model = InvariantRiskModel()
    logits = model(torch.rand(3, 8))
    assert logits.shape == (3, 4)
    loss = risk_loss(logits, torch.tensor([[0., 1., 0., 1.]] * 3))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for invalid in [torch.full((3, 4), .5), torch.full((3, 4), float("nan")), torch.ones(3, 4, dtype=torch.long)]:
        with pytest.raises(ValueError):
            risk_loss(logits, invalid)


def test_canal_mask_exact_semantics():
    mask = canal_mask(70, 32, torch.device("cpu"))
    for query in range(70):
        for key in range(70):
            assert mask[query, key].item() == (key <= query and key // 32 == query // 32)
    assert mask.diagonal().all()
    assert not mask.triu(1).any()


@pytest.mark.parametrize("length,width", [(0, 32), (129, 32), (5, 0), (5, 129), (True, 32)])
def test_canal_mask_rejects_invalid_sizes(length, width):
    with pytest.raises(ValueError):
        canal_mask(length, width, torch.device("cpu"))


def test_causal_prefix_and_cross_canal_isolation():
    model = CanalLanguageModel().eval()
    ids = torch.randint(0, 257, (1, 75))
    with torch.no_grad():
        full = model(ids)
        prefix = model(ids[:, :45])
        torch.testing.assert_close(full[:, :45], prefix, atol=2e-6, rtol=2e-6)
        changed = ids.clone()
        changed[:, 46:] = (changed[:, 46:] + 31) % 257
        torch.testing.assert_close(full[:, :46], model(changed)[:, :46], atol=2e-6, rtol=2e-6)
        changed = ids.clone()
        changed[:, :32] = (changed[:, :32] + 1) % 257
        torch.testing.assert_close(full[:, 32:], model(changed)[:, 32:], atol=2e-6, rtol=2e-6)


def test_causal_gradient_and_loss():
    model = CanalLanguageModel()
    ids = torch.arange(70).reshape(1, 70)
    # Output at token 40 must not backpropagate to previous-canal token 10
    # or future same-canal token 45. Token IDs here are unique and disjoint.
    model(ids)[0, 40].square().mean().backward()
    gradient = model.token_embedding.weight.grad
    assert gradient[10].abs().sum() == 0
    assert gradient[45].abs().sum() == 0
    assert gradient[40].abs().sum() > 0
    model.zero_grad(set_to_none=True)
    loss = model.loss(ids)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("ids", [torch.zeros(1, 3), torch.zeros(1, 129, dtype=torch.long),
                                  torch.tensor([[-1, 3]]), torch.tensor([[257, 3]])])
def test_token_contract(ids):
    with pytest.raises(ValueError):
        CanalLanguageModel()(ids)


@pytest.mark.parametrize("key", ["router", "invariant-risk", "yarqa-causal"])
def test_safetensors_roundtrip_is_not_a_release(key):
    model = build_candidate(key).eval()
    initial = copy.deepcopy(model.state_dict())
    data = save(initial, metadata={"state": "RANDOM_INITIALIZATION_TEST_ONLY"})
    restored = build_candidate(key).eval()
    restored.load_state_dict(load(data), strict=True)
    for name, tensor in initial.items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    assert candidate_spec(key)["publication_eligible"] is False


@pytest.mark.parametrize("key", ["router", "invariant-risk", "yarqa-causal"])
def test_nonfinite_parameters_fail_closed(key):
    model = build_candidate(key)
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    with pytest.raises(ValueError):
        if key == "router":
            model(torch.rand(1, 2, 8), torch.ones(1, 2, dtype=torch.bool))
        elif key == "invariant-risk":
            model(torch.rand(1, 8))
        else:
            model(torch.ones(1, 4, dtype=torch.long))


def test_dense_sdpa_matches_independent_masked_matmul(monkeypatch):
    import math
    from model_candidates import networks

    model = CanalLanguageModel().eval()
    ids = torch.randint(0, 257, (2, 49))
    with torch.no_grad():
        expected = model(ids)

    def reference(q, k, v, *, attn_mask, dropout_p):
        assert dropout_p == 0.0
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        return scores.masked_fill(~attn_mask, float("-inf")).softmax(-1) @ v

    monkeypatch.setattr(networks.F, "scaled_dot_product_attention", reference)
    with torch.no_grad():
        torch.testing.assert_close(model(ids), expected, atol=2e-6, rtol=2e-6)


def test_profiles_and_parameter_counts_are_bounded():
    counts = {key: sum(p.numel() for p in build_candidate(key).parameters())
              for key in ("router", "invariant-risk", "yarqa-causal")}
    assert counts == {"router": 1377, "invariant-risk": 1476, "yarqa-causal": 141184}
    assert len(candidate_spec("router")["feature_order"]) == 8
    assert len(candidate_spec("invariant-risk")["targets"]) == 4


@pytest.mark.parametrize("key", ["router", "invariant-risk"])
def test_named_features_are_ordered_and_strict(key):
    from model_candidates.specs import encode_features
    names = candidate_spec(key)["feature_order"]
    row = {name: index / 10 for index, name in reversed(list(enumerate(names)))}
    assert encode_features(key, [row]) == [[index / 10 for index in range(8)]]
    for invalid in [True, None, float("nan"), float("inf"), -1, 2]:
        with pytest.raises(ValueError):
            encode_features(key, [{**row, names[0]: invalid}])
    with pytest.raises(ValueError):
        encode_features(key, [{**row, "extra": .5}])
    with pytest.raises(ValueError):
        encode_features(key, [{name: value for name, value in row.items() if name != names[0]}])
    assert candidate_spec("router")["feature_order"] != candidate_spec("invariant-risk")["feature_order"]
