import json
import pytest
import torch
from szl_model_lab.catalog import TRACKS
from szl_model_lab.data import Dataset
from szl_model_lab.models import AdvisoryMLP, metrics
from szl_model_lab.training import fit

@pytest.mark.parametrize("track", list(TRACKS))
def test_actual_trainable_parameters_and_gradients(track):
    model = AdvisoryMLP(track)
    assert sum(p.numel() for p in model.parameters()) == 161
    x = torch.rand(4, 8)
    loss = model(x).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

def test_fixture_training_learns_separable_signal(dataset):
    model, result = fit(dataset, epochs=30)
    assert result["n"] == 8
    assert result["false_positives"] == result["false_negatives"] == 0
    assert model.score(dict.fromkeys(TRACKS["router"].features, 0.8)) > model.score(dict.fromkeys(TRACKS["router"].features, 0.2))
    assert result["calibration_validated"] is False

def test_fitting_never_reads_test_split(dataset, monkeypatch):
    original = dataset.select
    def checked(split):
        assert split != "test", "test rows accessed during fitting"
        return original(split)
    monkeypatch.setattr(dataset, "select", checked)
    fit(dataset, epochs=2)

def test_seed_repeats_within_this_environment(dataset):
    a, ma = fit(dataset, epochs=3, seed=22)
    b, mb = fit(dataset, epochs=3, seed=22)
    assert ma == mb
    assert all(torch.equal(a.state_dict()[k], v) for k, v in b.state_dict().items())

def test_rng_is_restored(dataset):
    before = torch.random.get_rng_state().clone()
    fit(dataset, epochs=1)
    assert torch.equal(before, torch.random.get_rng_state())

@pytest.mark.parametrize("epochs", [0, 501, True])
def test_budget_bounds(dataset, epochs):
    with pytest.raises(ValueError, match="budget"):
        fit(dataset, epochs=epochs)

@pytest.mark.parametrize("values", [[2.0]*8, [float("nan")]*8])
def test_bad_tensor(values):
    with pytest.raises(ValueError):
        AdvisoryMLP("router")(torch.tensor([values]))

def test_shape_bounds():
    with pytest.raises(ValueError):
        AdvisoryMLP("router")(torch.rand(3, 7))

def test_count_metrics():
    m = metrics(torch.tensor([2.0, -2.0, 1.0, -1.0]), torch.tensor([1.0, 0.0, 0.0, 1.0]))
    assert all(m[k] == 1 for k in ("true_positives", "false_positives", "true_negatives", "false_negatives"))
    assert m["n"] == 4

def test_invariant_data_adapter(rows):
    for row in rows:
        row["features"] = {n: float(row["label"]) for n in TRACKS["invariant"].features}
    data = Dataset(("\n".join(json.dumps(r) for r in rows)+"\n").encode(), "invariant")
    model, result = fit(data, epochs=3)
    assert model.track.target == "violation_observed" and result["n"] == 8
