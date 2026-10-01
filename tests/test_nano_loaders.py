"""Offline tests of receipted model schemas, not model quality or delivery."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SHAPES = {
    "TinyKhipu-Nano": {"E": (24, 12), "W": (2, 12), "b": (2,), "Wc": (12,)},
    "ReceiptAgent-Nano": {"W1": (16, 24), "b1": (16,), "W2": (8, 16),
                          "b2": (8,), "W3": (4, 8), "b3": (4,)},
}
EXAMPLE = {"query": "ask about F18", "handles": [{"id": "h.a", "note": "F18 knot"}]}


def module(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / name / "load.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def arrays(name):
    rng = np.random.default_rng(9)
    return {key: rng.normal(0, 0.2, shape) for key, shape in SHAPES[name].items()}


@pytest.fixture(params=tuple(SHAPES))
def model(request):
    return request.param, module(request.param)


def call(name, loader, weights):
    return loader.infer(EXAMPLE if name == "TinyKhipu-Nano" else np.zeros(24), weights=weights)


def test_named_archive_binding_is_order_independent(model, tmp_path):
    name, loader = model
    expected = arrays(name)
    archive = tmp_path / "weights.npz"
    np.savez(archive, **dict(reversed(list(expected.items()))))
    actual = loader.load(archive)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert call(name, loader, actual) in (loader.CLASSES if name == "TinyKhipu-Nano" else loader.GATES)


@pytest.mark.parametrize("mutation", ["missing", "extra", "transpose", "wrong_name", "nan", "inf", "complex", "object", "text"])
def test_malformed_archives_fail_closed(model, mutation, tmp_path):
    name, loader = model
    weights = arrays(name)
    key = next(iter(weights))
    if mutation == "missing":
        del weights[key]
    elif mutation == "extra":
        weights["unexpected"] = np.zeros(1)
    elif mutation == "transpose":
        weights[key] = weights[key].T
    elif mutation == "wrong_name":
        weights["renamed"] = weights.pop(key)
    elif mutation in ("nan", "inf"):
        weights[key].flat[0] = float(mutation)
    else:
        dtype = {"complex": np.complex128, "object": object, "text": str}[mutation]
        weights[key] = weights[key].astype(dtype)
    archive = tmp_path / "invalid.npz"
    np.savez(archive, **weights)
    with pytest.raises(ValueError):
        loader.load(archive)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_supplied_weights_cannot_bypass_finite_checks(model, value):
    name, loader = model
    weights = arrays(name)
    weights[next(iter(weights))].flat[0] = value
    with pytest.raises(ValueError):
        call(name, loader, weights)


def test_receipt_agent_matches_named_relu_three_layer_contract():
    loader = module("ReceiptAgent-Nano")
    w = arrays("ReceiptAgent-Nano")
    x = np.linspace(-1, 1, 24)
    h1 = np.maximum(w["W1"] @ x + w["b1"], 0)
    h2 = np.maximum(w["W2"] @ h1 + w["b2"], 0)
    logits = w["W3"] @ h2 + w["b3"]
    p = np.exp(logits - np.max(logits))
    p /= np.sum(p)
    np.testing.assert_allclose(loader.forward(x, weights=w), p, rtol=1e-14, atol=1e-14)
    assert loader.GATES == ("ALLOW", "WARN", "BLOCKED", "ESCALATE")
    assert loader.infer(x, weights=w) == loader.GATES[int(p.argmax())]


@pytest.mark.parametrize("index", range(4))
def test_receipt_class_order_has_no_invented_half_confidence_override(index):
    loader = module("ReceiptAgent-Nano")
    weights = {key: np.zeros(shape) for key, shape in SHAPES["ReceiptAgent-Nano"].items()}
    weights["b3"][index] = 0.1
    assert loader.infer(np.zeros(24), weights=weights) == loader.GATES[index]


@pytest.mark.parametrize("features", [np.zeros(4), np.zeros((1, 24)), ["0"] * 24,
                                      [True] * 24, [0j] * 24, [np.nan] * 24, [np.inf] * 24])
def test_receipt_agent_rejects_invalid_features(features):
    with pytest.raises(ValueError):
        module("ReceiptAgent-Nano").infer(features, weights=arrays("ReceiptAgent-Nano"))


def test_tiny_uses_embedding_class_order_and_offered_citation_scores():
    loader = module("TinyKhipu-Nano")
    w = {key: np.zeros(shape) for key, shape in SHAPES["TinyKhipu-Nano"].items()}
    w["E"][0, 0], w["E"][1, 0] = 2, 1
    w["W"][1, 0], w["Wc"][0] = 1, 1
    example = {"query": "F1", "handles": [{"id": "h.low", "note": "F4"},
                                             {"id": "h.high", "note": "F1"}]}
    result = loader.forward(example, weights=w)
    np.testing.assert_allclose(result["p"], np.array([1, np.exp(2)]) / (1 + np.exp(2)))
    np.testing.assert_allclose(result["cites"], [1, 2])
    assert result["decision"] == 1
    assert result["cited"] == ["h.high"]
    assert result["hallucinated"] == 0
    assert loader.infer(example, weights=w) == "NAVIGATE"
    w["b"][:] = [10, 0]
    assert loader.infer(example, weights=w) == "ABSTAIN"
    assert loader.forward(example, weights=w)["cited"] == []


def test_tiny_preserves_tokenizer_ties_and_empty_text_contract():
    loader = module("TinyKhipu-Nano")
    assert loader.tok_id("F18") == 0
    assert loader.tok_id("F4") == 1
    assert loader.tok_id("") == 16
    assert loader.tok_id("zzzzzzzz") == 16
    weights = {key: np.zeros(shape) for key, shape in SHAPES["TinyKhipu-Nano"].items()}
    result = loader.forward({"query": "", "handles": []}, weights=weights)
    np.testing.assert_array_equal(result["p"], [0.5, 0.5])
    assert result["decision"] == 1  # Raw prediction, not a valid navigation receipt.
    assert result["cited"] == []


@pytest.mark.parametrize("example", [np.zeros(4), {}, {"query": 1, "handles": []},
                                     {"query": "q", "handles": {}},
                                     {"query": "q", "handles": [{"id": "", "note": "x"}]},
                                     {"query": "q", "handles": [{"id": "a", "note": 1}]},
                                     {"query": "q", "handles": [{"id": "a", "note": "x"}] * 2}])
def test_tiny_rejects_malformed_example(example):
    with pytest.raises(ValueError):
        module("TinyKhipu-Nano").infer(example, weights=arrays("TinyKhipu-Nano"))


def test_forward_overflow_fails_closed(model):
    name, loader = model
    weights = {key: np.full(shape, 1e308) for key, shape in SHAPES[name].items()}
    with pytest.raises(ValueError, match="non-finite"):
        call(name, loader, weights)


def test_cli_accepts_actual_input_and_explicit_weights(model, tmp_path):
    name, _loader = model
    weights = tmp_path / "weights.npz"
    np.savez(weights, **arrays(name))
    example = EXAMPLE if name == "TinyKhipu-Nano" else [0] * 24
    result = subprocess.run([sys.executable, str(ROOT / name / "load.py"), json.dumps(example),
                             "--weights", str(weights)], text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() in ("ALLOW", "WARN", "BLOCKED", "ESCALATE", "ABSTAIN", "NAVIGATE")
