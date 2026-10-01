"""Offline canonical MiniEmbed and Moons contract regressions, not quality scores."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def loader(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / name / "load.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def table():
    return np.random.default_rng(11).normal(size=(64, 12))


def normalize(value, axis):
    return value / np.maximum(np.linalg.norm(value, axis=axis, keepdims=True), 1e-12)


@pytest.mark.parametrize("text", ["knot the run", "F18", "", "  ", "A A B", "caf\u00e9"])
def test_miniembed_matches_canonical_hash_normalization_and_empty_text(text, tmp_path):
    module = loader("MiniEmbed-Nano")
    raw = table()
    path = tmp_path / "mini.npz"
    np.savez(path, table=raw, V=np.array(64), D=np.array(12))
    normalized = normalize(raw, axis=1)
    tokens = text.split()
    ids = [int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little") % 64 for t in tokens]
    expected = normalize(normalized[ids].mean(axis=0) if ids else normalized[0], axis=0)
    np.testing.assert_allclose(module.embed(text, path=path), expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(module.embed(text, table=raw), expected, rtol=1e-14, atol=1e-14)


def test_miniembed_zero_and_tiny_rows_match_epsilon_contract():
    module = loader("MiniEmbed-Nano")
    np.testing.assert_array_equal(module.embed("", table=np.zeros((64, 12))), np.zeros(12))
    raw = np.full((64, 12), 1e-30)
    np.testing.assert_allclose(module.embed("", table=raw), normalize(normalize(raw, 1)[0], 0))


@pytest.mark.parametrize("value", [np.array(0), np.array(63), np.array(64.0), np.array([64]), np.array(True)])
def test_miniembed_rejects_invalid_metadata(value, tmp_path):
    path = tmp_path / "invalid.npz"
    np.savez(path, table=table(), V=value, D=np.array(12))
    with pytest.raises(ValueError):
        loader("MiniEmbed-Nano").load(path)


@pytest.mark.parametrize("mutation", ["shape", "nan", "inf", "complex", "object", "text", "extra", "missing"])
def test_miniembed_rejects_malformed_archives(mutation, tmp_path):
    values = {"table": table()}
    if mutation == "shape":
        values["table"] = values["table"].T
    elif mutation in ("nan", "inf"):
        values["table"][0, 0] = float(mutation)
    elif mutation in ("complex", "object", "text"):
        values["table"] = values["table"].astype({"complex": complex, "object": object, "text": str}[mutation])
    elif mutation == "extra":
        values["other"] = np.zeros(1)
    else:
        del values["table"]
    path = tmp_path / "invalid.npz"
    np.savez(path, **values)
    with pytest.raises(ValueError):
        loader("MiniEmbed-Nano").load(path)


@pytest.mark.parametrize("value", [np.nan, np.inf, 1e308])
def test_miniembed_supplied_table_cannot_bypass_validation(value):
    with pytest.raises(ValueError):
        loader("MiniEmbed-Nano").embed("text", table=np.full((64, 12), value))


def test_miniembed_rejects_nontext():
    with pytest.raises(ValueError):
        loader("MiniEmbed-Nano").embed([1, 2], table=table())


def moon_weights():
    rng = np.random.default_rng(8)
    return tuple(rng.normal(size=shape) for shape in ((8, 2), (8,), (2, 8), (2,)))


def test_moons_matches_canonical_formula_and_named_archive_order(tmp_path):
    module = loader("Moons-Nano")
    weights = moon_weights()
    path = tmp_path / "moons.npz"
    np.savez(path, **dict(reversed(list(zip(("W1", "b1", "W2", "b2"), weights)))))
    logits = weights[2] @ np.tanh(weights[0] @ [0.2, 0.3] + weights[1]) + weights[3]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    label, confidence = module.infer(0.2, 0.3, path=path)
    assert label == int(p.argmax())
    assert confidence == pytest.approx(float(p[label]), rel=1e-14)


@pytest.mark.parametrize("point", [(np.nan, 0), (0, np.inf), ("1", "2"), (1j, 0), ([0], [1])])
def test_moons_invalid_input_never_returns_a_prediction(point):
    with pytest.raises(ValueError):
        loader("Moons-Nano").infer(*point, weights=moon_weights())


@pytest.mark.parametrize("mutation", ["transpose", "nan", "complex", "object", "missing", "extra"])
def test_moons_rejects_invalid_archive(mutation, tmp_path):
    values = dict(zip(("W1", "b1", "W2", "b2"), moon_weights()))
    if mutation == "transpose":
        values["W1"] = values["W1"].T
    elif mutation == "nan":
        values["W1"][0, 0] = np.nan
    elif mutation in ("complex", "object"):
        values["W1"] = values["W1"].astype(complex if mutation == "complex" else object)
    elif mutation == "missing":
        del values["b2"]
    else:
        values["other"] = np.zeros(1)
    path = tmp_path / "invalid.npz"
    np.savez(path, **values)
    with pytest.raises(ValueError):
        loader("Moons-Nano").load(path)


def test_moons_supplied_weights_cannot_bypass_finite_validation():
    weights = moon_weights()
    weights[0][0, 0] = np.nan
    with pytest.raises(ValueError):
        loader("Moons-Nano").infer(0, 0, weights=weights)
