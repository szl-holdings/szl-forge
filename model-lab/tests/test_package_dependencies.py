"""The installed artifact path must not rely on incidental notebook packages."""
from pathlib import Path
import tomllib

import torch
from safetensors.torch import load, save


def test_numpy_is_a_declared_runtime_dependency():
    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    assert "numpy>=2.3.5,<3" in project["project"]["dependencies"]


def test_safetensors_serialization_works_in_declared_environment():
    # This tiny known vector is serialization evidence, not a trained candidate.
    expected = torch.tensor([[1.0, -2.0, 0.0]], dtype=torch.float32)
    restored = load(save({"serialization_fixture": expected}))
    assert set(restored) == {"serialization_fixture"}
    assert torch.equal(restored["serialization_fixture"], expected)
