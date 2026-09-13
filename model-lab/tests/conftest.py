"""Synthetic fixtures for contract testing only; never a publishable SZL corpus."""
from __future__ import annotations
import hashlib
import json
import pytest
import torch
from szl_model_lab.catalog import TRACKS
from szl_model_lab.data import Dataset

@pytest.fixture(autouse=True, scope="session")
def bounded_test_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)

@pytest.fixture
def rows():
    result = []
    for split in ("train", "validation", "test"):
        for n in range(8):
            label = n % 2 == 1
            result.append({"example_id": f"{split}-{n}", "group_id": f"group-{split}-{n}",
                "case_sha256": hashlib.sha256(f"case-{split}-{n}".encode()).hexdigest(),
                "split": split, "features": {f: float(0.8 if label else 0.2) for f in TRACKS["router"].features},
                "label": label, "provenance": {"source_id": f"fixture-{split}-{n}",
                    "rights_basis": "test-fixture", "synthetic": True,
                    "feature_time": "2026-09-01T00:00:00Z", "outcome_time": "2026-09-01T00:01:00Z",
                    "normalization_id": "test-unit-interval-v1"}})
    return result

@pytest.fixture
def raw(rows):
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode()

@pytest.fixture
def dataset(raw):
    return Dataset(raw, "router")
