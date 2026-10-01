import hashlib
import json
import pytest
import torch
from safetensors.torch import save
from szl_model_lab.artifacts import load_candidate, write_candidate
from szl_model_lab.models import AdvisoryMLP
from szl_model_lab.safeio import canonical_bytes
from szl_model_lab.training import evaluate_test, fit

@pytest.fixture
def artifact(tmp_path, dataset):
    model, result = fit(dataset, epochs=2)
    root = tmp_path / "candidate"
    write_candidate(root, model, source={"verification": "TEST_FIXTURE_ONLY"},
                    dataset=dataset.summary(), validation=result, recipe={"fixture": True})
    return root

def test_round_trip_is_safe_and_unqualified(artifact):
    model, cfg, sha = load_candidate(artifact)
    assert isinstance(model, AdvisoryMLP) and len(sha) == 64
    assert cfg["state"] == "TEST_FIXTURE_ONLY"
    assert cfg["publication_eligible"] is False
    assert not list(artifact.glob("*.joblib")) and not list(artifact.glob("*.pkl"))

def test_digest_tamper_detected(artifact):
    path = artifact / "model.safetensors"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="digest_mismatch"):
        load_candidate(artifact)

def test_manifest_cannot_add_executable_artifact(artifact):
    path = artifact / "manifest.json"
    obj = json.loads(path.read_bytes())
    obj["files"]["model.joblib"] = "0"*64
    path.write_bytes(canonical_bytes(obj))
    with pytest.raises(ValueError, match="manifest"):
        load_candidate(artifact)

def test_incomplete_directory_rejected(artifact):
    (artifact / "manifest.json").unlink()
    with pytest.raises(OSError):
        load_candidate(artifact)

def test_no_overwrite(artifact, dataset):
    with pytest.raises(FileExistsError):
        write_candidate(artifact, AdvisoryMLP("router"), source={}, dataset=dataset.summary(), validation={}, recipe={})

def test_symlink_artifact_rejected(artifact, tmp_path):
    p = tmp_path / "alias"
    try:
        p.symlink_to(artifact, target_is_directory=True)
    except OSError:
        pytest.skip("symlink permission unavailable on this host")
    with pytest.raises(ValueError, match="symlink"):
        load_candidate(p)

def test_same_length_changed_weights_not_trusted(artifact):
    path = artifact / "model.safetensors"
    b = bytearray(path.read_bytes()); b[-1] ^= 1; path.write_bytes(b)
    with pytest.raises(ValueError, match="digest"):
        load_candidate(artifact)

def test_wrong_tensor_shape_rejected_even_with_matching_local_hash(artifact):
    weights = save({"network.0.weight": torch.ones(2, 2)})
    (artifact / "model.safetensors").write_bytes(weights)
    path = artifact / "manifest.json"
    obj = json.loads(path.read_bytes())
    obj["files"]["model.safetensors"] = hashlib.sha256(weights).hexdigest()
    path.write_bytes(canonical_bytes(obj))
    with pytest.raises(ValueError, match="tensors"):
        load_candidate(artifact)

def test_explicit_test_eval_bound_to_identical_data(artifact, tmp_path, raw):
    data = tmp_path / "dataset.jsonl"; data.write_bytes(raw)
    result = evaluate_test(artifact, data)
    assert result["metrics"]["n"] == 8 and result["split"] == "test"
    assert result["publication_eligible"] is False

def test_test_dataset_substitution_rejected(artifact, tmp_path, rows):
    rows[-1]["features"]["input_load"] = 0.4
    p = tmp_path / "other.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    with pytest.raises(ValueError, match="not_the_training_bound"):
        evaluate_test(artifact, p)
