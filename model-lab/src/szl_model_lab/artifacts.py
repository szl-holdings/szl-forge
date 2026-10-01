"""Safetensors candidates with byte checks, never executable pickle/joblib files.

Local hashes detect corruption; they are NOT signatures or publication receipts.
The existing Forge publisher must independently authorize any eventual release.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from safetensors import SafetensorError
from safetensors.torch import load, save
from .catalog import track_for
from .models import AdvisoryMLP
from .safeio import canonical_bytes, read_regular, write_new, strict_json

SCHEMA = "szl.model-lab.candidate/v1"
_FILES = {"config.json": 65536, "model.safetensors": 1024 * 1024,
          "metrics.json": 65536, "README.md": 65536}


def write_candidate(root: Path, model: AdvisoryMLP, *, source: dict, dataset: dict,
                    validation: dict, recipe: dict) -> dict:
    root.mkdir(parents=False, exist_ok=False)
    # Models trained in unit tests must never be mistaken for owner-run candidates.
    stage = "TRAINED_UNQUALIFIED" if source.get("verification") == "GIT_CLEAN" else "TEST_FIXTURE_ONLY"
    config = {"schema": SCHEMA, "track": model.track.slug,
              "features": list(model.track.features), "architecture": "tabular-mlp-8x16x1-v1",
              "normalization_id": dataset["normalization_id"], "source": source,
              "dataset": dataset, "recipe": recipe, "state": stage,
              "purpose": "RESEARCH_ADVISORY_ONLY", "publication_eligible": False,
              "runtime_qualified": False, "calibration_validated": False}
    tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if not all(torch.isfinite(v).all() for v in tensors.values()):
        raise ValueError("nonfinite_candidate_weights")
    card = f'''---
tags:
- research
- advisory
- tabular-classification
---
# {model.track.display_name}

State: **{stage}**. This is a small tabular MLP, not an LLM.
Target: `{model.track.target}`. Source revision: `{source.get("revision", "UNVERIFIED")}`.

## Limits

{model.track.boundary}

The output is an uncalibrated sigmoid score, not proven trust or permission.
The validation counts in metrics.json are bounded dataset observations, not a
production qualification. Test evaluation is deliberately separate. No speed,
safety, broad benchmark, or frontier superiority is claimed.

This local candidate is unsigned. Its manifest detects corruption but is not an
authenticity or chain-of-custody attestation. Publication requires the existing
Forge release authority. No Hub upload, promotion, or license grant is implied.
Data rights are operator assertions requiring review before public publication.
Do not replace any existing kernel or signed model with this candidate.
'''
    bodies = {"config.json": canonical_bytes(config), "model.safetensors": save(tensors),
              "metrics.json": canonical_bytes({"split": "validation", "metrics": validation,
                                                "test_evaluated": False}),
              "README.md": card.encode("utf-8")}
    for name, data in bodies.items():
        write_new(root / name, data)
    manifest = {"schema": SCHEMA, "files": {k: hashlib.sha256(v).hexdigest() for k, v in bodies.items()},
                "signed": False, "publication_eligible": False}
    # Last write is the completion marker. Incomplete directories are never loadable.
    write_new(root / "manifest.json", canonical_bytes(manifest))
    return config


def load_candidate(root: Path) -> tuple[AdvisoryMLP, dict, str]:
    manifest_raw = read_regular(root / "manifest.json", 65536)
    manifest = strict_json(manifest_raw)
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "files", "signed", "publication_eligible"}
            or manifest["schema"] != SCHEMA or manifest["signed"] is not False
            or manifest["publication_eligible"] is not False
            or not isinstance(manifest["files"], dict) or set(manifest["files"]) != set(_FILES)):
        raise ValueError("invalid_local_candidate_manifest")
    bodies = {}
    for name, limit in _FILES.items():
        data = read_regular(root / name, limit)
        if hashlib.sha256(data).hexdigest() != manifest["files"][name]:
            raise ValueError("candidate_digest_mismatch")
        bodies[name] = data
    config = strict_json(bodies["config.json"])
    required = {"schema", "track", "features", "architecture", "normalization_id", "source", "dataset", "recipe",
                "state", "purpose", "publication_eligible", "runtime_qualified", "calibration_validated"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("invalid_candidate_config")
    track = track_for(config["track"])
    if (config["schema"] != SCHEMA or config["features"] != list(track.features)
            or config["architecture"] != "tabular-mlp-8x16x1-v1"
            or config["state"] not in ("TRAINED_UNQUALIFIED", "TEST_FIXTURE_ONLY")
            or config["purpose"] != "RESEARCH_ADVISORY_ONLY"
            or any(config[n] is not False for n in ("publication_eligible", "runtime_qualified", "calibration_validated"))):
        raise ValueError("candidate_contract_mismatch")
    model = AdvisoryMLP(track.slug)
    try:
        tensors = load(bodies["model.safetensors"])
    except SafetensorError as exc:
        raise ValueError("invalid_safetensors_serialization") from exc
    expected = model.state_dict()
    if set(tensors) != set(expected) or any(
            tensors[k].shape != expected[k].shape or tensors[k].dtype != torch.float32
            or not torch.isfinite(tensors[k]).all() for k in tensors):
        raise ValueError("invalid_candidate_tensors")
    model.load_state_dict(tensors, strict=True)
    model.eval()
    return model, config, hashlib.sha256(manifest_raw).hexdigest()
