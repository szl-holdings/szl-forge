"""Bounded LoRA identity-claim consistency, not artifact or loader verification.

The declaration and observation are separate caller-supplied claims. Matching
pins do not establish who observed them, current provider state, local weight
bytes, model use by a callback, or independently witnessed adapter application.
This module reads no files, contacts no service, loads no model and grants no
qualification, execution, publication or promotion authority.
"""

from __future__ import annotations

import hashlib
import json
import re

BINDING_SCHEMA = "szl.declared-lora-binding/v1"
SOURCE_REPOSITORY = "szl-holdings/szl-forge"
IDENTITY_FIELDS = frozenset({
    "schema", "artifact_repo_id", "artifact_revision", "artifact_file",
    "artifact_sha256", "base_repo_id", "base_revision", "source_repository",
    "source_revision", "probe_set_sha256", "expected_adapter", "loader_class",
})
GUARD_FIELDS = frozenset({
    "checkpoint_tensors", "applied", "unapplied", "unapplied_sample",
    "checkpoint_layout", "fully_applied", "expected_adapter",
    "adapter_activation", "coverage_basis", "numerical_application",
})
ACTIVATION_FIELDS = frozenset({
    "enabled", "active_adapters", "num_adapter_layers", "available_adapters",
    "merged_adapters",
})
MAX_STRUCTURAL_COUNT = (1 << 31) - 1


def _require(condition: bool, reason: str) -> None:
    if not condition:
        # Fixed reasons only; never echo an untrusted identifier, path or value.
        raise ValueError("BINDING_INVALID: " + reason)


def _digest(value, size: int) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{%d}" % size, value) is not None


def _identifier(value, limit: int = 128) -> bool:
    return (type(value) is str and 0 < len(value) <= limit
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is not None)


def _repo_id(value) -> bool:
    if type(value) is not str or len(value) > 193 or value.count("/") != 1:
        return False
    return all(
        0 < len(part) <= 96 and ".." not in part and "--" not in part
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part) is not None
        and not part.endswith((".", "-"))
        for part in value.split("/")
    )


def _artifact_path(value) -> bool:
    if (type(value) is not str or not 0 < len(value) <= 1024
            or not value.endswith(".safetensors")):
        return False
    return all(
        0 < len(part) <= 128 and part not in {".", ".."}
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part) is not None
        for part in value.split("/")
    )


def _identity(claim: dict) -> None:
    _require(type(claim["schema"]) is str and claim["schema"] == BINDING_SCHEMA,
             "unsupported identity schema")
    _require(_repo_id(claim["artifact_repo_id"]) and _repo_id(claim["base_repo_id"]),
             "invalid repository identity")
    _require(type(claim["source_repository"]) is str
             and claim["source_repository"] == SOURCE_REPOSITORY,
             "noncanonical source repository")
    _require(all(_digest(claim[key], 40) for key in (
        "artifact_revision", "base_revision", "source_revision")),
        "immutable revision required")
    _require(all(_digest(claim[key], 64) for key in (
        "artifact_sha256", "probe_set_sha256")), "exact SHA-256 required")
    _require(_artifact_path(claim["artifact_file"]), "unsafe or unsupported artifact path")
    _require(_identifier(claim["expected_adapter"]) and _identifier(claim["loader_class"]),
             "invalid adapter namespace or loader class")


def _positive_count(value) -> bool:
    return type(value) is int and 0 < value <= MAX_STRUCTURAL_COUNT


def _guard(claim, expected_adapter: str) -> None:
    _require(type(claim) is dict and set(claim) == GUARD_FIELDS,
             "complete structural adapter claim required")
    _require(claim["fully_applied"] is True
             and _positive_count(claim["checkpoint_tensors"])
             and _positive_count(claim["applied"])
             and claim["checkpoint_tensors"] == claim["applied"]
             and type(claim["unapplied"]) is int and claim["unapplied"] == 0
             and type(claim["unapplied_sample"]) is list and claim["unapplied_sample"] == [],
             "incomplete structural checkpoint coverage")
    _require(type(claim["checkpoint_layout"]) is str
             and claim["checkpoint_layout"] in {"text", "language_model"}
             and type(claim["expected_adapter"]) is str
             and claim["expected_adapter"] == expected_adapter
             and type(claim["coverage_basis"]) is str
             and claim["coverage_basis"] == "parameter_names_and_model_status"
             and type(claim["numerical_application"]) is str
             and claim["numerical_application"] == "UNKNOWN",
             "unsupported structural adapter claim")
    activation = claim["adapter_activation"]
    _require(type(activation) is dict and set(activation) == ACTIVATION_FIELDS,
             "complete adapter activation claim required")
    active, available, merged = (activation[key] for key in (
        "active_adapters", "available_adapters", "merged_adapters"))
    _require(activation["enabled"] is True
             and _positive_count(activation["num_adapter_layers"])
             and type(active) is list and len(active) == 1
             and _identifier(active[0]) and active == [expected_adapter]
             and type(available) is list and 0 < len(available) <= 128
             and all(_identifier(name) for name in available)
             and len(set(available)) == len(available) and expected_adapter in available
             and type(merged) is list and merged == [],
             "expected adapter must be enabled active available and unmerged")


def check_binding(*, artifact, probe_sha256, declared, observed) -> dict:
    """Compare strict, paired, LoRA-only claims before a harness callback.

    An absent pair is UNAVAILABLE; a one-sided or malformed pair is invalid.
    A matching pair is DECLARED only. No actual bytes, runtime loader, current
    source/provider revision or observer independence are verified here.
    Returned identity and claim hash cannot be altered by later input mutation.
    """
    boundary = {
        "artifact_bytes_verified": False, "loader_verified": False,
        "observation_independence": "UNKNOWN",
    }
    if declared is None and observed is None:
        return {"status": "UNAVAILABLE", "label": "UNAVAILABLE",
                "identity": None, "claims_sha256": None, **boundary}
    _require(type(declared) is dict and type(observed) is dict,
             "paired identity claims required")
    _require(set(declared) == IDENTITY_FIELDS
             and set(observed) == IDENTITY_FIELDS | {"adapter_admission"},
             "exact identity fields required")
    _identity(declared)
    _identity(observed)
    _require(_repo_id(artifact) and artifact == declared["artifact_repo_id"],
             "run artifact does not match declared identity")
    _require(_digest(probe_sha256, 64) and probe_sha256 == declared["probe_set_sha256"],
             "computed probe digest does not match declared identity")
    _require(all(declared[key] == observed[key] for key in IDENTITY_FIELDS),
             "declared and observed identity claims differ")
    _guard(observed["adapter_admission"], declared["expected_adapter"])
    # Only the strict fields above reach canonicalization. The declaration is
    # flat immutable values; the digest snapshots both complete validated claims.
    identity = dict(declared)
    canonical = json.dumps({"declared": identity, "observed": observed},
                           sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
    return {"status": "DECLARED_BINDING_MATCH", "label": "DECLARED",
            "identity": identity, "claims_sha256": hashlib.sha256(canonical).hexdigest(),
            **boundary}
