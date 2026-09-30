#!/usr/bin/env python3
"""Fail-closed verification for the SZL Hugging Face model/kernel portfolio."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORTFOLIO = ROOT / "portfolio" / "model_portfolio.json"
RECEIPT_FILES = (
    "owner_pubkey.json",
    "training_receipt.signed.json",
    "eval_receipt.signed.json",
)


class PortfolioError(RuntimeError):
    """A portfolio invariant failed."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_source(path: Path) -> str:
    """Hash committed source bytes, avoiding Windows checkout EOL translation."""

    try:
        relative = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return sha256_path(path)
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode == 0:
        return hashlib.sha256(completed.stdout).hexdigest()
    return sha256_path(path)


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def verify_signed_receipts(receipt_dir: Path) -> dict[str, Any]:
    from cryptography.hazmat.primitives.serialization import load_der_public_key

    missing = [name for name in RECEIPT_FILES if not (receipt_dir / name).is_file()]
    if missing:
        raise PortfolioError(
            f"{receipt_dir.relative_to(ROOT)} missing receipt files: {missing}"
        )
    declared_key = json.loads(
        (receipt_dir / "owner_pubkey.json").read_text(encoding="utf-8")
    )
    wrappers: dict[str, dict[str, Any]] = {}
    for name in RECEIPT_FILES[1:]:
        wrapper = json.loads((receipt_dir / name).read_text(encoding="utf-8"))
        expected = canonical_json(wrapper["payload"])
        if wrapper.get("canonical") != expected:
            raise PortfolioError(f"{name}: canonical payload mismatch")
        if wrapper.get("keyId") != declared_key.get("keyId"):
            raise PortfolioError(f"{name}: keyId differs from owner_pubkey.json")
        if wrapper.get("publicKeySpkiBase64") != declared_key.get(
            "publicKeySpkiBase64"
        ):
            raise PortfolioError(f"{name}: public key differs from owner_pubkey.json")
        public_key = load_der_public_key(
            base64.b64decode(wrapper["publicKeySpkiBase64"])
        )
        public_key.verify(
            base64.b64decode(wrapper["signatureBase64"]),
            wrapper["canonical"].encode("utf-8"),
        )
        wrappers[name] = wrapper

    training = wrappers["training_receipt.signed.json"]
    evaluation = wrappers["eval_receipt.signed.json"]
    training_digest = hashlib.sha256(training["canonical"].encode("utf-8")).hexdigest()
    if evaluation["payload"].get("trainingReceiptSha256") != training_digest:
        raise PortfolioError("evaluation receipt does not hash-chain to training")
    if evaluation["payload"].get("weightsArtifactSha256") != training["payload"].get(
        "weightsArtifactSha256"
    ):
        raise PortfolioError("training and evaluation receipts name different weights")
    for filename, expected in training["payload"].get("datasets", {}).items():
        local = receipt_dir / filename
        if not local.is_file():
            raise PortfolioError(f"receipt-bound dataset missing: {local}")
        if sha256_source(local) != expected:
            raise PortfolioError(f"receipt-bound dataset hash mismatch: {local}")

    result = {
        "status": "DECLARED_KEY_SIGNATURES_VALID",
        "key_id": declared_key["keyId"],
        "training_canonical_sha256": training_digest,
        "evaluation_canonical_sha256": hashlib.sha256(
            evaluation["canonical"].encode("utf-8")
        ).hexdigest(),
        "weights_artifact_sha256": training["payload"]["weightsArtifactSha256"],
        "weights_hash_recomputed": False,
        "weights_hash_boundary": (
            "merged weight directories are not stored in Git; their aggregate hash "
            "is signed but cannot be recomputed from this source checkout"
        ),
    }
    for field in (
        "evalTotal",
        "evalContractValid",
        "adversarialTotal",
        "adversarialRefused",
        "planTotal",
        "planValid",
        "groundingTotal",
        "groundingCorrect",
        "abstainTotal",
        "abstainCorrect",
        "hallucinatedCitationCount",
    ):
        if field in evaluation["payload"]:
            result[field] = evaluation["payload"][field]
    return result


def validate_portfolio(document: dict[str, Any]) -> list[str]:
    if document.get("schema") != "szl.model-kernel-portfolio/v1":
        raise PortfolioError("unsupported portfolio schema")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise PortfolioError("portfolio artifacts must be a non-empty list")
    repo_ids = [item.get("repo_id") for item in artifacts]
    if any(not isinstance(repo_id, str) or "/" not in repo_id for repo_id in repo_ids):
        raise PortfolioError("every artifact requires a namespace/repository repo_id")
    if len(repo_ids) != len(set(repo_ids)):
        raise PortfolioError("duplicate repo_id in portfolio")
    allowed = {"trained_model", "quantized_model", "learned_kernel", "software_kernel"}
    for artifact in artifacts:
        if artifact.get("kind") not in allowed:
            raise PortfolioError(f"{artifact['repo_id']}: unsupported kind")
        if artifact.get("autonomy_eligible") is not False:
            raise PortfolioError(
                f"{artifact['repo_id']}: autonomy requires a separate reviewed policy"
            )
        revision = artifact.get("hub_revision")
        if revision is not None and not (
            isinstance(revision, str)
            and len(revision) == 40
            and all(character in "0123456789abcdef" for character in revision)
        ):
            raise PortfolioError(
                f"{artifact['repo_id']}: hub_revision must be lowercase 40-hex"
            )
        expected_weights = artifact.get("expected_weight_sha256", {})
        if not isinstance(expected_weights, dict):
            raise PortfolioError(
                f"{artifact['repo_id']}: expected_weight_sha256 must be an object"
            )
        for filename, digest in expected_weights.items():
            if not isinstance(filename, str) or not (
                isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest)
            ):
                raise PortfolioError(
                    f"{artifact['repo_id']}: invalid expected weight digest"
                )
    return repo_ids


def sibling_record(sibling: Any) -> dict[str, Any]:
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        lfs_sha256 = None
    elif isinstance(lfs, dict):
        lfs_sha256 = lfs.get("sha256")
    else:
        lfs_sha256 = getattr(lfs, "sha256", None)
    return {
        "path": sibling.rfilename,
        "size": getattr(sibling, "size", None),
        "lfs_sha256": lfs_sha256,
    }


def is_auxiliary_gguf(path: str) -> bool:
    """Classify mmproj-named GGUF files by filename only, not inspected contents."""
    name = path.rsplit("/", 1)[-1].lower()
    return name.startswith("mmproj") and name.endswith(".gguf")


def classify_artifact_completeness(
    *,
    revision: str | None,
    inventory_complete: bool | None,
    files: list[dict[str, Any]],
    json_observations: dict[str, dict[str, Any]] | None = None,
    gguf_header_observations: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Describe principal-weight package metadata without I/O or qualification.

    This internal observation API does not change portfolio or model-card schemas.
    Each file is {path, revision, size}; size is an integer or None. Sidecars are
    keyed by exact repository-relative path and contain {revision, state, data?}.
    States are PARSED, ACCESS_DENIED, FETCH_FAILED, PARSE_FAILED, NOT_OBSERVED,
    or TRUNCATED. PARSED data must already be decoded JSON; this function never
    fetches or parses files. Every observation must use the same lowercase
    40-hex commit. Only inventory_complete=True with usable, consistently bound
    file observations establishes path absence.

    Supported indexes are model.safetensors.index.json, pytorch_model.bin.index.json,
    adapter_model.safetensors.index.json and adapter_model.bin.index.json. Their
    nonempty weight_map must reference safe paths relative to the index directory.
    A config must be in that same directory. Nonempty model config objects must
    contain model_type or architectures; adapter config objects must contain
    peft_type or base_model_name_or_path. This recognizes metadata shape only;
    it does not validate a loader's full schema or tensor/config compatibility.

    Optional GGUF header observations use the same envelope with normalized data
    {quantization: QUANTIZED|FLOATING|UNKNOWN, precision?: string}, supplied only
    after an independent header inspection. No GGUF parsing occurs here. Filename
    roles are observational; filenames never prove precision or quantization.
    Self-contained GGUF has no external-config requirement in this bounded check.

    COMPLETE_STRUCTURE means closure of the supplied metadata, not valid weights,
    training, evaluation, rights, promotion, deployment or release approval.
    Unknown lineage and uninspected GGUF quantization remain separate observations.
    A known missing/empty payload can be INCOMPLETE_STRUCTURE; unknown inventory or
    conflicting revisions force overall UNKNOWN_STRUCTURE. No payload is loaded.
    """
    unknown = "UNKNOWN_STRUCTURE"
    incomplete = "INCOMPLETE_STRUCTURE"
    complete = "COMPLETE_STRUCTURE"
    issues: list[dict[str, Any]] = []

    def issue(code: str, path: str | None = None) -> dict[str, Any]:
        record = {"code": code}
        if path is not None:
            record["path"] = path
        return record

    def immutable(value: Any) -> bool:
        return (
            isinstance(value, str) and len(value) == 40
            and all(character in "0123456789abcdef" for character in value)
        )

    def safe_path(value: Any) -> bool:
        return (
            isinstance(value, str) and bool(value)
            and not value.startswith("/") and "\\" not in value and ":" not in value
            and all(ord(character) >= 32 and ord(character) != 127 for character in value)
            and all(part not in {"", ".", ".."} for part in value.split("/"))
        )

    if not immutable(revision):
        issues.append(issue("INVALID_REVISION"))
    if inventory_complete is not True:
        issues.append(issue("INVENTORY_COMPLETENESS_UNPROVEN"))
    records: dict[str, dict[str, Any]] = {}
    roles: list[dict[str, str]] = []
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    ggufs: list[str] = []
    indexes = {
        "model.safetensors.index.json": ("MODEL", ".safetensors"),
        "pytorch_model.bin.index.json": ("MODEL", ".bin"),
        "adapter_model.safetensors.index.json": ("ADAPTER", ".safetensors"),
        "adapter_model.bin.index.json": ("ADAPTER", ".bin"),
    }
    if not isinstance(files, list):
        issues.append(issue("INVALID_INVENTORY_OBSERVATION"))
        files = []
    for row in files:
        if not isinstance(row, dict) or not safe_path(row.get("path")):
            issues.append(issue("INVALID_FILE_OBSERVATION"))
            continue
        path = row["path"]
        if not immutable(row.get("revision")) or row.get("revision") != revision:
            issues.append(issue("FILE_REVISION_MISMATCH", path))
        if path in records:
            issues.append(issue("DUPLICATE_FILE_OBSERVATION", path))
            continue
        records[path] = row
        directory, _, name = path.rpartition("/")
        lower = name.lower()
        kind = None
        role = "OTHER"
        if is_auxiliary_gguf(path):
            role = "AUXILIARY_GGUF_FILENAME"
        elif lower.endswith(".gguf"):
            role = "PRINCIPAL_GGUF_FILENAME"
            ggufs.append(path)
        elif name in indexes:
            kind = indexes[name][0]
            role = "WEIGHT_INDEX_METADATA"
        elif name in {"config.json", "adapter_config.json"}:
            kind = "ADAPTER" if name == "adapter_config.json" else "MODEL"
            role = "CONFIG_METADATA"
        elif lower.endswith((".safetensors", ".bin", ".pt", ".pth")):
            kind = "ADAPTER" if lower.startswith("adapter_model") else "MODEL"
            role = "WEIGHT_PAYLOAD_FILENAME"
        elif lower.endswith(".index.json"):
            role = "UNSUPPORTED_INDEX_METADATA"
            issues.append(issue("UNSUPPORTED_INDEX_SCHEMA", path))
        roles.append({"path": path, "role": role})
        if kind is not None:
            group = groups.setdefault((directory, kind), {
                "directory": directory, "kind": kind, "payloads": [], "indexes": [],
            })
            if role == "WEIGHT_PAYLOAD_FILENAME":
                group["payloads"].append(path)
            elif role == "WEIGHT_INDEX_METADATA":
                group["indexes"].append(path)

    absence_supported = inventory_complete is True and immutable(revision) and not any(
        item["code"] in {
            "INVALID_INVENTORY_OBSERVATION", "INVALID_FILE_OBSERVATION",
            "FILE_REVISION_MISMATCH", "DUPLICATE_FILE_OBSERVATION",
        } for item in issues
    )

    documents = json_observations if json_observations is not None else {}
    headers = gguf_header_observations if gguf_header_observations is not None else {}
    if not isinstance(documents, dict):
        issues.append(issue("INVALID_DOCUMENT_COLLECTION"))
        documents = {}
    if not isinstance(headers, dict):
        issues.append(issue("INVALID_HEADER_COLLECTION"))
        headers = {}
    valid_states = {"PARSED", "ACCESS_DENIED", "FETCH_FAILED", "PARSE_FAILED", "NOT_OBSERVED", "TRUNCATED"}
    for observations in (documents, headers):
        for path, observation in observations.items():
            if not safe_path(path) or not isinstance(observation, dict):
                issues.append(issue("INVALID_DOCUMENT_OBSERVATION"))
                continue
            if not immutable(observation.get("revision")) or observation.get("revision") != revision:
                issues.append(issue("DOCUMENT_REVISION_MISMATCH", path))
            if not isinstance(observation.get("state"), str) or observation.get("state") not in valid_states:
                issues.append(issue("UNKNOWN_OBSERVATION_STATE", path))
            if path not in records:
                issues.append(issue("DOCUMENT_NOT_IN_INVENTORY", path))

    def read_observation(path: str, observations: dict[str, dict[str, Any]], local: list[dict[str, Any]]) -> Any:
        observation = observations.get(path)
        if not isinstance(observation, dict):
            local.append(issue("NOT_OBSERVED", path))
            return None
        if observation.get("revision") != revision or not immutable(observation.get("revision")):
            local.append(issue("DOCUMENT_REVISION_MISMATCH", path))
            return None
        state = observation.get("state")
        if state != "PARSED":
            local.append(issue(state if isinstance(state, str) and state in valid_states else "UNKNOWN_OBSERVATION_STATE", path))
            return None
        metadata_size = records.get(path, {}).get("size")
        if isinstance(metadata_size, int) and not isinstance(metadata_size, bool) and metadata_size == 0:
            local.append(issue("PARSED_EMPTY_METADATA_CONFLICT", path))
            return None
        if metadata_size is not None and (isinstance(metadata_size, bool) or not isinstance(metadata_size, int) or metadata_size < 0):
            local.append(issue("INVALID_METADATA_SIZE", path))
            return None
        data = observation.get("data")
        if not isinstance(data, dict) or not data:
            local.append(issue("UNSUPPORTED_DOCUMENT_SCHEMA", path))
            return None
        return data

    def require_payload(path: str, local: list[dict[str, Any]]) -> None:
        row = records.get(path)
        if row is None:
            local.append(issue("ABSENT_AT_COMPLETE_INVENTORY" if absence_supported else "PRESENCE_UNKNOWN", path))
            return
        size = row.get("size")
        if isinstance(size, int) and not isinstance(size, bool) and size == 0:
            local.append(issue("EMPTY_PAYLOAD", path))
        elif isinstance(size, bool) or not isinstance(size, int) or size < 0:
            local.append(issue("PAYLOAD_SIZE_UNKNOWN" if size is None else "INVALID_PAYLOAD_SIZE", path))

    def status(local: list[dict[str, Any]]) -> str:
        if any(item["code"] in {"ABSENT_AT_COMPLETE_INVENTORY", "EMPTY_PAYLOAD", "NO_PRINCIPAL_PAYLOAD"} for item in local):
            return incomplete
        return unknown if local else complete

    packages: list[dict[str, Any]] = []
    adapters: list[dict[str, Any]] = []
    # A supported parent index owns its safe shard references. A shard-only
    # subdirectory is not a separate variant requiring another config.
    claimed_shards: set[str] = set()
    for (directory, _), group in groups.items():
        prefix = directory + "/" if directory else ""
        for index_path in group["indexes"]:
            index = read_observation(index_path, documents, [])
            weight_map = index.get("weight_map") if index is not None else None
            expected_suffix = indexes[index_path.rsplit("/", 1)[-1]][1]
            if isinstance(weight_map, dict) and weight_map and all(
                isinstance(tensor, str) and bool(tensor)
                and safe_path(shard) and shard.endswith(expected_suffix)
                for tensor, shard in weight_map.items()
            ):
                claimed_shards.update(prefix + shard for shard in weight_map.values())
    for (directory, kind), group in sorted(groups.items()):
        local: list[dict[str, Any]] = []
        prefix = directory + "/" if directory else ""
        config_path = prefix + ("adapter_config.json" if kind == "ADAPTER" else "config.json")
        if (
            not group["indexes"] and config_path not in records and group["payloads"]
            and all(path in claimed_shards for path in group["payloads"])
        ):
            continue
        config = None
        if config_path not in records:
            local.append(issue("ABSENT_AT_COMPLETE_INVENTORY" if absence_supported else "PRESENCE_UNKNOWN", config_path))
        else:
            config = read_observation(config_path, documents, local)
            if config is not None:
                model_shape = (
                    isinstance(config.get("model_type"), str) and bool(config["model_type"].strip())
                ) or (
                    isinstance(config.get("architectures"), list) and bool(config["architectures"])
                    and all(isinstance(item, str) and bool(item.strip()) for item in config["architectures"])
                )
                adapter_shape = (
                    isinstance(config.get("peft_type"), str) and bool(config["peft_type"].strip())
                ) or (
                    "base_model_name_or_path" in config
                    and (config["base_model_name_or_path"] is None or isinstance(config["base_model_name_or_path"], str))
                )
                if not (adapter_shape if kind == "ADAPTER" else model_shape):
                    local.append(issue("UNSUPPORTED_CONFIG_SCHEMA", config_path))
        referenced: set[str] = set()
        for index_path in sorted(group["indexes"]):
            index = read_observation(index_path, documents, local)
            if index is None:
                continue
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                local.append(issue("UNSUPPORTED_INDEX_SCHEMA", index_path))
                continue
            expected_suffix = indexes[index_path.rsplit("/", 1)[-1]][1]
            for tensor, shard in weight_map.items():
                if not isinstance(tensor, str) or not tensor or not safe_path(shard) or not shard.endswith(expected_suffix):
                    local.append(issue("UNSAFE_OR_UNSUPPORTED_SHARD_REFERENCE", index_path))
                    continue
                shard_path = prefix + shard
                referenced.add(shard_path)
                require_payload(shard_path, local)
        for path in group["payloads"]:
            require_payload(path, local)
        if not group["payloads"] and not referenced:
            local.append(issue("NO_PRINCIPAL_PAYLOAD" if absence_supported and not group["indexes"] else "PAYLOAD_CLOSURE_UNKNOWN", prefix or None))
        package = {**group, "config": config_path, "referenced_shards": sorted(referenced), "status": status(local), "issues": local}
        packages.append(package)
        if kind == "ADAPTER":
            base = config.get("base_model_name_or_path") if config is not None else None
            base_revision = config.get("revision") if config is not None else None
            recorded_base = isinstance(base, str) and bool(base.strip())
            adapters.append({
                "directory": directory, "config": config_path,
                "payloads": sorted(set(group["payloads"]) | referenced),
                "base_model_name_or_path": base, "base_revision": base_revision,
                "base_status": "RECORDED" if recorded_base else "UNKNOWN",
                "base_revision_status": "IMMUTABLE_RECORDED" if immutable(base_revision) else "UNKNOWN",
                "lineage_status": "RECORDED" if recorded_base and immutable(base_revision) else "UNKNOWN",
                "compatibility": "NOT_EVALUATED", "structure_status": package["status"],
            })
    quantization: list[dict[str, Any]] = []
    for path in sorted(ggufs):
        local = []
        require_payload(path, local)
        packages.append({"kind": "GGUF", "payloads": [path], "config": None, "status": status(local), "issues": local})
        header_issues: list[dict[str, Any]] = []
        header = read_observation(path, headers, header_issues)
        precision = header.get("precision") if header is not None else None
        quantized = header.get("quantization") if header is not None else "UNKNOWN"
        supported_header = isinstance(quantized, str) and quantized in {"QUANTIZED", "FLOATING", "UNKNOWN"}
        if not isinstance(quantized, str) or quantized not in {"QUANTIZED", "FLOATING", "UNKNOWN"}:
            header_issues.append(issue("UNSUPPORTED_HEADER_SCHEMA", path))
            quantized = "UNKNOWN"
        if precision is not None and not isinstance(precision, str):
            header_issues.append(issue("UNSUPPORTED_HEADER_SCHEMA", path))
            quantized = "UNKNOWN"
            supported_header = False
        if supported_header and isinstance(precision, str) and precision in {"F16", "BF16"}:
            if quantized == "QUANTIZED":
                header_issues.append(issue("CONFLICTING_HEADER_OBSERVATION", path))
                quantized = "UNKNOWN"
            else:
                quantized = "FLOATING"
        quantization.append({"path": path, "status": quantized, "precision": precision, "evidence_scope": "SUPPLIED_HEADER_OBSERVATION" if header is not None else "NOT_OBSERVED", "issues": header_issues})
    if not packages:
        issues.append(issue("NO_PRINCIPAL_PAYLOAD" if absence_supported else "PAYLOAD_CLOSURE_UNKNOWN"))
    known_bases = {item["base_model_name_or_path"] for item in adapters if item["base_status"] == "RECORDED"}
    mixed_bases = True if len(known_bases) > 1 else (None if any(item["base_status"] == "UNKNOWN" for item in adapters) else False)
    package_statuses = {item["status"] for item in packages}
    global_status = (
        unknown if issues and any(item["code"] != "NO_PRINCIPAL_PAYLOAD" for item in issues)
        else incomplete if incomplete in package_statuses or issues
        else unknown if unknown in package_statuses else complete
    )
    return {
        "scope": "principal-weight package metadata closure only",
        "revision": revision, "inventory_complete": inventory_complete,
        "absence_supported": absence_supported,
        "status": global_status, "issues": issues, "file_roles": roles,
        "packages": packages, "adapter_groups": adapters,
        "mixed_adapter_bases": mixed_bases, "gguf_quantization": quantization,
        "artifact_validity": "NOT_EVALUATED", "runtime_validity": "NOT_EVALUATED",
        "limits": "No tensor integrity, index tensor coverage, config/load compatibility, training, evaluation, rights or release qualification assessed.",
    }


def audit_live_artifact(
    artifact: dict[str, Any],
    *,
    api: Any,
    weight_extensions: tuple[str, ...],
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    pinned_revision = artifact.get("hub_revision")
    model_info_args: dict[str, Any] = {"files_metadata": True}
    if pinned_revision:
        model_info_args["revision"] = pinned_revision
    info = api.model_info(artifact["repo_id"], **model_info_args)
    files = [sibling_record(item) for item in info.siblings]
    paths = {item["path"] for item in files}
    missing = sorted(set(artifact.get("required_files", [])) - paths)
    weight_files = [
        item
        for item in files
        if item["path"].lower().endswith(weight_extensions)
        and not is_auxiliary_gguf(item["path"])
    ]
    errors: list[str] = []
    warnings: list[str] = []
    if missing:
        errors.append(f"required files absent: {missing}")
    if pinned_revision and info.sha != pinned_revision:
        errors.append(
            f"resolved revision {info.sha} differs from pin {pinned_revision}"
        )
    license_name = getattr(getattr(info, "card_data", None), "license", None)
    if str(license_name).lower() != "apache-2.0":
        errors.append(f"license is {license_name!r}, expected 'apache-2.0'")
    kind = artifact["kind"]
    if kind in {"trained_model", "quantized_model", "learned_kernel"} and not weight_files:
        errors.append(f"{kind} has no weight artifact")
    if kind == "software_kernel" and weight_files:
        errors.append(
            "software_kernel unexpectedly contains model weight files: "
            + ", ".join(item["path"] for item in weight_files)
        )
    known_weight_bytes = 0
    weight_sizes_known = True
    for item in weight_files:
        size = item["size"]
        if size is None:
            errors.append(f"{item['path']}: weight size is unknown")
            weight_sizes_known = False
        elif isinstance(size, bool) or not isinstance(size, int) or size < 0:
            errors.append(f"{item['path']}: invalid weight size {size!r}")
            weight_sizes_known = False
        elif size == 0:
            errors.append(f"{item['path']}: empty weight artifact")
        else:
            known_weight_bytes += size
    total_weight_bytes = known_weight_bytes if weight_sizes_known else None
    maximum = artifact.get("max_total_weight_bytes")
    if (
        maximum is not None
        and total_weight_bytes is not None
        and total_weight_bytes > int(maximum)
    ):
        errors.append(
            f"weight bytes {total_weight_bytes} exceed declared maximum {maximum}"
        )
    file_records = {item["path"]: item for item in files}
    for filename, expected_sha in artifact.get(
        "expected_weight_sha256",
        {},
    ).items():
        record = file_records.get(filename)
        if record is None:
            errors.append(f"expected weight absent: {filename}")
        elif record.get("lfs_sha256") != expected_sha:
            errors.append(
                f"{filename}: LFS SHA-256 {record.get('lfs_sha256')!r} "
                f"differs from pin {expected_sha}"
            )
    if artifact.get("github_source") is None:
        warnings.append("canonical GitHub source is unbound")

    receipt_parity: dict[str, Any] | None = None
    local_dir = artifact.get("local_receipt_dir")
    if local_dir:
        resolved_revision = info.sha
        valid_revision = (
            isinstance(resolved_revision, str)
            and len(resolved_revision) == 40
            and all(character in "0123456789abcdef" for character in resolved_revision)
        )
        if not valid_revision:
            errors.append("receipt parity requires an exact resolved Hub revision")
        elif pinned_revision and resolved_revision != pinned_revision:
            warnings.append("receipt parity skipped because the resolved Hub revision drifted")
        else:
            local_root = ROOT / local_dir
            receipt_parity = {}
            for name in RECEIPT_FILES:
                remote = Path(
                    hf_hub_download(
                        repo_id=artifact["repo_id"],
                        filename=name,
                        repo_type="model",
                        revision=resolved_revision,
                        force_download=True,
                    )
                )
                local_sha = sha256_source(local_root / name)
                remote_sha = sha256_path(remote)
                receipt_parity[name] = {
                    "local_sha256": local_sha,
                    "remote_sha256": remote_sha,
                    "matched": local_sha == remote_sha,
                }
                if local_sha != remote_sha:
                    errors.append(f"{name}: Hub bytes differ from canonical Git source")

    return {
        "repo_id": artifact["repo_id"],
        "hub_revision": info.sha,
        "kind": kind,
        "maturity": artifact["maturity"],
        "license": license_name,
        "downloads": getattr(info, "downloads", None),
        "files": len(files),
        "weight_files": weight_files,
        "total_weight_bytes": total_weight_bytes,
        "receipt_parity": receipt_parity,
        "metadata_structure": classify_artifact_completeness(
            revision=info.sha,
            inventory_complete=None,
            files=[{**item, "revision": info.sha} for item in files],
        ),
        "ok_scope": "existing catalog suffix/size/pin/license/receipt checks only",
        "artifact_validity": "NOT_EVALUATED",
        "runtime_validity": "NOT_EVALUATED",
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


def build_report(document: dict[str, Any], *, live: bool) -> dict[str, Any]:
    repo_ids = validate_portfolio(document)
    receipt_evidence: dict[str, Any] = {}
    for artifact in document["artifacts"]:
        local_dir = artifact.get("local_receipt_dir")
        if local_dir:
            receipt_evidence[artifact["repo_id"]] = verify_signed_receipts(
                ROOT / local_dir
            )

    live_results: list[dict[str, Any]] = []
    if live:
        from huggingface_hub import HfApi

        api = HfApi()
        extensions = tuple(document["policy"]["model_weight_extensions"])
        for artifact in document["artifacts"]:
            live_results.append(
                audit_live_artifact(
                    artifact,
                    api=api,
                    weight_extensions=extensions,
                )
            )

    errors = [
        f"{result['repo_id']}: {message}"
        for result in live_results
        for message in result["errors"]
    ]
    return {
        "schema": "szl.model-kernel-portfolio-report/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "LIVE_PUBLIC_HUB" if live else "OFFLINE_SOURCE",
        "portfolio_size": len(repo_ids),
        "trained_models": sum(
            item["kind"] == "trained_model" for item in document["artifacts"]
        ),
        "quantized_models": sum(
            item["kind"] == "quantized_model" for item in document["artifacts"]
        ),
        "learned_kernels": sum(
            item["kind"] == "learned_kernel" for item in document["artifacts"]
        ),
        "software_kernels": sum(
            item["kind"] == "software_kernel" for item in document["artifacts"]
        ),
        "receipt_evidence": receipt_evidence,
        "live_artifacts": live_results,
        "errors": errors,
        "ok": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio", default=str(DEFAULT_PORTFOLIO))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    try:
        document = json.loads(Path(args.portfolio).read_text(encoding="utf-8"))
        report = build_report(document, live=args.live)
    except Exception as exc:  # noqa: BLE001 - terminal verifier must emit evidence
        report = {
            "schema": "szl.model-kernel-portfolio-report/v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "fatal": f"{type(exc).__name__}: {exc}",
        }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        output = Path(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
