#!/usr/bin/env python3
"""Publish authorized SZL kernels data using only trusted Forge code."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

EXPECTED_REPO_ID = "SZLHOLDINGS/szl-kernels"
EXPECTED_SOURCE_REPOSITORY = "szl-holdings/szl-kernels"
EXPECTED_PUBLISHER_REPOSITORY = "szl-holdings/szl-forge"
EXPECTED_SOURCE_CHECKS = frozenset(
    {
        "Kernel contract",
        "MiniEmbed artifact replay",
        "Source binding dry run (no publication)",
    }
)
GITHUB_ACTIONS_APP_ID = 15368
EXPECTED_PUBLISHER_WORKFLOW = ".github/workflows/publish-szl-kernels.yml"
EXPECTED_PUBLISHER_WORKFLOW_REF = (
    f"{EXPECTED_PUBLISHER_REPOSITORY}/{EXPECTED_PUBLISHER_WORKFLOW}"
    "@refs/heads/main"
)
EXPECTED_SIGNER_IDENTITY = f"https://github.com/{EXPECTED_PUBLISHER_WORKFLOW_REF}"
EXPECTED_KERNEL_PACKAGE_VERSION = "0.2.0"
KERNEL_RUNTIME_CLIENT_VERSION = "0.16.0"
KERNEL_RUNTIME_IMAGE = f"szl-kernel-runtime:{KERNEL_RUNTIME_CLIENT_VERSION}"
KERNEL_RUNTIME_TIMEOUT_SECONDS = 300
KERNEL_RUNTIME_CONTROL_TIMEOUT_SECONDS = 30
KERNEL_RUNTIME_EVIDENCE_PATH = "/tmp/szl-kernel-runtime-evidence.json"
KERNEL_RUNTIME_LOG_PREFIX = "SZL_KERNEL_RUNTIME_EVIDENCE="
KERNEL_RUNTIME_LOG_LIMIT = 64 * 1024
CONTRACT_RELATIVE = Path("publishing/source-binding.json")
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DOCKER_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
LEGACY_REPO_TYPE = "model"
KERNEL_REPO_TYPE = "kernel"
KERNEL_BRANCHES = ("main", "v1")
KERNEL_VARIANT = "torch-cpu"
KERNEL_VERSION = 1
KERNEL_BUILDER_PACKAGE = "hf-kernel-builder"
KERNEL_BUILDER_VERSION = "0.17.0-dev0"
KERNEL_BUILDER_VERSION_OUTPUT = (
    f"{KERNEL_BUILDER_PACKAGE} {KERNEL_BUILDER_VERSION}"
)
KERNEL_BUILDER_SOURCE_REVISION = (
    "633246310320d85def0c67d62c7912fd444a842f"
)
KERNEL_BINDING_FILENAME = "source-binding.json"
KERNEL_SIGNATURE_FILENAME = "metadata.json.sigstore"
KERNEL_IMMUTABLE_ROOT_FILES = frozenset({".gitattributes", "LICENSE", "README.md"})
KERNEL_SIGNATURE_MANIFEST_SCHEMA = "szl.kernel-signature-transfer/v1"
COSIGN_VERSION = "v3.1.3"
SIGSTORE_OIDC_ISSUER = "https://token.actions.githubusercontent.com"
EXPECTED_WORKFLOW_REF = "refs/heads/main"
EXPECTED_WORKFLOW_TRIGGER = "workflow_dispatch"
COSIGN_TIMEOUT_SECONDS = 120
COSIGN_BUNDLE_MAX_BYTES = 1024 * 1024
SIGNATURE_MANIFEST_MAX_BYTES = 64 * 1024
SUBPROCESS_BASE_ENV_ALLOWLIST = (
    "CI",
    "GITHUB_ACTIONS",
    "HOME",
    "PATH",
    "RUNNER_TEMP",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TEMP",
    "TMP",
    "TMPDIR",
)
OIDC_ENV_ALLOWLIST = (
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
    "ACTIONS_ID_TOKEN_REQUEST_URL",
)
SENSITIVE_ENV_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "AUTH",
    "API_KEY",
    "ACCESS_KEY",
    "PRIVATE_KEY",
    "SIGNING_KEY",
)
FIRST_CLASS_README_SOURCE = "KERNEL_HUB.md"
FIRST_CLASS_KERNEL_FILES = {
    "build/torch-universal/szl_kernels/_kernel_api.py": (
        f"build/{KERNEL_VARIANT}/__init__.py"
    ),
    "build/torch-universal/szl_kernels/_chain.py": (
        f"build/{KERNEL_VARIANT}/_chain.py"
    ),
    "build/torch-universal/szl_kernels/_ops.py": (
        f"build/{KERNEL_VARIANT}/_ops.py"
    ),
    "build/torch-universal/szl_kernels/retrieval.py": (
        f"build/{KERNEL_VARIANT}/retrieval.py"
    ),
}
KERNEL_V1_DECLARED_BUILD_FILES = frozenset(
    {
        f"build/{KERNEL_VARIANT}/{Path(kernel_path).name}"
        for kernel_path in FIRST_CLASS_KERNEL_FILES.values()
    }
    | {
        f"build/{KERNEL_VARIANT}/szl_kernels/{Path(kernel_path).name}"
        for kernel_path in FIRST_CLASS_KERNEL_FILES.values()
    }
    | {
        f"build/{KERNEL_VARIANT}/{KERNEL_BINDING_FILENAME}",
        f"build/{KERNEL_VARIANT}/{KERNEL_SIGNATURE_FILENAME}",
        f"build/{KERNEL_VARIANT}/metadata.json",
    }
)
# These are pre-update checks: an existing v1 need not contain the new operation.
KERNEL_EXISTING_REQUIRED_FILES = {
    ".gitattributes",
    "LICENSE",
    "README.md",
    f"build/{KERNEL_VARIANT}/szl_kernels/__init__.py",
    f"build/{KERNEL_VARIANT}/szl_kernels/_chain.py",
    f"build/{KERNEL_VARIANT}/szl_kernels/_ops.py",
    f"build/{KERNEL_VARIANT}/metadata.json",
}
KERNEL_REQUIRED_FILES_BY_BRANCH = {
    "main": {"README.md"},
    "v1": set(KERNEL_EXISTING_REQUIRED_FILES),
}
FIRST_CLASS_REQUIRED_SOURCE_FILES = {
    FIRST_CLASS_README_SOURCE,
    *FIRST_CLASS_KERNEL_FILES.keys(),
}
KERNEL_REQUIRED_EXPORTS = (
    "UnifiedReceiptChain", "tensor_digest", "GENESIS", "governed_rms_norm",
    "governed_layer_norm", "governed_lambda_gate", "governed_measure_energy",
    "GovernedBlock", "list_kernels", "list_series", "get_member", "selfcheck",
    "DOCTRINE_FOOTER", "PROVENANCE", "__version__", "governed_cosine_topk",
)
RETRIEVAL_SMOKE_QUERY = [1.0, 0.0]
RETRIEVAL_SMOKE_DOCUMENTS = [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]
RETRIEVAL_SMOKE_INDICES = [[1, 2, 0, 3]]
RETRIEVAL_SMOKE_SCORES = [[1.0, 1.0, 0.0, -1.0]]


class PublicationError(RuntimeError):
    """Raised when publication evidence is insufficient."""


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def bounded_error_type(exc: BaseException) -> str:
    """Return a bounded identifier without serializing an exception message."""
    value = re.sub(r"[^A-Za-z0-9_.]", "_", type(exc).__name__)[:128]
    if not value or not (value[0].isalpha() or value[0] == "_"):
        value = f"_{value}"[:128]
    return value


def record_publication_failure(
    result: dict[str, Any],
    report_path: Path,
    *,
    stage: str,
    exc: BaseException,
    provider_write_attempted: bool,
) -> None:
    """Persist a terminal bounded failure state before propagating an error."""
    result["status"] = (
        "PUBLICATION_FAILED_AFTER_PROVIDER_WRITE_ATTEMPT"
        if provider_write_attempted
        else "PUBLICATION_FAILED_NO_PROVIDER_WRITE"
    )
    result["failure"] = {
        "stage": stage,
        "error_type": bounded_error_type(exc),
        "provider_write_attempted": provider_write_attempted,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(canonical_json(result), encoding="utf-8")


def runtime_evidence_from_logs(output: str) -> dict[str, Any] | None:
    if len(output) > KERNEL_RUNTIME_LOG_LIMIT:
        output = output[-KERNEL_RUNTIME_LOG_LIMIT:]
    for line in reversed(output.splitlines()):
        if not line.startswith(KERNEL_RUNTIME_LOG_PREFIX):
            continue
        try:
            evidence = json.loads(line.removeprefix(KERNEL_RUNTIME_LOG_PREFIX))
        except json.JSONDecodeError:
            continue
        if isinstance(evidence, dict):
            return evidence
    return None


def bounded_runtime_log_detail(output: str) -> str:
    return "".join(
        character if 32 <= ord(character) <= 126 else "?"
        for character in output[-2000:]
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_file(source_root: Path, relative: str) -> Path:
    root = source_root.resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents or not path.is_file():
        raise PublicationError(
            f"artifact file is missing or outside the source root: {relative}"
        )
    return path


def load_contract(source_root: Path) -> dict[str, Any]:
    path = safe_file(source_root, CONTRACT_RELATIVE.as_posix())
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "szl.kernel-source-binding/v1":
        raise PublicationError("unsupported source-binding schema")
    if payload.get("repo_id") != EXPECTED_REPO_ID:
        raise PublicationError("source contract cannot select another Hub repository")
    if payload.get("source_repository") != EXPECTED_SOURCE_REPOSITORY:
        raise PublicationError("source contract names an unexpected repository")
    artifact_files = payload.get("artifact_files")
    if (
        not isinstance(artifact_files, list)
        or not artifact_files
        or any(not isinstance(item, str) or not item for item in artifact_files)
        or len(artifact_files) != len(set(artifact_files))
    ):
        raise PublicationError("artifact_files must be a unique non-empty string list")
    missing_kernel_sources = FIRST_CLASS_REQUIRED_SOURCE_FILES - set(artifact_files)
    if missing_kernel_sources:
        raise PublicationError(
            "artifact_files must declare first-class Kernel source inputs: "
            f"{sorted(missing_kernel_sources)}"
        )
    expected = payload.get("expected_artifact_sha256")
    if not isinstance(expected, dict) or not expected:
        raise PublicationError("expected_artifact_sha256 must be a non-empty object")
    if not set(expected).issubset(set(artifact_files)):
        raise PublicationError("expected hashes must name declared artifact files")
    return payload


def load_authorization(
    path: Path,
    *,
    source_revision: str,
    publisher_revision: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "szl.kernels-release-authorization/v1":
        raise PublicationError("unsupported release authorization schema")
    if payload.get("status") != "AUTHORIZED_PROTECTED_MAIN":
        raise PublicationError("source release is not authorized")
    source = payload.get("source", {})
    publisher = payload.get("publisher", {})
    if (
        source.get("repository") != EXPECTED_SOURCE_REPOSITORY
        or source.get("revision") != source_revision
        or source.get("protected_main") != source_revision
        or source.get("branch_protection_observed") is not True
        or source.get("signature_verified") is not True
    ):
        raise PublicationError("source authorization does not bind the requested revision")
    if (
        publisher.get("repository") != EXPECTED_PUBLISHER_REPOSITORY
        or publisher.get("revision") != publisher_revision
        or publisher.get("protected_main") != publisher_revision
        or publisher.get("branch_protection_observed") is not True
    ):
        raise PublicationError("publisher authorization does not bind this Forge revision")
    checks = source.get("checks")
    if (
        type(checks) is not list
        or len(checks) != len(EXPECTED_SOURCE_CHECKS)
        or any(
            type(item) is not dict or type(item.get("name")) is not str
            for item in checks
        )
        or {item["name"] for item in checks} != EXPECTED_SOURCE_CHECKS
    ):
        raise PublicationError("source authorization required checks are incomplete")
    if any(
        type(item) is not dict
        or item.get("app_id") != GITHUB_ACTIONS_APP_ID
        or item.get("status") != "completed"
        or item.get("conclusion") != "success"
        for item in checks
    ):
        raise PublicationError("source authorization checks are not successful")
    return payload


def authorization_binding(payload: dict[str, Any]) -> dict[str, Any]:
    """Project fresh authorization into deterministic exact-head facts."""
    source = payload["source"]
    publisher = payload["publisher"]
    checks = sorted(source["checks"], key=lambda item: item["name"])
    return {
        "schema": "szl.kernels-release-authorization-binding/v1",
        "status": payload["status"],
        "source": {
            "repository": source["repository"],
            "revision": source["revision"],
            "protected_main": source["protected_main"],
            "branch_protection_observed": True,
            "signature_verified": True,
            "required_checks": [
                {
                    "name": item["name"],
                    "app_id": item["app_id"],
                    "status": item["status"],
                    "conclusion": item["conclusion"],
                }
                for item in checks
            ],
        },
        "publisher": {
            "repository": publisher["repository"],
            "revision": publisher["revision"],
            "protected_main": publisher["protected_main"],
            "branch_protection_observed": True,
        },
    }


def local_evidence(
    source_root: Path,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    expected = contract["expected_artifact_sha256"]
    evidence: list[dict[str, Any]] = []
    for relative in contract["artifact_files"]:
        path = safe_file(source_root, relative)
        observed = file_sha256(path)
        wanted = expected.get(relative)
        if wanted is not None and observed != wanted:
            raise PublicationError(
                f"{relative} SHA-256 drifted (expected {wanted}, observed {observed})"
            )
        evidence.append(
            {
                "path": Path(relative).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": observed,
            }
        )
    return evidence


def tree_sha256(evidence: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(evidence, key=lambda value: value["path"]):
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def legacy_model_before(
    api: HfApi,
    contract: dict[str, Any],
    *,
    token: str | None,
    download_fn: Callable[..., str],
) -> dict[str, Any]:
    info = api.model_info(EXPECTED_REPO_ID, files_metadata=True, token=token)
    observed_files = {sibling.rfilename for sibling in info.siblings or []}
    missing_critical = sorted(
        set(contract["expected_artifact_sha256"]) - observed_files
    )
    if missing_critical:
        raise PublicationError(
            f"Hub artifact is missing critical files: {missing_critical}"
        )
    critical = []
    for relative, wanted in sorted(contract["expected_artifact_sha256"].items()):
        downloaded = Path(
            download_fn(
                EXPECTED_REPO_ID,
                relative,
                repo_type=LEGACY_REPO_TYPE,
                revision=info.sha,
                token=token,
            )
        )
        observed = file_sha256(downloaded)
        if observed != wanted:
            raise PublicationError(
                f"Hub {relative} SHA-256 drifted "
                f"(expected {wanted}, observed {observed})"
            )
        critical.append({"path": relative, "sha256": observed})
    return {
        "revision": info.sha,
        "declared_files_present": len(
            set(contract["artifact_files"]) & observed_files
        ),
        "critical_artifacts": critical,
    }


def first_class_kernel_before(
    api: HfApi,
    contract: dict[str, Any],
    *,
    token: str | None,
) -> dict[str, Any]:
    declared = set(contract["artifact_files"])
    missing_sources = sorted(set(FIRST_CLASS_KERNEL_FILES) - declared)
    if missing_sources:
        raise PublicationError(
            "source contract is missing first-class Kernel files: "
            f"{missing_sources}"
        )

    info = api.repo_info(
        EXPECTED_REPO_ID,
        repo_type=KERNEL_REPO_TYPE,
        files_metadata=True,
        token=token,
    )
    refs = api.list_repo_refs(
        EXPECTED_REPO_ID,
        repo_type=KERNEL_REPO_TYPE,
        token=token,
    )
    branches = {branch.name: branch.target_commit for branch in refs.branches}
    missing_branches = sorted(set(KERNEL_BRANCHES) - set(branches))
    if missing_branches:
        raise PublicationError(
            f"first-class Kernel is missing release branches: {missing_branches}"
        )

    branch_evidence: dict[str, Any] = {}
    for branch in KERNEL_BRANCHES:
        expected_paths = KERNEL_REQUIRED_FILES_BY_BRANCH[branch]
        target = branches[branch]
        observed_files = set(
            api.list_repo_files(
                EXPECTED_REPO_ID,
                repo_type=KERNEL_REPO_TYPE,
                revision=target,
                token=token,
            )
        )
        missing = sorted(expected_paths - observed_files)
        if missing:
            raise PublicationError(
                f"first-class Kernel {branch} is missing package files: {missing}"
            )
        if branch == "v1":
            allowed_files = KERNEL_IMMUTABLE_ROOT_FILES | KERNEL_V1_DECLARED_BUILD_FILES
            undeclared_files = sorted(observed_files - allowed_files)
            if undeclared_files:
                raise PublicationError(
                    "first-class Kernel v1 contains undeclared immutable files before "
                    f"publication: {undeclared_files}"
                )
        branch_evidence[branch] = {
            "revision": target,
            "package_files_present": len(expected_paths & observed_files),
        }
    return {
        "repo_revision": info.sha,
        "branches": branch_evidence,
    }


def hub_before(
    api: HfApi,
    contract: dict[str, Any],
    *,
    token: str | None,
    download_fn: Callable[..., str],
) -> dict[str, Any]:
    return {
        "legacy_model": legacy_model_before(
            api,
            contract,
            token=token,
            download_fn=download_fn,
        ),
        "first_class_kernel": first_class_kernel_before(
            api,
            contract,
            token=token,
        ),
    }


def publisher_identity(
    *,
    repository: str,
    revision: str,
    workflow_ref: str,
    run_id: str,
    run_attempt: str,
) -> dict[str, Any]:
    if repository != EXPECTED_PUBLISHER_REPOSITORY:
        raise PublicationError("unexpected publisher repository")
    if FULL_SHA_RE.fullmatch(revision) is None:
        raise PublicationError("publisher revision must be an exact Git SHA")
    if not run_id.isdigit() or not run_attempt.isdigit():
        raise PublicationError("publisher run identity is malformed")
    workflow_path = EXPECTED_PUBLISHER_WORKFLOW
    if workflow_ref != EXPECTED_PUBLISHER_WORKFLOW_REF:
        raise PublicationError("publisher workflow must run from protected main")
    return {
        "repository": repository,
        "revision": revision,
        "workflow_path": workflow_path,
        "workflow_ref": workflow_ref,
        "certificate_identity": EXPECTED_SIGNER_IDENTITY,
        "workflow_url": (
            f"https://github.com/{repository}/blob/{revision}/{workflow_path}"
        ),
        "run_id": run_id,
        "run_attempt": run_attempt,
        "run_url": f"https://github.com/{repository}/actions/runs/{run_id}",
    }


def verify_legacy_readback(
    source_root: Path,
    contract: dict[str, Any],
    publication_bytes: bytes,
    *,
    revision: str,
    token: str,
    download_fn: Callable[..., str],
) -> None:
    for relative in list(contract["artifact_files"]) + ["publication.json"]:
        downloaded = Path(
            download_fn(
                EXPECTED_REPO_ID,
                relative,
                repo_type=LEGACY_REPO_TYPE,
                revision=revision,
                token=token,
            )
        )
        expected = (
            publication_bytes
            if relative == "publication.json"
            else safe_file(source_root, relative).read_bytes()
        )
        if downloaded.read_bytes() != expected:
            raise PublicationError(f"readback mismatch at {relative}")


def kernel_file_evidence(
    source_root: Path,
) -> list[dict[str, Any]]:
    readme = safe_file(source_root, FIRST_CLASS_README_SOURCE)
    evidence = [
        {
            "source_path": FIRST_CLASS_README_SOURCE,
            "kernel_path": "README.md",
            "bytes": readme.stat().st_size,
            "sha256": file_sha256(readme),
        }
    ]
    for source_path, kernel_path in FIRST_CLASS_KERNEL_FILES.items():
        path = safe_file(source_root, source_path)
        destinations = (
            kernel_path,
            f"build/{KERNEL_VARIANT}/szl_kernels/{Path(kernel_path).name}",
        )
        for destination in destinations:
            evidence.append(
                {
                    "source_path": source_path,
                    "kernel_path": destination,
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return evidence


def digest_base64(payload: bytes) -> str:
    return base64.b64encode(hashlib.sha256(payload).digest()).decode("ascii")


def stage_first_class_kernel(
    source_root: Path,
    binding_bytes: bytes,
    staging_root: Path,
) -> dict[str, dict[str, bytes]]:
    """Create a standards-compliant tree for the pinned kernel-builder."""
    build_root = staging_root / "build"
    variant_root = build_root / KERNEL_VARIANT
    compatibility_root = variant_root / "szl_kernels"
    compatibility_root.mkdir(parents=True, exist_ok=True)

    expected: dict[str, dict[str, bytes]] = {
        "main": {
            "README.md": safe_file(source_root, FIRST_CLASS_README_SOURCE).read_bytes(),
        },
        "v1": {},
    }
    digest_files: dict[str, str] = {}
    for source_path, kernel_path in FIRST_CLASS_KERNEL_FILES.items():
        payload = safe_file(source_root, source_path).read_bytes()
        filename = Path(kernel_path).name
        for relative in (filename, f"szl_kernels/{filename}"):
            destination = variant_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            repository_path = f"build/{KERNEL_VARIANT}/{relative}"
            expected["v1"][repository_path] = payload
            digest_files[relative] = digest_base64(payload)

    binding_path = variant_root / KERNEL_BINDING_FILENAME
    binding_path.write_bytes(binding_bytes)
    expected["v1"][
        f"build/{KERNEL_VARIANT}/{KERNEL_BINDING_FILENAME}"
    ] = binding_bytes
    digest_files[KERNEL_BINDING_FILENAME] = digest_base64(binding_bytes)

    binding_sha = hashlib.sha256(binding_bytes).hexdigest()
    metadata = {
        "name": "szl-kernels",
        "id": f"_szl_kernels_cpu_{binding_sha[:8]}",
        "version": KERNEL_VERSION,
        "license": "Apache-2.0",
        "python-depends": [],
        "backend": {"type": "cpu"},
        "digest": {
            "algorithm": "sha256",
            "files": digest_files,
        },
    }
    metadata_bytes = canonical_json(metadata).encode("utf-8")
    metadata_path = variant_root / "metadata.json"
    metadata_path.write_bytes(metadata_bytes)
    expected["v1"][f"build/{KERNEL_VARIANT}/metadata.json"] = metadata_bytes

    (build_root / "CARD.md").write_bytes(expected["main"]["README.md"])
    return expected


def _subprocess_base_environment() -> dict[str, str]:
    """Return a credentialless environment for pinned helper processes."""
    return {
        key: os.environ[key]
        for key in SUBPROCESS_BASE_ENV_ALLOWLIST
        if key in os.environ
    }


def _cosign_signing_environment() -> dict[str, str]:
    """Add GitHub OIDC authority only for the single signing subprocess."""
    environment = _subprocess_base_environment()
    missing = [key for key in OIDC_ENV_ALLOWLIST if not os.environ.get(key)]
    if missing:
        raise PublicationError("GitHub OIDC signing authority is unavailable")
    environment.update({key: os.environ[key] for key in OIDC_ENV_ALLOWLIST})
    return environment


def require_cosign_executable() -> str:
    executable = shutil.which("cosign")
    if executable is None:
        raise PublicationError("pinned cosign is not installed")
    try:
        observed = subprocess.run(
            [executable, "version", "--json"],
            check=False,
            capture_output=True,
            text=True,
            env=_subprocess_base_environment(),
            timeout=COSIGN_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PublicationError("unable to verify the pinned cosign version") from exc
    try:
        payload = json.loads(observed.stdout)
    except json.JSONDecodeError as exc:
        raise PublicationError("cosign returned malformed version evidence") from exc
    version = payload.get("gitVersion") if isinstance(payload, dict) else None
    if observed.returncode != 0 or version != COSIGN_VERSION:
        raise PublicationError(
            f"cosign version drifted (expected {COSIGN_VERSION!r}, "
            f"observed {version!r})"
        )
    return executable


def _kernel_signature_evidence(
    staging_root: Path,
    *,
    certificate_identity: str,
) -> tuple[dict[str, Any], bytes]:
    variant_root = staging_root / "build" / KERNEL_VARIANT
    metadata_path = variant_root / "metadata.json"
    bundle_path = variant_root / KERNEL_SIGNATURE_FILENAME
    try:
        bundle_size = bundle_path.stat().st_size
        if bundle_path.is_symlink() or not (0 < bundle_size <= COSIGN_BUNDLE_MAX_BYTES):
            raise PublicationError("kernel signature bundle failed local validation")
        bundle_bytes = bundle_path.read_bytes()
        bundle = json.loads(bundle_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError("kernel signature bundle failed local validation") from exc
    if (
        certificate_identity != EXPECTED_SIGNER_IDENTITY
        or not metadata_path.is_file()
        or not isinstance(bundle, dict)
        or len(bundle_bytes) != bundle_size
    ):
        raise PublicationError("kernel signature bundle failed local validation")
    return (
        {
            "status": "SIGNED_AND_IDENTITY_VERIFIED",
            "tool": "cosign",
            "tool_version": COSIGN_VERSION,
            "certificate_identity": certificate_identity,
            "oidc_issuer": SIGSTORE_OIDC_ISSUER,
            "metadata_sha256": file_sha256(metadata_path),
            "bundle_path": f"build/{KERNEL_VARIANT}/{KERNEL_SIGNATURE_FILENAME}",
            "bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
            "bundle_bytes": len(bundle_bytes),
        },
        bundle_bytes,
    )


def sign_kernel_metadata(
    staging_root: Path,
    *,
    certificate_identity: str,
    publisher_revision: str,
) -> dict[str, Any]:
    """Keylessly sign staged metadata and verify the exact workflow identity."""
    if certificate_identity != EXPECTED_SIGNER_IDENTITY:
        raise PublicationError("kernel signer identity is not the protected publisher workflow")
    variant_root = staging_root / "build" / KERNEL_VARIANT
    metadata_path = variant_root / "metadata.json"
    bundle_path = variant_root / KERNEL_SIGNATURE_FILENAME
    if not metadata_path.is_file():
        raise PublicationError("staged kernel metadata is missing")
    if bundle_path.exists():
        raise PublicationError("staged kernel signature bundle already exists")

    if FULL_SHA_RE.fullmatch(publisher_revision) is None:
        raise PublicationError("kernel signer revision is not an exact Git SHA")
    executable = require_cosign_executable()
    signing_environment = _cosign_signing_environment()
    try:
        signed = subprocess.run(
            [
                executable,
                "sign-blob",
                "--yes",
                "--bundle",
                str(bundle_path),
                str(metadata_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=signing_environment,
            timeout=COSIGN_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PublicationError("kernel metadata signing did not complete") from exc
    if signed.returncode != 0:
        # Cosign diagnostics are intentionally not propagated: they are outside
        # this publisher's schema and could contain ambient runner details.
        raise PublicationError("kernel metadata signing failed")
    verify_kernel_metadata_signature(
        staging_root,
        certificate_identity=certificate_identity,
        publisher_revision=publisher_revision,
        executable=executable,
    )
    evidence, _ = _kernel_signature_evidence(
        staging_root,
        certificate_identity=certificate_identity,
    )
    evidence["publisher_revision"] = publisher_revision
    return evidence


def verify_kernel_metadata_signature(
    staging_root: Path,
    *,
    certificate_identity: str,
    publisher_revision: str,
    executable: str | None = None,
) -> None:
    """Verify the bundle without giving Cosign OIDC or provider credentials."""
    if certificate_identity != EXPECTED_SIGNER_IDENTITY:
        raise PublicationError("kernel signer identity is not the protected publisher workflow")
    if FULL_SHA_RE.fullmatch(publisher_revision) is None:
        raise PublicationError("kernel signer revision is not an exact Git SHA")
    variant_root = staging_root / "build" / KERNEL_VARIANT
    metadata_path = variant_root / "metadata.json"
    bundle_path = variant_root / KERNEL_SIGNATURE_FILENAME
    executable = executable or require_cosign_executable()

    try:
        verified = subprocess.run(
            [
                executable,
                "verify-blob",
                str(metadata_path),
                "--bundle",
                str(bundle_path),
                "--certificate-identity",
                certificate_identity,
                "--certificate-oidc-issuer",
                SIGSTORE_OIDC_ISSUER,
                "--certificate-github-workflow-repository",
                EXPECTED_PUBLISHER_REPOSITORY,
                "--certificate-github-workflow-ref",
                EXPECTED_WORKFLOW_REF,
                "--certificate-github-workflow-sha",
                publisher_revision,
                "--certificate-github-workflow-trigger",
                EXPECTED_WORKFLOW_TRIGGER,
            ],
            check=False,
            capture_output=True,
            text=True,
            env=_subprocess_base_environment(),
            timeout=COSIGN_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PublicationError("kernel metadata signature verification did not complete") from exc
    if verified.returncode != 0:
        raise PublicationError("kernel metadata signature verification failed")


def validate_kernel_signature_evidence(
    evidence: Any,
    *,
    staging_root: Path,
    certificate_identity: str,
    publisher_revision: str,
) -> bytes:
    """Bind reported signature evidence to the staged files before upload."""
    expected, bundle_bytes = _kernel_signature_evidence(
        staging_root,
        certificate_identity=certificate_identity,
    )
    expected["publisher_revision"] = publisher_revision
    if type(evidence) is not dict or evidence != expected:
        raise PublicationError("kernel signature evidence failed validation")
    return bundle_bytes


def write_kernel_signature_transfer(
    *,
    staging_root: Path,
    signature: dict[str, Any],
    bundle_output: Path,
    manifest_output: Path,
    source_revision: str,
    publisher_revision: str,
    binding_sha256: str,
) -> dict[str, Any]:
    """Write the only data allowed to cross from the OIDC job to publish."""
    bundle_bytes = validate_kernel_signature_evidence(
        signature,
        staging_root=staging_root,
        certificate_identity=EXPECTED_SIGNER_IDENTITY,
        publisher_revision=publisher_revision,
    )
    manifest = {
        "schema": KERNEL_SIGNATURE_MANIFEST_SCHEMA,
        "source_revision": source_revision,
        "publisher_revision": publisher_revision,
        "binding_sha256": binding_sha256,
        "signature": signature,
    }
    bundle_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with bundle_output.open("xb") as stream:
            stream.write(bundle_bytes)
        with manifest_output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(manifest))
    except FileExistsError as exc:
        raise PublicationError("kernel signature transfer output already exists") from exc
    return manifest


def consume_kernel_signature_transfer(
    *,
    staging_root: Path,
    bundle_input: Path,
    manifest_input: Path,
    source_revision: str,
    publisher_revision: str,
    binding_sha256: str,
    verify_fn: Callable[..., None] = verify_kernel_metadata_signature,
) -> dict[str, Any]:
    """Validate and consume a signed bundle in the credential-only job."""
    try:
        bundle_size = bundle_input.stat().st_size
        manifest_size = manifest_input.stat().st_size
        if (
            bundle_input.is_symlink()
            or manifest_input.is_symlink()
            or not (0 < bundle_size <= COSIGN_BUNDLE_MAX_BYTES)
            or not (0 < manifest_size <= SIGNATURE_MANIFEST_MAX_BYTES)
        ):
            raise PublicationError("kernel signature transfer is invalid")
        bundle_bytes = bundle_input.read_bytes()
        manifest_bytes = manifest_input.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError("kernel signature transfer is unreadable") from exc
    if (
        type(manifest) is not dict
        or set(manifest)
        != {
            "schema",
            "source_revision",
            "publisher_revision",
            "binding_sha256",
            "signature",
        }
        or manifest["schema"] != KERNEL_SIGNATURE_MANIFEST_SCHEMA
        or manifest["source_revision"] != source_revision
        or manifest["publisher_revision"] != publisher_revision
        or manifest["binding_sha256"] != binding_sha256
        or len(bundle_bytes) != bundle_size
        or len(manifest_bytes) != manifest_size
    ):
        raise PublicationError("kernel signature transfer failed binding validation")
    bundle_target = (
        staging_root
        / "build"
        / KERNEL_VARIANT
        / KERNEL_SIGNATURE_FILENAME
    )
    if bundle_target.exists():
        raise PublicationError("staged kernel signature bundle already exists")
    bundle_target.write_bytes(bundle_bytes)
    signature = manifest["signature"]
    validate_kernel_signature_evidence(
        signature,
        staging_root=staging_root,
        certificate_identity=EXPECTED_SIGNER_IDENTITY,
        publisher_revision=publisher_revision,
    )
    verify_fn(
        staging_root,
        certificate_identity=EXPECTED_SIGNER_IDENTITY,
        publisher_revision=publisher_revision,
    )
    return signature


def require_kernel_builder_executable() -> str:
    executable = shutil.which("kernel-builder")
    if executable is None:
        raise PublicationError("pinned kernel-builder is not installed")
    version = subprocess.run(
        [executable, "--version"],
        check=False,
        capture_output=True,
        text=True,
        env=_subprocess_base_environment(),
    )
    observed_version = (version.stdout or version.stderr).strip()
    if version.returncode != 0 or observed_version != KERNEL_BUILDER_VERSION_OUTPUT:
        raise PublicationError(
            "kernel-builder version drifted "
            f"(expected {KERNEL_BUILDER_VERSION_OUTPUT!r}, "
            f"observed {observed_version!r})"
        )
    return executable


def upload_first_class_kernel(staging_root: Path, token: str) -> None:
    output_path = staging_root / "kernel-upload.json"
    environment = _subprocess_base_environment()
    environment["HF_TOKEN"] = token
    executable = require_kernel_builder_executable()
    command = [
        executable,
        "upload",
        str(staging_root),
        "--repo-id",
        EXPECTED_REPO_ID,
        "--branch",
        f"v{KERNEL_VERSION}",
        "--repo-type",
        KERNEL_REPO_TYPE,
        "--output-json",
        str(output_path),
        "--quiet",
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except FileNotFoundError as exc:
        raise PublicationError("pinned kernel-builder is not installed") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown uploader error").strip()
        raise PublicationError(f"kernel-builder upload failed: {detail[-2000:]}") from exc

    try:
        outcome = json.loads(output_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise PublicationError("kernel-builder did not write a valid outcome") from exc
    if outcome.get("repo_id") != EXPECTED_REPO_ID or outcome.get("branch") != "v1":
        raise PublicationError("kernel-builder outcome names an unexpected target")
    if outcome.get("status") not in {"uploaded", "no_changes"}:
        raise PublicationError("kernel-builder did not complete a direct upload")


def kernel_branch_targets(api: HfApi, *, token: str) -> dict[str, str]:
    refs = api.list_repo_refs(
        EXPECTED_REPO_ID,
        repo_type=KERNEL_REPO_TYPE,
        token=token,
    )
    targets = {
        branch.name: branch.target_commit
        for branch in refs.branches
        if branch.name in KERNEL_BRANCHES
    }
    if set(targets) != set(KERNEL_BRANCHES):
        raise PublicationError("first-class Kernel branches are incomplete")
    if any(
        FULL_SHA_RE.fullmatch(revision or "") is None
        for revision in targets.values()
    ):
        raise PublicationError("first-class Kernel branch target is not an exact SHA")
    return targets


def revalidate_kernel_branch_parents(
    api: HfApi,
    observed_branches: dict[str, dict[str, Any]],
    *,
    token: str,
) -> dict[str, str]:
    expected = {
        branch: observed_branches[branch]["revision"]
        for branch in KERNEL_BRANCHES
    }
    current = kernel_branch_targets(api, token=token)
    if current != expected:
        raise PublicationError(
            "first-class Kernel branch parents changed before upload"
        )
    return current


def raw_values_sha256(values: list[Any], kind: str, byte_order: str) -> str:
    if byte_order not in {"little", "big"}:
        raise PublicationError("retrieval evidence has an unsupported byte order")
    prefix = "<" if byte_order == "little" else ">"
    return hashlib.sha256(
        struct.pack(f"{prefix}{len(values)}{kind}", *values)
    ).hexdigest()


def same_json_value(observed: Any, expected: Any) -> bool:
    """Require the fixed JSON schema as well as values (True is not integer 1)."""
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            same_json_value(actual, wanted)
            for actual, wanted in zip(observed, expected)
        )
    if isinstance(expected, dict):
        return observed.keys() == expected.keys() and all(
            same_json_value(observed[key], value) for key, value in expected.items()
        )
    return observed == expected


def retrieval_smoke_attributes(byte_order: str) -> dict[str, Any]:
    """Compute raw-byte reference hashes without importing the published kernel."""
    return {
        "schema": "szl.governed-cosine-topk/v1",
        "query_shape": [2],
        "documents_shape": [4, 2],
        "output_shape": [1, 4],
        "dtype": "float32",
        "index_dtype": "int64",
        "device": "cpu",
        "byte_order": byte_order,
        "input_hash_format": "logical_c_order_raw_bytes",
        "hash_algorithm": "sha256",
        "query_sha256": raw_values_sha256(RETRIEVAL_SMOKE_QUERY, "f", byte_order),
        "documents_sha256": raw_values_sha256(
            [value for row in RETRIEVAL_SMOKE_DOCUMENTS for value in row], "f", byte_order
        ),
        "scores_sha256": raw_values_sha256(RETRIEVAL_SMOKE_SCORES[0], "f", byte_order),
        "indices_sha256": raw_values_sha256(RETRIEVAL_SMOKE_INDICES[0], "q", byte_order),
        "k": 4,
        "block_rows": 1,
        "actual_block_rows": 1,
        "max_similarity_elements": 1,
        "zero_document_count": 0,
        "zero_document_policy": "score_zero",
        "tie_break": "ascending_document_index",
        "implementation": "pytorch_float32_blocked_reference",
        "cuda_tf32_allowed": None,
        "receipt_authenticity": "UNSIGNED",
        "retrieval_quality": "NOT_MEASURED",
        "acceleration_claim": False,
    }


def validate_retrieval_runtime_evidence(evidence: Any) -> None:
    """Recheck retrieval values, raw hashes and receipt body in trusted code.

    This establishes consistency of the reported CPU smoke run. The hash chain
    has no signing key and is not evidence of authorship or retrieval quality.
    """
    try:
        if (
            type(evidence) is not dict
            or evidence.keys() != {
                "indices", "scores", "receipt", "receipt_depth", "chain_verified"
            }
            or not same_json_value(evidence.get("indices"), RETRIEVAL_SMOKE_INDICES)
            or not same_json_value(evidence.get("scores"), RETRIEVAL_SMOKE_SCORES)
            or evidence.get("chain_verified") is not True
            or type(evidence.get("receipt_depth")) is not int
            or evidence["receipt_depth"] != 1
        ):
            raise ValueError("values or chain depth")
        receipt = evidence["receipt"]
        if type(receipt) is not dict or receipt.keys() != {
            "seq", "kernel", "op", "attrs", "prev", "digest", "ts"
        }:
            raise ValueError("receipt object schema")
        attrs = receipt["attrs"]
        if type(attrs) is not dict:
            raise ValueError("receipt object schema")
        # The kernel runs in a separate, credentialless container. Its reported
        # byte order cannot select the trusted host's reference tensor hashes.
        byte_order = sys.byteorder
        if not same_json_value(attrs.get("byte_order"), byte_order):
            raise ValueError("receipt byte order differs from trusted host")
        expected = retrieval_smoke_attributes(byte_order)
        if (
            attrs.keys() != expected.keys() | {"torch_version", "matmul_precision"}
            or type(attrs.get("torch_version")) is not str
            or not attrs["torch_version"]
            or type(attrs.get("matmul_precision")) is not str
            or attrs["matmul_precision"] not in {"highest", "high", "medium"}
            or type(receipt["ts"]) is not float
            or not math.isfinite(receipt["ts"])
            or type(receipt.get("seq")) is not int
            or receipt["seq"] != 0
            or receipt.get("prev") != "0" * 64
            or receipt.get("kernel") != "governed_retrieval"
            or receipt.get("op") != "cosine_topk"
            or any(not same_json_value(attrs.get(key), value) for key, value in expected.items())
        ):
            raise ValueError("receipt contract")
        # The producer uses time.time(); ts has a finite-float shape but is
        # deliberately outside the digest. It proves neither freshness nor
        # timestamp authenticity, and is never used as a freshness gate.
        # Python equality treats -0.0 as +0.0. Bind the reported output values
        # themselves to the exact float32/int64 bytes recorded in the receipt.
        for field, kind in (("scores", "f"), ("indices", "q")):
            if raw_values_sha256(evidence[field][0], kind, byte_order) != attrs[f"{field}_sha256"]:
                raise ValueError("reported output raw hash")
        body = {key: receipt[key] for key in ("seq", "kernel", "op", "attrs", "prev")}
        digest = hashlib.sha3_256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
            .encode("utf-8")
        ).hexdigest()
        if receipt.get("digest") != digest:
            raise ValueError("receipt digest")
    except (AttributeError, KeyError, TypeError, ValueError, PublicationError) as exc:
        raise PublicationError("retrieval runtime evidence failed validation") from exc


def verify_stable_kernel_runtime(
    *,
    revision: str,
    get_kernel_fn: Callable[..., Any] | None = None,
    tensor_fn: Callable[[list[Any]], Any] | None = None,
    client_version: str | None = None,
) -> dict[str, Any]:
    if get_kernel_fn is None:
        from kernels import get_kernel

        get_kernel_fn = get_kernel
    if tensor_fn is None:
        import torch

        tensor_fn = torch.tensor
    if client_version is None:
        from importlib.metadata import version

        client_version = version("kernels")
    if client_version != KERNEL_RUNTIME_CLIENT_VERSION:
        raise PublicationError(
            "stable Kernel runtime client drifted "
            f"(expected {KERNEL_RUNTIME_CLIENT_VERSION}, observed {client_version})"
        )

    module = get_kernel_fn(
        EXPECTED_REPO_ID,
        revision=revision,
        backend="cpu",
        trust_remote_code=True,
    )
    missing_exports = [
        name for name in KERNEL_REQUIRED_EXPORTS if not hasattr(module, name)
    ]
    if missing_exports:
        raise PublicationError(f"stable get_kernel is missing public exports: {missing_exports}")
    selfcheck = module.selfcheck()
    if selfcheck.get("ok") is not True:
        raise PublicationError("stable get_kernel selfcheck did not pass")
    if (
        selfcheck.get("version") != EXPECTED_KERNEL_PACKAGE_VERSION
        or module.__version__ != EXPECTED_KERNEL_PACKAGE_VERSION
    ):
        raise PublicationError("stable get_kernel returned an unexpected package version")

    invalid_thresholds = (-0.01, 1.01, float("nan"), float("inf"))
    for threshold in invalid_thresholds:
        chain = module.UnifiedReceiptChain()
        try:
            module.governed_lambda_gate(
                chain,
                tensor_fn([0.5]),
                threshold=threshold,
            )
        except ValueError:
            pass
        else:
            raise PublicationError("invalid threshold did not fail closed")
        ok, depth, _ = chain.verify()
        if ok is not True or depth != 0:
            raise PublicationError("invalid threshold emitted a receipt")

    boundaries: dict[str, dict[str, Any]] = {}
    for threshold in (0.0, 1.0):
        expected_passed = threshold == 0.0
        chain = module.UnifiedReceiptChain()
        gate = module.governed_lambda_gate(
            chain,
            tensor_fn([0.5]),
            threshold=threshold,
        )
        ok, depth, first_break = chain.verify()
        if (
            ok is not True
            or depth != 1
            or first_break != -1
            or gate.get("threshold") != threshold
            or gate.get("passed") is not expected_passed
        ):
            raise PublicationError("inclusive threshold boundary contract failed")
        boundaries[str(int(threshold))] = {
            "passed": gate.get("passed"),
            "receipt_depth": depth,
        }

    chain = module.UnifiedReceiptChain()
    result = module.governed_cosine_topk(
        chain,
        tensor_fn(RETRIEVAL_SMOKE_QUERY),
        tensor_fn(RETRIEVAL_SMOKE_DOCUMENTS),
        k=4,
        block_rows=1,
    )
    if (
        str(result["indices"].dtype) != "torch.int64"
        or str(result["scores"].dtype) != "torch.float32"
        or str(result["indices"].device) != "cpu"
        or str(result["scores"].device) != "cpu"
        or chain.verify() != (True, 1, -1)
        or json.loads(chain.to_json()) != [result["receipt"]]
        or chain.head() != result["receipt"].get("digest")
        or result["receipt"].get("attrs", {}).get("byte_order") != sys.byteorder
    ):
        raise PublicationError("retrieval output or receipt chain contract failed")
    retrieval = {
        "indices": result["indices"].tolist(),
        "scores": result["scores"].tolist(),
        "receipt": result["receipt"],
        "receipt_depth": 1,
        "chain_verified": True,
    }
    validate_retrieval_runtime_evidence(retrieval)

    return {
        "status": "STABLE_GET_KERNEL_VERIFIED",
        "client_version": client_version,
        "revision": revision,
        "package_version": selfcheck["version"],
        "selfcheck_ok": True,
        "verified_exports": list(KERNEL_REQUIRED_EXPORTS),
        "invalid_thresholds_rejected_before_receipt": len(invalid_thresholds),
        "inclusive_boundaries": boundaries,
        "retrieval": retrieval,
    }


def verify_stable_kernel_runtime_isolated(*, revision: str) -> dict[str, Any]:
    """Run untrusted Hub code in a credentialless, workspace-free OCI sandbox."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in SENSITIVE_ENV_MARKERS)
    }
    create_command = [
        "docker",
        "create",
        "--log-driver=json-file",
        "--log-opt=max-size=64k",
        "--log-opt=max-file=1",
        # Docker's default PID namespace is private. Do not pass
        # ``--pid=private``: Docker rejects "private" as an explicit
        # selector, while omitting --pid preserves process isolation.
        "--network=bridge",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=64m",
        "--tmpfs",
        "/cache:rw,nosuid,nodev,noexec,size=2g",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=256",
        "--memory=4g",
        "--cpus=2",
        "--user=65532:65532",
        "--env=HF_HUB_DISABLE_IMPLICIT_TOKEN=1",
        "--env=HF_HOME=/cache/huggingface",
        "--env=XDG_CACHE_HOME=/cache",
        KERNEL_RUNTIME_IMAGE,
        "--revision",
        revision,
        "--output",
        KERNEL_RUNTIME_EVIDENCE_PATH,
    ]
    cleanup_timed_out = False
    try:
        created = subprocess.run(
            create_command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=KERNEL_RUNTIME_CONTROL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise PublicationError(
            "isolated stable Kernel runtime create timed out"
        ) from exc
    container_id = created.stdout.strip()
    if (
        created.returncode != 0
        or DOCKER_CONTAINER_ID_RE.fullmatch(container_id) is None
    ):
        detail = (created.stderr or created.stdout or "unknown create error").strip()
        raise PublicationError(
            f"isolated stable Kernel runtime create failed: {detail[-2000:]}"
        )
    try:
        try:
            started = subprocess.run(
                ["docker", "start", container_id],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=KERNEL_RUNTIME_CONTROL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise PublicationError(
                "isolated stable Kernel runtime start timed out"
            ) from exc
        if started.returncode != 0:
            detail = (started.stderr or started.stdout or "unknown start error").strip()
            raise PublicationError(
                f"isolated stable Kernel runtime start failed: {detail[-2000:]}"
            )
        try:
            waited = subprocess.run(
                ["docker", "wait", container_id],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=KERNEL_RUNTIME_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise PublicationError(
                "isolated stable Kernel runtime timed out after "
                f"{KERNEL_RUNTIME_TIMEOUT_SECONDS} seconds"
            ) from exc
        if waited.returncode != 0:
            detail = (waited.stderr or waited.stdout or "unknown wait error").strip()
            raise PublicationError(
                f"isolated stable Kernel runtime wait failed: {detail[-2000:]}"
            )
        container_exit_code = waited.stdout.strip()
        if (
            re.fullmatch(r"(?:0|[1-9][0-9]{0,2})", container_exit_code) is None
            or int(container_exit_code) > 255
        ):
            raise PublicationError(
                "isolated stable Kernel runtime returned an invalid exit code"
            )
        try:
            inspected = subprocess.run(
                ["docker", "inspect", "--format", "{{json .State}}", container_id],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=KERNEL_RUNTIME_CONTROL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise PublicationError(
                "isolated stable Kernel runtime state inspection timed out"
            ) from exc
        if inspected.returncode != 0:
            detail = (
                inspected.stderr or inspected.stdout or "unknown inspect error"
            ).strip()
            raise PublicationError(
                "isolated stable Kernel runtime state inspection failed: "
                f"{detail[-2000:]}"
            )
        try:
            state = json.loads(inspected.stdout)
        except json.JSONDecodeError as exc:
            raise PublicationError(
                "isolated stable Kernel runtime returned invalid state evidence"
            ) from exc
        observed_exit_code = state.get("ExitCode") if isinstance(state, dict) else None
        oom_killed = state.get("OOMKilled") if isinstance(state, dict) else None
        if (
            not isinstance(observed_exit_code, int)
            or isinstance(observed_exit_code, bool)
            or not isinstance(oom_killed, bool)
            or observed_exit_code != int(container_exit_code)
        ):
            raise PublicationError(
                "isolated stable Kernel runtime state evidence did not match docker wait"
            )
        state_summary = (
            f"exit_code={observed_exit_code}, "
            f"oom_killed={'true' if oom_killed else 'false'}"
        )
        if oom_killed:
            raise PublicationError(
                "isolated stable Kernel runtime was OOM-killed "
                f"({state_summary})"
            )
        with tempfile.TemporaryDirectory(prefix="szl-kernel-runtime-") as temporary:
            evidence_path = Path(temporary) / "evidence.json"
            try:
                copied = subprocess.run(
                    [
                        "docker",
                        "cp",
                        f"{container_id}:{KERNEL_RUNTIME_EVIDENCE_PATH}",
                        str(evidence_path),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                    timeout=KERNEL_RUNTIME_CONTROL_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                copied = None
            if copied is None or copied.returncode != 0:
                try:
                    logged = subprocess.run(
                        ["docker", "logs", "--tail", "100", container_id],
                        check=False,
                        capture_output=True,
                        text=True,
                        env=environment,
                        timeout=KERNEL_RUNTIME_CONTROL_TIMEOUT_SECONDS,
                    )
                except subprocess.TimeoutExpired:
                    logged = None
                log_output = (
                    ""
                    if logged is None
                    else f"{logged.stdout}\n{logged.stderr}"
                )
                evidence = runtime_evidence_from_logs(log_output)
                if evidence is None:
                    copy_detail = "evidence copy timed out"
                    if copied is not None:
                        copy_detail = (
                            copied.stderr
                            or copied.stdout
                            or "unknown evidence copy error"
                        ).strip()
                    log_detail = (
                        "logs timed out"
                        if logged is None
                        else bounded_runtime_log_detail(log_output)
                    )
                    raise PublicationError(
                        "isolated stable Kernel runtime exited without evidence "
                        f"({state_summary}): "
                        f"{copy_detail[-2000:]}; bounded logs: {log_detail}"
                    )
            else:
                try:
                    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                except (
                    FileNotFoundError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                ) as exc:
                    raise PublicationError(
                        "isolated stable Kernel runtime returned malformed evidence "
                        f"({state_summary})"
                    ) from exc
            if container_exit_code != "0":
                error_type = evidence.get("error_type")
                error = evidence.get("error")
                if (
                    evidence.get("status") != "FAILED"
                    or not isinstance(error_type, str)
                    or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,127}", error_type)
                    is None
                    or not isinstance(error, str)
                    or len(error) > 2000
                    or any(ord(character) < 32 or ord(character) > 126 for character in error)
                ):
                    raise PublicationError(
                        "isolated stable Kernel runtime exited without valid bounded failure evidence"
                    )
                raise PublicationError(
                    "isolated stable Kernel runtime failed: "
                    f"{error_type}: {error}"
                )
    finally:
        try:
            subprocess.run(
                ["docker", "rm", "--force", container_id],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=environment,
                timeout=KERNEL_RUNTIME_CONTROL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            cleanup_timed_out = True
    validate_stable_kernel_runtime_evidence(evidence, revision=revision)
    if cleanup_timed_out:
        raise PublicationError(
            "isolated stable Kernel runtime cleanup timed out"
        )
    return evidence


def validate_stable_kernel_runtime_evidence(
    evidence: Any, *, revision: str
) -> None:
    if type(evidence) is not dict:
        raise PublicationError("isolated stable Kernel runtime evidence failed validation")
    expected = {
        "status": "STABLE_GET_KERNEL_VERIFIED",
        "client_version": KERNEL_RUNTIME_CLIENT_VERSION,
        "revision": revision,
        "package_version": EXPECTED_KERNEL_PACKAGE_VERSION,
        "selfcheck_ok": True,
        "verified_exports": list(KERNEL_REQUIRED_EXPORTS),
        "invalid_thresholds_rejected_before_receipt": 4,
        "inclusive_boundaries": {
            "0": {"passed": True, "receipt_depth": 1},
            "1": {"passed": False, "receipt_depth": 1},
        },
    }
    if (
        evidence.keys() != expected.keys() | {"retrieval"}
        or any(not same_json_value(evidence.get(key), value) for key, value in expected.items())
    ):
        raise PublicationError(
            "isolated stable Kernel runtime evidence failed validation"
        )
    validate_retrieval_runtime_evidence(evidence.get("retrieval"))


def verify_kernel_readback(
    expected_files: dict[str, bytes],
    *,
    branch: str,
    revision: str,
    token: str,
    download_fn: Callable[..., str],
    list_files_fn: Callable[..., list[str]],
) -> None:
    observed_files = set(
        list_files_fn(
            EXPECTED_REPO_ID,
            repo_type=KERNEL_REPO_TYPE,
            revision=revision,
            token=token,
        )
    )
    if branch == "v1":
        expected_immutable_files = set(expected_files) | KERNEL_IMMUTABLE_ROOT_FILES
        if observed_files != expected_immutable_files:
            missing = sorted(expected_immutable_files - observed_files)
            unexpected = sorted(observed_files - expected_immutable_files)
            raise PublicationError(
                "first-class Kernel v1 immutable file-set mismatch "
                f"(missing={missing}, unexpected={unexpected})"
            )
    for kernel_path, expected in expected_files.items():
        downloaded = Path(
            download_fn(
                EXPECTED_REPO_ID,
                kernel_path,
                repo_type=KERNEL_REPO_TYPE,
                revision=revision,
                token=token,
            )
        )
        if downloaded.read_bytes() != expected:
            raise PublicationError(
                f"first-class Kernel {branch} readback mismatch at {kernel_path}"
            )


def run(
    *,
    source_root: Path,
    report_path: Path,
    authorization_path: Path,
    source_revision: str,
    publisher: dict[str, Any],
    publish: bool,
    token: str | None,
    prepare_signature: bool = False,
    signature_bundle_output: Path | None = None,
    signature_manifest_output: Path | None = None,
    signature_bundle_input: Path | None = None,
    signature_manifest_input: Path | None = None,
    api: HfApi | None = None,
    download_fn: Callable[..., str] = hf_hub_download,
    kernel_sign_fn: Callable[..., dict[str, Any]] | None = None,
    signature_verify_fn: Callable[..., None] = verify_kernel_metadata_signature,
    kernel_upload_fn: Callable[[Path, str], None] = upload_first_class_kernel,
    kernel_runtime_fn: Callable[..., dict[str, Any]] = (
        verify_stable_kernel_runtime_isolated
    ),
) -> dict[str, Any]:
    if prepare_signature and publish:
        raise PublicationError("signature preparation and publication are separate modes")
    if prepare_signature and token:
        raise PublicationError("HF_TOKEN must be absent from the signature preparation job")
    if publish and any(os.environ.get(key) for key in OIDC_ENV_ALLOWLIST):
        raise PublicationError("GitHub OIDC authority must be absent from the publication job")
    source_revision = source_revision.strip().lower()
    if FULL_SHA_RE.fullmatch(source_revision) is None:
        raise PublicationError("source revision must be an exact Git SHA")
    authorization_observation = load_authorization(
        authorization_path,
        source_revision=source_revision,
        publisher_revision=publisher["revision"],
    )
    authorization = authorization_binding(authorization_observation)
    contract = load_contract(source_root)
    files = local_evidence(source_root, contract)
    api = api or HfApi(token=token)
    observed_before = hub_before(
        api,
        contract,
        token=token,
        download_fn=download_fn,
    )
    legacy_publication = {
        "schema": "szl.hf-kernel-source-binding/v2",
        "artifact": {
            "repo_id": EXPECTED_REPO_ID,
            "repo_type": LEGACY_REPO_TYPE,
            "kind": "governed_kernel_suite_with_receipted_word_embeddings",
        },
        "source_repository": EXPECTED_SOURCE_REPOSITORY,
        "source_revision": source_revision,
        "source": {
            "url": f"https://github.com/{EXPECTED_SOURCE_REPOSITORY}",
            "revision": source_revision,
            "artifact_tree_sha256": tree_sha256(files),
            "declared_file_count": len(files),
            "files": files,
        },
        "publisher": publisher,
        "authorization": authorization_observation,
        "observed_hub_before_publication": observed_before["legacy_model"],
        "claims": contract["claims"],
        "limitations": contract["limitations"],
    }
    legacy_publication_bytes = canonical_json(legacy_publication).encode("utf-8")
    kernel_files = kernel_file_evidence(source_root)
    kernel_binding = {
        "schema": "szl.hf-first-class-kernel-binding/v2",
        "artifact": {
            "repo_id": EXPECTED_REPO_ID,
            "repo_type": KERNEL_REPO_TYPE,
            "backend": "torch-cpu",
            "package": "szl_kernels",
            "version": KERNEL_VERSION,
            "publication_interface": "kernel-builder",
            "publication_interface_version": KERNEL_BUILDER_VERSION,
            "publication_interface_source_revision": (
                KERNEL_BUILDER_SOURCE_REVISION
            ),
        },
        "source_repository": EXPECTED_SOURCE_REPOSITORY,
        "source_revision": source_revision,
        "source": {
            "url": f"https://github.com/{EXPECTED_SOURCE_REPOSITORY}",
            "revision": source_revision,
            "artifact_tree_sha256": tree_sha256(files),
            "kernel_files": kernel_files,
        },
        "publisher": publisher,
        "signature_policy": {
            "scheme": "sigstore-keyless",
            "signed_path": f"build/{KERNEL_VARIANT}/metadata.json",
            "bundle_path": f"build/{KERNEL_VARIANT}/{KERNEL_SIGNATURE_FILENAME}",
            "certificate_identity": publisher["certificate_identity"],
            "oidc_issuer": SIGSTORE_OIDC_ISSUER,
            "publisher_revision": publisher["revision"],
            "workflow_ref": EXPECTED_WORKFLOW_REF,
            "workflow_trigger": EXPECTED_WORKFLOW_TRIGGER,
            "authority_separation": "OIDC_SIGN_JOB_TO_HF_ONLY_PUBLISH_JOB",
            "verification": (
                "PREUPLOAD_IDENTITY_REVISION_AND_EXACT_POSTUPLOAD_READBACK"
            ),
        },
        "authorization_binding": authorization,
        "observed_hub_before_publication": observed_before["first_class_kernel"],
        "claims": contract["claims"],
        "limitations": contract["limitations"],
    }
    kernel_binding_bytes = canonical_json(kernel_binding).encode("utf-8")
    result: dict[str, Any] = {
        "schema": "szl.kernel-source-binding-report/v4",
        "mode": (
            "SIGNATURE_PREPARE"
            if prepare_signature
            else "PUBLISH" if publish else "DRY_RUN"
        ),
        "repo_id": EXPECTED_REPO_ID,
        "source_revision": source_revision,
        "publisher": publisher,
        "authorization_observation": authorization_observation,
        "artifact_tree_sha256": tree_sha256(files),
        "declared_file_count": len(files),
        "targets": {
            "legacy_model": {
                "repo_type": LEGACY_REPO_TYPE,
                "revision_before": observed_before["legacy_model"]["revision"],
                "publication_sha256": hashlib.sha256(
                    legacy_publication_bytes
                ).hexdigest(),
            },
            "first_class_kernel": {
                "repo_type": KERNEL_REPO_TYPE,
                "branches_before": observed_before["first_class_kernel"]["branches"],
                "binding_sha256": hashlib.sha256(kernel_binding_bytes).hexdigest(),
                "binding": kernel_binding,
                "mapped_file_count": len(kernel_files),
                "runtime": {
                    "status": "NOT_RUN",
                    "client_version": KERNEL_RUNTIME_CLIENT_VERSION,
                },
                "signature": {
                    "status": "NOT_RUN",
                    "tool": "cosign",
                    "tool_version": COSIGN_VERSION,
                },
            },
        },
        "status": "VERIFIED_DRY_RUN",
    }
    if prepare_signature:
        if signature_bundle_output is None or signature_manifest_output is None:
            raise PublicationError("signature transfer output paths are required")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        result["status"] = "SIGNATURE_PREPARATION_IN_PROGRESS"
        result["targets"]["first_class_kernel"]["signature"] = {
            "status": "PENDING",
            "tool": "cosign",
            "tool_version": COSIGN_VERSION,
        }
        report_path.write_text(canonical_json(result), encoding="utf-8")
        try:
            with tempfile.TemporaryDirectory(prefix="szl-kernel-sign-") as temporary:
                staging_root = Path(temporary)
                stage_first_class_kernel(
                    source_root,
                    kernel_binding_bytes,
                    staging_root,
                )
                signer = kernel_sign_fn or sign_kernel_metadata
                signature = signer(
                    staging_root,
                    certificate_identity=publisher["certificate_identity"],
                    publisher_revision=publisher["revision"],
                )
                write_kernel_signature_transfer(
                    staging_root=staging_root,
                    signature=signature,
                    bundle_output=signature_bundle_output,
                    manifest_output=signature_manifest_output,
                    source_revision=source_revision,
                    publisher_revision=publisher["revision"],
                    binding_sha256=hashlib.sha256(
                        kernel_binding_bytes
                    ).hexdigest(),
                )
        except Exception as exc:
            result["targets"]["first_class_kernel"]["signature"] = {
                "status": "FAILED",
                "tool": "cosign",
                "tool_version": COSIGN_VERSION,
                "error_type": bounded_error_type(exc),
            }
            result["status"] = "SIGNATURE_PREPARATION_FAILED"
            report_path.write_text(canonical_json(result), encoding="utf-8")
            raise
        result["targets"]["first_class_kernel"]["signature"] = signature
        result["status"] = "SIGNATURE_PREPARED_NO_PROVIDER_WRITE"
        report_path.write_text(canonical_json(result), encoding="utf-8")
        return result
    if publish:
        if not token:
            raise PublicationError("HF_TOKEN is required when --publish is used")

        report_path.parent.mkdir(parents=True, exist_ok=True)
        result["status"] = "PUBLICATION_IN_PROGRESS"
        result["targets"]["first_class_kernel"]["branches_after"] = {}
        result["targets"]["first_class_kernel"]["readback"] = {}
        result["targets"]["first_class_kernel"][
            "parents_revalidated_before_upload"
        ] = {}
        result["targets"]["first_class_kernel"]["runtime"] = {
            "status": "PENDING",
            "client_version": KERNEL_RUNTIME_CLIENT_VERSION,
        }
        result["targets"]["first_class_kernel"]["signature"] = {
            "status": "PENDING",
            "tool": "cosign",
            "tool_version": COSIGN_VERSION,
        }
        report_path.write_text(canonical_json(result), encoding="utf-8")

        with tempfile.TemporaryDirectory(prefix="szl-kernel-upload-") as temporary:
            staging_root = Path(temporary)
            expected_kernel_files = stage_first_class_kernel(
                source_root,
                kernel_binding_bytes,
                staging_root,
            )
            try:
                if kernel_sign_fn is not None:
                    signature = kernel_sign_fn(
                        staging_root,
                        certificate_identity=publisher["certificate_identity"],
                        publisher_revision=publisher["revision"],
                    )
                else:
                    if (
                        signature_bundle_input is None
                        or signature_manifest_input is None
                    ):
                        raise PublicationError(
                            "prepared signature transfer inputs are required"
                        )
                    signature = consume_kernel_signature_transfer(
                        staging_root=staging_root,
                        bundle_input=signature_bundle_input,
                        manifest_input=signature_manifest_input,
                        source_revision=source_revision,
                        publisher_revision=publisher["revision"],
                        binding_sha256=hashlib.sha256(
                            kernel_binding_bytes
                        ).hexdigest(),
                        verify_fn=signature_verify_fn,
                    )
                bundle_bytes = validate_kernel_signature_evidence(
                    signature,
                    staging_root=staging_root,
                    certificate_identity=publisher["certificate_identity"],
                    publisher_revision=publisher["revision"],
                )
            except Exception as exc:
                result["targets"]["first_class_kernel"]["signature"] = {
                    "status": "FAILED",
                    "tool": "cosign",
                    "tool_version": COSIGN_VERSION,
                    "error_type": bounded_error_type(exc),
                }
                result["status"] = "SIGNATURE_VALIDATION_FAILED_NO_PROVIDER_WRITE"
                report_path.write_text(canonical_json(result), encoding="utf-8")
                raise
            signature_path = f"build/{KERNEL_VARIANT}/{KERNEL_SIGNATURE_FILENAME}"
            expected_kernel_files["v1"][signature_path] = bundle_bytes
            result["targets"]["first_class_kernel"]["signature"] = signature
            report_path.write_text(canonical_json(result), encoding="utf-8")
            try:
                revalidated_parents = revalidate_kernel_branch_parents(
                    api,
                    observed_before["first_class_kernel"]["branches"],
                    token=token,
                )
            except Exception as exc:
                record_publication_failure(
                    result,
                    report_path,
                    stage="KERNEL_PARENT_REVALIDATION",
                    exc=exc,
                    provider_write_attempted=False,
                )
                raise
            result["targets"]["first_class_kernel"][
                "parents_revalidated_before_upload"
            ] = revalidated_parents
            report_path.write_text(canonical_json(result), encoding="utf-8")
            upload_error: Exception | None = None
            try:
                kernel_upload_fn(staging_root, token)
            except Exception as exc:  # preserve branch state after partial upload
                upload_error = exc
            try:
                branch_targets = kernel_branch_targets(api, token=token)
            except Exception as exc:
                record_publication_failure(
                    result,
                    report_path,
                    stage=(
                        "KERNEL_UPLOAD"
                        if upload_error is not None
                        else "KERNEL_BRANCH_DISCOVERY"
                    ),
                    exc=upload_error if upload_error is not None else exc,
                    provider_write_attempted=True,
                )
                if upload_error is not None:
                    raise upload_error
                raise
            result["targets"]["first_class_kernel"]["branches_after"] = (
                branch_targets
            )
            result["targets"]["first_class_kernel"]["readback"] = {
                branch: "PENDING" for branch in KERNEL_BRANCHES
            }
            report_path.write_text(canonical_json(result), encoding="utf-8")
            if upload_error is not None:
                record_publication_failure(
                    result,
                    report_path,
                    stage="KERNEL_UPLOAD",
                    exc=upload_error,
                    provider_write_attempted=True,
                )
                raise upload_error
            for branch in KERNEL_BRANCHES:
                try:
                    verify_kernel_readback(
                        expected_kernel_files[branch],
                        branch=branch,
                        revision=branch_targets[branch],
                        token=token,
                        download_fn=download_fn,
                        list_files_fn=api.list_repo_files,
                    )
                except Exception as exc:
                    record_publication_failure(
                        result,
                        report_path,
                        stage=f"KERNEL_READBACK_{branch.upper()}",
                        exc=exc,
                        provider_write_attempted=True,
                    )
                    raise
                result["targets"]["first_class_kernel"]["readback"][branch] = (
                    "EXACT_BYTES_VERIFIED"
                )
                report_path.write_text(canonical_json(result), encoding="utf-8")

            try:
                runtime = kernel_runtime_fn(revision=branch_targets["v1"])
                validate_stable_kernel_runtime_evidence(
                    runtime, revision=branch_targets["v1"]
                )
            except Exception as exc:
                result["targets"]["first_class_kernel"]["runtime"] = {
                    "status": "FAILED",
                    "client_version": KERNEL_RUNTIME_CLIENT_VERSION,
                    "error_type": bounded_error_type(exc),
                }
                record_publication_failure(
                    result,
                    report_path,
                    stage="KERNEL_RUNTIME",
                    exc=exc,
                    provider_write_attempted=True,
                )
                raise
            result["targets"]["first_class_kernel"]["runtime"] = runtime
            report_path.write_text(canonical_json(result), encoding="utf-8")

        legacy_operations = [
            CommitOperationAdd(
                path_in_repo=relative,
                path_or_fileobj=str(safe_file(source_root, relative)),
            )
            for relative in contract["artifact_files"]
        ]
        legacy_operations.append(
            CommitOperationAdd(
                path_in_repo="publication.json",
                path_or_fileobj=io.BytesIO(legacy_publication_bytes),
            )
        )
        try:
            legacy_commit = api.create_commit(
                repo_id=EXPECTED_REPO_ID,
                repo_type=LEGACY_REPO_TYPE,
                parent_commit=observed_before["legacy_model"]["revision"],
                operations=legacy_operations,
                commit_message=f"Publish authorized source {source_revision[:12]}",
                token=token,
            )
            legacy_revision = getattr(legacy_commit, "oid", None)
            if not legacy_revision:
                legacy_revision = api.model_info(EXPECTED_REPO_ID, token=token).sha
        except Exception as exc:
            record_publication_failure(
                result,
                report_path,
                stage="LEGACY_COMMIT",
                exc=exc,
                provider_write_attempted=True,
            )
            raise
        result["targets"]["legacy_model"]["revision_after"] = legacy_revision
        result["targets"]["legacy_model"]["readback"] = "PENDING"
        report_path.write_text(canonical_json(result), encoding="utf-8")
        try:
            verify_legacy_readback(
                source_root,
                contract,
                legacy_publication_bytes,
                revision=legacy_revision,
                token=token,
                download_fn=download_fn,
            )
        except Exception as exc:
            record_publication_failure(
                result,
                report_path,
                stage="LEGACY_READBACK",
                exc=exc,
                provider_write_attempted=True,
            )
            raise
        result["targets"]["legacy_model"]["readback"] = "EXACT_BYTES_VERIFIED"
        result["status"] = "PUBLISHED_AND_EXACT_READBACK_VERIFIED"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(canonical_json(result), encoding="utf-8")
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--authorization-report", type=Path, required=True)
    parser.add_argument("--publisher-repository", required=True)
    parser.add_argument("--publisher-revision", required=True)
    parser.add_argument("--publisher-workflow-ref", required=True)
    parser.add_argument("--publisher-run-id", required=True)
    parser.add_argument("--publisher-run-attempt", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-signature", action="store_true")
    mode.add_argument("--publish", action="store_true")
    parser.add_argument("--signature-bundle-output", type=Path)
    parser.add_argument("--signature-manifest-output", type=Path)
    parser.add_argument("--signature-bundle-input", type=Path)
    parser.add_argument("--signature-manifest-input", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/source-binding-published.json"),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    publisher = publisher_identity(
        repository=args.publisher_repository,
        revision=args.publisher_revision,
        workflow_ref=args.publisher_workflow_ref,
        run_id=args.publisher_run_id,
        run_attempt=args.publisher_run_attempt,
    )
    result = run(
        source_root=args.source_dir,
        report_path=args.report,
        authorization_path=args.authorization_report,
        source_revision=args.source_revision,
        publisher=publisher,
        publish=args.publish,
        token=os.getenv("HF_TOKEN"),
        prepare_signature=args.prepare_signature,
        signature_bundle_output=args.signature_bundle_output,
        signature_manifest_output=args.signature_manifest_output,
        signature_bundle_input=args.signature_bundle_input,
        signature_manifest_input=args.signature_manifest_input,
    )
    print(canonical_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
