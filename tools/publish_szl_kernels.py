#!/usr/bin/env python3
"""Publish authorized SZL kernels data using only trusted Forge code."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

EXPECTED_REPO_ID = "SZLHOLDINGS/szl-kernels"
EXPECTED_SOURCE_REPOSITORY = "szl-holdings/szl-kernels"
EXPECTED_PUBLISHER_REPOSITORY = "szl-holdings/szl-forge"
EXPECTED_KERNEL_PACKAGE_VERSION = "0.1.1"
KERNEL_RUNTIME_CLIENT_VERSION = "0.16.0"
KERNEL_RUNTIME_IMAGE = f"szl-kernel-runtime:{KERNEL_RUNTIME_CLIENT_VERSION}"
KERNEL_RUNTIME_TIMEOUT_SECONDS = 300
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
FIRST_CLASS_KERNEL_FILES = {
    "build/torch-universal/szl_kernels/__init__.py": (
        f"build/{KERNEL_VARIANT}/__init__.py"
    ),
    "build/torch-universal/szl_kernels/_chain.py": (
        f"build/{KERNEL_VARIANT}/_chain.py"
    ),
    "build/torch-universal/szl_kernels/_ops.py": (
        f"build/{KERNEL_VARIANT}/_ops.py"
    ),
}
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
    "v1": {
        f"build/{KERNEL_VARIANT}/szl_kernels/__init__.py",
        f"build/{KERNEL_VARIANT}/szl_kernels/_chain.py",
        f"build/{KERNEL_VARIANT}/szl_kernels/_ops.py",
        f"build/{KERNEL_VARIANT}/metadata.json",
    },
}
FIRST_CLASS_REQUIRED_SOURCE_FILES = {
    "README.md",
    *FIRST_CLASS_KERNEL_FILES.keys(),
}


class PublicationError(RuntimeError):
    """Raised when publication evidence is insufficient."""


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


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
        or source.get("signature_verified") is not True
    ):
        raise PublicationError("source authorization does not bind the requested revision")
    if (
        publisher.get("repository") != EXPECTED_PUBLISHER_REPOSITORY
        or publisher.get("revision") != publisher_revision
        or publisher.get("protected_main") != publisher_revision
    ):
        raise PublicationError("publisher authorization does not bind this Forge revision")
    return payload


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
        observed_files = {
            entry.path
            for entry in api.list_repo_tree(
                EXPECTED_REPO_ID,
                repo_type=KERNEL_REPO_TYPE,
                revision=target,
                recursive=True,
                token=token,
            )
            if getattr(entry, "path", None)
        }
        missing = sorted(expected_paths - observed_files)
        if missing:
            raise PublicationError(
                f"first-class Kernel {branch} is missing package files: {missing}"
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
    workflow_path = ".github/workflows/publish-szl-kernels.yml"
    if not workflow_ref.startswith(
        f"{EXPECTED_PUBLISHER_REPOSITORY}/{workflow_path}@"
    ):
        raise PublicationError("publisher workflow reference is malformed")
    return {
        "repository": repository,
        "revision": revision,
        "workflow_path": workflow_path,
        "workflow_ref": workflow_ref,
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
    readme = safe_file(source_root, "README.md")
    evidence = [
        {
            "source_path": "README.md",
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
            "README.md": safe_file(source_root, "README.md").read_bytes(),
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


def require_kernel_builder_executable() -> str:
    executable = shutil.which("kernel-builder")
    if executable is None:
        raise PublicationError("pinned kernel-builder is not installed")
    version = subprocess.run(
        [executable, "--version"],
        check=False,
        capture_output=True,
        text=True,
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
    environment = os.environ.copy()
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


def verify_stable_kernel_runtime(
    *,
    revision: str,
    get_kernel_fn: Callable[..., Any] | None = None,
    tensor_fn: Callable[[list[float]], Any] | None = None,
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
    selfcheck = module.selfcheck()
    if selfcheck.get("ok") is not True:
        raise PublicationError("stable get_kernel selfcheck did not pass")
    if selfcheck.get("version") != EXPECTED_KERNEL_PACKAGE_VERSION:
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

    return {
        "status": "STABLE_GET_KERNEL_VERIFIED",
        "client_version": client_version,
        "revision": revision,
        "package_version": selfcheck["version"],
        "selfcheck_ok": True,
        "invalid_thresholds_rejected_before_receipt": len(invalid_thresholds),
        "inclusive_boundaries": boundaries,
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
        "--log-driver=none",
        # Docker's default PID namespace is private. Do not pass
        # ``--pid=private``: Docker rejects "private" as an explicit
        # selector, while omitting --pid preserves process isolation.
        "--network=bridge",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=64m",
        "--tmpfs",
        "/cache:rw,nosuid,nodev,noexec,size=2g",
        "--tmpfs",
        "/output:rw,nosuid,nodev,noexec,size=1m",
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
        "/output/evidence.json",
    ]
    created = subprocess.run(
        create_command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
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
        started = subprocess.run(
            ["docker", "start", container_id],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
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
        if waited.returncode != 0 or waited.stdout.strip() != "0":
            raise PublicationError(
                "isolated stable Kernel runtime exited without verified evidence"
            )
        with tempfile.TemporaryDirectory(prefix="szl-kernel-runtime-") as temporary:
            evidence_path = Path(temporary) / "evidence.json"
            copied = subprocess.run(
                [
                    "docker",
                    "cp",
                    f"{container_id}:/output/evidence.json",
                    str(evidence_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            if copied.returncode != 0:
                detail = (
                    copied.stderr or copied.stdout or "unknown evidence copy error"
                ).strip()
                raise PublicationError(
                    "isolated stable Kernel runtime evidence copy failed: "
                    f"{detail[-2000:]}"
                )
            try:
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                raise PublicationError(
                    "isolated stable Kernel runtime returned invalid evidence"
                ) from exc
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container_id],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
        )
    boundaries = evidence.get("inclusive_boundaries", {})
    if (
        evidence.get("status") != "STABLE_GET_KERNEL_VERIFIED"
        or evidence.get("client_version") != KERNEL_RUNTIME_CLIENT_VERSION
        or evidence.get("revision") != revision
        or evidence.get("package_version") != EXPECTED_KERNEL_PACKAGE_VERSION
        or evidence.get("selfcheck_ok") is not True
        or evidence.get("invalid_thresholds_rejected_before_receipt") != 4
        or boundaries.get("0", {}).get("passed") is not True
        or boundaries.get("0", {}).get("receipt_depth") != 1
        or boundaries.get("1", {}).get("passed") is not False
        or boundaries.get("1", {}).get("receipt_depth") != 1
    ):
        raise PublicationError(
            "isolated stable Kernel runtime evidence failed validation"
        )
    return evidence


def verify_kernel_readback(
    expected_files: dict[str, bytes],
    *,
    branch: str,
    revision: str,
    token: str,
    download_fn: Callable[..., str],
) -> None:
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
    api: HfApi | None = None,
    download_fn: Callable[..., str] = hf_hub_download,
    kernel_upload_fn: Callable[[Path, str], None] = upload_first_class_kernel,
    kernel_runtime_fn: Callable[..., dict[str, Any]] = (
        verify_stable_kernel_runtime_isolated
    ),
) -> dict[str, Any]:
    source_revision = source_revision.strip().lower()
    if FULL_SHA_RE.fullmatch(source_revision) is None:
        raise PublicationError("source revision must be an exact Git SHA")
    authorization = load_authorization(
        authorization_path,
        source_revision=source_revision,
        publisher_revision=publisher["revision"],
    )
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
        "authorization": authorization,
        "observed_hub_before_publication": observed_before["legacy_model"],
        "claims": contract["claims"],
        "limitations": contract["limitations"],
    }
    legacy_publication_bytes = canonical_json(legacy_publication).encode("utf-8")
    kernel_files = kernel_file_evidence(source_root)
    kernel_binding = {
        "schema": "szl.hf-first-class-kernel-binding/v1",
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
        "authorization": authorization,
        "observed_hub_before_publication": observed_before["first_class_kernel"],
        "claims": contract["claims"],
        "limitations": contract["limitations"],
    }
    kernel_binding_bytes = canonical_json(kernel_binding).encode("utf-8")
    result: dict[str, Any] = {
        "schema": "szl.kernel-source-binding-report/v3",
        "mode": "PUBLISH" if publish else "DRY_RUN",
        "repo_id": EXPECTED_REPO_ID,
        "source_revision": source_revision,
        "publisher": publisher,
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
            },
        },
        "status": "VERIFIED_DRY_RUN",
    }
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
        report_path.write_text(canonical_json(result), encoding="utf-8")

        with tempfile.TemporaryDirectory(prefix="szl-kernel-upload-") as temporary:
            staging_root = Path(temporary)
            expected_kernel_files = stage_first_class_kernel(
                source_root,
                kernel_binding_bytes,
                staging_root,
            )
            revalidated_parents = revalidate_kernel_branch_parents(
                api,
                observed_before["first_class_kernel"]["branches"],
                token=token,
            )
            result["targets"]["first_class_kernel"][
                "parents_revalidated_before_upload"
            ] = revalidated_parents
            report_path.write_text(canonical_json(result), encoding="utf-8")
            upload_error: Exception | None = None
            try:
                kernel_upload_fn(staging_root, token)
            except Exception as exc:  # preserve branch state after partial upload
                upload_error = exc
            branch_targets = kernel_branch_targets(api, token=token)
            result["targets"]["first_class_kernel"]["branches_after"] = (
                branch_targets
            )
            result["targets"]["first_class_kernel"]["readback"] = {
                branch: "PENDING" for branch in KERNEL_BRANCHES
            }
            report_path.write_text(canonical_json(result), encoding="utf-8")
            if upload_error is not None:
                raise upload_error
            for branch in KERNEL_BRANCHES:
                verify_kernel_readback(
                    expected_kernel_files[branch],
                    branch=branch,
                    revision=branch_targets[branch],
                    token=token,
                    download_fn=download_fn,
                )
                result["targets"]["first_class_kernel"]["readback"][branch] = (
                    "EXACT_BYTES_VERIFIED"
                )
                report_path.write_text(canonical_json(result), encoding="utf-8")

            try:
                runtime = kernel_runtime_fn(revision=branch_targets["v1"])
            except Exception as exc:
                result["targets"]["first_class_kernel"]["runtime"] = {
                    "status": "FAILED",
                    "client_version": KERNEL_RUNTIME_CLIENT_VERSION,
                    "error": f"{type(exc).__name__}: {exc}"[:2000],
                }
                report_path.write_text(canonical_json(result), encoding="utf-8")
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
        result["targets"]["legacy_model"]["revision_after"] = legacy_revision
        result["targets"]["legacy_model"]["readback"] = "PENDING"
        report_path.write_text(canonical_json(result), encoding="utf-8")
        verify_legacy_readback(
            source_root,
            contract,
            legacy_publication_bytes,
            revision=legacy_revision,
            token=token,
            download_fn=download_fn,
        )
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
    parser.add_argument("--publish", action="store_true")
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
    )
    print(canonical_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
