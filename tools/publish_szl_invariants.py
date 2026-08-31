#!/usr/bin/env python3
"""Publish the authorized szl-invariants Python payload with exact readback."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


EXPECTED_REPO_ID = "SZLHOLDINGS/szl-invariants"
EXPECTED_SOURCE_REPOSITORY = "szl-holdings/szl-invariants"
EXPECTED_PUBLISHER_REPOSITORY = "szl-holdings/szl-forge"
CONTRACT_RELATIVE = Path("publishing/invariants-source-binding.json")
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
KERNEL_REPO_TYPE = "kernel"
MODEL_REPO_TYPE = "model"
KERNEL_BRANCHES = ("main", "v1")
KERNEL_RELEASE_BRANCH = "v1"
KERNEL_VERSION = 1
KERNEL_BUILDER_PACKAGE = "hf-kernel-builder"
KERNEL_BUILDER_VERSION = "0.17.0-dev0"
KERNEL_BUILDER_VERSION_OUTPUT = (
    f"{KERNEL_BUILDER_PACKAGE} {KERNEL_BUILDER_VERSION}"
)
KERNEL_BUILDER_SOURCE_REVISION = "633246310320d85def0c67d62c7912fd444a842f"
EXPECTED_ARTIFACT_FILES = frozenset(
    {
        "build/torch-universal/szl_invariants/__init__.py",
        "build/torch-universal/szl_invariants/metadata.json",
        "torch-ext/szl_invariants/__init__.py",
        "torch-ext/szl_invariants/metadata.json",
    }
)
EXPECTED_TARGETS = frozenset(
    {
        (
            "model",
            "build/torch-universal/szl_invariants/__init__.py",
            "build/torch-universal/szl_invariants/__init__.py",
        ),
        (
            "model",
            "build/torch-universal/szl_invariants/metadata.json",
            "build/torch-universal/szl_invariants/metadata.json",
        ),
        (
            "kernel",
            "build/torch-universal/szl_invariants/__init__.py",
            "build/torch-universal/szl_invariants/__init__.py",
        ),
        (
            "kernel",
            "build/torch-universal/szl_invariants/metadata.json",
            "build/torch-universal/szl_invariants/metadata.json",
        ),
        (
            "kernel",
            "torch-ext/szl_invariants/__init__.py",
            "build/torch-cpu/szl_invariants/__init__.py",
        ),
        (
            "kernel",
            "torch-ext/szl_invariants/metadata.json",
            "build/torch-cpu/szl_invariants/metadata.json",
        ),
    }
)


class PublicationError(RuntimeError):
    """Raised when authorization or publication evidence is insufficient."""


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


def _target_tuple(target: dict[str, Any]) -> tuple[str, str, str]:
    if set(target) != {"repo_type", "source_path", "path_in_repo"}:
        raise PublicationError("publication targets contain unexpected fields")
    return (
        str(target["repo_type"]),
        str(target["source_path"]),
        str(target["path_in_repo"]),
    )


def load_contract(source_root: Path) -> dict[str, Any]:
    path = safe_file(source_root, CONTRACT_RELATIVE.as_posix())
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PublicationError("publication contract must be a JSON object")
    if payload.get("schema") != "szl.invariants-source-binding/v1":
        raise PublicationError("unsupported invariants source-binding schema")
    if payload.get("repo_id") != EXPECTED_REPO_ID:
        raise PublicationError("source contract cannot select another Hub repository")
    if payload.get("source_repository") != EXPECTED_SOURCE_REPOSITORY:
        raise PublicationError("source contract names an unexpected repository")

    artifacts = payload.get("artifact_files")
    if (
        not isinstance(artifacts, list)
        or len(artifacts) != len(set(artifacts))
        or set(artifacts) != EXPECTED_ARTIFACT_FILES
    ):
        raise PublicationError("artifact_files must equal the closed publication set")
    expected = payload.get("expected_artifact_sha256")
    if (
        not isinstance(expected, dict)
        or set(expected) != EXPECTED_ARTIFACT_FILES
        or any(not isinstance(value, str) or HASH_RE.fullmatch(value) is None for value in expected.values())
    ):
        raise PublicationError("expected hashes must bind every declared artifact")

    targets = payload.get("publication_targets")
    if not isinstance(targets, list) or not all(isinstance(item, dict) for item in targets):
        raise PublicationError("publication_targets must be an object list")
    target_tuples = [_target_tuple(item) for item in targets]
    if len(target_tuples) != len(set(target_tuples)) or set(target_tuples) != EXPECTED_TARGETS:
        raise PublicationError("publication targets must equal the closed destination set")
    destinations = [(repo_type, path) for repo_type, _source, path in target_tuples]
    if len(destinations) != len(set(destinations)):
        raise PublicationError("publication destinations must be unique")

    claims = payload.get("claims")
    if not isinstance(claims, dict) or claims.get("trained_weights_present") is not False:
        raise PublicationError("source contract must deny trained weights")
    limitations = payload.get("limitations")
    if not isinstance(limitations, list) or not limitations:
        raise PublicationError("source contract limitations must be non-empty")
    return payload


def load_authorization(
    path: Path,
    *,
    source_revision: str,
    publisher_revision: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "szl.invariants-release-authorization/v1":
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
    evidence = []
    for relative in sorted(contract["artifact_files"]):
        path = safe_file(source_root, relative)
        observed = file_sha256(path)
        expected = contract["expected_artifact_sha256"][relative]
        if observed != expected:
            raise PublicationError(
                f"{relative} SHA-256 drifted (expected {expected}, observed {observed})"
            )
        evidence.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": observed}
        )

    universal = safe_file(
        source_root, "build/torch-universal/szl_invariants/__init__.py"
    ).read_bytes()
    extension = safe_file(source_root, "torch-ext/szl_invariants/__init__.py").read_bytes()
    if universal != extension:
        raise PublicationError("protected Python variants are not byte-identical")
    if b'"trained_weights_present": False' not in universal or b'"trained_weights_present": True' in universal:
        raise PublicationError("protected Python source does not deny trained weights")

    universal_metadata = safe_file(
        source_root, "build/torch-universal/szl_invariants/metadata.json"
    ).read_bytes()
    extension_metadata = safe_file(
        source_root, "torch-ext/szl_invariants/metadata.json"
    ).read_bytes()
    if universal_metadata != extension_metadata:
        raise PublicationError("protected metadata variants are not byte-identical")
    if b'"trained_weights_present": false' not in universal_metadata:
        raise PublicationError("protected metadata does not deny trained weights")
    return evidence


def tree_sha256(evidence: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(evidence, key=lambda value: value["path"]):
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


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
    workflow_path = ".github/workflows/publish-szl-invariants.yml"
    if not workflow_ref.startswith(f"{repository}/{workflow_path}@"):
        raise PublicationError("publisher workflow reference is malformed")
    return {
        "repository": repository,
        "revision": revision,
        "workflow_path": workflow_path,
        "workflow_ref": workflow_ref,
        "workflow_url": f"https://github.com/{repository}/blob/{revision}/{workflow_path}",
        "run_id": run_id,
        "run_attempt": run_attempt,
        "run_url": f"https://github.com/{repository}/actions/runs/{run_id}",
    }


def model_head(api: HfApi, token: str | None) -> str:
    revision = str(
        getattr(
            api.model_info(EXPECTED_REPO_ID, files_metadata=True, token=token),
            "sha",
            "",
        )
    ).lower()
    if FULL_SHA_RE.fullmatch(revision) is None:
        raise PublicationError("Hub model head is not an immutable Git SHA")
    return revision


def kernel_branch_targets(api: HfApi, *, token: str | None) -> dict[str, str]:
    refs = api.list_repo_refs(
        EXPECTED_REPO_ID,
        repo_type=KERNEL_REPO_TYPE,
        token=token,
    )
    targets = {
        branch.name: str(branch.target_commit).lower()
        for branch in refs.branches
        if branch.name in KERNEL_BRANCHES
    }
    if set(targets) != set(KERNEL_BRANCHES):
        raise PublicationError("first-class Kernel branches are incomplete")
    if any(FULL_SHA_RE.fullmatch(revision) is None for revision in targets.values()):
        raise PublicationError("first-class Kernel branch target is not an exact SHA")
    return targets


def revalidate_kernel_branch_parents(
    api: HfApi,
    expected: dict[str, str],
    *,
    token: str,
) -> dict[str, str]:
    current = kernel_branch_targets(api, token=token)
    if current != expected:
        raise PublicationError(
            "first-class Kernel branch parent changed before upload"
        )
    return current


def digest_base64(payload: bytes) -> str:
    return base64.b64encode(hashlib.sha256(payload).digest()).decode("ascii")


def stage_kernel_targets(
    source_root: Path,
    contract: dict[str, Any],
    staging_root: Path,
) -> dict[str, bytes]:
    """Create the v1 build tree consumed by the pinned kernel-builder."""
    expected: dict[str, bytes] = {}
    by_variant: dict[str, dict[str, bytes]] = {}
    for target in targets_for(contract, KERNEL_REPO_TYPE):
        destination = Path(target["path_in_repo"])
        parts = destination.parts
        if len(parts) < 3 or parts[0] != "build":
            raise PublicationError("Kernel destinations must be versioned build paths")
        variant = parts[1]
        relative = Path(*parts[2:]).as_posix()
        payload = safe_file(source_root, target["source_path"]).read_bytes()
        output = staging_root / destination
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
        repository_path = destination.as_posix()
        expected[repository_path] = payload
        by_variant.setdefault(variant, {})[relative] = payload

    for variant, files in sorted(by_variant.items()):
        digest = hashlib.sha256()
        for relative, payload in sorted(files.items()):
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(payload).digest())
        variant_id = variant.replace("-", "_")
        metadata = {
            "name": "szl-invariants",
            "id": f"_szl_invariants_{variant_id}_{digest.hexdigest()[:8]}",
            "version": KERNEL_VERSION,
            "license": "Apache-2.0",
            "python-depends": [],
            "backend": {"type": "cpu"},
            "digest": {
                "algorithm": "sha256",
                "files": {
                    relative: digest_base64(payload)
                    for relative, payload in sorted(files.items())
                },
            },
        }
        metadata_bytes = canonical_json(metadata).encode("utf-8")
        metadata_path = staging_root / "build" / variant / "metadata.json"
        metadata_path.write_bytes(metadata_bytes)
        expected[
            f"build/{variant}/metadata.json"
        ] = metadata_bytes
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
    observed = (version.stdout or version.stderr).strip()
    if version.returncode != 0 or observed != KERNEL_BUILDER_VERSION_OUTPUT:
        raise PublicationError(
            "kernel-builder version drifted "
            f"(expected {KERNEL_BUILDER_VERSION_OUTPUT!r}, observed {observed!r})"
        )
    return executable


def upload_first_class_kernel(staging_root: Path, token: str) -> None:
    output_path = staging_root / "kernel-upload.json"
    environment = os.environ.copy()
    environment["HF_TOKEN"] = token
    command = [
        require_kernel_builder_executable(),
        "upload",
        str(staging_root),
        "--repo-id",
        EXPECTED_REPO_ID,
        "--branch",
        KERNEL_RELEASE_BRANCH,
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
    if (
        outcome.get("repo_id") != EXPECTED_REPO_ID
        or outcome.get("branch") != KERNEL_RELEASE_BRANCH
    ):
        raise PublicationError("kernel-builder outcome names an unexpected target")
    if outcome.get("status") not in {"uploaded", "no_changes"}:
        raise PublicationError("kernel-builder did not complete a direct upload")


def _download_bytes(
    download_fn: Callable[..., str],
    *,
    repo_type: str,
    path: str,
    revision: str,
    token: str | None,
) -> bytes:
    downloaded = Path(
        download_fn(
            EXPECTED_REPO_ID,
            path,
            repo_type=repo_type,
            revision=revision,
            token=token,
        )
    )
    return downloaded.read_bytes()


def targets_for(contract: dict[str, Any], repo_type: str) -> list[dict[str, str]]:
    return [
        target
        for target in contract["publication_targets"]
        if target["repo_type"] == repo_type
    ]


def repository_paths(
    api: HfApi,
    *,
    repo_type: str,
    revision: str,
    token: str | None,
) -> set[str]:
    return {
        entry.path
        for entry in api.list_repo_tree(
            EXPECTED_REPO_ID,
            repo_type=repo_type,
            revision=revision,
            recursive=True,
            token=token,
        )
        if getattr(entry, "path", None)
    }


def observed_target_files(
    api: HfApi,
    contract: dict[str, Any],
    *,
    repo_type: str,
    revision: str,
    token: str | None,
    download_fn: Callable[..., str],
) -> list[dict[str, Any]]:
    paths = repository_paths(
        api,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )
    files = []
    for target in targets_for(contract, repo_type):
        path = target["path_in_repo"]
        if path not in paths:
            files.append({"path": path, "status": "MISSING"})
            continue
        payload = _download_bytes(
            download_fn,
            repo_type=repo_type,
            path=path,
            revision=revision,
            token=token,
        )
        files.append(
            {
                "path": path,
                "status": "PRESENT",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return files


def hub_before(
    api: HfApi,
    contract: dict[str, Any],
    *,
    token: str | None,
    download_fn: Callable[..., str],
) -> dict[str, Any]:
    model_revision = model_head(api, token)
    kernel_branches = kernel_branch_targets(api, token=token)
    kernel_revision = kernel_branches[KERNEL_RELEASE_BRANCH]
    return {
        MODEL_REPO_TYPE: {
            "revision": model_revision,
            "files": observed_target_files(
                api,
                contract,
                repo_type=MODEL_REPO_TYPE,
                revision=model_revision,
                token=token,
                download_fn=download_fn,
            ),
        },
        KERNEL_REPO_TYPE: {
            "branch": KERNEL_RELEASE_BRANCH,
            "revision": kernel_revision,
            "branches": kernel_branches,
            "files": observed_target_files(
                api,
                contract,
                repo_type=KERNEL_REPO_TYPE,
                revision=kernel_revision,
                token=token,
                download_fn=download_fn,
            ),
        },
    }


def verify_model_readback(
    api: HfApi,
    source_root: Path,
    contract: dict[str, Any],
    *,
    revision: str,
    token: str,
    download_fn: Callable[..., str],
) -> None:
    if model_head(api, token) != revision:
        raise PublicationError("Hub model head changed after publication")
    for target in targets_for(contract, MODEL_REPO_TYPE):
        expected = safe_file(source_root, target["source_path"]).read_bytes()
        observed = _download_bytes(
            download_fn,
            repo_type=MODEL_REPO_TYPE,
            path=target["path_in_repo"],
            revision=revision,
            token=token,
        )
        if observed != expected:
            raise PublicationError(
                f"Hub model readback mismatch at {target['path_in_repo']}"
            )


def verify_kernel_readback(
    api: HfApi,
    expected: dict[str, bytes],
    *,
    revision: str,
    expected_main: str,
    token: str,
    download_fn: Callable[..., str],
) -> dict[str, str]:
    branches = kernel_branch_targets(api, token=token)
    if branches["main"] != expected_main:
        raise PublicationError("first-class Kernel default head changed during v1 upload")
    if branches[KERNEL_RELEASE_BRANCH] != revision:
        raise PublicationError("first-class Kernel v1 head changed after publication")
    for path, wanted in sorted(expected.items()):
        observed = _download_bytes(
            download_fn,
            repo_type=KERNEL_REPO_TYPE,
            path=path,
            revision=revision,
            token=token,
        )
        if observed != wanted:
            raise PublicationError(
                f"first-class Kernel v1 readback mismatch at {path}"
            )
    return branches


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
    result: dict[str, Any] = {
        "schema": "szl.invariants-publication-report/v1",
        "mode": "PUBLISH" if publish else "DRY_RUN",
        "status": "VERIFIED_DRY_RUN",
        "repo_id": EXPECTED_REPO_ID,
        "source_repository": EXPECTED_SOURCE_REPOSITORY,
        "source_revision": source_revision,
        "publisher": publisher,
        "authorization": authorization,
        "artifact_tree_sha256": tree_sha256(files),
        "declared_files": files,
        "observed_before": observed_before,
        "targets": {
            MODEL_REPO_TYPE: {
                "status": "NOT_PUBLISHED",
                "revision_before": observed_before[MODEL_REPO_TYPE]["revision"],
                "publication_interface": "huggingface_hub.create_commit",
            },
            KERNEL_REPO_TYPE: {
                "status": "NOT_PUBLISHED",
                "branch": KERNEL_RELEASE_BRANCH,
                "revision_before": observed_before[KERNEL_REPO_TYPE]["revision"],
                "branches_before": observed_before[KERNEL_REPO_TYPE]["branches"],
                "publication_interface": "kernel-builder",
                "publication_interface_version": KERNEL_BUILDER_VERSION,
                "publication_interface_source_revision": (
                    KERNEL_BUILDER_SOURCE_REVISION
                ),
            },
        },
    }
    if not publish:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(canonical_json(result), encoding="utf-8")
        return result
    if not token:
        raise PublicationError("HF_TOKEN is required when --publish is used")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    result["status"] = "PUBLICATION_IN_PROGRESS"
    report_path.write_text(canonical_json(result), encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="szl-invariants-kernel-") as temporary:
        staging_root = Path(temporary)
        expected_kernel_files = stage_kernel_targets(
            source_root,
            contract,
            staging_root,
        )
        expected_branches = observed_before[KERNEL_REPO_TYPE]["branches"]
        revalidated = revalidate_kernel_branch_parents(
            api,
            expected_branches,
            token=token,
        )
        result["targets"][KERNEL_REPO_TYPE]["status"] = "PUBLISHING_V1"
        result["targets"][KERNEL_REPO_TYPE][
            "parents_revalidated_before_upload"
        ] = revalidated
        result["targets"][KERNEL_REPO_TYPE]["staged_files"] = sorted(
            expected_kernel_files
        )
        report_path.write_text(canonical_json(result), encoding="utf-8")
        upload_error: Exception | None = None
        try:
            kernel_upload_fn(staging_root, token)
        except Exception as exc:  # preserve branch state after a partial upload
            upload_error = exc
        branches_after = kernel_branch_targets(api, token=token)
        result["targets"][KERNEL_REPO_TYPE]["branches_after"] = branches_after
        result["targets"][KERNEL_REPO_TYPE]["revision_after"] = branches_after[
            KERNEL_RELEASE_BRANCH
        ]
        result["targets"][KERNEL_REPO_TYPE]["status"] = "V1_READBACK_PENDING"
        report_path.write_text(canonical_json(result), encoding="utf-8")
        if upload_error is not None:
            raise upload_error
        verified_branches = verify_kernel_readback(
            api,
            expected_kernel_files,
            revision=branches_after[KERNEL_RELEASE_BRANCH],
            expected_main=expected_branches["main"],
            token=token,
            download_fn=download_fn,
        )
        result["targets"][KERNEL_REPO_TYPE]["verified_branches"] = (
            verified_branches
        )
        result["targets"][KERNEL_REPO_TYPE]["status"] = (
            "V1_EXACT_READBACK_VERIFIED"
        )
        report_path.write_text(canonical_json(result), encoding="utf-8")

    model_parent = observed_before[MODEL_REPO_TYPE]["revision"]
    if model_head(api, token) != model_parent:
        raise PublicationError("Hub model parent changed before serialized publication")
    model_operations = [
        CommitOperationAdd(
            path_in_repo=target["path_in_repo"],
            path_or_fileobj=str(safe_file(source_root, target["source_path"])),
        )
        for target in targets_for(contract, MODEL_REPO_TYPE)
    ]
    result["targets"][MODEL_REPO_TYPE]["status"] = "PUBLISHING"
    report_path.write_text(canonical_json(result), encoding="utf-8")
    model_commit = api.create_commit(
        repo_id=EXPECTED_REPO_ID,
        repo_type=MODEL_REPO_TYPE,
        parent_commit=model_parent,
        operations=model_operations,
        commit_message=f"Publish authorized invariants source {source_revision[:12]}",
        token=token,
    )
    model_revision = str(
        getattr(model_commit, "oid", "") or model_head(api, token)
    ).lower()
    if FULL_SHA_RE.fullmatch(model_revision) is None:
        raise PublicationError("Hub model publication returned no immutable SHA")
    result["targets"][MODEL_REPO_TYPE]["revision_after"] = model_revision
    result["targets"][MODEL_REPO_TYPE]["status"] = "READBACK_PENDING"
    report_path.write_text(canonical_json(result), encoding="utf-8")
    verify_model_readback(
        api,
        source_root,
        contract,
        revision=model_revision,
        token=token,
        download_fn=download_fn,
    )
    result["targets"][MODEL_REPO_TYPE]["status"] = "EXACT_READBACK_VERIFIED"
    report_path.write_text(canonical_json(result), encoding="utf-8")

    result["status"] = "PUBLISHED_AND_EXACT_READBACK_VERIFIED"
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
        default=Path("reports/invariants-publication.json"),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    identity = publisher_identity(
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
        publisher=identity,
        publish=args.publish,
        token=os.getenv("HF_TOKEN"),
    )
    print(canonical_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
