#!/usr/bin/env python3
"""Publish authorized SZL kernels data using only trusted Forge code."""

from __future__ import annotations

import argparse
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
CONTRACT_RELATIVE = Path("publishing/source-binding.json")
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
LEGACY_REPO_TYPE = "model"
KERNEL_REPO_TYPE = "kernel"
KERNEL_BRANCHES = ("main", "v1")
KERNEL_VERSION_BRANCH = "v1"
KERNEL_BINDING_PATH = "build/torch-cpu/source-binding.json"
KERNEL_BUILDER_BINARY = "kernel-builder"
FIRST_CLASS_KERNEL_FILES = {
    "README.md": "README.md",
    "build/torch-universal/szl_kernels/__init__.py": (
        "build/torch-cpu/szl_kernels/__init__.py"
    ),
    "build/torch-universal/szl_kernels/_chain.py": (
        "build/torch-cpu/szl_kernels/_chain.py"
    ),
    "build/torch-universal/szl_kernels/_ops.py": (
        "build/torch-cpu/szl_kernels/_ops.py"
    ),
    "build/torch-universal/szl_kernels/metadata.json": (
        "build/torch-cpu/metadata.json"
    ),
}

KERNEL_BRANCH_FILES = {
    "main": {"README.md"},
    KERNEL_VERSION_BRANCH: set(FIRST_CLASS_KERNEL_FILES.values()) - {"README.md"}
    | {KERNEL_BINDING_PATH},
}
KERNEL_PREFLIGHT_BRANCH_FILES = {
    branch: paths - {KERNEL_BINDING_PATH}
    for branch, paths in KERNEL_BRANCH_FILES.items()
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
        raise PublicationError(
            "source authorization does not bind the requested revision"
        )
    if (
        publisher.get("repository") != EXPECTED_PUBLISHER_REPOSITORY
        or publisher.get("revision") != publisher_revision
        or publisher.get("protected_main") != publisher_revision
    ):
        raise PublicationError(
            "publisher authorization does not bind this Forge revision"
        )
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
        "declared_files_present": len(set(contract["artifact_files"]) & observed_files),
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
            f"source contract is missing first-class Kernel files: {missing_sources}"
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
        expected_paths = KERNEL_PREFLIGHT_BRANCH_FILES[branch]
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
    if not workflow_ref.startswith(f"{EXPECTED_PUBLISHER_REPOSITORY}/{workflow_path}@"):
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
    evidence = []
    for source_path, kernel_path in FIRST_CLASS_KERNEL_FILES.items():
        path = safe_file(source_root, source_path)
        evidence.append(
            {
                "source_path": source_path,
                "kernel_path": kernel_path,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return evidence


def verify_kernel_readback(
    source_root: Path,
    binding_bytes: bytes,
    *,
    branch: str,
    revision: str,
    token: str,
    download_fn: Callable[..., str],
) -> None:
    for source_path, kernel_path in FIRST_CLASS_KERNEL_FILES.items():
        if kernel_path not in KERNEL_BRANCH_FILES[branch]:
            continue
        downloaded = Path(
            download_fn(
                EXPECTED_REPO_ID,
                kernel_path,
                repo_type=KERNEL_REPO_TYPE,
                revision=revision,
                token=token,
            )
        )
        if downloaded.read_bytes() != safe_file(source_root, source_path).read_bytes():
            raise PublicationError(
                f"first-class Kernel {branch} readback mismatch at {kernel_path}"
            )
    if KERNEL_BINDING_PATH in KERNEL_BRANCH_FILES[branch]:
        binding = Path(
            download_fn(
                EXPECTED_REPO_ID,
                KERNEL_BINDING_PATH,
                repo_type=KERNEL_REPO_TYPE,
                revision=revision,
                token=token,
            )
        )
        if binding.read_bytes() != binding_bytes:
            raise PublicationError(
                f"first-class Kernel {branch} readback mismatch at {KERNEL_BINDING_PATH}"
            )


def stage_kernel_builder_upload(
    source_root: Path,
    binding_bytes: bytes,
    destination: Path,
) -> Path:
    """Create the exact official kernel-builder upload layout."""

    for source_path, kernel_path in FIRST_CLASS_KERNEL_FILES.items():
        target = destination / kernel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_file(source_root, source_path), target)
    binding = destination / KERNEL_BINDING_PATH
    binding.parent.mkdir(parents=True, exist_ok=True)
    binding.write_bytes(binding_bytes)
    return destination


def run_kernel_builder_upload(
    source_root: Path,
    binding_bytes: bytes,
    *,
    token: str,
) -> dict[str, Any]:
    """Publish through Hugging Face's supported first-class Kernel client."""

    with tempfile.TemporaryDirectory(prefix="szl-kernel-upload-") as temporary:
        stage = stage_kernel_builder_upload(
            source_root,
            binding_bytes,
            Path(temporary) / "kernel",
        )
        output = Path(temporary) / "kernel-builder-output.json"
        environment = dict(os.environ)
        environment["HF_TOKEN"] = token
        completed = subprocess.run(
            [
                KERNEL_BUILDER_BINARY,
                "upload",
                str(stage),
                "--repo-id",
                EXPECTED_REPO_ID,
                "--branch",
                KERNEL_VERSION_BRANCH,
                "--repo-type",
                KERNEL_REPO_TYPE,
                "--output-json",
                str(output),
                "--quiet",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise PublicationError(
                "official kernel-builder upload failed"
                + (f": {detail[-2000:]}" if detail else "")
            )
        if not output.is_file():
            raise PublicationError("kernel-builder did not write its output receipt")
        receipt = json.loads(output.read_text(encoding="utf-8"))
        if (
            receipt.get("status") not in {"uploaded", "no_changes"}
            or receipt.get("repo_id") != EXPECTED_REPO_ID
            or receipt.get("branch") != KERNEL_VERSION_BRANCH
            or receipt.get("pull_requests") != []
        ):
            raise PublicationError(
                "kernel-builder returned an unexpected upload receipt"
            )
        return receipt


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
    kernel_upload_fn: Callable[..., dict[str, Any]] = run_kernel_builder_upload,
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
            "version": json.loads(
                safe_file(
                    source_root,
                    "build/torch-universal/szl_kernels/metadata.json",
                ).read_text(encoding="utf-8")
            )["version"],
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
                "mapped_file_count": len(kernel_files),
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
        report_path.write_text(canonical_json(result), encoding="utf-8")

        before_refs = {
            branch.name: branch.target_commit
            for branch in api.list_repo_refs(
                EXPECTED_REPO_ID,
                repo_type=KERNEL_REPO_TYPE,
                token=token,
            ).branches
        }
        expected_refs = {
            branch: observed_before["first_class_kernel"]["branches"][branch][
                "revision"
            ]
            for branch in KERNEL_BRANCHES
        }
        if {
            branch: before_refs.get(branch) for branch in KERNEL_BRANCHES
        } != expected_refs:
            raise PublicationError(
                "first-class Kernel branches moved after authorization preflight"
            )

        kernel_receipt = kernel_upload_fn(
            source_root,
            kernel_binding_bytes,
            token=token,
        )
        result["targets"]["first_class_kernel"]["publisher"] = {
            "client": "kernel-builder",
            "receipt": kernel_receipt,
        }

        after_refs = {
            branch.name: branch.target_commit
            for branch in api.list_repo_refs(
                EXPECTED_REPO_ID,
                repo_type=KERNEL_REPO_TYPE,
                token=token,
            ).branches
        }
        for branch in KERNEL_BRANCHES:
            kernel_revision = after_refs.get(branch)
            if not kernel_revision:
                raise PublicationError(
                    f"kernel-builder upload did not leave branch {branch} readable"
                )
            result["targets"]["first_class_kernel"]["branches_after"][branch] = (
                kernel_revision
            )
            result["targets"]["first_class_kernel"]["readback"][branch] = "PENDING"
            report_path.write_text(canonical_json(result), encoding="utf-8")
            verify_kernel_readback(
                source_root,
                kernel_binding_bytes,
                branch=branch,
                revision=kernel_revision,
                token=token,
                download_fn=download_fn,
            )
            result["targets"]["first_class_kernel"]["readback"][branch] = (
                "EXACT_BYTES_VERIFIED"
            )
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
