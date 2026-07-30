#!/usr/bin/env python3
"""Publish authorized SZL kernels data using only trusted Forge code."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

EXPECTED_REPO_ID = "SZLHOLDINGS/szl-kernels"
EXPECTED_SOURCE_REPOSITORY = "szl-holdings/szl-kernels"
EXPECTED_PUBLISHER_REPOSITORY = "szl-holdings/szl-forge"
CONTRACT_RELATIVE = Path("publishing/source-binding.json")
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


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


def hub_before(
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
                repo_type="model",
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


def verify_readback(
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
                repo_type="model",
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
    publication = {
        "schema": "szl.hf-kernel-source-binding/v2",
        "artifact": {
            "repo_id": EXPECTED_REPO_ID,
            "repo_type": "model",
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
        "observed_hub_before_publication": observed_before,
        "claims": contract["claims"],
        "limitations": contract["limitations"],
    }
    publication_bytes = canonical_json(publication).encode("utf-8")
    result: dict[str, Any] = {
        "schema": "szl.kernel-source-binding-report/v2",
        "mode": "PUBLISH" if publish else "DRY_RUN",
        "repo_id": EXPECTED_REPO_ID,
        "source_revision": source_revision,
        "publisher": publisher,
        "artifact_tree_sha256": tree_sha256(files),
        "declared_file_count": len(files),
        "hub_revision_before": observed_before["revision"],
        "publication_sha256": hashlib.sha256(publication_bytes).hexdigest(),
        "status": "VERIFIED_DRY_RUN",
    }
    if publish:
        if not token:
            raise PublicationError("HF_TOKEN is required when --publish is used")
        operations = [
            CommitOperationAdd(
                path_in_repo=relative,
                path_or_fileobj=str(safe_file(source_root, relative)),
            )
            for relative in contract["artifact_files"]
        ]
        operations.append(
            CommitOperationAdd(
                path_in_repo="publication.json",
                path_or_fileobj=io.BytesIO(publication_bytes),
            )
        )
        commit = api.create_commit(
            repo_id=EXPECTED_REPO_ID,
            repo_type="model",
            operations=operations,
            commit_message=f"Publish authorized source {source_revision[:12]}",
            token=token,
        )
        revision = getattr(commit, "oid", None)
        if not revision:
            revision = api.model_info(EXPECTED_REPO_ID, token=token).sha
        verify_readback(
            source_root,
            contract,
            publication_bytes,
            revision=revision,
            token=token,
            download_fn=download_fn,
        )
        result.update(
            {
                "hub_revision_after": revision,
                "status": "PUBLISHED_AND_EXACT_READBACK_VERIFIED",
            }
        )
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
