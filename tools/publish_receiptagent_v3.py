#!/usr/bin/env python3
"""Validate and CAS-publish one authenticated ReceiptAgent v3 release packet.

Dry-run mode is local-only.  Publish mode requires an explicit Hub parent and
token, uses one atomic commit, then reads every accounted byte back at the
immutable resulting revision before writing a local publication receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Callable, Iterable

try:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
except ModuleNotFoundError:
    HfApi = None
    hf_hub_download = None

    @dataclass(frozen=True)
    class CommitOperationAdd:
        path_in_repo: str
        path_or_fileobj: Any


TARGET_REPO_ID = "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3-authenticated"
TARGET_REPO_TYPE = "model"
CANDIDATE_ID = "SZL-ReceiptAgent-Qwen3.5-0.8B-v3"
MANIFEST_SCHEMA = "szl.receiptagent-v3-release-manifest/v1"
RECEIPT_SCHEMA = "szl.receiptagent-v3-hf-publication-receipt/v1"
FAILURE_RECEIPT_SCHEMA = (
    "szl.receiptagent-v3-hf-publication-failure-receipt/v1"
)
ABSENT_PARENT = "ABSENT"
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
KEY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")
MAX_MANIFEST_BYTES = 32 * 1024 * 1024
MAX_INVENTORY_FILE_BYTES = 512 * 1024 * 1024
MAX_TRUSTED_PUBLIC_KEY_BYTES = 64 * 1024
MAX_SOURCE_FILE_BYTES = 2 * 1024 * 1024
MAX_GITATTRIBUTES_BYTES = 64 * 1024
READ_CHUNK_BYTES = 1024 * 1024
REPOSITORY = Path(__file__).resolve().parents[1]
RELEASE_DIRECTORY = REPOSITORY / "frontier" / "qwen35-receiptagent-v3"
RELEASE_MANIFEST_FILENAME = "release-manifest.json"
TRUST_POLICY_PATH = (
    "frontier/qwen35-receiptagent-v3/receipt-signing-trust-policy.json"
)
SOURCE_BOUND_PATHS = (
    "tools/publish_receiptagent_v3.py",
    "frontier/qwen35-receiptagent-v3/evidence_chain.py",
    "frontier/qwen35-receiptagent-v3/prepare_release.py",
    "frontier/qwen35-receiptagent-v3/release.schema.json",
    "frontier/qwen35-receiptagent-v3/schemas/authenticated-receipt.schema.json",
    "frontier/qwen35-receiptagent-v3/candidate.json",
    TRUST_POLICY_PATH,
)
MANIFEST_KEYS = {
    "schema",
    "candidateId",
    "state",
    "targetRepository",
    "sourceRevision",
    "runId",
    "authenticatedEvidenceChainValid",
    "receiptSigningTrustPolicy",
    "receiptEligible",
    "publicationEligible",
    "deploymentEligible",
    "runtimeValidated",
    "thirdPartyValidated",
    "lineage",
    "adapter",
    "benchmark",
    "reports",
    "receipts",
    "intendedRepositoryFiles",
    "localEvidenceDisposition",
    "localEvidenceFiles",
    "releaseManifestIsTargetRepositoryFile",
    "unmetPublicationGates",
    "claimBoundary",
    "manifestSha256",
}
TRUST_POLICY_IDENTITY_KEYS = {
    "policySchema",
    "policySha256",
    "candidateId",
    "algorithm",
    "keyId",
    "publicKeyFingerprintSha256",
    "usage",
    "state",
}
TARGET_KEYS = {
    "repoId",
    "repoType",
    "private",
    "declaredOnly",
    "immutableRevision",
    "immutableReadbackVerified",
}
RECEIPT_FILES = {
    "TRAINING": ("training-receipt.json", "AUTHENTICATED_TRAINING_RECEIPT"),
    "EVALUATION": ("evaluation-receipt.json", "AUTHENTICATED_EVALUATION_RECEIPT"),
    "COMPARISON": ("comparison-receipt.json", "AUTHENTICATED_COMPARISON_RECEIPT"),
}
REPORT_FILES = {
    "childTraining": ("training-report.json", "MEASURED_CHILD_TRAINING_REPORT"),
    "supervisor": ("supervisor-report.json", "MEASURED_SUPERVISOR_REPORT"),
    "devEvaluation": ("dev-evaluation-report.json", "MEASURED_DEV_EVALUATION_REPORT"),
    "testEvaluation": ("test-evaluation-report.json", "MEASURED_TEST_EVALUATION_REPORT"),
    "baseTestEvaluation": ("base-test-evaluation-report.json", "MEASURED_BASE_TEST_EVALUATION_REPORT"),
    "v2TestEvaluation": ("v2-test-evaluation-report.json", "MEASURED_V2_TEST_EVALUATION_REPORT"),
    "comparison": ("comparison-report.json", "MEASURED_COMPARISON_REPORT"),
}
PUBLIC_FIXED_FILE_ROLES = {
    "README.md": "MEASURED_MODEL_CARD",
    **{path: role for path, role in RECEIPT_FILES.values()},
}
ADAPTER_SOURCE_README = "adapter-source-readme.md"
ADAPTER_SOURCE_README_ROLE = "ADAPTER_SOURCE_README"
LOCAL_EVIDENCE_FILE_ROLES = {
    **{path: role for path, role in REPORT_FILES.values()},
    ADAPTER_SOURCE_README: ADAPTER_SOURCE_README_ROLE,
}


class PublicationError(RuntimeError):
    """Raised when publication cannot continue without weakening evidence."""


@dataclass(frozen=True)
class FileIdentity:
    path: str
    role: str
    size: int
    sha256: str
    body: bytes


@dataclass(frozen=True)
class SourceBinding:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class PreparedRelease:
    manifest_path: Path
    manifest_sha256: str
    chain_sha256: str
    private: bool
    source_revision: str
    source_bindings: tuple[SourceBinding, ...]
    trust_policy_sha256: str
    trust_key_id: str
    trust_key_fingerprint_sha256: str
    files: tuple[FileIdentity, ...]
    local_evidence_files: tuple[FileIdentity, ...]


def canonical_json(payload: Any) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PublicationError(f"value is not canonical JSON: {exc}") from exc
    return (rendered + "\n").encode("utf-8")


def compact_json(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PublicationError(f"value is not compact canonical JSON: {exc}") from exc


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute path without resolving links or reparse points."""

    return Path(os.path.abspath(os.fspath(path)))


def _metadata_is_reparse(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & marker)


def _path_is_junction(path: Path, label: str) -> bool:
    checker = getattr(path, "is_junction", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except OSError as exc:
        raise PublicationError(f"{label} junction status could not be inspected") from exc


def _lstat_non_reparse(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise PublicationError(f"{label} does not exist") from exc
    except OSError as exc:
        raise PublicationError(f"{label} metadata could not be inspected") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _metadata_is_reparse(metadata)
        or _path_is_junction(path, label)
    ):
        raise PublicationError(
            f"{label} must not be a symlink, junction, or reparse point"
        )
    return metadata


def _assert_non_reparse_ancestors(path: Path, label: str) -> Path:
    absolute = _lexical_absolute(path)
    for ancestor in reversed(absolute.parents):
        metadata = _lstat_non_reparse(ancestor, f"{label} ancestor {ancestor}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise PublicationError(f"{label} ancestor is not a directory: {ancestor}")
    return absolute


def _metadata_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _bounded_regular_file_read(
    path: Path,
    *,
    label: str,
    max_bytes: int,
    expected_size: int | None = None,
) -> bytes:
    """Read one regular file through one descriptor after bounding its size."""

    if max_bytes < 1:
        raise PublicationError(f"{label} maximum size is invalid")
    absolute = _assert_non_reparse_ancestors(path, label)
    before = _lstat_non_reparse(absolute, label)
    if not stat.S_ISREG(before.st_mode):
        raise PublicationError(f"{label} must be a regular file")
    if before.st_size < 0 or before.st_size > max_bytes:
        raise PublicationError(f"{label} is oversized")
    if expected_size is not None and before.st_size != expected_size:
        raise PublicationError(f"{label} size drift from the manifest")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise PublicationError(f"{label} could not be opened securely") from exc
    try:
        opened = os.fstat(descriptor)
        if _metadata_is_reparse(opened) or not stat.S_ISREG(opened.st_mode):
            raise PublicationError(f"{label} descriptor is not a regular non-reparse file")
        if _metadata_identity(opened) != _metadata_identity(before):
            raise PublicationError(f"{label} changed while it was opened")
        if opened.st_size > max_bytes:
            raise PublicationError(f"{label} is oversized")
        if expected_size is not None and opened.st_size != expected_size:
            raise PublicationError(f"{label} size drift from the manifest")

        remaining = opened.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, READ_CHUNK_BYTES))
            if not chunk:
                raise PublicationError(f"{label} was shortened while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise PublicationError(f"{label} grew while being read")
        after = os.fstat(descriptor)
        if _metadata_identity(after) != _metadata_identity(opened):
            raise PublicationError(f"{label} changed while being read")
        return b"".join(chunks)
    except OSError as exc:
        raise PublicationError(f"{label} could not be read securely") from exc
    finally:
        os.close(descriptor)


def _git_blob(expected_source_revision: str, relative_path: str) -> bytes:
    if FULL_SHA_RE.fullmatch(expected_source_revision or "") is None:
        raise PublicationError("expected source revision must be exact 40-hex")
    if relative_path not in SOURCE_BOUND_PATHS:
        raise PublicationError("source binding path is not allowlisted")
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                os.fspath(REPOSITORY),
                "cat-file",
                "blob",
                f"{expected_source_revision}:{relative_path}",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PublicationError("exact Git source binding could not be read") from exc
    if completed.returncode != 0:
        raise PublicationError(
            f"committed source binding is unavailable: {relative_path}"
        )
    if len(completed.stdout) > MAX_SOURCE_FILE_BYTES:
        raise PublicationError(f"committed source binding is oversized: {relative_path}")
    return completed.stdout


def _assert_exact_source_bindings(
    expected_source_revision: str,
) -> tuple[SourceBinding, ...]:
    if FULL_SHA_RE.fullmatch(expected_source_revision or "") is None:
        raise PublicationError("expected source revision must be exact 40-hex")
    bindings: list[SourceBinding] = []
    for relative_path in SOURCE_BOUND_PATHS:
        local = _bounded_regular_file_read(
            REPOSITORY / PurePosixPath(relative_path),
            label=f"local source binding {relative_path}",
            max_bytes=MAX_SOURCE_FILE_BYTES,
        )
        committed = _git_blob(expected_source_revision, relative_path)
        if local != committed:
            raise PublicationError(
                f"local source differs from exact revision: {relative_path}"
            )
        bindings.append(
            SourceBinding(
                path=relative_path,
                size=len(local),
                sha256=sha256_bytes(local),
            )
        )
    return tuple(bindings)


def _load_trust_policy(
    expected_source_revision: str,
) -> tuple[dict[str, Any], str]:
    body = _git_blob(expected_source_revision, TRUST_POLICY_PATH)
    policy = _strict_json_bytes(body, "receipt signing trust policy")
    _exact_keys(
        policy,
        {
            "schema",
            "candidateId",
            "algorithm",
            "keyId",
            "publicKeyFingerprintSha256",
            "usage",
            "state",
        },
        "receipt signing trust policy",
    )
    expected = {
        "schema": "szl.receiptagent-v3-receipt-signing-trust-policy/v1",
        "candidateId": CANDIDATE_ID,
        "algorithm": "Ed25519",
        "usage": "AUTHENTICATED_TRAINING_EVALUATION_COMPARISON_RECEIPTS",
        "state": "ACTIVE",
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise PublicationError(f"receipt signing trust policy {key} differs")
    if KEY_ID_RE.fullmatch(policy.get("keyId", "")) is None:
        raise PublicationError("receipt signing trust policy keyId is invalid")
    _require_sha256(
        policy.get("publicKeyFingerprintSha256"),
        "receipt signing trust policy fingerprint",
    )
    return policy, sha256_bytes(body)


def _assert_manifest_trust_policy_identity(
    document: dict[str, Any], trust_policy: dict[str, Any]
) -> None:
    observed = _exact_keys(
        document.get("receiptSigningTrustPolicy"),
        TRUST_POLICY_IDENTITY_KEYS,
        "receiptSigningTrustPolicy",
    )
    expected = {
        "policySchema": trust_policy["schema"],
        "policySha256": sha256_bytes(compact_json(trust_policy)),
        "candidateId": trust_policy["candidateId"],
        "algorithm": trust_policy["algorithm"],
        "keyId": trust_policy["keyId"],
        "publicKeyFingerprintSha256": trust_policy[
            "publicKeyFingerprintSha256"
        ],
        "usage": trust_policy["usage"],
        "state": trust_policy["state"],
    }
    if observed != expected:
        raise PublicationError(
            "release manifest receipt signing trust policy differs from exact source policy"
        )


def _enumerate_packet_files(root: Path, manifest_path: Path) -> set[str]:
    root = _assert_non_reparse_ancestors(root, "release directory")
    root_metadata = _lstat_non_reparse(root, "release directory")
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise PublicationError("release directory must be a regular directory")
    ignored = os.path.normcase(os.fspath(_lexical_absolute(manifest_path)))
    actual: set[str] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise PublicationError(f"release directory could not be enumerated: {directory}") from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise PublicationError(f"release entry could not be inspected: {path}") from exc
            if (
                stat.S_ISLNK(metadata.st_mode)
                or _metadata_is_reparse(metadata)
                or _path_is_junction(path, f"release entry {path}")
            ):
                raise PublicationError(
                    f"release directory contains a symlink, junction, or reparse point: {path}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                if os.path.normcase(os.fspath(_lexical_absolute(path))) != ignored:
                    actual.add(path.relative_to(root).as_posix())
            else:
                raise PublicationError(f"release directory contains a non-regular entry: {path}")
    return actual


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PublicationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_json_bytes(body: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            body.decode("utf-8", errors="strict"),
            object_pairs_hook=_object_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PublicationError(f"{label} contains non-finite number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PublicationError(f"{label} must be one JSON object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PublicationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PublicationError(f"{label} must be a non-empty POSIX path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or value != parsed.as_posix() or any(
        part in {"", ".", ".."} for part in parsed.parts
    ):
        raise PublicationError(f"{label} is not a canonical relative path: {value!r}")
    if value == ".gitattributes":
        raise PublicationError(".gitattributes is Hub-managed and cannot be in the packet")
    return value


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PublicationError(f"{label} must be one object")
    observed = set(value)
    if observed != expected:
        raise PublicationError(
            f"{label} keys differ (missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)})"
        )
    return value


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PublicationError(f"cannot load local release module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _release_modules() -> tuple[ModuleType, ModuleType]:
    evidence_module = _load_module(
        "_szl_ra3_evidence_chain", RELEASE_DIRECTORY / "evidence_chain.py"
    )
    missing = object()
    previous = sys.modules.get("evidence_chain", missing)
    sys.modules["evidence_chain"] = evidence_module
    try:
        release_module = _load_module(
            "_szl_ra3_prepare_release", RELEASE_DIRECTORY / "prepare_release.py"
        )
    finally:
        if previous is missing:
            sys.modules.pop("evidence_chain", None)
        else:
            sys.modules["evidence_chain"] = previous
    return evidence_module, release_module


def _adapt_manifest(
    document: Any, release_module: ModuleType
) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]], str]:
    document = _exact_keys(document, MANIFEST_KEYS, "release manifest")
    if dict(release_module.RECEIPT_OUTPUTS) != RECEIPT_FILES:
        raise PublicationError("publisher receipt filename/role contract drifted")
    if dict(release_module.REPORT_OUTPUTS) != REPORT_FILES:
        raise PublicationError("publisher report filename/role contract drifted")
    if document.get("schema") != MANIFEST_SCHEMA:
        raise PublicationError(f"release manifest schema must be {MANIFEST_SCHEMA}")
    if document.get("candidateId") != CANDIDATE_ID:
        raise PublicationError("release manifest candidate differs")
    if document.get("state") != "OFFLINE_RELEASE_PACKET_PREPARED_NOT_PUBLISHED":
        raise PublicationError("release manifest state differs")
    if FULL_SHA_RE.fullmatch(document.get("sourceRevision") or "") is None:
        raise PublicationError("release manifest sourceRevision must be exact 40-hex")
    if re.fullmatch(r"[0-9a-f]{32}", document.get("runId") or "") is None:
        raise PublicationError("release manifest runId must be exact 32-hex")

    target = _exact_keys(document.get("targetRepository"), TARGET_KEYS, "targetRepository")
    if target.get("repoId") != TARGET_REPO_ID:
        raise PublicationError(f"target repoId must be {TARGET_REPO_ID}")
    if target.get("repoType") != TARGET_REPO_TYPE:
        raise PublicationError("target repoType must be model")
    private = target.get("private")
    if not isinstance(private, bool):
        raise PublicationError("target private must be an explicit boolean")
    if (
        target.get("declaredOnly") is not True
        or target.get("immutableRevision") is not None
        or target.get("immutableReadbackVerified") is not False
    ):
        raise PublicationError("targetRepository crossed the offline claim boundary")
    if document.get("authenticatedEvidenceChainValid") is not True:
        raise PublicationError("authenticated evidence-chain flag is not true")
    if document.get("receiptEligible") is not True:
        raise PublicationError("receipt eligibility flag is not true")
    for key in (
        "publicationEligible",
        "deploymentEligible",
        "runtimeValidated",
        "thirdPartyValidated",
    ):
        if document.get(key) is not False:
            raise PublicationError(f"release manifest {key} crossed the false claim gate")
    if document.get("releaseManifestIsTargetRepositoryFile") is not True:
        raise PublicationError("release manifest must declare itself a target file")
    for key in ("lineage", "benchmark"):
        if not isinstance(document.get(key), dict):
            raise PublicationError(f"release manifest {key} must be an object")
    if not isinstance(document.get("unmetPublicationGates"), list):
        raise PublicationError("unmetPublicationGates must be an array")
    if not isinstance(document.get("claimBoundary"), str) or not document["claimBoundary"]:
        raise PublicationError("claimBoundary must be a non-empty string")

    adapter = _exact_keys(
        document.get("adapter"),
        {"aggregateSha256", "manifestSha256", "sourceFiles", "sourceReadmeDisposition"},
        "adapter",
    )
    _require_sha256(adapter.get("aggregateSha256"), "adapter aggregateSha256")
    _require_sha256(adapter.get("manifestSha256"), "adapter manifestSha256")
    if not isinstance(adapter.get("sourceFiles"), list):
        raise PublicationError("adapter sourceFiles must be an array")

    reports = _exact_keys(document.get("reports"), set(REPORT_FILES), "reports")
    for key, digest in reports.items():
        _require_sha256(digest, f"reports.{key}")
    receipts = _exact_keys(
        document.get("receipts"), {kind.lower() for kind in RECEIPT_FILES}, "receipts"
    )
    for key, digest in receipts.items():
        _require_sha256(digest, f"receipts.{key}")
    chain_sha256 = sha256_bytes(compact_json(receipts))

    claimed_manifest_sha = _require_sha256(
        document.get("manifestSha256"), "manifestSha256"
    )
    unsigned = dict(document)
    unsigned.pop("manifestSha256")
    if sha256_bytes(compact_json(unsigned)) != claimed_manifest_sha:
        raise PublicationError("release manifest self-digest differs")

    public_inventory = document.get("intendedRepositoryFiles")
    if not isinstance(public_inventory, list) or not public_inventory:
        raise PublicationError("intendedRepositoryFiles must be a non-empty array")
    allowed_adapter_files = set(release_module.ALLOWED_ADAPTER_FILES) - {"README.md"}
    observed_public_roles: dict[str, str] = {}
    for index, entry in enumerate(public_inventory):
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256", "role"}:
            raise PublicationError(
                f"intendedRepositoryFiles[{index}] must contain exactly path, bytes, sha256, role"
            )
        path = _safe_relative_path(entry.get("path"), f"intendedRepositoryFiles[{index}].path")
        if path in observed_public_roles:
            raise PublicationError(f"duplicate public inventory path: {path}")
        role = entry.get("role")
        expected_role = PUBLIC_FIXED_FILE_ROLES.get(path)
        if path in allowed_adapter_files:
            expected_role = "ADAPTER_ARTIFACT"
        if expected_role is None or role != expected_role:
            raise PublicationError(f"{path}: public inventory role differs")
        observed_public_roles[path] = role
    required_public = set(PUBLIC_FIXED_FILE_ROLES) | {
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    if not required_public.issubset(observed_public_roles):
        raise PublicationError(
            "public release inventory omits required files: "
            f"{sorted(required_public - set(observed_public_roles))}"
        )

    if document.get("localEvidenceDisposition") != "NOT_FOR_PUBLICATION":
        raise PublicationError("localEvidenceDisposition must be NOT_FOR_PUBLICATION")
    local_inventory = document.get("localEvidenceFiles")
    if not isinstance(local_inventory, list) or not local_inventory:
        raise PublicationError("localEvidenceFiles must be a non-empty array")
    observed_local_roles: dict[str, str] = {}
    for index, entry in enumerate(local_inventory):
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "bytes",
            "sha256",
            "role",
            "disposition",
        }:
            raise PublicationError(
                f"localEvidenceFiles[{index}] must contain exactly path, bytes, sha256, role, disposition"
            )
        path = _safe_relative_path(entry.get("path"), f"localEvidenceFiles[{index}].path")
        if path in observed_local_roles or path in observed_public_roles:
            raise PublicationError(f"duplicate public/local inventory path: {path}")
        if entry.get("disposition") != "NOT_FOR_PUBLICATION":
            raise PublicationError(f"{path}: local evidence disposition differs")
        expected_role = LOCAL_EVIDENCE_FILE_ROLES.get(path)
        if expected_role is None or entry.get("role") != expected_role:
            raise PublicationError(f"{path}: local evidence role differs")
        observed_local_roles[path] = entry["role"]
    required_local = {path for path, _role in REPORT_FILES.values()}
    if not required_local.issubset(observed_local_roles):
        raise PublicationError(
            "local evidence inventory omits required reports: "
            f"{sorted(required_local - set(observed_local_roles))}"
        )
    has_source_readme = ADAPTER_SOURCE_README in observed_local_roles
    expected_disposition = (
        "LOCAL_EVIDENCE_ONLY_NOT_FOR_PUBLICATION"
        if has_source_readme
        else "SOURCE_README_ABSENT"
    )
    if adapter.get("sourceReadmeDisposition") != expected_disposition:
        raise PublicationError("adapter sourceReadmeDisposition differs")
    return private, public_inventory, local_inventory, chain_sha256


def _validate_receipt_destination(packet_root: Path, receipt_path: Path) -> Path:
    destination = _assert_non_reparse_ancestors(
        receipt_path, "publication receipt destination"
    )
    try:
        existing = destination.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise PublicationError("publication receipt destination could not be inspected") from exc
    else:
        if (
            stat.S_ISLNK(existing.st_mode)
            or _metadata_is_reparse(existing)
            or _path_is_junction(destination, "publication receipt destination")
        ):
            raise PublicationError(
                "publication receipt destination must not be a symlink, junction, or reparse point"
            )
        raise PublicationError("publication receipt destination must not already exist")
    parent = destination.parent
    parent_metadata = _lstat_non_reparse(parent, "publication receipt parent")
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise PublicationError("publication receipt parent must be an existing directory")
    packet_root = _assert_non_reparse_ancestors(packet_root, "release packet directory")
    packet_metadata = _lstat_non_reparse(packet_root, "release packet directory")
    if not stat.S_ISDIR(packet_metadata.st_mode):
        raise PublicationError("release packet directory must be a directory")
    try:
        destination.relative_to(packet_root)
    except ValueError:
        return destination
    raise PublicationError("publication receipt must be outside the release packet directory")


def _compact_document(identity: FileIdentity, label: str) -> dict[str, Any]:
    value = _strict_json_bytes(identity.body, label)
    if identity.body != compact_json(value) + b"\n":
        raise PublicationError(f"{label} bytes are not compact canonical JSON")
    return value


def _read_inventory_files(
    root: Path,
    inventory: list[dict[str, Any]],
    *,
    inventory_label: str,
) -> tuple[list[FileIdentity], set[str]]:
    identities: list[FileIdentity] = []
    declared: set[str] = set()
    for index, entry in enumerate(inventory):
        relative = _safe_relative_path(
            entry["path"], f"{inventory_label}[{index}].path"
        )
        if relative in declared:
            raise PublicationError(f"duplicate {inventory_label} path: {relative}")
        declared.add(relative)
        size = entry["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise PublicationError(f"{relative}: bytes must be a positive integer")
        if size > MAX_INVENTORY_FILE_BYTES:
            raise PublicationError(f"{relative}: declared size exceeds publisher limit")
        expected_sha = _require_sha256(entry["sha256"], f"{relative} sha256")
        body = _bounded_regular_file_read(
            root.joinpath(*PurePosixPath(relative).parts),
            label=f"inventory file {relative}",
            max_bytes=MAX_INVENTORY_FILE_BYTES,
            expected_size=size,
        )
        if sha256_bytes(body) != expected_sha:
            raise PublicationError(f"{relative}: local SHA-256 drift")
        identities.append(FileIdentity(relative, entry["role"], size, expected_sha, body))
    return identities, declared


def _rebuild_packet(
    *,
    document: dict[str, Any],
    packet_identities: tuple[FileIdentity, ...],
    trusted_public_key: Path,
    expected_source_revision: str,
    evidence_module: ModuleType,
    release_module: ModuleType,
    candidate_loader: Callable[[str], dict[str, Any]],
    trust_policy: dict[str, Any],
) -> None:
    by_path = {item.path: item for item in packet_identities}
    receipts = {
        kind: _compact_document(by_path[path], f"{kind} receipt")
        for kind, (path, _role) in RECEIPT_FILES.items()
    }
    reports = {
        key: _compact_document(by_path[path], key)
        for key, (path, _role) in REPORT_FILES.items()
    }
    try:
        chain = evidence_module.verify_chain(
            receipts["TRAINING"],
            receipts["EVALUATION"],
            receipts["COMPARISON"],
            trusted_public_key=trusted_public_key,
        )
    except Exception as exc:
        raise PublicationError(f"authenticated evidence verification failed: {exc}") from exc
    if chain.get("sourceRevision") != expected_source_revision:
        raise PublicationError("authenticated receipt source differs from expected source")
    for kind, receipt in receipts.items():
        authentication = receipt.get("authentication")
        if not isinstance(authentication, dict):
            raise PublicationError(f"{kind} receipt authentication is absent")
        if authentication.get("keyId") != trust_policy["keyId"]:
            raise PublicationError(
                f"{kind} receipt keyId is not authorized by source policy"
            )
        if (
            authentication.get("publicKeyFingerprintSha256")
            != trust_policy["publicKeyFingerprintSha256"]
        ):
            raise PublicationError(
                f"{kind} receipt key fingerprint is not authorized by source policy"
            )
    candidate = candidate_loader(expected_source_revision)
    if not isinstance(candidate, dict):
        raise PublicationError("exact candidate loader did not return one object")

    with tempfile.TemporaryDirectory(prefix="szl-ra3-publisher-") as temporary:
        adapter_dir = Path(temporary) / "adapter"
        adapter_dir.mkdir(mode=0o700)
        for item in packet_identities:
            if item.role == "ADAPTER_ARTIFACT":
                (adapter_dir / item.path).write_bytes(item.body)
            elif item.role == ADAPTER_SOURCE_README_ROLE:
                (adapter_dir / "README.md").write_bytes(item.body)

        committed_policy_loader = getattr(
            release_module, "load_committed_trust_policy", None
        )
        if not callable(committed_policy_loader):
            raise PublicationError("release builder trust-policy loader is absent")

        def exact_policy_loader(source_revision: str) -> dict[str, Any]:
            if source_revision != expected_source_revision:
                raise PublicationError(
                    "release builder requested an unexpected trust-policy revision"
                )
            return dict(trust_policy)

        release_module.load_committed_trust_policy = exact_policy_loader
        try:
            rebuilt_manifest, rebuilt_files = release_module.build_release_packet(
                candidate=candidate,
                source_revision=expected_source_revision,
                receipts=receipts,
                reports=reports,
                adapter_dir=adapter_dir,
                trusted_public_key=trusted_public_key,
            )
        finally:
            release_module.load_committed_trust_policy = committed_policy_loader
    if rebuilt_manifest != document:
        raise PublicationError("release manifest differs from independently rebuilt packet")
    if not isinstance(rebuilt_files, dict) or any(
        not isinstance(path, str) or not isinstance(body, bytes)
        for path, body in rebuilt_files.items()
    ):
        raise PublicationError("packet builder returned an invalid repository file map")
    observed_files = {item.path: item.body for item in packet_identities}
    if rebuilt_files != observed_files:
        raise PublicationError("release packet bytes differ from independently rebuilt packet")


def prepare_release(
    manifest_path: Path,
    *,
    trusted_public_key: Path,
    expected_source_revision: str,
    receipt_path: Path | None = None,
    candidate_loader: Callable[[str], dict[str, Any]] | None = None,
) -> PreparedRelease:
    manifest_path = _lexical_absolute(manifest_path)
    if manifest_path.name != RELEASE_MANIFEST_FILENAME:
        raise PublicationError(
            f"release manifest filename must be exactly {RELEASE_MANIFEST_FILENAME}"
        )
    source_bindings = _assert_exact_source_bindings(expected_source_revision)
    trust_policy, trust_policy_sha256 = _load_trust_policy(
        expected_source_revision
    )
    manifest_body = _bounded_regular_file_read(
        manifest_path,
        label="release manifest",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    if not manifest_body:
        raise PublicationError("release manifest is empty")
    document = _strict_json_bytes(manifest_body, "release manifest")
    if manifest_body != canonical_json(document):
        raise PublicationError("release manifest bytes are not canonical JSON")
    if FULL_SHA_RE.fullmatch(expected_source_revision or "") is None:
        raise PublicationError("expected source revision must be exact 40-hex")
    if document.get("sourceRevision") != expected_source_revision:
        raise PublicationError("release manifest source differs from expected source revision")
    _assert_manifest_trust_policy_identity(document, trust_policy)
    trusted_public_key_body = _bounded_regular_file_read(
        trusted_public_key,
        label="trusted public key",
        max_bytes=MAX_TRUSTED_PUBLIC_KEY_BYTES,
    )
    if not trusted_public_key_body:
        raise PublicationError("trusted public key is empty")
    if receipt_path is not None:
        _validate_receipt_destination(manifest_path.parent, receipt_path)
    evidence_module, release_module = _release_modules()
    private, public_inventory, local_inventory, chain_sha256 = _adapt_manifest(
        document, release_module
    )

    root = manifest_path.parent
    actual = _enumerate_packet_files(root, manifest_path)
    public_identities, public_declared = _read_inventory_files(
        root, public_inventory, inventory_label="intendedRepositoryFiles"
    )
    local_identities, local_declared = _read_inventory_files(
        root, local_inventory, inventory_label="localEvidenceFiles"
    )
    if public_declared & local_declared:
        raise PublicationError("public and local evidence inventories overlap")
    declared = public_declared | local_declared
    if actual != declared:
        missing = sorted(declared - actual)
        unexpected = sorted(actual - declared)
        raise PublicationError(
            f"release directory inventory mismatch (missing={missing}, unexpected={unexpected})"
        )

    packet_identities = tuple(
        sorted(public_identities + local_identities, key=lambda item: item.path)
    )
    loader = candidate_loader or release_module.load_committed_candidate
    with tempfile.TemporaryDirectory(prefix="szl-ra3-trust-anchor-") as temporary:
        trusted_public_key_snapshot = Path(temporary) / "trusted-public-key.pem"
        _create_only_write(trusted_public_key_snapshot, trusted_public_key_body)
        _rebuild_packet(
            document=document,
            packet_identities=packet_identities,
            trusted_public_key=trusted_public_key_snapshot,
            expected_source_revision=expected_source_revision,
            evidence_module=evidence_module,
            release_module=release_module,
            candidate_loader=loader,
            trust_policy=trust_policy,
        )
    manifest_identity = FileIdentity(
        RELEASE_MANIFEST_FILENAME,
        "RELEASE_MANIFEST",
        len(manifest_body),
        sha256_bytes(manifest_body),
        manifest_body,
    )
    if manifest_identity.path in declared or manifest_identity.path == ".gitattributes":
        raise PublicationError("manifest filename collides with a published inventory path")
    published_identities = public_identities + [manifest_identity]
    published_identities.sort(key=lambda item: item.path)
    local_identities.sort(key=lambda item: item.path)
    return PreparedRelease(
        manifest_path=manifest_path,
        manifest_sha256=manifest_identity.sha256,
        chain_sha256=chain_sha256,
        private=private,
        source_revision=expected_source_revision,
        source_bindings=source_bindings,
        trust_policy_sha256=trust_policy_sha256,
        trust_key_id=trust_policy["keyId"],
        trust_key_fingerprint_sha256=trust_policy[
            "publicKeyFingerprintSha256"
        ],
        files=tuple(published_identities),
        local_evidence_files=tuple(local_identities),
    )


def _info_sha(info: Any) -> str:
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or FULL_SHA_RE.fullmatch(revision) is None:
        raise PublicationError("Hub repository did not expose an immutable revision")
    return revision


def _info_private(info: Any) -> bool:
    value = info.get("private") if isinstance(info, dict) else getattr(info, "private", None)
    if not isinstance(value, bool):
        raise PublicationError("Hub repository did not expose explicit visibility")
    return value


def _info_id(info: Any) -> str:
    value = info.get("id") if isinstance(info, dict) else getattr(info, "id", None)
    if value != TARGET_REPO_ID:
        raise PublicationError("Hub repository identity differs from the target")
    return value


def _repo_exists(api: Any, repo_type: str, token: str) -> bool:
    return bool(api.repo_exists(repo_id=TARGET_REPO_ID, repo_type=repo_type, token=token))


def _read_remote(
    downloader: Callable[..., str],
    path: str,
    revision: str,
    token: str,
    *,
    expected_size: int | None,
    max_bytes: int,
) -> bytes:
    downloaded = downloader(
        repo_id=TARGET_REPO_ID,
        filename=path,
        repo_type=TARGET_REPO_TYPE,
        revision=revision,
        token=token,
    )
    return _bounded_regular_file_read(
        Path(downloaded),
        label=f"downloaded immutable file {path}",
        max_bytes=max_bytes,
        expected_size=expected_size,
    )


def _remote_metadata_value(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, dict) else getattr(item, key, None)


def _assert_remote_path_sizes(
    api: Any,
    *,
    revision: str,
    token: str,
    expected_sizes: dict[str, int],
    maximum_sizes: dict[str, int] | None = None,
) -> None:
    """Reject remote size drift before the Hub downloader transfers a file."""

    maximum_sizes = maximum_sizes or {}
    requested = set(expected_sizes) | set(maximum_sizes)
    if not requested:
        return
    metadata = api.get_paths_info(
        repo_id=TARGET_REPO_ID,
        paths=sorted(requested),
        repo_type=TARGET_REPO_TYPE,
        revision=revision,
        token=token,
    )
    observed: dict[str, int] = {}
    for item in metadata:
        path = _remote_metadata_value(item, "path")
        if not isinstance(path, str) or path not in requested or path in observed:
            raise PublicationError("Hub path-size metadata is incomplete or ambiguous")
        item_type = _remote_metadata_value(item, "type")
        if item_type not in (None, "file"):
            raise PublicationError(f"{path}: Hub path-size metadata is not a file")
        size = _remote_metadata_value(item, "size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise PublicationError(f"{path}: Hub path-size metadata is invalid")
        observed[path] = size
    if set(observed) != requested:
        raise PublicationError("Hub path-size metadata is incomplete or ambiguous")
    for path, expected in expected_sizes.items():
        if observed[path] != expected:
            raise PublicationError(f"{path}: Hub metadata size drift from the manifest")
    for path, maximum in maximum_sizes.items():
        if observed[path] > maximum:
            raise PublicationError(f"{path}: Hub metadata reports an oversized file")


def _readback(
    api: Any,
    prepared: PreparedRelease,
    *,
    revision: str,
    token: str,
    downloader: Callable[..., str],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    observed_paths = set(
        api.list_repo_files(
            repo_id=TARGET_REPO_ID,
            repo_type=TARGET_REPO_TYPE,
            revision=revision,
            token=token,
        )
    )
    intended = {item.path for item in prepared.files}
    allowed = intended | {".gitattributes"}
    if observed_paths != intended and observed_paths != allowed:
        raise PublicationError(
            "immutable revision has an unaccounted or missing path "
            f"(expected={sorted(allowed)}, observed={sorted(observed_paths)})"
        )
    _assert_remote_path_sizes(
        api,
        revision=revision,
        token=token,
        expected_sizes={item.path: item.size for item in prepared.files},
        maximum_sizes=(
            {".gitattributes": MAX_GITATTRIBUTES_BYTES}
            if ".gitattributes" in observed_paths
            else None
        ),
    )
    evidence: list[dict[str, Any]] = []
    for item in prepared.files:
        body = _read_remote(
            downloader,
            item.path,
            revision,
            token,
            expected_size=item.size,
            max_bytes=max(item.size, 1),
        )
        if body != item.body:
            raise PublicationError(f"{item.path}: immutable readback mismatch")
        evidence.append(
            {
                "path": item.path,
                "bytes": len(body),
                "sha256": sha256_bytes(body),
                "role": item.role,
            }
        )
    attributes = None
    if ".gitattributes" in observed_paths:
        body = _read_remote(
            downloader,
            ".gitattributes",
            revision,
            token,
            expected_size=None,
            max_bytes=MAX_GITATTRIBUTES_BYTES,
        )
        attributes = {
            "path": ".gitattributes",
            "bytes": len(body),
            "sha256": sha256_bytes(body),
            "accounting": "GITATTRIBUTES_READBACK_VERIFIED_UNATTRIBUTED",
        }
    return evidence, attributes


def _create_only_write(path: Path, body: bytes) -> None:
    path = _assert_non_reparse_ancestors(path, "create-only destination")
    try:
        existing = path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise PublicationError("create-only destination could not be inspected") from exc
    else:
        if stat.S_ISLNK(existing.st_mode) or _metadata_is_reparse(existing):
            raise PublicationError("create-only destination is a symlink or reparse point")
        raise PublicationError("publication receipt destination already exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(path, flags, 0o600)
    except (FileExistsError, FileNotFoundError) as exc:
        raise PublicationError("publication receipt destination already exists") from exc
    try:
        opened = os.fstat(descriptor)
        if _metadata_is_reparse(opened) or not stat.S_ISREG(opened.st_mode):
            raise PublicationError("create-only destination descriptor is not a regular file")
        view = memoryview(body)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        try:
            path.unlink()
        except OSError:
            pass
        raise
    else:
        os.close(descriptor)


def _write_publication_failure_receipt(
    prepared: PreparedRelease,
    *,
    receipt_path: Path,
    expected_parent_revision: str,
    observed_parent_revision: str,
    published_revision: str | None,
    failure_phase: str,
    failure: BaseException,
    observed_main_revision: str | None,
    observed_private: bool | None,
    default_revision_verified: bool,
    visibility_verified: bool,
) -> None:
    committed = True if published_revision is not None else None
    receipt: dict[str, Any] = {
        "schema": FAILURE_RECEIPT_SCHEMA,
        "status": (
            "REMOTE_COMMIT_UNVERIFIED"
            if committed is True
            else "REMOTE_MUTATION_RESULT_UNKNOWN"
        ),
        "sourceRevision": prepared.source_revision,
        "target": {"repoId": TARGET_REPO_ID, "repoType": TARGET_REPO_TYPE},
        "expectedParentRevision": expected_parent_revision,
        "observedParentRevision": observed_parent_revision,
        "publishedRevision": published_revision,
        "observedMainRevision": observed_main_revision,
        "observedPrivate": observed_private,
        "mutationAttempted": True,
        "committed": committed,
        "failurePhase": failure_phase,
        "failureType": type(failure).__name__,
        "defaultRevisionVerified": default_revision_verified,
        "visibilityVerified": visibility_verified,
        "releaseManifestSha256": prepared.manifest_sha256,
        "authenticatedReceiptChainSha256": prepared.chain_sha256,
        "receiptSigningTrustPolicy": {
            "path": TRUST_POLICY_PATH,
            "sha256": prepared.trust_policy_sha256,
            "keyId": prepared.trust_key_id,
            "publicKeyFingerprintSha256": prepared.trust_key_fingerprint_sha256,
        },
        "sourceByteBindings": [
            {"path": item.path, "bytes": item.size, "sha256": item.sha256}
            for item in prepared.source_bindings
        ],
        "portfolioMutation": "NOT_PERFORMED",
    }
    receipt["receiptSha256"] = sha256_bytes(canonical_json(receipt))
    _create_only_write(receipt_path, canonical_json(receipt))


def publish_prepared(
    prepared: PreparedRelease,
    *,
    expected_parent_revision: str,
    receipt_path: Path,
    token: str,
    api: Any,
    downloader: Callable[..., str] | None = hf_hub_download,
) -> dict[str, Any]:
    receipt_path = _validate_receipt_destination(prepared.manifest_path.parent, receipt_path)
    if not token:
        raise PublicationError("a scoped Hugging Face token is required for publication")
    if downloader is None:
        raise PublicationError("huggingface_hub is required for publication readback")
    expected_parent_revision = expected_parent_revision.strip()
    if FULL_SHA_RE.fullmatch(expected_parent_revision) is None:
        raise PublicationError("expected parent revision must be an exact 40-character SHA")

    model_exists = _repo_exists(api, TARGET_REPO_TYPE, token)
    if not model_exists:
        for wrong_type in ("dataset", "space"):
            if _repo_exists(api, wrong_type, token):
                raise PublicationError(
                    f"target identifier already exists with repo type {wrong_type}, not model"
                )
        raise PublicationError(
            "target model repository is absent; initialize it in a separately authorized operation"
        )
    info = api.repo_info(
        repo_id=TARGET_REPO_ID,
        repo_type=TARGET_REPO_TYPE,
        revision="main",
        token=token,
    )
    _info_id(info)
    parent = _info_sha(info)
    observed_private = _info_private(info)
    if observed_private is not prepared.private:
        raise PublicationError(
            "Hub repository visibility differs from the release manifest"
        )
    if parent != expected_parent_revision:
        raise PublicationError(
            f"Hub parent drift (expected {expected_parent_revision}, observed {parent})"
        )

    intended = {item.path: item.body for item in prepared.files}
    current_paths = set(
        api.list_repo_files(
            repo_id=TARGET_REPO_ID,
            repo_type=TARGET_REPO_TYPE,
            revision=parent,
            token=token,
        )
    )
    if not current_paths.issubset(set(intended) | {".gitattributes"}):
        raise PublicationError(f"Hub parent has unaccounted paths: {sorted(current_paths - set(intended) - {'.gitattributes'})}")

    replay_candidate = current_paths.issuperset(intended)
    if replay_candidate:
        _assert_remote_path_sizes(
            api,
            revision=parent,
            token=token,
            expected_sizes={path: len(body) for path, body in intended.items()},
        )
    replay = replay_candidate and all(
        _read_remote(
            downloader,
            path,
            parent,
            token,
            expected_size=len(body),
            max_bytes=max(len(body), 1),
        )
        == body
        for path, body in intended.items()
    )
    mutation_attempted = False
    observed_main_revision: str | None = None
    observed_post_private: bool | None = None
    default_revision_verified = False
    visibility_verified = False
    failure_phase = "PRE_COMMIT_VERIFICATION"
    if replay:
        revision = parent
        status = "REPLAY_READBACK_VERIFIED"
    else:
        operations = [
            CommitOperationAdd(path_in_repo=item.path, path_or_fileobj=io.BytesIO(item.body))
            for item in prepared.files
        ]
        revision = None
        status = "PUBLISHED_AND_IMMUTABLE_READBACK_VERIFIED"

    try:
        if not replay:
            failure_phase = "COMMIT_INVOCATION"
            mutation_attempted = True
            commit = api.create_commit(
                repo_id=TARGET_REPO_ID,
                repo_type=TARGET_REPO_TYPE,
                revision="main",
                parent_commit=parent,
                operations=operations,
                token=token,
                commit_message=(
                    "Publish authenticated ReceiptAgent v3 "
                    f"{prepared.manifest_sha256[:12]}"
                ),
            )
            failure_phase = "COMMIT_REVISION_VALIDATION"
            revision = getattr(commit, "oid", None)
            if not isinstance(revision, str) or FULL_SHA_RE.fullmatch(revision) is None:
                revision = None
                raise PublicationError("Hub commit did not return an immutable revision")
        failure_phase = "IMMUTABLE_READBACK"
        files, attributes = _readback(
            api, prepared, revision=revision, token=token, downloader=downloader
        )
        failure_phase = "MAIN_IDENTITY_READBACK"
        post_info = api.repo_info(
            repo_id=TARGET_REPO_ID,
            repo_type=TARGET_REPO_TYPE,
            revision="main",
            token=token,
        )
        _info_id(post_info)
        observed_main_revision = _info_sha(post_info)
        observed_post_private = _info_private(post_info)
        failure_phase = "MAIN_REVISION_POSTCONDITION"
        if observed_main_revision != revision:
            raise PublicationError(
                "Hub main advanced before publication postcondition verification"
            )
        default_revision_verified = True
        failure_phase = "VISIBILITY_POSTCONDITION"
        if observed_post_private is not prepared.private:
            raise PublicationError(
                "Hub repository visibility changed before publication postcondition verification"
            )
        visibility_verified = True
    except Exception as exc:
        if mutation_attempted:
            try:
                _write_publication_failure_receipt(
                    prepared,
                    receipt_path=receipt_path,
                    expected_parent_revision=expected_parent_revision,
                    observed_parent_revision=parent,
                    published_revision=revision,
                    failure_phase=failure_phase,
                    failure=exc,
                    observed_main_revision=observed_main_revision,
                    observed_private=observed_post_private,
                    default_revision_verified=default_revision_verified,
                    visibility_verified=visibility_verified,
                )
            except Exception as receipt_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        "publication failure receipt could not be committed: "
                        f"{type(receipt_exc).__name__}"
                    )
        raise
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "sourceRevision": prepared.source_revision,
        "target": {"repoId": TARGET_REPO_ID, "repoType": TARGET_REPO_TYPE},
        "expectedParentRevision": expected_parent_revision,
        "observedParentRevision": parent,
        "publishedRevision": revision,
        "observedMainRevision": observed_main_revision,
        "defaultRevisionVerified": True,
        "visibilityVerified": True,
        "releaseManifestSha256": prepared.manifest_sha256,
        "authenticatedReceiptChainSha256": prepared.chain_sha256,
        "receiptSigningTrustPolicy": {
            "path": TRUST_POLICY_PATH,
            "sha256": prepared.trust_policy_sha256,
            "keyId": prepared.trust_key_id,
            "publicKeyFingerprintSha256": prepared.trust_key_fingerprint_sha256,
        },
        "sourceByteBindings": [
            {"path": item.path, "bytes": item.size, "sha256": item.sha256}
            for item in prepared.source_bindings
        ],
        "files": files,
        "gitattributes": attributes or {"accounting": "NOT_PRESENT"},
        "portfolioMutation": "NOT_PERFORMED",
    }
    receipt["receiptSha256"] = sha256_bytes(canonical_json(receipt))
    receipt_path = _validate_receipt_destination(
        prepared.manifest_path.parent, receipt_path
    )
    _create_only_write(receipt_path, canonical_json(receipt))
    return receipt


def run(
    *,
    manifest_path: Path,
    receipt_path: Path,
    publish: bool,
    trusted_public_key: Path,
    expected_source_revision: str,
    expected_parent_revision: str | None = None,
    token: str | None = None,
    api: Any | None = None,
    downloader: Callable[..., str] | None = hf_hub_download,
    candidate_loader: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prepared = prepare_release(
        manifest_path,
        trusted_public_key=trusted_public_key,
        expected_source_revision=expected_source_revision,
        receipt_path=receipt_path,
        candidate_loader=candidate_loader,
    )
    if not publish:
        return {
            "schema": RECEIPT_SCHEMA,
            "status": "LOCAL_RELEASE_PACKET_VERIFIED",
            "sourceRevision": prepared.source_revision,
            "target": {"repoId": TARGET_REPO_ID, "repoType": TARGET_REPO_TYPE},
            "releaseManifestSha256": prepared.manifest_sha256,
            "authenticatedReceiptChainSha256": prepared.chain_sha256,
            "receiptSigningTrustPolicy": {
                "path": TRUST_POLICY_PATH,
                "sha256": prepared.trust_policy_sha256,
                "keyId": prepared.trust_key_id,
                "publicKeyFingerprintSha256": (
                    prepared.trust_key_fingerprint_sha256
                ),
            },
            "sourceByteBindings": [
                {"path": item.path, "bytes": item.size, "sha256": item.sha256}
                for item in prepared.source_bindings
            ],
            "files": [
                {
                    "path": item.path,
                    "bytes": item.size,
                    "sha256": item.sha256,
                    "role": item.role,
                }
                for item in prepared.files
            ],
            "localEvidenceDisposition": "NOT_FOR_PUBLICATION",
            "localEvidenceFiles": [
                {
                    "path": item.path,
                    "bytes": item.size,
                    "sha256": item.sha256,
                    "role": item.role,
                    "publicationDisposition": "NOT_FOR_PUBLICATION",
                }
                for item in prepared.local_evidence_files
            ],
            "networkAccess": "NOT_PERFORMED",
            "portfolioMutation": "NOT_PERFORMED",
        }
    if expected_parent_revision is None or not expected_parent_revision.strip():
        raise PublicationError("--expected-parent-revision is required with --publish")
    if not token:
        raise PublicationError("HF_TOKEN is required with --publish")
    if api is None:
        if HfApi is None:
            raise PublicationError("huggingface_hub is required with --publish")
        api = HfApi(token=token)
    return publish_prepared(
        prepared,
        expected_parent_revision=expected_parent_revision,
        receipt_path=receipt_path,
        token=token,
        api=api,
        downloader=downloader,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--trusted-public-key", type=Path, required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--expected-parent-revision")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(
        manifest_path=args.manifest,
        receipt_path=args.receipt,
        publish=args.publish,
        trusted_public_key=args.trusted_public_key,
        expected_source_revision=args.expected_source_revision,
        expected_parent_revision=args.expected_parent_revision,
        token=os.getenv("HF_TOKEN"),
    )
    print(canonical_json(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
