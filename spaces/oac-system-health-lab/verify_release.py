"""Bounded local integrity and immutable Git-source verification for OAC Health.

The startup helper proves only local hashes. The release verifier additionally
compares each deployed artifact to an immutable object in the canonical Git
repository; a mutable checkout is never used as the source of truth.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import subprocess
from typing import Any


RELEASE_SCHEMA = "szl.oac-health-space-release/v1"
REPORT_SCHEMA = "szl.oac-health-space-release-verification/v1"
SOURCE_REPOSITORY = "szl-holdings/szl-forge"
SOURCE_DIRECTORY = "clinical-gateway/huggingface/model/oac-system-health-v1"
MODEL_REPOSITORY = "SZLHOLDINGS/oac-system-health-v1"
DATASET_REPOSITORY = "SZLHOLDINGS/oac-clinical-transport-observability-synthetic"
ARTIFACT_NAMES = ("oac_operational_health.py", "model.json", "artifact_receipt.json")
MAX_MANIFEST_BYTES = 16_384
MAX_ARTIFACT_BYTES = 1_048_576
MAX_JSON_DEPTH = 32
SHA1 = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ReleaseVerificationError(ValueError):
    """An artifact or source contract failed closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _closed(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ReleaseVerificationError(f"{label} fields do not match the closed schema")
    return value


def _hash(value: Any, expression: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or expression.fullmatch(value) is None:
        raise ReleaseVerificationError(f"{label} is not an exact lowercase digest")
    return value


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseVerificationError("duplicate JSON field")
        result[key] = value
    return result


def _invalid_constant(_value: str) -> None:
    raise ReleaseVerificationError("non-finite JSON constant")


def _integer(value: str) -> int:
    if len(value) > 128:
        raise ReleaseVerificationError("JSON numeric literal limit exceeded")
    return int(value)


def _number(value: str) -> float:
    if len(value) > 128:
        raise ReleaseVerificationError("JSON numeric literal limit exceeded")
    result = float(value)
    if not math.isfinite(result):
        raise ReleaseVerificationError("non-finite JSON number")
    return result


def strict_json(raw: bytes, *, limit: int = MAX_MANIFEST_BYTES) -> Any:
    if not isinstance(raw, bytes) or len(raw) > limit:
        raise ReleaseVerificationError("JSON byte limit exceeded")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_invalid_constant,
            parse_int=_integer,
            parse_float=_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ReleaseVerificationError("invalid bounded UTF-8 JSON") from exc
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > MAX_JSON_DEPTH:
            raise ReleaseVerificationError("JSON depth limit exceeded")
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
    return value


def _regular(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ReleaseVerificationError("required release path unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ReleaseVerificationError(
            "release paths may not be links or reparse points"
        )


def _read(path: Path, limit: int) -> bytes:
    _regular(path)
    if not path.is_file():
        raise ReleaseVerificationError("release artifact is not a regular file")
    try:
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
    except OSError as exc:
        raise ReleaseVerificationError(
            "unable to read bounded release artifact"
        ) from exc
    if len(raw) > limit:
        raise ReleaseVerificationError("release artifact byte limit exceeded")
    return raw


def validate_manifest(value: Any) -> dict[str, Any]:
    manifest = _closed(
        value,
        {"schema", "artifact_source", "hub_model", "hub_dataset", "artifacts"},
        "release",
    )
    if manifest["schema"] != RELEASE_SCHEMA:
        raise ReleaseVerificationError("release schema mismatch")
    source = _closed(
        manifest["artifact_source"],
        {"repository", "revision", "directory"},
        "artifact_source",
    )
    if (
        source["repository"] != SOURCE_REPOSITORY
        or source["directory"] != SOURCE_DIRECTORY
    ):
        raise ReleaseVerificationError(
            "artifact source is not the canonical repository and directory"
        )
    _hash(source["revision"], SHA1, "artifact source revision")
    for field, repo_id in (
        ("hub_model", MODEL_REPOSITORY),
        ("hub_dataset", DATASET_REPOSITORY),
    ):
        pin = _closed(manifest[field], {"repo_id", "revision"}, field)
        if pin["repo_id"] != repo_id:
            raise ReleaseVerificationError(f"{field} repository mismatch")
        _hash(pin["revision"], SHA1, f"{field} revision")
    artifacts = _closed(manifest["artifacts"], set(ARTIFACT_NAMES), "artifacts")
    for name in ARTIFACT_NAMES:
        _hash(artifacts[name], SHA256, f"artifact {name}")
    # Return a detached value, not a caller-owned mutable mapping.
    return json.loads(json.dumps(manifest))


def load_manifest(space_root: Path) -> dict[str, Any]:
    root = Path(space_root).absolute()
    _regular(root)
    return validate_manifest(
        strict_json(_read(root / "release.json", MAX_MANIFEST_BYTES))
    )


def verify_local_artifacts(
    space_root: Path, manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Startup integrity only: no immutable source or provider parity is claimed."""
    root = Path(space_root).absolute()
    release = load_manifest(root) if manifest is None else validate_manifest(manifest)
    _regular(root)
    _regular(root / "artifacts")
    if not (root / "artifacts").is_dir():
        raise ReleaseVerificationError("artifact directory is unavailable")
    evidence = {}
    for name in ARTIFACT_NAMES:
        raw = _read(root / "artifacts" / name, MAX_ARTIFACT_BYTES)
        observed = hashlib.sha256(raw).hexdigest()
        if observed != release["artifacts"][name]:
            raise ReleaseVerificationError(f"local artifact hash mismatch: {name}")
        evidence[name] = {
            "sha256": observed,
            "bytes": len(raw),
            "local_hash_matched": True,
        }
    return {
        "complete": True,
        "status": "PASS",
        "scope": "LOCAL_HASHES_ONLY",
        "artifacts": evidence,
    }


def _git(repository_root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=False,
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReleaseVerificationError("immutable Git source read unavailable") from exc
    if result.returncode != 0:
        raise ReleaseVerificationError("immutable Git source object unavailable")
    return result.stdout


def verify_release(space_root: Path, repository_root: Path) -> dict[str, Any]:
    manifest = load_manifest(space_root)
    local = verify_local_artifacts(space_root, manifest)
    remote = (
        _git(repository_root, "config", "--get", "remote.origin.url")
        .decode("utf-8")
        .strip()
    )
    if remote not in {
        "https://github.com/szl-holdings/szl-forge.git",
        "https://github.com/szl-holdings/szl-forge",
        "git@github.com:szl-holdings/szl-forge.git",
        "ssh://git@github.com/szl-holdings/szl-forge.git",
    }:
        raise ReleaseVerificationError(
            "Git origin is not the canonical source repository"
        )
    revision = manifest["artifact_source"]["revision"]
    observed_revision = _git(
        repository_root, "rev-parse", "--verify", f"{revision}^{{commit}}"
    )
    if observed_revision.decode("ascii").strip() != revision:
        raise ReleaseVerificationError("immutable source revision mismatch")
    for name in ARTIFACT_NAMES:
        object_name = f"{revision}:{SOURCE_DIRECTORY}/{name}"
        try:
            size = int(_git(repository_root, "cat-file", "-s", object_name).strip())
        except ValueError as exc:
            raise ReleaseVerificationError("invalid immutable artifact size") from exc
        if not 0 < size <= MAX_ARTIFACT_BYTES:
            raise ReleaseVerificationError("immutable artifact byte limit exceeded")
        canonical = _git(repository_root, "show", object_name)
        if (
            len(canonical) != size
            or hashlib.sha256(canonical).hexdigest() != manifest["artifacts"][name]
        ):
            raise ReleaseVerificationError(f"immutable Git artifact mismatch: {name}")
        local["artifacts"][name]["immutable_git_blob_matched"] = True
        local["artifacts"][name]["source_path"] = f"{SOURCE_DIRECTORY}/{name}"
    return {
        "schema": REPORT_SCHEMA,
        "generated_at": utc_now(),
        "complete": True,
        "status": "PASS",
        "scope": "IMMUTABLE_GIT_ARTIFACT_PARITY",
        "artifact_source": manifest["artifact_source"],
        "hub_model": manifest["hub_model"],
        "hub_dataset": manifest["hub_dataset"],
        "artifacts": local["artifacts"],
        "source_parity_count": len(ARTIFACT_NAMES),
        "provider_parity_claimed": False,
        "training_performed": False,
        "clinical_use_authorized": False,
        "receipt_minted": False,
    }


def write_report(output: Path, report: dict[str, Any]) -> None:
    """An existing receipt is never overwritten."""
    with Path(output).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            report,
            handle,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-root", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--repository-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = verify_release(args.space_root, args.repository_root)
    except ReleaseVerificationError as exc:
        report = {
            "schema": REPORT_SCHEMA,
            "generated_at": utc_now(),
            "complete": False,
            "status": "FAIL",
            "failure": str(exc),
            "receipt_minted": False,
            "training_performed": False,
            "clinical_use_authorized": False,
        }
    write_report(args.output, report)
    print(
        json.dumps(
            {"complete": report["complete"], "status": report["status"]}, sort_keys=True
        )
    )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
