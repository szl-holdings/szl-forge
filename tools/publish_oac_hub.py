#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish the staged OAC Hub packages as exact-source, parent-pinned commits.

A closed profile registry binds each Hub repository to one staged szl-forge
directory, its exact file set and an explicit replace list (HF upgrade plan D1
and P9: one committed writer per asset). A create-enabled profile may only add
its complete package to an empty target or accept a byte-identical target. Without
``--publish`` the run is a dry-run that reads the Hub anonymously and writes
nothing. With ``--publish`` the run must own current protected ``main``; it
makes at most one ``create_commit`` whose ``parent_commit`` is the Hub head it
inspected, then reads every file back. It never deletes a Hub file, never
changes visibility, and never records a credential.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPOSITORY = "szl-holdings/szl-forge"
SOURCE_URL = "https://github.com/" + SOURCE_REPOSITORY
CANONICAL_REMOTE = SOURCE_URL + ".git"
FULL_SHA = re.compile(r"[0-9a-f]{40}")
HUB_ID = re.compile(r"SZLHOLDINGS/[A-Za-z0-9][A-Za-z0-9._-]*")
HUB_METADATA = frozenset({".gitattributes"})
CARD_PATHS = frozenset({"README.md"})
MAX_FILE_BYTES = 2 * 1024 * 1024
REVIEWED_HUB_CLIENT = "1.23.0"
REPORT_SCHEMA = "szl.oac-hub-publication/v1"


@dataclass(frozen=True)
class HubProfile:
    """Immutable publication contract for one OAC Hub repository."""

    key: str
    repo_id: str
    repo_type: str
    staged_dir: str
    files: frozenset[str]
    replace_paths: frozenset[str]
    allow_create: bool

    @property
    def job_id(self) -> str:
        return "publish-" + self.key

    @property
    def lock_group(self) -> str:
        return f"hf-write/{self.repo_type}/{self.repo_id}"


PROFILES: Mapping[str, HubProfile] = {
    "oac-v1-model": HubProfile(
        key="oac-v1-model",
        repo_id="SZLHOLDINGS/oac-system-health-v1",
        repo_type="model",
        staged_dir="clinical-gateway/huggingface/model/oac-system-health-v1",
        files=frozenset(
            {
                "LICENSE",
                "README.md",
                "artifact_receipt.json",
                "example_input.json",
                "model.json",
                "oac_operational_health.py",
            }
        ),
        replace_paths=frozenset({"README.md"}),
        allow_create=False,
    ),
    "oac-v2-model": HubProfile(
        key="oac-v2-model",
        repo_id="SZLHOLDINGS/oac-ops-health-v2",
        repo_type="model",
        staged_dir="publishing/oac-ops-health-v2",
        files=frozenset(
            {
                "LICENSE",
                "README.md",
                "artifact_receipt.json",
                "example_input.json",
                "model.json",
                "ops_health.py",
            }
        ),
        replace_paths=frozenset(),
        allow_create=True,
    ),
    "oac-v1-dataset": HubProfile(
        key="oac-v1-dataset",
        repo_id="SZLHOLDINGS/oac-clinical-transport-observability-synthetic",
        repo_type="dataset",
        staged_dir=(
            "clinical-gateway/huggingface/dataset/"
            "oac-clinical-transport-observability-synthetic"
        ),
        files=frozenset(
            {
                "LICENSE",
                "README.md",
                "data/test.jsonl",
                "data/train.jsonl",
                "data/validation.jsonl",
                "dataset_receipt.json",
                "schema.json",
                "training_source_snapshot.py",
            }
        ),
        replace_paths=frozenset(),
        allow_create=False,
    ),
}


class PublicationRefused(RuntimeError):
    """A fail-closed publication contract did not hold."""


def _check_registry(profiles: Mapping[str, HubProfile]) -> None:
    for key, profile in profiles.items():
        if (
            key != profile.key
            or profile.repo_type not in {"model", "dataset"}
            or not HUB_ID.fullmatch(profile.repo_id)
            or not profile.files
            or not profile.replace_paths <= profile.files
            or profile.files & HUB_METADATA
            or (profile.allow_create and profile.replace_paths)
        ):
            raise PublicationRefused(f"closed profile registry is inconsistent: {key}")


_check_registry(PROFILES)


def resolve_profile(key: str) -> HubProfile:
    try:
        return PROFILES[str(key)]
    except KeyError as error:
        raise PublicationRefused(f"unknown OAC Hub profile: {key}") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def evidence_for(files: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        path: {"bytes": len(value), "sha256": sha256_bytes(value)}
        for path, value in sorted(files.items())
    }


def require_sha(value: str, what: str = "revision") -> str:
    if not isinstance(value, str) or not FULL_SHA.fullmatch(value):
        raise PublicationRefused(f"{what} must be a full lowercase 40-character commit SHA")
    return value


# --- Git source -------------------------------------------------------------


def _git(args: list[str]) -> bytes:
    """Run one fixed Git read in the repository root, never through a shell."""
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects", *args],
            cwd=ROOT,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PublicationRefused("Git object read failed") from error
    if result.returncode:
        raise PublicationRefused("Git object read failed")
    return result.stdout


def _read_tree(
    revision: str, directory: str, git: Callable[[list[str]], bytes]
) -> dict[str, bytes]:
    prefix = directory + "/"
    listing = git(["ls-tree", "-r", "-l", "-z", revision, "--", prefix])
    entries: dict[str, tuple[str, int]] = {}
    for record in listing.split(b"\0"):
        if not record:
            continue
        header, separator, name = record.partition(b"\t")
        fields = header.split()
        if not separator or len(fields) != 4:
            raise PublicationRefused("unexpected Git tree entry")
        mode, kind, oid, size = fields
        path = name.decode("utf-8")
        if not path.startswith(prefix):
            raise PublicationRefused("Git tree entry outside the staged directory")
        if (
            mode != b"100644"
            or kind != b"blob"
            or not re.fullmatch(rb"[0-9a-f]{40}", oid)
            or not size.isdigit()
        ):
            raise PublicationRefused("staged file must be a regular non-executable blob")
        if int(size) > MAX_FILE_BYTES:
            raise PublicationRefused("staged file exceeds the bounded publication size")
        entries[path[len(prefix):]] = (oid.decode("ascii"), int(size))
    files: dict[str, bytes] = {}
    for relative, (oid, size) in sorted(entries.items()):
        raw = git(["cat-file", "blob", oid])
        if len(raw) != size:
            raise PublicationRefused("Git blob size mismatch")
        files[relative] = raw
    return files


def validate_card(raw: bytes) -> None:
    """A card must be UTF-8, LF-only, and open with a YAML mapping."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PublicationRefused("card must be UTF-8") from error
    if "\r" in text:
        raise PublicationRefused("card must use LF line endings")
    if not text.startswith("---\n"):
        raise PublicationRefused("card front matter is missing")
    end = text.find("\n---\n", 3)
    if end < 0:
        raise PublicationRefused("card front matter is not closed")
    import yaml

    try:
        metadata = yaml.safe_load(text[4:end])
    except yaml.YAMLError as error:
        raise PublicationRefused("card front matter is not valid YAML") from error
    if not isinstance(metadata, dict):
        raise PublicationRefused("card front matter must be a mapping")


def staged_package(
    profile: HubProfile,
    revision: str,
    *,
    git: Callable[[list[str]], bytes] | None = None,
) -> dict[str, bytes]:
    """Return the exact staged bytes at ``revision`` for the closed file set."""
    require_sha(revision, "source revision")
    files = _read_tree(revision, profile.staged_dir, git or _git)
    if set(files) != set(profile.files):
        raise PublicationRefused("staged package file set drifted from the closed profile")
    for path in sorted(CARD_PATHS & set(files)):
        validate_card(files[path])
    return files


def staged_change_since(
    profile: HubProfile,
    base_revision: str,
    head_revision: str,
    *,
    git: Callable[[list[str]], bytes] | None = None,
) -> list[str]:
    """Staged paths whose bytes differ between ``base_revision`` and ``head_revision``.

    When the Hub equals the base's staged bytes, this is the exact delta a
    publication of ``head_revision`` must produce.
    """
    require_sha(base_revision, "base revision")
    base = _read_tree(base_revision, profile.staged_dir, git or _git)
    head = staged_package(profile, head_revision, git=git)
    return sorted(path for path in head if base.get(path) != head[path])


def assert_expected_delta(
    profile: HubProfile,
    base_revision: str,
    source_revision: str,
    report: dict[str, Any],
    *,
    git: Callable[[list[str]], bytes] | None = None,
) -> None:
    """Dry-run assertion of source delta and, when visible, Hub/base parity.

    A visible Hub snapshot must equal the base revision's staged bytes and the
    dry-run delta must equal the staged source change. A create-enabled target
    that returns RepositoryNotFoundError is unobservable: only an empty base
    and the source delta can be checked, while ``hub_equals_base`` stays null.
    The parent sha itself is not derivable from Git; ``--expect-parent`` pins it.
    """
    require_sha(base_revision, "base revision")
    base = evidence_for(_read_tree(base_revision, profile.staged_dir, git or _git))
    expected = staged_change_since(profile, base_revision, source_revision, git=git)
    if report.get("target_observation") == "UNRESOLVED_PRIVATE_OR_MISSING":
        report["expected_delta_since"] = {
            "base_revision": base_revision,
            "parent_revision": None,
            "hub_equals_base": None,
            "hub_drift": None,
            "delta": expected,
            "verification": "UNAVAILABLE",
        }
        if base:
            raise PublicationRefused("Hub target is unresolved while the base has staged files")
        if report.get("delta") != expected:
            raise PublicationRefused(
                "dry-run delta differs from the staged change since the base revision"
            )
        return
    hub = report.get("hub_before")
    if not isinstance(hub, dict):
        raise PublicationRefused("Hub snapshot is unavailable for expected-delta verification")
    drift = sorted(path for path in set(base) | set(hub) if base.get(path) != hub.get(path))
    report["expected_delta_since"] = {
        "base_revision": base_revision,
        "parent_revision": report.get("parent_revision"),
        "hub_equals_base": not drift,
        "hub_drift": drift,
        "delta": expected,
    }
    if drift:
        raise PublicationRefused(
            "Hub bytes at the inspected parent differ from the base revision's staged "
            "bytes: " + ", ".join(drift)
        )
    if report.get("delta") != expected:
        raise PublicationRefused(
            "dry-run delta differs from the staged change since the base revision"
        )


def assert_expected_parent(expected_parent: str, report: dict[str, Any]) -> None:
    """Dry-run assertion that the Hub head read is exactly ``expected_parent``."""
    require_sha(expected_parent, "expected parent")
    observed = report.get("parent_revision")
    report["expected_parent"] = {
        "revision": expected_parent,
        "observed": observed,
        "matches": observed == expected_parent,
    }
    if observed != expected_parent:
        raise PublicationRefused("Hub head differs from the expected parent revision")


def assert_checkout(
    source_revision: str, *, git: Callable[[list[str]], bytes] | None = None
) -> None:
    observed = (git or _git)(["rev-parse", "--verify", "HEAD"]).decode("ascii").strip()
    if observed != source_revision:
        raise PublicationRefused("checkout does not match the exact source revision")


def assert_current_main(source_revision: str) -> None:
    """Check the exact checkout against freshly queried canonical main.

    Runs only for publication, before the Hub lookup and again before the
    write. It does not make GitHub and the Hub one atomic transaction.
    """
    if not FULL_SHA.fullmatch(source_revision or ""):
        raise PublicationRefused("fresh-main guard requires an exact source revision")
    commands = (
        ["git", "rev-parse", "--verify", "HEAD"],
        ["git", "ls-remote", "--exit-code", CANONICAL_REMOTE, "refs/heads/main"],
    )
    outputs = []
    for command in commands:
        try:
            result = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True,
                encoding="utf-8", timeout=30, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise PublicationRefused("fresh-main lookup failed") from error
        if result.returncode:
            raise PublicationRefused("fresh-main lookup failed")
        outputs.append(result.stdout.strip())
        if len(outputs) == 1 and outputs[0] != source_revision:
            raise PublicationRefused("checkout does not match the exact source revision")
    remote_lines = outputs[1].splitlines()
    if len(remote_lines) != 1:
        raise PublicationRefused("canonical main did not expose one exact revision")
    fields = remote_lines[0].split()
    if (len(fields) != 2 or fields[1] != "refs/heads/main"
            or not FULL_SHA.fullmatch(fields[0])):
        raise PublicationRefused("canonical main did not expose one exact revision")
    if fields[0] != source_revision:
        raise PublicationRefused("publication source no longer owns current main")


def assert_publish_environment(environ: Mapping[str, str]) -> None:
    if (
        environ.get("GITHUB_ACTIONS") != "true"
        or environ.get("GITHUB_REPOSITORY") != SOURCE_REPOSITORY
        or environ.get("GITHUB_REF") != "refs/heads/main"
    ):
        raise PublicationRefused("publication is restricted to the canonical main workflow")


# --- Hub target -------------------------------------------------------------


def validate_hub_paths(paths: Any) -> list[str]:
    if not isinstance(paths, (list, tuple)) or any(not isinstance(p, str) for p in paths):
        raise PublicationRefused("Hub file list must contain strings")
    for path in paths:
        if (
            not path
            or "\\" in path
            or any(ord(character) < 32 for character in path)
            or any(part in ("", ".", "..") for part in path.split("/"))
        ):
            raise PublicationRefused("unsafe Hub file path")
    if len(paths) != len(set(paths)):
        raise PublicationRefused("duplicate Hub file path")
    return sorted(paths)


def lookup_target(api: Any, profile: HubProfile, *, not_found: type[BaseException]) -> Any | None:
    try:
        return api.repo_info(profile.repo_id, repo_type=profile.repo_type, revision="main")
    except not_found:
        return None


def check_identity(info: Any, profile: HubProfile) -> str:
    """Refuse a renamed, redirected or private target; return its exact head."""
    resolved = str(getattr(info, "id", "") or "")
    if resolved.casefold() != profile.repo_id.casefold():
        raise PublicationRefused(
            "Hub resolved the profile id to a different repository (renamed or redirected)"
        )
    if getattr(info, "private", None) is not False:
        raise PublicationRefused("target repository is private or its visibility is unknown")
    head = str(getattr(info, "sha", "") or "").strip().lower()
    if not FULL_SHA.fullmatch(head):
        raise PublicationRefused("Hub repository did not expose an exact head revision")
    return head


def hub_snapshot(
    api: Any,
    profile: HubProfile,
    revision: str,
    read_file: Callable[[HubProfile, str, str], bytes],
) -> tuple[list[str], dict[str, bytes]]:
    """Return the Hub file list and the bytes of every staged path present."""
    listing = validate_hub_paths(
        list(api.list_repo_files(profile.repo_id, repo_type=profile.repo_type, revision=revision))
    )
    extra = sorted(set(listing) - set(profile.files) - HUB_METADATA)
    if extra:
        raise PublicationRefused(
            "Hub holds files outside the staged package; no deletion is permitted: "
            + ", ".join(extra)
        )
    present = sorted(set(listing) & set(profile.files))
    return listing, {path: read_file(profile, path, revision) for path in present}


def plan_delta(
    profile: HubProfile,
    staged: Mapping[str, bytes],
    hub_files: Mapping[str, bytes],
) -> list[str]:
    """Plan an exact delta, never overwriting a create-enabled target."""
    if profile.allow_create:
        # The creation grant is one-time admission for a new closed package,
        # not permission to repair or overwrite a target someone already wrote.
        # hub_snapshot has already rejected unexpected paths and read every
        # present staged path at the inspected parent revision.
        if not hub_files:
            return sorted(staged)
        if set(hub_files) != set(staged):
            raise PublicationRefused("create-enabled target has a partial staged package")
        mismatched = sorted(path for path in staged if hub_files[path] != staged[path])
        if mismatched:
            raise PublicationRefused(
                "create-enabled target has differing staged bytes: " + ", ".join(mismatched)
            )
        return []
    delta = sorted(path for path in staged if hub_files.get(path) != staged[path])
    outside = [path for path in delta if path not in profile.replace_paths]
    if outside:
        raise PublicationRefused(
            "staged bytes differ from the Hub outside replace_paths: " + ", ".join(outside)
        )
    return delta


def commit_text(
    profile: HubProfile, source_revision: str, parent: str, delta: list[str]
) -> tuple[str, str]:
    message = f"Publish {', '.join(delta)} from szl-forge {source_revision[:12]}"
    lines = [
        f"Source: {SOURCE_URL}/commit/{source_revision}",
        f"Staged directory: {profile.staged_dir}",
        f"Profile: {profile.key}; parent: {parent}",
        "Written by the szl-forge publish-oac-hub workflow from protected main.",
    ]
    if set(delta) <= CARD_PATHS:
        lines.append(
            "Artifacts unchanged: only the card changed; every other file is "
            "byte-identical to the parent revision."
        )
    else:
        lines.append("Changed paths: " + ", ".join(delta))
    return message, "\n".join(lines)


def base_report(profile: HubProfile, source_revision: str, publish: bool) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile.key,
        "mode": "publish" if publish else "dry_run",
        "lookup": "authenticated" if publish else "anonymous",
        "state": "FAILED",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": source_revision,
            "staged_dir": profile.staged_dir,
        },
        "target": {
            "repo_id": profile.repo_id,
            "repo_type": profile.repo_type,
            "allow_create": profile.allow_create,
            "replace_paths": sorted(profile.replace_paths),
            "lock_group": profile.lock_group,
        },
        "resolved_repo_id": None,
        "target_observation": "UNKNOWN",
        "repository_created": False,
        "parent_revision": None,
        "hub_files": None,
        "staged": None,
        "hub_before": None,
        "delta": None,
        "artifacts_unchanged": None,
        "commit_attempted": False,
        "commit": None,
        "new_revision": None,
        "readback": None,
        "head_after": None,
        "huggingface_hub_version": None,
        "authority": {
            "paths": sorted(profile.files),
            "create": profile.allow_create,
            "delete": False,
            "visibility": False,
        },
        "secret_values_recorded": False,
    }


def execute(
    profile: HubProfile,
    source_revision: str,
    report: dict[str, Any],
    *,
    api: Any,
    read_file: Callable[[HubProfile, str, str], bytes],
    publish: bool,
    not_found: type[BaseException],
    operation_factory: Callable[..., Any],
    allow_create_grant: bool = False,
    fresh_main: Callable[[str], None] | None = None,
    git: Callable[[list[str]], bytes] | None = None,
) -> dict[str, Any]:
    """Run one profile; ``report`` keeps partial evidence when a step refuses."""
    require_sha(source_revision, "source revision")
    if allow_create_grant and not profile.allow_create:
        raise PublicationRefused("--allow-create is outside this profile's authority")
    fresh_main = fresh_main or assert_current_main
    git = git or _git
    if publish:
        fresh_main(source_revision)
    else:
        assert_checkout(source_revision, git=git)
    staged = staged_package(profile, source_revision, git=git)
    report["staged"] = evidence_for(staged)

    info = lookup_target(api, profile, not_found=not_found)
    if info is None:
        # The Hub uses one not-found exception for missing and inaccessible
        # private repositories. An anonymous read proves neither condition.
        report["target_observation"] = "UNRESOLVED_PRIVATE_OR_MISSING"
        if not profile.allow_create:
            raise PublicationRefused(
                "target repository is unresolved and this profile may not create it"
            )
        if not publish:
            report["delta"] = sorted(staged)
            report["artifacts_unchanged"] = False
            report["state"] = "CREATE_CANDIDATE_UNVERIFIED"
            return report
        if not allow_create_grant:
            raise PublicationRefused("create-enabled publication requires --allow-create")
        # This is an external effect. Check protected main immediately before
        # the one atomic create; a hidden target cannot be adopted.
        fresh_main(source_revision)
        try:
            api.create_repo(
                profile.repo_id, repo_type=profile.repo_type, private=False, exist_ok=False
            )
        except Exception as error:
            report["state"] = "BLOCKED"
            report["target_create_failure_type"] = type(error).__name__
            response = getattr(error, "response", None)
            status_code = getattr(response, "status_code", None)
            if isinstance(status_code, int):
                report["target_create_status_code"] = status_code
            raise PublicationRefused("unresolved target could not be atomically created") from error
        report["repository_created"] = True
        info = lookup_target(api, profile, not_found=not_found)
        if info is None:
            report["state"] = "BLOCKED"
            raise PublicationRefused("created repository is not visible to the lookup")

    parent = check_identity(info, profile)
    report["resolved_repo_id"] = str(info.id)
    report["target_observation"] = "PUBLIC"
    report["parent_revision"] = parent
    listing, hub_files = hub_snapshot(api, profile, parent, read_file)
    report["hub_files"] = listing
    report["hub_before"] = evidence_for(hub_files)
    delta = plan_delta(profile, staged, hub_files)
    report["delta"] = delta
    report["artifacts_unchanged"] = set(delta) <= CARD_PATHS
    if not delta:
        # The byte reads above were pinned to parent. A concurrent Hub writer
        # can move main while those reads run, so confirm the mutable head
        # still names that exact snapshot before reporting NO_CHANGE.
        unchanged_info = lookup_target(api, profile, not_found=not_found)
        if unchanged_info is None:
            raise PublicationRefused("target repository disappeared before NO_CHANGE confirmation")
        head = check_identity(unchanged_info, profile)
        report["head_after"] = head
        if head != parent:
            raise PublicationRefused("Hub head moved before NO_CHANGE confirmation")
        report["state"] = "NO_CHANGE"
        return report
    if not publish:
        report["state"] = "DELTA"
        return report

    if profile.allow_create and not allow_create_grant:
        raise PublicationRefused("create-enabled publication requires --allow-create")

    fresh_main(source_revision)
    operations = [
        operation_factory(path_in_repo=path, path_or_fileobj=staged[path]) for path in delta
    ]
    # The reviewed client marks an addition committed only after a server
    # commit. Without that contract a no-op could hide a skipped parent check.
    if any(getattr(operation, "_is_committed", None) is not False for operation in operations):
        raise PublicationRefused("reviewed client commit-origin contract is unavailable")
    message, description = commit_text(profile, source_revision, parent, delta)
    report["commit"] = {"message": message, "description": description}
    report["commit_attempted"] = True
    commit = api.create_commit(
        repo_id=profile.repo_id,
        repo_type=profile.repo_type,
        revision="main",
        create_pr=False,
        parent_commit=parent,
        operations=operations,
        commit_message=message,
        commit_description=description,
    )
    if not all(getattr(operation, "_is_committed", None) is True for operation in operations):
        raise PublicationRefused("Hub client returned without an expected-parent server commit")
    new_revision = str(getattr(commit, "oid", "") or "").strip().lower()
    if not FULL_SHA.fullmatch(new_revision):
        raise PublicationRefused("Hub commit did not return an exact revision")
    report["new_revision"] = new_revision

    after = validate_hub_paths(
        list(
            api.list_repo_files(
                profile.repo_id, repo_type=profile.repo_type, revision=new_revision
            )
        )
    )
    if set(after) - HUB_METADATA != set(staged):
        raise PublicationRefused("Hub file set after publication differs from the staged package")
    observed = {path: read_file(profile, path, new_revision) for path in sorted(staged)}
    report["readback"] = {
        path: {"sha256": sha256_bytes(observed[path]), "matches": observed[path] == staged[path]}
        for path in sorted(staged)
    }
    mismatched = sorted(path for path in staged if observed[path] != staged[path])
    if mismatched:
        raise PublicationRefused("Hub byte readback mismatch: " + ", ".join(mismatched))
    head_info = lookup_target(api, profile, not_found=not_found)
    if head_info is None:
        raise PublicationRefused("target repository disappeared after publication")
    head = check_identity(head_info, profile)
    report["head_after"] = head
    if head != new_revision:
        raise PublicationRefused("Hub head moved after publication")
    report["state"] = "PUBLISHED"
    return report


# --- Command line -----------------------------------------------------------


def hub_clients(token: str | bool, cache_dir: Path) -> dict[str, Any]:
    """Bind the real client; ``token=False`` keeps every read anonymous."""
    import huggingface_hub
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError

    def read_file(profile: HubProfile, path: str, revision: str) -> bytes:
        local = Path(
            hf_hub_download(
                repo_id=profile.repo_id,
                repo_type=profile.repo_type,
                filename=path,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
                force_download=True,
            )
        )
        if local.stat().st_size > MAX_FILE_BYTES:
            raise PublicationRefused("Hub file exceeds the bounded publication size")
        return local.read_bytes()

    return {
        "version": str(huggingface_hub.__version__),
        "api": HfApi(token=token),
        "read_file": read_file,
        "not_found": RepositoryNotFoundError,
        "operation_factory": CommitOperationAdd,
    }


def _reserve(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("x", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None, *, environ: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--allow-create", action="store_true",
        help="explicitly authorize additions for a create-enabled profile when publishing",
    )
    parser.add_argument(
        "--expect-delta-since",
        metavar="BASE_SHA",
        help=(
            "dry-run only: require the Hub to hold BASE_SHA's staged bytes and the "
            "delta to equal the staged change since BASE_SHA"
        ),
    )
    parser.add_argument(
        "--expect-parent",
        metavar="HUB_SHA",
        help="dry-run only: require the inspected Hub head to be exactly HUB_SHA",
    )
    args = parser.parse_args(argv)
    environ = os.environ if environ is None else environ
    profile = resolve_profile(args.profile)
    source_revision = args.source_revision.strip()

    # Reserve evidence before any provider access; never overwrite a receipt.
    try:
        handle = _reserve(args.report)
    except OSError as error:
        print(json.dumps({"state": "REFUSED_BEFORE_PUBLICATION",
                          "failure_type": type(error).__name__}, sort_keys=True))
        return 1

    report = base_report(profile, source_revision, args.publish)
    exit_code = 1
    try:
        require_sha(source_revision, "source revision")
        if args.allow_create and (not args.publish or not profile.allow_create):
            raise PublicationRefused("--allow-create requires a create-enabled publish profile")
        if args.publish:
            if args.expect_delta_since or args.expect_parent:
                raise PublicationRefused(
                    "--expect-delta-since and --expect-parent are dry-run assertions"
                )
            assert_publish_environment(environ)
            token: str | bool = str(environ.get("HF_TOKEN", "")).strip()
            if not token:
                raise PublicationRefused("HF_TOKEN is required for --publish")
        else:
            token = False
        with tempfile.TemporaryDirectory(prefix="oac-hub-") as cache:
            clients = hub_clients(token, Path(cache))
            report["huggingface_hub_version"] = clients["version"]
            if args.publish and clients["version"] != REVIEWED_HUB_CLIENT:
                raise PublicationRefused(
                    "publication requires the reviewed huggingface_hub "
                    f"{REVIEWED_HUB_CLIENT} client"
                )
            execute(
                profile,
                source_revision,
                report,
                api=clients["api"],
                read_file=clients["read_file"],
                publish=args.publish,
                not_found=clients["not_found"],
                operation_factory=clients["operation_factory"],
                allow_create_grant=args.allow_create,
            )
        if args.expect_parent:
            assert_expected_parent(args.expect_parent.strip(), report)
        if args.expect_delta_since:
            assert_expected_delta(profile, args.expect_delta_since.strip(), source_revision, report)
        exit_code = 0
    except PublicationRefused as error:
        if report["state"] != "BLOCKED":
            report["state"] = (
                "FAILED" if report["commit_attempted"] else
                "BLOCKED" if report["repository_created"] else "REFUSED"
            )
        report["refusal"] = str(error)
    except Exception as error:  # evidence only; never reflect provider text
        report["state"] = "FAILED"
        report["failure_type"] = type(error).__name__
        report["failure_sha256"] = sha256_bytes(str(error).encode("utf-8"))
    with handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "profile": profile.key,
                "mode": report["mode"],
                "state": report["state"],
                "repo_id": profile.repo_id,
                "parent_revision": report["parent_revision"],
                "delta": report["delta"],
                "new_revision": report["new_revision"],
                "refusal": report.get("refusal"),
                "failure_type": report.get("failure_type"),
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
