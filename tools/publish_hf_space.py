#!/usr/bin/env python3
"""Publish one Git-controlled Space subtree and verify exact live source."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


# Files the Hub rewrites after upload. ``.gitattributes`` gains
# ``filter=lfs`` rules for large uploads, so exact byte parity is impossible by
# design; the source rules must still all be present in the published copy.
HUB_MANAGED_METADATA = frozenset({".gitattributes"})

# Canonical Git LFS pointer. A pointer must never be uploaded in place of the
# object it names; a committed pointer binds the checked-out bytes by oid.
LFS_POINTER = re.compile(
    rb"version https://git-lfs\.github\.com/spec/v1\n"
    rb"oid sha256:([0-9a-f]{64})\n"
    rb"size (0|[1-9][0-9]{0,15})\n"
)
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
LFS_ATTRIBUTES = ("filter", "diff", "merge")


class PublishError(RuntimeError):
    """The Space publication or attestation contract failed."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_lfs_pointer(data: bytes) -> tuple[str, int] | None:
    """Return (oid, size) for an exact canonical LFS pointer, else None."""
    if len(data) > 1024:
        return None
    match = LFS_POINTER.fullmatch(data)
    if match is None:
        return None
    return match.group(1).decode("ascii"), int(match.group(2))


def gitattribute_states(text: str, path: str) -> dict[str, str]:
    """Evaluate one path against a Space-root ``.gitattributes`` (last match wins).

    This is what the Hub and the Docker builder's clone see. Values: the
    assigned value (e.g. ``lfs``), ``set``, ``unset`` (``-attr``) or absent.
    """
    states: dict[str, str] = {}
    name = PurePosixPath(path).name
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern, *tokens = line.split()
        anchored = "/" in pattern.rstrip("/")
        candidate = pattern.lstrip("/")
        if not (fnmatch.fnmatchcase(path, candidate) if anchored else fnmatch.fnmatchcase(name, pattern)):
            continue
        for token in tokens:
            if token == "binary":
                states.update({"diff": "unset", "merge": "unset", "text": "unset"})
            elif token.startswith("-"):
                states[token[1:]] = "unset"
            elif token.startswith("!"):
                states.pop(token[1:], None)
            elif "=" in token:
                key, value = token.split("=", 1)
                states[key] = value
            else:
                states[token] = "set"
    return states


def require_lfs_routing(gitattributes: str, target: str) -> dict[str, str]:
    """Fail closed if ``.gitattributes`` would not smudge ``target`` via LFS.

    Publishing a ``-filter`` rule makes the Hub Docker build check out the LFS
    pointer instead of the archive (the 2026-09-30 BUILD_ERROR), so the
    publisher refuses to regress the live LFS routing.
    """
    states = gitattribute_states(gitattributes, target)
    observed = {key: states.get(key, "unspecified") for key in LFS_ATTRIBUTES}
    if any(value != "lfs" for value in observed.values()):
        raise PublishError(
            f".gitattributes does not route {target} through LFS: {observed!r}"
        )
    return observed


def _lfs_tracked(repository_paths: list[str]) -> set[str]:
    """Repository paths whose checkout attribute is ``filter=lfs``."""
    if not repository_paths:
        return set()
    result = subprocess.run(
        ["git", "check-attr", "-z", "--stdin", "filter"],
        cwd=ROOT,
        input=b"\0".join(path.encode("utf-8") for path in repository_paths) + b"\0",
        check=False,
        capture_output=True,
    )
    fields = (result.stdout or b"").split(b"\0") if result.returncode == 0 else []
    return {
        fields[index].decode("utf-8", "replace")
        for index in range(0, len(fields) - 2, 3)
        if fields[index + 2] == b"lfs"
    }


def _committed_lfs_pointer(repository_path: str) -> tuple[str, int] | None:
    result = subprocess.run(
        ["git", "cat-file", "blob", f"HEAD:{repository_path}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or not isinstance(result.stdout, bytes):
        return None
    return parse_lfs_pointer(result.stdout)


def tracked_source_files(
    source_dir: Path, *, static: bool = False
) -> dict[str, dict[str, Any]]:
    source_dir = source_dir.resolve()
    try:
        relative_source = source_dir.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise PublishError("source directory must be inside the repository") from exc
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", relative_source],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    files: dict[str, dict[str, Any]] = {}
    prefix = relative_source.rstrip("/") + "/"
    repository_paths = [raw.decode("utf-8") for raw in result.stdout.split(b"\0") if raw]
    lfs_tracked = _lfs_tracked(repository_paths)
    for repository_path in repository_paths:
        if not repository_path.startswith(prefix):
            raise PublishError(f"tracked path escaped source directory: {repository_path}")
        target = repository_path[len(prefix) :]
        normalized = PurePosixPath(target)
        if normalized.is_absolute() or ".." in normalized.parts or not target:
            raise PublishError(f"unsafe Space target path: {target!r}")
        source = ROOT / Path(repository_path)
        data = source.read_bytes()
        if parse_lfs_pointer(data) is not None:
            raise PublishError(
                f"LFS object not materialized (refusing to upload a pointer): {target}; "
                "check out with lfs: true"
            )
        entry: dict[str, Any] = {
            "source_path": repository_path,
            "size": len(data),
            "sha256": sha256_bytes(data),
        }
        committed = _committed_lfs_pointer(repository_path) if repository_path in lfs_tracked else None
        if committed is not None:
            oid, size = committed
            if entry["sha256"] != oid or entry["size"] != size:
                raise PublishError(
                    f"materialized LFS bytes do not match the committed pointer: {target}"
                )
            entry["lfs_pointer"] = {"oid": oid, "size": size}
        files[target] = entry
    required_files = {"README.md", "index.html"} if static else {
        "README.md",
        "Dockerfile",
    }
    missing = sorted(required_files - set(files))
    if missing:
        mode = "static" if static else "Docker"
        raise PublishError(
            f"{mode} Space source is missing required files: {', '.join(missing)}"
        )
    return files


def parse_frozen_archives(values: list[str]) -> dict[str, str]:
    """Parse repeatable ``TARGET=SHA256`` declarations."""
    archives: dict[str, str] = {}
    for value in values:
        target, separator, digest = value.partition("=")
        normalized = PurePosixPath(target)
        if (
            not separator
            or not target
            or normalized.is_absolute()
            or ".." in normalized.parts
            or SHA256_HEX.fullmatch(digest) is None
            or target in archives
        ):
            raise PublishError(f"invalid --frozen-archive declaration: {value!r}")
        archives[target] = digest
    return archives


def verify_frozen_archives_before(
    files: dict[str, dict[str, Any]],
    source_dir: Path,
    frozen_archives: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Bind each frozen archive to its digest before anything is uploaded."""
    evidence: dict[str, dict[str, Any]] = {}
    if not frozen_archives:
        return evidence
    if ".gitattributes" not in files:
        raise PublishError("frozen archives require a tracked Space .gitattributes")
    gitattributes = (source_dir / ".gitattributes").read_text(encoding="utf-8")
    for target, digest in sorted(frozen_archives.items()):
        entry = files.get(target)
        if entry is None:
            raise PublishError(f"frozen archive is not a tracked Space file: {target}")
        if entry["sha256"] != digest:
            raise PublishError(f"frozen archive digest mismatch before upload: {target}")
        pointer = entry.get("lfs_pointer")
        if pointer is not None and pointer["oid"] != digest:
            raise PublishError(f"frozen archive LFS pointer oid mismatch: {target}")
        evidence[target] = {
            "frozen_sha256": digest,
            "size": entry["size"],
            "lfs_pointer_oid": pointer["oid"] if pointer else None,
            "gitattributes": require_lfs_routing(gitattributes, target),
            "verified_before_upload": True,
            "verified_after_publish": False,
        }
    return evidence


def verify_frozen_archives_after(
    evidence: dict[str, dict[str, Any]],
    fetch: Any,
) -> dict[str, dict[str, Any]]:
    """Re-read the published archive bytes and published LFS routing."""
    if not evidence:
        return evidence
    published_attributes = fetch(".gitattributes").decode("utf-8")
    for target, item in evidence.items():
        if sha256_bytes(fetch(target)) != item["frozen_sha256"]:
            raise PublishError(f"published frozen archive digest mismatch: {target}")
        item["published_gitattributes"] = require_lfs_routing(published_attributes, target)
        item["verified_after_publish"] = True
    return evidence


def build_plan(
    source_dir: Path,
    repo_id: str,
    source_revision: str,
    *,
    static: bool = False,
    frozen_archives: dict[str, str] | None = None,
) -> dict[str, Any]:
    revision = source_revision.strip().lower()
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise PublishError("source revision must be an exact lowercase Git SHA")
    files = tracked_source_files(source_dir, static=static)
    frozen = verify_frozen_archives_before(files, source_dir, frozen_archives or {})
    return {
        "schema": "szl.hf-space-publication/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": repo_id,
        "source_revision": revision,
        "source_revision_variable": "SZL_GITHUB_SOURCE_REVISION",
        "source_dir": source_dir.resolve().relative_to(ROOT).as_posix(),
        "files": files,
        "frozen_archives": frozen,
        "publish": False,
        "hf_commit": None,
        "live": None,
        "ok": True,
    }


def live_origin(repo_id: str, *, static: bool = False) -> str:
    owner, name = repo_id.split("/", 1)
    host = "".join(
        char if char.isalnum() or char == "-" else "-"
        for char in f"{owner}-{name}".lower()
    ).strip("-")
    suffix = ".static.hf.space" if static else ".hf.space"
    return f"https://{host}{suffix}"


def ensure_space_repository(
    api: Any,
    repo_id: str,
    *,
    static: bool = False,
) -> dict[str, Any]:
    """Create a missing Space or prove write access to the existing one.

    Credential validation happens before this function. A missing target is
    created once with the declared SDK; ``exist_ok`` makes concurrent recovery
    race-safe. The post-create readback and write check remain mandatory.
    """

    from huggingface_hub.utils import RepositoryNotFoundError

    sdk = "static" if static else "docker"
    created = False
    try:
        info = api.space_info(repo_id, files_metadata=False)
    except RepositoryNotFoundError:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk=sdk,
            private=False,
            exist_ok=True,
        )
        created = True
        info = api.space_info(repo_id, files_metadata=False)

    observed_id = str(
        getattr(info, "id", None)
        or getattr(info, "repo_id", None)
        or repo_id
    )
    if observed_id.casefold() != repo_id.casefold():
        raise PublishError(
            f"Space identity mismatch: {observed_id!r} != {repo_id!r}"
        )
    api.auth_check(repo_id=repo_id, repo_type="space", write=True)
    return {
        "state": (
            "CREATED_AND_WRITE_CONFIRMED"
            if created
            else "EXISTING_AND_WRITE_CONFIRMED"
        ),
        "repo_id": repo_id,
        "sdk": sdk,
        "created": created,
        "visibility_on_create": "public" if created else None,
        "write_access": "CONFIRMED",
    }


def space_info_before_deadline(api: Any, repo_id: str, deadline: float) -> Any:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise PublishError("Space publication deadline exhausted before control-plane read")
    info = api.space_info(repo_id, files_metadata=False, timeout=min(60, remaining))
    if time.monotonic() >= deadline:
        raise PublishError("Space publication deadline exhausted during control-plane read")
    return info


def observe_space_volumes(info: Any) -> dict[str, Any]:
    """Fail closed on explicit attachments without treating absent metadata as zero."""

    runtime = getattr(info, "runtime", None)
    volumes = getattr(runtime, "volumes", None)
    if volumes is None:
        return {"state": "UNKNOWN", "count": None}
    count = len(volumes)
    if count:
        raise PublishError(
            f"Space has {count} explicitly attached volume(s); "
            "refusing publication pending owner review"
        )
    return {"state": "OBSERVED_ZERO", "count": 0}


def clear_legacy_space_volumes(
    api: Any,
    repo_id: str,
    *,
    wait_seconds: int = 60,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Remove externally configured volumes and verify the control-plane state."""

    deadline = min(time.monotonic() + wait_seconds,
                   deadline if deadline is not None else float("inf"))
    info = space_info_before_deadline(api, repo_id, deadline)
    runtime = getattr(info, "runtime", None)
    before = list(getattr(runtime, "volumes", None) or [])
    if before:
        api.delete_space_volumes(repo_id=repo_id)
        while time.monotonic() < deadline:
            info = space_info_before_deadline(api, repo_id, deadline)
            runtime = getattr(info, "runtime", None)
            if not list(getattr(runtime, "volumes", None) or []):
                break
            time.sleep(min(2, max(0, deadline - time.monotonic())))
        else:
            raise PublishError("Space volumes remained configured after removal")
    return {
        "requested": True,
        "before_count": len(before),
        "after_count": len(
            list(getattr(runtime, "volumes", None) or [])
        ),
        "state": "CLEARED_AND_OBSERVED",
    }


def wait_for_exact_running_space(
    api: Any,
    repo_id: str,
    expected_sha: str,
    *,
    wait_seconds: int,
    require_zero_volumes: bool = False,
    deadline: float | None = None,
) -> Any:
    deadline = min(time.monotonic() + wait_seconds,
                   deadline if deadline is not None else float("inf"))
    stable_zero_volume_observations = 0
    while time.monotonic() < deadline:
        info = space_info_before_deadline(api, repo_id, deadline)
        stage = str(getattr(getattr(info, "runtime", None), "stage", "")).upper()
        if info.sha == expected_sha and stage == "RUNNING":
            if not require_zero_volumes:
                return info
            runtime = getattr(info, "runtime", None)
            if not list(getattr(runtime, "volumes", None) or []):
                stable_zero_volume_observations += 1
                if stable_zero_volume_observations >= 2:
                    return info
            else:
                stable_zero_volume_observations = 0
        else:
            stable_zero_volume_observations = 0
        time.sleep(min(10, max(0, deadline - time.monotonic())))
    requirement = " and zero volumes" if require_zero_volumes else ""
    raise PublishError(
        "Space did not reach RUNNING at the exact published Hugging Face commit"
        f"{requirement}"
    )


def wait_for_space_restart_transition(
    api: Any,
    repo_id: str,
    *,
    wait_seconds: int = 120,
    deadline: float | None = None,
) -> dict[str, Any]:
    deadline = min(time.monotonic() + wait_seconds,
                   deadline if deadline is not None else float("inf"))
    while time.monotonic() < deadline:
        runtime = getattr(space_info_before_deadline(api, repo_id, deadline), "runtime", None)
        stage = str(getattr(runtime, "stage", "")).upper()
        domains = (getattr(runtime, "raw", None) or {}).get("domains", [])
        domain_stages = [
            str(domain.get("stage", "")).upper()
            for domain in domains
            if isinstance(domain, dict)
        ]
        if stage and (stage != "RUNNING" or any(value and value != "READY" for value in domain_stages)):
            return {
                "observed": True,
                "runtime_stage": stage or None,
                "domain_stages": domain_stages,
            }
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    raise PublishError("Space restart was requested but no transition was observed")


def reconcile_final_space_volumes(
    api: Any,
    repo_id: str,
    expected_sha: str,
    *,
    wait_seconds: int,
    deadline: float | None = None,
) -> tuple[dict[str, Any], Any]:
    deadline = min(time.monotonic() + wait_seconds,
                   deadline if deadline is not None else float("inf"))
    evidence = clear_legacy_space_volumes(api, repo_id, deadline=deadline)
    restart_requested = evidence["before_count"] > 0
    if restart_requested:
        api.restart_space(repo_id=repo_id)
        evidence["restart_transition"] = wait_for_space_restart_transition(
            api,
            repo_id,
            deadline=deadline,
        )
    info = wait_for_exact_running_space(
        api,
        repo_id,
        expected_sha,
        wait_seconds=wait_seconds,
        require_zero_volumes=True,
        deadline=deadline,
    )
    evidence["restart_requested"] = restart_requested
    evidence["final_count"] = 0
    return evidence, info


def verify_published_bytes(
    files: dict[str, dict[str, Any]],
    source_dir: Path,
    fetch: Any,
) -> dict[str, dict[str, Any]]:
    """Require exact bytes for every payload file at the published revision.

    Hub-managed metadata (``HUB_MANAGED_METADATA``) is verified by content
    instead: every non-empty source line must appear verbatim in the published
    copy, and any extra published lines must be Hub LFS tracking rules. The
    relaxation is recorded in the returned receipt fragment so the publication
    receipt never claims byte parity for those files.
    """
    receipt: dict[str, dict[str, Any]] = {}
    for target, expected in files.items():
        remote = fetch(target)
        if target not in HUB_MANAGED_METADATA:
            if sha256_bytes(remote) != expected["sha256"]:
                raise PublishError(f"immutable Space byte mismatch: {target}")
            continue
        source_lines = [
            line for line in (source_dir / Path(target)).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        remote_lines = [line for line in remote.decode("utf-8").splitlines() if line.strip()]
        missing = [line for line in source_lines if line not in remote_lines]
        extra = [line for line in remote_lines if line not in source_lines]
        foreign = [line for line in extra if "filter=lfs" not in line]
        if missing or foreign:
            raise PublishError(
                f"Hub-managed metadata drift: {target} missing={missing!r} foreign={foreign!r}"
            )
        receipt[target] = {
            "verification": "SOURCE_LINES_PRESENT_NOT_BYTE_PARITY",
            "source_sha256": expected["sha256"],
            "published_sha256": sha256_bytes(remote),
            "hub_appended_lines": extra,
        }
    return receipt


def wait_for_exact_runtime_source(
    session: Any,
    origin: str,
    expected_revision: str,
    *,
    deadline: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Observe the new process after an asynchronous Space-variable rollout.

    The deadline is established before waiting for Hub RUNNING, so an old
    healthy process cannot start a second full publication wait budget.
    """
    import requests

    started = time.monotonic()
    attempts = 0
    observed_revisions: list[str] = []
    last_status: int | None = None
    while (remaining := deadline - time.monotonic()) > 0:
        attempts += 1
        build = None
        try:
            response = session.get(
                origin + "/api/build-info",
                timeout=min(90, remaining),
                allow_redirects=False,
            )
            last_status = response.status_code
            if response.status_code == 200:
                try:
                    candidate = response.json()
                except ValueError:
                    candidate = None
                if isinstance(candidate, dict):
                    build = candidate
        except requests.RequestException:
            last_status = None

        identity = build.get("build") if build else None
        if isinstance(identity, dict):
            revision = identity.get("revision")
            if (isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision)
                    and revision not in observed_revisions and len(observed_revisions) < 8):
                observed_revisions.append(revision)
            if (identity.get("state") == "OBSERVED"
                    and revision == expected_revision
                    and build.get("receipt_minted") is False
                    and time.monotonic() < deadline):
                return build, {
                    "state": "EXACT_SOURCE_OBSERVED",
                    "attempts": attempts,
                    "observed_revisions": observed_revisions,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "shared_publication_deadline": True,
                    "redirects_followed": False,
                }
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    raise PublishError(
        "runtime source binding deadline exhausted: "
        f"expected={expected_revision}, observed={observed_revisions!r}, "
        f"last_http_status={last_status!r}, attempts={attempts}"
    )


def verify_live_smoke_paths(
    session: Any,
    origin: str,
    paths: list[str],
    *,
    deadline: float,
    probes: dict[str, Any],
    max_attempts: int = 30,
    static: bool = False,
) -> None:
    """Probe directly, with one explicit static-host entry-point contract.

    Static root may return 302 to the exact relative /index.html, which must
    also be probed independently. Never enable automatic redirect following.
    Transient retries retain the shared deadline and every observed status.
    """

    import requests

    static_entry_contract = (
        static
        and re.fullmatch(r"https://[a-z0-9][a-z0-9-]*\.static\.hf\.space", origin) is not None
        and "/index.html" in paths
    )
    transient_statuses = {408, 429, 500, 502, 503, 504}
    for path in paths:
        observations: list[dict[str, Any]] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or len(observations) >= max_attempts:
                raise PublishError(
                    f"live smoke probe failed: {path}: deadline or attempt limit "
                    f"exhausted, attempts={len(observations)}"
                )
            try:
                response = session.get(
                    origin + path,
                    timeout=min(90, remaining),
                    allow_redirects=False,
                )
                observation = {
                    "status": response.status_code,
                    "bytes": len(response.content),
                    "content_type": response.headers.get("content-type"),
                }
                if (static_entry_contract and path == "/" and response.status_code == 302
                        and response.headers.get("location") == "/index.html"):
                    observation.update({
                        "verification": "EXPECTED_STATIC_ENTRY_REDIRECT",
                        "redirect_target_probe": "/index.html",
                        "redirects_followed": False,
                    })
            except requests.RequestException as exc:
                observation = {
                    "status": None,
                    "bytes": 0,
                    "content_type": None,
                    "error_class": type(exc).__name__,
                }
            observations.append(observation)
            probes[path] = {
                **observation,
                "attempt_count": len(observations),
                "attempts": observations,
            }
            status = observation["status"]
            if status == 200 and observation["bytes"] and time.monotonic() < deadline:
                break
            if (observation.get("verification") == "EXPECTED_STATIC_ENTRY_REDIRECT"
                    and time.monotonic() < deadline):
                break
            if status not in transient_statuses and status not in (None, 200):
                raise PublishError(
                    f"live smoke probe failed: {path}: status={status}, "
                    f"attempts={len(observations)}"
                )
            if len(observations) >= max_attempts:
                raise PublishError(
                    f"live smoke probe failed: {path}: status={status}, "
                    f"attempt limit exhausted, attempts={len(observations)}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PublishError(
                    f"live smoke probe failed: {path}: status={status}, "
                    f"deadline exhausted, attempts={len(observations)}"
                )
            time.sleep(min(2, remaining))


def publish_and_verify(
    plan: dict[str, Any],
    *,
    token: str,
    source_dir: Path,
    smoke_paths: list[str],
    wait_seconds: int,
    static: bool,
    clear_space_volumes: bool,
    reject_attached_space_volumes: bool = False,
) -> dict[str, Any]:
    import requests
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    if clear_space_volumes and reject_attached_space_volumes:
        raise PublishError("Space volume deletion and read-only guard are mutually exclusive")
    api = HfApi(token=token)
    repo_id = plan["repo_id"]
    plan["repository_reconciliation"] = ensure_space_repository(
        api,
        repo_id,
        static=static,
    )
    if clear_space_volumes:
        plan["volume_reconciliation"] = {
            "pre_publish": clear_legacy_space_volumes(api, repo_id)
        }
    if reject_attached_space_volumes:
        plan["volume_observation"] = {
            "policy": "READ_ONLY_REJECT_EXPLICIT_ATTACHMENTS",
            "pre_publish": observe_space_volumes(
                api.space_info(repo_id, files_metadata=False)
            ),
        }
    live_files = set(api.list_repo_files(repo_id=repo_id, repo_type="space"))
    expected_files = set(plan["files"])
    operations: list[Any] = [
        CommitOperationAdd(
            path_in_repo=target,
            path_or_fileobj=str(source_dir / Path(target)),
        )
        for target in sorted(expected_files)
    ]
    operations.extend(
        CommitOperationDelete(path_in_repo=target)
        for target in sorted(live_files - expected_files)
    )
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="space",
        operations=operations,
        commit_message=(
            f"deploy: exact szl-forge source {plan['source_revision'][:12]}"
        ),
        commit_description=(
            "Git-controlled Space subtree publication with post-commit byte and "
            "runtime verification.\n\nSigned-off-by: SZL Holdings "
            "<noreply@szlholdings.ai>"
        ),
    )
    plan["publish"] = True
    plan["hf_commit"] = commit.oid
    if not static:
        api.add_space_variable(
            repo_id=repo_id,
            key=plan["source_revision_variable"],
            value=plan["source_revision"],
            description=(
                "Exact szl-holdings/szl-forge protected Git revision deployed to this Space."
            ),
        )
        values = api.get_space_variables(repo_id=repo_id)
        value_item = values[plan["source_revision_variable"]]
        observed_variable = (
            value_item.get("value")
            if isinstance(value_item, dict)
            else getattr(value_item, "value", None)
        )
        if observed_variable != plan["source_revision"]:
            raise PublishError(
                f"Space source variable mismatch: {observed_variable!r}"
            )

    publication_deadline = time.monotonic() + wait_seconds
    info = wait_for_exact_running_space(
        api,
        repo_id,
        commit.oid,
        wait_seconds=wait_seconds,
        deadline=publication_deadline,
    )

    if clear_space_volumes:
        post_publish, info = reconcile_final_space_volumes(
            api,
            repo_id,
            commit.oid,
            wait_seconds=wait_seconds,
            deadline=publication_deadline,
        )
        plan["volume_reconciliation"]["post_publish"] = post_publish
        plan["volume_reconciliation"]["final_count"] = 0
    if reject_attached_space_volumes:
        plan["volume_observation"]["post_publish"] = observe_space_volumes(info)

    from huggingface_hub import hf_hub_download

    def fetch_published(target: str) -> bytes:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="space",
                filename=target,
                revision=commit.oid,
                token=token,
                force_download=True,
            )
        ).read_bytes()

    plan["hub_managed_metadata"] = verify_published_bytes(
        plan["files"], source_dir, fetch_published
    )
    plan["frozen_archives"] = verify_frozen_archives_after(
        plan.get("frozen_archives") or {}, fetch_published
    )

    origin = live_origin(repo_id, static=static)
    probes: dict[str, Any] = {}
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "Cache-Control": "no-cache, no-store, max-age=0",
            "User-Agent": "szl-forge-space-publisher/1",
        }
    )
    build = None
    if not static:
        build, plan["runtime_source_wait"] = wait_for_exact_runtime_source(
            session,
            origin,
            plan["source_revision"],
            deadline=publication_deadline,
        )
    plan["smoke_verification"] = {"state": "IN_PROGRESS", "probes": probes}
    try:
        verify_live_smoke_paths(
            session,
            origin,
            smoke_paths,
            deadline=publication_deadline,
            probes=probes,
            static=static,
        )
    except Exception:  # noqa: BLE001 - preserve failed smoke evidence on any error
        plan["smoke_verification"]["state"] = "FAILED"
        raise
    plan["smoke_verification"]["state"] = "VERIFIED"
    plan["live"] = {
        "origin": origin,
        "hf_commit": info.sha,
        "runtime_stage": getattr(getattr(info, "runtime", None), "stage", None),
        "source_revision": (
            build["build"]["revision"] if build is not None else plan["source_revision"]
        ),
        "source_revision_evidence": (
            "RUNTIME_VARIABLE_READBACK"
            if build is not None
            else "EXACT_HF_COMMIT_PLUS_BYTE_PARITY"
        ),
        "receipt_minted": False,
        "probes": probes,
    }
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--clear-space-volumes",
        action="store_true",
        help=(
            "remove externally configured Space volumes before publication and "
            "reconcile the final exact runtime to zero volumes"
        ),
    )
    parser.add_argument(
        "--reject-attached-space-volumes",
        action="store_true",
        help=(
            "read Space volume metadata and block explicitly attached volumes "
            "without deleting any volumes; omitted metadata remains UNKNOWN"
        ),
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="use the static Space host and exact byte parity instead of runtime variable readback",
    )
    parser.add_argument(
        "--frozen-archive",
        action="append",
        default=[],
        metavar="TARGET=SHA256",
        help=(
            "repeatable: Space file whose bytes (and LFS pointer oid) must equal the "
            "frozen digest before upload and after publication, and which "
            ".gitattributes must route through LFS"
        ),
    )
    parser.add_argument("--wait-seconds", type=int, default=1800)
    parser.add_argument(
        "--smoke-path",
        action="append",
        default=[],
        help="repeatable same-host GET path",
    )
    args = parser.parse_args()
    report_path = Path(args.report)
    plan: dict[str, Any] = {}
    try:
        source_dir = (ROOT / args.source_dir).resolve()
        plan = build_plan(
            source_dir,
            args.repo_id,
            args.source_revision,
            static=args.static,
            frozen_archives=parse_frozen_archives(args.frozen_archive),
        )
        if args.publish:
            token = os.environ.get("HF_TOKEN", "")
            if not token:
                raise PublishError("HF_TOKEN is required with --publish")
            plan = publish_and_verify(
                plan,
                token=token,
                source_dir=source_dir,
                smoke_paths=args.smoke_path
                or ["/live", "/health", "/api/build-info", "/api/v1/identity"],
                wait_seconds=args.wait_seconds,
                static=args.static,
                clear_space_volumes=args.clear_space_volumes,
                reject_attached_space_volumes=args.reject_attached_space_volumes,
            )
    except Exception as exc:  # noqa: BLE001 - always emit terminal evidence
        partial = plan
        plan = {
            "schema": "szl.hf-space-publication/v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "fatal": f"{type(exc).__name__}: {exc}",
        }
        for key in (
            "repo_id", "source_revision", "hf_commit", "volume_observation",
            "runtime_source_wait", "smoke_verification",
        ):
            if key in partial:
                plan[key] = partial[key]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    report_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if plan.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
