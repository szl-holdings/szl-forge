#!/usr/bin/env python3
"""Publish one Git-controlled Space subtree and verify exact live source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class PublishError(RuntimeError):
    """The Space publication or attestation contract failed."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        repository_path = raw.decode("utf-8")
        if not repository_path.startswith(prefix):
            raise PublishError(f"tracked path escaped source directory: {repository_path}")
        target = repository_path[len(prefix) :]
        normalized = PurePosixPath(target)
        if normalized.is_absolute() or ".." in normalized.parts or not target:
            raise PublishError(f"unsafe Space target path: {target!r}")
        source = ROOT / Path(repository_path)
        data = source.read_bytes()
        files[target] = {
            "source_path": repository_path,
            "size": len(data),
            "sha256": sha256_bytes(data),
        }
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


def build_plan(
    source_dir: Path,
    repo_id: str,
    source_revision: str,
    *,
    static: bool = False,
) -> dict[str, Any]:
    revision = source_revision.strip().lower()
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise PublishError("source revision must be an exact lowercase Git SHA")
    files = tracked_source_files(source_dir, static=static)
    return {
        "schema": "szl.hf-space-publication/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": repo_id,
        "source_revision": revision,
        "source_revision_variable": "SZL_GITHUB_SOURCE_REVISION",
        "source_dir": source_dir.resolve().relative_to(ROOT).as_posix(),
        "files": files,
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


def mount_recovery_reason(info: Any, expected_sha: str) -> str | None:
    """Return the exact mount failure eligible for one factory reboot."""

    if getattr(info, "sha", None) != expected_sha:
        return None
    runtime = getattr(info, "runtime", None)
    stage = str(getattr(runtime, "stage", "")).upper()
    raw = getattr(runtime, "raw", {})
    error = str(raw.get("errorMessage", "")) if isinstance(raw, dict) else ""
    if stage == "RUNTIME_ERROR" and "hf-mount" in error.lower():
        return error
    return None


def publish_and_verify(
    plan: dict[str, Any],
    *,
    token: str,
    source_dir: Path,
    smoke_paths: list[str],
    wait_seconds: int,
    static: bool,
) -> dict[str, Any]:
    import requests
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    api = HfApi(token=token)
    repo_id = plan["repo_id"]
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

    deadline = time.monotonic() + wait_seconds
    info = None
    mount_recovery_attempted = False
    while time.monotonic() < deadline:
        info = api.space_info(repo_id, files_metadata=False)
        stage = str(getattr(getattr(info, "runtime", None), "stage", "")).upper()
        if info.sha == commit.oid and stage == "RUNNING":
            break
        recovery_reason = mount_recovery_reason(info, commit.oid)
        if recovery_reason is not None and not mount_recovery_attempted:
            api.restart_space(repo_id, factory_reboot=True)
            mount_recovery_attempted = True
            plan["mount_recovery"] = {
                "attempted": True,
                "mode": "FACTORY_REBOOT",
                "hf_commit": commit.oid,
                "reason": recovery_reason,
            }
        time.sleep(10)
    else:
        raise PublishError(
            "Space did not reach RUNNING at the exact published Hugging Face commit"
        )

    from huggingface_hub import hf_hub_download

    for target, expected in plan["files"].items():
        remote = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="space",
                filename=target,
                revision=commit.oid,
                token=token,
                force_download=True,
            )
        )
        if sha256_bytes(remote.read_bytes()) != expected["sha256"]:
            raise PublishError(f"immutable Space byte mismatch: {target}")

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
    for path in smoke_paths:
        response = session.get(origin + path, timeout=90)
        probes[path] = {
            "status": response.status_code,
            "bytes": len(response.content),
            "content_type": response.headers.get("content-type"),
        }
        if response.status_code != 200 or not response.content:
            raise PublishError(f"live smoke probe failed: {path}")
    build = None
    if not static:
        build = session.get(origin + "/api/build-info", timeout=90).json()
        if (
            build.get("build", {}).get("state") != "OBSERVED"
            or build.get("build", {}).get("revision") != plan["source_revision"]
            or build.get("receipt_minted") is not False
        ):
            raise PublishError(f"runtime source binding mismatch: {build!r}")
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
        "--static",
        action="store_true",
        help="use the static Space host and exact byte parity instead of runtime variable readback",
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
    try:
        source_dir = (ROOT / args.source_dir).resolve()
        plan = build_plan(
            source_dir,
            args.repo_id,
            args.source_revision,
            static=args.static,
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
            )
    except Exception as exc:  # noqa: BLE001 - always emit terminal evidence
        plan = {
            "schema": "szl.hf-space-publication/v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "fatal": f"{type(exc).__name__}: {exc}",
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    report_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if plan.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
