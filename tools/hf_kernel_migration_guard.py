#!/usr/bin/env python3
"""Fail-closed Hugging Face Kernel Hub migration checks for SZL Forge.

This module is intentionally additive to the existing release publisher so the
0.16.1 migration can be validated independently before the legacy compatibility
lane is retired. GitHub remains the source of truth; Hugging Face is a governed
runtime projection.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Iterable

from huggingface_hub import HfApi

KERNELS_CLIENT_VERSION = "0.16.1"
FIRST_CLASS_REPO_TYPE = "kernel"
LEGACY_REPO_TYPE = "model"
REQUIRED_BRANCHES = ("main", "v1")
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class KernelMigrationError(RuntimeError):
    """Raised when the first-class Kernel Hub contract is not satisfied."""


@dataclass(frozen=True)
class KernelEvidence:
    repo_id: str
    repo_type: str
    client_version: str
    repo_revision: str
    branch_revisions: dict[str, str]
    runtime_revision: str | None = None
    runtime_loaded: bool = False


def installed_kernels_version() -> str:
    try:
        return version("kernels")
    except PackageNotFoundError as exc:
        raise KernelMigrationError("kernels package is not installed") from exc


def require_supported_client(observed: str | None = None) -> str:
    observed = observed or installed_kernels_version()
    if observed != KERNELS_CLIENT_VERSION:
        raise KernelMigrationError(
            "Hugging Face kernels client drifted: "
            f"expected {KERNELS_CLIENT_VERSION}, observed {observed}"
        )
    return observed


def require_first_class_repo_type(repo_type: str) -> None:
    if repo_type == LEGACY_REPO_TYPE:
        raise KernelMigrationError(
            "legacy model-type kernel repositories are forbidden in the migrated lane"
        )
    if repo_type != FIRST_CLASS_REPO_TYPE:
        raise KernelMigrationError(f"unsupported kernel repository type: {repo_type}")


def _branch_targets(api: HfApi, repo_id: str, token: str | None) -> dict[str, str]:
    refs = api.list_repo_refs(repo_id, repo_type=FIRST_CLASS_REPO_TYPE, token=token)
    branches = {
        branch.name: branch.target_commit
        for branch in refs.branches
        if branch.name in REQUIRED_BRANCHES
    }
    missing = sorted(set(REQUIRED_BRANCHES) - set(branches))
    if missing:
        raise KernelMigrationError(f"first-class kernel branches missing: {missing}")
    for branch, revision in branches.items():
        if FULL_SHA_RE.fullmatch(revision or "") is None:
            raise KernelMigrationError(
                f"branch {branch} is not bound to an exact 40-character Git SHA"
            )
    return branches


def verify_first_class_kernel(
    repo_id: str,
    *,
    api: HfApi | None = None,
    token: str | None = None,
    client_version: str | None = None,
) -> KernelEvidence:
    """Verify the Hub object is a first-class Kernel repo with exact branch heads."""
    client_version = require_supported_client(client_version)
    api = api or HfApi(token=token)
    info = api.repo_info(
        repo_id,
        repo_type=FIRST_CLASS_REPO_TYPE,
        files_metadata=True,
        token=token,
    )
    repo_revision = getattr(info, "sha", None)
    if FULL_SHA_RE.fullmatch(repo_revision or "") is None:
        raise KernelMigrationError("kernel repository revision is not an exact Git SHA")
    branches = _branch_targets(api, repo_id, token)
    return KernelEvidence(
        repo_id=repo_id,
        repo_type=FIRST_CLASS_REPO_TYPE,
        client_version=client_version,
        repo_revision=repo_revision,
        branch_revisions=branches,
    )


def verify_runtime_load(
    evidence: KernelEvidence,
    *,
    backend: str = "cpu",
    get_kernel_fn: Any | None = None,
) -> KernelEvidence:
    """Load the v1 package through kernels 0.16.1 using the exact Hub revision."""
    require_first_class_repo_type(evidence.repo_type)
    require_supported_client(evidence.client_version)
    revision = evidence.branch_revisions["v1"]
    if get_kernel_fn is None:
        from kernels import get_kernel

        get_kernel_fn = get_kernel
    module = get_kernel_fn(
        evidence.repo_id,
        revision=revision,
        backend=backend,
        trust_remote_code=True,
    )
    if module is None:
        raise KernelMigrationError("get_kernel returned no module")
    return KernelEvidence(
        repo_id=evidence.repo_id,
        repo_type=evidence.repo_type,
        client_version=evidence.client_version,
        repo_revision=evidence.repo_revision,
        branch_revisions=evidence.branch_revisions,
        runtime_revision=revision,
        runtime_loaded=True,
    )


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="SZLHOLDINGS/szl-kernels")
    parser.add_argument("--token", default=None)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--backend", default="cpu")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    evidence = verify_first_class_kernel(args.repo_id, token=args.token)
    if args.runtime:
        evidence = verify_runtime_load(evidence, backend=args.backend)
    print(canonical_json(asdict(evidence)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
