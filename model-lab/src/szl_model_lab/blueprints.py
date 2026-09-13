"""Code-only HF projections. Reuses Forge credentials; never publishes weights.

Only an explicitly dispatched protected-main workflow may publish. The default
operation builds a plan from exact Git objects without contacting any service.
Neither a successful upload nor a manifest is a trained-model qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from .catalog import TRACKS, track_for
from .safeio import canonical_bytes, read_regular, strict_json, write_new

REPOSITORY = "szl-holdings/szl-forge"
SCHEMA = "szl.model-lab.blueprint/v1"
SOURCE_FILES = (
    "pyproject.toml", "LICENSE", "README.md", "constraints-test.txt",
    "docs/DATA_CONTRACT.md", "docs/THREAT_MODEL.md", "docs/POOL_VIEW.md",
    "src/szl_model_lab/__init__.py", "src/szl_model_lab/catalog.py",
    "src/szl_model_lab/models.py", "src/szl_model_lab/data.py",
    "src/szl_model_lab/safeio.py", "src/szl_model_lab/artifacts.py",
    "src/szl_model_lab/training.py", "src/szl_model_lab/probes.py",
    "src/szl_model_lab/pool_view.py",
    "src/szl_model_lab/app.py", "src/szl_model_lab/cli.py",
    "src/szl_model_lab/blueprints.py", "src/szl_model_lab/templates/index.html",
)


def source_payload(root: Path, revision: str, track: str) -> dict[str, bytes]:
    """Export an explicit source allowlist, never weights, secrets or test fixtures."""
    item = track_for(track)
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("full_github_revision_required")
    payload = {}
    for relative in SOURCE_FILES:
        process = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:model-lab/{relative}"],
            check=True, capture_output=True, timeout=15,
        )
        if not 0 < len(process.stdout) <= 1024 * 1024:
            raise ValueError("source_file_size_rejected")
        payload["source/" + relative] = process.stdout
    binding = {
        "schema": SCHEMA, "repo_id": item.proposed_hf_id, "track": track,
        "github_repository": REPOSITORY, "github_source_revision": revision,
        "state": "BLUEPRINT_NOT_TRAINED", "weights_present": False,
        "model_qualified": False, "signed": False,
        "files_sha256": {name: hashlib.sha256(body).hexdigest() for name, body in payload.items()},
    }
    payload["source-binding.json"] = canonical_bytes(binding)
    payload["README.md"] = f'''---
license: apache-2.0
library_name: pytorch
tags:
- model-blueprint
- not-trained
- advisory
- tabular-classification
---
# {item.display_name}

**BLUEPRINT_NOT_TRAINED. No trained weights or benchmark results are published.**

This repository contains executable architecture, training, evaluation, backend,
and Python-rendered workbench source. It is not an inference-ready checkpoint,
not an LLM, and not a conversion or replacement of an existing SZL kernel.

Architecture: 8 -> 16 -> 1 MLP, 161 trainable parameters. Target: `{item.target}`.
Features, in order: {', '.join(item.features)}.

{item.boundary}

## Exact source

Canonical GitHub source: https://github.com/{REPOSITORY}/tree/{revision}/model-lab

The `source/` directory contains exact committed bytes from that revision.
`source-binding.json` records their SHA-256 hashes. This unsigned content binding
is not independent provenance, a model evaluation, or an authorization receipt.

## Local use

Read `source/README.md`, then install the package from `source/` in an isolated
Python environment. `szl-model-lab plan` is read-only. Training is an explicit
owner operation from a clean canonical Git checkout, not from this HF mirror.
The authenticated workbench binds to loopback and has no training/publish API.
Do not call Transformers `from_pretrained` or Ollama on this code-only repository.

## Evidence and limits

No production data was used to create this blueprint. No benchmark, GPU support,
calibrated trust, safety, latency or deployment readiness is claimed. Dataset
rights, genuine training, held-out evaluation, and existing Forge release gates
must be satisfied before any trained artifact is published. Existing signed
models and Kernel Hub identities remain separate and unchanged.
'''.encode()
    return payload


def publish_context(environment: dict[str, str], revision: str) -> None:
    expected = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPOSITORY,
                "GITHUB_REF": "refs/heads/main", "GITHUB_SHA": revision,
                "GITHUB_EVENT_NAME": "workflow_dispatch"}
    if any(environment.get(key) != value for key, value in expected.items()):
        raise ValueError("protected_main_dispatch_context_required")


def publish_payload(api, track: str, payload: dict[str, bytes], *, download: Callable,
                    operation: Callable, journal: Callable[[str], None]) -> dict:
    """Conditional, no-delete publication with immutable byte readback; no retries."""
    from huggingface_hub.errors import RepositoryNotFoundError

    repo = track_for(track).proposed_hf_id
    binding = strict_json(payload.get("source-binding.json", b"{}"))
    expected_paths = {"source/" + name for name in SOURCE_FILES} | {"README.md", "source-binding.json"}
    if (set(payload) != expected_paths or not isinstance(binding, dict)
            or binding.get("schema") != SCHEMA or binding.get("repo_id") != repo
            or binding.get("track") != track or binding.get("github_repository") != REPOSITORY
            or binding.get("state") != "BLUEPRINT_NOT_TRAINED"
            or binding.get("weights_present") is not False
            or not re.fullmatch(r"[0-9a-f]{40}", str(binding.get("github_source_revision", "")))
            or any(not isinstance(body, bytes) or not 0 < len(body) <= 1024 * 1024 for body in payload.values())):
        raise ValueError("invalid_blueprint_publication_payload")
    expected_hashes = {name: hashlib.sha256(payload[name]).hexdigest()
                       for name in expected_paths - {"README.md", "source-binding.json"}}
    if binding.get("files_sha256") != expected_hashes:
        raise ValueError("blueprint_source_digest_mismatch")
    info = None
    try:
        info = api.model_info(repo, files_metadata=True)
    except RepositoryNotFoundError as exc:
        if getattr(exc.response, "status_code", None) != 404:
            raise ValueError("repo_visibility_unknown") from None
    if info is None:
        journal("CREATE_REQUEST_STARTED_READBACK_REQUIRED")
        api.create_repo(repo_id=repo, repo_type="model", private=False, exist_ok=False)
        info = api.model_info(repo, files_metadata=True)
    if not re.fullmatch(r"[0-9a-f]{40}", info.sha):
        raise ValueError("invalid_hf_revision")
    paths = {item.rfilename for item in info.siblings}
    if paths - set(payload) - {".gitattributes"}:
        raise ValueError("existing_nonblueprint_artifacts_preserved")
    with tempfile.TemporaryDirectory(prefix="szl-blueprint-readback-") as directory:
        def read(name: str, revision: str) -> bytes:
            path = download(repo_id=repo, filename=name, repo_type="model", revision=revision,
                            local_dir=directory, token=api.token)
            return read_regular(Path(path), 1024 * 1024)

        if paths - {".gitattributes"}:
            if "source-binding.json" not in paths:
                raise ValueError("existing_identity_without_blueprint_binding_preserved")
            prior = strict_json(read("source-binding.json", info.sha))
            if (not isinstance(prior, dict) or prior.get("schema") != SCHEMA
                    or prior.get("repo_id") != repo or prior.get("github_repository") != REPOSITORY
                    or prior.get("state") != "BLUEPRINT_NOT_TRAINED"
                    or prior.get("weights_present") is not False):
                raise ValueError("existing_identity_not_owned_by_blueprint_contract")
        matching = paths - {".gitattributes"} == set(payload) and all(
            read(name, info.sha) == body for name, body in payload.items())
        revision = info.sha
        if not matching:
            journal("COMMIT_REQUEST_STARTED_READBACK_REQUIRED")
            commit = api.create_commit(
                repo_id=repo, repo_type="model", revision="main", parent_commit=info.sha,
                commit_message="Publish exact-source untrained model blueprint",
                operations=[operation(path_in_repo=name, path_or_fileobj=body)
                            for name, body in payload.items()],
            )
            revision = commit.oid
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("invalid_publication_revision")
        observed = api.model_info(repo, revision=revision, files_metadata=True)
        if observed.sha != revision or {x.rfilename for x in observed.siblings} - {".gitattributes"} != set(payload):
            raise ValueError("publication_inventory_mismatch")
        if any(read(name, revision) != body for name, body in payload.items()):
            raise ValueError("publication_byte_mismatch")
    if api.model_info(repo).sha != revision:
        raise ValueError("hf_main_moved_during_publication")
    return {"schema": SCHEMA, "repo_id": repo, "hf_revision": revision,
            "github_source_revision": strict_json(payload["source-binding.json"])["github_source_revision"],
            "state": "CODE_ONLY_PUBLICATION_VERIFIED", "model_state": "BLUEPRINT_NOT_TRAINED",
            "weights_present": False, "signed": False, "model_qualified": False,
            "files_sha256": {name: hashlib.sha256(body).hexdigest() for name, body in payload.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=TRACKS, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    # Report destination is an owner-controlled local path, never a web input.
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.report.exists():
        raise ValueError("report_exists_do_not_replay")
    stages = []
    def journal(stage: str) -> None:
        stages.append(stage)
        write_new(args.report.with_name(args.report.name + f".attempt-{len(stages)}.json"),
                  canonical_bytes({"stage": stage, "track": args.track,
                                   "github_source_revision": args.source_revision}))
    try:
        root = Path(__file__).resolve().parents[3]
        payload = source_payload(root, args.source_revision, args.track)
        result = {"state": "PLAN_ONLY", "binding": strict_json(payload["source-binding.json"])}
        if args.publish:
            publish_context(dict(os.environ), args.source_revision)
            from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise ValueError("existing_forge_publisher_credential_required")
            result = publish_payload(HfApi(token=token), args.track, payload,
                                     download=hf_hub_download, operation=CommitOperationAdd, journal=journal)
        write_new(args.report, canonical_bytes(result))
        print(result["state"])
        return 0
    except Exception as exc:
        write_new(args.report, canonical_bytes({"state": "BLOCKED_OR_MUTATION_UNCONFIRMED",
                  "error_type": type(exc).__name__, "stages": stages, "training_started": False}))
        print("BLOCKED_OR_MUTATION_UNCONFIRMED; inspect redacted report before retry")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
