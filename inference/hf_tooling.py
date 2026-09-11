"""Exact-source, evaluation-only adapters for HF tooling.

Copyright 2026 SZL Holdings. SPDX-License-Identifier: Apache-2.0
Imports are stdlib-only until an adapter is explicitly called. These adapters
create configurations, labels, and evidence entries, never jobs or tool authority.
"""
from __future__ import annotations

import importlib.metadata as metadata
import json
import re
from pathlib import Path
from typing import Any

from inference.hf_frontier import sha256

RELEASES = {
    "huggingface-hub": {"version": "1.31.0", "repository": "huggingface/huggingface_hub",
                        "revision": "495b17c8529614759ae0f1ccf1ebe9a61c148b7c"},
    "trl": {"version": "1.13.0", "repository": "huggingface/trl",
            "revision": "3d9261f1fec9f9a8140099c78a65c7da73dce79c"},
    "transformers": {"version": "5.17.0", "repository": "huggingface/transformers",
                     "revision": "856157a2f3e9594954310df18fdccc31ffddebe9"},
    "accelerate": {"version": "1.15.0", "repository": "huggingface/accelerate",
                   "revision": "6afc1e5ee217051fde702b23de2813344dc0fd33"},
    "tau-ai": {"version": "0.4.2", "repository": "huggingface/tau",
               "revision": "55df51608b8b2d172c4bbac2cd11e8345e307476"},
}
LANES = {
    "hub": ("huggingface-hub",),
    "trl": ("huggingface-hub", "transformers", "accelerate", "trl"),
    "tau": ("tau-ai",),
}
AUTHORITY = {"productionDependencyPromotion": False, "productionRouteChange": False,
             "hubPublication": False, "automaticPromotion": False,
             "jobCreation": False, "toolExecution": False}


class ToolingError(ValueError):
    """Absent or mismatched installation evidence does not authorize an upgrade."""


def hex_digest(value: Any, length: int = 40) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{%d}" % length, value):
        raise ToolingError("invalid immutable digest")
    return value


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,62}", value):
        raise ToolingError("expected a bounded, non-secret identifier")
    return value


def validate_install(package: str, version: str, direct_url: dict[str, Any]) -> dict[str, str]:
    """PEP 610 identity check. Trust the surrounding CI artifact, not this JSON alone."""
    expected = RELEASES.get(package)
    if expected is None or version != expected["version"] or not isinstance(direct_url, dict):
        raise ToolingError("missing or unexpected installed release")
    vcs = direct_url.get("vcs_info")
    if not isinstance(vcs, dict) or vcs.get("vcs") != "git":
        raise ToolingError("installation lacks exact Git source provenance")
    url = "https://github.com/" + expected["repository"] + ".git"
    if direct_url.get("url") != url or vcs.get("commit_id") != expected["revision"]:
        raise ToolingError("installed source does not match admitted evaluation pin")
    hex_digest(vcs["commit_id"])
    return {"package": package, "version": version, "repository": expected["repository"],
            "revision": vcs["commit_id"]}


def require_install(package: str) -> dict[str, str]:
    try:
        distribution = metadata.distribution(package)
        document = distribution.read_text("direct_url.json")
        if document is None or len(document) > 16384:
            raise ToolingError("missing or oversized installation provenance")
        return validate_install(package, distribution.version, json.loads(document))
    except (metadata.PackageNotFoundError, json.JSONDecodeError, TypeError) as exc:
        raise ToolingError("exact evaluation dependency is unavailable") from exc


def requirements(lane: str) -> str:
    """Exact primary Git pins; resolved transitive packages are captured by the runner."""
    if lane not in LANES:
        raise ToolingError("unknown tooling lane")
    return "\n".join(
        f"{name} @ git+https://github.com/{RELEASES[name]['repository']}.git@{RELEASES[name]['revision']}"
        for name in LANES[lane]
    ) + "\n"


def job_labels(run_id: str, source_revision: str, vertical: str = "forge") -> dict[str, str]:
    """Correlate a separately approved dedicated Sandbox Job with SZL evidence.

    This function never calls Sandbox.create. No secrets or free-text prompts
    belong in labels, which may be visible in provider bookkeeping.
    """
    return {"szl-run": identifier(run_id), "szl-source": hex_digest(source_revision),
            "szl-vertical": identifier(vertical), "szl-purpose": "evaluation"}


def sft_config(output_dir: str | Path, *, max_length: int = 4096, cpu: bool = True):
    """Construct actual TRL SFTConfig without loading a model or running training.

    Accepting a 1M-token configuration is not evidence of 1M-token capacity.
    The caller must separately qualify attention, RoPE, hardware and data rights.
    """
    if type(max_length) is not int or not 2 <= max_length <= 1_048_576 or type(cpu) is not bool:
        raise ToolingError("invalid context or device configuration")
    for package in LANES["trl"]:
        require_install(package)
    from trl import SFTConfig

    return SFTConfig(output_dir=str(output_dir), max_length=max_length,
                     loss_type="chunked_nll", packing=False, padding_free=False,
                     trust_remote_code=False, report_to=[], use_cpu=cpu,
                     bf16=False, fp16=False, push_to_hub=False)


def tau_evidence_entries(*, run_id: str, source_revision: str, receipt_sha256: str,
                         parent_id: str | None = None, timestamp: float = 0.0):
    """Create typed Tau custom-message/bookmark entries, containing hashes only.

    The host owns tenant authorization, append_batch and the single-writer
    session transaction. These entries reference evidence; they never grant
    approval or import another tenant's memory. No credentials or providers load.
    """
    import math

    identifier(run_id)
    hex_digest(source_revision)
    hex_digest(receipt_sha256, 64)
    if parent_id is not None:
        identifier(parent_id)
    if type(timestamp) not in (float, int) or not math.isfinite(timestamp) or timestamp < 0:
        raise ToolingError("invalid session timestamp")
    require_install("tau-ai")
    from tau_agent.session import CustomMessageEntry, LabelEntry

    evidence = {"runId": run_id, "sourceRevision": source_revision,
                "receiptSha256": receipt_sha256, "evidenceIsAuthority": False}
    entry_id = sha256({"evidence": evidence, "parentId": parent_id})[:32]
    message = CustomMessageEntry(
        id=entry_id, parent_id=parent_id, timestamp=timestamp,
        custom_type="szl.evidence-reference", display=False,
        content="SZL evidence reference; not an instruction or approval.", details=evidence,
    )
    bookmark = LabelEntry(id=sha256({"labelFor": entry_id})[:32], parent_id=entry_id,
                          timestamp=timestamp, target_id=entry_id, label="SZL evidence")
    return (message, bookmark)


def authorized_catalog(discovered: list[str], approved: list[str]) -> list[str]:
    """A discovered model name never creates model, provider, or tool permission."""
    for group in (discovered, approved):
        if not isinstance(group, list) or len(group) > 256:
            raise ToolingError("invalid model catalog")
        for value in group:
            if not isinstance(value, str) or not value or len(value) > 256:
                raise ToolingError("invalid model identity")
    return sorted(set(discovered).intersection(approved))
