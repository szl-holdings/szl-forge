"""Fail-closed evidence contract for the Repo2RLEnv synthesis evaluation lane.

This module validates receipts only. It does not clone repositories, invoke providers,
run Docker, publish Hugging Face datasets, admit training data, or authorize production.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

UPSTREAM_REPOSITORY = "huggingface/Repo2RLEnv"
UPSTREAM_REVISION = "d1f7677265b10ae5deec1b9cc43425d715564d47"
UPSTREAM_RELEASE = "v0.8.7"
FRONTIER_MERGE = "2094061aa783c8c7aa89adc7b530a41de3469208"
ALLOWED_PIPELINES = frozenset({"pr_runtime", "commit_runtime"})
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SynthesisContractError(ValueError):
    """Invalid or incomplete synthesis evidence cannot become authority."""


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise SynthesisContractError(f"{field} must be an exact lowercase sha256")
    return value


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA40.fullmatch(value):
        raise SynthesisContractError(f"{field} must be an exact lowercase SHA-40")
    return value


def validate_task_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one generated task without trusting generated text as policy."""
    if receipt.get("upstream_repository") != UPSTREAM_REPOSITORY:
        raise SynthesisContractError("unexpected Repo2RLEnv repository")
    if receipt.get("upstream_revision") != UPSTREAM_REVISION:
        raise SynthesisContractError("receipt is not bound to admitted functional release")
    _sha(receipt.get("upstream_revision"), "upstream_revision")

    pipeline = receipt.get("pipeline")
    if pipeline not in ALLOWED_PIPELINES:
        raise SynthesisContractError("only stable runtime pipelines are admitted")
    if receipt.get("source_visibility") != "PUBLIC_NON_SECRET":
        raise SynthesisContractError("first lane accepts public non-secret repositories only")
    if receipt.get("provider_called") is not False:
        raise SynthesisContractError("external provider calls are not authorized")
    if receipt.get("hub_write_performed") is not False:
        raise SynthesisContractError("Hugging Face publication is not authorized")
    if receipt.get("training_admitted") is not False:
        raise SynthesisContractError("generated task is not admitted training data")
    if receipt.get("oracle_leak_detected") is not False:
        raise SynthesisContractError("oracle/gold-patch leakage fails closed")

    status = receipt.get("status")
    if status not in {"PASS", "FAIL", "UNAVAILABLE"}:
        raise SynthesisContractError("status must be PASS, FAIL, or UNAVAILABLE")

    normalized = {
        "upstreamRepository": UPSTREAM_REPOSITORY,
        "upstreamRevision": UPSTREAM_REVISION,
        "upstreamRelease": UPSTREAM_RELEASE,
        "frontierMerge": FRONTIER_MERGE,
        "sourceRepository": receipt.get("source_repository"),
        "sourceRevision": _sha(receipt.get("source_revision"), "source_revision"),
        "sourceEventId": receipt.get("source_event_id"),
        "pipeline": pipeline,
        "bootstrapDigest": _digest(receipt.get("bootstrap_sha256"), "bootstrap_sha256"),
        "taskContentHash": _digest(receipt.get("task_sha256"), "task_sha256"),
        "verifierDigest": _digest(receipt.get("verifier_sha256"), "verifier_sha256"),
        "rewardDigest": _digest(receipt.get("reward_sha256"), "reward_sha256"),
        "status": status,
        "productionAuthorized": False,
        "publicationAuthorized": False,
        "trainingAuthorized": False,
        "automaticPromotionAuthorized": False,
    }
    if not isinstance(normalized["sourceRepository"], str) or "/" not in normalized["sourceRepository"]:
        raise SynthesisContractError("source_repository must be explicit")
    if not isinstance(normalized["sourceEventId"], str) or not normalized["sourceEventId"].strip():
        raise SynthesisContractError("source_event_id must identify the mined PR or commit")
    return normalized


def evaluate_repeatability(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare repeated synthesis receipts; missing capability is HOLD, never PASS."""
    if len(receipts) < 2:
        raise SynthesisContractError("repeatability needs at least two receipts")
    rows = [validate_task_receipt(receipt) for receipt in receipts]
    available = all(row["status"] != "UNAVAILABLE" for row in rows)
    passed = available and all(row["status"] == "PASS" for row in rows)
    stable_source = len({(row["sourceRepository"], row["sourceRevision"], row["sourceEventId"], row["pipeline"]) for row in rows}) == 1
    stable_bootstrap = len({row["bootstrapDigest"] for row in rows}) == 1
    stable_task = len({row["taskContentHash"] for row in rows}) == 1
    stable_verifier = len({row["verifierDigest"] for row in rows}) == 1
    stable_reward = len({row["rewardDigest"] for row in rows}) == 1
    reproducible = passed and stable_source and stable_bootstrap and stable_task and stable_verifier and stable_reward
    return {
        "runCount": len(rows),
        "available": available,
        "allRunsPassed": passed,
        "stableSource": stable_source,
        "stableBootstrap": stable_bootstrap,
        "stableTask": stable_task,
        "stableVerifier": stable_verifier,
        "stableReward": stable_reward,
        "reproducible": reproducible,
        "disposition": "EVALUATION" if reproducible else "HOLD",
        "productionAuthorized": False,
        "publicationAuthorized": False,
        "trainingAuthorized": False,
        "automaticPromotionAuthorized": False,
    }
