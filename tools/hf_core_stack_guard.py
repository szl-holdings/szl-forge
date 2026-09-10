"""Fail-closed contract for the HF 2026-09-10 core-stack evaluation.

This module does not install packages, run models, change production defaults, or
claim hardware performance. It validates exact release identity and emits the
bounded test plan that a real evaluator must satisfy.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

SHA40 = re.compile(r"^[0-9a-f]{40}$")
EXPECTED = {
    "transformers": {
        "version": "5.17.0",
        "revision": "856157a2f3e9594954310df18fdccc31ffddebe9",
    },
    "accelerate": {
        "version": "1.15.0",
        "revision": "6afc1e5ee217051fde702b23de2813344dc0fd33",
    },
    "trl": {
        "version": "1.13.0",
        "revision": "3d9261f1fec9f9a8140099c78a65c7da73dce79c",
    },
}

DENIED_AUTHORITY = {
    "productionDependencyPromotion": False,
    "productionRouteChange": False,
    "hubPublication": False,
    "weightRehosting": False,
    "automaticPromotion": False,
}

REQUIRED_CHECKS = (
    "transformers_import_api",
    "transformers_generation_no_unconditional_hub_download",
    "transformers_cache_negative_path",
    "transformers_kernel_fallback_visibility",
    "transformers_vision_rope_compatibility",
    "accelerate_fsdp2_activation_checkpointing",
    "accelerate_fsdp2_checkpoint_offload",
    "accelerate_checkpoint_save_load",
    "accelerate_dtensor_grad_clip",
    "accelerate_peft_full_state",
    "trl_chunked_nll",
    "trl_fused_loss_parity",
    "trl_long_context_config",
    "trl_vllm_fail_fast",
    "trl_removed_ppo_migration",
    "rollback_to_prior_stack",
)

ALLOWED_RESULTS = frozenset({"PASS", "FAIL", "UNAVAILABLE"})


class CoreStackError(ValueError):
    """Incomplete or mismatched evidence must not become permission."""


@dataclass(frozen=True)
class ReleaseObservation:
    package: str
    version: str
    revision: str

    def validate(self) -> None:
        expected = EXPECTED.get(self.package)
        if expected is None:
            raise CoreStackError(f"unexpected package: {self.package}")
        if self.version != expected["version"]:
            raise CoreStackError(f"version mismatch for {self.package}")
        if not SHA40.fullmatch(self.revision) or self.revision != expected["revision"]:
            raise CoreStackError(f"revision mismatch for {self.package}")


def plan() -> dict[str, Any]:
    """Return the exact-source bounded evaluation plan; never a promotion grant."""
    return {
        "disposition": "HOLD",
        "exactSources": EXPECTED,
        "requiredChecks": list(REQUIRED_CHECKS),
        "allowedResults": sorted(ALLOWED_RESULTS),
        "hardwareRule": "Use UNAVAILABLE when the required backend is absent; never simulate PASS.",
        "authority": dict(DENIED_AUTHORITY),
        "claimBoundary": "Upstream benchmark numbers are references only until reproduced under SZL-controlled evidence.",
    }


def validate_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CoreStackError("evidence must be an object")
    releases = payload.get("releases")
    checks = payload.get("checks")
    if not isinstance(releases, list) or not isinstance(checks, dict):
        raise CoreStackError("missing releases/checks")
    observed = set()
    for row in releases:
        if not isinstance(row, dict):
            raise CoreStackError("invalid release observation")
        obs = ReleaseObservation(str(row.get("package")), str(row.get("version")), str(row.get("revision")))
        obs.validate()
        if obs.package in observed:
            raise CoreStackError("duplicate release observation")
        observed.add(obs.package)
    if observed != set(EXPECTED):
        raise CoreStackError("incomplete release set")

    normalized: dict[str, str] = {}
    for name in REQUIRED_CHECKS:
        result = checks.get(name)
        if result not in ALLOWED_RESULTS:
            raise CoreStackError(f"invalid or missing result for {name}")
        normalized[name] = result

    # Evaluation evidence can establish compatibility, never automatic promotion.
    return {
        "disposition": "HOLD",
        "allChecksPass": all(value == "PASS" for value in normalized.values()),
        "checks": normalized,
        "authority": dict(DENIED_AUTHORITY),
    }


def main() -> None:
    print(json.dumps(plan(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
