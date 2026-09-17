"""Fail-closed evidence contract for Forge issue #325.

This module validates measured qualification evidence for Hugging Face Kernels
v0.17.0 at the exact admitted tag commit. It does not widen trust, publish
kernels, or alter a production default.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

REVISION = "7a11103e4a85074344de419993d48c1ff3064baf"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_CASES = (
    "trusted_repo_allowlist_positive",
    "trusted_repo_allowlist_negative",
    "min_version_incompatible_fails_before_execution",
    "min_version_compatible_passes",
    "conditional_kernel_false_uses_reference",
    "conditional_kernel_true_uses_kernel_path",
    "reference_output_parity",
    "kernels_data_packaging_migration",
    "helion",
)
ALLOWED = {"PASS", "FAIL", "UNAVAILABLE"}


class EvidenceError(ValueError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def plan() -> dict[str, Any]:
    return {
        "schema": "szl.forge.hf-kernels-0.17-evidence.v1",
        "issue": "szl-holdings/szl-forge#325",
        "disposition": "HOLD",
        "version": "0.17.0",
        "revision": REVISION,
        "requiredCases": list(REQUIRED_CASES),
        "authority": {
            "remoteKernelPublication": False,
            "productionTrustListWidening": False,
            "productionKernelDefault": False,
            "automaticPromotion": False,
        },
    }


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != plan()["schema"]:
        raise EvidenceError("schema mismatch")
    if payload.get("version") != "0.17.0":
        raise EvidenceError("version mismatch")
    revision = str(payload.get("revision", ""))
    if not SHA40.fullmatch(revision) or revision != REVISION:
        raise EvidenceError("revision mismatch")
    artifact = payload.get("artifact")
    if not isinstance(artifact, dict):
        raise EvidenceError("missing artifact identity")
    digest = str(artifact.get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise EvidenceError("artifact sha256 required")
    env = payload.get("environment")
    if not isinstance(env, dict):
        raise EvidenceError("missing environment")
    for key in ("python", "torch", "backend", "device", "driver", "compiler"):
        if not isinstance(env.get(key), str) or not env[key].strip():
            raise EvidenceError(f"missing environment.{key}")
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        raise EvidenceError("missing checks")
    normalized: dict[str, str] = {}
    for name in REQUIRED_CASES:
        result = checks.get(name)
        if result not in ALLOWED:
            raise EvidenceError(f"invalid or missing check: {name}")
        normalized[name] = result
    if normalized["helion"] == "UNAVAILABLE" and not env.get("helionUnavailableReason"):
        raise EvidenceError("Helion UNAVAILABLE requires reason")
    if normalized["trusted_repo_allowlist_negative"] != "PASS":
        raise EvidenceError("negative trust test must PASS before qualification")
    if normalized["min_version_incompatible_fails_before_execution"] != "PASS":
        raise EvidenceError("incompatible capability must fail before execution")
    return {
        "disposition": "HOLD",
        "allRequiredAvailableChecksPass": all(v in {"PASS", "UNAVAILABLE"} for v in normalized.values()),
        "checks": normalized,
        "authority": plan()["authority"],
    }


def validate_file(path: str) -> dict[str, Any]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    result = validate(payload)
    result["evidenceSha256"] = _sha256(p)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--evidence")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(plan(), sort_keys=True, separators=(",", ":")))
        return
    if not args.evidence:
        parser.error("--evidence is required unless --plan is used")
    print(json.dumps(validate_file(args.evidence), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
