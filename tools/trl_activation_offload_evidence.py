"""Fail-closed evidence contract for Forge issue #324.

This module does not install TRL, change production dependencies, or infer hardware
success. It validates a sealed result emitted by an exact-source evaluator.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

SHA40 = re.compile(r"^[0-9a-f]{40}$")
STREAM_RACE = "61c04cd306a41c536b44c172d840398d3e3e1a73"
STORAGE_DEDUPE = "8d58043c4c1c4d736b0350f82eebff6269b0ff1b"
EXACT_SOURCES = [STREAM_RACE, STORAGE_DEDUPE]
REQUIRED_CASES = (
    "forward_parity_no_offload_single_stream_streams",
    "gradient_parity_no_offload_single_stream_streams",
    "shared_storage_views_single_stream",
    "shared_storage_views_streams_mode",
    "nonzero_storage_offset_view",
    "noncontiguous_view",
    "nondefault_compute_stream",
    "checkpoint_or_custom_autograd",
    "fsdp2",
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
        "schema": "szl.forge.trl-activation-offload-evidence.v1",
        "issue": "szl-holdings/szl-forge#324",
        "disposition": "HOLD",
        "exactSources": EXACT_SOURCES,
        "requiredCases": list(REQUIRED_CASES),
        "correctnessBeforePerformance": True,
        "fsdp2Rule": "PASS only on real supported hardware; otherwise UNAVAILABLE with environment identity",
        "authority": {
            "productionDependencyPromotion": False,
            "trainingDefaultChange": False,
            "hubPublication": False,
            "automaticPromotion": False,
        },
    }


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != plan()["schema"]:
        raise EvidenceError("schema mismatch")
    sources = payload.get("exactSources")
    if sources != EXACT_SOURCES or any(not SHA40.fullmatch(str(x)) for x in sources or []):
        raise EvidenceError("exact source set mismatch")
    env = payload.get("environment")
    if not isinstance(env, dict):
        raise EvidenceError("missing environment")
    for key in ("python", "torch", "device", "acceleratorRuntime", "driver"):
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
    if normalized["fsdp2"] == "UNAVAILABLE" and not env.get("fsdp2UnavailableReason"):
        raise EvidenceError("FSDP2 UNAVAILABLE requires reason")
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise EvidenceError("metrics must be an object")
    all_correctness_pass = all(v == "PASS" for k, v in normalized.items() if k != "fsdp2")
    return {
        "disposition": "HOLD",
        "allCorrectnessPass": all_correctness_pass,
        "fsdp2": normalized["fsdp2"],
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
