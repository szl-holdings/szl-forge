#!/usr/bin/env python3
"""Public, read-only witness for the synthetic OAC model-audit demonstration.

Run from a Forge checkout with ``python -I -B demos/oac-audit-demo/run.py``.
The canonical OAC verifier supplies the actual source and live probes; this
wrapper emits a small public-safe decision, not a new release receipt.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SPACE = ROOT / "spaces" / "oac-system-health-lab"
VERIFIER = SPACE / "verify_live.py"
SOURCE_SHA = "234a2dfb4c511318febfd20ca2f695fa9cb2ea8c"
ARTIFACT_SOURCE_SHA = "48cb7630dc5c6982cf14755da72d3d07cd45740c"
MODEL_ID = "SZLHOLDINGS/oac-system-health-v1"
MODEL_SHA = "dd7d109813abcd90c5250106dc2eabd2804e7ac3"
DATASET_ID = "SZLHOLDINGS/oac-clinical-transport-observability-synthetic"
DATASET_SHA = "f7ab6170bf78138b187b8cb707d374a25ad85375"
MODEL_BYTES_SHA256 = "f111b7fc65db561c80763b25e538366a19bed18526ca881915777487935d9557"
PUBLIC_ORIGIN = "https://szlholdings-oac-system-health-lab.hf.space"
EXPECTED_CHECKS = (
    "/healthz",
    "/readyz",
    "/api/build-info",
    "/api/v1/identity",
    "authored_healthy_synthetic",
    "authored_degraded_synthetic",
    "clinical_like_extra_field_refused_without_echo",
)


class DemoError(ValueError):
    """A public demo claim lacks its required evidence."""


def load_verifier() -> Any:
    """Load only the verifier adjacent to this checked-out Forge source."""
    spec = importlib.util.spec_from_file_location("_szl_oac_demo_live", VERIFIER)
    if spec is None or spec.loader is None:
        raise DemoError("VERIFIER_SOURCE_MISSING")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    """Project a closed, public-safe subset of the canonical verifier report."""
    if not isinstance(report, dict) or report.get("status") != "PASS" or report.get("complete") is not True:
        raise DemoError("OAC_LIVE_WITNESS_FAILED")
    if report.get("schema") != "szl.oac-health-space-live-verification/v1":
        raise DemoError("WITNESS_SCHEMA_CHANGED")
    if report.get("expected_application_source_sha") != SOURCE_SHA:
        raise DemoError("APPLICATION_SOURCE_DRIFT")
    if report.get("observation_scope") != "PUBLIC_SYNTHETIC_PROBE":
        raise DemoError("WITNESS_SCOPE_CHANGED")
    if report.get("origin") != PUBLIC_ORIGIN:
        raise DemoError("PUBLIC_ORIGIN_CHANGED")
    if report.get("artifact_source") != {
        "repository": "szl-holdings/szl-forge",
        "revision": ARTIFACT_SOURCE_SHA,
        "directory": "clinical-gateway/huggingface/model/oac-system-health-v1",
    }:
        raise DemoError("ARTIFACT_SOURCE_DRIFT")
    if report.get("hub_model") != {"repo_id": MODEL_ID, "revision": MODEL_SHA}:
        raise DemoError("MODEL_REVISION_DRIFT")
    if report.get("hub_dataset") != {"repo_id": DATASET_ID, "revision": DATASET_SHA}:
        raise DemoError("DATASET_REVISION_DRIFT")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict) or artifacts.get("model.json") != MODEL_BYTES_SHA256:
        raise DemoError("MODEL_DIGEST_DRIFT")
    source_check = report.get("release_verification")
    if not isinstance(source_check, dict) or any(
        source_check.get(key) != value
        for key, value in {
            "status": "PASS",
            "complete": True,
            "scope": "IMMUTABLE_GIT_ARTIFACT_PARITY",
            "source_parity_count": 3,
            "provider_parity_claimed": False,
        }.items()
    ):
        raise DemoError("IMMUTABLE_SOURCE_PARITY_MISSING")
    if any(report.get(key) is not False for key in (
        "training_performed",
        "clinical_use_authorized",
        "production_promotion_allowed",
        "receipt_minted",
    )):
        raise DemoError("AUTHORITY_BOUNDARY_CHANGED")
    checks = report.get("checks")
    if not isinstance(checks, list) or tuple(
        check.get("name") if isinstance(check, dict) else None for check in checks
    ) != EXPECTED_CHECKS:
        raise DemoError("LIVE_CHECK_COVERAGE_CHANGED")
    if any(check.get("status") != "PASS" for check in checks):
        raise DemoError("LIVE_CHECK_FAILED")
    if any(check.get("http_status") != (400 if index == 6 else 200)
           for index, check in enumerate(checks)):
        raise DemoError("LIVE_HTTP_CONTRACT_CHANGED")
    if any(check.get("method") != ("GET" if index < 4 else "POST")
           for index, check in enumerate(checks)):
        raise DemoError("LIVE_METHOD_CONTRACT_CHANGED")

    return {
        "schema": "szl.oac-audit-demo/v1",
        "status": "SYNTHETIC_WITNESS_PASS",
        "observed_at_utc": report.get("completed_at"),
        "application_source_sha": SOURCE_SHA,
        "artifact_source_sha": ARTIFACT_SOURCE_SHA,
        "hub_model": {"repo_id": MODEL_ID, "revision": MODEL_SHA},
        "hub_dataset": {"repo_id": DATASET_ID, "revision": DATASET_SHA},
        "immutable_git_artifact_parity_count": 3,
        "live_checks": [
            {"name": check["name"], "http_status": check["http_status"]}
            for check in checks
        ],
        "training_performed": False,
        "clinical_use_authorized": False,
        "production_promotion_allowed": False,
        "receipt_minted": False,
        "scope": "invented numerical telemetry; no patient, device, LIS or result path",
        "limits": "not an all-model audit, clinical validation, or provider byte attestation",
    }


def main() -> int:
    try:
        verifier = load_verifier()
        if verifier.DEFAULT_ORIGIN != PUBLIC_ORIGIN:
            raise DemoError("PUBLIC_ORIGIN_CHANGED")
        report = verifier.verify_live(
            origin=PUBLIC_ORIGIN,
            expected_source_sha=SOURCE_SHA,
            space_root=SPACE,
            repository_root=ROOT,
        )
        result = summarize(report)
    except DemoError as exc:
        result = {"schema": "szl.oac-audit-demo/v1", "status": "FAIL", "error": str(exc)}
    except Exception:
        result = {
            "schema": "szl.oac-audit-demo/v1",
            "status": "FAIL",
            "error": "VERIFIER_EXECUTION_FAILED",
        }
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "SYNTHETIC_WITNESS_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
