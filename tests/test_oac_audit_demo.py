"""Offline contract tests for the public synthetic OAC demo wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "demos" / "oac-audit-demo" / "run.py"
SPEC = importlib.util.spec_from_file_location("oac_audit_demo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
demo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)


def good_report() -> dict:
    return {
        "schema": "szl.oac-health-space-live-verification/v1",
        "status": "PASS",
        "complete": True,
        "completed_at": "2026-10-03T00:00:00+00:00",
        "expected_application_source_sha": demo.SOURCE_SHA,
        "observation_scope": "PUBLIC_SYNTHETIC_PROBE",
        "origin": demo.PUBLIC_ORIGIN,
        "artifact_source": {
            "repository": "szl-holdings/szl-forge",
            "revision": demo.ARTIFACT_SOURCE_SHA,
            "directory": "clinical-gateway/huggingface/model/oac-system-health-v1",
        },
        "hub_model": {"repo_id": demo.MODEL_ID, "revision": demo.MODEL_SHA},
        "hub_dataset": {"repo_id": demo.DATASET_ID, "revision": demo.DATASET_SHA},
        "artifacts": {"model.json": demo.MODEL_BYTES_SHA256},
        "release_verification": {
            "status": "PASS",
            "complete": True,
            "scope": "IMMUTABLE_GIT_ARTIFACT_PARITY",
            "source_parity_count": 3,
            "provider_parity_claimed": False,
        },
        "checks": [
            {
                "name": name,
                "method": "GET" if i < 4 else "POST",
                "status": "PASS",
                "http_status": 400 if i == 6 else 200,
            }
            for i, name in enumerate(demo.EXPECTED_CHECKS)
        ],
        "training_performed": False,
        "clinical_use_authorized": False,
        "production_promotion_allowed": False,
        "receipt_minted": False,
    }


class AuditDemoTests(unittest.TestCase):
    def test_closed_public_projection(self) -> None:
        summary = demo.summarize(good_report())
        self.assertEqual(summary["status"], "SYNTHETIC_WITNESS_PASS")
        self.assertEqual(len(summary["live_checks"]), 7)
        self.assertFalse(summary["clinical_use_authorized"])
        self.assertNotIn(str(ROOT), str(summary))

    def test_source_drift_fails_closed(self) -> None:
        report = good_report()
        report["expected_application_source_sha"] = "0" * 40
        with self.assertRaisesRegex(demo.DemoError, "APPLICATION_SOURCE_DRIFT"):
            demo.summarize(report)

    def test_public_origin_drift_fails_closed(self) -> None:
        report = good_report()
        report["origin"] = "https://example.invalid"
        with self.assertRaisesRegex(demo.DemoError, "PUBLIC_ORIGIN_CHANGED"):
            demo.summarize(report)

    def test_missing_refusal_fails_closed(self) -> None:
        report = good_report()
        report["checks"].pop()
        with self.assertRaisesRegex(demo.DemoError, "LIVE_CHECK_COVERAGE_CHANGED"):
            demo.summarize(report)

    def test_clinical_authority_fails_closed(self) -> None:
        report = good_report()
        report["clinical_use_authorized"] = True
        with self.assertRaisesRegex(demo.DemoError, "AUTHORITY_BOUNDARY_CHANGED"):
            demo.summarize(report)

    def test_provider_parity_is_not_inferred(self) -> None:
        report = good_report()
        report["release_verification"]["provider_parity_claimed"] = True
        with self.assertRaisesRegex(demo.DemoError, "IMMUTABLE_SOURCE_PARITY_MISSING"):
            demo.summarize(report)

    def test_model_digest_drift_fails_closed(self) -> None:
        report = good_report()
        report["artifacts"]["model.json"] = "0" * 64
        with self.assertRaisesRegex(demo.DemoError, "MODEL_DIGEST_DRIFT"):
            demo.summarize(report)

    def test_method_drift_fails_closed(self) -> None:
        report = good_report()
        report["checks"][6]["method"] = "GET"
        with self.assertRaisesRegex(demo.DemoError, "LIVE_METHOD_CONTRACT_CHANGED"):
            demo.summarize(report)


if __name__ == "__main__":
    unittest.main()
