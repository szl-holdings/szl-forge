"""Host-probe regressions for AsyncGRPO LoRA / vLLM. No GPU execution."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from frontier.asyncgrpo_lora_contract import DENIED_AUTHORITY, TRL_REVISION
from frontier.asyncgrpo_lora_runner import (
    ALLOWED_DISPOSITIONS,
    REQUIRED_EVIDENCE,
    EvaluationRunnerError,
    admit_measurements,
    evaluate_host,
    hardware_ready,
    probe_host,
)

MANIFEST = Path(__file__).resolve().parents[1] / "frontier" / "asyncgrpo_lora_evaluation.json"
FAKE_MEASUREMENTS = {
    "adapterOnlyTransferredBytes": 128,
    "mergedTransferredBytes": 4096,
    "adapterOnlySyncPauseSeconds": 0.01,
    "mergedSyncPauseSeconds": 0.2,
    "adapterOnlyGenerationThroughput": 12.0,
    "mergedGenerationThroughput": 9.0,
    "policyVersionCorrect": True,
    "resourceUse": {"vramMiB": 1024},
}


class AsyncGrpoLoraRunnerTests(unittest.TestCase):
    def test_default_host_is_unavailable_not_pass(self) -> None:
        result = evaluate_host()
        self.assertEqual(result["disposition"], "UNAVAILABLE")
        self.assertNotIn("PASS", ALLOWED_DISPOSITIONS)
        self.assertNotIn(result["disposition"], {"PASS", "OK"})
        self.assertEqual(result["sourceRevision"], TRL_REVISION)
        self.assertTrue(result["evaluationOnly"])
        self.assertFalse(result["productionEligible"])
        for key in DENIED_AUTHORITY:
            self.assertIs(result[key], False)
        self.assertEqual(result["measured"]["adapterOnlyTransferredBytes"], "UNAVAILABLE")
        self.assertEqual(result["measured"]["mergedGenerationThroughput"], "UNAVAILABLE")
        self.assertIn("measured_comparison_absent", result["reasons"])
        self.assertIn("adapter_only_sync_api_unverified_at_pin", result["reasons"])
        self.assertEqual(result["satisfiedEvidence"], [])
        self.assertEqual(set(result["requiredEvidence"]), set(REQUIRED_EVIDENCE))
        self.assertEqual(set(result["missingEvidence"]), set(REQUIRED_EVIDENCE))

    def test_injected_measurements_are_refused_without_hardware(self) -> None:
        with self.assertRaises(EvaluationRunnerError):
            evaluate_host(measurements=FAKE_MEASUREMENTS)
        probe = probe_host()
        self.assertFalse(hardware_ready(probe))
        with self.assertRaises(EvaluationRunnerError):
            admit_measurements(probe, FAKE_MEASUREMENTS)

    def test_boolean_and_string_measurements_are_rejected(self) -> None:
        ready = {
            "cudaAvailable": True,
            "vllmPresent": True,
            "exactTrlRevisionMatched": True,
            "peftPresent": True,
        }
        bad = dict(FAKE_MEASUREMENTS)
        bad["adapterOnlyTransferredBytes"] = True
        with self.assertRaises(EvaluationRunnerError):
            admit_measurements(ready, bad)
        bad = dict(FAKE_MEASUREMENTS)
        bad["mergedSyncPauseSeconds"] = "0.2"
        with self.assertRaises(EvaluationRunnerError):
            admit_measurements(ready, bad)

    def test_plan_hold_reasons_are_preserved(self) -> None:
        result = evaluate_host(
            plan={
                "lora_rank": 32,
                "max_lora_rank": 32,
                "max_loras": 1,
                "max_staleness": 4,
                "lora_server_enabled": False,
                "shared_storage_verified": False,
                "checkpoints_enabled": False,
                "adapter_servable": False,
            }
        )
        self.assertEqual(result["plan"]["disposition"], "HOLD")
        self.assertIn("vllm_lora_not_enabled", result["reasons"])
        self.assertIn("insufficient_version_capacity", result["reasons"])
        self.assertEqual(result["disposition"], "UNAVAILABLE")

    def test_malformed_plan_still_fails_closed(self) -> None:
        with self.assertRaises(Exception):
            evaluate_host(plan={"lora_server_enabled": "false"})

    def test_manifest_stays_unavailable_and_denies_authority(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(payload["disposition"], "UNAVAILABLE")
        self.assertEqual(payload["upstreamRevision"], TRL_REVISION)
        self.assertFalse(payload["productionEligible"])
        self.assertFalse(payload["automaticPromotionAuthorized"])
        self.assertFalse(payload["syncContract"]["adapterOnlyApiVerifiedAtPin"])
        self.assertTrue(payload["evaluationOnly"])
        for value in payload["authority"].values():
            self.assertIs(value, False)
        self.assertTrue(set(REQUIRED_EVIDENCE).issubset(set(payload["requiredEvidence"])))

    def test_probe_does_not_assume_adapter_only_api(self) -> None:
        probe = probe_host()
        self.assertIs(probe["adapterOnlySyncApiAssumed"], False)


if __name__ == "__main__":
    unittest.main()
