"""Executable AsyncGRPO LoRA / vLLM lane. No GPU numbers are invented."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from frontier.asyncgrpo_lora_contract import DENIED_AUTHORITY, TRL_REVISION
from frontier.asyncgrpo_lora_runner import (
    ALLOWED_DISPOSITIONS,
    REQUIRED_EVIDENCE,
    SIBLING_HOLDS,
    EvaluationRunnerError,
    admit_measurements,
    evaluate_host,
    hardware_ready,
    probe_host,
    write_receipt,
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
        self.assertNotIn(result["disposition"], {"PASS", "OK", "ALL_DONE"})
        self.assertEqual(result["sourceRevision"], TRL_REVISION)
        self.assertTrue(result["evaluationOnly"])
        self.assertFalse(result["productionEligible"])
        self.assertFalse(result["ciGreenClaimed"])
        self.assertFalse(result["allDone"])
        self.assertFalse(result["ato"])
        self.assertFalse(result["live"])
        for key in DENIED_AUTHORITY:
            self.assertIs(result[key], False)
        self.assertEqual(result["measured"]["adapterOnlyTransferredBytes"], "UNAVAILABLE")
        self.assertEqual(result["measured"]["mergedGenerationThroughput"], "UNAVAILABLE")
        self.assertEqual(result["gpuStatus"], "UNAVAILABLE")
        self.assertEqual(result["protocolStatus"], "MEASURED")
        self.assertEqual(result["plan"]["disposition"], "EVALUATION")
        self.assertIn("measured_comparison_absent", result["reasons"])
        self.assertIn("cuda_unavailable", result["reasons"])
        self.assertIn("gpu_generation_throughput_unavailable", result["reasons"])
        self.assertNotIn("adapter_only_sync_api_unverified_at_pin", result["reasons"])
        self.assertEqual(set(result["requiredEvidence"]), set(REQUIRED_EVIDENCE))
        self.assertIn("generation-throughput", result["missingEvidence"])
        self.assertIn("job-local-env-not-production-runtime", result["satisfiedEvidence"])
        self.assertIn("transferred-sync-bytes", result["satisfiedEvidence"])
        self.assertIn("adapter-only-sync-api-at-pinned-trl-revision", result["satisfiedEvidence"])

    def test_protocol_measures_both_sync_paths_without_upgrading_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = evaluate_host(output_dir=Path(tmp))
        self.assertEqual(result["disposition"], "UNAVAILABLE")
        self.assertEqual(result["gpuStatus"], "UNAVAILABLE")
        measured = result["protocolMeasured"]
        assert isinstance(measured, dict)
        self.assertGreater(measured["mergedTransferredBytes"], measured["adapterOnlyTransferredBytes"])
        self.assertTrue(measured["adapterOnlySmallerThanMerged"])
        self.assertTrue(measured["policyVersionCorrect"])
        self.assertTrue(measured["checkpointPresent"])
        self.assertEqual(measured["gpuGenerationThroughput"], "UNAVAILABLE")
        self.assertIs(measured["gpuSyncExecuted"], False)
        self.assertGreater(measured["adapterOnlyTransferredBytes"], 0)
        self.assertIn("sync-pause", result["satisfiedEvidence"])
        self.assertNotIn("generation-throughput", result["satisfiedEvidence"])

    def test_negative_paths_all_fire(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = evaluate_host(output_dir=Path(tmp))
        observed = result["negativePathsObserved"]
        assert isinstance(observed, dict)
        required = {
            "vllm_without_lora_support",
            "runtime_adapter_update_unavailable",
            "unsupported_or_insufficient_max_rank",
            "insufficient_max_loras",
            "shared_path_unavailable",
            "path_escape_symlink",
            "partially_written_adapter",
            "load_failure",
            "load_before_evict_ordering",
            "server_restart",
            "serving_cache_eviction",
            "checkpoint_missing",
            "hybrid_recurrent_state_logprob_mismatch",
            "serving_cache_ephemeral_not_durable_artifact",
        }
        self.assertTrue(required.issubset(set(observed)))
        failed = [key for key in required if not observed[key]]
        self.assertEqual(failed, [])

    def test_injected_measurements_are_refused_without_hardware(self) -> None:
        with self.assertRaises(EvaluationRunnerError):
            evaluate_host(measurements=FAKE_MEASUREMENTS)
        probe = probe_host()
        self.assertFalse(hardware_ready(probe))
        self.assertFalse(probe["cudaAvailable"])
        self.assertTrue(probe["adapterOnlyApiAtPinnedRevision"])
        self.assertFalse(probe["adapterOnlySyncApiAssumed"])
        self.assertFalse(probe["adapterOnlyHardwareExecuted"])
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
        self.assertEqual(payload["gpuStatus"], "UNAVAILABLE")
        self.assertFalse(payload["productionEligible"])
        self.assertFalse(payload["automaticPromotionAuthorized"])
        self.assertFalse(payload["ciGreenClaimed"])
        self.assertFalse(payload["allDone"])
        self.assertTrue(payload["syncContract"]["adapterOnlyApiVerifiedAtPin"])
        self.assertFalse(payload["syncContract"]["adapterOnlyHardwareExecuted"])
        self.assertFalse(payload["syncContract"]["inventedExactVersions"])
        self.assertTrue(payload["evaluationOnly"])
        for value in payload["authority"].values():
            self.assertIs(value, False)
        self.assertTrue(set(REQUIRED_EVIDENCE).issubset(set(payload["requiredEvidence"])))
        sibling_ids = {item["id"] for item in payload["siblingHolds"]}
        self.assertEqual(sibling_ids, {item["id"] for item in SIBLING_HOLDS})
        for item in payload["siblingHolds"]:
            self.assertEqual(item["disposition"], "HOLD")

    def test_write_receipt_refuses_production_authority_and_live_stamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            receipt = evaluate_host(output_dir=Path(tmp) / "run")
            forged = dict(receipt)
            forged["productionServingAuthorized"] = True
            target = Path(tmp) / "receipt.json"
            with self.assertRaises(EvaluationRunnerError):
                write_receipt(target, forged)
            self.assertFalse(target.exists())
            stamped = dict(receipt)
            stamped["allDone"] = True
            with self.assertRaises(EvaluationRunnerError):
                write_receipt(target, stamped)
            written = write_receipt(target, receipt)
            payload = json.loads(written.read_text(encoding="utf-8"))
            self.assertEqual(payload["disposition"], "UNAVAILABLE")

    def test_required_evidence_is_unique(self) -> None:
        self.assertEqual(len(REQUIRED_EVIDENCE), len(set(REQUIRED_EVIDENCE)))

    def test_gpu_negatives_stay_unavailable_without_host(self) -> None:
        result = evaluate_host()
        gpu_neg = result["gpuNegativePaths"]
        assert isinstance(gpu_neg, dict)
        self.assertTrue(gpu_neg)
        for value in gpu_neg.values():
            self.assertEqual(value["status"], "UNAVAILABLE")
            self.assertIs(value["observed"], False)


if __name__ == "__main__":
    unittest.main()
