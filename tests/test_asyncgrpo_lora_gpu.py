"""GPU backend stays UNAVAILABLE without CUDA. No invented throughput."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from frontier.asyncgrpo_lora_contract import EvaluationContractError
from frontier.asyncgrpo_lora_gpu import (
    GPU_FIELDS,
    admit_gpu_measurements,
    gpu_blocker,
    try_gpu_negative_paths,
    try_gpu_sync_paths,
    unavailable_measurements,
    write_tiny_fixture,
)


READY = {
    "cudaAvailable": True,
    "vllmPresent": True,
    "exactTrlRevisionMatched": True,
    "peftPresent": True,
}
FAKE = {
    "adapterOnlyTransferredBytes": 128,
    "mergedTransferredBytes": 4096,
    "adapterOnlySyncPauseSeconds": 0.01,
    "mergedSyncPauseSeconds": 0.2,
    "adapterOnlyGenerationThroughput": 12.0,
    "mergedGenerationThroughput": 9.0,
    "policyVersionCorrect": True,
    "resourceUse": {"vramMiB": 1024},
}


class GpuBackendTests(unittest.TestCase):
    def test_blocker_on_this_host_is_cuda_or_packages(self) -> None:
        from frontier.asyncgrpo_lora_runner import probe_host

        probe = probe_host()
        self.assertIsNotNone(gpu_blocker(probe))
        self.assertEqual(gpu_blocker(READY), None)

    def test_try_gpu_sync_does_not_invent_numbers(self) -> None:
        from frontier.asyncgrpo_lora_runner import probe_host

        with tempfile.TemporaryDirectory() as tmp:
            result = try_gpu_sync_paths(probe_host(), Path(tmp))
        self.assertIs(result["executed"], False)
        self.assertEqual(result["status"], "UNAVAILABLE")
        for field in GPU_FIELDS:
            self.assertEqual(result[field], "UNAVAILABLE")
        self.assertEqual(result["gpuGenerationThroughput"], "UNAVAILABLE")

    def test_ready_probe_without_weights_is_still_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = try_gpu_sync_paths(READY, Path(tmp), execute=True)
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertEqual(result["adapterOnlyGenerationThroughput"], "UNAVAILABLE")
        self.assertIn(
            result["reason"],
            {
                "gpu_model_fixture_weights_absent",
                "vllm_uninstalled",
                "vllm_spawn_failed",
                "vllm_server_exited",
                "vllm_server_unready",
            },
        )

    def test_admit_rejects_fakes_without_hardware(self) -> None:
        cold = {
            "cudaAvailable": False,
            "vllmPresent": False,
            "exactTrlRevisionMatched": False,
            "peftPresent": False,
        }
        with self.assertRaises(EvaluationContractError):
            admit_gpu_measurements(cold, FAKE)

    def test_admit_accepts_complete_ready_measurements(self) -> None:
        admitted = admit_gpu_measurements(READY, FAKE)
        self.assertIs(admitted["executed"], True)
        self.assertEqual(admitted["adapterOnlyTransferredBytes"], 128)
        self.assertEqual(admitted["status"], "MEASURED")

    def test_gpu_negatives_unavailable_without_host(self) -> None:
        from frontier.asyncgrpo_lora_runner import probe_host

        with tempfile.TemporaryDirectory() as tmp:
            result = try_gpu_negative_paths(probe_host(), Path(tmp))
        self.assertTrue(result)
        for item in result.values():
            self.assertEqual(item["status"], "UNAVAILABLE")
            self.assertIs(item["observed"], False)

    def test_tiny_fixture_is_non_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_tiny_fixture(Path(tmp))
            self.assertTrue((path / "config.json").is_file())
            text = (path / "config.json").read_text(encoding="utf-8")
            self.assertIn("szl_fixture", text)
            self.assertFalse(any(path.glob("*.safetensors")))

    def test_unavailable_helper_has_every_gpu_field(self) -> None:
        payload = unavailable_measurements("cuda_unavailable")
        for field in GPU_FIELDS:
            self.assertEqual(payload[field], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
