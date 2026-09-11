from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from frontier.asyncgrpo_lora_contract import (
    EvaluationContractError,
    TRL_REVISION,
    evaluate_plan,
    required_max_loras,
    serving_cache_path,
)


class AsyncGrpoLoraContractTests(unittest.TestCase):
    def test_exact_source_is_fixed(self) -> None:
        self.assertEqual(TRL_REVISION, "f540773f5250c816e992ae3d35a41142ef3625c0")

    def test_staleness_capacity_reserves_load_before_evict_slot(self) -> None:
        self.assertEqual(required_max_loras(0), 2)
        self.assertEqual(required_max_loras(4), 6)
        with self.assertRaises(EvaluationContractError):
            required_max_loras(-1)

    def test_valid_adapter_plan_is_still_evaluation_only(self) -> None:
        result = evaluate_plan(
            lora_rank=32,
            max_lora_rank=32,
            max_loras=6,
            max_staleness=4,
            lora_server_enabled=True,
            shared_storage_verified=True,
            checkpoints_enabled=True,
            adapter_servable=True,
        )
        self.assertEqual(result["disposition"], "EVALUATION")
        self.assertEqual(result["syncModeCandidate"], "adapter-only")
        self.assertFalse(result["productionTrainingAuthorized"])
        self.assertFalse(result["productionServingAuthorized"])
        self.assertFalse(result["providerCredentialUseAuthorized"])
        self.assertFalse(result["hubPublicationAuthorized"])
        self.assertFalse(result["automaticPromotionAuthorized"])

    def test_missing_runtime_preconditions_fail_closed(self) -> None:
        result = evaluate_plan(
            lora_rank=4,
            max_lora_rank=8,
            max_loras=2,
            max_staleness=4,
            lora_server_enabled=False,
            shared_storage_verified=False,
            checkpoints_enabled=False,
            adapter_servable=False,
        )
        self.assertEqual(result["disposition"], "HOLD")
        self.assertEqual(result["syncModeCandidate"], "merged-fallback-or-unavailable")
        self.assertEqual(
            set(result["reasons"]),
            {
                "vllm_lora_not_enabled",
                "insufficient_version_capacity",
                "shared_storage_unverified",
                "durable_checkpoint_missing",
                "adapter_not_vllm_servable",
            },
        )

    def test_rank_capacity_is_validated(self) -> None:
        with self.assertRaises(EvaluationContractError):
            evaluate_plan(
                lora_rank=33,
                max_lora_rank=32,
                max_loras=6,
                max_staleness=4,
                lora_server_enabled=True,
                shared_storage_verified=True,
                checkpoints_enabled=True,
                adapter_servable=True,
            )
        with self.assertRaises(EvaluationContractError):
            evaluate_plan(
                lora_rank=4,
                max_lora_rank=7,
                max_loras=6,
                max_staleness=4,
                lora_server_enabled=True,
                shared_storage_verified=True,
                checkpoints_enabled=True,
                adapter_servable=True,
            )

    def test_serving_cache_cannot_escape_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            root.mkdir()
            inside = root / ".vllm_lora" / "policy-v3"
            self.assertEqual(serving_cache_path(root, inside), inside.resolve())
            with self.assertRaises(EvaluationContractError):
                serving_cache_path(root, root / ".." / "outside")


if __name__ == "__main__":
    unittest.main()
