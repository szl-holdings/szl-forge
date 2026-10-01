from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "frontier" / "transformers-fsdp2-expert-parallel-415e6d2" / "evaluation-plan.json"
REVISION = "415e6d2f596ef2bd44fdee4261799200a7fc02bf"
PARENT = "5474a55e920f358d8382f3ecd3377edca979baa1"


class TransformersFsdp2ExpertParallelPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))

    def test_exact_source_and_owner_identity(self) -> None:
        self.assertEqual(self.plan["schema"], "szl.forge.frontier-evaluation.v1")
        self.assertEqual(self.plan["frontierIssue"], 108)
        self.assertEqual(self.plan["forgeIssue"], 255)
        self.assertEqual(self.plan["source"]["repository"], "huggingface/transformers")
        self.assertEqual(self.plan["source"]["revision"], REVISION)
        self.assertEqual(self.plan["source"]["parentRevision"], PARENT)
        self.assertEqual(self.plan["source"]["upstreamPullRequest"], 48516)
        self.assertEqual(self.plan["source"]["sourceClass"], "MAIN_ONLY_WATCH")

    def test_plan_grants_no_production_or_publication_authority(self) -> None:
        authority = self.plan["authority"]
        self.assertEqual(authority["disposition"], "EVALUATION_HOLD")
        for key in (
            "productionTrainingAuthorized",
            "modelQualificationAuthorized",
            "publicationAuthorized",
            "providerWritesAuthorized",
            "automaticPromotion",
            "paidComputeAuthorizedByThisPlan",
            "weightRehostingForInventory",
        ):
            self.assertIs(authority[key], False, key)

    def test_execution_requires_complete_reproducibility_pins(self) -> None:
        required = set(self.plan["requiredPins"])
        expected = {
            "transformers_git_revision",
            "torch_version",
            "accelerate_version",
            "cuda_version",
            "driver_version",
            "nccl_version",
            "model_revision",
            "tokenizer_revision",
            "fixture_or_dataset_digest",
            "hardware_inventory",
            "world_size",
            "fsdp_size",
            "tp_size",
            "seed",
            "launch_command",
        }
        self.assertEqual(required, expected)

    def test_upstream_numbers_cannot_be_mislabeled_as_szl_measurements(self) -> None:
        labels = self.plan["evidenceLabels"]
        self.assertEqual(labels["upstreamNumbers"], "UPSTREAM_REPORTED_NOT_SZL_MEASURED")
        self.assertEqual(labels["staticPreflight"], "PREPARATION_ONLY")
        self.assertEqual(labels["executedEvidence"], "UNAVAILABLE_UNTIL_RUN")

    def test_correctness_and_failure_gates_precede_performance(self) -> None:
        stages = {stage["id"]: stage for stage in self.plan["stages"]}
        self.assertEqual(set(stages), {"S0", "S1", "S2", "S3", "S4", "S5", "S6"})
        self.assertEqual(stages["S6"]["blockedUntil"], ["S1", "S2", "S3", "S4", "S5"])
        acceptance = "\n".join(
            rule for stage in self.plan["stages"] for rule in stage["acceptance"]
        )
        self.assertIn("NaN or Inf", acceptance)
        self.assertIn("sentinel-tail", acceptance)
        self.assertIn("router-score gradient all-reduce", acceptance)
        self.assertIn("per-mesh total-gradient-norm", acceptance)
        self.assertIn("save_pretrained", acceptance)
        self.assertIn("optimizer checkpoint resumability", acceptance)
        self.assertIn("rank loss, collective timeout, malformed mesh, OOM and cancellation", acceptance)
        self.assertIn("Measure only after all correctness and failure gates pass", acceptance)

    def test_mesh_contract_is_explicit_and_pipeline_composition_remains_denied(self) -> None:
        stage = next(stage for stage in self.plan["stages"] if stage["id"] == "S3")
        text = "\n".join(stage["acceptance"])
        self.assertIn("two-dimensional fsdp/tp device mesh", text)
        self.assertIn("world_size equal to fsdp_size times tp_size", text)
        self.assertIn("Reject pipeline-parallel composition", text)

    def test_claim_boundary_refuses_execution_claims(self) -> None:
        boundary = self.plan["claimBoundary"]
        for phrase in (
            "defines evidence requirements only",
            "No FSDP2 plus Expert Parallel training run",
            "model quality result",
            "performance result",
            "production readiness claim",
            "publication",
            "promotion",
        ):
            self.assertIn(phrase, boundary)


if __name__ == "__main__":
    unittest.main()
