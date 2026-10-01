from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "frontier" / "peft-distributed-lora-export-78bce7c" / "evaluation-plan.json"
REVISION = "78bce7cb48f800a7ad0d352b68a46302e13e1687"


class PeftDistributedLoraExportPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(PLAN.read_text(encoding="utf-8"))

    def test_exact_source_and_owners(self) -> None:
        self.assertEqual(self.plan["schema"], "szl.forge.frontier-evaluation.v1")
        self.assertEqual(self.plan["frontierIssue"], 110)
        self.assertEqual(self.plan["forgeIssue"], 257)
        self.assertEqual(self.plan["source"]["repository"], "huggingface/peft")
        self.assertEqual(self.plan["source"]["revision"], REVISION)
        self.assertEqual(self.plan["source"]["upstreamPullRequest"], 3251)

    def test_no_authority_is_granted(self) -> None:
        authority = self.plan["authority"]
        self.assertEqual(authority["disposition"], "EVALUATION_HOLD")
        for key in (
            "trainingAuthorized",
            "adapterPublicationAuthorized",
            "modelPublicationAuthorized",
            "providerWritesAuthorized",
            "automaticPromotion",
            "paidComputeAuthorizedByThisPlan",
            "weightRehostingForInventory",
        ):
            self.assertIs(authority[key], False, key)

    def test_warning_boundary_is_honest(self) -> None:
        boundary = self.plan["truthBoundary"]
        self.assertEqual(boundary["upstreamBehavior"], "warning_only_for_inferred_state_dict")
        self.assertEqual(boundary["explicitStateDictHeuristic"], "NOT_APPLIED")
        self.assertIs(boundary["upstreamWarningIsPublicationDenial"], False)
        self.assertIs(boundary["szlFailClosedPublisherPolicyAdmitted"], False)

    def test_reproducibility_pins_cover_distributed_export(self) -> None:
        self.assertEqual(
            set(self.plan["requiredPins"]),
            {
                "peft_git_revision",
                "torch_version",
                "deepspeed_or_fsdp_version",
                "safetensors_version",
                "base_model_revision",
                "adapter_config_digest",
                "fixture_digest",
                "hardware_inventory",
                "launch_command",
            },
        )

    def test_stages_cover_positive_negative_bypass_and_receipt_checks(self) -> None:
        stages = {stage["id"]: stage for stage in self.plan["stages"]}
        self.assertEqual(set(stages), {"P0", "P1", "P2", "P3", "P4", "P5"})
        text = "\n".join(rule for stage in stages.values() for rule in stage["acceptance"])
        for phrase in (
            "serialized lora_A and lora_B tensor is at least 2-D and non-empty",
            "emit PeftWarning",
            "warning does not itself prevent adapter files from being written",
            "explicitly supplied state_dict bypasses the heuristic",
            "DoRA magnitude vectors",
            "AdaLoRA lora_E",
            "immutable-byte readable",
            "successfully reloadable",
            "separate code review and tests",
            "rollback",
        ):
            self.assertIn(phrase, text)

    def test_claim_boundary_rejects_false_safety_claim(self) -> None:
        boundary = self.plan["claimBoundary"]
        self.assertIn("upstream PEFT warning does not certify adapter completeness", boundary)
        self.assertIn("deny malformed writes", boundary)
        self.assertIn("no SZL fail-closed publication policy is admitted", boundary)


if __name__ == "__main__":
    unittest.main()
