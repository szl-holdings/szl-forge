# Copyright 2026 SZL Holdings. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "frontier" / "transformers-peft-020-load-floor" / "evaluation-plan.json"
TRANSFORMERS_REVISION = "75c3583e042c0765874784306fb9ace2137925d1"
PEFT_SUCCESSOR_REVISION = "e99fdd275075068d093f8996a59bc0eb865f71a2"


class TransformersPeft020Frontier142Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(PLAN.read_text(encoding="utf-8"))

    def test_exact_source_and_hold_are_immutable_contract(self) -> None:
        self.assertEqual(self.plan["frontierIssue"], 142)
        self.assertEqual(self.plan["forgeIssue"], 316)
        self.assertEqual(self.plan["source"]["repository"], "huggingface/transformers")
        self.assertEqual(self.plan["source"]["revision"], TRANSFORMERS_REVISION)
        self.assertEqual(self.plan["authority"]["disposition"], "EVALUATION_HOLD")
        self.assertFalse(self.plan["authority"]["automaticPromotion"])
        self.assertFalse(self.plan["authority"]["trainingAuthorized"])
        self.assertFalse(self.plan["authority"]["providerWritesAuthorized"])

    def test_peft_020_is_stable_floor_and_dtype_fix_stays_separate_watch(self) -> None:
        self.assertEqual(self.plan["peft"]["stableMinimum"], "0.20.0")
        self.assertEqual(self.plan["peft"]["stableTag"], "v0.20.0")
        self.assertEqual(self.plan["peft"]["unreleasedSuccessorRevision"], PEFT_SUCCESSOR_REVISION)
        self.assertEqual(self.plan["peft"]["unreleasedSuccessorQualification"], "SEPARATE_WATCH")

    def test_incumbent_0191_pins_are_not_relabelled_as_current_breakage(self) -> None:
        observations = self.plan["currentEstatePins"]
        self.assertEqual(len(observations), 3)
        self.assertTrue(all(item["peft"] == "0.19.1" for item in observations))
        self.assertTrue(
            all(
                item["classification"] == "INCUMBENT_PIN_NOT_DECLARED_BROKEN_BY_THIS_PLAN"
                for item in observations
            )
        )

    def test_negative_version_floor_and_existing_integrity_lane_are_required(self) -> None:
        stages = {stage["id"]: stage for stage in self.plan["stages"]}
        p0 = " ".join(stages["P0"]["acceptance"])
        p3 = " ".join(stages["P3"]["acceptance"])
        self.assertIn("requiring PEFT >=0.20.0", p0)
        self.assertIn("pins 0.19.1", p0)
        self.assertIn("Frontier #110 / Forge #257/#258", p3)
        self.assertIn("UNKNOWN", p3)

    def test_no_projection_or_promotion_from_partial_evidence(self) -> None:
        p5 = next(stage for stage in self.plan["stages"] if stage["id"] == "P5")
        acceptance = " ".join(p5["acceptance"])
        self.assertIn("No Hugging Face artifact/runtime projection", acceptance)
        self.assertIn("a-11-oy.com remains unchanged", acceptance)
        self.assertFalse(self.plan["authority"]["adapterPublicationAuthorized"])
        self.assertFalse(self.plan["authority"]["modelPublicationAuthorized"])
        self.assertFalse(self.plan["authority"]["weightRehostingForInventory"])


if __name__ == "__main__":
    unittest.main()
