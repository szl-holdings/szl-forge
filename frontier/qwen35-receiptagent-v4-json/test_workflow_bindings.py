# SPDX-License-Identifier: Apache-2.0
"""Blocking source-only CI ownership, no model launch or publishing permission."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class WorkflowBindingTests(unittest.TestCase):
    def test_nemo_lane_triggers_and_pins_reviewed_kernel(self):
        workflow = (ROOT / ".github/workflows/nemo-doctrine-gate.yml").read_text(encoding="utf-8")
        self.assertEqual(workflow.count('- "frontier/qwen35-receiptagent-v4-json/**"'), 2)
        self.assertIn("szl-nemo.git@f44b468a60c97978897bc610cf4729fc16b271af", workflow)
        region = workflow.split("      - name: Gate ReceiptAgent v4 training references", 1)[1].split("      - name: Gate contract tests", 1)[0]
        for text in ("set -euo pipefail", "jsonschema==4.26.0", "curriculum_admission.py", "--check-conformance",
                     "--train frontier/qwen35-receiptagent-v4-json/train.jsonl", "--manifest"):
            self.assertIn(text, region)
        for text in ("dev.jsonl", "heldout/test.jsonl", "continue-on-error", "secrets.", "train_candidate.py"):
            self.assertNotIn(text, region)

    def test_frontier_lane_checks_generation_and_discovers_all_software_tests(self):
        workflow = (ROOT / ".github/workflows/model-kernel-frontier.yml").read_text(encoding="utf-8")
        region = workflow.split("      - name: Compile and run offline contracts", 1)[1].split("      - name: Build the stable Kernel runtime image", 1)[0]
        self.assertIn("python -B frontier/qwen35-receiptagent-v4-json/generate_curriculum.py --check", region)
        self.assertIn("python -B frontier/qwen35-receiptagent-v4-json/heldout/generate_heldout.py --check", region)
        for text in ("curriculum_admission.py", "test_generate_curriculum.py", "test_cross_split_integrity.py",
                     "-s frontier/qwen35-receiptagent-v4-json", "-p 'test_*.py'", "set -euo pipefail"):
            self.assertIn(text, region)
        self.assertNotIn("continue-on-error", region)

    def test_preregistration_retains_launch_blockers_and_future_denominators(self):
        candidate = json.loads((HERE / "candidate.json").read_text(encoding="utf-8"))
        prereg = json.loads((HERE / "preregistration.json").read_text(encoding="utf-8"))
        manifest = json.loads((HERE / "curriculum-manifest.json").read_text(encoding="utf-8"))
        for item in (candidate, prereg, manifest):
            for key in ("training_eligible", "publication_eligible", "execution_authority"):
                self.assertIs(item[key], False)
        self.assertEqual(prereg["curriculum_manifest_sha256"], candidate["curriculum"]["manifest_sha256"])
        self.assertEqual(prereg["future_model_gate"]["full_pair_conformance_required"], 180)
        self.assertEqual(prereg["future_model_gate"]["required_per_class"], {"DRAFT": 60, "RECOVERY": 60, "REFUSAL": 60})
        self.assertEqual(prereg["model_training"], "NOT_RUN")
        self.assertEqual(prereg["optimizer_steps"], 0)
        self.assertIsNone(prereg["future_model_gate"]["actual_model_metrics"])
        self.assertEqual(prereg["signature"], "UNAVAILABLE")
        self.assertEqual(prereg["trainer_supervisor_binding"], "UNAVAILABLE")
        self.assertFalse(prereg["scope"]["semantic_intent_detection"])
        self.assertFalse(prereg["scope"]["predecessor_scores_comparable"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
