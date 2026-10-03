from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "nemo-doctrine-gate.yml"
V3_TRAIN = "frontier/qwen35-receiptagent-v3/train.jsonl"


class NemoDoctrineWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        cls.pull_request = cls.workflow.split("  pull_request:\n", 1)[1].split(
            "  push:\n", 1
        )[0]
        cls.push = cls.workflow.split("  push:\n", 1)[1].split(
            "\npermissions:", 1
        )[0]

    def test_both_events_cover_v3_and_its_schema_gate(self) -> None:
        for region in (self.pull_request, self.push):
            self.assertIn('"frontier/qwen35-receiptagent-v3/**"', region)
            self.assertIn('"tools/validate_sft_dataset.py"', region)
            self.assertIn('"tools/test_nemo_doctrine_workflow.py"', region)

    def test_v3_training_data_gets_both_blocking_gates(self) -> None:
        marker = "      - name: Gate ReceiptAgent v3 training data (fail closed)"
        self.assertIn(marker, self.workflow)
        region = self.workflow.split(marker, 1)[1].split("      - name:", 1)[0]
        self.assertIn("set -euo pipefail", region)
        self.assertIn(
            f"python tools/validate_sft_dataset.py {V3_TRAIN} --min-examples 180 --json",
            region,
        )
        self.assertIn(
            f"python tools/nemo_doctrine_gate.py {V3_TRAIN} --persona finetuned --json",
            region,
        )
        self.assertNotIn("continue-on-error", region)
        self.assertNotIn("|| true", region)
        self.assertNotIn("dev.jsonl", region)
        self.assertNotIn("test.jsonl", region)

    def test_reviewed_kernel_pin_and_existing_datasets_are_preserved(self) -> None:
        self.assertIn(
            "szl-nemo.git@f44b468a60c97978897bc610cf4729fc16b271af",
            self.workflow,
        )
        for dataset in (
            "szl_dataset.jsonl",
            "receiptagent/train.jsonl",
            "receiptagent/train.refusals.jsonl",
            "receiptagent/adversarial.jsonl",
            "receiptagent/eval.jsonl",
        ):
            self.assertIn(dataset, self.workflow)
        self.assertIn('python tools/nemo_doctrine_gate.py "$ds" --persona finetuned', self.workflow)
        self.assertIn("szl-nemo selftest", self.workflow)
        self.assertNotIn("continue-on-error", self.workflow)

    def test_dedicated_workflow_runs_its_ownership_contract(self) -> None:
        self.assertIn(
            "python -B -m unittest -v tools/test_nemo_doctrine_workflow.py",
            self.workflow,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
