from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "model-kernel-frontier.yml"


class ModelKernelFrontierWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        cls.offline_region = cls.workflow.split(
            "      - name: Compile and run offline contracts", 1
        )[1].split("      - name: Build the stable Kernel runtime image", 1)[0]
        cls.live_region = cls.workflow.split(
            "      - name: Acquire validated private-Hub read credential", 1
        )[1].split("      - name: Upload audit evidence", 1)[0]
        cls.pre_live_region = cls.workflow.split(
            "      - name: Acquire validated private-Hub read credential", 1
        )[0]

    def test_pull_request_lane_is_credentialless_and_network_free_for_model_bindings(
        self,
    ) -> None:
        self.assertIn("  pull_request:\n    branches: [main]", self.workflow)
        self.assertIn("python tools/verify_model_portfolio.py --offline", self.offline_region)
        self.assertNotIn(
            'python tools/publish_model_source_bindings.py \\\n            --source-revision',
            self.offline_region,
        )
        self.assertNotIn("verify_model_portfolio.py \\\n            --live", self.offline_region)
        self.assertNotIn("secrets.", self.pre_live_region)

    def test_private_hub_read_is_bound_to_a_trusted_main_push(self) -> None:
        required_guard = (
            "if: github.event_name == 'push' && github.ref == 'refs/heads/main'"
        )
        self.assertGreaterEqual(self.live_region.count(required_guard), 2)
        self.assertIn(
            "HF_ORG_TOKEN_CANDIDATE: ${{ secrets.HF_ORG_TOKEN }}",
            self.live_region,
        )
        self.assertIn(
            "HF_ORG_TOKEN1_CANDIDATE: ${{ secrets.HF_ORG_TOKEN1 }}",
            self.live_region,
        )
        self.assertIn(
            "--target-repo SZLHOLDINGS/SZL-Forge-1.5B-ReceiptAgent",
            self.live_region,
        )
        self.assertIn(
            'python tools/publish_model_source_bindings.py \\\n            --source-revision "${GITHUB_SHA}"',
            self.live_region,
        )
        self.assertIn("python tools/verify_model_portfolio.py \\\n            --live", self.live_region)

    def test_reports_are_retained_without_making_skipped_live_proofs_look_present(
        self,
    ) -> None:
        self.assertIn("reports/model-portfolio-live.json", self.workflow)
        self.assertIn("reports/model-source-bindings-dry-run.json", self.workflow)
        self.assertIn("reports/hf-private-read-credential.json", self.workflow)
        self.assertIn("if-no-files-found: warn", self.workflow)


if __name__ == "__main__":
    unittest.main(verbosity=2)
