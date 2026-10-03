"""Credentialless, exact-source CI ownership for qualification admission controls."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/model-qualification-admission.yml"


class QualificationWorkflowTests(unittest.TestCase):
    def test_both_hosts_and_supported_python_versions_are_blocking(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn('os: [ubuntu-latest, windows-latest]', text)
        self.assertIn('python: ["3.11", "3.12"]', text)
        self.assertIn("fail-fast: false", text)
        self.assertIn("timeout-minutes: 10", text)
        self.assertNotIn("continue-on-error", text)

    def test_runs_exact_source_without_credentials_or_model_dependencies(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("permissions:\n  contents: read", text)
        self.assertIn("persist-credentials: false", text)
        self.assertIn("ref: ${{ github.event.pull_request.head.sha || github.sha }}", text)
        self.assertNotIn("secrets.", text)
        self.assertNotIn("pip install", text)
        self.assertNotIn("workflow_dispatch", text)

    def test_normal_and_optimized_python_run_the_same_admission_tests(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        folded = " ".join(text.split())
        for flag in ("-B", "-B -O"):
            self.assertIn(
                f"python {flag} -m unittest -v chaski/test_adapter_guard.py "
                "tests/frontier/test_public_smoke_boundary.py "
                "tests/test_model_qualification_workflow.py",
                folded,
            )


if __name__ == "__main__":
    unittest.main()
