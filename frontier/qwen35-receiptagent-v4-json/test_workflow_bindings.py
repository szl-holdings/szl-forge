# SPDX-License-Identifier: Apache-2.0
"""Blocking source-only CI ownership, no model launch or publishing permission."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import sys
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NEMO_REVISION = "f7b33e1b1fe8f3b9b729a27567bcba375e2e0d6e"


class WorkflowBindingTests(unittest.TestCase):
    def cpu_job(self):
        workflow = (ROOT / ".github/workflows/model-kernel-frontier.yml").read_text(encoding="utf-8")
        self.assertEqual(workflow.count("  native-cpu-contracts:\n"), 1)
        return workflow.split("  native-cpu-contracts:\n", 1)[1].split("\n  verify:", 1)[0]

    def cpu_script(self):
        return textwrap.dedent(self.cpu_job().split("python -I -B <<'PY'\n", 1)[1].split("\n          PY", 1)[0])

    def test_native_cpu_job_is_bounded_read_only_and_separate_from_admission(self):
        job = self.cpu_job()
        for text in (
            "runs-on: ubuntu-latest", "timeout-minutes: 10", "contents: read",
            "persist-credentials: false", 'python-version: "3.12"',
            "step-security/harden-runner@dc2769abcc0fcfbadfc1df54081c1608579ef6b2",
            "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
            "actions/setup-python@6d4fd22b20465ccc0f7354476588e8df7c9ca0e0",
            '"torch==2.10.0" --index-url https://download.pytorch.org/whl/cpu',
            '"jsonschema==4.26.0"', "set -euo pipefail", "python -I -B <<'PY'",
        ):
            self.assertIn(text, job)
        for text in ("continue-on-error", "if:", "secrets.", "||", "set +", "exit 0",
                     "self-hosted", "train_candidate.py", "native_supervisor.py", "heldout/", "hf "):
            self.assertNotIn(text, job)
        script = self.cpu_script()
        self.assertIn('runpy.run_path("frontier/qwen35-receiptagent-v4-json/test_native_worker.py")', script)
        self.assertIn('namespace["CPUAutogradTests"]', script)
        self.assertNotIn('__main__', script)

    def run_cpu_script(self, *, count=8, version="2.10.0+cpu", cuda=None, **changes):
        outcome = dict(testsRun=8, failures=[], errors=[], skipped=[], expectedFailures=[],
                       unexpectedSuccesses=[], wasSuccessful=lambda: True)
        outcome.update(changes)
        suite = SimpleNamespace(countTestCases=lambda: count)
        torch = SimpleNamespace(__version__=version, version=SimpleNamespace(cuda=cuda))
        with redirect_stdout(io.StringIO()), patch.dict(sys.modules, {"torch": torch}), \
                patch("runpy.run_path", return_value={"CPUAutogradTests": object}), \
                patch.object(unittest.defaultTestLoader, "loadTestsFromTestCase", return_value=suite), \
                patch.object(unittest, "TextTestRunner") as runner:
            runner.return_value.run.return_value = SimpleNamespace(**outcome)
            exec(compile(self.cpu_script(), "<native-cpu-ci>", "exec"), {})
            runner.return_value.run.assert_called_once_with(suite)

    def test_native_cpu_ci_accepts_only_eight_successful_unskipped_tests(self):
        self.run_cpu_script()

    def test_required_frontier_check_fails_when_cpu_job_does_not_succeed(self):
        workflow = (ROOT / ".github/workflows/model-kernel-frontier.yml").read_text(encoding="utf-8")
        required = workflow.split("\n  verify:\n", 1)[1].split("      - name: Harden runner", 1)[0]
        self.assertEqual([line.strip() for line in required.splitlines() if line.strip()], [
            "name: Receipts, source contracts, and portfolio",
            "runs-on: ubuntu-latest",
            "timeout-minutes: 20",
            "needs: native-cpu-contracts",
            "if: ${{ always() }}",
            "steps:",
            "- name: Require successful synthetic CPU contracts",
            "env:",
            "NATIVE_CPU_RESULT: ${{ needs.native-cpu-contracts.result }}",
            'run: test "$NATIVE_CPU_RESULT" = success',
        ])

    def test_native_cpu_ci_rejects_incomplete_or_non_success_results(self):
        cases = [{"count": 0}, {"count": 9}, {"testsRun": 0}, {"testsRun": 7},
                 {"version": "2.9.0+cpu"}, {"cuda": "12.8"},
                 {"wasSuccessful": lambda: False}]
        cases.extend({name: [("fixture", "not a pass")]} for name in
                     ("failures", "errors", "skipped", "expectedFailures", "unexpectedSuccesses"))
        for case in cases:
            with self.subTest(case=case), self.assertRaises(SystemExit):
                self.run_cpu_script(**case)

    def test_nemo_lane_triggers_and_pins_reviewed_kernel(self):
        workflow = (ROOT / ".github/workflows/nemo-doctrine-gate.yml").read_text(encoding="utf-8")
        self.assertEqual(workflow.count('- "frontier/qwen35-receiptagent-v4-json/**"'), 2)
        self.assertIn("szl-nemo.git@f44b468a60c97978897bc610cf4729fc16b271af", workflow)
        region = workflow.split("      - name: Gate ReceiptAgent v4 training references", 1)[1].split("      - name:", 1)[0]
        for text in ("set -euo pipefail", "jsonschema==4.26.0", "probe_nemo_binding.py", "curriculum_admission.py", "--check-conformance",
                     "--train frontier/qwen35-receiptagent-v4-json/train.jsonl", "--manifest",
                     "--nemo-source-root .nemo-v4-source"):
            self.assertIn(text, region)
        for text in ("dev.jsonl", "heldout/test.jsonl", "continue-on-error", "secrets.", "train_candidate.py",
                     "PYTHONPATH", "szl-nemo @", "|| true"):
            self.assertNotIn(text, region)

    def test_v4_report_redirection_preserves_blocking_gate_commands(self):
        workflow = (ROOT / ".github/workflows/nemo-doctrine-gate.yml").read_text(encoding="utf-8")
        region = workflow.split("      - name: Gate ReceiptAgent v4 training references", 1)[1].split("      - name:", 1)[0]
        script = region.split("        run: |\n", 1)[1]
        lines = [line.strip() for line in script.splitlines() if line.strip()]
        self.assertEqual(lines, [
            "set -euo pipefail",
            'python -m pip install --disable-pip-version-check "jsonschema==4.26.0"',
            'mkdir -p "$RUNNER_TEMP/v4-source-conformance"',
            "python -B frontier/qwen35-receiptagent-v4-json/probe_nemo_binding.py \\",
            "--nemo-source-root .nemo-v4-source \\",
            '> "$RUNNER_TEMP/v4-source-conformance/numeric-probe.json"',
            "python -B frontier/qwen35-receiptagent-v4-json/curriculum_admission.py \\",
            "--check-conformance \\",
            "--train frontier/qwen35-receiptagent-v4-json/train.jsonl \\",
            "--manifest frontier/qwen35-receiptagent-v4-json/curriculum-manifest.json \\",
            "--nemo-source-root .nemo-v4-source \\",
            '> "$RUNNER_TEMP/v4-source-conformance/curriculum-conformance.json"',
        ])
        for text in ("continue-on-error", "||", "set +", "exit 0", "if:"):
            self.assertNotIn(text, region)

    def test_v4_upload_retains_only_two_reports_and_never_changes_gate_authority(self):
        workflow = (ROOT / ".github/workflows/nemo-doctrine-gate.yml").read_text(encoding="utf-8")
        marker = "      - name: Retain v4 source-conformance reports (not training authority)"
        self.assertEqual(workflow.count(marker), 1)
        region = workflow.split(marker, 1)[1].split("      - name:", 1)[0]
        self.assertEqual([line.strip() for line in region.splitlines() if line.strip()], [
            "if: always()",
            "uses: actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9 # v7.0.2",
            "with:",
            "name: v4-source-conformance-${{ github.run_id }}",
            "path: |",
            "${{ runner.temp }}/v4-source-conformance/numeric-probe.json",
            "${{ runner.temp }}/v4-source-conformance/curriculum-conformance.json",
            "if-no-files-found: error",
            "retention-days: 30",
        ])
        for text in ("continue-on-error", "train.jsonl", "dev.jsonl", "heldout", "secrets.",
                     "github.workspace", "include-hidden-files", "*"):
            self.assertNotIn(text, region)
        self.assertLess(workflow.index("      - name: Gate ReceiptAgent v4 training references"),
                        workflow.index(marker))
        readme = (HERE / "README.md").read_text(encoding="utf-8")
        for text in ("numeric-probe.json", "curriculum-conformance.json", "30 days",
                     "no curriculum", "not independent witnessing"):
            self.assertIn(text, readme)

    def test_v4_uses_separate_immutable_source_checkout(self):
        workflow = (ROOT / ".github/workflows/nemo-doctrine-gate.yml").read_text(encoding="utf-8")
        marker = "      - name: Check out ReceiptAgent v4 doctrine source (pinned commit)"
        self.assertEqual(workflow.count(marker), 1)
        region = workflow.split(marker, 1)[1].split("      - name:", 1)[0]
        for line in (
            "        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1",
            "          repository: szl-holdings/szl-nemo",
            "          ref: " + NEMO_REVISION,
            "          path: .nemo-v4-source",
            "          persist-credentials: false",
            "          sparse-checkout: szl_nemo",
        ):
            self.assertIn(line, region.splitlines())
        for text in ("continue-on-error", "secrets.", "token:", "|| true"):
            self.assertNotIn(text, region)
        self.assertLess(workflow.index(marker), workflow.index("      - name: Gate ReceiptAgent v4 training references"))

    def test_v4_documentation_and_preregistration_bind_reviewed_kernel(self):
        prereg = json.loads((HERE / "preregistration.json").read_text(encoding="utf-8"))
        readme = (HERE / "README.md").read_text(encoding="utf-8")
        self.assertEqual(prereg["gate_sources"]["nemo_kernel_revision"], NEMO_REVISION)
        binding = prereg["gate_sources"]["nemo_source_binding"]
        self.assertEqual(binding["schema"], "szl.receiptagent.v4-nemo-source-binding/v1")
        self.assertEqual(binding["module"], "nemo_source_binding.py")
        self.assertEqual(binding["required_for_conformance_cli_argument"], "--nemo-source-root")
        self.assertEqual(binding["source_modules"], 6)
        self.assertIs(binding["ambient_package_imports"], False)
        self.assertIs(binding["ambient_evidence_core_imports"], False)
        self.assertEqual(binding["canonical_json_provider"], "reviewed_kernel_local_fallback")
        self.assertIn("https://github.com/szl-holdings/szl-nemo/commit/" + NEMO_REVISION, readme)
        self.assertIn("--nemo-source-root .nemo-v4-source", readme)
        self.assertIn("git checkout --detach " + NEMO_REVISION, readme)
        self.assertNotIn("szl-nemo.git@f44b468", readme)

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
