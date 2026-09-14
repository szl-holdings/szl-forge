# SPDX-License-Identifier: Apache-2.0
"""Offline CPU identity negatives; no Torch import, install, or provider access."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rlt_cpu_contract", ROOT / "tools/rlt_cpu_environment.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def project(*items):
    return ('[project]\ndependencies = ' + json.dumps(list(items)) + '\n').encode()


class RltCpuEnvironmentTests(unittest.TestCase):
    def test_requirement_follows_single_project_pin(self):
        for version in ("2.10.0", "2.13.0", "3.0.1"):
            with self.subTest(version=version):
                self.assertEqual(M.declared_cpu_requirement(project("numpy==2.3.5", "torch==" + version)), "torch==" + version + "+cpu")

    def test_cpu_suffix_is_not_duplicated(self):
        self.assertEqual(M.declared_cpu_requirement(project("torch==2.13.0+cpu")), "torch==2.13.0+cpu")

    def test_ambiguous_or_unpinned_declaration_rejected(self):
        for items in ((), ("torch>=2.13",), ("torch==2.13.*",), ("torch==2.13.0", "torch==2.13.0"), ("torch==2.13.0", "Torch>=2"), ("torch==2.13.0; python_version > '3.11'",), ("torch @ https://example.invalid/x",), ("torch[cuda]==2.13.0",), ("torch==2.13.0+cu128",), ("torch==2.13.0\n--extra-index-url=canary",)):
            with self.subTest(items=items), self.assertRaises(M.ContractError):
                M.declared_cpu_requirement(project(*items))

    def test_invalid_schema_or_oversized_bytes_rejected(self):
        for raw in (None, b"", b"\xff", b"[project]\ndependencies = 1", project(1), b"x" * (M.MAX_PROJECT_BYTES + 1)):
            with self.subTest(kind=type(raw).__name__), self.assertRaises(M.ContractError):
                M.declared_cpu_requirement(raw)

    def test_matching_cpu_metadata_is_narrowly_accepted(self):
        M.validate_runtime("torch==2.13.0+cpu", "2.13.0+cpu", "2.13.0+cpu", None, None)

    def test_torch_version_string_subclass_is_supported(self):
        class TorchVersion(str):
            pass
        M.validate_runtime("torch==2.13.0+cpu", "2.13.0+cpu", TorchVersion("2.13.0+cpu"), None, None)

    def test_gpu_build_or_wrong_version_is_rejected_without_gpu_availability(self):
        for distribution, module, cuda, hip in (("2.10.0+cpu", "2.10.0+cpu", None, None), ("2.13.0", "2.13.0+cu128", "12.8", None), ("2.13.0+cpu", "2.13.0+cpu", "12.8", None), ("2.13.0+cpu", "2.13.0+cpu", None, "6.4"), ("2.13.0+cpu", "2.10.0+cpu", None, None)):
            with self.subTest(distribution=distribution, module=module, cuda=cuda, hip=hip), self.assertRaises(M.ContractError):
                M.validate_runtime("torch==2.13.0+cpu", distribution, module, cuda, hip)

    def test_requirement_command_never_imports_torch(self):
        with patch.object(M, "read_project", return_value=project("torch==2.13.0")), patch.object(M.importlib, "import_module") as imported, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(M.run(["requirement"]), 0)
            imported.assert_not_called()
            self.assertEqual(output.getvalue(), "torch==2.13.0+cpu\n")

    def test_wrong_distribution_fails_before_import(self):
        with patch.object(M, "read_project", return_value=project("torch==2.13.0")), patch.object(M.importlib.metadata, "version", return_value="2.10.0+cpu"), patch.object(M.importlib, "import_module") as imported, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(M.run(["verify"]), 2)
            imported.assert_not_called()

    def test_import_failure_is_sanitized(self):
        with patch.object(M, "read_project", return_value=project("torch==2.13.0")), patch.object(M.importlib.metadata, "version", return_value="2.13.0+cpu"), patch.object(M.importlib, "import_module", side_effect=RuntimeError("PRIVATE_CANARY")), contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(M.run(["verify"]), 2)
            self.assertNotIn("PRIVATE_CANARY", output.getvalue())
            self.assertEqual(json.loads(output.getvalue())["reason"], "ENVIRONMENT_UNAVAILABLE")

    def test_real_verifier_control_flow_preserves_claim_bounds(self):
        fake = SimpleNamespace(__version__="2.13.0+cpu", version=SimpleNamespace(cuda=None, hip=None))
        with patch.object(M, "read_project", return_value=project("torch==2.13.0")), patch.object(M.importlib.metadata, "version", return_value="2.13.0+cpu"), patch.object(M.importlib, "import_module", return_value=fake), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(M.run(["verify"]), 0)
            report = json.loads(output.getvalue())
            self.assertFalse(report["wheel_bytes_verified"])
            self.assertFalse(report["production_authorized"])
            self.assertEqual(report["state"], "PASS_CPU_PACKAGE_IDENTITY_ONLY")

    def test_source_movement_fails(self):
        fake = SimpleNamespace(__version__="2.13.0+cpu", version=SimpleNamespace(cuda=None, hip=None))
        with patch.object(M, "read_project", side_effect=[project("torch==2.13.0"), project("torch==2.14.0")]), patch.object(M.importlib.metadata, "version", return_value="2.13.0+cpu"), patch.object(M.importlib, "import_module", return_value=fake), contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(M.run(["verify"]), 2)
            self.assertEqual(json.loads(output.getvalue())["reason"], "PROJECT_CHANGED_DURING_VERIFICATION")

    def test_workflow_uses_constraint_and_verifies_both_install_stages(self):
        workflow = (ROOT / ".github/workflows/rlt-continuity.yml").read_text()
        self.assertIn("python tools/rlt_cpu_environment.py requirement", workflow)
        self.assertIn('PIP_CONSTRAINT: ${{ runner.temp }}/rlt-cpu-constraint.txt', workflow)
        self.assertEqual(workflow.count("python tools/rlt_cpu_environment.py verify"), 2)
        self.assertNotIn("torch==2.10.0", workflow)
        self.assertIn("tools/rlt_cpu_environment.py", workflow.split("  push:")[0])
        self.assertIn("tests/test_rlt_cpu_environment.py", workflow.split("  push:")[0])
        self.assertNotIn("paths:", workflow.split("  push:")[1].split("permissions:")[0])
        self.assertIn("if: always()", workflow)
        self.assertIn("--steps 120", workflow)
        self.assertIn("--steps 8 --single-seed 17", workflow)


if __name__ == "__main__":
    unittest.main()
