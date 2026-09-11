"""Job-local closure recording. Does not pip-install TRL/vLLM."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from frontier.asyncgrpo_lora_closure import (
    PEFT_SPEC,
    TRL_GIT_REQUIREMENT,
    TRL_REVISION,
    VLLM_SPEC,
    recorded_requirements_text,
    spec_satisfied,
)
from frontier.asyncgrpo_lora_install import JobLocalInstallError, prepare_job_local


class ClosureRecordTests(unittest.TestCase):
    def test_recorded_specs_match_trl_pin_extras(self) -> None:
        self.assertEqual(PEFT_SPEC, "peft>=0.13.0")
        self.assertEqual(VLLM_SPEC, "vllm>=0.19.1,<=0.28.0")
        text = recorded_requirements_text()
        self.assertIn(TRL_GIT_REQUIREMENT, text)
        self.assertIn(PEFT_SPEC, text)
        self.assertIn(VLLM_SPEC, text)
        self.assertIn(TRL_REVISION, text)
        self.assertIn("not invented pins", text)
        self.assertNotIn("peft==", text)
        self.assertNotIn("vllm==", text)

    def test_spec_floor_and_cap(self) -> None:
        self.assertTrue(spec_satisfied("0.13.0", PEFT_SPEC))
        self.assertTrue(spec_satisfied("0.17.1", PEFT_SPEC))
        self.assertFalse(spec_satisfied("0.12.9", PEFT_SPEC))
        self.assertTrue(spec_satisfied("0.19.1", VLLM_SPEC))
        self.assertTrue(spec_satisfied("0.28.0", VLLM_SPEC))
        self.assertFalse(spec_satisfied("0.19.0", VLLM_SPEC))
        self.assertFalse(spec_satisfied("0.28.1", VLLM_SPEC))
        self.assertFalse(spec_satisfied(True, PEFT_SPEC))
        self.assertFalse(spec_satisfied(None, PEFT_SPEC))

    def test_prepare_writes_closure_and_never_uses_production_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = prepare_job_local(Path(tmp) / "job")
            self.assertIs(result["productionRuntime"], False)
            req = Path(result["requirementsPath"]).read_text(encoding="utf-8")
            self.assertIn(TRL_REVISION, req)
            closure = result["closure"]
            assert isinstance(closure, dict)
            self.assertFalse(closure["compatibleExactPeftVllmClosure"])
            self.assertEqual(closure["peft"]["spec"], PEFT_SPEC)
            self.assertEqual(closure["vllm"]["spec"], VLLM_SPEC)
            self.assertIs(closure["peft"]["pinInvented"], False)
            self.assertEqual(closure["install"]["status"], "NOT_ATTEMPTED")
            self.assertNotEqual(result.get("venvPython"), sys.executable)
            payload = json.loads(Path(result["closurePath"]).read_text(encoding="utf-8"))
            self.assertFalse(payload["productionRuntime"])

    def test_non_boolean_install_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(JobLocalInstallError):
                prepare_job_local(Path(tmp), install="false")  # type: ignore[arg-type]

    def test_symlink_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "real"
            real.mkdir()
            link = Path(tmp) / "link"
            try:
                link.symlink_to(real, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlink unavailable: {type(exc).__name__}")
            with self.assertRaises(JobLocalInstallError):
                prepare_job_local(link)


if __name__ == "__main__":
    unittest.main()
