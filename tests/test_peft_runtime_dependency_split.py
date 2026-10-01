"""Keep the real PEFT and predecessor interoperability dependency scopes separate."""
from pathlib import Path
import unittest

from tools import evaluate_peft_runtime as runtime


class RuntimeDependencySplitTests(unittest.TestCase):
    def test_transformers_517_safetensors_requirement_is_satisfied(self):
        self.assertEqual(runtime.VERSIONS["safetensors"], "0.8.0")
        self.assertIn("safetensors==0.8.0", runtime.requirements())
        self.assertNotIn("safetensors==0.7.0", runtime.requirements())

    def test_legacy_interop_is_separate_not_disabled(self):
        path = Path(__file__).resolve().parents[1] / ".github/workflows/peft-export-runtime.yml"
        text = path.read_text()
        runtime_job, legacy = text.split("  legacy-interop:", 1)
        self.assertIn("python -m tools.evaluate_peft_runtime run", runtime_job)
        self.assertNotIn("-m tools.evaluate_peft_export_interop", runtime_job)
        self.assertIn("interop-venv/bin/python", legacy)
        self.assertIn("'safetensors==0.7.0'", legacy)
        self.assertIn("-m tools.evaluate_peft_export_interop", legacy)
        self.assertNotIn("continue-on-error", text)
        self.assertNotIn("--no-deps", text)
        self.assertNotIn("--ignore-installed", text)


if __name__ == "__main__":
    unittest.main()
