"""The prohibited-key denylist is a distinct control: it decides the refusal reason.

The closed feature schema alone already refuses unknown keys, so accept/refuse tests
cannot tell whether the denylist exists. These tests pin the clinical-boundary reason,
and the mutation test proves they fail when the denylist is emptied.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import oac_operational_health as kernel_module  # noqa: E402

MODEL = ROOT / "operational-model" / "artifacts" / "model.json"
RECEIPT = ROOT / "operational-model" / "artifacts" / "model-receipt.json"
FEATURES = {"listener_running": 1, "tls_enabled": 1, "peer_allowlist_configured": 1,
            "queue_utilization": 0.25, "consecutive_failures": 0, "seconds_since_last_success": 120.0,
            "ledger_integrity_ok": 1, "configuration_valid": 1}
PROHIBITED = "prohibited non-operational field"


def kernel():
    return kernel_module.OperationalHealthKernel(MODEL, RECEIPT)


class DenylistReasonTests(unittest.TestCase):
    def test_clean_features_are_accepted(self) -> None:
        self.assertIn("operator_attention_score", kernel().score(FEATURES))

    def test_prohibited_feature_keys_carry_the_boundary_reason(self) -> None:
        for key in ("patient", "Patient", "PATIENT_ID", "hl7", "HL7", "fhir", "mrn", "ssn", "specimen_id"):
            with self.subTest(key=key), self.assertRaisesRegex(kernel_module.ModelInputError, PROHIBITED):
                kernel().score({**FEATURES, key: 1})

    def test_unknown_operational_key_is_a_schema_refusal_not_a_boundary_refusal(self) -> None:
        with self.assertRaises(kernel_module.ModelInputError) as caught:
            kernel().score({**FEATURES, "cpu_temp": 41.0})
        self.assertIn("feature schema mismatch", str(caught.exception))
        self.assertNotIn(PROHIBITED, str(caught.exception))

    def test_emptied_denylist_is_detected(self) -> None:
        """Mutation: without the denylist the input is still refused, but for the wrong reason."""
        with mock.patch.object(kernel_module, "_PROHIBITED_KEYS", frozenset()):
            with self.assertRaises(kernel_module.ModelInputError) as caught:
                kernel().score({**FEATURES, "patient": 1})
        self.assertNotIn(PROHIBITED, str(caught.exception))
        self.assertIn("feature schema mismatch", str(caught.exception))


class CliDocumentBoundaryTests(unittest.TestCase):
    def run_cli(self, document: object) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            proc = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "src" / "oac_operational_health.py"),
                                   "--model", str(MODEL), "--receipt", str(RECEIPT), "--input", str(path)],
                                  capture_output=True, text=True, check=False)
        return proc.returncode, proc.stdout + proc.stderr

    def test_wrapped_clean_document_is_accepted(self) -> None:
        code, output = self.run_cli({"features": FEATURES})
        self.assertEqual(0, code, output)

    def test_prohibited_keys_outside_features_carry_the_boundary_reason(self) -> None:
        for document, location in (({"features": FEATURES, "patient": {"id": 1}}, "input.patient"),
                                   ({"features": FEATURES, "fhir": {"resourceType": "Bundle"}}, "input.fhir"),
                                   ({"features": FEATURES, "metadata": {"mrn": "x"}}, "input.metadata.mrn")):
            with self.subTest(location=location):
                code, output = self.run_cli(document)
                self.assertNotEqual(0, code)
                self.assertIn(f"{PROHIBITED} at {location}", output)

    def test_raw_hl7_segment_is_refused_by_content(self) -> None:
        code, output = self.run_cli({"features": FEATURES, "note": "MSH|^~\\&|SYNTHETIC"})
        self.assertNotEqual(0, code)
        self.assertIn("raw HL7-like content is prohibited", output)


if __name__ == "__main__":
    unittest.main()
