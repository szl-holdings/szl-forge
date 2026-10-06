# SPDX-License-Identifier: Apache-2.0
"""SIMULATED capture-interface tests, not signing or model evidence.

Fixtures are independently constructed here through existing software helpers.
No actual training, dev or held-out curriculum file is opened by this suite.
The root separately replays the real gates with the frozen train-only bytes.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curriculum_admission as admission
import json_contract as contract
import training_data
from test_curriculum_admission import fixture_manifest, fixture_row, serialize


class TrainOnlyCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [fixture_row(family, kind, index)
                    for family in sorted(admission.TRAIN_FAMILIES)
                    for kind in contract.CLAIMS for index in range(30)]
        cls.raw = serialize(cls.rows)
        cls.manifest_raw = (contract.canonical_json(fixture_manifest(cls.raw)) + "\n").encode("utf-8")

    def simulated_gate_report(self):
        return {"generic_sft_valid": True, "nemo_valid": True,
                "nemo_receipt_chain_tip": "SIMULATED-not-a-kernel-receipt",
                "nemo_receipt_chain": [], "nemo_receipt_count": 360,
                "gate_reports_authenticated": False,
                "nemo_source_binding": {"evidence_class": "SIMULATED"}}

    def test_capture_retains_checked_buffers_and_does_not_read_other_splits(self):
        opened = []
        real_read = admission._read_once

        def recorded(path, maximum):
            opened.append(Path(path))
            self.assertNotIn("heldout", Path(path).parts)
            self.assertNotEqual(Path(path).name, "dev.jsonl")
            return real_read(path, maximum)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, manifest = root / "train.jsonl", root / "curriculum-manifest.json"
            train.write_bytes(self.raw)
            manifest.write_bytes(self.manifest_raw)
            with patch.object(admission, "MANIFEST_SHA256", hashlib.sha256(self.manifest_raw).hexdigest()), \
                 patch.object(admission, "run_gates", return_value=self.simulated_gate_report()) as gates, \
                 patch.object(admission, "_read_once", side_effect=recorded):
                captured = admission.capture_train_only(train, manifest,
                                                        nemo_source_root=root / "SIMULATED-source")
            gates.assert_called_once_with(self.raw, root / "SIMULATED-source")
            train.write_bytes(b"changed after capture")
            manifest.write_bytes(b"changed after capture")
        self.assertEqual(captured.train_bytes, self.raw)
        self.assertEqual(captured.manifest_bytes, self.manifest_raw)
        self.assertEqual(opened.count(train), 1)
        self.assertEqual(opened.count(manifest), 1)
        report = json.loads(captured.conformance_report_bytes)
        self.assertEqual(report["rows"], 360)
        self.assertEqual(report["state"], "BLOCKED")
        for field in ("training_eligible", "publication_eligible", "execution_authority",
                      "signature_valid", "heldout_content_opened", "gate_reports_authenticated"):
            self.assertIs(report[field], False)
        self.assertEqual(report["key_trust"], "REPO_DECLARED")
        self.assertEqual(report["trainer_supervisor_binding"], "UNAVAILABLE")
        with self.assertRaises(FrozenInstanceError):
            captured.train_bytes = b"replacement"

    def test_snapshot_rejects_mutable_or_nonbytes_fields(self):
        for fields in ((bytearray(b"manifest"), b"train", b"report"),
                       (b"manifest", "train", b"report"),
                       (b"manifest", b"train", {"state": "VALID"})):
            with self.subTest(fields=fields), self.assertRaises(admission.AdmissionError):
                admission.TrainOnlySnapshot(*fields)

    def test_fresh_capture_to_converter_retains_exact_messages_without_rereads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, manifest = root / "train.jsonl", root / "curriculum-manifest.json"
            train.write_bytes(self.raw)
            manifest.write_bytes(self.manifest_raw)
            with patch.object(admission, "MANIFEST_SHA256", hashlib.sha256(self.manifest_raw).hexdigest()), \
                 patch.object(admission, "run_gates", return_value=self.simulated_gate_report()):
                captured = admission.capture_train_only(train, manifest)
                train.write_bytes(b"changed after successful fixture capture")
                with patch.object(admission, "_read_once", side_effect=AssertionError("source reread")), \
                     patch.object(admission, "run_gates", side_effect=AssertionError("gate rerun")), \
                     patch.object(admission, "validate_train_bytes", side_effect=AssertionError("schema reread")):
                    messages = training_data.training_messages(captured)
        expected = tuple(tuple((message["role"], message["content"])
                               for message in row["messages"]) for row in self.rows)
        self.assertEqual(messages, expected)
        self.assertEqual(json.loads(captured.conformance_report_bytes)["state"], "BLOCKED")

    def test_report_buffer_does_not_retain_mutable_gate_report_references(self):
        gate_report = self.simulated_gate_report()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, manifest = root / "train.jsonl", root / "curriculum-manifest.json"
            train.write_bytes(self.raw)
            manifest.write_bytes(self.manifest_raw)
            with patch.object(admission, "MANIFEST_SHA256", hashlib.sha256(self.manifest_raw).hexdigest()), \
                 patch.object(admission, "run_gates", return_value=gate_report):
                captured = admission.capture_train_only(train, manifest)
        gate_report["nemo_source_binding"]["evidence_class"] = "fake replacement"
        self.assertEqual(json.loads(captured.conformance_report_bytes)["nemo_source_binding"],
                         {"evidence_class": "SIMULATED"})

    def test_failures_never_yield_a_snapshot_and_preserve_cli_error_semantics(self):
        for failure, exit_code in ((admission.GateNotReady("SIMULATED missing kernel"), 2),
                                   (admission.AdmissionError("SIMULATED gate violation"), 1)):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                train, manifest = root / "train.jsonl", root / "curriculum-manifest.json"
                train.write_bytes(self.raw)
                manifest.write_bytes(self.manifest_raw)
                with patch.object(admission, "MANIFEST_SHA256", hashlib.sha256(self.manifest_raw).hexdigest()), \
                     patch.object(admission, "run_gates", side_effect=failure):
                    with self.assertRaises(type(failure)):
                        admission.capture_train_only(train, manifest)
                    report, code = admission.check_conformance(train, manifest)
                self.assertEqual(code, exit_code)
                self.assertIs(report["contract_conforms"], True)
                self.assertIs(report["training_eligible"], False)
                self.assertIs(report["nemo_valid"], False)

    def test_invalid_inputs_fail_before_gates(self):
        with patch.object(admission, "_read_once") as reads, \
             patch.object(admission, "run_gates") as gates:
            with self.assertRaises(admission.AdmissionError):
                admission.capture_train_only(Path("dev.jsonl"), Path("curriculum-manifest.json"))
            with self.assertRaises(admission.AdmissionError):
                admission.capture_train_only(Path("one/train.jsonl"), Path("two/curriculum-manifest.json"))
            reads.assert_not_called()
            gates.assert_not_called()

    def test_capture_has_no_saved_report_override(self):
        with patch.object(admission, "run_gates") as gates:
            with self.assertRaises(TypeError):
                admission.capture_train_only(Path("train.jsonl"), Path("curriculum-manifest.json"),
                                             conformance_report={"state": "VALID"})
            gates.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
