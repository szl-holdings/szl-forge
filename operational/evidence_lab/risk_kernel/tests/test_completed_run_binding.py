"""Regression checks for receipts bound to completed, stable run artifacts."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "retrieval_risk_kernel.py"
SPEC = importlib.util.spec_from_file_location("completed_run_binding_kernel", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CompletedRunBindingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="szl-completed-run-")
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name) / "runs" / "baseline-review"
        self.run.mkdir(parents=True)
        self.write("payload.json", {"documents": [{"id": "one", "text": "fixture passage"}]})
        self.write("freeze.json", {"payload_sha256": self.sha("payload.json")})
        self.write("calibration.json", {"freeze_sha256": self.sha("freeze.json"), "threshold": 2.0})
        (self.run / "documents.npy").write_bytes(b"synthetic index bytes; no NumPy is loaded")
        self.result = {
            "status": MODULE.STATUS,
            "freeze_sha256": self.sha("freeze.json"),
            "index_sha256": self.sha("documents.npy"),
            "calibration_sha256": self.sha("calibration.json"),
        }
        self.write("result.json", self.result)
        self.status = {
            "run_id": self.run.name, "documents": 1, "threshold": 2.0,
            "freeze_sha256": self.sha("freeze.json"),
            "result_sha256": self.sha("result.json"),
            "preview_sha256": MODULE.sha256_file(SOURCE), "publication_eligible": False,
        }

    def write(self, name, value):
        (self.run / name).write_bytes(MODULE.canonical_bytes(value))

    def sha(self, name):
        return MODULE.sha256_file(self.run / name)

    def rebind_calibration(self):
        self.result["calibration_sha256"] = self.sha("calibration.json")
        self.write("result.json", self.result)

    def test_valid_completed_artifact_graph(self):
        hashes = MODULE.artifact_hashes(self.run)
        bound = MODULE.validate_completed_run(self.run, hashes)
        self.assertEqual(bound, {"artifacts": hashes, "run_id": "baseline-review",
                                 "documents": 1, "threshold": 2.0})
        MODULE.verify_status(self.status, self.run, hashes, SOURCE)

    def test_index_changed_before_initial_snapshot_is_rejected(self):
        (self.run / "documents.npy").write_bytes(b"different index loaded after server startup")
        with self.assertRaisesRegex(RuntimeError, "artifact binding"):
            MODULE.validate_completed_run(self.run)
        with self.assertRaisesRegex(RuntimeError, "artifact binding"):
            MODULE.verify_status(self.status, self.run, MODULE.artifact_hashes(self.run), SOURCE)

    def test_calibration_changed_before_initial_snapshot_is_rejected(self):
        self.write("calibration.json", {"freeze_sha256": self.sha("freeze.json"), "threshold": 999.0})
        with self.assertRaisesRegex(RuntimeError, "artifact binding"):
            MODULE.validate_completed_run(self.run)
        with self.assertRaisesRegex(RuntimeError, "artifact binding"):
            MODULE.verify_status(self.status, self.run, MODULE.artifact_hashes(self.run), SOURCE)

    def test_earlier_snapshot_must_match_current_artifacts(self):
        earlier = MODULE.artifact_hashes(self.run)
        (self.run / "documents.npy").write_bytes(b"changed since caller snapshot")
        with self.assertRaisesRegex(RuntimeError, "supplied snapshot"):
            MODULE.validate_completed_run(self.run, earlier)

    def test_calibration_must_link_back_to_selected_freeze(self):
        self.write("calibration.json", {"freeze_sha256": "0" * 64, "threshold": 2.0})
        self.rebind_calibration()
        with self.assertRaisesRegex(RuntimeError, "artifact binding"):
            MODULE.validate_completed_run(self.run)

    def test_noncompleted_result_is_rejected(self):
        self.result["status"] = "IN_PROGRESS"
        self.write("result.json", self.result)
        with self.assertRaisesRegex(RuntimeError, "completed evaluation"):
            MODULE.validate_completed_run(self.run)

    def test_invalid_threshold_types_are_rejected_even_with_matching_hashes(self):
        for threshold in (True, None, "2.0", [], {}):
            with self.subTest(threshold=threshold):
                self.write("calibration.json", {"freeze_sha256": self.sha("freeze.json"), "threshold": threshold})
                self.rebind_calibration()
                with self.assertRaisesRegex(ValueError, "calibration threshold"):
                    MODULE.validate_completed_run(self.run)

    def test_changed_artifact_during_parsing_is_rejected(self):
        original = MODULE.load_json

        def concurrent_change(path):
            value = original(path)
            if Path(path).name == "calibration.json":
                (self.run / "documents.npy").write_bytes(b"changed during validator")
            return value

        with mock.patch.object(MODULE, "load_json", side_effect=concurrent_change):
            with self.assertRaisesRegex(RuntimeError, "while validating"):
                MODULE.validate_completed_run(self.run)

    def test_live_status_must_name_selected_run(self):
        self.status["run_id"] = "other-run"
        with self.assertRaisesRegex(RuntimeError, "run identifier"):
            MODULE.verify_status(self.status, self.run, MODULE.artifact_hashes(self.run), SOURCE)

    def test_live_status_threshold_must_match_calibration(self):
        for threshold in (True, "2.0", 3.0, float("nan")):
            with self.subTest(threshold=threshold):
                self.status["threshold"] = threshold
                with self.assertRaisesRegex(RuntimeError, "preview threshold"):
                    MODULE.verify_status(self.status, self.run, MODULE.artifact_hashes(self.run), SOURCE)

    def test_live_status_document_count_cannot_be_boolean(self):
        self.status["documents"] = True
        with self.assertRaisesRegex(RuntimeError, "document count"):
            MODULE.verify_status(self.status, self.run, MODULE.artifact_hashes(self.run), SOURCE)

    def test_response_threshold_cannot_drift_after_status_verification(self):
        response = {
            "run_id": self.status["run_id"], "freeze_sha256": self.status["freeze_sha256"],
            "preview_sha256": self.status["preview_sha256"], "publication_eligible": False,
            "status": "ABSTAIN", "answer": None, "evidence": None,
            "scope": "retrieved_passages_only", "global_unanswerability": "UNVERIFIED",
            "margin": 1.0, "threshold": 2.0, "passages": [{"id": "one", "text": "fixture"}],
        }
        MODULE.validate_response(response, self.status)
        response["threshold"] = 3.0
        with self.assertRaisesRegex(RuntimeError, "response threshold"):
            MODULE.validate_response(response, self.status)


if __name__ == "__main__":
    unittest.main()
