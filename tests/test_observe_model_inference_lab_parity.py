"""Offline negative controls for the anonymous model-inference-lab parity observer."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "tools" / "observe_model_inference_lab_parity.py"
sys.path.insert(0, str(SOURCE.parent))
spec = importlib.util.spec_from_file_location("model_lab_observer", SOURCE)
observer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(observer)

REVISION = "b" * 40


class ModelLabObservationTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.gettempdir()) / "synthetic-source-unused"
        self.called = []
        self.metadata = {"stage": "RUNNING", "sha": REVISION}
        self.parity = {
            "state": "REPOSITORY_BYTES_MATCH_ONLY",
            "space_repository_parity": True,
            "reason": "MANIFEST_BYTES_AND_LISTED_FILES_MATCH",
            "matched_count": 26,
        }

    def fetch(self, url, limit):
        self.assertEqual(url, observer.METADATA_URL)
        self.assertEqual(limit, observer.METADATA_LIMIT)
        return json.dumps(self.metadata).encode("utf-8")

    def compare(self, source_dir, space, revision):
        self.called.append((source_dir, space, revision))
        return self.parity

    def run_observe(self):
        return observer.observe(self.path, fetch=self.fetch, compare=self.compare)

    def test_positive_remains_unsigned_source_only(self):
        report = self.run_observe()
        self.assertEqual(report["state"], "REPOSITORY_BYTES_MATCH_ONLY")
        self.assertTrue(report["source_match_only"])
        self.assertEqual(self.called, [(self.path, observer.SPACE, REVISION)])
        for field in ("model_inference_verified", "served_image_verified", "signer_verified", "release_admitted", "provider_writes"):
            self.assertFalse(report[field])

    def test_readme_drift_fails_even_when_provider_running(self):
        self.parity = {"state": "DRIFT", "reason": "REMOTE_SOURCE_DIGEST_DRIFT", "mismatches": ["README.md"], "matched_count": 25, "space_repository_parity": False}
        report = self.run_observe()
        self.assertEqual(report["state"], "DRIFT")
        self.assertEqual(report["parity"]["mismatches"], ["README.md"])
        self.assertFalse(report["source_match_only"])

    def test_paused_sleeping_and_unknown_never_probe_files(self):
        for stage in ("PAUSED", "SLEEPING", "BUILDING", "RUNTIME_ERROR", "UNKNOWN"):
            with self.subTest(stage=stage):
                self.metadata = {"stage": stage, "sha": REVISION}
                report = self.run_observe()
                self.assertEqual(report["reason"], "PROVIDER_NOT_RUNNING")
                self.assertEqual(report["state"], "HOLD")
                self.assertFalse(self.called)

    def test_missing_and_malformed_revisions_fail_before_remote_files(self):
        for bad in (None, "", "main", "z" * 40, True, 123):
            with self.subTest(revision=repr(bad)):
                self.metadata = {"stage": "RUNNING", "sha": bad}
                report = self.run_observe()
                self.assertEqual(report["reason"], "IMMUTABLE_HF_REVISION_UNAVAILABLE")
                self.assertFalse(self.called)

    def test_duplicate_json_keys_and_nonfinite_numbers_refuse(self):
        for raw in (b'{"stage":"RUNNING","stage":"PAUSED","sha":"' + REVISION.encode() + b'"}',
                    b'{"stage":"RUNNING","sha":"' + REVISION.encode() + b'","data":NaN}',
                    b'{"stage":"RUNNING","sha":"' + REVISION.encode() + b'","data":1e999}'):
            with self.subTest(raw=raw[-30:]):
                with mock.patch.object(self, "fetch", return_value=raw):
                    report = self.run_observe()
                self.assertEqual(report["reason"], "RUNTIME_METADATA_INVALID")
                self.assertFalse(self.called)

    def test_invalid_json_and_nonobject_refuse(self):
        for raw in (b"[]", b"{}", b"{", b'{"stage":true}', b"null"):
            with self.subTest(raw=raw):
                with mock.patch.object(self, "fetch", return_value=raw):
                    report = self.run_observe()
                self.assertEqual(report["reason"], "RUNTIME_METADATA_INVALID")
                self.assertFalse(self.called)

    def test_rate_limit_is_unavailable_not_zero_spaces(self):
        with mock.patch.object(self, "fetch", side_effect=observer.parity.RemoteUnavailable("HTTP_429")):
            report = self.run_observe()
        self.assertEqual(report["state"], "UNAVAILABLE")
        self.assertEqual(report["reason"], "RUNTIME_METADATA_HTTP_429")
        self.assertIsNone(report["hf_repository_revision"])
        self.assertFalse(self.called)

    def test_manifest_mismatch_holds(self):
        self.parity = {"state": "LOCAL_SOURCE_CONFLICT", "reason": "LOCAL_FILE_DIGEST_DRIFT", "space_repository_parity": False}
        report = self.run_observe()
        self.assertEqual(report["state"], "HOLD")
        self.assertEqual(report["reason"], "LOCAL_FILE_DIGEST_DRIFT")

    def test_workflow_live_witness_is_readonly_and_off_on_pull_requests(self):
        workflow = (SOURCE.parents[1] / ".github/workflows/public-space-source-parity.yml").read_text()
        self.assertIn("github.event_name != 'pull_request'", workflow)
        self.assertIn("github.ref == 'refs/heads/main'", workflow)
        self.assertIn("schedule:", workflow)
        self.assertIn("contents: read", workflow)
        self.assertIn("persist-credentials: false", workflow)
        self.assertIn("--source-dir spaces/szl-model-inference-lab", workflow)
        self.assertNotIn("--publish", workflow)
        self.assertNotIn("secrets.", workflow)
        self.assertNotIn("id-token: write", workflow)

    def test_comparer_cannot_assert_runtime_authority(self):
        self.parity.update({"release_admitted": True, "model_inference_verified": True})
        report = self.run_observe()
        self.assertFalse(report["release_admitted"])
        self.assertFalse(report["model_inference_verified"])


if __name__ == "__main__":
    unittest.main()
