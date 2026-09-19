"""HTTP integration: optional guard, unchanged base path and source bindings."""
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_external_retrieval_preview import StubService
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import external_retrieval_preview as preview
from fastapi.testclient import TestClient


class GuardIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.service = StubService()
        self.service.documents = [self.service.document]
        self.app = preview.create_preview(self.service, {"status": "LOCAL_READY_NOT_PRODUCTION"})
        self.client = TestClient(self.app, base_url="http://127.0.0.1:8766")
        self.addCleanup(self.client.close)

    def post(self, route, question):
        return self.client.post("/api/" + route, json={"question": question}, headers={"X-SZL-Preview": "1"})

    def test_unknown_identifier_abstains_without_leaking_base_answer(self):
        question = "Who founded LongSubjectIdentifier012345?"
        base = self.post("query", question).json()
        result = self.post("query-guarded", question).json()
        self.assertEqual(base["status"], "ANSWER")
        self.assertEqual(result["status"], "ABSTAIN")
        self.assertIsNone(result["answer"])
        self.assertIsNone(result["evidence"])
        self.assertEqual(result["identifier_guard"]["reason"], "UNRESOLVED_IDENTIFIER")

    def test_ordinary_positive_preserves_answer_and_source_span(self):
        base = self.post("query", "Which word?").json()
        guarded = self.post("query-guarded", "Which word?").json()
        self.assertEqual(base["answer"], guarded["answer"])
        self.assertEqual(base["evidence"], guarded["evidence"])

    def test_guard_rejects_unknown_body_fields(self):
        response = self.client.post("/api/query-guarded", json={"question": "word?", "threshold": 0}, headers={"X-SZL-Preview": "1"})
        self.assertEqual(response.status_code, 422)

    def test_guard_retains_cross_origin_and_header_checks(self):
        self.assertEqual(self.client.post("/api/query-guarded", json={"question": "word?"}).status_code, 403)
        self.assertEqual(self.client.post("/api/query-guarded", json={"question": "word?"}, headers={"X-SZL-Preview": "1", "Origin": "https://invalid.example"}).status_code, 403)

    def test_source_drift_disables_runtime(self):
        with mock.patch.object(preview, "digest", return_value="0" * 64):
            response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 503)

    def test_status_reports_guard_and_complete_source_set(self):
        status = self.client.get("/api/status").json()
        self.assertTrue(status["capabilities"]["identifier_guard"])
        self.assertEqual(len(status["runtime_source_sha256"]), 8)


if __name__ == "__main__":
    unittest.main()
