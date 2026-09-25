"""Synthetic preview contract tests: no model loading, network, or quality claims."""
from __future__ import annotations

import ast
import concurrent.futures
import contextlib
import hashlib
from importlib.metadata import PackageNotFoundError
import io
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import external_retrieval_preview as preview


class StubService:
    """Model-free service double; validation errors are delegated service errors."""

    def __init__(self):
        self.lock = threading.Lock()
        self.run_id = "synthetic-preview-run"
        self.freeze_sha = "a" * 64
        self.document = {"id": "synthetic-unicode", "title": "Synthetic fixture",
                         "text": "\U0001f34e caf\u00e9 beside the river", "citation": "Synthetic only", "rank": 1}
        self.query = mock.Mock(side_effect=self._query)
        self.search = mock.Mock(side_effect=self._search)
        self.answer_context = mock.Mock(side_effect=self._answer_context)

    @staticmethod
    def _validate_question(question):
        if not isinstance(question, str) or not question.strip() or len(question) > 512:
            raise ValueError("synthetic question validation detail")

    @staticmethod
    def _answer(document, scope):
        context = document["text"]
        start = context.index("caf\u00e9") if "caf\u00e9" in context else 0
        end = start + 4 if "caf\u00e9" in context else min(len(context), 1)
        return {"status": "ANSWER", "answer": context[start:end], "scope": scope,
                "confidence_probability": None,
                "evidence": {"document_id": document["id"], "start": start, "end": end,
                             "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest()}}

    def _query(self, question):
        if not self.lock.locked():
            raise RuntimeError("Preview must own the lock during inference")
        self._validate_question(question)
        return {**self._answer(self.document, "retrieved_passages_only"),
                "passages": [self.document], "global_unanswerability": "UNVERIFIED",
                "false_answer_rate_after_retrieval_selection": "UNMEASURED"}

    def _search(self, question, k=5):
        if not self.lock.locked():
            raise RuntimeError("Preview must own the lock during inference")
        self._validate_question(question)
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10:
            raise ValueError("synthetic search validation detail")
        return [self.document]

    def _answer_context(self, question, context):
        if not self.lock.locked():
            raise RuntimeError("Preview must own the lock during inference")
        self._validate_question(question)
        if not isinstance(context, str) or not context.strip() or len(context) > 16000:
            raise ValueError("synthetic context validation detail")
        document = {"id": "provided-" + hashlib.sha256(context.encode("utf-8")).hexdigest(),
                    "text": context}
        return self._answer(document, "provided_context_only")


class PreviewTests(unittest.TestCase):
    routes = ("query", "search", "answer-context")

    def setUp(self):
        self.service = StubService()
        self.summary = {"status": "LOCAL_READY_NOT_PRODUCTION", "run_id": self.service.run_id,
                        "freeze_sha256": self.service.freeze_sha, "publication_eligible": False,
                        "limitations": ["Synthetic fixture, not quality evidence"]}
        self.app = preview.create_preview(self.service, self.summary, port=8766)
        self.client = TestClient(self.app, base_url="http://127.0.0.1:8766",
                                 raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def post(self, route="query", body=None, headers=None):
        if body is None:
            body = {"question": "Which word?"}
            if route == "answer-context":
                body["context"] = self.service.document["text"]
        return self.client.post("/api/" + route, json=body,
                                headers={"x-szl-preview": "1", **(headers or {})})

    def assert_security_headers(self, response):
        self.assertEqual("nosniff", response.headers.get("x-content-type-options"))
        self.assertEqual("no-referrer", response.headers.get("referrer-policy"))
        self.assertEqual("DENY", response.headers.get("x-frame-options"))
        self.assertEqual("no-store", response.headers.get("cache-control"))
        self.assertEqual("noindex, nofollow, noarchive", response.headers.get("x-robots-tag"))
        self.assertEqual("camera=(), microphone=(), geolocation=()",
                         response.headers.get("permissions-policy"))
        csp = response.headers.get("content-security-policy", "")
        for directive in ("default-src 'self'", "script-src 'self'", "style-src 'self'",
                          "connect-src 'self'", "object-src 'none'", "base-uri 'none'",
                          "frame-ancestors 'none'", "form-action 'self'"):
            self.assertIn(directive, csp)
        self.assertNotIn("unsafe-inline", csp)
        self.assertNotIn("unsafe-eval", csp)
        self.assertNotIn("access-control-allow-origin", response.headers)

    def assert_rejected_without_inference(self, response, status):
        self.assertEqual(status, response.status_code, response.text)
        self.assert_security_headers(response)
        self.service.query.assert_not_called()
        self.service.search.assert_not_called()
        self.service.answer_context.assert_not_called()
        self.assertFalse(self.service.lock.locked())

    def assert_provenance(self, response):
        self.assertEqual(200, response.status_code, response.text)
        self.assert_security_headers(response)
        body = response.json()
        self.assertEqual(self.service.run_id, body["run_id"])
        self.assertEqual(self.service.freeze_sha, body["freeze_sha256"])
        self.assertEqual(preview.digest(preview.__file__), body["preview_sha256"])
        self.assertFalse(body["publication_eligible"])
        self.assertGreaterEqual(body["elapsed_seconds"], 0)
        self.assertFalse(self.service.lock.locked())
        return body

    def test_exact_host_and_configured_port_are_required(self):
        for host in ("127.0.0.1", "localhost", "127.0.0.1:8765", "localhost:80",
                     "evil.example:8766", "127.0.0.1.evil.example:8766", "[::1]:8766", ""):
            with self.subTest(host=host):
                self.assert_rejected_without_inference(self.post(headers={"host": host}), 400)
        for host in ("localhost:8766", "127.0.0.1:8766"):
            with self.subTest(host=host):
                self.assertEqual(200, self.client.get("/api/status", headers={"host": host}).status_code)

    def test_custom_port_replaces_default_port(self):
        client = TestClient(preview.create_preview(self.service, self.summary, port=9988),
                            base_url="http://localhost:9988")
        self.addCleanup(client.close)
        self.assertEqual(200, client.get("/api/status").status_code)
        self.assertEqual(400, client.get("/api/status", headers={"host": "localhost:8766"}).status_code)

    def test_origin_must_match_exact_request_host(self):
        for host, origin in (("127.0.0.1:8766", "http://localhost:8766"),
                             ("localhost:8766", "http://127.0.0.1:8766"),
                             ("127.0.0.1:8766", "http://127.0.0.1:8765"),
                             ("127.0.0.1:8766", "https://127.0.0.1:8766"),
                             ("127.0.0.1:8766", "http://127.0.0.1:8766/"),
                             ("127.0.0.1:8766", "https://evil.example"),
                             ("127.0.0.1:8766", "null"), ("127.0.0.1:8766", "")):
            with self.subTest(host=host, origin=origin):
                self.assert_rejected_without_inference(self.post(headers={"host": host, "origin": origin}), 403)
        for host in ("127.0.0.1:8766", "localhost:8766"):
            self.assertEqual(200, self.post(headers={"host": host, "origin": "http://" + host}).status_code)

    def test_fetch_site_blocks_cross_site_and_same_site(self):
        for site in ("cross-site", "same-site", "", "unexpected"):
            with self.subTest(site=site):
                self.assert_rejected_without_inference(self.post(headers={"sec-fetch-site": site}), 403)
        for site in ("same-origin", "none"):
            self.assertEqual(200, self.post(headers={"sec-fetch-site": site}).status_code)
        self.assertEqual(200, self.post().status_code)

    def test_guard_applies_to_static_status_and_preflight(self):
        for route in ("/", "/app.js", "/app.css", "/api/status", "/api/notices"):
            for headers, status in (({"host": "evil.example"}, 400),
                                    ({"origin": "https://evil.example"}, 403),
                                    ({"sec-fetch-site": "cross-site"}, 403)):
                with self.subTest(route=route, headers=headers):
                    self.assert_rejected_without_inference(self.client.get(route, headers=headers), status)
        self.assert_rejected_without_inference(self.client.options("/api/query", headers={
            "origin": "https://evil.example", "access-control-request-method": "POST",
            "access-control-request-headers": "x-szl-preview"}), 403)

    def test_custom_request_header_is_required_on_every_inference_route(self):
        for route in self.routes:
            for value in (None, "", "0", "true", "1 "):
                headers = {} if value is None else {"x-szl-preview": value}
                with self.subTest(route=route, value=value):
                    self.assert_rejected_without_inference(self.client.post(
                        "/api/" + route, json={"question": "Which?"}, headers=headers), 403)

    def test_json_mime_required_with_charset_allowed(self):
        content = b'{"question":"Which?"}'
        for mime in (None, "text/plain", "application/x-www-form-urlencoded", "application/problem+json"):
            headers = {"x-szl-preview": "1"}
            if mime is not None:
                headers["content-type"] = mime
            with self.subTest(mime=mime):
                self.assert_rejected_without_inference(self.client.post(
                    "/api/query", content=content, headers=headers), 415)
        self.assertEqual(200, self.client.post("/api/query", content=content, headers={
            "x-szl-preview": "1", "content-type": "application/json; charset=utf-8"}).status_code)

    def test_body_over_64_kib_rejected_but_exact_byte_limit_accepted(self):
        content = '{"question":"caf\u00e9?"}'.encode("utf-8")
        headers = {"x-szl-preview": "1", "content-type": "application/json"}
        oversized = content + b" " * (65537 - len(content))
        self.assert_rejected_without_inference(self.client.post(
            "/api/query", content=oversized, headers=headers), 413)
        exact = content + b" " * (65536 - len(content))
        self.assertEqual(200, self.client.post("/api/query", content=exact, headers=headers).status_code)

    def test_declared_content_length_cannot_bypass_actual_body_limit(self):
        content = b'{"question":"Which?"}' + b" " * 65536
        self.assert_rejected_without_inference(self.client.post("/api/query", content=content, headers={
            "x-szl-preview": "1", "content-type": "application/json", "content-length": "1"}), 413)

    def test_unknown_missing_and_cross_route_fields_rejected(self):
        for route in self.routes:
            invalid = [{}, {"question": "Which?", "unexpected": 1}]
            invalid.append({"question": "Which?", "context": "passage"} if route != "answer-context"
                           else {"question": "Which?", "context": "passage", "k": 1})
            for body in invalid:
                with self.subTest(route=route, body=body):
                    self.assert_rejected_without_inference(self.post(route, body), 422)

    def test_nonobject_and_malformed_json_rejected(self):
        headers = {"x-szl-preview": "1", "content-type": "application/json"}
        for content in (b"", b"{", b'{"question":}', b"{} {}", b"\xff",
                        b"[]", b"null", b'"question"', b"7", b"true"):
            with self.subTest(content=content):
                self.assert_rejected_without_inference(self.client.post(
                    "/api/query", content=content, headers=headers), 422)

    def test_duplicate_json_fields_rejected_before_service(self):
        for route, content in (("query", b'{"question":"first","question":"second"}'),
                               ("search", b'{"question":"Which?","k":1,"k":2}'),
                               ("answer-context", b'{"question":"Which?","context":"a","context":"b"}')):
            with self.subTest(route=route):
                self.assert_rejected_without_inference(self.client.post("/api/" + route, content=content,
                    headers={"x-szl-preview": "1", "content-type": "application/json"}), 422)

    def test_nonfinite_json_constants_rejected_before_service(self):
        for constant in ("NaN", "Infinity", "-Infinity"):
            for route, content in (("query", '{"question":' + constant + '}'),
                                   ("search", '{"question":"Which?","k":' + constant + '}'),
                                   ("answer-context", '{"question":"Which?","context":' + constant + '}')):
                with self.subTest(route=route, constant=constant):
                    self.assert_rejected_without_inference(self.client.post("/api/" + route, content=content,
                        headers={"x-szl-preview": "1", "content-type": "application/json"}), 422)

    def test_deeply_nested_json_is_422_not_500(self):
        content = b'{"question":' + b"[" * 1100 + b"0" + b"]" * 1100 + b"}"
        self.assert_rejected_without_inference(self.client.post("/api/query", content=content,
            headers={"x-szl-preview": "1", "content-type": "application/json"}), 422)

    def test_delegated_question_type_and_length_errors_map_to_422(self):
        for route in self.routes:
            for question in (None, True, 3, [], {}, "", " \t\n", "x" * 513):
                body = {"question": question}
                if route == "answer-context":
                    body["context"] = self.service.document["text"]
                with self.subTest(route=route, question=repr(question)[:20]):
                    response = self.post(route, body)
                    self.assertEqual(422, response.status_code, response.text)
                    self.assert_security_headers(response)
                    self.assertNotIn("synthetic question validation detail", response.text)
                    self.assertFalse(self.service.lock.locked())

    def test_delegated_k_type_and_range_errors_map_to_422(self):
        for value in (None, True, False, 0, -1, 11, 1.0, "1", [], {}):
            with self.subTest(k=value):
                self.assertEqual(422, self.post("search", {"question": "Which?", "k": value}).status_code)
                self.assertFalse(self.service.lock.locked())
        for value in (1, 10):
            self.assertEqual(200, self.post("search", {"question": "Which?", "k": value}).status_code)

    def test_delegated_context_type_and_length_errors_map_to_422(self):
        bodies = [{"question": "Which?"}]
        bodies.extend({"question": "Which?", "context": value}
                      for value in (None, True, 3, [], {}, "", " \n", "x" * 16001))
        for body in bodies:
            with self.subTest(context=repr(body.get("context"))[:20]):
                response = self.post("answer-context", body)
                self.assertEqual(422, response.status_code, response.text)
                self.assertNotIn("synthetic context validation detail", response.text)
                self.assertFalse(self.service.lock.locked())

    def test_busy_lock_denies_all_inference_without_releasing_another_owner(self):
        self.service.lock.acquire()
        try:
            for route in self.routes:
                response = self.post(route)
                self.assertEqual(429, response.status_code, response.text)
                self.assert_security_headers(response)
                self.assertTrue(self.service.lock.locked())
            self.assertEqual(200, self.client.get("/api/status").status_code)
            self.assertEqual(200, self.client.get("/").status_code)
            self.service.query.assert_not_called()
            self.service.search.assert_not_called()
            self.service.answer_context.assert_not_called()
        finally:
            self.service.lock.release()

    def test_concurrent_request_sees_real_inflight_lock_and_status_remains_available(self):
        entered = threading.Event()
        finish = threading.Event()
        original = self.service._query

        def blocking_query(question):
            entered.set()
            if not finish.wait(timeout=10):
                raise RuntimeError("Synthetic test synchronization timeout")
            return original(question)

        self.service.query.side_effect = blocking_query
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(self.post)
            try:
                self.assertTrue(entered.wait(timeout=5))
                self.assertTrue(self.service.lock.locked())
                self.assertEqual(429, self.post("search").status_code)
                self.assertEqual(200, self.client.get("/api/status").status_code)
                self.service.search.assert_not_called()
            finally:
                finish.set()
            self.assertEqual(200, first.result(timeout=5).status_code)
        self.assertFalse(self.service.lock.locked())

    def test_delegated_input_exceptions_release_lock_and_do_not_leak_details(self):
        for route, method in (("query", self.service.query), ("search", self.service.search),
                              ("answer-context", self.service.answer_context)):
            original = method.side_effect
            for error in (ValueError, TypeError, KeyError, RecursionError):
                with self.subTest(route=route, error=error.__name__):
                    method.side_effect = error("synthetic private exception detail")
                    response = self.post(route)
                    self.assertEqual(422, response.status_code, response.text)
                    self.assertNotIn("synthetic private exception detail", response.text)
                    self.assertFalse(self.service.lock.locked())
                    method.side_effect = original
                    self.assertEqual(200, self.post(route).status_code)

    def test_runtime_exception_releases_lock_without_disclosing_private_detail(self):
        for route, method in (("query", self.service.query), ("search", self.service.search),
                              ("answer-context", self.service.answer_context)):
            original = method.side_effect
            with self.subTest(route=route):
                method.side_effect = RuntimeError("synthetic private runtime detail")
                response = self.post(route)
                self.assertEqual(500, response.status_code)
                self.assert_security_headers(response)
                self.assertEqual({"detail": "Local inference failed; inspect the runtime and retry."}, response.json())
                self.assertNotIn("synthetic private runtime detail", response.text)
                self.assertFalse(self.service.lock.locked())
                method.side_effect = original
                self.assertEqual(200, self.post(route).status_code)

    def test_query_preserves_evidence_limitations_and_unicode_offsets(self):
        body = self.assert_provenance(self.post())
        self.service.query.assert_called_once_with("Which word?")
        self.assertEqual("retrieved_passages_only", body["scope"])
        self.assertEqual("UNVERIFIED", body["global_unanswerability"])
        self.assertEqual("UNMEASURED", body["false_answer_rate_after_retrieval_selection"])
        self.assertIsNone(body["confidence_probability"])
        evidence = body["evidence"]
        self.assertEqual((2, 6), (evidence["start"], evidence["end"]))
        self.assertEqual("caf\u00e9", body["answer"])
        self.assertEqual(body["answer"], body["passages"][0]["text"][evidence["start"]:evidence["end"]])

    def test_search_wraps_passages_and_preserves_retrieval_only_scope(self):
        body = self.assert_provenance(self.post("search"))
        self.service.search.assert_called_once_with("Which word?", 5)
        self.assertEqual("RETRIEVED", body["status"])
        self.assertEqual("retrieval_only_not_answer_validation", body["scope"])
        self.assertEqual([self.service.document], body["passages"])
        self.assertNotIn("answer", body)
        self.assert_provenance(self.post("search", {"question": "Which?", "k": 2}))
        self.service.search.assert_called_with("Which?", 2)

    def test_context_adds_exact_utf8_digest_bound_passage_without_changing_offsets(self):
        context = "\U0001f34e caf\u00e9 with e\u0301 and \u6771\u4eac"
        body = self.assert_provenance(self.post("answer-context", {"question": "Which word?", "context": context}))
        self.service.answer_context.assert_called_once_with("Which word?", context)
        self.assertEqual("provided_context_only", body["scope"])
        passage = body["passages"][0]
        expected_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
        self.assertEqual(context, passage["text"])
        self.assertEqual("provided-" + expected_hash, passage["id"])
        self.assertEqual(passage["id"], body["evidence"]["document_id"])
        self.assertEqual(expected_hash, body["evidence"]["context_sha256"])
        self.assertEqual((2, 6), (body["evidence"]["start"], body["evidence"]["end"]))
        self.assertEqual(body["answer"], context[2:6])
        self.assertEqual(1, passage["rank"])
        self.assertIn("rights not reviewed", passage["citation"])

    def test_status_is_honest_bound_to_current_preview_and_assets_without_inference(self):
        response = self.client.get("/api/status")
        self.assertEqual(200, response.status_code)
        self.assert_security_headers(response)
        body = response.json()
        for key, value in self.summary.items():
            self.assertEqual(value, body[key])
        self.assertEqual(preview.digest(preview.__file__), body["preview_sha256"])
        self.assertEqual({name: preview.digest(preview.HERE / "retrieval_web" / name)
                          for name in ("index.html", "app.js", "app.css")}, body["asset_sha256"])
        self.service.query.assert_not_called()
        self.service.search.assert_not_called()
        self.service.answer_context.assert_not_called()

    def test_static_assets_and_notices_are_served_with_security_headers(self):
        for route, path, mime in (("/", preview.HERE / "retrieval_web" / "index.html", "text/html"),
                                  ("/app.js", preview.HERE / "retrieval_web" / "app.js", "javascript"),
                                  ("/app.css", preview.HERE / "retrieval_web" / "app.css", "text/css"),
                                  ("/api/notices", preview.HERE / "EXTERNAL_RETRIEVAL_NOTICES.md", "text/plain")):
            with self.subTest(route=route):
                response = self.client.get(route)
                self.assertEqual(200, response.status_code, response.text[:200])
                self.assertEqual(path.read_bytes(), response.content)
                self.assertIn(mime, response.headers["content-type"])
                self.assert_security_headers(response)

    def test_asset_hash_mismatch_fails_closed_without_editing_real_assets(self):
        for route in ("/", "/app.js", "/app.css"):
            with self.subTest(route=route), mock.patch.object(preview, "digest", return_value="0" * 64):
                response = self.client.get(route)
                self.assertEqual(503, response.status_code, response.text)
                self.assertIn("restart required", response.json()["detail"])
                self.assert_security_headers(response)

    def test_unknown_paths_docs_and_wrong_methods_are_not_exposed(self):
        for route in ("/docs", "/redoc", "/openapi.json", "/missing", "/external_retrieval_lab.py"):
            with self.subTest(route=route):
                response = self.client.get(route)
                self.assertEqual(404, response.status_code)
                self.assert_security_headers(response)
        for route in self.routes:
            response = self.client.get("/api/" + route)
            self.assertEqual(405, response.status_code)
            self.assert_security_headers(response)


class RunSummaryTests(unittest.TestCase):
    """Mock disk reads and hashes; do not create or modify frozen run artifacts."""

    def setUp(self):
        self.service = StubService()
        self.service.documents = [self.service.document]
        self.service.threshold = 2.0
        self.service.payload = {"calibration": [{"id": "calibration"}], "hotpot": [{"id": "hotpot"}],
            "evaluation": [{"id": "positive", "question": "Which word?", "context_id": "synthetic-unicode"},
                           {"id": "negative", "question": "Which planet?", "context_id": "synthetic-unicode"}]}
        self.result = {"freeze_sha256": self.service.freeze_sha,
            "status": "LOCAL_EVALUATION_COMPLETED_NOT_PRODUCTION", "index_sha256": "b" * 64,
            "calibration_sha256": "c" * 64, "runtime": {"gpu": "SYNTHETIC_NO_GPU"},
            "squad_retrieval": {"synthetic": True}, "hotpot_support_retrieval": {"synthetic": True},
            "given_context_answerability": {"synthetic": True},
            "retrieved_context_qa_answerable_only": {"synthetic": True},
            "completed_at": "SYNTHETIC_TIMESTAMP", "limitations": ["Synthetic fixture, not quality evidence"],
            "records": {"given_context": [
                {"id": "positive", "answerable": True, "margin": 3.0, "prediction": "caf\u00e9", "gold": ["caf\u00e9"]},
                {"id": "negative", "answerable": False, "margin": 2.0, "prediction": "caf\u00e9", "gold": []}]}}
        self.run = Path("synthetic-run-never-read")

    def summary(self, result):
        raw = json.dumps(result, ensure_ascii=False).encode("utf-8")
        hashes = {"documents.npy": "b" * 64, "calibration.json": "c" * 64}
        with mock.patch.object(Path, "read_bytes", return_value=raw), \
                mock.patch.object(preview, "digest", side_effect=lambda path: hashes[path.name]):
            return preview.run_summary(self.service, self.run), raw

    def test_summary_preserves_result_digest_and_labels_replayed_examples(self):
        summary, raw = self.summary(self.result)
        self.assertEqual("LOCAL_READY_NOT_PRODUCTION", summary["status"])
        self.assertFalse(summary["publication_eligible"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), summary["result_sha256"])
        self.assertEqual(self.service.freeze_sha, summary["freeze_sha256"])
        self.assertEqual(self.service.run_id, summary["run_id"])
        self.assertEqual("SYNTHETIC_NO_GPU", summary["device"])
        self.assertEqual(1, summary["documents"])
        self.assertEqual(1, summary["calibration_questions"])
        self.assertEqual(2, summary["evaluation_questions"])
        self.assertEqual(1, summary["hotpot_cases"])
        self.assertEqual(["positive", "negative"], [example["source_id"] for example in summary["examples"]])
        for example in summary["examples"]:
            self.assertEqual(self.service.document["text"], example["context"])
            self.assertEqual("Previously evaluated example, not new quality evidence", example["scope"])
        self.assertEqual(self.result["limitations"], summary["limitations"])

    def test_summary_rejects_freeze_status_index_and_calibration_mismatches(self):
        for field in ("freeze_sha256", "status", "index_sha256", "calibration_sha256"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "source mismatch"):
                self.summary({**self.result, field: "SYNTHETIC_MISMATCH"})


class RuntimeVerificationTests(unittest.TestCase):
    """Version metadata is stubbed; never import or construct model packages."""

    def setUp(self):
        self.run = Path("synthetic-run-never-read")
        self.recorded = {"torch": "2.11.0+cu128", "transformers": "5.3.0"}
        self.expected = {**self.recorded, "huggingface-hub": "1.29.0", "tokenizers": "0.23.2"}

    def test_matching_runtime_returns_exact_recorded_and_pinned_versions(self):
        with mock.patch.object(preview, "load_json", return_value={"runtime": self.recorded}) as read, \
                mock.patch.object(preview, "version", side_effect=self.expected.__getitem__) as version:
            self.assertEqual(self.expected, preview.verify_runtime(self.run))
        read.assert_called_once_with(self.run / "result.json")
        self.assertEqual([mock.call(name) for name in self.expected], version.call_args_list)

    def test_every_package_must_match_including_torch_local_build_tag(self):
        for name in self.expected:
            actual = {**self.expected, name: "SYNTHETIC_DIFFERENT_VERSION"}
            if name == "torch":
                actual[name] = "2.11.0"
            with self.subTest(package=name), \
                    mock.patch.object(preview, "load_json", return_value={"runtime": self.recorded}), \
                    mock.patch.object(preview, "version", side_effect=actual.__getitem__):
                with self.assertRaisesRegex(ValueError, "Preview dependency drift"):
                    preview.verify_runtime(self.run)

    def test_torch_and_transformers_expectations_are_read_from_frozen_result(self):
        recorded = {"torch": "SYNTHETIC_ALTERNATE_TORCH", "transformers": "SYNTHETIC_ALTERNATE_TRANSFORMERS",
                    "huggingface-hub": "IGNORED_UNTRUSTED_OVERRIDE", "tokenizers": "IGNORED_UNTRUSTED_OVERRIDE"}
        expected = {**self.expected, "torch": recorded["torch"], "transformers": recorded["transformers"]}
        with mock.patch.object(preview, "load_json", return_value={"runtime": recorded}), \
                mock.patch.object(preview, "version", side_effect=expected.__getitem__):
            self.assertEqual(expected, preview.verify_runtime(self.run))

    def test_missing_installed_package_fails_closed(self):
        for missing in self.expected:
            def lookup(name):
                if name == missing:
                    raise PackageNotFoundError(name)
                return self.expected[name]

            with self.subTest(package=missing), \
                    mock.patch.object(preview, "load_json", return_value={"runtime": self.recorded}), \
                    mock.patch.object(preview, "version", side_effect=lookup):
                with self.assertRaises(PackageNotFoundError):
                    preview.verify_runtime(self.run)

    def test_missing_recorded_versions_fail_before_metadata_lookup(self):
        for result in ({}, {"runtime": {}}, {"runtime": {"torch": self.recorded["torch"]}}):
            with self.subTest(result=result), mock.patch.object(preview, "load_json", return_value=result), \
                    mock.patch.object(preview, "version") as version:
                with self.assertRaises(KeyError):
                    preview.verify_runtime(self.run)
                version.assert_not_called()

    def test_main_rejects_runtime_drift_before_constructing_local_service(self):
        # Execute the actual startup body with all disk/version/model boundaries
        # stubbed, so ordering is tested without starting a server or model.
        source = Path(preview.__file__)
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        startup = next(node for node in tree.body if isinstance(node, ast.If)
                       and ast.unparse(node.test) == "__name__ == '__main__'")
        code = compile(ast.Module(body=startup.body, type_ignores=[]), str(source), "exec")
        actual = {**self.expected, "tokenizers": "SYNTHETIC_DIFFERENT_VERSION"}
        with mock.patch.object(sys, "argv", [str(source), "--lab-root", "synthetic-root", "--run-id", "synthetic-run"]), \
                mock.patch.object(preview, "paths", return_value=(Path("synthetic-root"), self.run)), \
                mock.patch.object(preview, "digest", return_value="a" * 64), \
                mock.patch.object(preview, "load_json", return_value={"runtime": self.recorded}), \
                mock.patch.object(preview, "version", side_effect=actual.__getitem__), \
                mock.patch.object(preview, "LocalService") as service, \
                mock.patch.object(preview, "run_summary") as summary, \
                mock.patch.object(preview, "create_preview") as create, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "Preview dependency drift"):
                exec(code, dict(vars(preview)))
        service.assert_not_called()
        summary.assert_not_called()
        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
