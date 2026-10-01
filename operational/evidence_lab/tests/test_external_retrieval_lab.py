"""Synthetic local boundary tests; no downloads, real models, or benchmark runs."""
import concurrent.futures
import hashlib
import math
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

import numpy as np
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "operational"))
import external_retrieval_lab as lab


class ArtifactTests(unittest.TestCase):
    def test_simple_run_ids_resolve_under_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            for run_id in ("a", "Trial_09-ab", "x" * 64):
                root, run = lab.paths(directory, run_id)
                self.assertEqual(Path(directory).resolve(), root)
                self.assertEqual(root / "runs" / run_id, run)

    def test_run_id_path_traversal_and_non_simple_names_rejected(self):
        for run_id in ("", ".", "..", "../escape", "..\\escape", "/absolute",
                       "C:\\escape", "a/b", "a\\b", "a:b", "a\x00b", " space",
                       "a ", "a\n", "_a", "-a", "caf\u00e9", "x" * 65):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                lab.paths(".", run_id)

    def test_exclusive_save_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            lab.save_new(target, {"message": "caf\u00e9", "number": 4})
            first = target.read_bytes()
            self.assertEqual({"message": "caf\u00e9", "number": 4}, lab.load_json(target))
            self.assertTrue(first.endswith(b"\n"))
            with self.assertRaises(FileExistsError):
                lab.save_new(target, {"number": 5})
            self.assertEqual(first, target.read_bytes())

    def test_concurrent_exclusive_saves_have_one_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            barrier = threading.Barrier(4)

            def save(index):
                barrier.wait(timeout=10)
                try:
                    lab.save_new(target, {"winner": index})
                    return index
                except FileExistsError:
                    return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                winners = [item for item in pool.map(save, range(4)) if item is not None]
            self.assertEqual(1, len(winners))
            self.assertEqual({"winner": winners[0]}, lab.load_json(target))

    def test_invalid_serialization_does_not_create_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, value in enumerate(({"margin": math.nan}, {"margin": math.inf}, {"bad": object()})):
                target = Path(directory) / f"invalid-{index}.json"
                with self.subTest(index=index):
                    with self.assertRaises((ValueError, TypeError)):
                        lab.save_new(target, value)
                    self.assertFalse(target.exists(), "Rejected JSON must not leave an empty frozen artifact")

    def test_json_loading_rejects_nonfinite_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "invalid.json"
            for value in (b"NaN", b"Infinity", b"-Infinity", b'{"x":NaN}'):
                target.write_bytes(value)
                with self.subTest(value=value), self.assertRaises(ValueError):
                    lab.load_json(target)

    def test_asset_receipt_rejects_missing_or_tampered_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "model.safetensors"
            content = b"synthetic pinned bytes only"
            expected = {target.name: hashlib.sha256(content).hexdigest()}
            with self.assertRaisesRegex(ValueError, "integrity"):
                lab.verify_files(root, expected)
            target.write_bytes(content)
            self.assertEqual({target.name: {"sha256": expected[target.name], "bytes": len(content)}},
                             lab.verify_files(root, expected))
            target.write_bytes(b"X" + content[1:])
            with self.assertRaisesRegex(ValueError, "integrity"):
                lab.verify_files(root, expected)

    def test_symbolic_asset_is_not_accepted_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"same content")
            target = root / "model.safetensors"
            try:
                target.symlink_to(source)
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"Local symbolic link privilege unavailable: {type(error).__name__}")
            with self.assertRaisesRegex(ValueError, "integrity"):
                lab.verify_files(root, {target.name: lab.digest(source)})

    def test_symlink_guard_rejects_before_reading_asset_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.safetensors"
            target.write_bytes(b"synthetic asset")
            with mock.patch.object(Path, "is_symlink", return_value=True), \
                    mock.patch.object(lab, "digest") as digest:
                with self.assertRaisesRegex(ValueError, "integrity"):
                    lab.verify_files(Path(directory), {target.name: "0" * 64})
                digest.assert_not_called()

    def test_unpinned_loader_files_fail_asset_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder = root / "encoder"
            reader = root / "assets" / "reader"
            encoder.mkdir()
            reader.mkdir(parents=True)
            with mock.patch.object(lab, "QWEN_FILES", {}), \
                    mock.patch.object(lab, "READER_FILES", {}), \
                    mock.patch.object(lab, "DATA_FILES", {}):
                self.assertEqual({"encoder": {}, "reader": {}, "datasets": {}},
                                 lab.verify_assets(root, encoder))
                for model_dir in (encoder, reader):
                    injected = model_dir / "adapter_config.json"
                    injected.write_bytes(b"{}")
                    with self.subTest(directory=model_dir.name), self.assertRaisesRegex(ValueError, "Unexpected file"):
                        lab.verify_assets(root, encoder)
                    injected.unlink()


class PoolingTests(unittest.TestCase):
    def test_last_actual_token_for_left_right_and_no_padding(self):
        hidden = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
        masks = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
        pooled = lab.last_token_pool(hidden, masks)
        torch.testing.assert_close(pooled, torch.stack([hidden[0, 1], hidden[1, 3], hidden[2, 3]]))
        self.assertEqual(hidden.dtype, pooled.dtype)

    def test_boolean_mask_and_noncontiguous_padding_select_last_active(self):
        hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1)
        masks = torch.tensor([[True, False, True, False], [False, True, False, False]])
        torch.testing.assert_close(lab.last_token_pool(hidden, masks), torch.tensor([[2.0], [5.0]]))

    def test_any_all_padding_row_fails_closed(self):
        hidden = torch.zeros((2, 3, 4))
        for mask in ([[0, 0, 0], [1, 0, 0]], [[1, 1, 1], [0, 0, 0]]):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "All-padding"):
                lab.last_token_pool(hidden, torch.tensor(mask))

    def test_malformed_pooling_dimensions_raise(self):
        for hidden, mask in ((torch.zeros(2, 3), torch.ones(2, 3)),
                             (torch.zeros(2, 3, 4, 1), torch.ones(2, 3)),
                             (torch.zeros(2, 3, 4), torch.ones(2, 2)),
                             (torch.zeros(2, 3, 4), torch.ones(3)),
                             (torch.zeros(2, 0, 4), torch.ones(2, 0)),
                             (torch.zeros(0, 3, 4), torch.ones(0, 3)),
                             (torch.zeros(2, 3, 0), torch.ones(2, 3))):
            with self.subTest(hidden=hidden.shape, mask=mask.shape), self.assertRaises(ValueError):
                lab.last_token_pool(hidden, mask)

    def test_non_binary_or_nonfinite_attention_mask_rejected(self):
        hidden = torch.ones(1, 3, 2)
        for value in (-1, 2, 0.5, math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "attention mask"):
                lab.last_token_pool(hidden, torch.tensor([[1, value, 0]]))


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.doc = {"id": "synthetic-1", "title": "Toy", "citation": "synthetic test fixture",
                    "text": "The red apple"}
        self.pred = {"prediction": "red", "margin": 3.0, "start": 4, "end": 7}

    def test_answer_preserves_literal_exclusive_span_and_source_digest(self):
        answer = lab.evidence_answer(self.pred, self.doc, 2.0)
        self.assertEqual("ANSWER", answer["status"])
        self.assertEqual("red", answer["answer"])
        self.assertIsNone(answer["confidence_probability"])
        self.assertEqual({"document_id": "synthetic-1", "title": "Toy",
                          "citation": "synthetic test fixture",
                          "context_sha256": hashlib.sha256(self.doc["text"].encode()).hexdigest(),
                          "start": 4, "end": 7}, answer["evidence"])

    def test_threshold_is_strict_and_abstention_has_no_evidence(self):
        for margin, threshold, expected in ((3, 3, "ABSTAIN"), (2, 3, "ABSTAIN"),
                                             (math.nextafter(3, math.inf), 3, "ANSWER")):
            with self.subTest(margin=margin):
                answer = lab.evidence_answer({**self.pred, "margin": margin}, self.doc, threshold)
                self.assertEqual(expected, answer["status"])
                if expected == "ABSTAIN":
                    self.assertIsNone(answer["answer"])
                    self.assertIsNone(answer["evidence"])

    def test_whitespace_only_span_abstains_even_above_threshold(self):
        pred = {"prediction": " ", "margin": 30, "start": 3, "end": 4}
        self.assertEqual("ABSTAIN", lab.evidence_answer(pred, self.doc, 0)["status"])

    def test_invalid_span_bounds_types_and_source_mismatch_rejected(self):
        for start, end in ((-1, 7), (4, 14), (7, 4), (4, 4), (True, 7), (4, False),
                           (4.0, 7), (4, 7.0), (None, 7)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                lab.evidence_answer({**self.pred, "start": start, "end": end}, self.doc, 0)
        with self.assertRaises(ValueError):
            lab.evidence_answer({**self.pred, "prediction": "invented"}, self.doc, 0)
        # Invalid evidence must also fail when its score would have abstained.
        with self.assertRaises(ValueError):
            lab.evidence_answer({**self.pred, "start": -1, "margin": -100}, self.doc, 0)

    def test_nonfinite_or_boolean_threshold_and_margin_rejected(self):
        for value in (math.nan, math.inf, -math.inf, True, False):
            with self.subTest(value=value, field="threshold"), self.assertRaises(ValueError):
                lab.evidence_answer(self.pred, self.doc, value)
            with self.subTest(value=value, field="margin"), self.assertRaises(ValueError):
                lab.evidence_answer({**self.pred, "margin": value}, self.doc, 0)

    def test_unicode_offsets_are_character_indices_not_utf8_bytes(self):
        doc = {**self.doc, "text": "\U0001f34e caf\u00e9"}
        pred = {**self.pred, "prediction": "caf\u00e9", "start": 2, "end": 6}
        answer = lab.evidence_answer(pred, doc, 0)
        self.assertEqual("caf\u00e9", answer["answer"])
        self.assertEqual(hashlib.sha256(doc["text"].encode("utf-8")).hexdigest(),
                         answer["evidence"]["context_sha256"])


class FakeService(lab.LocalService):
    """Use real service validation/routing with fake encoder and reader outputs."""

    def __init__(self):
        self.documents = [{"id": "synthetic-1", "title": "Toy",
                           "citation": "synthetic test fixture", "text": "The red apple"}]
        self.vectors = np.array([[1.0, 0.0]], dtype=np.float32)
        self.encoder = mock.Mock()
        self.encoder.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
        self.reader = mock.Mock()
        self.reader.predict.return_value = {"prediction": "red", "margin": 3.0, "start": 4, "end": 7}
        self.lexical = lab.BM25([item["text"] for item in self.documents])
        self.run_id, self.freeze_sha, self.threshold = "synthetic-run", "0" * 64, 2.0
        self.lock = threading.Lock()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.service = FakeService()
        self.client = TestClient(lab.create_app(self.service), raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def assert_rejected_without_inference(self, response, status):
        self.assertEqual(status, response.status_code, response.text)
        self.service.encoder.encode.assert_not_called()
        self.service.reader.predict.assert_not_called()

    def test_health_is_honest_and_does_not_run_inference(self):
        response = self.client.get("/health")
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual("LOCAL_READY_NOT_PRODUCTION", body["status"])
        self.assertEqual(lab.SCOPE, body["scope"])
        self.assertFalse(body["training_performed"])
        self.assertEqual(1, body["documents"])
        self.assertEqual(self.service.freeze_sha, body["freeze_sha256"])
        self.service.encoder.encode.assert_not_called()

    def test_search_query_and_context_routes_have_real_service_contracts(self):
        search = self.client.post("/search", json={"question": "Which color?", "k": 1})
        self.assertEqual(200, search.status_code, search.text)
        self.assertEqual("synthetic-1", search.json()[0]["id"])
        self.assertEqual(1, search.json()[0]["rank"])
        query = self.client.post("/query", json={"question": "Which color?"})
        self.assertEqual(200, query.status_code, query.text)
        self.assertEqual("red", query.json()["answer"])
        self.assertEqual("retrieved_passages_only", query.json()["scope"])
        self.assertEqual("UNVERIFIED", query.json()["global_unanswerability"])
        self.assertEqual("UNMEASURED", query.json()["false_answer_rate_after_retrieval_selection"])
        context = self.client.post("/answer-context", json={"question": "Which color?", "context": "The red apple"})
        self.assertEqual(200, context.status_code, context.text)
        self.assertEqual("provided_context_only", context.json()["scope"])
        self.assertTrue(context.json()["evidence"]["document_id"].startswith("provided-"))
        self.assertFalse(self.service.lock.locked())

    def test_unknown_fields_missing_question_and_nonobject_body_rejected(self):
        for route in ("/search", "/query", "/answer-context"):
            for body in ({}, {"question": "Which?", "unexpected": 1}, [], None, "question", 7):
                with self.subTest(route=route, body=body):
                    self.assert_rejected_without_inference(
                        self.client.post(route, content=lab.json_bytes(body),
                                         headers={"content-type": "application/json"}), 422)

    def test_question_types_and_character_budget_rejected_before_inference(self):
        for route in ("/search", "/query", "/answer-context"):
            for question in (None, True, 3, [], {}, "", " \t\n", "x" * 513):
                body = {"question": question}
                if route == "/answer-context":
                    body["context"] = "The red apple"
                with self.subTest(route=route, question=repr(question)[:20]):
                    self.assert_rejected_without_inference(self.client.post(route, json=body), 422)

    def test_search_k_is_strict_integer_in_one_through_ten(self):
        for value in (None, True, False, 0, -1, 11, 1.0, "1", [], {}):
            with self.subTest(k=value):
                self.assert_rejected_without_inference(
                    self.client.post("/search", json={"question": "Which?", "k": value}), 422)
        for value in (1, 10):
            self.assertEqual(200, self.client.post("/search", json={"question": "Which?", "k": value}).status_code)

    def test_context_types_missing_context_and_character_budget_rejected(self):
        bodies = [{"question": "Which?"}]
        bodies.extend({"question": "Which?", "context": value}
                      for value in (None, True, 3, [], {}, "", " \n", "x" * 16001))
        for body in bodies:
            with self.subTest(context=repr(body.get("context"))[:20]):
                self.assert_rejected_without_inference(self.client.post("/answer-context", json=body), 422)

    def test_malformed_json_and_invalid_utf8_rejected(self):
        for content in (b"", b"{", b'{"question":}', b'{} {}', b'\xff'):
            with self.subTest(content=content):
                self.assert_rejected_without_inference(
                    self.client.post("/query", content=content, headers={"content-type": "application/json"}), 422)

    def test_deeply_nested_json_is_input_error_not_server_failure(self):
        body = b'{"question":' + b"[" * 1100 + b"0" + b"]" * 1100 + b"}"
        self.assert_rejected_without_inference(
            self.client.post("/query", content=body, headers={"content-type": "application/json"}), 422)

    def test_content_type_required_but_json_charset_accepted(self):
        body = b'{"question":"Which?"}'
        for content_type in (None, "text/plain", "application/x-www-form-urlencoded"):
            headers = {} if content_type is None else {"content-type": content_type}
            with self.subTest(content_type=content_type):
                self.assert_rejected_without_inference(self.client.post("/query", content=body, headers=headers), 415)
        response = self.client.post("/query", content=body, headers={"content-type": "application/json; charset=utf-8"})
        self.assertEqual(200, response.status_code, response.text)

    def test_body_over_64_kib_rejected_and_exact_limit_accepted(self):
        body = b'{"question":"Which?"}'
        headers = {"content-type": "application/json"}
        oversized = body + b" " * (65537 - len(body))
        self.assert_rejected_without_inference(self.client.post("/query", content=oversized, headers=headers), 413)
        exact = body + b" " * (65536 - len(body))
        response = self.client.post("/query", content=exact, headers=headers)
        self.assertEqual(200, response.status_code, response.text)

    def test_any_browser_origin_rejected_before_inference(self):
        for origin in ("https://example.invalid", "http://127.0.0.1:8765", "null", ""):
            for route in ("/search", "/query", "/answer-context"):
                with self.subTest(origin=origin, route=route):
                    self.assert_rejected_without_inference(
                        self.client.post(route, json={"question": "Which?"}, headers={"origin": origin}), 403)

    def test_untrusted_host_rejected(self):
        self.assert_rejected_without_inference(
            self.client.post("/query", json={"question": "Which?"}, headers={"host": "evil.example"}), 400)

    def test_busy_inference_is_429_but_health_remains_available(self):
        self.service.lock.acquire()
        try:
            self.assert_rejected_without_inference(self.client.post("/query", json={"question": "Which?"}), 429)
            self.assertEqual(200, self.client.get("/health").status_code)
            self.assertTrue(self.service.lock.locked())
        finally:
            self.service.lock.release()

    def test_input_error_releases_lock_and_next_valid_call_succeeds(self):
        with mock.patch.object(self.service, "query", side_effect=ValueError("synthetic private detail")):
            response = self.client.post("/query", json={"question": "Which?"})
        self.assertEqual(422, response.status_code)
        self.assertNotIn("synthetic private detail", response.text)
        self.assertFalse(self.service.lock.locked())
        self.assertEqual(200, self.client.post("/query", json={"question": "Which?"}).status_code)

    def test_unexpected_runtime_failure_releases_lock_without_leaking_details(self):
        with mock.patch.object(self.service, "query", side_effect=RuntimeError("synthetic private detail")):
            response = self.client.post("/query", json={"question": "Which?"})
        self.assertEqual(500, response.status_code)
        self.assertNotIn("synthetic private detail", response.text)
        self.assertFalse(self.service.lock.locked())


class HttpVerifierTests(unittest.TestCase):
    """Mock every HTTP/process operation: these are verifier tests, not a live witness."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.run = self.root / "runs" / "synthetic-run"
        self.run.mkdir(parents=True)
        self.freeze_sha = "0" * 64
        self.documents = [{"id": f"doc-{index}", "text": "The red apple", "title": f"Toy {index}",
                           "citation": "synthetic test fixture"} for index in range(3)]
        self.payload = {"documents": self.documents,
                        "evaluation": [{"id": "positive", "context_id": "doc-0", "question": "Which color?",
                                        "answers": ["red"]},
                                       {"id": "negative", "context_id": "doc-1", "question": "Which country?",
                                        "answers": []}]}
        self.result = {"freeze_sha256": self.freeze_sha, "records": {"given_context": [
            {"id": "positive", "answerable": True, "margin": 3, "prediction": "red", "gold": ["red"]},
            {"id": "negative", "answerable": False, "margin": 1, "prediction": "apple", "gold": []}]}}
        for name, value in (("payload.json", self.payload), ("result.json", self.result),
                            ("calibration.json", {"threshold": 2}), ("freeze.json", {"synthetic": True})):
            lab.save_new(self.run / name, value)
        source = Path(__file__).resolve().parents[1] / "operational" / "verify_external_retrieval_http.py"
        self.verifier = types.ModuleType("http_verifier_under_optimization_test")
        self.verifier.__file__ = str(source)
        # This is intentional: operational proof checks must still execute under -O.
        exec(compile(source.read_text(encoding="utf-8"), str(source), "exec", optimize=2), self.verifier.__dict__)
        self.child = mock.Mock()
        self.child.poll.return_value = None

        def terminate():
            self.child.poll.return_value = 0

        self.child.terminate.side_effect = terminate
        self.child.wait.return_value = 0
        spawn_patch = mock.patch.object(self.verifier.subprocess, "Popen", return_value=self.child)
        self.spawn = spawn_patch.start()
        self.addCleanup(spawn_patch.stop)
        socket_patch = mock.patch("socket.socket")
        socket_patch.start()
        self.addCleanup(socket_patch.stop)

    def exchange(self, base, route, payload=None, *, origin=None):
        if route == "/health":
            return 200, {"status": "LOCAL_READY_NOT_PRODUCTION", "scope": lab.SCOPE,
                         "freeze_sha256": self.freeze_sha, "run_id": "synthetic-run", "documents": 3,
                         "training_performed": False}
        if origin:
            return 403, {"detail": "Browser cross-origin access disabled"}
        if len(payload.get("question", "")) > 65536:
            return 413, {"detail": "body too large"}
        if not payload.get("question") or payload.get("k", 1) > 10 or len(payload.get("context", "")) > 16000:
            return 422, {"detail": "invalid input"}
        if route == "/search":
            return 200, [{**doc, "rank": index + 1} for index, doc in enumerate(self.documents)]
        if route == "/query":
            return 200, {"status": "ABSTAIN", "answer": None, "evidence": None,
                         "global_unanswerability": "UNVERIFIED", "passages": self.documents,
                         "scope": "retrieved_passages_only", "run_id": "synthetic-run",
                         "freeze_sha256": self.freeze_sha,
                         "false_answer_rate_after_retrieval_selection": "UNMEASURED"}
        if payload["question"] == "Which country?":
            return 200, {"status": "ABSTAIN", "answer": None, "evidence": None,
                         "scope": "provided_context_only"}
        context = payload["context"]
        return 200, {"status": "ANSWER", "answer": "red", "scope": "provided_context_only",
                     "evidence": {"start": 4, "end": 7,
                                  "document_id": "provided-" + hashlib.sha256(context.encode()).hexdigest(),
                                  "context_sha256": hashlib.sha256(context.encode()).hexdigest()}}

    def test_verified_receipt_requires_valid_health_even_under_optimization(self):
        def invalid_health(base, route, payload=None, *, origin=None):
            status, body = self.exchange(base, route, payload, origin=origin)
            if route == "/health":
                body["status"] = "BROKEN_SYNTHETIC_SERVICE"
            return status, body

        with mock.patch.object(self.verifier, "exchange", side_effect=invalid_health):
            with self.assertRaisesRegex(RuntimeError, "readiness status"):
                self.verifier.verify(self.root, "synthetic-run", 8765)
        self.assertFalse((self.run / "http-verification.json").exists())
        self.child.terminate.assert_called_once()

    def test_result_mutation_during_checks_cannot_receive_verified_receipt(self):
        def mutate_result(base, route, payload=None, *, origin=None):
            if origin:
                (self.run / "result.json").write_bytes(lab.json_bytes({**self.result, "unexpected_mutation": True}))
            return self.exchange(base, route, payload, origin=origin)

        with mock.patch.object(self.verifier, "exchange", side_effect=mutate_result):
            with self.assertRaisesRegex(RuntimeError, "artifacts changed"):
                self.verifier.verify(self.root, "synthetic-run", 8765)
        self.assertFalse((self.run / "http-verification.json").exists())
        self.child.terminate.assert_called_once()

    def test_valid_mocked_contract_reaches_receipt_and_stops_only_its_child(self):
        with mock.patch.object(self.verifier, "exchange", side_effect=self.exchange):
            receipt = self.verifier.verify(self.root, "synthetic-run", 8765)
        self.assertEqual("LIVE_LOOPBACK_HTTP_VERIFIED_SERVER_STOPPED", receipt["status"])
        self.assertEqual("local_replay_wiring_only_not_independent_quality_evidence", receipt["scope"])
        self.assertEqual(10, len(receipt["checks"]))
        self.assertTrue(receipt["child_stopped"])
        self.assertFalse(receipt["production_deployed"])
        self.assertEqual(lab.digest(self.run / "result.json"), receipt["result_sha256"])
        self.assertEqual(receipt, lab.load_json(self.run / "http-verification.json"))
        self.spawn.assert_called_once()
        self.child.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
