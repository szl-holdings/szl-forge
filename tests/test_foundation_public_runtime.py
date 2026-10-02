"""Public Foundation admission, resource bounds and actual checkpoint execution.

Copyright SZL Holdings LLC. SPDX-License-Identifier: Apache-2.0
Pure ASGI tests run without ML dependencies. The dedicated Linux gate sets
FOUNDATION_REQUIRE_INTEGRATION=1 so real three-checkpoint execution cannot skip.
"""
from __future__ import annotations

import asyncio
import hashlib
import http.client
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from email.message import Message
from urllib.error import HTTPError

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "spaces" / "szl-foundation-confirmation"
SPEC = importlib.util.spec_from_file_location("foundation_public_adapter_tests", SPACE / "app.py")
adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter
SPEC.loader.exec_module(adapter)
WITNESS_SPEC = importlib.util.spec_from_file_location("foundation_witness_tests", ROOT / "foundation-confirmation" / "verify_public_runtime.py")
public_witness = importlib.util.module_from_spec(WITNESS_SPEC)
WITNESS_SPEC.loader.exec_module(public_witness)
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"


def require_materialized_archive(case: unittest.TestCase) -> None:
    """release.zip is an LFS pointer bound to ARCHIVE_SHA256; content needs LFS.

    A checkout without LFS smudge holds only the pointer. Its oid must still be
    the frozen digest; the content assertions then require the dedicated gate
    (FOUNDATION_REQUIRE_INTEGRATION=1, checkout with ``lfs: true``) and may not
    be skipped there.
    """
    head = (SPACE / "release.zip").read_bytes()[:256]
    if not head.startswith(LFS_POINTER_PREFIX):
        return
    if os.getenv("FOUNDATION_REQUIRE_INTEGRATION") == "1":
        case.fail("release.zip is an unmaterialized LFS pointer; check out with lfs: true")
    case.assertIn(b"\noid sha256:" + adapter.ARCHIVE_SHA256.encode() + b"\n", head)
    case.skipTest("LFS pointer bound to ARCHIVE_SHA256; archive bytes not materialized in this checkout")


REQUEST = {"seed": 20260929, "index": 0, "family": "shared_bias", "policy": "learned", "model_seed": 17}


async def request(app, *, method="GET", path="/api/status", value=None, raw=None, headers=None, chunks=None, delays=None, query=b""):
    body = raw if raw is not None else adapter.canonical(value or REQUEST)
    if headers is None:
        headers = [(b"host", adapter.PUBLIC_HOST.encode())]
        if method == "POST":
            headers += [(b"origin", adapter.PUBLIC_ORIGIN.encode()), (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode())]
    blocks = chunks if chunks is not None else [body]
    cursor = 0
    async def receive():
        nonlocal cursor
        if cursor >= len(blocks):
            return {"type": "http.disconnect"}
        index = cursor
        cursor += 1
        if delays:
            await asyncio.sleep(delays[index])
        return {"type": "http.request", "body": blocks[index], "more_body": index < len(blocks) - 1}
    sent = []
    async def send(message):
        sent.append(message)
    await app({"type": "http", "method": method, "path": path, "query_string": query,
               "headers": headers}, receive, send)
    start = sent[0]
    payload = b"".join(message.get("body", b"") for message in sent[1:])
    response_headers = dict(start["headers"])
    parsed = json.loads(payload) if response_headers[b"content-type"].startswith(b"application/json") else payload
    return start["status"], response_headers, parsed, payload


class FixtureRuntime:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.started = threading.Event()
        self.release = None

    def run(self, value):
        self.calls.append(value)
        self.started.set()
        if self.release is not None:
            self.release.wait(timeout=3)
        if self.fail:
            raise ValueError("Fixture failed at /private/operator/path")
        return {"policy": value["policy"], "accuracy": .75, "paid_cost": .05, "net_utility": .7,
                "history": [{"source": "reference", "charged": True}], "fixture": True}, {
                    "checkpoint_sha256": "a" * 64, "model_fingerprint": "b" * 64}


def qualified(runtime):
    benchmark = {"available": True, "verification": {"status": "SOURCE_BOUND_AND_RECOMPUTED", "rows_verified": 5184},
                 "data": {"primary": {"overall_pass": False}}}
    probes = [{"role": "STARTUP_EXECUTION_PROBE", "model_seed": seed} for seed in adapter.MODEL_SEEDS]
    return adapter.QualifiedRuntime(runtime, benchmark, probes)


class ASGIContracts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runtime = FixtureRuntime()
        self.app = adapter.ConfirmationApp(qualifier=lambda: qualified(self.runtime), body_deadline=.06)
        await self.app.startup()

    async def asyncTearDown(self):
        if self.runtime.release is not None:
            self.runtime.release.set()
        await self.app.shutdown()

    async def post(self, **kwargs):
        return await request(self.app, method="POST", path="/api/trial", **kwargs)

    async def test_liveness_and_qualified_readiness_are_distinct(self):
        self.app.state = "STARTING"
        self.assertEqual((await request(self.app, path="/live"))[0], 200)
        self.assertEqual((await request(self.app, path="/readyz"))[0], 503)
        self.assertEqual((await self.post())[0], 503)
        self.assertEqual(self.runtime.calls, [])
        self.app.state = "READY"
        status = (await request(self.app, path="/readyz"))[2]
        self.assertEqual(status["models_loaded"], [17, 23, 41])
        self.assertEqual(len(status["startup_execution_probes"]), 3)
        self.assertFalse(status["startup_probes_are_user_receipts"])
        self.assertEqual(status["scientific_overall_gate"], "FAILED")
        self.assertEqual(self.app.receipts.count(), 0)

    async def test_startup_failure_is_closed_and_does_not_disclose_paths(self):
        def fail():
            raise PermissionError("/private/token/location")
        broken = adapter.ConfirmationApp(qualifier=fail)
        try:
            await broken.startup()
            code, _, result, _ = await request(broken, path="/readyz")
            self.assertEqual(code, 503)
            self.assertEqual(result["state"], "FAILED")
            self.assertNotIn("private", result["error"])
        finally:
            await broken.shutdown()

    async def test_production_host_and_post_origin_are_exact(self):
        good = [(b"host", adapter.PUBLIC_HOST.encode()), (b"origin", adapter.PUBLIC_ORIGIN.encode()),
                (b"content-type", b"application/json")]
        for bad in (
            [(b"host", b"example.invalid")],
            [(b"host", adapter.PUBLIC_HOST.encode()), (b"host", adapter.PUBLIC_HOST.encode())],
            [(b"host", b"127.0.0.1:7860")],
            [row for row in good if row[0] != b"origin"],
            [(b"host", adapter.PUBLIC_HOST.encode()), (b"origin", b"null")],
            [(b"host", adapter.PUBLIC_HOST.encode()), (b"origin", adapter.PUBLIC_ORIGIN.encode()), (b"origin", adapter.PUBLIC_ORIGIN.encode())],
            [(b"host", b"example.invalid"), (b"x-forwarded-host", adapter.PUBLIC_HOST.encode())],
        ):
            with self.subTest(headers=bad):
                self.assertEqual((await self.post(headers=bad))[0], 403)
        self.assertEqual(self.runtime.calls, [])
        self.assertEqual((await request(self.app, path="/live", headers=[(b"host", b"127.0.0.1:7860")]))[0], 200)
        self.app.allow_local = True
        local = [(b"host", b"127.0.0.1:18769"), (b"origin", b"http://127.0.0.1:18769"), (b"content-type", b"application/json")]
        self.assertEqual((await self.post(headers=local))[0], 201)

    async def test_exact_json_schema_and_duplicate_fields(self):
        bad_values = [{**REQUEST, "seed": True}, {**REQUEST, "index": -1}, {**REQUEST, "index": 2**32},
                      {**REQUEST, "model_seed": "17"}, {**REQUEST, "family": "remote"},
                      {**REQUEST, "policy": "train"}, {**REQUEST, "url": "https://example.invalid"}]
        for value in bad_values:
            self.assertEqual((await self.post(value=value))[0], 400)
        for raw in (b'{"seed":1,"seed":2}', b'NaN', b'{"seed":Infinity}', b'[]', b'\xff', b'{', b'[' * 1100 + b'0' + b']' * 1100):
            self.assertEqual((await self.post(raw=raw))[0], 400)
        self.assertEqual(self.runtime.calls, [])

    async def test_body_size_is_measured_even_without_content_length(self):
        headers = [(b"host", adapter.PUBLIC_HOST.encode()), (b"origin", adapter.PUBLIC_ORIGIN.encode()), (b"content-type", b"application/json")]
        self.assertEqual((await self.post(headers=headers, chunks=[b"x" * 3000, b"x" * 1097]))[0], 413)
        self.assertEqual((await self.post(raw=b"x" * 4097))[0], 413)
        valid = adapter.canonical(REQUEST)
        self.assertEqual((await self.post(headers=headers, chunks=[valid[:20], valid[20:]]))[0], 201)
        self.assertEqual(len(self.runtime.calls), 1)

    async def test_one_absolute_body_deadline_rejects_dripped_chunks(self):
        raw = adapter.canonical(REQUEST)
        headers = [(b"host", adapter.PUBLIC_HOST.encode()), (b"origin", adapter.PUBLIC_ORIGIN.encode()), (b"content-type", b"application/json")]
        code = (await self.post(headers=headers, chunks=[raw[:5], raw[5:10], raw[10:]], delays=[.03, .03, .03]))[0]
        self.assertEqual(code, 408)
        self.assertEqual(self.runtime.calls, [])

    async def test_ambiguous_length_media_type_and_partial_body_are_refused(self):
        base = [(b"host", adapter.PUBLIC_HOST.encode()), (b"origin", adapter.PUBLIC_ORIGIN.encode()), (b"content-type", b"application/json")]
        for extra in ([(b"content-length", b"1"), (b"content-length", b"1")],
                      [(b"content-length", b"1"), (b"transfer-encoding", b"chunked")],
                      [(b"content-length", b"-1")], [(b"content-length", b"999")]):
            self.assertEqual((await self.post(headers=base + extra))[0], 400)
        self.assertEqual((await self.post(headers=[(b"host", adapter.PUBLIC_HOST.encode()), (b"origin", adapter.PUBLIC_ORIGIN.encode()), (b"content-type", b"text/plain")]))[0], 415)
        self.assertEqual(self.runtime.calls, [])

    async def test_busy_and_global_rate_do_not_queue_execution(self):
        with self.app.trial_lock:
            response = await self.post()
            self.assertEqual(response[0], 429)
            self.assertIn(b"retry-after", response[1])
        self.assertEqual(self.runtime.calls, [])
        self.assertEqual((await self.post())[0], 201)
        self.assertEqual((await self.post())[0], 201)
        limited = await self.post()
        self.assertEqual(limited[0], 429)
        self.assertEqual(len(self.runtime.calls), 2)
        self.assertFalse(self.app.trial_lock.locked())

    async def test_disconnect_does_not_release_active_execution_lock(self):
        self.runtime.release = threading.Event()
        task = asyncio.create_task(self.post())
        while not self.runtime.started.is_set():
            await asyncio.sleep(.001)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.app.trial_lock.locked())
        self.assertEqual((await self.post())[0], 429)
        self.runtime.release.set()
        for _ in range(100):
            if not self.app.trial_lock.locked():
                break
            await asyncio.sleep(.005)
        self.assertFalse(self.app.trial_lock.locked())
        self.assertEqual(len(self.runtime.calls), 1)

    async def test_receipt_hashes_readback_and_exact_source_identity(self):
        with patch.dict(os.environ, {"SZL_GITHUB_SOURCE_REVISION": "a" * 40}):
            code, _, receipt, raw = await self.post()
        self.assertEqual(code, 201)
        self.assertEqual(receipt["request_sha256"], adapter.digest(REQUEST))
        self.assertEqual(receipt["result_sha256"], adapter.digest(receipt["result"]))
        self.assertEqual(receipt["receipt_sha256"], adapter.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"}))
        self.assertEqual(receipt["source"], {"state": "OBSERVED", "repository": "szl-holdings/szl-forge", "revision": "a" * 40})
        self.assertTrue(receipt["unsigned"])
        self.assertFalse(receipt["authenticity_established"])
        self.assertEqual((await request(self.app, path="/api/trials/" + receipt["id"]))[3], raw)
        self.assertLessEqual(len(raw), adapter.MAX_RECEIPT_BYTES)

    async def test_runtime_failure_is_sanitized_and_lock_released(self):
        self.runtime.fail = True
        code, _, receipt, _ = await self.post()
        self.assertEqual(code, 503)
        self.assertEqual(receipt["status"], "FAILED")
        self.assertNotIn("private", json.dumps(receipt))
        self.assertEqual((await request(self.app, path="/api/trials/" + receipt["id"]))[2], receipt)
        self.assertFalse(self.app.trial_lock.locked())

    async def test_gets_are_observations_without_receipts_or_listing(self):
        for path in ("/live", "/readyz", "/api/status", "/api/results", "/api/build-info", "/", "/app.js", "/style.css"):
            response = await request(self.app, path=path)
            self.assertEqual(response[0], 200)
            self.assertEqual(response[1][b"x-content-type-options"], b"nosniff")
        self.assertEqual(self.app.receipts.count(), 0)
        self.assertEqual(self.runtime.calls, [])
        for path in ("/api/trials", "/state", "/../app.py", "/api/trials/../../app.py", "/release.zip"):
            self.assertEqual((await request(self.app, path=path))[0], 404)
        self.assertEqual((await request(self.app, query=b"url=example.invalid"))[0], 400)
        self.assertEqual((await request(self.app, method="OPTIONS"))[0], 405)

    async def test_build_identity_is_unknown_without_exact_revision(self):
        for revision in ("", "main", "a" * 39, "A" * 40):
            with patch.dict(os.environ, {"SZL_GITHUB_SOURCE_REVISION": revision}):
                result = (await request(self.app, path="/api/build-info"))[2]
                self.assertEqual(result["build"]["state"], "UNKNOWN")
                self.assertIsNone(result["build"]["revision"])
                self.assertFalse(result["receipt_minted"])


class StorageAndAdmission(unittest.TestCase):
    def test_space_card_is_admitted_before_publication(self):
        card = (SPACE / "README.md").read_text(encoding="utf-8")
        self.assertTrue(card.startswith("---\n"))
        metadata = yaml.safe_load(card.split("---", 2)[1])
        self.assertEqual(metadata["sdk"], "docker")
        self.assertIs(type(metadata["app_port"]), int)
        self.assertEqual(metadata["app_port"], 7860)
        description = metadata["short_description"]
        self.assertIsInstance(description, str)
        self.assertGreater(len(description.strip()), 0)
        self.assertLessEqual(len(description), 60,
                             "Hub rejects Space descriptions longer than 60 characters")

    def test_retention_capacity_ttl_and_integrity(self):
        now = [0.0]
        store = adapter.EphemeralReceipts(capacity=2, ttl=10, clock=lambda: now[0])
        receipts = []
        for index in range(3):
            ident = f"{index:032x}"
            receipts.append(store.save({"id": ident, "status": "FAILED", "fixture": index}))
        self.assertEqual(store.count(), 2)
        with self.assertRaises(KeyError):
            store.get(receipts[0]["id"])
        ident = receipts[1]["id"]
        created, raw = store.rows[ident]
        altered = json.loads(raw)
        altered["fixture"] = 999
        store.rows[ident] = (created, adapter.canonical(altered))
        with self.assertRaises(ValueError):
            store.get(ident)
        now[0] = 10
        self.assertEqual(store.count(), 0)
        with self.assertRaises(ValueError):
            adapter.EphemeralReceipts(max_bytes=32).save({"id": "a" * 32, "data": "large"})

    def test_global_rate_refills_and_has_no_ip_keys(self):
        now = [0.0]
        bucket = adapter.TokenBucket(clock=lambda: now[0])
        self.assertTrue(bucket.take()[0])
        self.assertTrue(bucket.take()[0])
        self.assertEqual(bucket.take(), (False, 5))
        now[0] = 5
        self.assertTrue(bucket.take()[0])
        self.assertFalse(bucket.take()[0])

    def test_archive_tamper_refused_before_execution(self):
        with unittest.mock.patch.object(Path, "read_bytes", return_value=b"not a released archive"):
            with self.assertRaisesRegex(ValueError, "archive digest"):
                adapter.release_files(SPACE / "release.zip")

    def test_zip_and_full_manifest_are_pinned_and_unchanged(self):
        require_materialized_archive(self)
        files = adapter.release_files(SPACE / "release.zip")
        self.assertEqual(len(files), 71)
        self.assertEqual(len(json.loads(files["release-manifest.json"])["files"]), 70)
        self.assertIs(json.loads(files["benchmark-summary.json"])["primary"]["overall_pass"], False)
        self.assertEqual(hashlib.sha256((SPACE / "release.zip").read_bytes()).hexdigest(), adapter.ARCHIVE_SHA256)

    def test_ui_keeps_failure_and_ephemeral_boundary_visible(self):
        html = (SPACE / "web" / "index.html").read_text(encoding="utf-8")
        javascript = (SPACE / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn("FAILED", html)
        self.assertIn("Receipts from other visitors are not listed", html)
        self.assertIn("does not establish AGI", html)
        for forbidden in ("innerHTML", "localStorage", "sessionStorage", "document.cookie", "setInterval"):
            self.assertNotIn(forbidden, javascript)


class WitnessFailureEvidence(unittest.TestCase):
    def failure(self, body=b"Service Unavailable"):
        headers = Message()
        headers["Content-Type"] = "text/plain; charset=utf-8"
        headers["Connection"] = "close"
        headers["Authorization"] = "fixture-private-header"
        return HTTPError(adapter.PUBLIC_ORIGIN + "/api/trial", 503, "Service Unavailable", headers, io.BytesIO(body))

    def test_actual_http_failure_body_survives_as_a_terminal_report(self):
        failure = self.failure()
        with patch.object(public_witness, "urlopen", side_effect=failure):
            with self.assertRaises(HTTPError) as raised:
                public_witness.fetch(adapter.PUBLIC_ORIGIN, "/api/trial", REQUEST)
        captured = raised.exception
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "failure.json"
            argv = ["witness", "--origin", adapter.PUBLIC_ORIGIN, "--expected-source", "a" * 40, "--report", str(report)]
            with patch.object(sys, "argv", argv), patch.object(public_witness, "witness", side_effect=captured):
                self.assertEqual(public_witness.main(), 1)
            evidence = json.loads(report.read_bytes())
        self.assertFalse(evidence["ok"])
        detail = evidence["http_failure"]
        self.assertEqual((detail["method"], detail["endpoint"], detail["status"]), ("POST", "/api/trial", 503))
        self.assertEqual(detail["body_utf8"], "Service Unavailable")
        self.assertFalse(detail["body_truncated"])
        self.assertEqual(detail["captured_body_sha256"], hashlib.sha256(b"Service Unavailable").hexdigest())
        self.assertEqual(detail["connection"], "close")
        self.assertNotIn("fixture-private-header", json.dumps(evidence))

    def test_http_error_evidence_is_byte_bounded_and_not_retried(self):
        body = b"x" * 8192
        with patch.object(public_witness, "urlopen", side_effect=self.failure(body)) as transport:
            with self.assertRaises(HTTPError) as raised:
                public_witness.fetch(adapter.PUBLIC_ORIGIN, "/api/trial", REQUEST)
        transport.assert_called_once()
        detail = raised.exception.runtime_http_evidence
        self.assertTrue(detail["body_truncated"])
        self.assertEqual(detail["captured_body_bytes"], 4096)
        self.assertEqual(len(detail["body_utf8"]), 4096)
        self.assertEqual(detail["captured_body_sha256"], hashlib.sha256(body[:4096]).hexdigest())

    def test_incomplete_error_body_preserves_original_http_failure(self):
        class IncompleteBody(io.BytesIO):
            def read(self, size=-1):
                raise http.client.IncompleteRead(b"partial", 1)
        headers = Message()
        headers["Content-Type"] = "text/plain"
        failure = HTTPError(adapter.PUBLIC_ORIGIN + "/api/trial", 503, "Service Unavailable", headers, IncompleteBody())
        with patch.object(public_witness, "urlopen", side_effect=failure):
            with self.assertRaises(HTTPError) as raised:
                public_witness.fetch(adapter.PUBLIC_ORIGIN, "/api/trial", REQUEST)
        self.assertIs(raised.exception, failure)
        detail = failure.runtime_http_evidence
        self.assertEqual(detail["status"], 503)
        self.assertTrue(detail["body_unavailable"])
        self.assertIsNone(detail["captured_body_sha256"])
        self.assertIsNone(detail["body_utf8"])
        self.assertTrue(failure.file.closed)


class TransportIdleAdmission(unittest.TestCase):
    def test_completed_idle_requests_do_not_consume_the_next_trial_slot(self):
        absent = [name for name in ("uvicorn", "h11") if importlib.util.find_spec(name) is None]
        if absent:
            if os.getenv("FOUNDATION_REQUIRE_INTEGRATION") == "1":
                self.fail("Dedicated runtime gate lacks transport dependencies: " + ", ".join(absent))
            self.skipTest("Transport dependencies unavailable in general development environment")
        docker = (SPACE / "Dockerfile").read_text()
        command = json.loads(next(line[4:] for line in docker.splitlines() if line.startswith("CMD ")))
        concurrency = int(command[command.index("--limit-concurrency") + 1])
        keepalive = int(command[command.index("--timeout-keep-alive") + 1])
        self.assertEqual(concurrency, 8)
        self.assertEqual(command[command.index("--workers") + 1], "1")
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        child = '''import uvicorn,sys
async def app(scope,receive,send):
 body=b"transport-control-only"
 await send({"type":"http.response.start","status":201 if scope["method"]=="POST" else 200,"headers":[(b"content-length",str(len(body)).encode())]})
 await send({"type":"http.response.body","body":body})
uvicorn.run(app,host="127.0.0.1",port=int(sys.argv[1]),workers=1,limit_concurrency=int(sys.argv[2]),timeout_keep_alive=int(sys.argv[3]),proxy_headers=False,server_header=False,http="h11",lifespan="off",log_level="warning")
'''
        process = subprocess.Popen([sys.executable, "-B", "-c", child, str(port), str(concurrency), str(keepalive)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        connections, raw_sockets = [], []
        try:
            deadline = time.monotonic() + 8
            while True:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.2):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        self.fail("Owned controlled transport process did not listen")
                    time.sleep(.05)
            time.sleep(.1)

            def exchange(method="GET"):
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                connections.append(connection)
                connection.request(method, "/", headers={"Connection": "keep-alive"})
                response = connection.getresponse()
                raw = response.read(1024)
                return response.status, raw

            for _ in range(7):
                self.assertEqual(exchange()[0], 200)
            self.assertEqual(exchange("POST")[0], 201, "Completed keepalive sockets obstruct the fresh trial")
            for connection in connections:
                connection.close()
            time.sleep(.15)
            # Initial sockets still count: disabling idle response retention must
            # preserve admission against eight concurrent HTTP connections.
            for _ in range(7):
                raw_sockets.append(socket.create_connection(("127.0.0.1", port), timeout=2))
            time.sleep(.1)
            self.assertEqual(exchange(), (503, b"Service Unavailable"))
        finally:
            for connection in connections:
                connection.close()
            for opened in raw_sockets:
                opened.close()
            if process.poll() is None:
                process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)


class ActualCheckpointIntegration(unittest.TestCase):
    def test_three_actual_model_startup_and_public_trial_execution(self):
        required = os.getenv("FOUNDATION_REQUIRE_INTEGRATION") == "1"
        absent = [name for name in ("torch", "numpy", "safetensors") if importlib.util.find_spec(name) is None]
        if absent:
            if required:
                self.fail("Dedicated runtime gate lacks required dependencies: " + ", ".join(absent))
            self.skipTest("ML dependencies unavailable in the general development environment")
        require_materialized_archive(self)
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--actual-integration"],
                                   cwd=str(ROOT), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                                   capture_output=True, text=True, timeout=180)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn('"actual_checkpoint_executions": 3', completed.stdout)
        self.assertIn('"archive_unchanged": true', completed.stdout)
        print(completed.stdout.strip())


async def actual_integration():
    original = (SPACE / "release.zip").read_bytes()
    app = adapter.ConfirmationApp()
    try:
        await app.startup()
        assert app.state == "READY", app.startup_error
        status = (await request(app, path="/readyz"))[2]
        assert status["models_loaded"] == [17, 23, 41]
        assert [row["model_seed"] for row in status["startup_execution_probes"]] == [17, 23, 41]
        assert app.receipts.count() == 0
        frozen_summary = app.qualified.benchmark
        assert frozen_summary["verification"]["rows_verified"] == 5184
        assert frozen_summary["data"]["primary"]["overall_pass"] is False
        seen = []
        for index, seed in enumerate((17, 23, 41)):
            if index == 2:
                await asyncio.sleep(5.1)
            value = {**REQUEST, "seed": 20260929, "index": 8762, "model_seed": seed}
            code, _, receipt, raw = await request(app, method="POST", path="/api/trial", value=value)
            assert code == 201, receipt
            assert receipt["request_sha256"] == adapter.digest(value)
            assert receipt["result_sha256"] == adapter.digest(receipt["result"])
            assert receipt["receipt_sha256"] == adapter.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
            assert (await request(app, path="/api/trials/" + receipt["id"]))[3] == raw
            assert len(receipt["result"]["history"]) <= 8
            assert receipt["result"]["resources_used"] <= 4
            assert receipt["binding"]["checkpoint_sha256"]
            assert receipt["binding"]["model_fingerprint"]
            seen.append(seed)
        assert app.qualified.runtime.admitted()
        assert frozen_summary == (await request(app, path="/api/results"))[2]
        assert (SPACE / "release.zip").read_bytes() == original
        # Alter only this integration subprocess's temporary extracted copy.
        targets = [app.qualified.runtime.root / "core.py", app.qualified.runtime.admitted()["checkpoints"][17]["path"]]
        for target in targets:
            saved = target.read_bytes()
            try:
                target.write_bytes(saved + b"tampered")
                assert app.qualified.runtime.status()["ready"] is False
                try:
                    app.qualified.runtime.run(REQUEST)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Source or checkpoint tamper was admitted")
            finally:
                target.write_bytes(saved)
        assert app.qualified.runtime.status()["ready"] is True
        import torch
        print(json.dumps({"actual_checkpoint_executions": len(seen), "startup_probes": len(status["startup_execution_probes"]),
                          "archive_unchanged": True, "benchmark_overall": "FAILED", "torch_observed": torch.__version__,
                          "device": "cpu", "receipt_hashes_verified": True}, sort_keys=True))
    finally:
        await app.shutdown()


if __name__ == "__main__":
    if sys.argv[1:] == ["--actual-integration"]:
        asyncio.run(actual_integration())
    else:
        unittest.main(verbosity=2)
