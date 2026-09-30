"""Hermetic artifact, refusal, wire-bound, and HTTP service regression tests."""

from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
import marshal
from pathlib import Path
import shutil
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("oac_health_space_app", ROOT / "app.py")
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)
SOURCE = "a" * 40
FEATURES = {
    "listener_running": 1,
    "tls_enabled": 1,
    "peer_allowlist_configured": 1,
    "queue_utilization": 0.1,
    "consecutive_failures": 0,
    "seconds_since_last_success": 1,
    "ledger_integrity_ok": 1,
    "configuration_valid": 1,
}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="szl-oac-http-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "artifacts").mkdir()
        for name in app.ARTIFACTS:
            shutil.copyfile(ROOT / "artifacts" / name, self.root / "artifacts" / name)
        shutil.copyfile(ROOT / "release.json", self.root / "release.json")
        self.html = b'<!doctype html><style>body{color:#123}</style><script>"use strict";</script><h1>Numeric preview</h1>'
        (self.root / "index.html").write_bytes(self.html)

    def application(self, source=SOURCE):
        return app.Application(self.root, source)

    def server(self, application=None, timeout=1, workers=4):
        server = app.create_server(
            ("127.0.0.1", 0),
            application or self.application(),
            request_timeout=timeout,
            workers=workers,
        )
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()

        def close():
            server.shutdown()
            server.server_close()
            thread.join(1)

        self.addCleanup(close)
        return server

    def request(
        self, server, path="/api/score", body=None, method="POST", headers=None
    ):
        connection = http.client.HTTPConnection(*server.server_address, timeout=2)
        self.addCleanup(connection.close)
        if body is None:
            body = app.canonical({"features": FEATURES})
        if headers is None:
            headers = {"Content-Type": "application/json", "X-SZL-Preview": "1"}
        connection.request(
            method, path, body=body if method == "POST" else None, headers=headers
        )
        response = connection.getresponse()
        raw = response.read()
        return response.status, dict(response.getheaders()), raw


class ArtifactTests(Fixture):
    def test_ready_and_fixed_identity(self):
        application = self.application()
        self.assertTrue(application.ready)
        identity = application.identity()
        self.assertEqual(identity["application"]["revision"], SOURCE)
        self.assertEqual(identity["artifacts"], app.ARTIFACTS)
        self.assertFalse(identity["clinical_use_authorized"])
        self.assertFalse(identity["production_promotion_allowed"])

    def test_missing_invalid_source_is_not_ready(self):
        for source in ("", "A" * 40, "a" * 39, "a" * 40 + "\n", "main"):
            with self.subTest(source=source):
                application = self.application(source)
                self.assertFalse(application.ready)
                self.assertTrue(application.artifacts_verified)
                self.assertIsNone(application.identity()["application"]["revision"])
                self.assertEqual(application.failure_code, "SOURCE_BINDING_UNAVAILABLE")

    def test_each_artifact_drift_fails_before_readiness(self):
        for name in app.ARTIFACTS:
            with self.subTest(name=name):
                path = self.root / "artifacts" / name
                raw = path.read_bytes()
                path.write_bytes(raw + b"\n")
                application = self.application()
                self.assertFalse(application.ready)
                self.assertIsNone(application.kernel)
                self.assertIsNone(application.identity()["artifacts"])
                path.write_bytes(raw)

    def test_untrusted_kernel_cannot_execute(self):
        marker = self.root / "must-not-exist"
        (self.root / "artifacts" / "oac_operational_health.py").write_text(
            "from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n",
            encoding="utf-8",
        )
        self.assertFalse(self.application().ready)
        self.assertFalse(marker.exists())

    def test_release_pin_change_and_unknown_field_rejected(self):
        release = json.loads((self.root / "release.json").read_bytes())
        release["hub_model"]["revision"] = "b" * 40
        (self.root / "release.json").write_bytes(app.canonical(release))
        self.assertFalse(self.application().ready)
        release["hub_model"]["revision"] = app.MODEL_REVISION
        release["extra"] = True
        (self.root / "release.json").write_bytes(app.canonical(release))
        self.assertFalse(self.application().ready)

    def test_missing_artifact_is_unavailable(self):
        (self.root / "artifacts" / "model.json").unlink()
        self.assertFalse(self.application().ready)

    def test_oversized_release_is_bounded_and_unavailable(self):
        (self.root / "release.json").write_bytes(b" " * 16385)
        self.assertFalse(self.application().ready)

    def test_kernel_uses_verified_bytes_not_reopened_paths(self):
        baseline = self.application()
        changed = json.loads((self.root / "artifacts" / "model.json").read_bytes())
        changed["intercept"] = 10.0
        changed_bytes = app.canonical(changed)
        receipt = json.loads(
            (self.root / "artifacts" / "artifact_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        receipt["model_sha256"] = hashlib.sha256(changed_bytes).hexdigest()
        original_bytes, original_text = Path.read_bytes, Path.read_text
        reopened = []

        def raced_bytes(path):
            reopened.append(path.name)
            return changed_bytes if path.name == "model.json" else original_bytes(path)

        def raced_text(path, *args, **kwargs):
            reopened.append(path.name)
            return (
                json.dumps(receipt)
                if path.name == "artifact_receipt.json"
                else original_text(path, *args, **kwargs)
            )

        # bounded_file(open/read) still sees the pinned originals. These mocks
        # reproduce a different model+matching receipt on any subsequent path
        # reopen without modifying the canonical artifacts or filesystem.
        with (
            mock.patch.object(Path, "read_bytes", raced_bytes),
            mock.patch.object(Path, "read_text", raced_text),
        ):
            canonical_kernel = type(baseline.kernel).__mro__[1]
            unadapted = canonical_kernel(
                self.root / "artifacts" / "model.json",
                self.root / "artifacts" / "artifact_receipt.json",
            )
            self.assertEqual(reopened, ["model.json", "artifact_receipt.json"])
            self.assertNotEqual(
                unadapted.score(FEATURES), baseline.kernel.score(FEATURES)
            )
            reopened.clear()
            raced = self.application()
            self.assertEqual(reopened, [])
        self.assertTrue(raced.ready)
        self.assertEqual(raced.identity()["artifacts"], app.ARTIFACTS)
        self.assertEqual(raced.kernel.score(FEATURES), baseline.kernel.score(FEATURES))
        self.assertEqual(
            hashlib.sha256(raced.kernel.model_path.read_bytes()).hexdigest(),
            app.ARTIFACTS["model.json"],
        )
        self.assertEqual(
            hashlib.sha256(raced.kernel.receipt_path.read_bytes()).hexdigest(),
            app.ARTIFACTS["artifact_receipt.json"],
        )
        with self.assertRaises(app.InvalidInput):
            raced.kernel.model_path = Path("another-model.json")
        with self.assertRaises(app.InvalidInput):
            raced.kernel.receipt_path = Path("another-receipt.json")
        with self.assertRaises(AttributeError):
            raced.kernel.model_path.raw = changed_bytes

    def test_unverified_python_bytecode_is_never_loaded(self):
        kernel_path = self.root / "artifacts" / "oac_operational_health.py"
        marker = self.root / "bytecode-must-not-execute"
        code = compile(
            "from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n",
            str(kernel_path),
            "exec",
        )
        cached = Path(importlib.util.cache_from_source(str(kernel_path)))
        cached.parent.mkdir()
        stat = kernel_path.stat()
        header = importlib.util.MAGIC_NUMBER + struct.pack(
            "<III", 0, int(stat.st_mtime) & 0xFFFFFFFF, stat.st_size
        )
        cached.write_bytes(header + marshal.dumps(code))
        module_name = (
            "_szl_oac_health_" + app.ARTIFACTS["oac_operational_health.py"][:16]
        )
        with mock.patch.dict(sys.modules):
            sys.modules.pop(module_name, None)
            self.assertTrue(self.application().ready)
        self.assertFalse(marker.exists())


class ScoreTests(Fixture):
    def test_hashes_and_canonical_kernel_output(self):
        application = self.application()
        result = application.score(app.canonical({"features": FEATURES}))
        self.assertEqual(result["advisory"], application.kernel.score(FEATURES))
        self.assertEqual(
            result["input_sha256"], hashlib.sha256(app.canonical(FEATURES)).hexdigest()
        )
        self.assertEqual(
            result["output_sha256"],
            hashlib.sha256(app.canonical(result["advisory"])).hexdigest(),
        )
        self.assertFalse(result["receipt_minted"])
        self.assertTrue(
            all(value is False for value in result["advisory"]["authority"].values())
        )

    def test_closed_and_numeric_only_contract(self):
        cases = [
            [],
            {"features": FEATURES, "extra": 1},
            {"features": {**FEATURES, "patient_id": "DO-NOT-ECHO"}},
            {"features": {**FEATURES, "queue_utilization": "MSH|PRIVATE"}},
            {"features": {**FEATURES, "queue_utilization": 2}},
            {"features": {**FEATURES, "listener_running": 1.0}},
            {"features": {**FEATURES, "configuration_valid": None}},
            {
                "features": {
                    name: value
                    for name, value in FEATURES.items()
                    if name != "tls_enabled"
                }
            },
        ]
        application = self.application()
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(app.InvalidInput):
                    application.score(app.canonical(case))

    def test_strict_json_refusals(self):
        cases = [
            b'{"features":{},"features":{}}',
            b'{"n":NaN}',
            b'{"n":Infinity}',
            b'{"n":1e999}',
            b'{"n":' + b"9" * 25 + b"}",
            b'{"n":"\\ud800"}',
            b"\xff",
            b"[" * 10 + b"0" + b"]" * 10,
            b"x" * 4097,
        ]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(app.InvalidInput):
                    app.strict_json(case)


class HTTPTests(Fixture):
    def test_score_and_identity(self):
        server = self.server()
        status, headers, body = self.request(server)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["identity"]["state"], "READY")
        self.assertEqual(headers["Connection"], "close")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_live_ready_build_and_identity(self):
        server = self.server()
        for path in ("/healthz", "/readyz", "/api/build-info", "/api/v1/identity"):
            with self.subTest(path=path):
                status, _, body = self.request(server, path=path, method="GET")
                self.assertEqual(status, 200)
                result = json.loads(body)
                if path == "/api/build-info":
                    self.assertEqual(result["build"]["revision"], SOURCE)
                    self.assertFalse(result["receipt_minted"])

    def test_binding_absence_liveness_but_no_ready_or_score(self):
        server = self.server(self.application(""))
        self.assertEqual(self.request(server, path="/healthz", method="GET")[0], 200)
        self.assertEqual(self.request(server, path="/readyz", method="GET")[0], 503)
        self.assertEqual(self.request(server)[0], 503)

    def test_preview_header_required(self):
        status, _, body = self.request(
            self.server(), headers={"Content-Type": "application/json"}
        )
        self.assertEqual(status, 403)
        self.assertEqual(json.loads(body)["error"], "PREVIEW_HEADER_REQUIRED")

    def test_refusal_never_echoes_sensitive_input(self):
        body = app.canonical(
            {"features": {**FEATURES, "patient_id": "SENSITIVE-SENTINEL"}}
        )
        status, _, result = self.request(self.server(), body=body)
        self.assertEqual(status, 400)
        self.assertNotIn(b"SENSITIVE-SENTINEL", result)
        self.assertNotIn(b"patient_id", result)

    def test_oversize_content_type_duplicate_and_unknown_route(self):
        server = self.server()
        self.assertEqual(self.request(server, body=b"x" * 4097)[0], 413)
        self.assertEqual(
            self.request(
                server, headers={"X-SZL-Preview": "1", "Content-Type": "text/plain"}
            )[0],
            415,
        )
        self.assertEqual(
            self.request(server, body=b'{"features":{},"features":{}}')[0], 400
        )
        self.assertEqual(self.request(server, path="/api/score?extra=1")[0], 404)
        self.assertEqual(
            self.request(server, path="/api/score", method="OPTIONS")[0], 405
        )

    def test_ui_inline_csp_and_embed_origins(self):
        status, headers, body = self.request(self.server(), path="/", method="GET")
        self.assertEqual(status, 200)
        self.assertEqual(body, self.html)
        policy = headers["Content-Security-Policy"]
        self.assertEqual(policy, app.content_security_policy(self.html))
        self.assertIn("https://a-11-oy.com https://a11oy.net", policy)
        self.assertNotIn("unsafe-inline", policy)
        self.assertNotIn("X-Frame-Options", headers)

    def test_csp_hashes_every_case_variant_block(self):
        html = b'<SCRIPT>one</SCRIPT ><script type="text/javascript">two</script><STYLE>three</STYLE>'
        policy = app.content_security_policy(html)
        self.assertEqual(policy.count("sha256-"), 3)

    def test_duplicate_length_and_chunked_framing_rejected(self):
        server = self.server()
        for framing in (
            b"Content-Length: 2\r\nContent-Length: 2\r\n",
            b"Content-Length: 2\r\nTransfer-Encoding: chunked\r\n",
        ):
            with self.subTest(framing=framing):
                with socket.create_connection(server.server_address, timeout=2) as peer:
                    peer.sendall(
                        b"POST /api/score HTTP/1.1\r\nHost: localhost\r\nX-SZL-Preview: 1\r\nContent-Type: application/json\r\n"
                        + framing
                        + b"\r\n{}"
                    )
                    response = peer.recv(4096)
                    self.assertNotIn(b" 200 ", response)
                    self.assertTrue(b" 411 " in response or b" 400 " in response)

    def test_absolute_slow_header_and_body_deadline(self):
        server = self.server(timeout=0.25)
        for prefix in (
            b"GET /healthz HTTP/1.1\r\nX-Slow: ",
            b"POST /api/score HTTP/1.1\r\nHost: localhost\r\nX-SZL-Preview: 1\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n",
        ):
            with self.subTest(prefix=prefix):
                with socket.create_connection(server.server_address, timeout=1) as peer:
                    peer.sendall(prefix)
                    started = time.monotonic()
                    while time.monotonic() - started < 0.65:
                        try:
                            peer.sendall(b"x")
                        except OSError:
                            break
                        time.sleep(0.035)
                    peer.settimeout(0.3)
                    try:
                        response = peer.recv(4096)
                    except (ConnectionResetError, ConnectionAbortedError):
                        response = b""
                    self.assertLess(time.monotonic() - started, 1.0)
                    self.assertNotIn(b" 200 ", response)
                    if response:
                        self.assertIn(b" 408 ", response)

    def test_worker_limit_and_fast_busy_refusal(self):
        server = self.server(timeout=5, workers=4)
        peers = [
            socket.create_connection(server.server_address, timeout=1) for _ in range(4)
        ]
        for peer in peers:
            self.addCleanup(peer.close)
            peer.sendall(b"GET /healthz HTTP/1.1\r\nX-Slow: ")
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            with server.lock:
                if len(server.connections) == 4:
                    break
            time.sleep(0.005)
        with server.lock:
            self.assertEqual(len(server.connections), 4)
        with socket.create_connection(server.server_address, timeout=1) as extra:
            extra.sendall(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n")
            self.assertIn(b" 503 ", extra.recv(4096))
        self.assertEqual(server.max_active_observed, 4)

    def test_total_wire_size_is_bounded(self):
        server = self.server()
        with socket.create_connection(server.server_address, timeout=2) as peer:
            peer.sendall(
                b"GET /healthz HTTP/1.1\r\nX-Huge: " + b"x" * 17000 + b"\r\n\r\n"
            )
            self.assertIn(b" 431 ", peer.recv(4096))

    def test_server_close_releases_incomplete_connections(self):
        server = self.server(timeout=1)
        peer = socket.create_connection(server.server_address, timeout=1)
        self.addCleanup(peer.close)
        peer.sendall(b"GET /healthz HTTP/1.1\r\nX-Slow: ")
        time.sleep(0.03)
        server.shutdown()
        server.server_close()
        try:
            closed = peer.recv(4096)
        except (ConnectionResetError, ConnectionAbortedError):
            closed = b""
        self.assertEqual(closed, b"")

    def test_parser_refusals_do_not_reflect_request_line_or_method(self):
        server = self.server()
        for request in (
            b"PRIVATE-SENTINEL / HTTP/1.1\r\nHost: localhost\r\n\r\n",
            b"GET /PRIVATE-SENTINEL HTTP/bad\r\n\r\n",
        ):
            with self.subTest(request=request):
                with socket.create_connection(server.server_address, timeout=2) as peer:
                    peer.sendall(request)
                    response = bytearray()
                    while True:
                        chunk = peer.recv(4096)
                        if not chunk:
                            break
                        response.extend(chunk)
                    self.assertNotIn(b"PRIVATE-SENTINEL", response)
                    self.assertIn(b"INVALID_HTTP_REQUEST", response)


if __name__ == "__main__":
    unittest.main()
