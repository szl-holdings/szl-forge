"""A local adapter filename cannot qualify the unimplemented Chaski server."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

def load_server(filename):
    source = Path(__file__).resolve().parents[1] / "chaski" / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChaskiReadinessTests(unittest.TestCase):
    server_module = load_server("serve_chaski.py")
    local_error = "NOT_PINNED"
    expected_pins = {"serve_pin": False, "inference_lab_pin": False}

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.adapter = Path(self.directory.name) / "adapter"
        self.adapter_patch = patch.object(self.server_module, "ADAPTER_DIR", self.adapter)
        self.adapter_patch.start()
        self.addCleanup(self.adapter_patch.stop)

    def add_unqualified_adapter(self) -> None:
        self.adapter.mkdir()
        # Deliberately invalid SAMPLE bytes; these must never be loaded.
        (self.adapter / "adapter_model.safetensors").write_bytes(b"SAMPLE fixture")

    def test_missing_adapter_is_unavailable(self) -> None:
        state = self.server_module.availability()
        self.assertEqual(state["status"], "UNAVAILABLE")
        self.assertEqual(state["adapter"], "UNAVAILABLE")

    def test_local_adapter_presence_does_not_make_runtime_ready(self) -> None:
        self.add_unqualified_adapter()
        state = self.server_module.availability()
        self.assertEqual(state["adapter"], "LOCAL")
        self.assertEqual(state["weights"], "LOCAL")
        self.assertEqual(state["status"], "UNAVAILABLE")
        for key, value in self.expected_pins.items():
            self.assertEqual(state[key], value)

    def test_check_reports_unavailable_with_local_adapter(self) -> None:
        self.add_unqualified_adapter()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(self.server_module.check(), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "UNAVAILABLE")

    def test_http_routes_preserve_failure_and_report_no_readiness(self) -> None:
        httpd = self.server_module.ThreadingHTTPServer(("127.0.0.1", 0), self.server_module.Handler)
        worker = threading.Thread(target=httpd.serve_forever, daemon=True)
        worker.start()
        try:
            origin = f"http://127.0.0.1:{httpd.server_port}"
            for present, expected_code, expected_error in (
                (False, 503, "UNAVAILABLE"),
                (True, 501, self.local_error),
            ):
                if present:
                    self.add_unqualified_adapter()
                with self.subTest(adapter_present=present):
                    for route in ("/", "/health", "/v1/models"):
                        with urlopen(origin + route, timeout=5) as response:
                            self.assertEqual(response.status, 200)
                            self.assertEqual(json.load(response)["status"], "UNAVAILABLE")
                    request = Request(origin + "/generate", data=b"{}", method="POST")
                    with self.assertRaises(HTTPError) as raised:
                        urlopen(request, timeout=5)
                    with raised.exception as response:
                        self.assertEqual(response.code, expected_code)
                        self.assertEqual(json.load(response)["error"], expected_error)
        finally:
            httpd.shutdown()
            worker.join(timeout=5)
            httpd.server_close()


class Chaski5050ReadinessTests(ChaskiReadinessTests):
    server_module = load_server("serve_chaski_5050.py")
    local_error = "NOT_PINNED_TO_LIVE"
    expected_pins = {
        "serve_pin": "SZLHOLDINGS/chaski-5050",
        "live_chaski_pin": False,
        "khipu_lab_pin": False,
    }


if __name__ == "__main__":
    unittest.main()
