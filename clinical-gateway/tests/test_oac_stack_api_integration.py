from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))

import oac_stack_api as api  # noqa: E402
import oac_stack_integration as integration  # noqa: E402
from oac_operational_health import OperationalHealthKernel  # noqa: E402
from oac_stack_integration import ClinicalKernel  # noqa: E402


class _Status:
    def asdict(self) -> dict[str, object]:
        return {
            "transport_id": "test-transport",
            "running": False,
            "clinical_use_authorized": False,
        }


class _Runtime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def start(self, **kwargs: object) -> _Status:
        self.calls.append(dict(kwargs))
        return _Status()


def _bare_handler(path: str = "/") -> tuple[api.OACStackHandler, list[tuple[str, str]], list[int]]:
    handler = object.__new__(api.OACStackHandler)
    handler.path = path
    handler.headers = {}  # type: ignore[assignment]
    handler.wfile = BytesIO()  # type: ignore[assignment]
    headers: list[tuple[str, str]] = []
    statuses: list[int] = []
    handler.send_response = lambda status, _message=None: statuses.append(status)  # type: ignore[method-assign]
    handler.send_header = lambda name, value: headers.append((name, value))  # type: ignore[method-assign]
    handler.end_headers = lambda: None  # type: ignore[method-assign]
    return handler, headers, statuses


class HandlerBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="oac-api-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.state = self.root / "state"
        self.state.mkdir()
        self.original = {
            "api_key": api.OACStackHandler.api_key,
            "allowed_origins": api.OACStackHandler.allowed_origins,
            "data_root": api.OACStackHandler.data_root,
            "default_state_dir": api.OACStackHandler.default_state_dir,
            "default_limit": api.OACStackHandler.default_limit,
            "ui_file": api.OACStackHandler.ui_file,
            "_operational_kernel": api.OACStackHandler._operational_kernel,
        }
        api.OACStackHandler.api_key = "correct-secret"
        api.OACStackHandler.allowed_origins = frozenset({"https://console.example"})
        api.OACStackHandler.data_root = self.root
        api.OACStackHandler.default_state_dir = self.state

    def tearDown(self) -> None:
        for key, value in self.original.items():
            setattr(api.OACStackHandler, key, value)

    def test_bearer_authentication_is_constant_time_admission_boundary(self) -> None:
        handler, _headers, _statuses = _bare_handler()
        self.assertFalse(handler._authorized())
        handler.headers = {"Authorization": "Bearer wrong-secret"}  # type: ignore[assignment]
        self.assertFalse(handler._authorized())
        handler.headers = {"Authorization": "Bearer correct-secret"}  # type: ignore[assignment]
        self.assertTrue(handler._authorized())

        api.OACStackHandler.api_key = ""
        self.assertFalse(handler._authorized())

    def test_private_key_signing_is_not_exposed_by_the_http_api(self) -> None:
        self.assertNotIn("clinical-review-sign", api.API_ALLOWED_COMMANDS)

    def test_omitted_state_uses_nondefault_server_directory(self) -> None:
        configured = self.root / ".runtime" / "site-state"
        api.OACStackHandler.default_state_dir = configured
        handler, _headers, _statuses = _bare_handler()
        self.assertEqual(handler._bounded_state_dir(None), str(configured))
        with self.assertRaisesRegex(ValueError, "override is disabled"):
            handler._bounded_state_dir("state")

    def test_operational_model_endpoint_is_strictly_advisory_and_rejects_identity(self) -> None:
        artifacts = Path(__file__).resolve().parents[1] / "operational-model" / "artifacts"
        api.OACStackHandler._operational_kernel = OperationalHealthKernel(
            artifacts / "model.json",
            artifacts / "model-receipt.json",
        )
        features = {
            "listener_running": 1,
            "tls_enabled": 1,
            "peer_allowlist_configured": 1,
            "queue_utilization": 0.05,
            "consecutive_failures": 0,
            "seconds_since_last_success": 10.0,
            "ledger_integrity_ok": 1,
            "configuration_valid": 1,
        }
        encoded = json.dumps({"features": features}).encode("utf-8")
        handler, _headers, statuses = _bare_handler("/api/operational-health/score")
        handler.headers = {  # type: ignore[assignment]
            "Authorization": "Bearer correct-secret",
            "Content-Length": str(len(encoded)),
        }
        handler.rfile = BytesIO(encoded)  # type: ignore[assignment]
        handler.do_POST()
        payload = json.loads(handler.wfile.getvalue())
        self.assertEqual(statuses, [200])
        self.assertTrue(payload["ok"])
        self.assertTrue(all(value is False for value in payload["advisory"]["authority"].values()))

        encoded = json.dumps({"features": {**features, "patient_id": "SECRET"}}).encode("utf-8")
        rejected, _headers, rejected_statuses = _bare_handler(
            "/api/operational-health/score"
        )
        rejected.headers = {  # type: ignore[assignment]
            "Authorization": "Bearer correct-secret",
            "Content-Length": str(len(encoded)),
        }
        rejected.rfile = BytesIO(encoded)  # type: ignore[assignment]
        rejected.do_POST()
        self.assertEqual(rejected_statuses, [400])
        self.assertNotIn("SECRET", rejected.wfile.getvalue().decode("utf-8"))

    def test_cors_only_reflects_an_explicitly_allowed_origin(self) -> None:
        handler, headers, _statuses = _bare_handler()
        handler.headers = {"Origin": "https://console.example"}  # type: ignore[assignment]
        self.assertTrue(handler._origin_allowed())
        handler._set_cors_headers()
        self.assertIn(("Access-Control-Allow-Origin", "https://console.example"), headers)
        self.assertIn(("Vary", "Origin"), headers)

        denied, denied_headers, _statuses = _bare_handler()
        denied.headers = {"Origin": "https://attacker.example"}  # type: ignore[assignment]
        self.assertFalse(denied._origin_allowed())
        denied._set_cors_headers()
        self.assertNotIn("Access-Control-Allow-Origin", {name for name, _ in denied_headers})

    def test_relative_paths_are_data_root_relative_and_escape_is_denied(self) -> None:
        resolved = api._resolve_bounded_path("bindings/site.json", self.root, "binding_path")
        self.assertEqual(Path(resolved), self.root / "bindings" / "site.json")
        with self.assertRaisesRegex(ValueError, "within data_root"):
            api._resolve_bounded_path(self.root.parent / "outside.json", self.root, "binding_path")

    def test_transport_request_accepts_fixed_binding_or_registry(self) -> None:
        handler, _headers, _statuses = _bare_handler()
        fixed = handler._normalize_transport_start_request(
            {
                "transport_id": "drop-fixed",
                "kind": "file-drop",
                "source_id": "source-1",
                "binding_path": "bindings/fixed.json",
                "config": {
                    "watch_dir": "drop/inbox",
                    "archive_dir": "drop/archive",
                    "quarantine_dir": "drop/quarantine",
                    "tls_certfile": "tls/server.crt",
                    "tls_keyfile": "tls/server.key",
                    "tls_ca_file": "tls/ca.crt",
                },
            }
        )
        self.assertEqual(fixed["binding_path"], self.root / "bindings" / "fixed.json")
        self.assertIsNone(fixed["binding_dir"])
        for key in (
            "watch_dir",
            "archive_dir",
            "quarantine_dir",
            "tls_certfile",
            "tls_keyfile",
            "tls_ca_file",
        ):
            self.assertTrue(Path(fixed["config"][key]).is_relative_to(self.root))
        self.assertNotIn("inbox_dir", fixed["config"])

        registry = handler._normalize_transport_start_request(
            {
                "transport_id": "drop-registry",
                "kind": "file-drop",
                "source_id": "source-1",
                "binding_dir": "bindings/registry",
                "config": {"watch_dir": "drop/inbox"},
            }
        )
        self.assertIsNone(registry["binding_path"])
        self.assertEqual(registry["binding_dir"], self.root / "bindings" / "registry")

    def test_ambiguous_missing_legacy_and_outside_transport_paths_fail_closed(self) -> None:
        handler, _headers, _statuses = _bare_handler()
        base = {
            "transport_id": "drop",
            "kind": "file-drop",
            "source_id": "source-1",
        }
        with self.assertRaisesRegex(ValueError, "binding_path or binding_dir"):
            handler._normalize_transport_start_request(base)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            handler._normalize_transport_start_request(
                {**base, "binding_path": "binding.json", "binding_dir": "bindings"}
            )
        with self.assertRaisesRegex(ValueError, "watch_dir"):
            handler._normalize_transport_start_request(
                {**base, "binding_path": "binding.json", "config": {"inbox_dir": "drop"}}
            )
        with self.assertRaisesRegex(ValueError, "within data_root"):
            handler._normalize_transport_start_request(
                {
                    **base,
                    "binding_path": "binding.json",
                    "config": {"tls_keyfile": self.root.parent / "outside.key"},
                }
            )

    def test_api_projection_redacts_hl7_fhir_and_identity_fields(self) -> None:
        projected = api._sanitize_api_payload(
            {
                "safe": "VISIBLE",
                "diagnostic": "MSH|^~\\&|LIAT\rPID|||PATIENT-SECRET\r",
                "source_subject_token": "PATIENT-SECRET",
                "resource": {
                    "resourceType": "Patient",
                    "name": [{"family": "SECRET-NAME"}],
                },
            }
        )
        encoded = json.dumps(projected)
        self.assertIn("VISIBLE", encoded)
        self.assertNotIn("PATIENT-SECRET", encoded)
        self.assertNotIn("SECRET-NAME", encoded)
        self.assertIn("REDACTED_HL7", encoded)

    def test_ui_routes_serve_only_the_configured_file_with_security_headers(self) -> None:
        ui = self.root / "index.html"
        ui.write_text("<html>safe console</html>", encoding="utf-8")
        api.OACStackHandler.ui_file = ui
        api.OACStackHandler.api_key = ""

        handler, headers, statuses = _bare_handler("/")
        handler.do_GET()
        self.assertEqual(statuses, [200])
        self.assertEqual(handler.wfile.getvalue(), ui.read_bytes())
        header_map = dict(headers)
        self.assertEqual(header_map["X-Frame-Options"], "DENY")
        self.assertIn("frame-ancestors 'none'", header_map["Content-Security-Policy"])

        escaped, _headers, escaped_statuses = _bare_handler("/../index.html")
        escaped.do_GET()
        self.assertEqual(escaped_statuses, [401])
        self.assertNotEqual(escaped.wfile.getvalue(), ui.read_bytes())


class IntegrationNormalizationTests(unittest.TestCase):
    def test_ui_never_overrides_server_state_directory(self) -> None:
        html = (
            Path(__file__).resolve().parents[1] / "frontend" / "index.html"
        ).read_text(encoding="utf-8")
        self.assertIn("Server configured (locked)", html)
        self.assertNotIn("state_dir", html)
        self.assertNotIn("stateQuery", html)

    def test_packaged_ui_wires_only_typed_operational_model_inputs(self) -> None:
        html = (
            Path(__file__).resolve().parents[1] / "frontend" / "index.html"
        ).read_text(encoding="utf-8")
        self.assertIn('id="scoreModel"', html)
        self.assertIn('id="modelQueueUtilization" type="number"', html)
        self.assertIn('/api/operational-health/score', html)
        for forbidden_id in ("patient", "result", "specimen", "order", "hl7", "fhir"):
            self.assertNotIn(f'id="{forbidden_id}', html.lower())

    def test_import_has_no_runtime_filesystem_side_effect(self) -> None:
        module_dir = Path(api.__file__).resolve().parent
        script = (
            "import pathlib, sys; "
            f"sys.path.insert(0, {str(module_dir)!r}); "
            "import oac_stack_api; "
            "raise SystemExit(1 if pathlib.Path('state').exists() else 0)"
        )
        with tempfile.TemporaryDirectory(prefix="oac-import-test-") as temporary:
            completed = subprocess.run(
                [sys.executable, "-I", "-B", "-c", script],
                cwd=temporary,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_capabilities_api_returns_successful_stateless_payload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oac-capabilities-test-") as temporary:
            root = Path(temporary).resolve()
            kernel = ClinicalKernel(root / "state", data_root=root)
            handler, _headers, statuses = _bare_handler("/api/capabilities")
            handler.headers = {"Authorization": "Bearer test-secret"}  # type: ignore[assignment]
            original = {
                "api_key": api.OACStackHandler.api_key,
                "allowed_origins": api.OACStackHandler.allowed_origins,
                "data_root": api.OACStackHandler.data_root,
                "default_state_dir": api.OACStackHandler.default_state_dir,
                "_kernel": api.OACStackHandler._kernel,
            }
            try:
                api.OACStackHandler.api_key = "test-secret"
                api.OACStackHandler.allowed_origins = frozenset()
                api.OACStackHandler.data_root = root
                api.OACStackHandler.default_state_dir = root / "state"
                api.OACStackHandler._kernel = kernel
                handler.do_GET()
            finally:
                for key, value in original.items():
                    setattr(api.OACStackHandler, key, value)

            payload = json.loads(handler.wfile.getvalue())
            self.assertEqual(statuses, [200])
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["payload"]["ok"])
            self.assertEqual(payload["payload"]["operation"], "clinical-capabilities")

    def test_kernel_normalizes_bridge_paths_without_starting_a_socket(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oac-kernel-test-") as temporary:
            root = Path(temporary).resolve()
            state = root / "state"
            kernel = ClinicalKernel(state, data_root=root)
            runtime = _Runtime()
            kernel._runtime_by_state[str(state.resolve())] = runtime

            result = kernel.transport_start(
                transport_id="registry-test",
                kind="file-drop",
                source_id="source-1",
                binding_dir="bindings",
                config={
                    "watch_dir": "drop/inbox",
                    "archive_dir": "drop/archive",
                    "quarantine_dir": "drop/quarantine",
                    "tls_certfile": "tls/server.crt",
                    "tls_keyfile": "tls/server.key",
                    "tls_ca_file": "tls/ca.crt",
                },
            )

            self.assertTrue(result["ok"])
            call = runtime.calls[-1]
            self.assertIsNone(call["binding_path"])
            self.assertEqual(call["state_dir"], state.resolve())
            self.assertEqual(
                call["config"]["binding_dir"],
                str(root / "bindings"),
            )
            self.assertEqual(call["config"]["watch_dir"], str(root / "drop" / "inbox"))

            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                kernel.transport_start(
                    transport_id="ambiguous",
                    kind="file-drop",
                    source_id="source-1",
                    binding_path="binding.json",
                    binding_dir="bindings",
                )
            with self.assertRaisesRegex(ValueError, "within data_root"):
                kernel.transport_start(
                    transport_id="outside",
                    kind="file-drop",
                    source_id="source-1",
                    binding_path=root.parent / "outside.json",
                )

    def test_command_and_dataset_state_paths_remain_confined_and_operational_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oac-dataset-test-") as temporary:
            root = Path(temporary).resolve()
            kernel = ClinicalKernel(root / "state", data_root=root)
            outside = root.parent / "outside-state"
            for operation in (
                lambda: kernel.run("clinical-status", state_dir=outside),
                lambda: kernel.dataset_tail(state_dir=outside),
                lambda: kernel.dataset_summary(state_dir=outside),
            ):
                with self.assertRaisesRegex(ValueError, "within data_root"):
                    operation()

            with mock.patch.object(
                integration,
                "run_owned_command",
                return_value=(
                    {
                        "ok": True,
                        "operation": "clinical-status",
                        "clinical_use_authorized": False,
                        "patient_name": "SECRET^PERSON",
                        "result_value": "SECRET-RESULT",
                        "resource": {
                            "resourceType": "DiagnosticReport",
                            "subject": {"reference": "Patient/SECRET"},
                        },
                    },
                    "{}",
                    0,
                ),
            ):
                result = kernel.run("clinical-status")
            self.assertEqual(result["dataset_schema"], integration.DATASET_SCHEMA)
            self.assertEqual(result["dataset_semantics"], integration.DATASET_SEMANTICS)
            rows = kernel.dataset_tail()
            self.assertEqual(rows[0]["schema"], integration.DATASET_SCHEMA)
            self.assertEqual(rows[0]["dataset_semantics"], integration.DATASET_SEMANTICS)
            serialized = json.dumps(rows[0], sort_keys=True)
            self.assertNotIn("SECRET", serialized)
            self.assertEqual(
                rows[0]["payload"],
                {
                    "clinical_use_authorized": False,
                    "ok": True,
                    "operation": "clinical-status",
                },
            )
            summary = kernel.dataset_summary()
            self.assertEqual(summary["schema"], integration.DATASET_SCHEMA)
            self.assertEqual(summary["dataset_semantics"], integration.DATASET_SEMANTICS)

    def test_callable_main_forwards_packaging_configuration(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oac-main-test-") as temporary:
            root = Path(temporary).resolve()
            ui = root / "console.html"
            ui.write_text("<html></html>", encoding="utf-8")
            with mock.patch.object(api, "run_server") as run_server:
                with mock.patch.dict(
                    api.os.environ,
                    {
                        "OAC_API_KEY": "k" * 32,
                        "OAC_ALLOWED_ORIGINS": "https://console.example",
                    },
                    clear=False,
                ):
                    result = api.main(
                        [
                            "--host",
                            "127.0.0.1",
                            "--port",
                            "8123",
                            "--data-root",
                            str(root),
                            "--state-dir",
                            "state",
                            "--ui-file",
                            str(ui),
                        ]
                    )
            self.assertEqual(result, 0)
            run_server.assert_called_once_with(
                host="127.0.0.1",
                port=8123,
                state_dir=Path("state"),
                data_root=root,
                api_key="k" * 32,
                allowed_origins=frozenset({"https://console.example"}),
                ui_file=ui,
                operational_model_path=None,
                operational_receipt_path=None,
            )

    def test_run_server_discovers_only_the_confined_default_ui_without_opening_socket(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oac-ui-default-test-") as temporary:
            root = Path(temporary).resolve()
            default_ui = root / "clinical-gateway" / "frontend" / "index.html"
            default_ui.parent.mkdir(parents=True)
            default_ui.write_text("<html>default</html>", encoding="utf-8")
            source_artifacts = Path(__file__).resolve().parents[1] / "operational-model" / "artifacts"
            default_artifacts = root / "clinical-gateway" / "operational-model" / "artifacts"
            default_artifacts.mkdir(parents=True)
            for name in ("model.json", "model-receipt.json"):
                (default_artifacts / name).write_bytes((source_artifacts / name).read_bytes())
            fake_server = mock.Mock()
            original = {
                "allowed_origins": api.OACStackHandler.allowed_origins,
                "data_root": api.OACStackHandler.data_root,
                "default_state_dir": api.OACStackHandler.default_state_dir,
                "ui_file": api.OACStackHandler.ui_file,
                "_kernel": api.OACStackHandler._kernel,
                "_operational_kernel": api.OACStackHandler._operational_kernel,
            }
            try:
                with mock.patch.object(api, "ThreadingHTTPServer", return_value=fake_server):
                    api.run_server(
                        port=0,
                        state_dir=Path("state"),
                        data_root=root,
                        api_key="x" * 32,
                        allowed_origins=frozenset(),
                    )
                self.assertEqual(api.OACStackHandler.ui_file, default_ui)
                self.assertEqual(api.OACStackHandler.allowed_origins, frozenset())
                self.assertIsInstance(
                    api.OACStackHandler._operational_kernel,
                    OperationalHealthKernel,
                )
                fake_server.serve_forever.assert_called_once_with()
            finally:
                for key, value in original.items():
                    setattr(api.OACStackHandler, key, value)

    def test_run_server_rejects_missing_or_short_api_key_before_opening_socket(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oac-api-key-test-") as temporary:
            root = Path(temporary).resolve()
            with mock.patch.object(api, "ThreadingHTTPServer") as server_class:
                for candidate in ("", "short-secret", " " * 32):
                    with self.subTest(candidate_length=len(candidate)):
                        with self.assertRaisesRegex(RuntimeError, "at least 32 characters"):
                            api.run_server(
                                state_dir=Path("state"),
                                data_root=root,
                                api_key=candidate,
                            )
                server_class.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
