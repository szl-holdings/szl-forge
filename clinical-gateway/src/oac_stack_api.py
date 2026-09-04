from __future__ import annotations

"""HTTP API for Owned Agent Clinical Control integration."""

import argparse
import hmac
import ipaddress
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

from oac_operational_health import (
    FEATURE_NAMES,
    MODEL_AUTHORITY,
    MODEL_PURPOSE,
    OperationalHealthKernel,
    OperationalModelError,
)
from oac_stack_integration import ClinicalKernel


MAX_BODY_BYTES = 1024 * 1024
API_ALLOWED_COMMANDS = {
    "clinical-capabilities",
    "clinical-export-fhir",
    "clinical-export-verify",
    "clinical-ingest",
    "clinical-init",
    "clinical-ledger-verify",
    "clinical-review-apply",
    "clinical-review-new",
    "clinical-reviewer-add",
    "clinical-self-test",
    "clinical-source-add",
    "clinical-status",
    "clinical-trust-seal",
}
COMMAND_PATH_ARGUMENTS = {
    "artifact",
    "assay_map",
    "authorization_manifest",
    "binding",
    "hl7_file",
    "out",
    "private_key",
    "public_key",
    "request",
    "trusted_reviewer_key",
}
TRANSPORT_PATH_ARGUMENTS = {
    "archive_dir",
    "binding_dir",
    "quarantine_dir",
    "tls_ca_file",
    "tls_certfile",
    "tls_keyfile",
    "watch_dir",
}
API_REDACTED_KEYS = frozenset(
    {
        "address",
        "authorization_bearer",
        "birth_date",
        "date_of_birth",
        "hl7",
        "hl7_file",
        "medical_record_number",
        "mrn",
        "note_text",
        "patient_id",
        "patient_identifier",
        "patient_name",
        "patient_reference",
        "raw",
        "raw_bytes",
        "raw_hl7",
        "raw_message",
        "recipient_id",
        "source_order_token",
        "source_subject_token",
        "specimen_reference",
        "telecom",
    }
)
FHIR_RESOURCE_TYPES = frozenset(
    {
        "Bundle",
        "DiagnosticReport",
        "Immunization",
        "Observation",
        "Patient",
        "ServiceRequest",
        "Specimen",
    }
)
HL7_MARKERS = ("MSH|", "PID|", "OBR|", "OBX|")


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _resolve_bounded_path(value: Any, root: Path, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must not be empty")
    resolved_root = root.resolve(strict=False)
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    candidate = candidate.resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay within data_root") from exc
    return str(candidate)


def _sanitize_api_payload(value: Any, *, key: str | None = None) -> Any:
    """Return a JSON-safe projection that cannot echo raw HL7 or FHIR PHI."""

    normalized_key = key.lower() if isinstance(key, str) else None
    if normalized_key in API_REDACTED_KEYS:
        return "[REDACTED]"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "[REDACTED_BINARY]"
    if isinstance(value, str):
        if any(marker in value for marker in HL7_MARKERS):
            return "[REDACTED_HL7]"
        return value
    if isinstance(value, Mapping):
        resource_type = value.get("resourceType")
        if isinstance(resource_type, str) and resource_type in FHIR_RESOURCE_TYPES:
            return {"resourceType": resource_type, "redacted": True}
        return {
            str(item_key): _sanitize_api_payload(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize_api_payload(item) for item in value]
    return value


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(_sanitize_api_payload(payload), ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler._set_cors_headers()  # type: ignore[attr-defined]
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, body: bytes, status: int = 200) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler._set_cors_headers()  # type: ignore[attr-defined]
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.send_header("X-Frame-Options", "DENY")
    handler.send_header("Referrer-Policy", "no-referrer")
    handler.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    handler.send_header(
        "Content-Security-Policy",
        "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
    )
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    try:
        size = int(handler.headers.get("Content-Length", "0") or 0)
    except ValueError as exc:
        raise ValueError("invalid Content-Length") from exc
    if size <= 0:
        return {}
    if size > MAX_BODY_BYTES:
        raise ValueError(f"request body exceeds {MAX_BODY_BYTES} bytes")
    raw = handler.rfile.read(size).decode("utf-8")
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("JSON payload must be an object")
        return value
    except json.JSONDecodeError:
        raise ValueError("invalid JSON payload")


class OACStackHandler(BaseHTTPRequestHandler):
    default_state_dir: Path = Path("state").resolve()
    data_root: Path = Path.cwd().resolve()
    default_limit: int = 50
    # Constructed explicitly by configure(). Importing this module must not
    # create state directories or other runtime artifacts in the caller's CWD.
    _kernel: ClinicalKernel | None = None
    _operational_kernel: OperationalHealthKernel | None = None
    ui_file: Path | None = None
    api_key: str = os.environ.get("OAC_API_KEY", "")
    allowed_origins: frozenset[str] = frozenset(
        origin.strip()
        for origin in os.environ.get(
            "OAC_ALLOWED_ORIGINS",
            "http://127.0.0.1:8080,http://localhost:8080",
        ).split(",")
        if origin.strip()
    )

    def _set_cors_headers(self) -> None:
        origin = self.headers.get("Origin")
        if origin and origin in OACStackHandler.allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _set_default_headers(self) -> None:
        self._set_cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or origin in OACStackHandler.allowed_origins

    def _authorized(self) -> bool:
        expected = OACStackHandler.api_key
        if not expected:
            return False
        supplied = self.headers.get("Authorization", "")
        prefix = "Bearer "
        return supplied.startswith(prefix) and hmac.compare_digest(
            supplied[len(prefix):],
            expected,
        )

    def _admit_request(self, *, health: bool = False) -> bool:
        if not self._origin_allowed():
            _json_response(self, {"ok": False, "code": "ORIGIN_DENIED"}, status=403)
            return False
        if not health and not self._authorized():
            _json_response(self, {"ok": False, "code": "AUTHENTICATION_REQUIRED"}, status=401)
            return False
        return True

    def _bounded_state_dir(self, value: Any | None) -> str:
        candidate = Path(
            _resolve_bounded_path(
                value or OACStackHandler.default_state_dir,
                OACStackHandler.data_root,
                "state_dir",
            )
        )
        if candidate != OACStackHandler.default_state_dir:
            raise ValueError("API state_dir override is disabled")
        return str(candidate)

    def _bound_command_paths(self, args: dict[str, Any]) -> dict[str, Any]:
        bounded = dict(args)
        for key in COMMAND_PATH_ARGUMENTS:
            if bounded.get(key) is not None:
                bounded[key] = _resolve_bounded_path(
                    bounded[key], OACStackHandler.data_root, key
                )
        return bounded

    def _bound_transport_config(self, config: dict[str, Any]) -> dict[str, Any]:
        bounded = dict(config)
        if "inbox_dir" in bounded:
            raise ValueError("config.inbox_dir is unsupported; use config.watch_dir")
        for key in TRANSPORT_PATH_ARGUMENTS:
            if bounded.get(key) is not None:
                bounded[key] = _resolve_bounded_path(
                    bounded[key], OACStackHandler.data_root, key
                )
        return bounded

    def _normalize_transport_start_request(self, body: Mapping[str, Any]) -> dict[str, Any]:
        transport_id = str(body.get("transport_id", "")).strip()
        kind = str(body.get("kind", "")).strip()
        source_id = str(body.get("source_id", "")).strip()
        if not transport_id or not kind or not source_id:
            raise ValueError("transport_id, kind, and source_id are required")

        raw_config = body.get("config")
        if raw_config is None:
            config: dict[str, Any] = {}
        elif isinstance(raw_config, dict):
            config = dict(raw_config)
        else:
            raise ValueError("config must be an object")

        top_level_binding_dir = body.get("binding_dir")
        configured_binding_dir = config.get("binding_dir")
        if top_level_binding_dir is not None and configured_binding_dir is not None:
            top_level_resolved = _resolve_bounded_path(
                top_level_binding_dir,
                OACStackHandler.data_root,
                "binding_dir",
            )
            configured_resolved = _resolve_bounded_path(
                configured_binding_dir,
                OACStackHandler.data_root,
                "binding_dir",
            )
            if top_level_resolved != configured_resolved:
                raise ValueError("binding_dir is defined more than once with different values")
            config["binding_dir"] = top_level_resolved
        elif top_level_binding_dir is not None:
            config["binding_dir"] = top_level_binding_dir

        bounded_config = self._bound_transport_config(config)
        fixed_binding_value = body.get("binding_path")
        fixed_binding = None
        if fixed_binding_value is not None:
            fixed_binding = _resolve_bounded_path(
                fixed_binding_value,
                OACStackHandler.data_root,
                "binding_path",
            )
        bounded_binding_dir = bounded_config.pop("binding_dir", None)
        if fixed_binding is None and bounded_binding_dir is None:
            raise ValueError("binding_path or binding_dir is required")
        if fixed_binding is not None and bounded_binding_dir is not None:
            raise ValueError("binding_path and binding_dir are mutually exclusive")

        return {
            "transport_id": transport_id,
            "kind": kind,
            "source_id": source_id,
            "binding_path": Path(fixed_binding) if fixed_binding is not None else None,
            "binding_dir": Path(bounded_binding_dir) if bounded_binding_dir is not None else None,
            "config": bounded_config,
            "state_dir": self._bounded_state_dir(body.get("state_dir")),
        }

    def do_OPTIONS(self) -> None:
        if not self._origin_allowed():
            _json_response(self, {"ok": False, "code": "ORIGIN_DENIED"}, status=403)
            return
        self.send_response(204)
        self._set_default_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}

        if path in {"", "/index.html"}:
            if not self._admit_request(health=True):
                return
            ui_file = OACStackHandler.ui_file
            if ui_file is None:
                _json_response(self, {"ok": False, "error": "UI_NOT_CONFIGURED"}, status=404)
                return
            try:
                body = ui_file.read_bytes()
            except OSError:
                _json_response(self, {"ok": False, "error": "UI_UNAVAILABLE"}, status=503)
                return
            _html_response(self, body)
            return

        if path == "/api/health":
            if not self._admit_request(health=True):
                return
            _json_response(
                self,
                {
                    "ok": True,
                    "service": "oac_stack_api",
                    "clinical_use_authorized": False,
                    "real_phi_authorized": False,
                    "site_validated": False,
                    "truth_boundary": "local_gateway_health_not_device_or_clinical_proof",
                },
            )
            return

        if not self._admit_request():
            return

        if path == "/api/model/summary":
            try:
                state_dir = self._bounded_state_dir(query.get("state_dir"))
            except ValueError as exc:
                _json_response(self, {"ok": False, "error": str(exc)}, status=403)
                return
            _json_response(self, self.kernel().dataset_summary(state_dir))
            return

        if path == "/api/operational-health":
            configured = OACStackHandler._operational_kernel is not None
            _json_response(
                self,
                {
                    "ok": True,
                    "configured": configured,
                    "purpose": MODEL_PURPOSE,
                    "authority": MODEL_AUTHORITY,
                    "features": list(FEATURE_NAMES),
                    "truth_boundary": "synthetic_operational_advisory_not_clinical_authority",
                },
            )
            return

        if path == "/api/dataset":
            try:
                state_dir = self._bounded_state_dir(query.get("state_dir"))
                limit = max(1, min(500, int(query.get("limit", str(OACStackHandler.default_limit)))))
            except ValueError as exc:
                _json_response(self, {"ok": False, "error": str(exc)}, status=400)
                return
            _json_response(
                self,
                {
                    "ok": True,
                    "state_dir": state_dir,
                    "rows": self.kernel().dataset_tail(state_dir=state_dir, limit=limit),
                },
            )
            return

        if path == "/api/capabilities":
            kernel = self.kernel()
            payload = kernel.run_with_default("clinical-capabilities")
            payload["api"] = {
                "authentication_configured": bool(OACStackHandler.api_key),
                "data_root": str(OACStackHandler.data_root),
                "loopback_only": True,
            }
            _json_response(self, payload)
            return

        if path == "/api/transports":
            try:
                state_dir = self._bounded_state_dir(query.get("state_dir"))
            except ValueError as exc:
                _json_response(self, {"ok": False, "error": str(exc)}, status=403)
                return
            kernel = self.kernel()
            payload = kernel.transport_list(state_dir=state_dir)
            _json_response(self, payload)
            return

        if path == "/api/transport/status":
            try:
                state_dir = self._bounded_state_dir(query.get("state_dir"))
            except ValueError as exc:
                _json_response(self, {"ok": False, "error": str(exc)}, status=403)
                return
            transport_id = query.get("transport_id")
            kernel = self.kernel()
            payload = kernel.transport_status(transport_id=transport_id, state_dir=state_dir)
            _json_response(self, payload, status=200 if payload.get("ok") else 404)
            return

        _json_response(self, {"ok": False, "error": "unknown GET route"}, status=404)

    def do_POST(self) -> None:
        if not self._admit_request():
            return
        path = urlparse(self.path).path.rstrip("/")
        if path == "/api/operational-health/score":
            try:
                body = _read_json_body(self)
            except ValueError as exc:
                _json_response(self, {"ok": False, "error": str(exc)}, status=400)
                return
            if set(body) != {"features"} or not isinstance(body.get("features"), dict):
                _json_response(
                    self,
                    {"ok": False, "error": "request must contain only a features object"},
                    status=400,
                )
                return
            kernel = OACStackHandler._operational_kernel
            if kernel is None:
                _json_response(
                    self,
                    {"ok": False, "code": "OPERATIONAL_MODEL_NOT_CONFIGURED"},
                    status=503,
                )
                return
            try:
                advisory = kernel.score(body["features"])
            except OperationalModelError as exc:
                _json_response(self, {"ok": False, "error": str(exc)}, status=400)
                return
            _json_response(self, {"ok": True, "advisory": advisory})
            return
        if path == "/api/command":
            try:
                payload = _read_json_body(self)
            except ValueError as exc:
                _json_response(self, {"ok": False, "error": str(exc)}, status=400)
                return

            command = str(payload.get("command", "")).strip()
            args = payload.get("args", {})
            state_override = payload.get("state_dir")
            if not isinstance(args, dict):
                _json_response(self, {"ok": False, "error": "args must be an object"}, status=400)
                return
            if not command:
                _json_response(self, {"ok": False, "error": "command required"}, status=400)
                return
            if command not in API_ALLOWED_COMMANDS:
                _json_response(
                    self,
                    {
                        "ok": False,
                        "code": "COMMAND_NOT_ALLOWED",
                        "error": "the API exposes clinical commands only",
                    },
                    status=403,
                )
                return

            try:
                requested_state = state_override or args.get("state_dir")
                args = self._bound_command_paths(dict(args))
                args["state_dir"] = self._bounded_state_dir(requested_state)
            except ValueError as exc:
                _json_response(self, {"ok": False, "error": str(exc)}, status=403)
                return
            kernel = self.kernel()
            result = kernel.run(command, args)
            _json_response(self, result, status=200 if result.get("ok") else 422)
            return
        if path == "/api/transport/start":
            try:
                body = _read_json_body(self)
            except ValueError as exc:
                _json_response(self, {"ok": False, "error": str(exc)}, status=400)
                return
            try:
                request = self._normalize_transport_start_request(body)
                result = self.kernel().transport_start(
                    transport_id=request["transport_id"],
                    kind=request["kind"],
                    source_id=request["source_id"],
                    binding_path=request["binding_path"],
                    binding_dir=request["binding_dir"],
                    config=request["config"],
                    state_dir=request["state_dir"],
                )
            except (ValueError, FileNotFoundError, RuntimeError) as exc:
                _json_response(self, {"ok": False, "error": str(exc)}, status=422)
                return
            _json_response(self, result, status=200 if result.get("ok") else 422)
            return
        if path == "/api/transport/stop":
            try:
                body = _read_json_body(self)
            except ValueError as exc:
                _json_response(self, {"ok": False, "error": str(exc)}, status=400)
                return
            transport_id = str(body.get("transport_id", "")).strip()
            state_dir = body.get("state_dir")
            if not transport_id:
                _json_response(self, {"ok": False, "error": "transport_id is required"}, status=400)
                return
            try:
                bounded_state = self._bounded_state_dir(state_dir)
                result = self.kernel().transport_stop(
                    transport_id=transport_id,
                    state_dir=bounded_state,
                )
            except ValueError as exc:
                _json_response(self, {"ok": False, "error": str(exc)}, status=403)
                return
            _json_response(self, result, status=200 if result.get("ok") else 404)
            return

        _json_response(self, {"ok": False, "error": "unknown POST route"}, status=404)

    def kernel(self) -> ClinicalKernel:
        kernel = OACStackHandler._kernel
        if kernel is None:
            raise RuntimeError("OAC stack handler is not configured")
        return kernel

    @classmethod
    def configure(
        cls,
        *,
        default_state_dir: Path,
        data_root: Path,
        api_key: str,
        allowed_origins: frozenset[str],
        ui_file: Path | None = None,
        operational_model_path: Path | None = None,
        operational_receipt_path: Path | None = None,
    ) -> None:
        cls.data_root = data_root.resolve()
        cls.default_state_dir = Path(
            _resolve_bounded_path(default_state_dir, cls.data_root, "state_dir")
        )
        cls.api_key = api_key
        cls.allowed_origins = allowed_origins
        if ui_file is None:
            cls.ui_file = None
        else:
            resolved_ui = Path(_resolve_bounded_path(ui_file, cls.data_root, "ui_file"))
            if not resolved_ui.is_file():
                raise ValueError("ui_file must point to an existing file")
            cls.ui_file = resolved_ui
        if (operational_model_path is None) != (operational_receipt_path is None):
            raise ValueError("operational model and receipt must be configured together")
        if operational_model_path is None:
            cls._operational_kernel = None
        else:
            resolved_model = Path(
                _resolve_bounded_path(operational_model_path, cls.data_root, "operational_model")
            )
            resolved_receipt = Path(
                _resolve_bounded_path(
                    operational_receipt_path,
                    cls.data_root,
                    "operational_receipt",
                )
            )
            if not resolved_model.is_file() or not resolved_receipt.is_file():
                raise ValueError("operational model and receipt must be existing files")
            cls._operational_kernel = OperationalHealthKernel(
                resolved_model,
                resolved_receipt,
            )
        cls._kernel = ClinicalKernel(cls.default_state_dir, data_root=cls.data_root)

    def log_message(self, format: str, *args: object) -> None:
        # keep logs compact for service-level observability
        return None


def run_server(
    host: str = "127.0.0.1",
    port: int = 8010,
    state_dir: Path = Path("state"),
    *,
    data_root: Path = Path.cwd(),
    api_key: str = "",
    allowed_origins: frozenset[str] | None = None,
    ui_file: Path | None = None,
    operational_model_path: Path | None = None,
    operational_receipt_path: Path | None = None,
) -> None:
    if not _is_loopback(host):
        raise RuntimeError(
            "the control API is loopback-only; place an authenticated TLS reverse proxy in front"
        )
    if len(api_key) < 32 or not api_key.strip():
        raise RuntimeError(
            "OAC_API_KEY must be configured with at least 32 characters before server startup"
        )
    resolved_data_root = data_root.resolve()
    try:
        resolved_state = Path(
            _resolve_bounded_path(state_dir, resolved_data_root, "state_dir")
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    selected_ui = ui_file
    if selected_ui is None:
        default_ui = resolved_data_root / "clinical-gateway" / "frontend" / "index.html"
        if default_ui.is_file():
            selected_ui = default_ui
    resolved_ui: Path | None = None
    if selected_ui is not None:
        try:
            resolved_ui = Path(
                _resolve_bounded_path(selected_ui, resolved_data_root, "ui_file")
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if not resolved_ui.is_file():
            raise RuntimeError("ui_file must point to an existing file")
    selected_model = operational_model_path
    selected_receipt = operational_receipt_path
    if selected_model is None and selected_receipt is None:
        default_artifacts = (
            resolved_data_root / "clinical-gateway" / "operational-model" / "artifacts"
        )
        candidate_model = default_artifacts / "model.json"
        candidate_receipt = default_artifacts / "model-receipt.json"
        if candidate_model.is_file() or candidate_receipt.is_file():
            if not candidate_model.is_file() or not candidate_receipt.is_file():
                raise RuntimeError("default operational model package is incomplete")
            selected_model = candidate_model
            selected_receipt = candidate_receipt
    if (selected_model is None) != (selected_receipt is None):
        raise RuntimeError("operational model and receipt must be configured together")
    origins = OACStackHandler.allowed_origins if allowed_origins is None else allowed_origins
    OACStackHandler.configure(
        default_state_dir=resolved_state,
        data_root=resolved_data_root,
        api_key=api_key,
        allowed_origins=origins,
        ui_file=resolved_ui,
        operational_model_path=selected_model,
        operational_receipt_path=selected_receipt,
    )
    server = ThreadingHTTPServer((host, port), OACStackHandler)
    server.daemon_threads = True
    print(f"Serving OAC stack API at http://{host}:{port}/api/health")
    server.serve_forever()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Owned Agent Clinical Control stack API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--ui-file")
    parser.add_argument("--operational-model")
    parser.add_argument("--operational-receipt")
    parser.add_argument("--default-limit", type=int, default=50)
    args = parser.parse_args(argv)
    if args.default_limit < 1 or args.default_limit > 500:
        parser.error("--default-limit must be between 1 and 500")

    OACStackHandler.default_limit = args.default_limit
    configured_origins = frozenset(
        origin.strip()
        for origin in os.environ.get(
            "OAC_ALLOWED_ORIGINS",
            "http://127.0.0.1:8080,http://localhost:8080",
        ).split(",")
        if origin.strip()
    )
    run_server(
        host=args.host,
        port=args.port,
        state_dir=Path(args.state_dir),
        data_root=Path(args.data_root),
        api_key=os.environ.get("OAC_API_KEY", ""),
        allowed_origins=configured_origins,
        ui_file=Path(args.ui_file) if args.ui_file else None,
        operational_model_path=(
            Path(args.operational_model) if args.operational_model else None
        ),
        operational_receipt_path=(
            Path(args.operational_receipt) if args.operational_receipt else None
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
