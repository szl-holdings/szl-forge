"""Bounded, stateless HTTP preview of the fixed synthetic OAC health model.

This process never ingests results, connects to a device, writes a receipt,
trains weights, or grants clinical authority. Artifact hashes are integrity
controls; they are not signatures or an independent trust root.
"""

from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import socketserver
import sys
import threading
import time
from typing import Any, NamedTuple


ROOT = Path(__file__).resolve().parent
SOURCE_REPOSITORY = "szl-holdings/szl-forge"
ARTIFACT_REVISION = "48cb7630dc5c6982cf14755da72d3d07cd45740c"
MODEL_ID = "SZLHOLDINGS/oac-system-health-v1"
MODEL_REVISION = "dd7d109813abcd90c5250106dc2eabd2804e7ac3"
DATASET_ID = "SZLHOLDINGS/oac-clinical-transport-observability-synthetic"
DATASET_REVISION = "f7ab6170bf78138b187b8cb707d374a25ad85375"
ARTIFACTS = {
    "oac_operational_health.py": "2fd8ffd3fd3ca8d0f960d4e0d471593b86e6cccb03607116390cbe9300992669",
    "model.json": "f111b7fc65db561c80763b25e538366a19bed18526ca881915777487935d9557",
    "artifact_receipt.json": "23239bb39b0a5b6db1e41284e6372f352cee61d7938da1e84472246dcdaf95a1",
}
V2_ARTIFACT_REVISION = "56a00821858825f529c40c7322c2f1584608d6e5"
V2_MODEL_ID = "SZLHOLDINGS/oac-ops-health-v2"
V2_MODEL_REVISION = "ff107198aa257ce1bb1841377d553bee3f90be41"
V2_ARTIFACTS = {
    "ops_health.py": "b0a64ff3f26ea284b0588de351ed7795a6089203b35fded0e7d82cbd4871aed9",
    "model.json": "b830a5edca271d667ab09b378dd5d3ab505d71a7e3451de8d9ca2bacfeed977c",
    "artifact_receipt.json": "442486a3b451f0ad765aac253830cdc5455bad3a34186b4cf73388cf207056f2",
    "example_input.json": "afa7c8e5081c682877ad489259592951f854ba2558d409fdcf071ba845d4fb63",
}
V2_AUTHORITY = {
    "acknowledgement": False,
    "clinical_decision": False,
    "device_control": False,
    "result_interpretation": False,
    "result_release": False,
}
FEATURES = frozenset(
    {
        "listener_running",
        "tls_enabled",
        "peer_allowlist_configured",
        "queue_utilization",
        "consecutive_failures",
        "seconds_since_last_success",
        "ledger_integrity_ok",
        "configuration_valid",
    }
)
MAX_BODY_BYTES = 4096
MAX_REQUEST_BYTES = 16384
REQUEST_TIMEOUT_SECONDS = 5.0
MAX_WORKERS = 4
MAX_JSON_DEPTH = 8
SHA = re.compile(r"[0-9a-f]{40}\Z")


class InvalidInput(ValueError):
    """Public input refusal; details are deliberately not reflected."""


class RequestLimit(ValueError):
    """The bounded wire request exceeded its allowance."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def bounded_file(path: Path, maximum: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise InvalidInput("file kind")
    with path.open("rb") as source:
        raw = source.read(maximum + 1)
    if len(raw) > maximum:
        raise InvalidInput("file size")
    return raw


class _VerifiedSnapshot(NamedTuple):
    """Immutable, bounded bytes with only the canonical kernel's read API."""

    name: str
    raw: bytes

    def read_bytes(self) -> bytes:
        return self.raw

    def read_text(self, encoding="utf-8", errors="strict") -> str:
        if encoding != "utf-8" or errors != "strict":
            raise InvalidInput("unsupported snapshot decoding")
        return self.raw.decode("utf-8", errors="strict")


def kernel_from_snapshots(kernel_type, model_bytes: bytes, receipt_bytes: bytes):
    """Run the unchanged canonical constructor against verified memory only.

    The pinned constructor assigns resolved Path objects, then calls
    self.model_path.read_bytes() and self.receipt_path.read_text(). These
    properties bind those reads to immutable snapshots instead of reopening
    mutable paths. Constructor validation and scoring math stay canonical.
    The class is private, accepts no request-controlled paths, and locks the
    adapter after initialization. There are no temporary files or global Path
    monkeypatches. A future kernel pin needs a renewed adapter contract review.
    """

    class SnapshotKernel(kernel_type):
        def __init__(self):
            self._model_snapshot = _VerifiedSnapshot("model.json", bytes(model_bytes))
            self._receipt_snapshot = _VerifiedSnapshot(
                "artifact_receipt.json", bytes(receipt_bytes)
            )
            self._snapshot_paths_locked = False
            super().__init__("model.json", "artifact_receipt.json")
            self._snapshot_paths_locked = True

        @property
        def model_path(self):
            return self._model_snapshot

        @model_path.setter
        def model_path(self, value):
            if self._snapshot_paths_locked or value.name != self._model_snapshot.name:
                raise InvalidInput("snapshot model path is fixed")

        @property
        def receipt_path(self):
            return self._receipt_snapshot

        @receipt_path.setter
        def receipt_path(self, value):
            if self._snapshot_paths_locked or value.name != self._receipt_snapshot.name:
                raise InvalidInput("snapshot receipt path is fixed")

    return SnapshotKernel()


def v2_scorer_from_verified_bytes(module, source: bytes, model: bytes, receipt: bytes):
    """Apply the pinned v2 constructor's checks to one immutable byte snapshot.

    Unlike v1, v2's OpsHealthKernel resolves and reopens local Paths. Its
    canonical receipt validator and ArtifactScorer are used here after the
    same kernel/model/generator checks, so neither model nor receipt can race
    a second disk read. The scoring math is unchanged canonical code.
    """
    record = module.validate_receipt(
        module.loads_strict(receipt.decode("utf-8"), module.ModelArtifactError)
    )
    if hashlib.sha256(source).hexdigest() != record["kernel_sha256"]:
        raise InvalidInput("v2 kernel receipt hash")
    if hashlib.sha256(model).hexdigest() != record["model_sha256"]:
        raise InvalidInput("v2 model receipt hash")
    artifact = module.loads_strict(
        model.decode("utf-8"), module.ModelArtifactError
    )
    scorer = module.ArtifactScorer(artifact)
    if scorer._artifact["generator"]["source_sha256"] != record["generator_sha256"]:
        raise InvalidInput("v2 generator receipt hash")
    return scorer


def strict_json(raw: bytes, maximum: int = MAX_BODY_BYTES) -> Any:
    if len(raw) > maximum:
        raise InvalidInput("size")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InvalidInput("duplicate")
            result[key] = value
        return result

    def integer(value):
        if len(value.lstrip("-")) > 24:
            raise InvalidInput("integer")
        return int(value)

    def number(value):
        if len(value) > 128:
            raise InvalidInput("number")
        result = float(value)
        if not math.isfinite(result):
            raise InvalidInput("number")
        return result

    def constant(_):
        raise InvalidInput("constant")

    def inspect(value, depth=0):
        if depth > MAX_JSON_DEPTH:
            raise InvalidInput("depth")
        if isinstance(value, dict):
            for key, child in value.items():
                if len(key) > 128:
                    raise InvalidInput("key")
                key.encode("utf-8", errors="strict")
                inspect(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                inspect(child, depth + 1)
        elif isinstance(value, str):
            value.encode("utf-8", errors="strict")

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_int=integer,
            parse_float=number,
            parse_constant=constant,
        )
        inspect(value)
        return value
    except (ValueError, UnicodeError, RecursionError, OverflowError) as exc:
        raise InvalidInput("invalid JSON") from exc


class Application:
    """Independent v1 and opt-in v2 fixed models; no network or persistence."""

    def __init__(self, root: Path = ROOT, source_revision: str | None = None):
        self.root = Path(root)
        candidate = (
            os.environ.get("SZL_GITHUB_SOURCE_REVISION", "")
            if source_revision is None
            else source_revision
        )
        self.source_revision = candidate if SHA.fullmatch(candidate) else None
        self.release = None
        self.kernel = None
        self.artifacts_verified = False
        try:
            release = strict_json(
                bounded_file(self.root / "release.json", 16384), 16384
            )
            expected = {
                "schema": "szl.oac-health-space-release/v1",
                "artifact_source": {
                    "repository": SOURCE_REPOSITORY,
                    "revision": ARTIFACT_REVISION,
                    "directory": "clinical-gateway/huggingface/model/oac-system-health-v1",
                },
                "hub_model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
                "hub_dataset": {"repo_id": DATASET_ID, "revision": DATASET_REVISION},
                "artifacts": ARTIFACTS,
            }
            if release != expected:
                raise InvalidInput("release mismatch")
            paths = {name: self.root / "artifacts" / name for name in ARTIFACTS}
            if (self.root / "artifacts").is_symlink():
                raise InvalidInput("artifact directory")
            verified_bytes = {}
            for name, path in paths.items():
                raw = bounded_file(path, 131072)
                if hashlib.sha256(raw).hexdigest() != ARTIFACTS[name]:
                    raise InvalidInput("artifact digest")
                verified_bytes[name] = raw
            receipt = strict_json(verified_bytes["artifact_receipt.json"], 16384)
            if (
                receipt.get("kernel_sha256") != ARTIFACTS["oac_operational_health.py"]
                or receipt.get("model_sha256") != ARTIFACTS["model.json"]
            ):
                raise InvalidInput("receipt binding")
            # The module is verified before any of its Python can execute. A
            # stable module name permits dataclasses to resolve its namespace.
            module_name = (
                "_szl_oac_health_" + ARTIFACTS["oac_operational_health.py"][:16]
            )
            module = sys.modules.get(module_name)
            if module is None:
                spec = importlib.util.spec_from_file_location(
                    module_name, paths["oac_operational_health.py"]
                )
                if spec is None or spec.loader is None:
                    raise InvalidInput("kernel load")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                try:
                    # Execute the bytes just hashed, never a second pathname
                    # read or an unverified pre-existing __pycache__ entry.
                    exec(
                        compile(
                            verified_bytes["oac_operational_health.py"],
                            str(paths["oac_operational_health.py"]),
                            "exec",
                        ),
                        module.__dict__,
                    )
                except Exception:
                    sys.modules.pop(module_name, None)
                    raise
            self.kernel = kernel_from_snapshots(
                module.OperationalHealthKernel,
                verified_bytes["model.json"],
                verified_bytes["artifact_receipt.json"],
            )
            self.release = release
            self.artifacts_verified = True
        except (OSError, ValueError, TypeError, AttributeError, ImportError):
            # No filenames, submitted keys, source text, or exception strings
            # cross the public API boundary.
            self.failure_code = "ARTIFACT_VERIFICATION_FAILED"
        else:
            self.failure_code = (
                None if self.source_revision else "SOURCE_BINDING_UNAVAILABLE"
            )
        # The v2 research preview has its own readiness. A bad or absent v2
        # package cannot make the already-published v1 path unavailable.
        self.v2_release = None
        self.v2_kernel = None
        self.v2_artifacts_verified = False
        try:
            release = strict_json(
                bounded_file(self.root / "release_v2.json", 16384), 16384
            )
            expected = {
                "schema": "szl.oac-health-space-release/v2",
                "artifact_source": {
                    "repository": SOURCE_REPOSITORY,
                    "revision": V2_ARTIFACT_REVISION,
                    "directory": "ops-health/v2/ops-health",
                },
                "hub_model": {
                    "repo_id": V2_MODEL_ID,
                    "revision": V2_MODEL_REVISION,
                },
                "artifacts": V2_ARTIFACTS,
                "synthetic_training_data": True,
                "clinical_use_authorized": False,
                "production_promotion_allowed": False,
            }
            if release != expected or any(
                type(release.get(key)) is not bool
                for key in (
                    "synthetic_training_data",
                    "clinical_use_authorized",
                    "production_promotion_allowed",
                )
            ):
                raise InvalidInput("v2 release mismatch")
            artifact_dir = self.root / "artifacts" / "v2"
            if artifact_dir.is_symlink():
                raise InvalidInput("v2 artifact directory")
            paths = {name: artifact_dir / name for name in V2_ARTIFACTS}
            verified_bytes = {}
            for name, path in paths.items():
                raw = bounded_file(path, 131072)
                if hashlib.sha256(raw).hexdigest() != V2_ARTIFACTS[name]:
                    raise InvalidInput("v2 artifact digest")
                verified_bytes[name] = raw
            receipt = strict_json(verified_bytes["artifact_receipt.json"], 16384)
            if (
                receipt.get("kernel_sha256") != V2_ARTIFACTS["ops_health.py"]
                or receipt.get("model_sha256") != V2_ARTIFACTS["model.json"]
                or receipt.get("schema") != "szl-oac/ops-health-artifact-receipt/v2"
            ):
                raise InvalidInput("v2 receipt binding")
            module_name = "_szl_oac_ops_v2_" + V2_ARTIFACTS["ops_health.py"][:16]
            spec = importlib.util.spec_from_file_location(
                module_name, paths["ops_health.py"]
            )
            if spec is None or spec.loader is None:
                raise InvalidInput("v2 kernel load")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                # Execute only the bytes already hashed, without loading an
                # unverified cached bytecode file. The v2 constructor reopens
                # disk paths, so the equivalent admission checks below use the
                # canonical validator and scorer against these same snapshots.
                exec(
                    compile(
                        verified_bytes["ops_health.py"],
                        str(paths["ops_health.py"]),
                        "exec",
                    ),
                    module.__dict__,
                )
                if module.AUTHORITY_MAP != V2_AUTHORITY:
                    raise InvalidInput("v2 authority boundary")
                kernel = v2_scorer_from_verified_bytes(
                    module,
                    verified_bytes["ops_health.py"],
                    verified_bytes["model.json"],
                    verified_bytes["artifact_receipt.json"],
                )
                example = strict_json(verified_bytes["example_input.json"])
                if kernel.advise(example).get("authority") != V2_AUTHORITY:
                    raise InvalidInput("v2 example authority boundary")
            except Exception:
                sys.modules.pop(module_name, None)
                raise
            self.v2_kernel = kernel
            self.v2_release = release
            self.v2_artifacts_verified = True
        except Exception:
            # Do not expose request content, filenames, or kernel exceptions.
            self.v2_failure_code = "V2_ARTIFACT_VERIFICATION_FAILED"
        else:
            self.v2_failure_code = (
                None if self.source_revision else "SOURCE_BINDING_UNAVAILABLE"
            )

    @property
    def ready(self) -> bool:
        return self.artifacts_verified and self.source_revision is not None

    @property
    def v2_ready(self) -> bool:
        return self.v2_artifacts_verified and self.source_revision is not None

    def v2_identity(self) -> dict[str, Any]:
        release = self.v2_release or {}
        return {
            "schema": "szl.oac-health-space-identity/v2",
            "state": "READY" if self.v2_ready else "UNAVAILABLE",
            "application": {
                "repository": SOURCE_REPOSITORY,
                "revision": self.source_revision,
            },
            "artifact_source": release.get("artifact_source"),
            "hub_model": release.get("hub_model"),
            "artifacts": release.get("artifacts"),
            "synthetic_training_data": True,
            "clinical_use_authorized": False,
            "production_promotion_allowed": False,
            "authority": dict(V2_AUTHORITY),
            "receipt_minted": False,
        }

    def identity(self) -> dict[str, Any]:
        release = self.release or {}
        return {
            "schema": "szl.oac-health-space-identity/v1",
            "state": "READY" if self.ready else "UNAVAILABLE",
            "application": {
                "repository": SOURCE_REPOSITORY,
                "revision": self.source_revision,
            },
            "artifact_source": release.get("artifact_source"),
            "hub_model": release.get("hub_model"),
            "hub_dataset": release.get("hub_dataset"),
            "artifacts": release.get("artifacts"),
            "synthetic_training_data": True,
            "clinical_use_authorized": False,
            "production_promotion_allowed": False,
        }

    def build_info(self) -> dict[str, Any]:
        return {
            "service": "oac-system-health-lab",
            "build": {
                "state": "OBSERVED" if self.source_revision else "UNKNOWN",
                "revision": self.source_revision,
                "revision_source": "env:SZL_GITHUB_SOURCE_REVISION"
                if self.source_revision
                else None,
            },
            "runtime": {"state": "READY" if self.ready else "UNAVAILABLE"},
            "receipt_minted": False,
        }

    def score(self, raw: bytes) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("unavailable")
        value = strict_json(raw)
        if (
            not isinstance(value, dict)
            or set(value) != {"features"}
            or not isinstance(value["features"], dict)
            or set(value["features"]) != FEATURES
        ):
            raise InvalidInput("closed schema")
        if any(
            not isinstance(item, (bool, int, float))
            for item in value["features"].values()
        ):
            raise InvalidInput("numeric fields only")
        try:
            advisory = self.kernel.score(value["features"])
        except (ValueError, OverflowError, TypeError) as exc:
            raise InvalidInput("feature contract") from exc
        return {
            "ok": True,
            "advisory": advisory,
            "identity": self.identity(),
            "receipt_minted": False,
            "input_sha256": digest(value["features"]),
            "output_sha256": digest(advisory),
        }

    def score_v2(self, raw: bytes) -> dict[str, Any]:
        if not self.v2_ready:
            raise RuntimeError("v2 unavailable")
        value = strict_json(raw)
        if (
            not isinstance(value, dict)
            or set(value) != {"features"}
            or not isinstance(value["features"], dict)
            or set(value["features"]) != FEATURES
            or any(
                not isinstance(item, (bool, int, float))
                for item in value["features"].values()
            )
        ):
            raise InvalidInput("closed v2 schema")
        try:
            advisory = self.v2_kernel.advise(value)
        except (ValueError, OverflowError, TypeError) as exc:
            raise InvalidInput("v2 feature contract") from exc
        if (
            advisory.get("schema") != "szl-oac/ops-health-advisory/v2"
            or advisory.get("advisory") not in {"ALERT", "NO_ALERT", "ABSTAIN"}
            or advisory.get("authority") != V2_AUTHORITY
        ):
            raise RuntimeError("v2 advisory boundary")
        return {
            "ok": True,
            "advisory": advisory,
            "identity": self.v2_identity(),
            "receipt_minted": False,
            "input_sha256": digest(value["features"]),
            "output_sha256": digest(advisory),
        }


class _DeadlineReader(io.RawIOBase):
    """Guard every socket receive, including buffered header/body reads."""

    def __init__(self, connection: socket.socket, deadline: float):
        super().__init__()
        self.connection = connection
        self.deadline = deadline
        self.received = 0

    def readable(self):
        return True

    def readinto(self, buffer):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("request deadline")
        self.connection.settimeout(remaining)
        allowed = MAX_REQUEST_BYTES + 1 - self.received
        if allowed <= 0:
            raise RequestLimit("request size")
        count = self.connection.recv_into(memoryview(buffer)[:allowed])
        self.received += count
        if self.received > MAX_REQUEST_BYTES:
            raise RequestLimit("request size")
        return count


def content_security_policy(html: bytes) -> str:
    text = html.decode("utf-8", errors="strict")
    import base64

    values = {}
    for tag in ("script", "style"):
        matches = re.findall(
            r"<" + tag + r"(?:\s[^>]*)?>(.*?)</" + tag + r"\s*[^>]*>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        hashes = [
            "'sha256-"
            + base64.b64encode(hashlib.sha256(item.encode("utf-8")).digest()).decode(
                "ascii"
            )
            + "'"
            for item in matches
        ]
        values[tag] = " ".join(hashes) or "'none'"
    return (
        "default-src 'none'; script-src "
        + values["script"]
        + "; style-src "
        + values["style"]
        + "; connect-src 'self'; img-src 'none'; base-uri 'none'; form-action 'none'; "
        + "frame-ancestors 'self' https://a-11-oy.com https://a11oy.net"
    )


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "SZL-OAC-Preview"
    sys_version = ""

    def setup(self):
        self.connection = self.request
        self.deadline = self.server.request_deadline(self.connection)
        self.rfile = io.BufferedReader(
            _DeadlineReader(self.connection, self.deadline), 8192
        )
        self.wfile = self.connection.makefile("wb", buffering=0)

    def handle(self):
        self.close_connection = True
        try:
            self.handle_one_request()
        except RequestLimit:
            self._json(431, {"ok": False, "error": "REQUEST_TOO_LARGE"})
        except (OSError, ValueError):
            pass
        self.close_connection = True

    def log_message(self, *_):
        # Requests, paths, headers, bodies, and input keys are not logged.
        pass

    def send_error(self, code, message=None, explain=None):
        # BaseHTTPRequestHandler's default HTML error can echo an untrusted
        # request line or method. Replace it even for parser-level failures.
        self._json(code, {"ok": False, "error": "INVALID_HTTP_REQUEST"})

    def _send(self, status, body, content_type="application/json", csp=None):
        self.close_connection = True
        self.connection.settimeout(max(0.05, self.deadline - time.monotonic()))
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Connection", "close")
            if csp:
                self.send_header("Content-Security-Policy", csp)
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass

    def _json(self, status, value):
        self._send(status, canonical(value))

    def do_GET(self):
        app = self.server.application
        if self.path == "/healthz":
            self._json(
                200,
                {
                    "alive": True,
                    "state": "READY" if app.ready else "UNAVAILABLE",
                    "receipt_minted": False,
                },
            )
        elif self.path == "/readyz":
            self._json(
                200 if app.ready else 503,
                {
                    "state": "READY" if app.ready else "UNAVAILABLE",
                    "error": app.failure_code,
                    "receipt_minted": False,
                },
            )
        elif self.path == "/api/build-info":
            self._json(200, app.build_info())
        elif self.path == "/api/v1/identity":
            self._json(200, app.identity())
        elif self.path == "/api/v2/identity":
            self._json(200, app.v2_identity())
        elif self.path == "/api/v2/readyz":
            self._json(
                200 if app.v2_ready else 503,
                {
                    "state": "READY" if app.v2_ready else "UNAVAILABLE",
                    "error": app.v2_failure_code,
                    "receipt_minted": False,
                },
            )
        elif self.path in ("/", "/index.html"):
            try:
                html = bounded_file(app.root / "index.html", 262144)
                csp = content_security_policy(html)
            except (OSError, UnicodeError, ValueError):
                self._json(503, {"ok": False, "error": "UI_UNAVAILABLE"})
            else:
                self._send(200, html, "text/html; charset=utf-8", csp)
        else:
            self._json(404, {"ok": False, "error": "NOT_FOUND"})

    def do_POST(self):
        if self.path not in ("/api/score", "/api/v2/score"):
            self._json(404, {"ok": False, "error": "NOT_FOUND"})
            return
        application = self.server.application
        if not (application.ready if self.path == "/api/score" else application.v2_ready):
            self._json(503, {"ok": False, "error": "SERVICE_UNAVAILABLE"})
            return
        if self.headers.get_all("X-SZL-Preview", []) != ["1"]:
            self._json(403, {"ok": False, "error": "PREVIEW_HEADER_REQUIRED"})
            return
        if self.headers.get_all("Transfer-Encoding", []):
            self._json(400, {"ok": False, "error": "UNSUPPORTED_FRAMING"})
            return
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or re.fullmatch(r"[0-9]{1,10}", lengths[0]) is None:
            self._json(411, {"ok": False, "error": "EXACT_CONTENT_LENGTH_REQUIRED"})
            return
        length = int(lengths[0])
        if length > MAX_BODY_BYTES:
            self._json(413, {"ok": False, "error": "BODY_TOO_LARGE"})
            return
        types = self.headers.get_all("Content-Type", [])
        if len(types) != 1 or types[0].strip().lower() not in (
            "application/json",
            "application/json; charset=utf-8",
        ):
            self._json(415, {"ok": False, "error": "JSON_REQUIRED"})
            return
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise InvalidInput("incomplete")
            if time.monotonic() >= self.deadline:
                raise TimeoutError("deadline")
            result = (
                application.score(raw)
                if self.path == "/api/score"
                else application.score_v2(raw)
            )
        except (TimeoutError, socket.timeout):
            self._json(408, {"ok": False, "error": "REQUEST_TIMEOUT"})
        except (InvalidInput, RequestLimit):
            self._json(400, {"ok": False, "error": "INVALID_SCORE_INPUT"})
        except RuntimeError:
            self._json(503, {"ok": False, "error": "SERVICE_UNAVAILABLE"})
        except OSError:
            self.close_connection = True
        else:
            self._json(200, result)

    def _unsupported(self):
        self._json(405, {"ok": False, "error": "METHOD_NOT_ALLOWED"})

    do_OPTIONS = _unsupported
    do_HEAD = _unsupported
    do_PUT = _unsupported
    do_PATCH = _unsupported
    do_DELETE = _unsupported
    do_TRACE = _unsupported


class BoundedServer(socketserver.TCPServer):
    """At most four connection threads; never queues accepted work."""

    allow_reuse_address = True
    request_queue_size = 4

    def __init__(
        self,
        address,
        application,
        request_timeout=REQUEST_TIMEOUT_SECONDS,
        workers=MAX_WORKERS,
    ):
        if (
            not 0 < request_timeout <= REQUEST_TIMEOUT_SECONDS
            or not 1 <= workers <= MAX_WORKERS
        ):
            raise ValueError("invalid service bounds")
        self.application = application
        self.request_timeout = request_timeout
        self.slots = threading.BoundedSemaphore(workers)
        self.lock = threading.Lock()
        self.connections = {}
        self.threads = set()
        self.max_active_observed = 0
        super().__init__(address, Handler)

    def request_deadline(self, connection):
        with self.lock:
            return self.connections[connection]

    def process_request(self, request, client_address):
        deadline = time.monotonic() + self.request_timeout
        if not self.slots.acquire(blocking=False):
            body = b'{"ok":false,"error":"BUSY"}'
            try:
                request.settimeout(0.05)
                request.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                    + str(len(body)).encode("ascii")
                    + b"\r\n\r\n"
                    + body
                )
                # Closing a TCP connection with unread inbound bytes can send
                # RST on Windows and discard the already-written 503. Half
                # close the response, then drain a bounded amount for a single
                # absolute 50 ms budget. This never creates or queues a worker.
                request.shutdown(socket.SHUT_WR)
                drain_deadline = time.monotonic() + 0.05
                remaining_bytes = MAX_REQUEST_BYTES
                while remaining_bytes > 0:
                    remaining_time = drain_deadline - time.monotonic()
                    if remaining_time <= 0:
                        break
                    request.settimeout(remaining_time)
                    chunk = request.recv(min(4096, remaining_bytes))
                    if not chunk:
                        break
                    remaining_bytes -= len(chunk)
            except OSError:
                pass
            self.shutdown_request(request)
            return
        thread = threading.Thread(
            target=self._worker, args=(request, client_address), daemon=True
        )
        with self.lock:
            self.connections[request] = deadline
            self.threads.add(thread)
            self.max_active_observed = max(
                self.max_active_observed, len(self.connections)
            )
        try:
            thread.start()
        except RuntimeError:
            with self.lock:
                self.connections.pop(request, None)
                self.threads.discard(thread)
            self.slots.release()
            self.shutdown_request(request)

    def _worker(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            # A handler failure never emits input/path-bearing tracebacks.
            pass
        finally:
            self.shutdown_request(request)
            with self.lock:
                self.connections.pop(request, None)
                self.threads.discard(threading.current_thread())
            self.slots.release()

    def server_close(self):
        super().server_close()
        with self.lock:
            connections = list(self.connections)
            threads = list(self.threads)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(timeout=0.5)


def create_server(
    address=("0.0.0.0", 7860),
    application=None,
    *,
    request_timeout=REQUEST_TIMEOUT_SECONDS,
    workers=MAX_WORKERS,
):
    return BoundedServer(
        address,
        application if application is not None else Application(),
        request_timeout,
        workers,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    with create_server((args.host, args.port)) as server:
        try:
            server.serve_forever(poll_interval=0.1)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
