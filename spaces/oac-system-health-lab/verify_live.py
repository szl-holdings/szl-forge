"""Source-bound, synthetic-only live proof for the fixed OAC Health preview.

Only the one declared public Space may be contacted. There are no redirects,
retries, authentication, model downloads, private inputs, or authority changes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import functools
import hashlib
import http.client
import io
import json
from pathlib import Path
import re
import socket
import sys
import time
import types
from typing import Any
from urllib.parse import urlsplit

# Isolated Python excludes the script directory from sys.path. Admit only this
# verifier's resolved sibling directory, never CWD, --space-root or PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_release import (  # noqa: E402
    ARTIFACT_NAMES,
    MAX_ARTIFACT_BYTES,
    SHA1,
    SOURCE_DIRECTORY,
    SOURCE_REPOSITORY,
    V2_ARTIFACTS,
    V2_SOURCE_DIRECTORY,
    V2_SOURCE_REVISION,
    V2_HUB_REVISION,
    ReleaseVerificationError,
    _closed,
    _git,
    load_manifest,
    strict_json,
    utc_now,
    verify_release,
    verify_v2_release,
    write_report,
)

REPORT_SCHEMA = "szl.oac-health-space-live-verification/v1"
IDENTITY_SCHEMA = "szl.oac-health-space-identity/v1"
PUBLIC_HOST = "szlholdings-oac-system-health-lab.hf.space"
DEFAULT_ORIGIN = "https://" + PUBLIC_HOST
MAX_RESPONSE_BYTES = 65_536
REQUEST_SECONDS = 20.0
AUTHORITY_KEYS = {
    "acknowledgement",
    "clinical_decision",
    "device_control",
    "result_interpretation",
    "result_release",
}
CASES = (
    (
        "authored_healthy_synthetic",
        {
            "listener_running": 1,
            "tls_enabled": 1,
            "peer_allowlist_configured": 1,
            "queue_utilization": 0.05,
            "consecutive_failures": 0,
            "seconds_since_last_success": 1,
            "ledger_integrity_ok": 1,
            "configuration_valid": 1,
        },
    ),
    (
        "authored_degraded_synthetic",
        {
            "listener_running": 0,
            "tls_enabled": 0,
            "peer_allowlist_configured": 0,
            "queue_utilization": 0.95,
            "consecutive_failures": 15,
            "seconds_since_last_success": 7200,
            "ledger_integrity_ok": 0,
            "configuration_valid": 0,
        },
    ),
)
V2_CASES = CASES + (
    (
        "authored_ambiguous_synthetic",
        {
            "listener_running": 1,
            "tls_enabled": 1,
            "peer_allowlist_configured": 1,
            "queue_utilization": 0.1,
            "consecutive_failures": 5,
            "seconds_since_last_success": 1,
            "ledger_integrity_ok": 1,
            "configuration_valid": 1,
        },
    ),
)
REJECTION_MARKER = "SYNTHETIC_CLINICAL_REJECTION_ONLY"


class LiveVerificationError(ValueError):
    """A live observation cannot support the requested claim."""


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


@dataclass(frozen=True)
class Observation:
    status: int
    content_type: str
    body: bytes


def target(origin: str, *, allow_localhost: bool = False) -> tuple[str, str, int]:
    try:
        parsed = urlsplit(origin)
        port = parsed.port
    except ValueError as exc:
        raise LiveVerificationError("TARGET_NOT_ALLOWED") from exc
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise LiveVerificationError("TARGET_NOT_ALLOWED")
    if (
        parsed.scheme == "https"
        and parsed.hostname == PUBLIC_HOST
        and port in (None, 443)
    ):
        return "https", PUBLIC_HOST, 443
    if (
        allow_localhost
        and parsed.scheme == "http"
        and parsed.hostname in ("localhost", "127.0.0.1", "::1")
        and port is not None
        and 1 <= port <= 65535
    ):
        return "http", parsed.hostname, port
    raise LiveVerificationError("TARGET_NOT_ALLOWED")


class _DeadlineReader(io.RawIOBase):
    def __init__(self, raw: io.RawIOBase, connection: socket.socket, deadline: float):
        self.raw = raw
        self.connection = connection
        self.deadline = deadline
        self.received = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("absolute response deadline")
        self.connection.settimeout(remaining)
        count = self.raw.readinto(buffer)
        self.received += count or 0
        # Includes status, headers, framing and body; even unbounded HTTP
        # header/trailer streams cannot extend the byte or absolute-time budget.
        if self.received > MAX_RESPONSE_BYTES + 16_384:
            raise LiveVerificationError("HTTP_RECEIVE_BYTE_LIMIT")
        return count

    def close(self) -> None:
        try:
            self.raw.close()
        finally:
            super().close()


class _DeadlineResponse(http.client.HTTPResponse):
    def __init__(
        self, connection: socket.socket, *args: Any, deadline: float, **kwargs: Any
    ):
        super().__init__(connection, *args, **kwargs)
        self.fp = io.BufferedReader(
            _DeadlineReader(self.fp.detach(), connection, deadline)
        )


class Transport:
    def __init__(self, origin: str, *, allow_localhost: bool = False):
        self.scheme, self.host, self.port = target(
            origin, allow_localhost=allow_localhost
        )

    def request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Observation:
        if (method, path) not in {
            ("GET", "/healthz"),
            ("GET", "/readyz"),
            ("GET", "/api/build-info"),
            ("GET", "/api/v1/identity"),
            ("GET", "/api/v2/readyz"),
            ("GET", "/api/v2/identity"),
            ("POST", "/api/score"),
            ("POST", "/api/v2/score"),
        }:
            raise LiveVerificationError("REQUEST_NOT_ALLOWED")
        deadline = time.monotonic() + REQUEST_SECONDS
        connection_type = (
            http.client.HTTPSConnection
            if self.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(self.host, self.port, timeout=REQUEST_SECONDS)
        connection.response_class = functools.partial(
            _DeadlineResponse, deadline=deadline
        )
        headers = {"Accept": "application/json", "Connection": "close"}
        body = None
        if payload is not None:
            if method != "POST":
                raise LiveVerificationError("REQUEST_NOT_ALLOWED")
            body = canonical(payload)
            if len(body) > 4096:
                raise LiveVerificationError("REQUEST_BYTE_LIMIT")
            headers.update({"Content-Type": "application/json", "X-SZL-Preview": "1"})
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise LiveVerificationError("RESPONSE_BYTE_LIMIT")
            if time.monotonic() > deadline:
                raise LiveVerificationError("RESPONSE_DEADLINE_EXCEEDED")
            return Observation(
                response.status, response.getheader("Content-Type", ""), raw
            )
        except (OSError, http.client.HTTPException, TimeoutError) as exc:
            raise LiveVerificationError("HTTP_TRANSPORT_FAILURE") from exc
        finally:
            connection.close()


def _json(observation: Observation, *, status: int = 200) -> dict[str, Any]:
    if type(observation.status) is not int or observation.status != status:
        raise LiveVerificationError("UNEXPECTED_HTTP_STATUS")
    if (
        not isinstance(observation.content_type, str)
        or observation.content_type.split(";", 1)[0].strip().lower()
        != "application/json"
    ):
        raise LiveVerificationError("JSON_CONTENT_TYPE_REQUIRED")
    value = strict_json(observation.body, limit=MAX_RESPONSE_BYTES)
    if not isinstance(value, dict):
        raise LiveVerificationError("JSON_OBJECT_REQUIRED")
    return value


def expected_identity(manifest: dict[str, Any], source_sha: str) -> dict[str, Any]:
    return {
        "schema": IDENTITY_SCHEMA,
        "state": "READY",
        "application": {"repository": SOURCE_REPOSITORY, "revision": source_sha},
        "artifact_source": manifest["artifact_source"],
        "hub_model": manifest["hub_model"],
        "hub_dataset": manifest["hub_dataset"],
        "artifacts": manifest["artifacts"],
        "synthetic_training_data": True,
        "clinical_use_authorized": False,
        "production_promotion_allowed": False,
    }


def _identity(observed: Any, expected: dict[str, Any]) -> None:
    if canonical(observed) != canonical(expected):
        raise LiveVerificationError("IDENTITY_OR_SOURCE_BINDING_MISMATCH")


@dataclass(frozen=True)
class _ArtifactSnapshot:
    name: str
    content: bytes

    def read_bytes(self) -> bytes:
        return self.content

    def read_text(self, encoding: str = "utf-8") -> str:
        if encoding != "utf-8":
            raise LiveVerificationError("CANONICAL_SNAPSHOT_DECODING_MISMATCH")
        return self.content.decode("utf-8", errors="strict")


def _kernel_from_immutable_snapshots(
    kernel_type: Any, model: bytes, receipt: bytes
) -> Any:
    """Adapt the pinned constructor's reads without opening mutable files.

    This independent verifier adapter does not import the application's kernel
    or adapter. The unchanged canonical constructor and score method execute;
    its two path properties return only the Git blobs already hash-verified.
    Updating the kernel pin requires re-reviewing this narrow read contract.
    """

    class SnapshotKernel(kernel_type):
        def __init__(self) -> None:
            self._model_snapshot = _ArtifactSnapshot("model.json", bytes(model))
            self._receipt_snapshot = _ArtifactSnapshot(
                "artifact_receipt.json", bytes(receipt)
            )
            self._paths_locked = False
            super().__init__("model.json", "artifact_receipt.json")
            self._paths_locked = True

        @property
        def model_path(self) -> _ArtifactSnapshot:
            return self._model_snapshot

        @model_path.setter
        def model_path(self, value: Path) -> None:
            if self._paths_locked or value.name != self._model_snapshot.name:
                raise LiveVerificationError("CANONICAL_MODEL_PATH_CHANGED")

        @property
        def receipt_path(self) -> _ArtifactSnapshot:
            return self._receipt_snapshot

        @receipt_path.setter
        def receipt_path(self, value: Path) -> None:
            if self._paths_locked or value.name != self._receipt_snapshot.name:
                raise LiveVerificationError("CANONICAL_RECEIPT_PATH_CHANGED")

    return SnapshotKernel()


def _load_kernel(
    space_root: Path, repository_root: Path, manifest: dict[str, Any]
) -> Any:
    # Independently execute immutable Git bytes, never app.py or its loaded
    # kernel. The constructor is adapted to verified immutable snapshots, so
    # there is no post-verification file reopen or temporary artifact path.
    immutable = {}
    revision = manifest["artifact_source"]["revision"]
    for name in ARTIFACT_NAMES:
        raw = _git(repository_root, "show", f"{revision}:{SOURCE_DIRECTORY}/{name}")
        if (
            len(raw) > MAX_ARTIFACT_BYTES
            or hashlib.sha256(raw).hexdigest() != manifest["artifacts"][name]
        ):
            raise LiveVerificationError("CANONICAL_KERNEL_ARTIFACT_MISMATCH")
        immutable[name] = raw
    name = "_szl_oac_live_verifier_canonical"
    module = types.ModuleType(name)
    prior = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(
            compile(
                immutable["oac_operational_health.py"], "<immutable-oac-kernel>", "exec"
            ),
            module.__dict__,
        )
        return _kernel_from_immutable_snapshots(
            module.OperationalHealthKernel,
            immutable["model.json"],
            immutable["artifact_receipt.json"],
        )
    finally:
        if prior is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prior


def _load_v2_scorer(repository_root: Path) -> Any:
    """Independently score from v2's immutable Git blobs, never app.py."""
    immutable = {}
    for name, expected in V2_ARTIFACTS.items():
        raw = _git(
            repository_root,
            "show",
            f"{V2_SOURCE_REVISION}:{V2_SOURCE_DIRECTORY}/{name}",
        )
        if len(raw) > MAX_ARTIFACT_BYTES or hashlib.sha256(raw).hexdigest() != expected:
            raise LiveVerificationError("V2_CANONICAL_ARTIFACT_MISMATCH")
        immutable[name] = raw
    name = "_szl_oac_live_verifier_v2"
    module = types.ModuleType(name)
    prior = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(immutable["ops_health.py"], "<immutable-oac-v2>", "exec"), module.__dict__)
        receipt = module.validate_receipt(
            module.loads_strict(
                immutable["artifact_receipt.json"].decode("utf-8"),
                module.ModelArtifactError,
            )
        )
        if (
            receipt["kernel_sha256"] != V2_ARTIFACTS["ops_health.py"]
            or receipt["model_sha256"] != V2_ARTIFACTS["model.json"]
        ):
            raise LiveVerificationError("V2_RECEIPT_BINDING_MISMATCH")
        artifact = module.loads_strict(
            immutable["model.json"].decode("utf-8"), module.ModelArtifactError
        )
        scorer = module.ArtifactScorer(artifact)
        if scorer._artifact["generator"]["source_sha256"] != receipt["generator_sha256"]:
            raise LiveVerificationError("V2_GENERATOR_BINDING_MISMATCH")
        return scorer
    finally:
        if prior is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prior


def verify_live(
    *,
    origin: str,
    expected_source_sha: str,
    space_root: Path,
    repository_root: Path,
    allow_localhost: bool = False,
    transport: Transport | None = None,
    verify_v2: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "started_at": utc_now(),
        "origin": origin,
        "expected_application_source_sha": expected_source_sha,
        "complete": False,
        "status": "FAIL",
        "checks": [],
        "training_performed": False,
        "clinical_use_authorized": False,
        "production_promotion_allowed": False,
        "receipt_minted": False,
    }
    try:
        scheme, _host, _port = target(origin, allow_localhost=allow_localhost)
        report["observation_scope"] = (
            "PUBLIC_SYNTHETIC_PROBE" if scheme == "https" else "LOCAL_TEST_ONLY"
        )
        if (
            not isinstance(expected_source_sha, str)
            or SHA1.fullmatch(expected_source_sha) is None
        ):
            raise LiveVerificationError("EXACT_APPLICATION_SOURCE_SHA_REQUIRED")
        manifest = load_manifest(space_root)
        report.update(
            {
                key: manifest[key]
                for key in ("artifact_source", "hub_model", "hub_dataset", "artifacts")
            }
        )
        report["release_verification"] = verify_release(space_root, repository_root)
        kernel = _load_kernel(space_root, repository_root, manifest)
        identity = expected_identity(manifest, expected_source_sha)
        client = (
            transport
            if transport is not None
            else Transport(origin, allow_localhost=allow_localhost)
        )
        checks = report["checks"]
        for path in ("/healthz", "/readyz", "/api/build-info", "/api/v1/identity"):
            value = _json(client.request("GET", path))
            if path == "/healthz":
                if canonical(value) != canonical(
                    {"alive": True, "state": "READY", "receipt_minted": False}
                ):
                    raise LiveVerificationError("LIVENESS_CONTRACT_MISMATCH")
            elif path == "/readyz":
                if canonical(value) != canonical(
                    {"state": "READY", "error": None, "receipt_minted": False}
                ):
                    raise LiveVerificationError("READINESS_CONTRACT_MISMATCH")
            elif path == "/api/build-info":
                expected = {
                    "service": "oac-system-health-lab",
                    "build": {
                        "state": "OBSERVED",
                        "revision": expected_source_sha,
                        "revision_source": "env:SZL_GITHUB_SOURCE_REVISION",
                    },
                    "runtime": {"state": "READY"},
                    "receipt_minted": False,
                }
                if canonical(value) != canonical(expected):
                    raise LiveVerificationError("BUILD_SOURCE_CONTRACT_MISMATCH")
            else:
                _identity(value, identity)
            checks.append(
                {
                    "name": path,
                    "method": "GET",
                    "status": "PASS",
                    "http_status": 200,
                    "response_sha256": digest(value),
                    "observed_at": utc_now(),
                }
            )
        for label, features in CASES:
            payload = {"features": dict(features)}
            expected_advisory = kernel.score(features)
            value = _json(client.request("POST", "/api/score", payload))
            _closed(
                value,
                {
                    "ok",
                    "advisory",
                    "identity",
                    "receipt_minted",
                    "input_sha256",
                    "output_sha256",
                },
                "score response",
            )
            if value["ok"] is not True or value["receipt_minted"] is not False:
                raise LiveVerificationError("SCORING_OR_RECEIPT_CONTRACT_MISMATCH")
            _identity(value["identity"], identity)
            advisory = value["advisory"]
            if (
                not isinstance(advisory, dict)
                or not isinstance(advisory.get("authority"), dict)
                or set(advisory["authority"]) != AUTHORITY_KEYS
                or any(flag is not False for flag in advisory["authority"].values())
            ):
                raise LiveVerificationError("AUTHORITY_BOUNDARY_MISMATCH")
            if canonical(advisory) != canonical(expected_advisory):
                raise LiveVerificationError("CANONICAL_SCORE_MISMATCH")
            if value["input_sha256"] != digest(features) or value[
                "output_sha256"
            ] != digest(expected_advisory):
                raise LiveVerificationError("SCORING_DIGEST_MISMATCH")
            checks.append(
                {
                    "name": label,
                    "method": "POST",
                    "status": "PASS",
                    "http_status": 200,
                    "input_sha256": digest(features),
                    "output_sha256": digest(expected_advisory),
                    "observed_at": utc_now(),
                }
            )
        invalid = {"features": dict(CASES[0][1], patient_id=REJECTION_MARKER)}
        observation = client.request("POST", "/api/score", invalid)
        value = _json(observation, status=400)
        if (
            set(value) != {"ok", "error"}
            or value["ok"] is not False
            or not isinstance(value["error"], str)
            or re.fullmatch(r"[A-Z_]{1,64}", value["error"]) is None
        ):
            raise LiveVerificationError("CLINICAL_INPUT_REJECTION_CONTRACT_MISMATCH")
        decoded_response = canonical(value)
        if (
            REJECTION_MARKER.encode() in decoded_response
            or b"patient_id" in decoded_response.lower()
        ):
            raise LiveVerificationError("REJECTED_INPUT_ECHOED")
        checks.append(
            {
                "name": "clinical_like_extra_field_refused_without_echo",
                "method": "POST",
                "status": "PASS",
                "http_status": 400,
                "observed_at": utc_now(),
            }
        )
        if verify_v2:
            report["v2_release_verification"] = verify_v2_release(
                space_root, repository_root
            )
            v2_scorer = _load_v2_scorer(repository_root)
            v2_identity = {
                "schema": "szl.oac-health-space-identity/v2",
                "state": "READY",
                "application": {
                    "repository": SOURCE_REPOSITORY,
                    "revision": expected_source_sha,
                },
                "artifact_source": {
                    "repository": SOURCE_REPOSITORY,
                    "revision": V2_SOURCE_REVISION,
                    "directory": V2_SOURCE_DIRECTORY,
                },
                "hub_model": {
                    "repo_id": "SZLHOLDINGS/oac-ops-health-v2",
                    "revision": V2_HUB_REVISION,
                },
                "artifacts": V2_ARTIFACTS,
                "synthetic_training_data": True,
                "clinical_use_authorized": False,
                "production_promotion_allowed": False,
                "authority": {name: False for name in AUTHORITY_KEYS},
                "receipt_minted": False,
            }
            for path, expected in (
                (
                    "/api/v2/readyz",
                    {"state": "READY", "error": None, "receipt_minted": False},
                ),
                ("/api/v2/identity", v2_identity),
            ):
                observed = _json(client.request("GET", path))
                if canonical(observed) != canonical(expected):
                    raise LiveVerificationError("V2_IDENTITY_OR_READINESS_MISMATCH")
                checks.append(
                    {
                        "name": path,
                        "method": "GET",
                        "status": "PASS",
                        "http_status": 200,
                        "response_sha256": digest(observed),
                        "observed_at": utc_now(),
                    }
                )
            for label, features in V2_CASES:
                expected_advisory = v2_scorer.advise({"features": dict(features)})
                observed = _json(
                    client.request("POST", "/api/v2/score", {"features": dict(features)})
                )
                if canonical(observed) != canonical(
                    {
                        "ok": True,
                        "advisory": expected_advisory,
                        "identity": v2_identity,
                        "receipt_minted": False,
                        "input_sha256": digest(features),
                        "output_sha256": digest(expected_advisory),
                    }
                ):
                    raise LiveVerificationError("V2_CANONICAL_SCORE_MISMATCH")
                if (
                    set(expected_advisory["authority"]) != AUTHORITY_KEYS
                    or any(expected_advisory["authority"].values())
                ):
                    raise LiveVerificationError("V2_AUTHORITY_BOUNDARY_MISMATCH")
                checks.append(
                    {
                        "name": "v2_" + label,
                        "method": "POST",
                        "status": "PASS",
                        "http_status": 200,
                        "input_sha256": digest(features),
                        "output_sha256": digest(expected_advisory),
                        "observed_at": utc_now(),
                    }
                )
            forbidden = {"features": dict(CASES[0][1], patient_id=REJECTION_MARKER)}
            refused = _json(
                client.request("POST", "/api/v2/score", forbidden), status=400
            )
            if (
                refused != {"ok": False, "error": "INVALID_SCORE_INPUT"}
                or REJECTION_MARKER.encode() in canonical(refused)
            ):
                raise LiveVerificationError("V2_FORBIDDEN_INPUT_REFUSAL_MISMATCH")
            checks.append(
                {
                    "name": "v2_clinical_like_extra_field_refused_without_echo",
                    "method": "POST",
                    "status": "PASS",
                    "http_status": 400,
                    "observed_at": utc_now(),
                }
            )
            report["v2_synthetic_case_count"] = len(V2_CASES)
        report.update(
            {"complete": True, "status": "PASS", "synthetic_case_count": len(CASES)}
        )
    except (
        LiveVerificationError,
        ReleaseVerificationError,
        ValueError,
        TypeError,
        KeyError,
        RecursionError,
    ) as exc:
        report["failure"] = (
            str(exc)
            if isinstance(exc, (LiveVerificationError, ReleaseVerificationError))
            else "INVALID_APPLICATION_CONTRACT"
        )
    report["completed_at"] = utc_now()
    report["boundaries"] = [
        "Only authored synthetic numerical telemetry and an invented clinical-like rejection marker were submitted.",
        "Source binding and matching fixed-model scores are not accuracy, clinical authorization, availability, or production qualification.",
        "No training, model downloads, retries, authentication, redirects, device connection, or authoritative receipt minting was performed.",
    ]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default=DEFAULT_ORIGIN)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--space-root", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--repository-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--allow-localhost",
        action="store_true",
        help="Test-only: permit explicit HTTP loopback origin; never a provider deployment proof.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-v2", action="store_true", help="Also require the opt-in v2 runtime witness"
    )
    args = parser.parse_args(argv)
    report = verify_live(
        origin=args.origin,
        expected_source_sha=args.expected_source_sha,
        space_root=args.space_root,
        repository_root=args.repository_root,
        allow_localhost=args.allow_localhost,
        verify_v2=args.verify_v2,
    )
    write_report(args.output, report)
    print(
        json.dumps(
            {"complete": report["complete"], "status": report["status"]}, sort_keys=True
        )
    )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
