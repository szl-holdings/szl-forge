#!/usr/bin/env python3
"""Owned Agent Clinical Control: fail-closed Windows supervision plus an offline gateway.

Enforcement boundary:
  * launches only explicitly registered local executables;
  * assigns the launched process to a Windows Job Object before it can run;
  * terminates the supervised process tree;
  * durably blocks later starts for the registered owned-agent ID;
  * revokes only the supervisor-local generation credential;
  * verifies short-lived Ed25519 quorum authorization and one-shot replay state;
  * records a locally tamper-evident (not immutable) hash-chained audit ledger.

It does NOT control external providers, remote credentials, network policy,
messages, model state, containers, VMs, services, or processes it did not start.

Clinical boundary:
  * ingests tightly constrained synthetic MOCK files and deidentified LIVE_SHADOW input;
  * validates bounded message profiles shaped from a cited Roche host-interface manual;
  * preserves source values, quarantines failed gates, and binds each version;
  * requires exact-version Ed25519 reviewer-key authorization before an offline artifact;
  * never releases a patient result or delivers anything to a LIS, EHR, clinician,
    or patient. Network transport lives in the separately supervised bridge module.
"""

from __future__ import annotations

import argparse
import base64
import copy
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
import uuid

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    CRYPTOGRAPHY_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - exercised by deployment environments
    InvalidSignature = Exception  # type: ignore[assignment,misc]
    serialization = None  # type: ignore[assignment]
    Ed25519PrivateKey = None  # type: ignore[assignment,misc]
    Ed25519PublicKey = None  # type: ignore[assignment,misc]
    CRYPTOGRAPHY_IMPORT_ERROR = exc


PROGRAM = "owned-agent-control"
PROGRAM_VERSION = "2.5.0"
STATE_SCHEMA_VERSION = "2"
ENVELOPE_SCHEMA = "owned-agent-isolation/v1"
ENVELOPE_ACTION = "isolate_owned_agent"
SIGNATURE_SCHEME = "Ed25519"
SIGNATURE_DOMAIN = b"owned-agent-control/isolate/v1\x00"
QUORUM = 2
MAX_TTL_SECONDS = 300
MAX_CLOCK_SKEW_SECONDS = 30
MAX_JSON_BYTES = 64 * 1024
MAX_JSON_DEPTH = 20
MAX_SIGNATURES = 8
TARGET_PATTERN = re.compile(r"^owned-agent:[a-z0-9][a-z0-9._-]{1,127}$")
OPERATOR_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")
B64URL_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
UTC_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
ZERO_HASH = "00" * 32

EFFECTS = {
    "block_supervisor_starts": True,
    "preserve_local_audit": True,
    "revoke_supervisor_local_credential": True,
    "terminate_supervised_process_tree": True,
}
SCOPE = {
    "enforcement": "windows_job_object",
    "platform": "windows",
    "stop_only": True,
}

EXIT_OK = 0
EXIT_INPUT = 2
EXIT_DENIED = 3
EXIT_CONFLICT = 4
EXIT_UNCONFIRMED = 5
EXIT_INTERNAL = 6

SUPERVISOR_CAP_ENV = "OAC_SUPERVISOR_CAPABILITY"
LOCAL_CREDENTIAL_ENV = "OAC_LOCAL_CREDENTIAL"
AGENT_ID_ENV = "OAC_AGENT_ID"
GENERATION_ENV = "OAC_GENERATION"
PASSPHRASE_ENV = "OAC_KEY_PASSPHRASE"

# Clinical schema v5 supports an external transport bridge, but remains an
# offline review/export system. LIVE_SHADOW is intentionally deidentified,
# not site-validated, and never authorizes clinical delivery.
CLINICAL_SCHEMA_VERSION = "5"
CLINICAL_MOCK_PROFILE_ID = "illustrative-mock-roche-cobas-liat-hl7-sw-3.4-3.5-him-11.3"
CLINICAL_LIVE_SHADOW_PROFILE_ID = (
    "unvalidated-live-shadow-roche-cobas-liat-hl7-sw-3.4-3.5-him-11.3"
)
# Backward-compatible name used by the synthetic fixture helpers and callers.
CLINICAL_PROFILE_ID = CLINICAL_MOCK_PROFILE_ID
CLINICAL_PROFILE_IDS = {
    "LIVE_SHADOW": CLINICAL_LIVE_SHADOW_PROFILE_ID,
    "MOCK": CLINICAL_MOCK_PROFILE_ID,
}
CLINICAL_POLICY_VERSION = "owned-agent-clinical-policy/v5"
CLINICAL_REVIEW_SCHEMA = "owned-agent-clinical-review/v5"
CLINICAL_REVIEW_ACTION = "authorize_offline_fhir_export"
CLINICAL_REVIEW_DOMAIN = b"owned-agent-control/clinical-review/v5\x00"
CLINICAL_TRANSFORM_VERSION = "owned-agent-clinical-fhir-r4-offline/v5"
CLINICAL_EXPORT_AUTHORIZATION_SCHEMA = "owned-agent-clinical-export-authorization/v2"
CLINICAL_EXPORT_AUTHORIZATION_SUFFIX = ".authorization.json"
CLINICAL_MAX_HL7_BYTES = 1024 * 1024
CLINICAL_MAX_SEGMENTS = 1024
CLINICAL_MAX_OBSERVATIONS = 256
CLINICAL_MODES = {"MOCK", "LIVE_SHADOW"}
CLINICAL_SYNTHETIC_SOURCE_ID = "synthetic-liat"
CLINICAL_SYNTHETIC_SENDER_APPLICATION = "LIAT-SIM"
CLINICAL_SYNTHETIC_SENDER_FACILITY = "LAB-SIM"
CLINICAL_SYNTHETIC_ASSAY_MAP = {
    "SYNTH-FLU": {
        "display": "Synthetic influenza assay",
        "local_system": "urn:synthetic:assay",
    }
}
CLINICAL_SYNTHETIC_ASSAY_CODES = {"SYNTH-FLU", "SYNTH-UNMAPPED"}
CLINICAL_SYNTHETIC_OBSERVATION_PROFILE = (
    {
        "identifier_raw": "SYNTH-FLUA^Synthetic influenza A^urn:synthetic:observation",
        "set_id": "1",
        "units_raw": "0",
        "value_type": "NM",
    },
    {
        "identifier_raw": "SYNTH-FLUA^Synthetic influenza A^urn:synthetic:observation",
        "set_id": "2",
        "units_raw": "",
        "value_type": "ST",
    },
    {
        "identifier_raw": (
            "SYNTH-FLUA-CT^Synthetic influenza A Ct^urn:synthetic:observation^"
            "S_OTHER^Synthetic Supplemental^IHE LPOCT"
        ),
        "set_id": "3",
        "units_raw": "",
        "value_type": "NM",
    },
)
CLINICAL_SYNTHETIC_MESSAGE_ID_PATTERN = re.compile(r"^SYNTH-MSG-[0-9]{3}$")
CLINICAL_SYNTHETIC_SUBJECT_ID_PATTERN = re.compile(r"^SYNTH-SUBJECT-[0-9]{3}$")
CLINICAL_SYNTHETIC_ORDER_ID_PATTERN = re.compile(r"^SYNTH-SOURCE-ORDER-[0-9]{3}$")
CLINICAL_SYNTHETIC_REPORT_ID_PATTERN = re.compile(r"^SYNTH-REPORT-[0-9]{3}$")
CLINICAL_LIVE_SOURCE_ASSERTION = (
    "deidentified_live_transport_claim_not_device_identity_or_site_validation"
)
CLINICAL_DEIDENTIFIED_TOKEN_PATTERN = re.compile(r"^DEID-[A-Za-z0-9.-]{8,64}$")
CLINICAL_DEIDENTIFIED_REFERENCE_PATTERNS = {
    "order_reference": re.compile(r"^ServiceRequest/DEID-[A-Za-z0-9.-]{8,59}$"),
    "patient_reference": re.compile(r"^Patient/DEID-[A-Za-z0-9.-]{8,59}$"),
    "specimen_reference": re.compile(r"^Specimen/DEID-[A-Za-z0-9.-]{8,59}$"),
}
CLINICAL_EXPORT_TRUTH_BOUNDARIES = {
    "MOCK": "synthetic_offline_artifact_not_clinical_delivery",
    "LIVE_SHADOW": "deidentified_live_shadow_artifact_not_clinical_delivery",
}
CLINICAL_REQUIRED_GATES = (
    "SOURCE_PROFILE_ID_MATCH",
    "SOURCE_SENDER_CONFIG_MATCH",
    "SOURCE_CONFIG_SELF_HASH_VALID",
    "RAW_HASH_RECOMPUTED",
    "MESSAGE_CONTROL_ID_NOT_PREVIOUSLY_SEEN",
    "MESSAGE_TYPE_SHAPE_ALLOWED",
    "HL7_VERSION_FIELD_ALLOWED",
    "CHARSET_FIELD_ALLOWED",
    "BOUNDED_MESSAGE_SYNTAX_VALID",
    "ASSAY_MAP_MATCH",
    "OBSERVATION_SCHEMA_VALID",
    "SOURCE_RESULT_STATUS_CAPTURED",
    "FINAL_STATUS_REQUIRED_FOR_EXPORT",
    "SPECIAL_VALUE_LEXEMES_PRESERVED",
    "EMPTY_VALUE_PRESERVED",
    "SUBJECT_BINDING_SELF_CONSISTENT",
    "ORDER_BINDING_SELF_CONSISTENT",
    "SPECIMEN_REFERENCE_PRESENT",
    "RECIPIENT_ID_PRESENT",
    "LOCAL_PROVENANCE_FIELDS_PRESENT",
    "LOCAL_POLICY_HASH_BOUND",
    "LOCAL_MAPPING_HASH_BOUND",
    "SUPERSESSION_LINEAGE_VALID",
)
CLINICAL_APPEND_ONLY_TABLES = (
    "clinical_sources",
    "clinical_reviewers",
    "clinical_messages",
    "clinical_ingest_attempts",
    "clinical_results",
    "clinical_supersession_claims",
    "clinical_observations",
    "clinical_policy_events",
    "clinical_transition_events",
    "clinical_review_attestations",
    "clinical_exports",
    "clinical_audit_events",
)


class ControlError(Exception):
    """Stable controlled failure."""

    def __init__(self, code: str, message: str, exit_code: int = EXIT_DENIED):
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code


class IntegrityFailure(ControlError):
    def __init__(self, message: str):
        super().__init__("AUDIT_INTEGRITY_FAILURE", message, EXIT_INTERNAL)


class StrictArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        emit_error("CLI_USAGE_ERROR", message, EXIT_INPUT)


def require_cryptography() -> None:
    if CRYPTOGRAPHY_IMPORT_ERROR is not None:
        raise ControlError(
            "DEPENDENCY_MISSING",
            "install the tested dependency range with: python -m pip install 'cryptography>=43,<51'",
            EXIT_INTERNAL,
        )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def parse_time(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not UTC_TIMESTAMP_PATTERN.fullmatch(value):
        raise ControlError(
            "INVALID_ENVELOPE",
            f"{field} must be an RFC3339 UTC timestamp ending in Z",
            EXIT_INPUT,
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ControlError(
            "INVALID_ENVELOPE", f"{field} is not a valid RFC3339 timestamp", EXIT_INPUT
        ) from exc


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ControlError("INVALID_JSON", f"value is not canonical JSON: {exc}", EXIT_INPUT) from exc


def pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)


def pretty_json_bytes(value: Any) -> bytes:
    """Exact UTF-8 JSON artifact bytes, including one terminal LF."""
    return (pretty_json(value) + "\n").encode("utf-8")


def emit(value: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")


def emit_error(code: str, message: str, exit_code: int) -> None:
    sys.stderr.write(
        json.dumps(
            {"code": code, "message": message, "ok": False},
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    )
    raise SystemExit(exit_code)


def b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def b64url_decode(value: Any, *, exact_bytes: int | None = None, minimum_bytes: int = 0) -> bytes:
    if not isinstance(value, str) or not value or not B64URL_PATTERN.fullmatch(value):
        raise ControlError("INVALID_BASE64URL", "invalid unpadded base64url value", EXIT_INPUT)
    try:
        decoded = base64.b64decode(
            value + "=" * (-len(value) % 4), altchars=b"-_", validate=True
        )
    except (ValueError, TypeError) as exc:
        raise ControlError("INVALID_BASE64URL", "invalid unpadded base64url value", EXIT_INPUT) from exc
    if b64url_encode(decoded) != value:
        raise ControlError("INVALID_BASE64URL", "non-canonical base64url value", EXIT_INPUT)
    if exact_bytes is not None and len(decoded) != exact_bytes:
        raise ControlError(
            "INVALID_BASE64URL", f"decoded value must be exactly {exact_bytes} bytes", EXIT_INPUT
        )
    if len(decoded) < minimum_bytes:
        raise ControlError(
            "INVALID_BASE64URL", f"decoded value must contain at least {minimum_bytes} bytes", EXIT_INPUT
        )
    return decoded


def _reject_constant(value: str) -> Any:
    raise ControlError("INVALID_JSON", f"non-finite JSON number is forbidden: {value}", EXIT_INPUT)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ControlError("DUPLICATE_JSON_KEY", f"duplicate JSON key: {key}", EXIT_INPUT)
        result[key] = value
    return result


def _validate_json_tree(value: Any, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ControlError("JSON_TOO_DEEP", "JSON nesting exceeds the limit", EXIT_INPUT)
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise ControlError("INVALID_JSON_TYPE", "floating-point values are forbidden", EXIT_INPUT)
    if isinstance(value, list):
        for child in value:
            _validate_json_tree(child, depth + 1)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ControlError("INVALID_JSON_TYPE", "JSON object keys must be strings", EXIT_INPUT)
            _validate_json_tree(child, depth + 1)
        return
    raise ControlError("INVALID_JSON_TYPE", f"unsupported JSON type: {type(value).__name__}", EXIT_INPUT)


def parse_json_bytes(raw: bytes) -> Any:
    if len(raw) > MAX_JSON_BYTES:
        raise ControlError("JSON_TOO_LARGE", "JSON input exceeds 64 KiB", EXIT_INPUT)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ControlError("INVALID_UTF8", "JSON input is not valid UTF-8", EXIT_INPUT) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except ControlError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ControlError("INVALID_JSON", f"unable to parse JSON: {exc}", EXIT_INPUT) from exc
    _validate_json_tree(value)
    return value


def load_json_file(path: Path) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ControlError("FILE_READ_FAILED", f"unable to read {path}: {exc}", EXIT_INPUT) from exc
    return parse_json_bytes(raw)


def atomic_write(path: Path, data: bytes, mode: int = 0o600) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, mode)
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def safe_new_or_identical_output(path: Path, data: bytes, label: str) -> Path:
    """Publish complete bytes with an atomic no-replace link, or accept an identical file."""
    requested = Path(os.path.abspath(path.expanduser()))
    cursor = requested.parent
    while cursor != cursor.parent:
        if cursor.exists() and _is_reparse_point(cursor):
            raise ControlError(
                "UNSAFE_OUTPUT_PATH",
                f"{label} parent path may not contain a reparse point",
                EXIT_DENIED,
            )
        cursor = cursor.parent
    requested.parent.mkdir(parents=True, exist_ok=True)

    def accept_identical_existing() -> None:
        if _is_reparse_point(requested) or not requested.is_file():
            raise ControlError("UNSAFE_OUTPUT_PATH", f"{label} target is unsafe", EXIT_DENIED)
        try:
            existing = requested.read_bytes()
        except OSError as exc:
            raise ControlError("OUTPUT_READ_FAILED", f"unable to inspect {label}: {exc}", EXIT_INPUT) from exc
        if not secrets.compare_digest(existing, data):
            raise ControlError(
                "OUTPUT_ALREADY_EXISTS",
                f"{label} already exists with different content; refusing overwrite",
                EXIT_CONFLICT,
            )

    if requested.exists():
        accept_identical_existing()
    else:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{requested.name}.", suffix=".publish", dir=requested.parent
        )
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb", closefd=True) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            staged = Path(temp_name)
            _apply_owner_only_acl(staged)
            try:
                os.link(staged, requested)
            except FileExistsError:
                accept_identical_existing()
            except OSError as exc:
                raise ControlError(
                    "OUTPUT_PUBLISH_FAILED",
                    f"unable to publish {label} without replacement: {exc}",
                    EXIT_INPUT,
                ) from exc
        finally:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
    _apply_owner_only_acl(requested)
    return requested.resolve()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ControlError("EXECUTABLE_READ_FAILED", f"unable to hash executable: {exc}", EXIT_INPUT) from exc
    return digest.hexdigest()


def validate_target(value: Any) -> str:
    if not isinstance(value, str) or not TARGET_PATTERN.fullmatch(value):
        raise ControlError(
            "INVALID_TARGET",
            "target must match owned-agent:[a-z0-9][a-z0-9._-]{1,127}",
            EXIT_INPUT,
        )
    return value


def validate_operator(value: Any) -> str:
    if not isinstance(value, str) or not OPERATOR_PATTERN.fullmatch(value):
        raise ControlError(
            "INVALID_OPERATOR", "operator id must match [a-z][a-z0-9_-]{1,31}", EXIT_INPUT
        )
    return value


def validate_timeout(value: Any, field: str = "timeout") -> float:
    if isinstance(value, bool):
        raise ControlError("INVALID_TIMEOUT", f"{field} must be a finite number", EXIT_INPUT)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ControlError("INVALID_TIMEOUT", f"{field} must be a finite number", EXIT_INPUT) from exc
    if not math.isfinite(parsed) or not 0.1 <= parsed <= 60.0:
        raise ControlError(
            "INVALID_TIMEOUT", f"{field} must be from 0.1 through 60 seconds", EXIT_INPUT
        )
    return parsed


def require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ControlError("INVALID_ENVELOPE", f"{label} must be an object", EXIT_INPUT)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ControlError(
            "INVALID_ENVELOPE",
            f"{label} fields mismatch; missing={missing}, extra={extra}",
            EXIT_INPUT,
        )
    return value


@dataclass(frozen=True)
class StatePaths:
    root: Path
    database: Path
    lock: Path
    logs: Path
    demo: Path


def windows_fixed_drive(path: Path) -> bool:
    if os.name != "nt":
        return True
    anchor = path.anchor
    if not re.fullmatch(r"[A-Za-z]:\\", anchor):
        return False
    drive_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    drive_kernel32.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
    drive_kernel32.GetDriveTypeW.restype = wintypes.UINT
    DRIVE_FIXED = 3
    return int(drive_kernel32.GetDriveTypeW(anchor)) == DRIVE_FIXED


def state_paths(value: str | Path) -> StatePaths:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise ControlError("STATE_PATH_NOT_ABSOLUTE", "--state-dir must be absolute", EXIT_INPUT)
    root = Path(os.path.abspath(raw))
    if os.name == "nt" and (
        str(root).startswith("\\\\")
        or not re.match(r"^[A-Za-z]:[\\/]", str(root))
    ):
        raise ControlError(
            "STATE_PATH_NOT_LOCAL",
            "--state-dir must be a local drive path, not UNC or a device path",
            EXIT_INPUT,
        )
    if not windows_fixed_drive(root):
        raise ControlError(
            "STATE_PATH_NOT_LOCAL",
            "--state-dir must be on a fixed local drive, not mapped, remote, or removable media",
            EXIT_INPUT,
        )
    return StatePaths(
        root=root,
        database=root / "control.sqlite3",
        lock=root / "config.lock",
        logs=root / "logs",
        demo=root / "demo",
    )


def _is_reparse_point(path: Path) -> bool:
    if os.name != "nt":
        return path.is_symlink()
    try:
        attrs = os.stat(path, follow_symlinks=False).st_file_attributes
    except (AttributeError, OSError):
        return path.is_symlink()
    return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def ensure_state_root(paths: StatePaths, *, create: bool = False) -> None:
    if create:
        if paths.root.exists():
            try:
                if any(paths.root.iterdir()):
                    raise ControlError(
                        "STATE_DIRECTORY_NOT_EMPTY",
                        "initialization requires a new or empty state directory",
                        EXIT_CONFLICT,
                    )
            except OSError as exc:
                raise ControlError(
                    "STATE_PATH_UNREADABLE", f"unable to inspect state directory: {exc}", EXIT_INPUT
                ) from exc
        paths.root.mkdir(parents=True, exist_ok=True)
    if not paths.root.is_dir():
        raise ControlError("STATE_NOT_INITIALIZED", "state directory does not exist", EXIT_CONFLICT)
    if _is_reparse_point(paths.root):
        raise ControlError("STATE_REPARSE_POINT", "state directory may not be a reparse point", EXIT_DENIED)
    for fixed in (paths.database, paths.lock, paths.logs, paths.demo):
        if fixed.exists() and _is_reparse_point(fixed):
            raise ControlError(
                "STATE_REPARSE_POINT",
                f"fixed controller path may not be a reparse point: {fixed.name}",
                EXIT_DENIED,
            )


def prepare_controller_initialization_root(paths: StatePaths) -> None:
    """Allow only fixed controller artifacts so a failed initialization can retry safely."""
    paths.root.mkdir(parents=True, exist_ok=True)
    ensure_state_root(paths)
    allowed = {paths.database.name, paths.lock.name, paths.logs.name, paths.demo.name}
    try:
        unexpected = sorted(entry.name for entry in paths.root.iterdir() if entry.name not in allowed)
    except OSError as exc:
        raise ControlError(
            "STATE_PATH_UNREADABLE", f"unable to inspect state directory: {exc}", EXIT_INPUT
        ) from exc
    if unexpected:
        raise ControlError(
            "STATE_DIRECTORY_NOT_EMPTY",
            "initialization root contains unexpected entries: " + ",".join(unexpected),
            EXIT_CONFLICT,
        )
    if paths.lock.exists() and not paths.lock.is_file():
        raise ControlError("STATE_PATH_CONFLICT", "config.lock is not a regular file", EXIT_CONFLICT)
    for directory in (paths.logs, paths.demo):
        if directory.exists() and not directory.is_dir():
            raise ControlError(
                "STATE_PATH_CONFLICT", f"{directory.name} is not a directory", EXIT_CONFLICT
            )


def _apply_owner_only_acl(path: Path) -> dict[str, Any]:
    if os.name != "nt":
        os.chmod(path, 0o700 if path.is_dir() else 0o600)
        return {"platform": os.name, "owner_only_mode_applied": True, "windows_dacl_applied": None}

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    TOKEN_QUERY = 0x0008
    TOKEN_USER = 1
    DACL_SECURITY_INFORMATION = 0x00000004
    PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
    SDDL_REVISION_1 = 1

    class SID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]

    class TOKEN_USER_STRUCT(ctypes.Structure):
        _fields_ = [("User", SID_AND_ATTRIBUTES)]

    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.ConvertSidToStringSidW.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.LPWSTR),
    ]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
    advapi32.SetFileSecurityW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    advapi32.SetFileSecurityW.restype = wintypes.BOOL

    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), TOKEN_QUERY, ctypes.byref(token)):
        raise ControlError("STATE_ACL_FAILED", f"OpenProcessToken failed: {ctypes.get_last_error()}", EXIT_INTERNAL)
    try:
        needed = wintypes.DWORD(0)
        advapi32.GetTokenInformation(token, TOKEN_USER, None, 0, ctypes.byref(needed))
        buffer = ctypes.create_string_buffer(needed.value)
        if not advapi32.GetTokenInformation(
            token, TOKEN_USER, buffer, needed, ctypes.byref(needed)
        ):
            raise ControlError(
                "STATE_ACL_FAILED", f"GetTokenInformation failed: {ctypes.get_last_error()}", EXIT_INTERNAL
            )
        token_user = ctypes.cast(buffer, ctypes.POINTER(TOKEN_USER_STRUCT)).contents
        sid_pointer = token_user.User.Sid
        sid_string = wintypes.LPWSTR()
        if not advapi32.ConvertSidToStringSidW(sid_pointer, ctypes.byref(sid_string)):
            raise ControlError(
                "STATE_ACL_FAILED", f"ConvertSidToStringSidW failed: {ctypes.get_last_error()}", EXIT_INTERNAL
            )
        try:
            sid = ctypes.wstring_at(sid_string)
        finally:
            kernel32.LocalFree(sid_string)
    finally:
        kernel32.CloseHandle(token)

    sddl = f"D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;{sid})"
    security_descriptor = ctypes.c_void_p()
    if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        sddl, SDDL_REVISION_1, ctypes.byref(security_descriptor), None
    ):
        raise ControlError(
            "STATE_ACL_FAILED",
            f"ConvertStringSecurityDescriptor failed: {ctypes.get_last_error()}",
            EXIT_INTERNAL,
        )
    try:
        if not advapi32.SetFileSecurityW(
            str(path),
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            security_descriptor,
        ):
            raise ControlError(
                "STATE_ACL_FAILED", f"SetFileSecurity failed: {ctypes.get_last_error()}", EXIT_INTERNAL
            )
    finally:
        kernel32.LocalFree(security_descriptor)
    return {"platform": "windows", "owner_only_mode_applied": True, "windows_dacl_applied": True}


class ConfigurationLock:
    def __init__(self, path: Path):
        self.path = path
        self.stream: Any = None

    def __enter__(self) -> "ConfigurationLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0, os.SEEK_END)
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            self.stream = None
            raise ControlError("CONFIGURATION_BUSY", "another configuration operation is active", EXIT_CONFLICT) from exc
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.stream is None:
            return
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()
            self.stream = None


class ClosingSQLiteConnection(sqlite3.Connection):
    """Commit or roll back like sqlite3.Connection, then actually close the handle."""

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc, tb))
        finally:
            self.close()


def configure_controller_connection(
    connection: sqlite3.Connection,
    *,
    initialize_wal: bool = False,
) -> None:
    connection.execute("PRAGMA busy_timeout = 10000")
    if initialize_wal:
        journal_mode = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0])
    else:
        journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0])
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA synchronous = FULL")
    if journal_mode.lower() != "wal":
        raise IntegrityFailure("controller database journal mode is not WAL")
    if int(connection.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
        raise IntegrityFailure("controller database foreign-key enforcement is disabled")
    if int(connection.execute("PRAGMA synchronous").fetchone()[0]) < 2:
        raise IntegrityFailure("controller database synchronous durability is below FULL")


def refuse_existing_controller_database(database: Path) -> None:
    if not database.exists():
        return
    complete = False
    if database.is_file() and not _is_reparse_point(database):
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                database.resolve().as_uri() + "?mode=ro",
                timeout=2.0,
                isolation_level=None,
                uri=True,
                factory=ClosingSQLiteConnection,
            )
            connection.row_factory = sqlite3.Row
            rows = {
                str(row["key"]): str(row["value"])
                for row in connection.execute(
                    "SELECT key,value FROM metadata "
                    "WHERE key IN ('schema_version','initialization_complete')"
                )
            }
            complete = rows == {
                "initialization_complete": "1",
                "schema_version": STATE_SCHEMA_VERSION,
            }
        except sqlite3.DatabaseError:
            complete = False
        finally:
            if connection is not None:
                connection.close()
    if complete:
        raise ControlError("STATE_ALREADY_INITIALIZED", "control database already exists", EXIT_CONFLICT)
    raise ControlError(
        "STATE_INCOMPLETE",
        "a pre-existing controller database is incomplete, unsafe, or from another schema; preserving it",
        EXIT_CONFLICT,
    )


def connect(paths: StatePaths) -> sqlite3.Connection:
    ensure_state_root(paths)
    if not paths.database.is_file():
        raise ControlError("STATE_NOT_INITIALIZED", "control database does not exist", EXIT_CONFLICT)
    if _is_reparse_point(paths.database):
        raise ControlError(
            "STATE_DATABASE_REPARSE_POINT",
            "control database may not be a reparse point",
            EXIT_DENIED,
        )
    connection = sqlite3.connect(
        paths.database,
        timeout=10.0,
        isolation_level=None,
        factory=ClosingSQLiteConnection,
    )
    connection.row_factory = sqlite3.Row
    try:
        configure_controller_connection(connection)
        complete = connection.execute(
            "SELECT value FROM metadata WHERE key='initialization_complete'"
        ).fetchone()
    except (sqlite3.DatabaseError, ControlError) as exc:
        connection.close()
        if isinstance(exc, ControlError) and not isinstance(exc, IntegrityFailure):
            raise
        raise ControlError(
            "STATE_INCOMPLETE", "controller state is not a complete trusted schema", EXIT_CONFLICT
        ) from exc
    if complete is None or str(complete["value"]) != "1":
        connection.close()
        raise ControlError(
            "STATE_INCOMPLETE", "controller initialization-completion marker is missing", EXIT_CONFLICT
        )
    return connection


class immediate_transaction:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def __enter__(self) -> sqlite3.Connection:
        self.connection.execute("BEGIN IMMEDIATE")
        return self.connection

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc_type is None:
            self.connection.execute("COMMIT")
        else:
            self.connection.execute("ROLLBACK")
        return False


def initialize_state(paths: StatePaths) -> dict[str, Any]:
    if os.name != "nt":
        raise ControlError(
            "WINDOWS_REQUIRED",
            "the operational enforcement path requires Windows Job Objects",
            EXIT_INTERNAL,
        )
    prepare_controller_initialization_root(paths)
    acl_result = _apply_owner_only_acl(paths.root)
    paths.logs.mkdir(exist_ok=True)
    paths.demo.mkdir(exist_ok=True)
    _apply_owner_only_acl(paths.logs)
    _apply_owner_only_acl(paths.demo)
    refuse_existing_controller_database(paths.database)
    with ConfigurationLock(paths.lock):
        refuse_existing_controller_database(paths.database)
        descriptor, staged_name = tempfile.mkstemp(
            prefix=".control.sqlite3.", suffix=".initialize", dir=paths.root
        )
        os.close(descriptor)
        staged = Path(staged_name)
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                staged,
                timeout=10.0,
                isolation_level=None,
                factory=ClosingSQLiteConnection,
            )
            connection.row_factory = sqlite3.Row
            configure_controller_connection(connection, initialize_wal=True)
            connection.executescript(
                """
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                ) STRICT;

                CREATE TABLE operators (
                    operator_id TEXT PRIMARY KEY,
                    public_key_b64 TEXT NOT NULL,
                    fingerprint TEXT NOT NULL UNIQUE,
                    enabled INTEGER NOT NULL CHECK(enabled IN (0, 1)),
                    created_at TEXT NOT NULL
                ) STRICT;

                CREATE TABLE targets (
                    target TEXT PRIMARY KEY,
                    argv_json TEXT NOT NULL,
                    cwd TEXT NOT NULL,
                    executable_sha256 TEXT NOT NULL,
                    control_state TEXT NOT NULL CHECK(control_state IN ('READY', 'ISOLATED')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                ) STRICT;

                CREATE TABLE runs (
                    run_id TEXT PRIMARY KEY,
                    target TEXT NOT NULL REFERENCES targets(target),
                    state TEXT NOT NULL CHECK(state IN (
                        'STARTING', 'RUNNING', 'ISOLATING', 'ISOLATED', 'EXITED', 'FAILED'
                    )),
                    job_name TEXT NOT NULL UNIQUE,
                    supervisor_token_hash TEXT,
                    supervisor_pid INTEGER,
                    supervisor_created_filetime INTEGER,
                    child_pid INTEGER,
                    credential_hash TEXT,
                    credential_expires_at TEXT,
                    credential_revoked_at TEXT,
                    job_active_processes INTEGER,
                    heartbeat_at TEXT,
                    started_at TEXT,
                    ended_at TEXT,
                    log_path TEXT NOT NULL,
                    error_code TEXT,
                    error_message TEXT,
                    exit_code INTEGER,
                    created_at TEXT NOT NULL
                ) STRICT;

                CREATE UNIQUE INDEX one_live_run_per_target
                ON runs(target)
                WHERE state IN ('STARTING', 'RUNNING', 'ISOLATING');

                CREATE TABLE requests (
                    request_id TEXT PRIMARY KEY,
                    nonce TEXT NOT NULL UNIQUE,
                    envelope_sha256 TEXT NOT NULL UNIQUE,
                    target TEXT NOT NULL REFERENCES targets(target),
                    status TEXT NOT NULL CHECK(status IN (
                        'ENFORCING', 'APPLIED', 'ENFORCEMENT_UNCONFIRMED'
                    )),
                    accepted_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    applied_at TEXT,
                    result_json TEXT
                ) STRICT;

                CREATE UNIQUE INDEX one_enforcing_request_per_target
                ON requests(target)
                WHERE status IN ('ENFORCING', 'ENFORCEMENT_UNCONFIRMED');

                CREATE TABLE audit_events (
                    sequence INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    target TEXT,
                    actor TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE
                ) STRICT;

                CREATE TRIGGER audit_events_no_update
                BEFORE UPDATE ON audit_events
                BEGIN SELECT RAISE(ABORT, 'audit_events are append-only'); END;

                CREATE TRIGGER audit_events_no_delete
                BEFORE DELETE ON audit_events
                BEGIN SELECT RAISE(ABORT, 'audit_events are append-only'); END;

                CREATE TRIGGER operators_no_update
                BEFORE UPDATE ON operators
                BEGIN SELECT RAISE(ABORT, 'operators are immutable'); END;

                CREATE TRIGGER operators_no_delete
                BEFORE DELETE ON operators
                BEGIN SELECT RAISE(ABORT, 'operators are immutable'); END;

                CREATE TRIGGER targets_no_unisolate
                BEFORE UPDATE OF control_state ON targets
                WHEN OLD.control_state='ISOLATED' AND NEW.control_state<>'ISOLATED'
                BEGIN SELECT RAISE(ABORT, 'target isolation is irreversible'); END;

                CREATE TRIGGER trust_store_no_unseal
                BEFORE UPDATE OF value ON metadata
                WHEN OLD.key='trust_store_sealed' AND OLD.value='1' AND NEW.value<>'1'
                BEGIN SELECT RAISE(ABORT, 'trust store sealing is irreversible'); END;

                CREATE TRIGGER metadata_no_delete
                BEFORE DELETE ON metadata
                BEGIN SELECT RAISE(ABORT, 'controller metadata may not be deleted'); END;
                """
            )
            now = format_time(utc_now())
            with immediate_transaction(connection):
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
                    (STATE_SCHEMA_VERSION,),
                )
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('initialization_complete', '1')"
                )
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('audit_tip', ?)",
                    (ZERO_HASH,),
                )
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('trust_store_sealed', '0')"
                )
                append_audit(
                    connection,
                    "STATE_INITIALIZED",
                    None,
                    "local-operator",
                    {
                        "program": PROGRAM,
                        "program_version": PROGRAM_VERSION,
                        "schema_version": STATE_SCHEMA_VERSION,
                        "truth_boundary": "local_windows_supervised_processes_only",
                    },
                    timestamp=now,
                )
            verify_audit(connection)
            checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint is None or int(checkpoint[0]) != 0 or int(checkpoint[1]) != int(checkpoint[2]):
                raise IntegrityFailure("controller initialization WAL checkpoint did not complete")
            connection.close()
            connection = None
            _apply_owner_only_acl(staged)
            try:
                os.link(staged, paths.database)
            except FileExistsError as exc:
                raise ControlError(
                    "STATE_ALREADY_INITIALIZED", "control database already exists", EXIT_CONFLICT
                ) from exc
            except OSError as exc:
                raise ControlError(
                    "STATE_PUBLISH_FAILED",
                    f"unable to atomically publish controller state: {exc}",
                    EXIT_INPUT,
                ) from exc
        finally:
            if connection is not None:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                connection.close()
            for suffix in ("", "-journal", "-shm", "-wal"):
                try:
                    Path(str(staged) + suffix).unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
    return {
        "acl": acl_result,
        "database": str(paths.database),
        "enforcement_scope": "local_windows_supervised_processes_only",
        "ok": True,
        "operation": "init",
        "state_dir": str(paths.root),
        "trust_store_sealed": False,
    }


def audit_body(
    sequence: int,
    timestamp: str,
    event_type: str,
    target: str | None,
    actor: str,
    payload: Any,
) -> dict[str, Any]:
    return {
        "actor": actor,
        "event_type": event_type,
        "payload": payload,
        "sequence": sequence,
        "target": target,
        "timestamp": timestamp,
        "version": 1,
    }


def append_audit(
    connection: sqlite3.Connection,
    event_type: str,
    target: str | None,
    actor: str,
    payload: Any,
    *,
    timestamp: str | None = None,
) -> str:
    row = connection.execute(
        "SELECT sequence, event_hash FROM audit_events ORDER BY sequence DESC LIMIT 1"
    ).fetchone()
    if row is None:
        sequence = 1
        previous_hash = ZERO_HASH
    else:
        sequence = int(row["sequence"]) + 1
        previous_hash = str(row["event_hash"])
    tip_row = connection.execute(
        "SELECT value FROM metadata WHERE key='audit_tip'"
    ).fetchone()
    if tip_row is None or str(tip_row["value"]) != previous_hash:
        raise IntegrityFailure("stored audit tip does not match the event chain")
    timestamp_value = timestamp or format_time(utc_now())
    payload_bytes = canonical_json(payload)
    payload_json = payload_bytes.decode("utf-8")
    body = audit_body(sequence, timestamp_value, event_type, target, actor, payload)
    event_hash = hashlib.sha256(bytes.fromhex(previous_hash) + canonical_json(body)).hexdigest()
    connection.execute(
        """
        INSERT INTO audit_events(
            sequence, timestamp, event_type, target, actor, payload_json,
            previous_hash, event_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sequence,
            timestamp_value,
            event_type,
            target,
            actor,
            payload_json,
            previous_hash,
            event_hash,
        ),
    )
    connection.execute(
        "UPDATE metadata SET value=? WHERE key='audit_tip'", (event_hash,)
    )
    return event_hash


def verify_audit(connection: sqlite3.Connection) -> dict[str, Any]:
    """Verify one stable SQLite snapshot, even when called in autocommit mode."""

    if connection.in_transaction:
        return _verify_audit_in_current_transaction(connection)
    connection.execute("BEGIN")
    try:
        result = _verify_audit_in_current_transaction(connection)
        connection.execute("COMMIT")
        return result
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _verify_audit_in_current_transaction(connection: sqlite3.Connection) -> dict[str, Any]:
    schema_row = connection.execute(
        "SELECT value FROM metadata WHERE key='schema_version'"
    ).fetchone()
    if schema_row is None or str(schema_row["value"]) != STATE_SCHEMA_VERSION:
        raise IntegrityFailure("controller schema version is missing or unsupported")
    complete_row = connection.execute(
        "SELECT value FROM metadata WHERE key='initialization_complete'"
    ).fetchone()
    if complete_row is None or str(complete_row["value"]) != "1":
        raise IntegrityFailure("controller initialization-completion marker is invalid")
    trigger_rows = connection.execute(
        """
        SELECT name, sql FROM sqlite_master
        WHERE type='trigger' AND name IN (
            'audit_events_no_update','audit_events_no_delete',
            'operators_no_update','operators_no_delete','targets_no_unisolate',
            'trust_store_no_unseal','metadata_no_delete'
        )
        """
    ).fetchall()
    trigger_sql = {
        str(row["name"]): " ".join(str(row["sql"] or "").upper().split())
        for row in trigger_rows
    }
    expected_trigger_fragments = {
        "audit_events_no_update": (
            "BEFORE UPDATE ON AUDIT_EVENTS",
            "RAISE(ABORT, 'AUDIT_EVENTS ARE APPEND-ONLY')",
        ),
        "audit_events_no_delete": (
            "BEFORE DELETE ON AUDIT_EVENTS",
            "RAISE(ABORT, 'AUDIT_EVENTS ARE APPEND-ONLY')",
        ),
        "operators_no_update": (
            "BEFORE UPDATE ON OPERATORS",
            "RAISE(ABORT, 'OPERATORS ARE IMMUTABLE')",
        ),
        "operators_no_delete": (
            "BEFORE DELETE ON OPERATORS",
            "RAISE(ABORT, 'OPERATORS ARE IMMUTABLE')",
        ),
        "targets_no_unisolate": (
            "BEFORE UPDATE OF CONTROL_STATE ON TARGETS",
            "OLD.CONTROL_STATE='ISOLATED'",
            "RAISE(ABORT, 'TARGET ISOLATION IS IRREVERSIBLE')",
        ),
        "trust_store_no_unseal": (
            "BEFORE UPDATE OF VALUE ON METADATA",
            "OLD.KEY='TRUST_STORE_SEALED'",
            "RAISE(ABORT, 'TRUST STORE SEALING IS IRREVERSIBLE')",
        ),
        "metadata_no_delete": (
            "BEFORE DELETE ON METADATA",
            "RAISE(ABORT, 'CONTROLLER METADATA MAY NOT BE DELETED')",
        ),
    }
    for name, fragments in expected_trigger_fragments.items():
        sql = trigger_sql.get(name, "")
        if not all(fragment in sql for fragment in fragments):
            raise IntegrityFailure(f"required append-only audit trigger is missing or changed: {name}")
    index_row = connection.execute(
        """
        SELECT sql FROM sqlite_master
        WHERE type='index' AND name='one_enforcing_request_per_target'
        """
    ).fetchone()
    index_sql = " ".join(str(index_row["sql"] or "").upper().split()) if index_row else ""
    if not all(
        fragment in index_sql
        for fragment in (
            "UNIQUE INDEX ONE_ENFORCING_REQUEST_PER_TARGET",
            "ON REQUESTS(TARGET)",
            "STATUS IN ('ENFORCING', 'ENFORCEMENT_UNCONFIRMED')",
        )
    ):
        raise IntegrityFailure("required single-enforcement request index is missing or changed")
    live_index_row = connection.execute(
        """
        SELECT sql FROM sqlite_master
        WHERE type='index' AND name='one_live_run_per_target'
        """
    ).fetchone()
    live_index_sql = (
        " ".join(str(live_index_row["sql"] or "").upper().split())
        if live_index_row
        else ""
    )
    if not all(
        fragment in live_index_sql
        for fragment in (
            "UNIQUE INDEX ONE_LIVE_RUN_PER_TARGET",
            "ON RUNS(TARGET)",
            "STATE IN ('STARTING', 'RUNNING', 'ISOLATING')",
        )
    ):
        raise IntegrityFailure("required single-live-run index is missing or changed")
    expected_previous = ZERO_HASH
    expected_sequence = 1
    count = 0
    audited_operators: set[tuple[str, str]] = set()
    trust_store_seal_events = 0
    for row in connection.execute("SELECT * FROM audit_events ORDER BY sequence"):
        sequence = int(row["sequence"])
        if sequence != expected_sequence:
            raise IntegrityFailure(
                f"audit sequence gap: expected {expected_sequence}, observed {sequence}"
            )
        if str(row["previous_hash"]) != expected_previous:
            raise IntegrityFailure(f"audit predecessor mismatch at sequence {sequence}")
        try:
            payload = parse_json_bytes(str(row["payload_json"]).encode("utf-8"))
        except ControlError as exc:
            raise IntegrityFailure(f"audit payload is invalid at sequence {sequence}: {exc.message}") from exc
        if canonical_json(payload).decode("utf-8") != str(row["payload_json"]):
            raise IntegrityFailure(f"audit payload is not canonical at sequence {sequence}")
        body = audit_body(
            sequence,
            str(row["timestamp"]),
            str(row["event_type"]),
            row["target"],
            str(row["actor"]),
            payload,
        )
        calculated = hashlib.sha256(
            bytes.fromhex(expected_previous) + canonical_json(body)
        ).hexdigest()
        if calculated != str(row["event_hash"]):
            raise IntegrityFailure(f"audit hash mismatch at sequence {sequence}")
        if str(row["event_type"]) == "OPERATOR_ADDED":
            try:
                audited_operators.add((str(payload["operator_id"]), str(payload["fingerprint"])))
            except (KeyError, TypeError) as exc:
                raise IntegrityFailure(
                    f"operator audit payload is invalid at sequence {sequence}"
                ) from exc
        elif str(row["event_type"]) == "TRUST_STORE_SEALED":
            trust_store_seal_events += 1
        expected_previous = calculated
        expected_sequence += 1
        count += 1
    tip_row = connection.execute(
        "SELECT value FROM metadata WHERE key='audit_tip'"
    ).fetchone()
    if tip_row is None or str(tip_row["value"]) != expected_previous:
        raise IntegrityFailure("audit tip does not match the verified chain")
    database_operators = {
        (str(row["operator_id"]), str(row["fingerprint"]))
        for row in connection.execute("SELECT operator_id, fingerprint FROM operators")
    }
    if database_operators != audited_operators:
        raise IntegrityFailure("trusted operator table does not match audited operator additions")
    sealed_row = connection.execute(
        "SELECT value FROM metadata WHERE key='trust_store_sealed'"
    ).fetchone()
    if sealed_row is None or str(sealed_row["value"]) not in {"0", "1"}:
        raise IntegrityFailure("trust-store seal metadata is missing or invalid")
    sealed = str(sealed_row["value"]) == "1"
    if trust_store_seal_events != (1 if sealed else 0):
        raise IntegrityFailure("trust-store seal state does not match the audit chain")
    inconsistent_target = connection.execute(
        """
        SELECT t.target FROM targets t
        WHERE t.control_state='READY'
          AND EXISTS (SELECT 1 FROM requests r WHERE r.target=t.target)
        LIMIT 1
        """
    ).fetchone()
    if inconsistent_target is not None:
        raise IntegrityFailure("an accepted isolation request exists for a READY target")
    isolated_without_request = connection.execute(
        """
        SELECT t.target FROM targets t
        WHERE t.control_state='ISOLATED'
          AND NOT EXISTS (SELECT 1 FROM requests r WHERE r.target=t.target)
        LIMIT 1
        """
    ).fetchone()
    if isolated_without_request is not None:
        raise IntegrityFailure("an ISOLATED target has no accepted isolation request")
    return {
        "audit_tip": expected_previous,
        "events_verified": count,
        "integrity": "VERIFIED_LOCAL_HASH_CHAIN",
        "immutability": False,
        "ok": True,
        "tail_truncation_without_external_anchor_detectable": False,
        "trust_store_sealed": sealed,
    }


def public_key_raw(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def public_fingerprint(public_key: Ed25519PublicKey) -> str:
    return hashlib.sha256(public_key_raw(public_key)).hexdigest()


def generate_operator_keypair(
    operator_id: str,
    private_path: Path,
    public_path: Path,
    passphrase: bytes,
) -> dict[str, Any]:
    require_cryptography()
    operator_id = validate_operator(operator_id)
    if len(passphrase) < 12:
        raise ControlError(
            "WEAK_KEY_PASSPHRASE", "private-key passphrase must contain at least 12 bytes", EXIT_INPUT
        )
    private_path = Path(os.path.abspath(private_path.expanduser()))
    public_path = Path(os.path.abspath(public_path.expanduser()))
    if private_path.exists() or public_path.exists():
        raise ControlError("KEY_FILE_EXISTS", "refusing to overwrite an existing key file", EXIT_CONFLICT)
    if private_path == public_path:
        raise ControlError("INVALID_KEY_PATH", "private and public key paths must differ", EXIT_INPUT)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    key = Ed25519PrivateKey.generate()
    encrypted_private = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.BestAvailableEncryption(passphrase),
    )
    public_bytes = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    private_created = False
    public_created = False
    try:
        private_fd = os.open(
            private_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        private_created = True
        with os.fdopen(private_fd, "wb") as stream:
            stream.write(encrypted_private)
            stream.flush()
            os.fsync(stream.fileno())
        public_fd = os.open(
            public_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        public_created = True
        with os.fdopen(public_fd, "wb") as stream:
            stream.write(public_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        _apply_owner_only_acl(private_path)
        _apply_owner_only_acl(public_path)
    except BaseException:
        for path, created in (
            (private_path, private_created),
            (public_path, public_created),
        ):
            if not created:
                continue
            try:
                path.unlink()
            except OSError:
                pass
        raise
    return {
        "fingerprint": public_fingerprint(key.public_key()),
        "ok": True,
        "operation": "keygen",
        "operator_id": operator_id,
        "private_key": str(private_path),
        "public_key": str(public_path),
    }


def load_public_key(path: Path) -> Ed25519PublicKey:
    require_cryptography()
    try:
        value = serialization.load_pem_public_key(path.read_bytes())
    except (OSError, ValueError, TypeError) as exc:
        raise ControlError("PUBLIC_KEY_INVALID", f"unable to load public key: {exc}", EXIT_INPUT) from exc
    if not isinstance(value, Ed25519PublicKey):
        raise ControlError("PUBLIC_KEY_INVALID", "public key must be Ed25519", EXIT_INPUT)
    return value


def load_private_key(path: Path, passphrase: bytes) -> Ed25519PrivateKey:
    require_cryptography()
    try:
        value = serialization.load_pem_private_key(path.read_bytes(), password=passphrase)
    except (OSError, ValueError, TypeError) as exc:
        raise ControlError("PRIVATE_KEY_INVALID", f"unable to load private key: {exc}", EXIT_INPUT) from exc
    if not isinstance(value, Ed25519PrivateKey):
        raise ControlError("PRIVATE_KEY_INVALID", "private key must be Ed25519", EXIT_INPUT)
    return value


def add_operator_raw(
    paths: StatePaths,
    operator_id: str,
    public_key: Ed25519PublicKey,
    *,
    actor: str = "local-operator",
) -> dict[str, Any]:
    require_cryptography()
    operator_id = validate_operator(operator_id)
    raw = public_key_raw(public_key)
    fingerprint = hashlib.sha256(raw).hexdigest()
    with ConfigurationLock(paths.lock), connect(paths) as connection:
        with immediate_transaction(connection):
            verify_audit(connection)
            sealed_row = connection.execute(
                "SELECT value FROM metadata WHERE key='trust_store_sealed'"
            ).fetchone()
            if sealed_row is None or str(sealed_row["value"]) != "0":
                raise ControlError(
                    "TRUST_STORE_SEALED",
                    "operator registration is permanently closed for this state root",
                    EXIT_DENIED,
                )
            try:
                connection.execute(
                    """
                    INSERT INTO operators(
                        operator_id, public_key_b64, fingerprint, enabled, created_at
                    ) VALUES (?, ?, ?, 1, ?)
                    """,
                    (operator_id, b64url_encode(raw), fingerprint, format_time(utc_now())),
                )
            except sqlite3.IntegrityError as exc:
                raise ControlError(
                    "OPERATOR_OR_KEY_ALREADY_REGISTERED",
                    "operator id or public-key fingerprint is already registered",
                    EXIT_CONFLICT,
                ) from exc
            append_audit(
                connection,
                "OPERATOR_ADDED",
                None,
                actor,
                {"fingerprint": fingerprint, "operator_id": operator_id, "scheme": SIGNATURE_SCHEME},
            )
    return {
        "enabled": True,
        "fingerprint": fingerprint,
        "ok": True,
        "operation": "operator-add",
        "operator_id": operator_id,
    }


def add_operator(paths: StatePaths, operator_id: str, public_path: Path) -> dict[str, Any]:
    return add_operator_raw(paths, operator_id, load_public_key(public_path))


def require_trust_store_sealed(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT value FROM metadata WHERE key='trust_store_sealed'"
    ).fetchone()
    if row is None or str(row["value"]) != "1":
        raise ControlError(
            "TRUST_STORE_NOT_SEALED",
            "seal the operator trust store after registering at least two distinct keys",
            EXIT_CONFLICT,
        )


def seal_trust_store(paths: StatePaths, *, actor: str = "local-operator") -> dict[str, Any]:
    with ConfigurationLock(paths.lock), connect(paths) as connection:
        with immediate_transaction(connection):
            audit = verify_audit(connection)
            if audit["trust_store_sealed"]:
                return {
                    "ok": True,
                    "operation": "trust-seal",
                    "operator_count": int(
                        connection.execute(
                            "SELECT COUNT(*) FROM operators WHERE enabled=1"
                        ).fetchone()[0]
                    ),
                    "trust_store_sealed": True,
                }
            operator_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM operators WHERE enabled=1"
                ).fetchone()[0]
            )
            if operator_count < QUORUM:
                raise ControlError(
                    "QUORUM_UNAVAILABLE",
                    "at least two distinct enabled operator keys are required before sealing",
                    EXIT_CONFLICT,
                )
            connection.execute(
                "UPDATE metadata SET value='1' WHERE key='trust_store_sealed' AND value='0'"
            )
            append_audit(
                connection,
                "TRUST_STORE_SEALED",
                None,
                actor,
                {"operator_count": operator_count, "quorum": QUORUM},
            )
    return {
        "ok": True,
        "operation": "trust-seal",
        "operator_count": operator_count,
        "trust_store_sealed": True,
    }


def validate_argv(argv: Any) -> list[str]:
    if not isinstance(argv, list) or not argv or len(argv) > 128:
        raise ControlError("INVALID_COMMAND", "command must be a non-empty argv list", EXIT_INPUT)
    result: list[str] = []
    for item in argv:
        if not isinstance(item, str) or not item or len(item) > 8192 or "\x00" in item:
            raise ControlError("INVALID_COMMAND", "each argv item must be a bounded non-empty string", EXIT_INPUT)
        result.append(item)
    executable = Path(result[0]).expanduser()
    if not executable.is_absolute():
        raise ControlError("INVALID_COMMAND", "registered executable path must be absolute", EXIT_INPUT)
    executable = Path(os.path.abspath(executable))
    if os.name == "nt" and (
        str(executable).startswith("\\\\")
        or not re.match(r"^[A-Za-z]:[\\/]", str(executable))
    ):
        raise ControlError("INVALID_COMMAND", "executable must be on a local drive", EXIT_INPUT)
    if not windows_fixed_drive(executable):
        raise ControlError(
            "INVALID_COMMAND", "executable must be on a fixed local drive", EXIT_INPUT
        )
    if not executable.is_file():
        raise ControlError("INVALID_COMMAND", "registered executable does not exist", EXIT_INPUT)
    if _is_reparse_point(executable):
        raise ControlError("INVALID_COMMAND", "executable may not be a reparse point", EXIT_DENIED)
    if executable.suffix.lower() in {".bat", ".cmd"}:
        raise ControlError(
            "INVALID_COMMAND",
            "batch files are not accepted as executables; register powershell.exe -File explicitly",
            EXIT_INPUT,
        )
    result[0] = str(executable)
    return result


def register_agent(
    paths: StatePaths,
    target: str,
    argv: list[str],
    cwd: Path,
    *,
    actor: str = "local-operator",
) -> dict[str, Any]:
    target = validate_target(target)
    argv = validate_argv(argv)
    cwd = Path(os.path.abspath(cwd.expanduser()))
    if not cwd.is_absolute() or not cwd.is_dir():
        raise ControlError("INVALID_WORKING_DIRECTORY", "working directory must exist and be absolute", EXIT_INPUT)
    if os.name == "nt" and (
        str(cwd).startswith("\\\\") or not re.match(r"^[A-Za-z]:[\\/]", str(cwd))
    ):
        raise ControlError(
            "INVALID_WORKING_DIRECTORY", "working directory must be on a local drive", EXIT_INPUT
        )
    if not windows_fixed_drive(cwd):
        raise ControlError(
            "INVALID_WORKING_DIRECTORY",
            "working directory must be on a fixed local drive",
            EXIT_INPUT,
        )
    if _is_reparse_point(cwd):
        raise ControlError(
            "INVALID_WORKING_DIRECTORY", "working directory may not be a reparse point", EXIT_DENIED
        )
    executable_hash = hash_file(Path(argv[0]))
    with ConfigurationLock(paths.lock), connect(paths) as connection:
        with immediate_transaction(connection):
            verify_audit(connection)
            try:
                connection.execute(
                    """
                    INSERT INTO targets(
                        target, argv_json, cwd, executable_sha256, control_state,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'READY', ?, ?)
                    """,
                    (
                        target,
                        canonical_json(argv).decode("utf-8"),
                        str(cwd),
                        executable_hash,
                        format_time(utc_now()),
                        format_time(utc_now()),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ControlError("TARGET_ALREADY_REGISTERED", "target is already registered", EXIT_CONFLICT) from exc
            append_audit(
                connection,
                "TARGET_REGISTERED",
                target,
                actor,
                {
                    "argv_sha256": hashlib.sha256(canonical_json(argv)).hexdigest(),
                    "cwd": str(cwd),
                    "executable_sha256": executable_hash,
                    "shell": False,
                },
            )
    return {
        "executable": argv[0],
        "executable_sha256": executable_hash,
        "ok": True,
        "operation": "register",
        "restart_blocked": False,
        "target": target,
    }


def demo_paths(paths: StatePaths, target: str) -> tuple[Path, Path]:
    safe = target.replace(":", "_")
    directory = paths.demo / safe
    directory.mkdir(parents=True, exist_ok=True)
    if _is_reparse_point(directory):
        raise ControlError("DEMO_PATH_INVALID", "demo directory may not be a reparse point", EXIT_DENIED)
    _apply_owner_only_acl(directory)
    return directory / "heartbeat.json", directory / "child.json"


def register_demo(paths: StatePaths, target: str) -> dict[str, Any]:
    target = validate_target(target)
    heartbeat, child = demo_paths(paths, target)
    argv = [
        str(Path(sys.executable).resolve()),
        "-I",
        "-B",
        str(Path(__file__).resolve()),
        "_demo-target",
        "--state-dir",
        str(paths.root),
        "--heartbeat",
        str(heartbeat),
        "--child-record",
        str(child),
    ]
    result = register_agent(paths, target, argv, paths.root)
    result["demo_child_record"] = str(child)
    result["demo_heartbeat"] = str(heartbeat)
    result["operation"] = "register-demo"
    return result


def unsigned_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(envelope))
    result["authorization"]["signatures"] = {}
    return result


def signature_message(envelope: Mapping[str, Any]) -> bytes:
    return SIGNATURE_DOMAIN + canonical_json(unsigned_envelope(envelope))


def validate_envelope_structure(
    envelope: Any,
    *,
    now: datetime | None = None,
    check_time: bool = True,
) -> dict[str, Any]:
    top = require_exact_keys(
        envelope,
        {
            "action",
            "authorization",
            "effects",
            "expires_at",
            "issued_at",
            "nonce",
            "request_id",
            "schema",
            "scope",
            "target",
        },
        "envelope",
    )
    if top["schema"] != ENVELOPE_SCHEMA or top["action"] != ENVELOPE_ACTION:
        raise ControlError("INVALID_ENVELOPE", "schema or action is not accepted", EXIT_INPUT)
    target = validate_target(top["target"])
    try:
        uuid.UUID(str(top["request_id"]))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ControlError("INVALID_ENVELOPE", "request_id must be a UUID", EXIT_INPUT) from exc
    b64url_decode(top["nonce"], minimum_bytes=32)
    scope = require_exact_keys(top["scope"], set(SCOPE), "scope")
    if any(type(scope[key]) is not type(expected) or scope[key] != expected for key, expected in SCOPE.items()):
        raise ControlError("INVALID_ENVELOPE", "scope was changed or broadened", EXIT_INPUT)
    effects = require_exact_keys(top["effects"], set(EFFECTS), "effects")
    if any(
        type(effects[key]) is not type(expected) or effects[key] != expected
        for key, expected in EFFECTS.items()
    ):
        raise ControlError("INVALID_ENVELOPE", "effects were changed or broadened", EXIT_INPUT)
    authorization = require_exact_keys(
        top["authorization"], {"quorum", "scheme", "signatures"}, "authorization"
    )
    if authorization["scheme"] != SIGNATURE_SCHEME or type(authorization["quorum"]) is not int:
        raise ControlError("INVALID_ENVELOPE", "authorization scheme or quorum type is invalid", EXIT_INPUT)
    if authorization["quorum"] != QUORUM:
        raise ControlError("INVALID_ENVELOPE", "authorization quorum must be exactly two", EXIT_INPUT)
    signatures = authorization["signatures"]
    if not isinstance(signatures, dict) or len(signatures) > MAX_SIGNATURES:
        raise ControlError("INVALID_ENVELOPE", "signature set is invalid or oversized", EXIT_INPUT)
    for operator_id, signature in signatures.items():
        validate_operator(operator_id)
        b64url_decode(signature, exact_bytes=64)
    issued_at = parse_time(top["issued_at"], "issued_at")
    expires_at = parse_time(top["expires_at"], "expires_at")
    if expires_at <= issued_at:
        raise ControlError("INVALID_ENVELOPE", "expires_at must be after issued_at", EXIT_INPUT)
    if (expires_at - issued_at).total_seconds() > MAX_TTL_SECONDS:
        raise ControlError("INVALID_ENVELOPE", "request TTL exceeds 300 seconds", EXIT_INPUT)
    if check_time:
        current = (now or utc_now()).astimezone(timezone.utc)
        if issued_at > current + timedelta(seconds=MAX_CLOCK_SKEW_SECONDS):
            raise ControlError("REQUEST_FROM_FUTURE", "request is dated too far in the future", EXIT_DENIED)
        if expires_at <= current:
            raise ControlError("REQUEST_EXPIRED", "request has expired", EXIT_DENIED)
    if target != top["target"]:
        raise AssertionError("validated target changed unexpectedly")
    return dict(top)


def build_request(paths: StatePaths, target: str, ttl_seconds: int) -> dict[str, Any]:
    target = validate_target(target)
    if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= MAX_TTL_SECONDS:
        raise ControlError("INVALID_TTL", "TTL must be an integer from 1 through 300", EXIT_INPUT)
    with connect(paths) as connection:
        verify_audit(connection)
        require_trust_store_sealed(connection)
        if connection.execute("SELECT 1 FROM targets WHERE target=?", (target,)).fetchone() is None:
            raise ControlError("TARGET_NOT_FOUND", "target is not registered", EXIT_CONFLICT)
        enabled_count = int(
            connection.execute("SELECT COUNT(*) FROM operators WHERE enabled=1").fetchone()[0]
        )
        if enabled_count < QUORUM:
            raise ControlError("QUORUM_UNAVAILABLE", "fewer than two operators are enabled", EXIT_CONFLICT)
    now = utc_now()
    return {
        "action": ENVELOPE_ACTION,
        "authorization": {
            "quorum": QUORUM,
            "scheme": SIGNATURE_SCHEME,
            "signatures": {},
        },
        "effects": copy.deepcopy(EFFECTS),
        "expires_at": format_time(now + timedelta(seconds=ttl_seconds)),
        "issued_at": format_time(now),
        "nonce": b64url_encode(secrets.token_bytes(32)),
        "request_id": str(uuid.uuid4()),
        "schema": ENVELOPE_SCHEMA,
        "scope": copy.deepcopy(SCOPE),
        "target": target,
    }


def sign_request_with_key(
    paths: StatePaths,
    envelope: dict[str, Any],
    operator_id: str,
    private_key: Ed25519PrivateKey,
) -> dict[str, Any]:
    envelope = validate_envelope_structure(envelope)
    operator_id = validate_operator(operator_id)
    signatures = envelope["authorization"]["signatures"]
    if operator_id in signatures:
        raise ControlError("DUPLICATE_APPROVAL", "operator already signed this request", EXIT_CONFLICT)
    _verify_signature_set(paths, envelope, require_quorum=False)
    with connect(paths) as connection:
        verify_audit(connection)
        require_trust_store_sealed(connection)
        row = connection.execute(
            "SELECT public_key_b64, fingerprint FROM operators WHERE operator_id=? AND enabled=1",
            (operator_id,),
        ).fetchone()
    if row is None:
        raise ControlError("OPERATOR_NOT_TRUSTED", "operator is not enabled", EXIT_DENIED)
    expected_raw = b64url_decode(row["public_key_b64"], exact_bytes=32)
    actual_raw = public_key_raw(private_key.public_key())
    if not secrets.compare_digest(expected_raw, actual_raw):
        raise ControlError("PRIVATE_KEY_MISMATCH", "private key does not match the trusted operator", EXIT_DENIED)
    signature = private_key.sign(signature_message(envelope))
    result = copy.deepcopy(envelope)
    result["authorization"]["signatures"][operator_id] = b64url_encode(signature)
    validate_envelope_structure(result)
    return result


def _verify_signature_set(
    paths: StatePaths,
    envelope: dict[str, Any],
    *,
    require_quorum: bool,
) -> list[str]:
    require_cryptography()
    signatures: dict[str, str] = envelope["authorization"]["signatures"]
    if require_quorum and len(signatures) < QUORUM:
        raise ControlError("QUORUM_NOT_MET", "two valid operator signatures are required", EXIT_DENIED)
    with connect(paths) as connection:
        verify_audit(connection)
        require_trust_store_sealed(connection)
        if connection.execute(
            "SELECT 1 FROM targets WHERE target=?", (envelope["target"],)
        ).fetchone() is None:
            raise ControlError("TARGET_NOT_FOUND", "target is not registered", EXIT_CONFLICT)
        message = signature_message(envelope)
        valid_fingerprints: set[str] = set()
        valid_operators: list[str] = []
        for operator_id, encoded_signature in sorted(signatures.items()):
            row = connection.execute(
                "SELECT public_key_b64, fingerprint FROM operators WHERE operator_id=? AND enabled=1",
                (operator_id,),
            ).fetchone()
            if row is None:
                raise ControlError("OPERATOR_NOT_TRUSTED", f"operator is not enabled: {operator_id}", EXIT_DENIED)
            public_key = Ed25519PublicKey.from_public_bytes(
                b64url_decode(row["public_key_b64"], exact_bytes=32)
            )
            try:
                public_key.verify(
                    b64url_decode(encoded_signature, exact_bytes=64), message
                )
            except InvalidSignature as exc:
                raise ControlError(
                    "SIGNATURE_INVALID", f"signature verification failed for {operator_id}", EXIT_DENIED
                ) from exc
            fingerprint = str(row["fingerprint"])
            if fingerprint in valid_fingerprints:
                raise ControlError("NON_DISTINCT_APPROVAL_KEYS", "approval keys are not distinct", EXIT_DENIED)
            valid_fingerprints.add(fingerprint)
            valid_operators.append(operator_id)
    if require_quorum and len(valid_operators) < QUORUM:
        raise ControlError("QUORUM_NOT_MET", "two valid operator signatures are required", EXIT_DENIED)
    return valid_operators


def verify_request(
    paths: StatePaths,
    envelope: dict[str, Any],
    *,
    check_time: bool = True,
) -> dict[str, Any]:
    envelope = validate_envelope_structure(envelope, check_time=check_time)
    valid_operators = _verify_signature_set(paths, envelope, require_quorum=True)
    return {
        "authorization_valid": True,
        "operation_status": "NOT_APPLIED",
        "request_id": envelope["request_id"],
        "signers": valid_operators,
        "target": envelope["target"],
    }


if os.name == "nt":
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
    CREATE_SUSPENDED = 0x00000004
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    CREATE_UNICODE_ENVIRONMENT = 0x00000400
    DETACHED_PROCESS = 0x00000008
    CREATE_NO_WINDOW = 0x08000000
    STARTF_USESTDHANDLES = 0x00000100
    WAIT_OBJECT_0 = 0x00000000
    WAIT_TIMEOUT = 0x00000102
    INFINITE = 0xFFFFFFFF
    JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS = 1
    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9

    ULONG_PTR = wintypes.WPARAM
    SIZE_T = ctypes.c_size_t

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", SIZE_T),
            ("MaximumWorkingSetSize", SIZE_T),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ULONG_PTR),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", SIZE_T),
            ("JobMemoryLimit", SIZE_T),
            ("PeakProcessMemoryUsed", SIZE_T),
            ("PeakJobMemoryUsed", SIZE_T),
        ]

    class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_longlong),
            ("TotalKernelTime", ctypes.c_longlong),
            ("ThisPeriodTotalUserTime", ctypes.c_longlong),
            ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    class STARTUPINFOW(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("lpReserved", wintypes.LPWSTR),
            ("lpDesktop", wintypes.LPWSTR),
            ("lpTitle", wintypes.LPWSTR),
            ("dwX", wintypes.DWORD),
            ("dwY", wintypes.DWORD),
            ("dwXSize", wintypes.DWORD),
            ("dwYSize", wintypes.DWORD),
            ("dwXCountChars", wintypes.DWORD),
            ("dwYCountChars", wintypes.DWORD),
            ("dwFillAttribute", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("wShowWindow", wintypes.WORD),
            ("cbReserved2", wintypes.WORD),
            ("lpReserved2", ctypes.POINTER(ctypes.c_ubyte)),
            ("hStdInput", wintypes.HANDLE),
            ("hStdOutput", wintypes.HANDLE),
            ("hStdError", wintypes.HANDLE),
        ]

    class PROCESS_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("hProcess", wintypes.HANDLE),
            ("hThread", wintypes.HANDLE),
            ("dwProcessId", wintypes.DWORD),
            ("dwThreadId", wintypes.DWORD),
        ]

    class PROCESS_FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", wintypes.DWORD),
            ("dwHighDateTime", wintypes.DWORD),
        ]

    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.CreateProcessW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.LPCWSTR,
        ctypes.POINTER(STARTUPINFOW),
        ctypes.POINTER(PROCESS_INFORMATION),
    ]
    kernel32.CreateProcessW.restype = wintypes.BOOL
    kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel32.ResumeThread.restype = wintypes.DWORD
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_FILETIME),
        ctypes.POINTER(PROCESS_FILETIME),
        ctypes.POINTER(PROCESS_FILETIME),
        ctypes.POINTER(PROCESS_FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL


def win_error(operation: str) -> ControlError:
    code = ctypes.get_last_error()
    return ControlError("WINDOWS_API_FAILED", f"{operation} failed with Windows error {code}", EXIT_INTERNAL)


def process_handle_creation_filetime(handle: Any) -> int:
    creation = PROCESS_FILETIME()
    exit_time = PROCESS_FILETIME()
    kernel_time = PROCESS_FILETIME()
    user_time = PROCESS_FILETIME()
    if not kernel32.GetProcessTimes(
        handle,
        ctypes.byref(creation),
        ctypes.byref(exit_time),
        ctypes.byref(kernel_time),
        ctypes.byref(user_time),
    ):
        raise win_error("GetProcessTimes")
    return (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)


def safe_child_environment(extra: Mapping[str, str]) -> dict[str, str]:
    allowed = {
        "COMSPEC",
        "NUMBER_OF_PROCESSORS",
        "OS",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_IDENTIFIER",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "WINDIR",
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in allowed and isinstance(value, str)
    }
    environment.update(extra)
    return environment


def windows_environment_block(environment: Mapping[str, str]) -> ctypes.Array[Any]:
    entries = []
    for key, value in sorted(environment.items(), key=lambda item: item[0].upper()):
        if not key or "=" in key or "\x00" in key or "\x00" in value:
            raise ControlError("INVALID_ENVIRONMENT", "child environment contains an invalid entry", EXIT_INTERNAL)
        entries.append(f"{key}={value}")
    return ctypes.create_unicode_buffer("\x00".join(entries) + "\x00\x00")


@dataclass
class WindowsManagedJob:
    job_handle: Any
    process_handle: Any
    process_id: int
    job_name: str
    closed: bool = False

    def active_processes(self) -> int:
        if self.closed:
            return 0
        info = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        returned = wintypes.DWORD(0)
        if not kernel32.QueryInformationJobObject(
            self.job_handle,
            JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
            ctypes.byref(returned),
        ):
            raise win_error("QueryInformationJobObject")
        return int(info.ActiveProcesses)

    def root_exited(self) -> bool:
        if self.closed:
            return True
        result = kernel32.WaitForSingleObject(self.process_handle, 0)
        if result == WAIT_OBJECT_0:
            return True
        if result == WAIT_TIMEOUT:
            return False
        raise win_error("WaitForSingleObject")

    def terminate_and_confirm(self, timeout_seconds: float = 10.0) -> int:
        timeout_seconds = validate_timeout(timeout_seconds)
        if self.closed:
            return 0
        if not kernel32.TerminateJobObject(self.job_handle, 0xE0000001):
            error = ctypes.get_last_error()
            if error not in {5, 6, 87}:
                raise win_error("TerminateJobObject")
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            active = self.active_processes()
            if active == 0:
                return 0
            time.sleep(0.05)
        return self.active_processes()

    def exit_code(self) -> int | None:
        code = wintypes.DWORD(0)
        if not kernel32.GetExitCodeProcess(self.process_handle, ctypes.byref(code)):
            return None
        return int(code.value)

    def close(self) -> None:
        if self.closed:
            return
        if self.process_handle:
            kernel32.CloseHandle(self.process_handle)
        if self.job_handle:
            kernel32.CloseHandle(self.job_handle)
        self.closed = True


def launch_windows_job(
    job_name: str,
    argv: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
) -> WindowsManagedJob:
    if os.name != "nt":
        raise ControlError("WINDOWS_REQUIRED", "Windows Job Objects are required", EXIT_INTERNAL)
    import msvcrt

    # The Job Object is deliberately unnamed. A named object would allow another
    # process under the same Windows account to pre-create or retain a handle to
    # it. The detached supervisor is the sole owner; KILL_ON_JOB_CLOSE handles a
    # supervisor crash, and signed isolation is delivered through durable state.
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise win_error("CreateJobObjectW")
    process_info = PROCESS_INFORMATION()
    log_stream: Any = None
    null_stream: Any = None
    try:
        limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        limits.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
        )
        if not kernel32.SetInformationJobObject(
            job,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            raise win_error("SetInformationJobObject")

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("ab", buffering=0)
        null_stream = open(os.devnull, "rb", buffering=0)
        stdout_handle = msvcrt.get_osfhandle(log_stream.fileno())
        stdin_handle = msvcrt.get_osfhandle(null_stream.fileno())
        os.set_handle_inheritable(stdout_handle, True)
        os.set_handle_inheritable(stdin_handle, True)

        startup = STARTUPINFOW()
        startup.cb = ctypes.sizeof(startup)
        startup.dwFlags = STARTF_USESTDHANDLES
        startup.hStdInput = stdin_handle
        startup.hStdOutput = stdout_handle
        startup.hStdError = stdout_handle
        command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(list(argv)))
        environment_block = windows_environment_block(environment)
        flags = CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW
        if not kernel32.CreateProcessW(
            str(argv[0]),
            command_line,
            None,
            None,
            True,
            flags,
            ctypes.cast(environment_block, ctypes.c_void_p),
            str(cwd),
            ctypes.byref(startup),
            ctypes.byref(process_info),
        ):
            raise win_error("CreateProcessW")
        if not kernel32.AssignProcessToJobObject(job, process_info.hProcess):
            kernel32.TerminateProcess(process_info.hProcess, 0xE0000002)
            raise win_error("AssignProcessToJobObject")
        resume_result = kernel32.ResumeThread(process_info.hThread)
        if resume_result == 0xFFFFFFFF:
            kernel32.TerminateProcess(process_info.hProcess, 0xE0000003)
            raise win_error("ResumeThread")
        kernel32.CloseHandle(process_info.hThread)
        process_info.hThread = None
        return WindowsManagedJob(
            job_handle=job,
            process_handle=process_info.hProcess,
            process_id=int(process_info.dwProcessId),
            job_name=job_name,
        )
    except BaseException:
        if process_info.hThread:
            kernel32.CloseHandle(process_info.hThread)
        if process_info.hProcess:
            kernel32.TerminateProcess(process_info.hProcess, 0xE0000004)
            kernel32.CloseHandle(process_info.hProcess)
        kernel32.CloseHandle(job)
        raise
    finally:
        if log_stream is not None:
            try:
                os.set_handle_inheritable(msvcrt.get_osfhandle(log_stream.fileno()), False)
            except OSError:
                pass
            log_stream.close()
        if null_stream is not None:
            try:
                os.set_handle_inheritable(msvcrt.get_osfhandle(null_stream.fileno()), False)
            except OSError:
                pass
            null_stream.close()


def spawn_run_supervisor(paths: StatePaths, run_id: str, capability: str) -> tuple[int, int]:
    environment = safe_child_environment(
        {
            SUPERVISOR_CAP_ENV: capability,
            "PYTHONIOENCODING": "utf-8",
        }
    )
    command = [
        str(Path(sys.executable).resolve()),
        "-I",
        "-B",
        str(Path(__file__).resolve()),
        "_supervisor",
        "--state-dir",
        str(paths.root),
        "--run-id",
        run_id,
    ]
    creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    try:
        process = subprocess.Popen(
            command,
            shell=False,
            cwd=paths.root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=creationflags,
        )
    except OSError as exc:
        raise ControlError("SUPERVISOR_START_FAILED", f"unable to start supervisor: {exc}", EXIT_INTERNAL) from exc
    pid = int(process.pid)
    created_filetime = 0
    if os.name == "nt":
        # This is an intentionally detached supervisor. Close only the start
        # client's duplicate process handle; the new process and its Job handle
        # remain alive independently.
        try:
            created_filetime = process_handle_creation_filetime(process._handle)  # type: ignore[attr-defined]
        except (AttributeError, ControlError):
            created_filetime = 0
        try:
            process._handle.Close()  # type: ignore[attr-defined]
            process._child_created = False  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            # The OS will close this duplicate when the short-lived start client
            # exits. Never convert a successful detached spawn into a false
            # launch failure solely because this optimization was unavailable.
            pass
    return pid, created_filetime


def preflight_supervisor_runtime() -> None:
    """Prove the detached isolated interpreter can import its sole dependency."""

    require_cryptography()
    command = [
        str(Path(sys.executable).resolve()),
        "-I",
        "-B",
        "-c",
        "import cryptography; from cryptography.hazmat.primitives.asymmetric import ed25519",
    ]
    try:
        result = subprocess.run(
            command,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10.0,
            creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ControlError(
            "SUPERVISOR_RUNTIME_PREFLIGHT_FAILED",
            f"unable to preflight the isolated supervisor interpreter: {exc}",
            EXIT_INTERNAL,
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or "dependency import failed").strip().splitlines()[-1]
        raise ControlError(
            "SUPERVISOR_RUNTIME_PREFLIGHT_FAILED",
            f"{sys.executable} -I cannot import cryptography: {detail[:512]}",
            EXIT_INTERNAL,
        )


def start_agent(paths: StatePaths, target: str, timeout_seconds: float = 15.0) -> dict[str, Any]:
    if os.name != "nt":
        raise ControlError("WINDOWS_REQUIRED", "Windows Job Objects are required", EXIT_INTERNAL)
    timeout_seconds = validate_timeout(timeout_seconds)
    target = validate_target(target)
    preflight_supervisor_runtime()
    reconcile_stale_runs(paths, target)
    run_id = str(uuid.uuid4())
    job_name = f"UNNAMED:{run_id}"
    capability = b64url_encode(secrets.token_bytes(32))
    capability_hash = hashlib.sha256(capability.encode("ascii")).hexdigest()
    log_path = paths.logs / f"{run_id}.log"
    now = format_time(utc_now())
    with connect(paths) as connection:
        with immediate_transaction(connection):
            verify_audit(connection)
            require_trust_store_sealed(connection)
            target_row = connection.execute(
                "SELECT * FROM targets WHERE target=?", (target,)
            ).fetchone()
            if target_row is None:
                raise ControlError("TARGET_NOT_FOUND", "target is not registered", EXIT_CONFLICT)
            if target_row["control_state"] == "ISOLATED":
                raise ControlError("TARGET_ISOLATED", "future starts are durably blocked", EXIT_DENIED)
            live = connection.execute(
                "SELECT run_id FROM runs WHERE target=? AND state IN ('STARTING','RUNNING','ISOLATING')",
                (target,),
            ).fetchone()
            if live is not None:
                raise ControlError("TARGET_ALREADY_RUNNING", "target already has a live run", EXIT_CONFLICT)
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, target, state, job_name, supervisor_token_hash,
                    log_path, created_at
                ) VALUES (?, ?, 'STARTING', ?, ?, ?, ?)
                """,
                (run_id, target, job_name, capability_hash, str(log_path), now),
            )
            append_audit(
                connection,
                "START_RESERVED",
                target,
                "local-operator",
                {"run_id": run_id, "shell": False},
            )
    try:
        supervisor_pid, supervisor_created_filetime = spawn_run_supervisor(
            paths, run_id, capability
        )
    except BaseException as exc:
        isolation_won = False
        with connect(paths) as connection:
            with immediate_transaction(connection):
                verify_audit(connection)
                current = connection.execute(
                    """
                    SELECT r.state, t.control_state FROM runs r
                    JOIN targets t ON t.target=r.target WHERE r.run_id=?
                    """,
                    (run_id,),
                ).fetchone()
                isolation_won = bool(
                    current is not None
                    and current["state"] in {"ISOLATING", "ISOLATED"}
                    and current["control_state"] == "ISOLATED"
                )
                if not isolation_won:
                    connection.execute(
                        """
                        UPDATE runs SET state='FAILED', error_code='SUPERVISOR_START_FAILED',
                            error_message=?, ended_at=?, heartbeat_at=?
                        WHERE run_id=? AND state='STARTING'
                        """,
                        (
                            str(exc)[:2048],
                            format_time(utc_now()),
                            format_time(utc_now()),
                            run_id,
                        ),
                    )
                    append_audit(
                        connection,
                        "START_FAILED",
                        target,
                        "local-operator",
                        {"error_code": "SUPERVISOR_START_FAILED", "run_id": run_id},
                    )
        if isolation_won:
            reconcile_stale_runs(paths, target)
            raise ControlError(
                "TARGET_ISOLATED",
                "isolation won the launch race; no target process was created",
                EXIT_DENIED,
            ) from exc
        raise

    with connect(paths) as connection:
        with immediate_transaction(connection):
            verify_audit(connection)
            connection.execute(
                """
                UPDATE runs SET supervisor_pid=?,
                    supervisor_created_filetime=CASE
                        WHEN ? > 0 THEN ? ELSE supervisor_created_filetime END,
                    heartbeat_at=?
                WHERE run_id=? AND state IN ('STARTING','ISOLATING')
                """,
                (
                    supervisor_pid,
                    supervisor_created_filetime,
                    supervisor_created_filetime,
                    format_time(utc_now()),
                    run_id,
                ),
            )

    deadline = time.monotonic() + timeout_seconds
    last_reconcile = 0.0
    while time.monotonic() < deadline:
        if time.monotonic() - last_reconcile >= 1.0:
            reconcile_stale_runs(paths, target)
            last_reconcile = time.monotonic()
        with connect(paths) as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is not None and row["state"] == "RUNNING":
            if int(row["job_active_processes"] or 0) < 1:
                time.sleep(0.05)
                continue
            return {
                "child_pid": row["child_pid"],
                "credential_issued": True,
                "enforcement": "windows_job_object",
                "generation": run_id,
                "job_active_processes": row["job_active_processes"],
                "ok": True,
                "operation": "start",
                "operation_status": "VERIFIED_RUNNING",
                "observed_at": row["heartbeat_at"],
                "run_id": run_id,
                "supervisor_created_filetime": row["supervisor_created_filetime"],
                "supervisor_pid": row["supervisor_pid"],
                "target": target,
            }
        if row is not None and row["state"] == "EXITED":
            return {
                "child_pid": row["child_pid"],
                "credential_currently_valid": False,
                "credential_issued": bool(row["credential_hash"]),
                "enforcement": "windows_job_object",
                "exit_code": row["exit_code"],
                "generation": run_id,
                "job_active_processes": 0,
                "ok": True,
                "operation": "start",
                "operation_status": "VERIFIED_STARTED_AND_EXITED",
                "observed_at": row["ended_at"],
                "run_id": run_id,
                "supervisor_created_filetime": row["supervisor_created_filetime"],
                "supervisor_pid": row["supervisor_pid"],
                "target": target,
            }
        if row is not None and row["state"] == "ISOLATED":
            raise ControlError(
                "TARGET_ISOLATED",
                "isolation won the launch race and the process tree is absent",
                EXIT_DENIED,
            )
        if row is not None and row["state"] == "FAILED":
            raise ControlError(
                str(row["error_code"] or "START_FAILED"),
                str(row["error_message"] or "supervisor failed to launch target"),
                EXIT_INTERNAL,
            )
        time.sleep(0.1)
    raise ControlError(
        "START_UNCONFIRMED",
        f"run {run_id} did not reach RUNNING before timeout; inspect status",
        EXIT_UNCONFIRMED,
    )


def _record_supervisor_failure(
    paths: StatePaths,
    run_id: str,
    target: str | None,
    code: str,
    message: str,
) -> None:
    try:
        with connect(paths) as connection:
            with immediate_transaction(connection):
                verify_audit(connection)
                row = connection.execute(
                    "SELECT target, state FROM runs WHERE run_id=?", (run_id,)
                ).fetchone()
                if row is None:
                    return
                target_value = target or str(row["target"])
                target_row = connection.execute(
                    "SELECT control_state FROM targets WHERE target=?", (target_value,)
                ).fetchone()
                isolated = target_row is not None and target_row["control_state"] == "ISOLATED"
                connection.execute(
                    """
                    UPDATE runs SET state=?, credential_revoked_at=COALESCE(credential_revoked_at, ?),
                        ended_at=?, error_code=?, error_message=?, heartbeat_at=?
                    WHERE run_id=?
                    """,
                    (
                        "FAILED",
                        format_time(utc_now()),
                        format_time(utc_now()),
                        code,
                        message[:2048],
                        format_time(utc_now()),
                        run_id,
                    ),
                )
                if isolated:
                    connection.execute(
                        """
                        UPDATE requests SET status='ENFORCEMENT_UNCONFIRMED', result_json=?
                        WHERE target=? AND status='ENFORCING'
                        """,
                        (
                            canonical_json(
                                {
                                    "error_code": code,
                                    "message": message[:2048],
                                    "operation_status": "ENFORCEMENT_UNCONFIRMED",
                                }
                            ).decode("utf-8"),
                            target_value,
                        ),
                    )
                append_audit(
                    connection,
                    "SUPERVISOR_FAILURE",
                    target_value,
                    "run-supervisor",
                    {
                        "error_code": code,
                        "message": message[:512],
                        "run_id": run_id,
                        "target_remains_blocked": bool(isolated),
                    },
                )
    except BaseException:
        return


def _build_isolation_result(
    request_id: str,
    target: str,
    run_id: str | None,
    *,
    termination_performed: bool,
    credential_was_issued: bool,
    job_close_cleanup_basis: bool = False,
) -> dict[str, Any]:
    if termination_performed:
        absence_basis = "terminate_job_object_zero_active_processes"
    elif job_close_cleanup_basis:
        absence_basis = "supervisor_exit_with_unnamed_job_kill_on_close"
    else:
        absence_basis = "no_live_supervised_process_tree"
    return {
        "authorization_valid": True,
        "enforcement": {
            "evidence_retention": "local_hash_chain_not_immutable",
            "external_sessions_revoked": False,
            "future_supervisor_starts_blocked": True,
            "job_active_processes": 0,
            "local_process_tree_absence_basis": absence_basis,
            "local_process_tree_absent_verified": True,
            "local_process_tree_termination_performed": termination_performed,
            "network_policy_changed": False,
            "provider_credentials_revoked": False,
            "supervisor_local_credential_absent_or_revoked_verified": True,
            "supervisor_local_credential_revoked": credential_was_issued,
        },
        "ok": True,
        "operation": "apply-isolation",
        "operation_status": "VERIFIED_ISOLATED",
        "request_id": request_id,
        "run_id": run_id,
        "target": target,
        "truth_boundary": "local_windows_supervised_processes_only",
    }


def _finalize_isolation(
    paths: StatePaths,
    request_id: str,
    target: str,
    run_id: str | None,
    *,
    termination_performed: bool,
    credential_was_issued: bool,
    actor: str,
    job_close_cleanup_basis: bool = False,
) -> dict[str, Any]:
    result = _build_isolation_result(
        request_id,
        target,
        run_id,
        termination_performed=termination_performed,
        credential_was_issued=credential_was_issued,
        job_close_cleanup_basis=job_close_cleanup_basis,
    )
    with connect(paths) as connection:
        with immediate_transaction(connection):
            verify_audit(connection)
            return _finalize_isolation_in_transaction(
                connection,
                result,
                request_id,
                target,
                run_id,
                termination_performed=termination_performed,
                actor=actor,
            )


def _finalize_isolation_in_transaction(
    connection: sqlite3.Connection,
    result: dict[str, Any],
    request_id: str,
    target: str,
    run_id: str | None,
    *,
    termination_performed: bool,
    actor: str,
) -> dict[str, Any]:
    request_row = connection.execute(
        "SELECT status, result_json FROM requests WHERE request_id=?", (request_id,)
    ).fetchone()
    if request_row is None:
        raise IntegrityFailure("accepted isolation request disappeared")
    observed = format_time(utc_now())
    if run_id is not None:
        connection.execute(
            """
            UPDATE runs SET state='ISOLATED', job_active_processes=0,
                credential_revoked_at=COALESCE(credential_revoked_at, ?),
                ended_at=COALESCE(ended_at, ?), heartbeat_at=?
            WHERE run_id=?
            """,
            (observed, observed, observed, run_id),
        )
    if request_row["status"] == "APPLIED":
        if request_row["result_json"]:
            return parse_json_bytes(str(request_row["result_json"]).encode("utf-8"))
        return result
    connection.execute(
        """
        UPDATE requests SET status='APPLIED', applied_at=?, result_json=?
        WHERE request_id=?
        """,
        (observed, canonical_json(result).decode("utf-8"), request_id),
    )
    append_audit(
        connection,
        "ISOLATION_VERIFIED",
        target,
        actor,
        {
            "job_active_processes": 0,
            "local_process_tree_absence_basis": result["enforcement"][
                "local_process_tree_absence_basis"
            ],
            "request_id": request_id,
            "run_id": run_id,
            "termination_performed": termination_performed,
        },
    )
    return result


def run_supervisor(paths: StatePaths, run_id: str) -> int:
    if os.name != "nt":
        return EXIT_INTERNAL
    capability = os.environ.pop(SUPERVISOR_CAP_ENV, None)
    target: str | None = None
    managed: WindowsManagedJob | None = None
    prelaunch_request_id: str | None = None
    if not capability:
        _record_supervisor_failure(
            paths, run_id, None, "SUPERVISOR_CAPABILITY_MISSING", "capability was not supplied"
        )
        return EXIT_DENIED
    try:
        capability_hash = hashlib.sha256(capability.encode("ascii")).hexdigest()
        with connect(paths) as connection:
            with immediate_transaction(connection):
                verify_audit(connection)
                row = connection.execute(
                    """
                    SELECT r.*, t.argv_json, t.cwd, t.executable_sha256, t.control_state
                    FROM runs r JOIN targets t ON t.target=r.target
                    WHERE r.run_id=?
                    """,
                    (run_id,),
                ).fetchone()
                if row is None:
                    raise ControlError("RUN_NOT_FOUND", "reserved run does not exist", EXIT_CONFLICT)
                if not secrets.compare_digest(
                    str(row["supervisor_token_hash"] or ""), capability_hash
                ):
                    raise ControlError("SUPERVISOR_CAPABILITY_INVALID", "capability did not match", EXIT_DENIED)
                launch_allowed = row["state"] == "STARTING" and row["control_state"] == "READY"
                isolation_won = (
                    row["state"] in {"ISOLATING", "ISOLATED"}
                    and row["control_state"] == "ISOLATED"
                )
                if not launch_allowed and not isolation_won:
                    raise IntegrityFailure(
                        "run and target state are inconsistent at the supervisor launch gate"
                    )
                target = str(row["target"])
                argv = parse_json_bytes(str(row["argv_json"]).encode("utf-8"))
                argv = validate_argv(argv)
                cwd = Path(str(row["cwd"]))
                expected_executable_hash = str(row["executable_sha256"])
                job_name = str(row["job_name"])
                log_path = Path(str(row["log_path"]))
                cursor = connection.execute(
                    """
                    UPDATE runs SET supervisor_token_hash=NULL, supervisor_pid=?,
                        supervisor_created_filetime=?, heartbeat_at=?
                    WHERE run_id=? AND supervisor_token_hash=?
                    """,
                    (
                        os.getpid(),
                        process_handle_creation_filetime(kernel32.GetCurrentProcess()),
                        format_time(utc_now()),
                        run_id,
                        capability_hash,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ControlError("SUPERVISOR_CAPABILITY_REPLAY", "capability was already consumed", EXIT_DENIED)
                append_audit(
                    connection,
                    "SUPERVISOR_CLAIMED",
                    target,
                    "run-supervisor",
                    {"run_id": run_id, "supervisor_pid": os.getpid()},
                )
                if isolation_won:
                    request_row = connection.execute(
                        """
                        SELECT request_id FROM requests
                        WHERE target=? AND status IN (
                            'ENFORCING','ENFORCEMENT_UNCONFIRMED','APPLIED'
                        )
                        ORDER BY CASE WHEN status='APPLIED' THEN 1 ELSE 0 END, accepted_at
                        LIMIT 1
                        """,
                        (target,),
                    ).fetchone()
                    if request_row is None:
                        raise IntegrityFailure("isolated pre-launch run has no accepted request")
                    prelaunch_request_id = str(request_row["request_id"])
        capability = ""
        if prelaunch_request_id is not None:
            _finalize_isolation(
                paths,
                prelaunch_request_id,
                target,
                run_id,
                termination_performed=False,
                credential_was_issued=bool(row["credential_hash"]),
                actor="run-supervisor-prelaunch",
            )
            return EXIT_OK
        if hash_file(Path(argv[0])) != expected_executable_hash:
            raise ControlError(
                "EXECUTABLE_CHANGED",
                "registered executable hash changed; re-register intentionally",
                EXIT_DENIED,
            )
        local_credential = b64url_encode(secrets.token_bytes(32))
        credential_hash = hashlib.sha256(local_credential.encode("ascii")).hexdigest()
        credential_expires_at = format_time(utc_now() + timedelta(hours=1))
        child_environment = safe_child_environment(
            {
                AGENT_ID_ENV: target,
                GENERATION_ENV: run_id,
                LOCAL_CREDENTIAL_ENV: local_credential,
                "PYTHONIOENCODING": "utf-8",
            }
        )

        started_and_exited = False
        connection = connect(paths)
        try:
            connection.execute("BEGIN IMMEDIATE")
            verify_audit(connection)
            current = connection.execute(
                """
                SELECT r.state, t.control_state
                FROM runs r JOIN targets t ON t.target=r.target WHERE r.run_id=?
                """,
                (run_id,),
            ).fetchone()
            if current is None:
                raise IntegrityFailure("reserved run disappeared before the launch gate")
            if current["state"] in {"ISOLATING", "ISOLATED"} and current["control_state"] == "ISOLATED":
                request_row = connection.execute(
                    """
                    SELECT request_id FROM requests
                    WHERE target=? AND status IN (
                        'ENFORCING','ENFORCEMENT_UNCONFIRMED','APPLIED'
                    )
                    ORDER BY CASE WHEN status='APPLIED' THEN 1 ELSE 0 END, accepted_at
                    LIMIT 1
                    """,
                    (target,),
                ).fetchone()
                if request_row is None:
                    raise IntegrityFailure("isolated launch gate has no accepted request")
                prelaunch_request_id = str(request_row["request_id"])
            elif current["state"] == "STARTING" and current["control_state"] == "READY":
                managed = launch_windows_job(job_name, argv, cwd, child_environment, log_path)
                active = managed.active_processes()
                root_exited_after_resume = managed.root_exited()
                observed = format_time(utc_now())
                if root_exited_after_resume:
                    residual_tree_terminated = active > 0
                    if residual_tree_terminated:
                        remaining = managed.terminate_and_confirm()
                        if remaining != 0:
                            raise ControlError(
                                "RESIDUAL_TREE_TERMINATION_UNCONFIRMED",
                                f"Job Object still reports {remaining} active processes",
                                EXIT_UNCONFIRMED,
                            )
                    exit_code = managed.exit_code()
                    connection.execute(
                        """
                        UPDATE runs SET state='EXITED', child_pid=?, credential_hash=?,
                            credential_expires_at=?, credential_revoked_at=?,
                            job_active_processes=0, started_at=?, ended_at=?,
                            heartbeat_at=?, exit_code=? WHERE run_id=?
                        """,
                        (
                            managed.process_id,
                            credential_hash,
                            credential_expires_at,
                            observed,
                            observed,
                            observed,
                            observed,
                            exit_code,
                            run_id,
                        ),
                    )
                    append_audit(
                        connection,
                        "PROCESS_TREE_EXITED",
                        target,
                        "run-supervisor",
                        {
                            "child_pid": managed.process_id,
                            "exit_code": exit_code,
                            "exited_before_running_observation": True,
                            "residual_tree_terminated": residual_tree_terminated,
                            "run_id": run_id,
                            "windows_job_assigned_before_resume": True,
                        },
                    )
                    started_and_exited = True
                elif active < 1:
                    raise ControlError(
                        "PROCESS_STATE_UNCONFIRMED",
                        "the root handle is live but the Job Object reports no active process",
                        EXIT_UNCONFIRMED,
                    )
                else:
                    connection.execute(
                        """
                        UPDATE runs SET state='RUNNING', child_pid=?, credential_hash=?,
                            credential_expires_at=?, job_active_processes=?, started_at=?, heartbeat_at=?
                        WHERE run_id=?
                        """,
                        (
                            managed.process_id,
                            credential_hash,
                            credential_expires_at,
                            active,
                            observed,
                            observed,
                            run_id,
                        ),
                    )
                    append_audit(
                        connection,
                        "PROCESS_TREE_STARTED",
                        target,
                        "run-supervisor",
                        {
                            "child_pid": managed.process_id,
                            "job_active_processes": active,
                            "observed_at": observed,
                            "run_id": run_id,
                            "windows_job_assigned_before_resume": True,
                        },
                    )
            else:
                raise IntegrityFailure("run and target state are inconsistent at the final launch gate")
            connection.execute("COMMIT")
        except BaseException:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            if managed is not None:
                try:
                    managed.terminate_and_confirm()
                finally:
                    managed.close()
                    managed = None
            raise
        finally:
            connection.close()
        if prelaunch_request_id is not None:
            _finalize_isolation(
                paths,
                prelaunch_request_id,
                target,
                run_id,
                termination_performed=False,
                credential_was_issued=False,
                actor="run-supervisor-prelaunch",
            )
            return EXIT_OK
        child_environment.pop(LOCAL_CREDENTIAL_ENV, None)
        local_credential = ""
        if started_and_exited:
            managed.close()
            managed = None
            return EXIT_OK

        last_durable_heartbeat = 0.0
        while True:
            active = managed.active_processes()
            root_exited = managed.root_exited()
            poll_time = time.monotonic()
            integrity_checked = False
            with connect(paths) as connection:
                if poll_time - last_durable_heartbeat >= 1.0:
                    with immediate_transaction(connection):
                        verify_audit(connection)
                        row = connection.execute(
                            "SELECT control_state FROM targets WHERE target=?", (target,)
                        ).fetchone()
                        request_row = connection.execute(
                            """
                            SELECT request_id, status FROM requests
                            WHERE target=? AND status IN (
                                'ENFORCING','ENFORCEMENT_UNCONFIRMED','APPLIED'
                            )
                            ORDER BY CASE WHEN status='APPLIED' THEN 1 ELSE 0 END, accepted_at
                            LIMIT 1
                            """,
                            (target,),
                        ).fetchone()
                        connection.execute(
                            "UPDATE runs SET heartbeat_at=?, job_active_processes=? WHERE run_id=?",
                            (format_time(utc_now()), active, run_id),
                        )
                    last_durable_heartbeat = poll_time
                    integrity_checked = True
                else:
                    connection.execute("BEGIN")
                    try:
                        row = connection.execute(
                            "SELECT control_state FROM targets WHERE target=?", (target,)
                        ).fetchone()
                        request_row = connection.execute(
                            """
                            SELECT request_id, status FROM requests
                            WHERE target=? AND status IN (
                                'ENFORCING','ENFORCEMENT_UNCONFIRMED','APPLIED'
                            )
                            ORDER BY CASE WHEN status='APPLIED' THEN 1 ELSE 0 END, accepted_at
                            LIMIT 1
                            """,
                            (target,),
                        ).fetchone()
                        connection.execute("COMMIT")
                    except BaseException:
                        if connection.in_transaction:
                            connection.execute("ROLLBACK")
                        raise
            if row is None:
                raise IntegrityFailure("registered target disappeared")
            if row["control_state"] == "ISOLATED":
                if not integrity_checked:
                    with connect(paths) as connection:
                        with immediate_transaction(connection):
                            verify_audit(connection)
                            row = connection.execute(
                                "SELECT control_state FROM targets WHERE target=?", (target,)
                            ).fetchone()
                            request_row = connection.execute(
                                """
                                SELECT request_id, status FROM requests
                                WHERE target=? AND status IN (
                                    'ENFORCING','ENFORCEMENT_UNCONFIRMED','APPLIED'
                                )
                                ORDER BY CASE WHEN status='APPLIED' THEN 1 ELSE 0 END, accepted_at
                                LIMIT 1
                                """,
                                (target,),
                            ).fetchone()
                if row is None or row["control_state"] != "ISOLATED":
                    continue
                remaining = managed.terminate_and_confirm()
                if remaining != 0:
                    raise ControlError(
                        "ENFORCEMENT_UNCONFIRMED",
                        f"Job Object still reports {remaining} active processes",
                        EXIT_UNCONFIRMED,
                    )
                if request_row is None:
                    raise IntegrityFailure("isolated state has no enforcing request")
                _finalize_isolation(
                    paths,
                    str(request_row["request_id"]),
                    target,
                    run_id,
                    termination_performed=True,
                    credential_was_issued=True,
                    actor="run-supervisor",
                )
                return EXIT_OK
            if root_exited:
                exit_code = managed.exit_code()
                if active > 0:
                    remaining = managed.terminate_and_confirm()
                    if remaining != 0:
                        raise ControlError(
                            "RESIDUAL_TREE_TERMINATION_UNCONFIRMED",
                            f"Job Object still reports {remaining} active processes",
                            EXIT_UNCONFIRMED,
                        )
                isolation_request_id: str | None = None
                with connect(paths) as connection:
                    with immediate_transaction(connection):
                        verify_audit(connection)
                        latest_target = connection.execute(
                            "SELECT control_state FROM targets WHERE target=?", (target,)
                        ).fetchone()
                        if latest_target is None:
                            raise IntegrityFailure("registered target disappeared")
                        if latest_target["control_state"] == "ISOLATED":
                            latest_request = connection.execute(
                                """
                                SELECT request_id FROM requests
                                WHERE target=? AND status IN (
                                    'ENFORCING','ENFORCEMENT_UNCONFIRMED','APPLIED'
                                )
                                ORDER BY CASE WHEN status='APPLIED' THEN 1 ELSE 0 END, accepted_at
                                LIMIT 1
                                """,
                                (target,),
                            ).fetchone()
                            if latest_request is None:
                                raise IntegrityFailure("isolated target has no accepted request")
                            isolation_request_id = str(latest_request["request_id"])
                        else:
                            observed = format_time(utc_now())
                            connection.execute(
                                """
                                UPDATE runs SET state='EXITED', job_active_processes=0,
                                    credential_revoked_at=?, ended_at=?, heartbeat_at=?, exit_code=?
                                WHERE run_id=?
                                """,
                                (observed, observed, observed, exit_code, run_id),
                            )
                            append_audit(
                                connection,
                                "PROCESS_TREE_EXITED",
                                target,
                                "run-supervisor",
                                {
                                    "exit_code": exit_code,
                                    "residual_tree_terminated": active > 0,
                                    "run_id": run_id,
                                },
                            )
                if isolation_request_id is not None:
                    _finalize_isolation(
                        paths,
                        isolation_request_id,
                        target,
                        run_id,
                        termination_performed=active > 0,
                        credential_was_issued=True,
                        actor="run-supervisor-exit-race",
                    )
                return EXIT_OK
            time.sleep(0.1)
    except ControlError as exc:
        if managed is not None:
            try:
                managed.terminate_and_confirm()
            except BaseException:
                pass
            managed.close()
            managed = None
        _record_supervisor_failure(paths, run_id, target, exc.code, exc.message)
        return exc.exit_code
    except BaseException as exc:
        if managed is not None:
            try:
                managed.terminate_and_confirm()
            except BaseException:
                pass
            managed.close()
            managed = None
        _record_supervisor_failure(paths, run_id, target, "SUPERVISOR_INTERNAL_ERROR", repr(exc))
        return EXIT_INTERNAL
    finally:
        if managed is not None:
            try:
                if managed.active_processes() > 0:
                    managed.terminate_and_confirm()
            except BaseException:
                pass
            managed.close()


def apply_isolation(
    paths: StatePaths,
    envelope: dict[str, Any],
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    if os.name != "nt":
        raise ControlError("WINDOWS_REQUIRED", "Windows Job Objects are required", EXIT_INTERNAL)
    timeout_seconds = validate_timeout(timeout_seconds)
    envelope = validate_envelope_structure(envelope, check_time=False)
    authorization = verify_request(paths, envelope, check_time=False)
    envelope_hash = hashlib.sha256(canonical_json(envelope)).hexdigest()
    target = str(envelope["target"])
    request_id = str(envelope["request_id"])
    nonce = str(envelope["nonce"])

    with connect(paths) as connection:
        replay = connection.execute(
            """
            SELECT request_id FROM requests
            WHERE request_id=? OR nonce=? OR envelope_sha256=?
            """,
            (request_id, nonce, envelope_hash),
        ).fetchone()
        if replay is not None:
            raise ControlError(
                "REPLAY_DENIED",
                "request_id, nonce, or envelope hash was already consumed",
                EXIT_DENIED,
            )

    validate_envelope_structure(envelope, check_time=True)
    accepted_at = format_time(utc_now())
    live_run: sqlite3.Row | None = None
    no_live_result: dict[str, Any] | None = None
    with connect(paths) as connection:
        with immediate_transaction(connection):
            verify_audit(connection)
            validate_envelope_structure(envelope, now=utc_now(), check_time=True)
            replay = connection.execute(
                """
                SELECT request_id FROM requests
                WHERE request_id=? OR nonce=? OR envelope_sha256=?
                """,
                (request_id, nonce, envelope_hash),
            ).fetchone()
            if replay is not None:
                raise ControlError(
                    "REPLAY_DENIED",
                    "request_id, nonce, or envelope hash was already consumed",
                    EXIT_DENIED,
                )
            target_row = connection.execute(
                "SELECT control_state FROM targets WHERE target=?", (target,)
            ).fetchone()
            if target_row is None:
                raise ControlError("TARGET_NOT_FOUND", "target is not registered", EXIT_CONFLICT)
            if target_row["control_state"] == "ISOLATED":
                raise ControlError(
                    "TARGET_ALREADY_ISOLATED",
                    "target is already durably isolated; this new request was not consumed",
                    EXIT_DENIED,
                )
            inflight = connection.execute(
                """
                SELECT request_id FROM requests
                WHERE target=? AND status IN ('ENFORCING','ENFORCEMENT_UNCONFIRMED')
                LIMIT 1
                """,
                (target,),
            ).fetchone()
            if inflight is not None:
                raise ControlError(
                    "ISOLATION_ALREADY_IN_PROGRESS",
                    f"request {inflight['request_id']} is already enforcing isolation",
                    EXIT_CONFLICT,
                )
            try:
                connection.execute(
                    """
                    INSERT INTO requests(
                        request_id, nonce, envelope_sha256, target, status,
                        accepted_at, expires_at
                    ) VALUES (?, ?, ?, ?, 'ENFORCING', ?, ?)
                    """,
                    (
                        request_id,
                        nonce,
                        envelope_hash,
                        target,
                        accepted_at,
                        envelope["expires_at"],
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ControlError(
                    "REPLAY_DENIED",
                    "request_id, nonce, or envelope hash was already consumed",
                    EXIT_DENIED,
                ) from exc
            connection.execute(
                "UPDATE targets SET control_state='ISOLATED', updated_at=? WHERE target=?",
                (accepted_at, target),
            )
            live_run = connection.execute(
                """
                SELECT * FROM runs
                WHERE target=? AND state IN ('STARTING','RUNNING','ISOLATING')
                ORDER BY created_at DESC LIMIT 1
                """,
                (target,),
            ).fetchone()
            if live_run is not None:
                connection.execute(
                    """
                    UPDATE runs SET state='ISOLATING', credential_revoked_at=COALESCE(credential_revoked_at, ?)
                    WHERE run_id=?
                    """,
                    (accepted_at, live_run["run_id"]),
                )
            append_audit(
                connection,
                "ISOLATION_ACCEPTED",
                target,
                "authorized-quorum",
                {
                    "authorization_valid": True,
                    "request_id": request_id,
                    "run_id": live_run["run_id"] if live_run else None,
                    "signers": authorization["signers"],
                    "supervisor_local_credential_revoked_before_termination": bool(
                        live_run is not None and live_run["credential_hash"]
                    ),
                },
            )
            if live_run is None:
                no_live_result = _finalize_isolation_in_transaction(
                    connection,
                    _build_isolation_result(
                        request_id,
                        target,
                        None,
                        termination_performed=False,
                        credential_was_issued=False,
                    ),
                    request_id,
                    target,
                    None,
                    termination_performed=False,
                    actor="apply-client",
                )

    if live_run is None:
        if no_live_result is None:
            raise IntegrityFailure("no-live-run isolation did not finalize atomically")
        return no_live_result

    run_id = str(live_run["run_id"])
    deadline = time.monotonic() + timeout_seconds
    last_reconcile = 0.0
    while time.monotonic() < deadline:
        if time.monotonic() - last_reconcile >= 1.0:
            reconcile_stale_runs(paths, target)
            last_reconcile = time.monotonic()
        with connect(paths) as connection:
            row = connection.execute(
                "SELECT status, result_json FROM requests WHERE request_id=?", (request_id,)
            ).fetchone()
        if row is not None and row["status"] == "APPLIED" and row["result_json"]:
            return parse_json_bytes(str(row["result_json"]).encode("utf-8"))
        if row is not None and row["status"] == "ENFORCEMENT_UNCONFIRMED":
            reconcile_stale_runs(paths, target)
        time.sleep(0.1)

    reconcile_stale_runs(paths, target)
    with connect(paths) as connection:
        with immediate_transaction(connection):
            verify_audit(connection)
            final_row = connection.execute(
                "SELECT status, result_json FROM requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if final_row is None:
                raise IntegrityFailure("accepted isolation request disappeared")
            if final_row["status"] == "APPLIED" and final_row["result_json"]:
                return parse_json_bytes(str(final_row["result_json"]).encode("utf-8"))
            append_audit(
                connection,
                "ISOLATION_CLIENT_WAIT_TIMEOUT",
                target,
                "apply-client",
                {
                    "durable_request_status": str(final_row["status"]),
                    "request_id": request_id,
                    "run_id": run_id,
                    "target_remains_blocked": True,
                },
            )
    raise ControlError(
        "ENFORCEMENT_UNCONFIRMED",
        "client wait expired; restart block is durable and reconciliation resumes on the next controller command",
        EXIT_UNCONFIRMED,
    )


def target_status(paths: StatePaths, target: str) -> dict[str, Any]:
    target = validate_target(target)
    reconciliation = reconcile_stale_runs(paths, target)
    with connect(paths) as connection:
        connection.execute("BEGIN")
        try:
            audit = verify_audit(connection)
            target_row = connection.execute(
                "SELECT * FROM targets WHERE target=?", (target,)
            ).fetchone()
            if target_row is None:
                raise ControlError("TARGET_NOT_FOUND", "target is not registered", EXIT_CONFLICT)
            run = connection.execute(
                "SELECT * FROM runs WHERE target=? ORDER BY created_at DESC LIMIT 1", (target,)
            ).fetchone()
            request = connection.execute(
                "SELECT * FROM requests WHERE target=? ORDER BY accepted_at DESC LIMIT 1", (target,)
            ).fetchone()
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
    credential_state = "NOT_ISSUED"
    if run is not None and run["credential_hash"]:
        if run["credential_revoked_at"]:
            credential_state = "REVOKED"
        elif parse_time(run["credential_expires_at"], "credential_expires_at") <= utc_now():
            credential_state = "EXPIRED"
        elif (
            run["state"] == "RUNNING"
            and target_row["control_state"] == "READY"
            and heartbeat_is_fresh(run["heartbeat_at"])
            and query_process_status(
                int(run["supervisor_pid"] or 0),
                int(run["supervisor_created_filetime"] or 0),
            )
            == PROCESS_ALIVE
        ):
            credential_state = "VALID_FOR_LOCAL_SUPERVISOR_CHECKS"
        else:
            credential_state = "INVALID"
    return {
        "audit_integrity": audit["integrity"],
        "enforcement_scope": "local_windows_supervised_processes_only",
        "latest_request_id": request["request_id"] if request else None,
        "latest_request_status": request["status"] if request else None,
        "local_credential_state": credential_state,
        "ok": True,
        "operation": "status",
        "reconciliation": reconciliation,
        "process_tree": {
            "child_pid": run["child_pid"] if run else None,
            "exit_code": run["exit_code"] if run else None,
            "job_active_processes_last_observed": run["job_active_processes"] if run else 0,
            "run_id": run["run_id"] if run else None,
            "state": run["state"] if run else "NEVER_STARTED",
            "supervisor_pid": run["supervisor_pid"] if run else None,
            "supervisor_created_filetime": (
                run["supervisor_created_filetime"] if run else None
            ),
            "supervisor_heartbeat_at": run["heartbeat_at"] if run else None,
        },
        "restart_blocked": target_row["control_state"] == "ISOLATED",
        "target": target,
        "trust_store_sealed": audit["trust_store_sealed"],
        "truth_boundary": {
            "external_sessions_revoked": False,
            "network_policy_enforced": False,
            "provider_credentials_revoked": False,
        },
    }


def credential_check(paths: StatePaths, target: str, token: str) -> dict[str, Any]:
    target = validate_target(target)
    if not isinstance(token, str) or len(token) > 4096:
        raise ControlError("LOCAL_CREDENTIAL_INVALID", "credential input is invalid", EXIT_DENIED)
    reconcile_stale_runs(paths, target)
    supplied_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with connect(paths) as connection:
        connection.execute("BEGIN")
        try:
            verify_audit(connection)
            row = connection.execute(
                """
                SELECT r.*, t.control_state FROM runs r
                JOIN targets t ON t.target=r.target
                WHERE r.target=? ORDER BY r.created_at DESC LIMIT 1
                """,
                (target,),
            ).fetchone()
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
    valid = bool(
        row is not None
        and row["state"] == "RUNNING"
        and row["control_state"] == "READY"
        and row["credential_revoked_at"] is None
        and row["credential_hash"] is not None
        and heartbeat_is_fresh(row["heartbeat_at"])
        and query_process_status(
            int(row["supervisor_pid"] or 0),
            int(row["supervisor_created_filetime"] or 0),
        )
        == PROCESS_ALIVE
        and parse_time(row["credential_expires_at"], "credential_expires_at") > utc_now()
        and secrets.compare_digest(str(row["credential_hash"]), supplied_hash)
    )
    return {
        "credential_valid": valid,
        "ok": valid,
        "operation": "credential-check",
        "scope": "local_supervisor_generation_only",
        "target": target,
    }


PROCESS_ALIVE = "ALIVE"
PROCESS_EXITED = "EXITED"
PROCESS_UNKNOWN = "UNKNOWN"


def query_process_status(pid: int, expected_created_filetime: int = 0) -> str:
    if os.name != "nt" or pid <= 0:
        return PROCESS_EXITED
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    SYNCHRONIZE = 0x00100000
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid
    )
    if not handle:
        error = ctypes.get_last_error()
        if error in {87, 1168}:  # invalid/nonexistent PID
            return PROCESS_EXITED
        return PROCESS_UNKNOWN
    try:
        wait_result = kernel32.WaitForSingleObject(handle, 0)
        if wait_result == WAIT_OBJECT_0:
            return PROCESS_EXITED
        if wait_result != WAIT_TIMEOUT:
            return PROCESS_UNKNOWN
        if expected_created_filetime:
            try:
                observed_created_filetime = process_handle_creation_filetime(handle)
            except ControlError:
                return PROCESS_UNKNOWN
            if observed_created_filetime != expected_created_filetime:
                # The persisted PID was reused; the recorded supervisor exited.
                return PROCESS_EXITED
        return PROCESS_ALIVE
    finally:
        kernel32.CloseHandle(handle)


def process_running(pid: int, expected_created_filetime: int = 0) -> bool:
    # UNKNOWN is deliberately conservative: callers waiting for verified
    # absence must not treat an access/query failure as proof of exit.
    return query_process_status(pid, expected_created_filetime) != PROCESS_EXITED


def terminate_exact_process_identity(
    pid: int, expected_created_filetime: int, timeout_seconds: float = 5.0
) -> str:
    """Terminate only the exact PID + creation-time identity, never a bare PID."""

    if os.name != "nt" or pid <= 0 or expected_created_filetime <= 0:
        return PROCESS_UNKNOWN
    PROCESS_TERMINATE = 0x0001
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    SYNCHRONIZE = 0x00100000
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(
        PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE,
        False,
        pid,
    )
    if not handle:
        return query_process_status(pid, expected_created_filetime)
    try:
        wait_result = kernel32.WaitForSingleObject(handle, 0)
        if wait_result == WAIT_OBJECT_0:
            return PROCESS_EXITED
        if wait_result != WAIT_TIMEOUT:
            return PROCESS_UNKNOWN
        try:
            observed_created_filetime = process_handle_creation_filetime(handle)
        except ControlError:
            return PROCESS_UNKNOWN
        if observed_created_filetime != expected_created_filetime:
            return PROCESS_EXITED
        if not kernel32.TerminateProcess(handle, 0xE0000003):
            if kernel32.WaitForSingleObject(handle, 0) == WAIT_OBJECT_0:
                return PROCESS_EXITED
            return PROCESS_UNKNOWN
        timeout_ms = max(1, min(int(timeout_seconds * 1000), 60_000))
        return (
            PROCESS_EXITED
            if kernel32.WaitForSingleObject(handle, timeout_ms) == WAIT_OBJECT_0
            else PROCESS_UNKNOWN
        )
    finally:
        kernel32.CloseHandle(handle)


def heartbeat_is_fresh(value: Any, maximum_age_seconds: float = 3.0) -> bool:
    if not value:
        return False
    try:
        observed = parse_time(value, "heartbeat_at")
    except ControlError:
        return False
    age = (utc_now() - observed).total_seconds()
    return -MAX_CLOCK_SKEW_SECONDS <= age <= maximum_age_seconds


def reconcile_stale_runs(paths: StatePaths, target: str | None = None) -> list[dict[str, Any]]:
    """Reconcile runs using exact Windows process identity.

    The Job Object is unnamed and its only handle is held by the detached
    supervisor. Windows KILL_ON_JOB_CLOSE therefore removes the supervised tree
    when that process exits. A stale live supervisor is terminated only after a
    signed isolation has durably blocked the target, and only when both its PID
    and creation FILETIME still match. A bare persisted PID is never killed.
    """

    if os.name != "nt":
        return []
    parameters: tuple[Any, ...] = ()
    predicate = ""
    if target is not None:
        target = validate_target(target)
        predicate = " AND r.target=?"
        parameters = (target,)
    with connect(paths) as connection:
        connection.execute("BEGIN")
        try:
            verify_audit(connection)
            rows = connection.execute(
                f"""
                SELECT r.*, t.control_state
                FROM runs r JOIN targets t ON t.target=r.target
                WHERE (
                    r.state IN ('STARTING','RUNNING','ISOLATING')
                    OR (r.state='FAILED' AND t.control_state='ISOLATED')
                ){predicate}
                """,
                parameters,
            ).fetchall()
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
    results: list[dict[str, Any]] = []
    for row in rows:
        supervisor_pid = int(row["supervisor_pid"] or 0)
        supervisor_status = query_process_status(
            supervisor_pid, int(row["supervisor_created_filetime"] or 0)
        )
        heartbeat_fresh = heartbeat_is_fresh(row["heartbeat_at"])
        created_age = (
            utc_now() - parse_time(row["created_at"], "created_at")
        ).total_seconds()
        supervisor_termination_fallback = False
        if supervisor_status == PROCESS_ALIVE:
            # READY targets are never killed by reconciliation. After signed
            # isolation, an exact PID + creation-time match can be terminated
            # to close the supervisor's sole unnamed Job handle.
            if heartbeat_fresh or (row["state"] == "STARTING" and created_age < 10.0):
                continue
            if row["control_state"] == "ISOLATED":
                supervisor_status = terminate_exact_process_identity(
                    supervisor_pid,
                    int(row["supervisor_created_filetime"] or 0),
                )
                supervisor_termination_fallback = supervisor_status == PROCESS_EXITED
            if supervisor_status != PROCESS_EXITED:
                results.append(
                    {
                        "operation_status": "STALE_SUPERVISOR_UNCONFIRMED",
                        "run_id": row["run_id"],
                        "target": row["target"],
                    }
                )
                continue

        if supervisor_status == PROCESS_UNKNOWN:
            results.append(
                {
                    "operation_status": "SUPERVISOR_IDENTITY_UNCONFIRMED",
                    "run_id": row["run_id"],
                    "target": row["target"],
                }
            )
            continue

        if (
            row["state"] == "STARTING"
            and row["control_state"] == "READY"
            and created_age < 10.0
        ):
            # The start client records the detached bootstrap PID immediately
            # after Popen. Do not let a concurrent status call fail that tiny
            # reservation-to-Popen window.
            continue

        run_id = str(row["run_id"])
        run_target = str(row["target"])
        if row["control_state"] == "ISOLATED":
            with connect(paths) as connection:
                request = connection.execute(
                    """
                    SELECT request_id FROM requests
                    WHERE target=? AND status IN (
                        'ENFORCING','ENFORCEMENT_UNCONFIRMED','APPLIED'
                    )
                    ORDER BY CASE WHEN status='APPLIED' THEN 1 ELSE 0 END, accepted_at
                    LIMIT 1
                    """,
                    (run_target,),
                ).fetchone()
            if request is not None:
                _finalize_isolation(
                    paths,
                    str(request["request_id"]),
                    run_target,
                    run_id,
                    termination_performed=False,
                    credential_was_issued=bool(row["credential_hash"]),
                    actor="stale-run-reconciler",
                    job_close_cleanup_basis=bool(row["child_pid"]),
                )
                results.append(
                    {
                        "operation_status": "VERIFIED_ISOLATED_AFTER_SUPERVISOR_EXIT",
                        "run_id": run_id,
                        "supervisor_termination_fallback_performed": (
                            supervisor_termination_fallback
                        ),
                        "target": run_target,
                    }
                )
                continue

        with connect(paths) as connection:
            with immediate_transaction(connection):
                verify_audit(connection)
                current = connection.execute(
                    """
                    SELECT state, supervisor_pid, supervisor_created_filetime, heartbeat_at
                    FROM runs WHERE run_id=?
                    """,
                    (run_id,),
                ).fetchone()
                if current is None or current["state"] not in {
                    "STARTING",
                    "RUNNING",
                    "ISOLATING",
                }:
                    continue
                current_status = query_process_status(
                    int(current["supervisor_pid"] or 0),
                    int(current["supervisor_created_filetime"] or 0),
                )
                if current_status != PROCESS_EXITED:
                    continue
                connection.execute(
                    """
                    UPDATE runs SET state='FAILED', job_active_processes=0,
                        credential_revoked_at=COALESCE(credential_revoked_at, ?),
                        ended_at=?, heartbeat_at=?, error_code='SUPERVISOR_EXIT_RECONCILED',
                        error_message='supervisor absent; unnamed Job handle closure ended the tree'
                    WHERE run_id=?
                    """,
                    (
                        format_time(utc_now()),
                        format_time(utc_now()),
                        format_time(utc_now()),
                        run_id,
                    ),
                )
                append_audit(
                    connection,
                    "SUPERVISOR_EXIT_RECONCILED",
                    run_target,
                    "stale-run-reconciler",
                    {
                        "run_id": run_id,
                        "supervisor_pid_observed_absent": True,
                        "unnamed_job_kill_on_close": True,
                    },
                )
        results.append(
            {
                "operation_status": "SUPERVISOR_EXIT_RECONCILED",
                "run_id": run_id,
                "target": run_target,
            }
        )
    return results


def run_demo_target(
    paths: StatePaths, heartbeat_path: Path, child_record_path: Path
) -> int:
    raw_target = os.environ.get(AGENT_ID_ENV)
    generation = os.environ.get(GENERATION_ENV)
    local_credential = os.environ.pop(LOCAL_CREDENTIAL_ENV, None)
    if not raw_target or not generation or not local_credential:
        raise ControlError(
            "DEMO_CAPABILITY_MISSING",
            "the demo target can only run as a credentialed supervised generation",
            EXIT_DENIED,
        )
    target = validate_target(raw_target)
    expected_directory = (paths.demo / target.replace(":", "_")).resolve()
    heartbeat_path = Path(os.path.abspath(heartbeat_path))
    child_record_path = Path(os.path.abspath(child_record_path))
    if (
        heartbeat_path.parent.resolve() != expected_directory
        or child_record_path.parent.resolve() != expected_directory
        or heartbeat_path.name != "heartbeat.json"
        or child_record_path.name != "child.json"
    ):
        raise ControlError(
            "DEMO_PATH_INVALID",
            "demo evidence paths must be the registered target's fixed state paths",
            EXIT_DENIED,
        )
    deadline = time.monotonic() + 5.0
    credential_valid = False
    while time.monotonic() < deadline:
        credential_valid = credential_check(paths, target, local_credential)["credential_valid"]
        if credential_valid:
            break
        time.sleep(0.05)
    local_credential = ""
    if not credential_valid:
        raise ControlError(
            "DEMO_CAPABILITY_INVALID",
            "the demo target's supervisor-local generation credential was not valid",
            EXIT_DENIED,
        )
    child_environment = safe_child_environment({"PYTHONIOENCODING": "utf-8"})
    child = subprocess.Popen(
        [
            str(Path(sys.executable).resolve()),
            "-I",
            "-B",
            "-c",
            "import time; time.sleep(3600)",
        ],
        shell=False,
        env=child_environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    atomic_write(
        child_record_path,
        (pretty_json({"child_pid": child.pid, "parent_pid": os.getpid()}) + "\n").encode("utf-8"),
    )
    counter = 0
    while True:
        atomic_write(
            heartbeat_path,
            (
                pretty_json(
                    {
                        "agent_id": os.environ.get(AGENT_ID_ENV),
                        "child_pid": child.pid,
                        "counter": counter,
                        "generation": generation,
                        "parent_pid": os.getpid(),
                        "timestamp": format_time(utc_now()),
                    }
                )
                + "\n"
            ).encode("utf-8"),
        )
        counter += 1
        time.sleep(0.2)
def read_passphrase(environment_name: str | None, *, confirm: bool) -> bytes:
    if environment_name:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,63}", environment_name):
            raise ControlError(
                "INVALID_PASSPHRASE_ENV",
                "passphrase environment variable name must be uppercase and bounded",
                EXIT_INPUT,
            )
        value = os.environ.get(environment_name)
        if value is None:
            raise ControlError(
                "PASSPHRASE_ENV_MISSING",
                f"environment variable {environment_name} is not set",
                EXIT_INPUT,
            )
        return value.encode("utf-8")
    first = getpass.getpass("Private-key passphrase: ").encode("utf-8")
    if confirm:
        second = getpass.getpass("Confirm passphrase: ").encode("utf-8")
        if not secrets.compare_digest(first, second):
            raise ControlError("PASSPHRASE_MISMATCH", "passphrases did not match", EXIT_INPUT)
    return first


def wait_for_file(path: Path, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if path.is_file():
                value = load_json_file(path)
                if isinstance(value, dict):
                    return value
        except Exception as exc:  # bounded retry for an atomic producer
            last_error = exc
        time.sleep(0.05)
    raise ControlError(
        "DEMO_EVIDENCE_TIMEOUT",
        f"timed out waiting for {path}; last_error={last_error!r}",
        EXIT_UNCONFIRMED,
    )


def self_test() -> dict[str, Any]:
    if os.name != "nt":
        raise ControlError("WINDOWS_REQUIRED", "self-test requires Windows", EXIT_INTERNAL)
    require_cryptography()
    checks: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(
        prefix="owned-agent-control-self-test-", ignore_cleanup_errors=True
    ) as directory:
        paths = state_paths(Path(directory).resolve())
        initialize_state(paths)
        key_a = Ed25519PrivateKey.generate()
        key_b = Ed25519PrivateKey.generate()
        add_operator_raw(paths, "synthetic_a", key_a.public_key(), actor="self-test")
        add_operator_raw(paths, "synthetic_b", key_b.public_key(), actor="self-test")
        seal_trust_store(paths, actor="self-test")
        target = "owned-agent:self-test"
        register_demo(paths, target)
        start_result = start_agent(paths, target, timeout_seconds=20.0)
        checks["start"] = start_result["operation_status"]
        heartbeat_path, child_record_path = demo_paths(paths, target)
        heartbeat = wait_for_file(heartbeat_path, 10.0)
        child_record = wait_for_file(child_record_path, 10.0)
        parent_pid = int(heartbeat["parent_pid"])
        child_pid = int(child_record["child_pid"])
        checks["parent_running_before"] = process_running(parent_pid)
        checks["child_running_before"] = process_running(child_pid)
        if not checks["parent_running_before"] or not checks["child_running_before"]:
            raise ControlError(
                "SELF_TEST_PROCESS_NOT_RUNNING",
                "demo parent and child were not both observed running",
                EXIT_UNCONFIRMED,
            )

        unsigned = build_request(paths, target, 300)
        one_signature = sign_request_with_key(paths, unsigned, "synthetic_a", key_a)
        try:
            verify_request(paths, one_signature)
        except ControlError as exc:
            checks["one_signature_denial"] = exc.code
        else:
            raise ControlError("SELF_TEST_FAILED", "one-signature request was accepted", EXIT_INTERNAL)
        approved = sign_request_with_key(paths, one_signature, "synthetic_b", key_b)
        verified = verify_request(paths, approved)
        checks["two_signature_verification"] = verified["authorization_valid"]

        tampered = copy.deepcopy(approved)
        tampered["expires_at"] = format_time(
            parse_time(tampered["expires_at"], "expires_at") - timedelta(microseconds=1)
        )
        try:
            verify_request(paths, tampered)
        except ControlError as exc:
            checks["tamper_denial"] = exc.code
        else:
            raise ControlError("SELF_TEST_FAILED", "tampered request was accepted", EXIT_INTERNAL)

        isolation = apply_isolation(paths, approved, timeout_seconds=20.0)
        checks["isolation"] = isolation["operation_status"]
        with connect(paths) as connection:
            supervisor_row = connection.execute(
                "SELECT supervisor_pid FROM runs WHERE run_id=?", (start_result["run_id"],)
            ).fetchone()
        supervisor_pid = int(supervisor_row["supervisor_pid"] or 0)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and (
            process_running(parent_pid) or process_running(child_pid)
        ):
            time.sleep(0.05)
        checks["parent_running_after"] = process_running(parent_pid)
        checks["child_running_after"] = process_running(child_pid)
        if checks["parent_running_after"] or checks["child_running_after"]:
            raise ControlError(
                "SELF_TEST_TERMINATION_UNCONFIRMED",
                "demo process tree remained active after isolation",
                EXIT_UNCONFIRMED,
            )
        supervisor_deadline = time.monotonic() + 5.0
        while time.monotonic() < supervisor_deadline and process_running(supervisor_pid):
            time.sleep(0.05)
        checks["supervisor_running_after"] = process_running(supervisor_pid)
        if checks["supervisor_running_after"]:
            raise ControlError(
                "SELF_TEST_SUPERVISOR_EXIT_UNCONFIRMED",
                "run supervisor did not exit after verified isolation",
                EXIT_UNCONFIRMED,
            )

        try:
            apply_isolation(paths, approved, timeout_seconds=1.0)
        except ControlError as exc:
            checks["replay_denial"] = exc.code
        else:
            raise ControlError("SELF_TEST_FAILED", "replayed request was accepted", EXIT_INTERNAL)

        try:
            start_agent(paths, target, timeout_seconds=1.0)
        except ControlError as exc:
            checks["restart_denial"] = exc.code
        else:
            raise ControlError("SELF_TEST_FAILED", "isolated target restarted", EXIT_INTERNAL)

        status = target_status(paths, target)
        checks["restart_blocked"] = status["restart_blocked"]
        checks["credential_state"] = status["local_credential_state"]
        if not checks["restart_blocked"] or checks["credential_state"] != "REVOKED":
            raise ControlError(
                "SELF_TEST_FAILED",
                "restart block or local credential revocation was not observed",
                EXIT_INTERNAL,
            )
        with connect(paths) as connection:
            audit = verify_audit(connection)
        checks["audit_events_verified"] = audit["events_verified"]
        checks["audit_integrity"] = audit["integrity"]

        required = {
            "one_signature_denial": "QUORUM_NOT_MET",
            "tamper_denial": "SIGNATURE_INVALID",
            "replay_denial": "REPLAY_DENIED",
            "restart_denial": "TARGET_ISOLATED",
        }
        for name, expected in required.items():
            if checks.get(name) != expected:
                raise ControlError(
                    "SELF_TEST_FAILED",
                    f"{name} expected {expected}, observed {checks.get(name)}",
                    EXIT_INTERNAL,
                )
    return {
        "checks": checks,
        "ok": True,
        "operation": "self-test",
        "operation_status": "VERIFIED_LOCAL_WINDOWS_ENFORCEMENT",
        "truth_boundary": "local_windows_supervised_processes_only",
    }


CLINICAL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{1,127}$")
CLINICAL_TRANSITIONS = {
    "NONE": {"RECEIVED"},
    "RECEIVED": {"PARSED", "QUARANTINED"},
    "PARSED": {"VALIDATED", "QUARANTINED"},
    "VALIDATED": {"PENDING_REVIEW", "QUARANTINED"},
    "PENDING_REVIEW": {"AUTHORIZED_FOR_EXPORT", "REJECTED", "QUARANTINED"},
    "AUTHORIZED_FOR_EXPORT": {"ARTIFACT_CREATED", "QUARANTINED"},
    "ARTIFACT_CREATED": {"SUPERSEDED"},
    "QUARANTINED": set(),
    "REJECTED": set(),
    "SUPERSEDED": set(),
}


@dataclass(frozen=True)
class ClinicalPolicyCheck:
    gate_code: str
    outcome: str
    reason_code: str
    evidence_sha256: str


class EntropyDepthAllocator:
    """Bound validation work only; never changes or scores a clinical result."""

    LAYERS = ("transport", "semantic", "release")

    def __init__(self, budget: int = 3):
        if type(budget) is not int or not 1 <= budget <= len(self.LAYERS):
            raise ControlError("INVALID_VALIDATION_BUDGET", "validation budget must be 1..3", EXIT_INPUT)
        self.budget = budget

    def allocate(self, required_layers: Sequence[str]) -> tuple[str, ...]:
        requested = tuple(required_layers)
        if any(layer not in self.LAYERS for layer in requested) or len(requested) > self.budget:
            raise ControlError(
                "VALIDATION_BUDGET_EXHAUSTED",
                "required deterministic validation layers exceed the configured budget",
                EXIT_DENIED,
            )
        return requested


class CrossStepConsistency:
    """Exact invariants, not a confidence score or clinical interpretation."""

    @staticmethod
    def evaluate(parsed: Mapping[str, Any], binding: Mapping[str, Any]) -> dict[str, bool]:
        return {
            "source_subject_bound": parsed["source_subject_token"] == binding["source_subject_token"],
            "source_order_bound": parsed["source_order_token"] == binding["source_order_token"],
            "patient_reference_bound": str(binding["patient_reference"]).startswith("Patient/"),
            "order_reference_bound": str(binding["order_reference"]).startswith("ServiceRequest/"),
            "specimen_reference_bound": str(binding["specimen_reference"]).startswith("Specimen/"),
            "recipient_bound": bool(binding["recipient_id"]),
        }


class LoopKernelStateTransition:
    """Append-only clinical-result lifecycle; no bypass or auto-approval edge."""

    @staticmethod
    def allowed(from_state: str, to_state: str) -> bool:
        return to_state in CLINICAL_TRANSITIONS.get(from_state, set())


def clinical_database_path(paths: StatePaths) -> Path:
    return paths.root / "clinical.sqlite3"


def clinical_database_sidecars(database: Path) -> tuple[Path, Path, Path]:
    return tuple(Path(str(database) + suffix) for suffix in ("-journal", "-shm", "-wal"))


def configure_clinical_connection(
    connection: sqlite3.Connection,
    *,
    initialize_wal: bool = False,
) -> None:
    """Apply and verify every connection-local clinical durability control."""
    connection.execute("PRAGMA busy_timeout = 10000")
    if initialize_wal:
        journal_mode = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0])
    else:
        journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0])
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA synchronous = FULL")
    foreign_keys = int(connection.execute("PRAGMA foreign_keys").fetchone()[0])
    synchronous = int(connection.execute("PRAGMA synchronous").fetchone()[0])
    if journal_mode.lower() != "wal":
        raise IntegrityFailure("clinical database journal mode is not WAL")
    if foreign_keys != 1:
        raise IntegrityFailure("clinical database foreign-key enforcement is disabled")
    if synchronous < 2:
        raise IntegrityFailure("clinical database synchronous durability is below FULL")


def refuse_existing_clinical_database(database: Path) -> None:
    """Preserve pre-existing state and distinguish complete from partial initialization."""
    if not database.exists():
        return
    complete = False
    if database.is_file() and not _is_reparse_point(database):
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                database.resolve().as_uri() + "?mode=ro",
                timeout=2.0,
                isolation_level=None,
                uri=True,
                factory=ClosingSQLiteConnection,
            )
            connection.row_factory = sqlite3.Row
            rows = {
                str(row["key"]): str(row["value"])
                for row in connection.execute(
                    "SELECT key,value FROM clinical_metadata "
                    "WHERE key IN ('schema_version','initialization_complete')"
                )
            }
            complete = rows == {
                "initialization_complete": "1",
                "schema_version": CLINICAL_SCHEMA_VERSION,
            }
        except sqlite3.DatabaseError:
            complete = False
        finally:
            if connection is not None:
                connection.close()
    if complete:
        raise ControlError(
            "CLINICAL_ALREADY_INITIALIZED", "clinical database already exists", EXIT_CONFLICT
        )
    raise ControlError(
        "CLINICAL_STATE_INCOMPLETE",
        "a pre-existing clinical database is incomplete, unsafe, or from another schema; preserving it",
        EXIT_CONFLICT,
    )


def clinical_write_output(paths: StatePaths, path: Path, data: bytes, label: str) -> Path:
    resolved = Path(os.path.abspath(path.expanduser()))
    state_databases = {paths.database.resolve(), clinical_database_path(paths).resolve()}
    forbidden = {
        paths.database.resolve(),
        paths.lock.resolve(),
        clinical_database_path(paths).resolve(),
    }
    for database in state_databases:
        for suffix in ("-journal", "-shm", "-wal"):
            forbidden.add(Path(str(database) + suffix).resolve())
    if resolved.resolve() in forbidden:
        raise ControlError(
            "UNSAFE_OUTPUT_PATH",
            f"{label} may not target controller or clinical state files",
            EXIT_DENIED,
        )
    return safe_new_or_identical_output(resolved, data, label)


def clinical_connect(paths: StatePaths) -> sqlite3.Connection:
    ensure_state_root(paths)
    database = clinical_database_path(paths)
    if not database.is_file() or _is_reparse_point(database):
        raise ControlError("CLINICAL_NOT_INITIALIZED", "clinical state is missing or unsafe", EXIT_CONFLICT)
    connection = sqlite3.connect(
        database,
        timeout=10.0,
        isolation_level=None,
        factory=ClosingSQLiteConnection,
    )
    connection.row_factory = sqlite3.Row
    try:
        configure_clinical_connection(connection)
        complete = connection.execute(
            "SELECT value FROM clinical_metadata WHERE key='initialization_complete'"
        ).fetchone()
    except (sqlite3.DatabaseError, ControlError) as exc:
        connection.close()
        if isinstance(exc, ControlError) and not isinstance(exc, IntegrityFailure):
            raise
        raise ControlError(
            "CLINICAL_STATE_INCOMPLETE",
            "clinical state exists but is not a complete trusted schema",
            EXIT_CONFLICT,
        ) from exc
    if complete is None or str(complete["value"]) != "1":
        connection.close()
        raise ControlError(
            "CLINICAL_STATE_INCOMPLETE",
            "clinical initialization-completion marker is missing",
            EXIT_CONFLICT,
        )
    return connection


def clinical_audit_body(
    sequence: int,
    timestamp: str,
    event_type: str,
    result_id: str | None,
    actor: str,
    payload: Any,
) -> dict[str, Any]:
    return {
        "actor": actor,
        "event_type": event_type,
        "payload": payload,
        "result_id": result_id,
        "sequence": sequence,
        "timestamp": timestamp,
        "version": 1,
    }


def clinical_append_only_trigger_sql(table: str, operation: str) -> str:
    sql_operation = {"no_update": "UPDATE", "no_delete": "DELETE"}[operation]
    return (
        f"CREATE TRIGGER {table}_{operation} BEFORE {sql_operation} ON {table} "
        f"BEGIN SELECT RAISE(ABORT,'{table} is append-only'); END"
    )


def normalize_schema_sql(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def clinical_schema_fingerprint(connection: sqlite3.Connection) -> str:
    definitions = [
        {
            "name": str(row["name"]),
            "sql": normalize_schema_sql(str(row["sql"])),
            "type": str(row["type"]),
        }
        for row in connection.execute(
            """
            SELECT type,name,sql FROM sqlite_master
            WHERE type IN ('table','trigger') AND name LIKE 'clinical_%' AND sql IS NOT NULL
            ORDER BY type,name
            """
        )
    ]
    return hashlib.sha256(canonical_json(definitions)).hexdigest()


def clinical_append_audit(
    connection: sqlite3.Connection,
    event_type: str,
    result_id: str | None,
    actor: str,
    payload: Any,
) -> str:
    row = connection.execute(
        "SELECT sequence, event_hash FROM clinical_audit_events ORDER BY sequence DESC LIMIT 1"
    ).fetchone()
    sequence = 1 if row is None else int(row["sequence"]) + 1
    previous_hash = ZERO_HASH if row is None else str(row["event_hash"])
    tip = connection.execute(
        "SELECT value FROM clinical_metadata WHERE key='audit_tip'"
    ).fetchone()
    if tip is None or str(tip["value"]) != previous_hash:
        raise IntegrityFailure("clinical audit tip does not match the event chain")
    timestamp = format_time(utc_now())
    body = clinical_audit_body(sequence, timestamp, event_type, result_id, actor, payload)
    event_hash = hashlib.sha256(bytes.fromhex(previous_hash) + canonical_json(body)).hexdigest()
    connection.execute(
        """
        INSERT INTO clinical_audit_events(
            sequence,timestamp,event_type,result_id,actor,payload_json,previous_hash,event_hash
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (sequence, timestamp, event_type, result_id, actor, canonical_json(payload).decode("utf-8"), previous_hash, event_hash),
    )
    connection.execute("UPDATE clinical_metadata SET value=? WHERE key='audit_tip'", (event_hash,))
    return event_hash


def clinical_initialize(paths: StatePaths) -> dict[str, Any]:
    paths.root.mkdir(parents=True, exist_ok=True)
    ensure_state_root(paths)
    acl = _apply_owner_only_acl(paths.root)
    database = clinical_database_path(paths)
    refuse_existing_clinical_database(database)
    with ConfigurationLock(paths.lock):
        refuse_existing_clinical_database(database)
        descriptor, staged_name = tempfile.mkstemp(
            prefix=".clinical.sqlite3.", suffix=".initialize", dir=paths.root
        )
        os.close(descriptor)
        staged = Path(staged_name)
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                staged,
                timeout=10.0,
                isolation_level=None,
                factory=ClosingSQLiteConnection,
            )
            connection.row_factory = sqlite3.Row
            configure_clinical_connection(connection, initialize_wal=True)
            connection.executescript(
                """
                CREATE TABLE clinical_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) STRICT;
                CREATE TABLE clinical_sources(
                    source_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL CHECK(mode IN ('MOCK','LIVE_SHADOW')),
                    profile_id TEXT NOT NULL,sender_application TEXT NOT NULL,sender_facility TEXT NOT NULL,
                    assay_map_json TEXT NOT NULL,config_json TEXT NOT NULL,config_sha256 TEXT NOT NULL UNIQUE,
                    active INTEGER NOT NULL CHECK(active IN (0,1)),created_at TEXT NOT NULL
                ) STRICT;
                CREATE TABLE clinical_reviewers(
                    reviewer_id TEXT PRIMARY KEY,public_key_b64 TEXT NOT NULL,
                    fingerprint TEXT NOT NULL UNIQUE,enabled INTEGER NOT NULL CHECK(enabled IN (0,1)),
                    created_at TEXT NOT NULL
                ) STRICT;
                CREATE TABLE clinical_messages(
                    message_id TEXT PRIMARY KEY,source_id TEXT NOT NULL REFERENCES clinical_sources(source_id),
                    received_at TEXT NOT NULL,raw_sha256 TEXT NOT NULL,raw_b64 TEXT NOT NULL,
                    message_control_id TEXT NOT NULL,message_type TEXT NOT NULL,hl7_version TEXT NOT NULL,
                    character_set TEXT NOT NULL,ingest_outcome TEXT NOT NULL,
                    UNIQUE(source_id,message_control_id)
                ) STRICT;
                CREATE TABLE clinical_ingest_attempts(
                    attempt_id TEXT PRIMARY KEY,source_id TEXT NOT NULL,message_control_id TEXT NOT NULL,
                    attempted_sha256 TEXT NOT NULL,attempted_binding_sha256 TEXT NOT NULL,
                    outcome TEXT NOT NULL,existing_message_id TEXT,created_at TEXT NOT NULL
                ) STRICT;
                CREATE TABLE clinical_results(
                    result_id TEXT PRIMARY KEY,message_id TEXT NOT NULL UNIQUE REFERENCES clinical_messages(message_id),
                    supersedes_result_id TEXT REFERENCES clinical_results(result_id),
                    result_version INTEGER NOT NULL CHECK(result_version>=1),
                    source_report_id TEXT NOT NULL,
                    source_subject_token TEXT NOT NULL,source_order_token TEXT NOT NULL,assay_code TEXT NOT NULL,
                    source_status TEXT NOT NULL,canonical_json TEXT NOT NULL,canonical_sha256 TEXT NOT NULL UNIQUE,
                    profile_sha256 TEXT NOT NULL,mapping_sha256 TEXT NOT NULL,binding_json TEXT NOT NULL,
                    binding_sha256 TEXT NOT NULL,created_at TEXT NOT NULL
                ) STRICT;
                CREATE TABLE clinical_supersession_claims(
                    predecessor_result_id TEXT PRIMARY KEY REFERENCES clinical_results(result_id),
                    successor_result_id TEXT NOT NULL UNIQUE REFERENCES clinical_results(result_id),
                    created_at TEXT NOT NULL
                ) STRICT;
                CREATE TABLE clinical_observations(
                    result_id TEXT NOT NULL REFERENCES clinical_results(result_id),ordinal INTEGER NOT NULL,
                    value_type TEXT NOT NULL,local_code TEXT NOT NULL,local_display TEXT NOT NULL,
                    local_system TEXT NOT NULL,raw_value_lexeme TEXT NOT NULL,unit_code TEXT NOT NULL,
                    unit_display TEXT NOT NULL,source_status TEXT NOT NULL,note_text TEXT NOT NULL,
                    PRIMARY KEY(result_id,ordinal)
                ) STRICT;
                CREATE TABLE clinical_policy_events(
                    event_id TEXT PRIMARY KEY,result_id TEXT NOT NULL REFERENCES clinical_results(result_id),
                    gate_code TEXT NOT NULL,outcome TEXT NOT NULL CHECK(outcome IN ('PASS','FAIL','NOT_EVALUATED')),
                    reason_code TEXT NOT NULL,evidence_sha256 TEXT NOT NULL,policy_sha256 TEXT NOT NULL,
                    evaluated_at TEXT NOT NULL,UNIQUE(result_id,gate_code)
                ) STRICT;
                CREATE TABLE clinical_transition_events(
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,result_id TEXT NOT NULL REFERENCES clinical_results(result_id),
                    from_state TEXT NOT NULL,to_state TEXT NOT NULL,actor_kind TEXT NOT NULL,
                    actor_identifier TEXT NOT NULL,reason_code TEXT NOT NULL,created_at TEXT NOT NULL
                ) STRICT;
                CREATE TABLE clinical_review_attestations(
                    review_id TEXT PRIMARY KEY,result_id TEXT NOT NULL UNIQUE REFERENCES clinical_results(result_id),
                    reviewer_id TEXT NOT NULL REFERENCES clinical_reviewers(reviewer_id),recipient_id TEXT NOT NULL,
                    nonce TEXT NOT NULL UNIQUE,envelope_json TEXT NOT NULL,envelope_sha256 TEXT NOT NULL UNIQUE,
                    signature_b64 TEXT NOT NULL,applied_at TEXT NOT NULL
                ) STRICT;
                CREATE TABLE clinical_exports(
                    export_id TEXT PRIMARY KEY,result_id TEXT NOT NULL UNIQUE REFERENCES clinical_results(result_id),
                    review_id TEXT NOT NULL REFERENCES clinical_review_attestations(review_id),format TEXT NOT NULL,
                    artifact_json TEXT NOT NULL,artifact_sha256 TEXT NOT NULL UNIQUE,
                    authorization_manifest_json TEXT NOT NULL,
                    authorization_manifest_sha256 TEXT NOT NULL UNIQUE,created_at TEXT NOT NULL
                ) STRICT;
                CREATE TABLE clinical_audit_events(
                    sequence INTEGER PRIMARY KEY,timestamp TEXT NOT NULL,event_type TEXT NOT NULL,result_id TEXT,
                    actor TEXT NOT NULL,payload_json TEXT NOT NULL,previous_hash TEXT NOT NULL,event_hash TEXT NOT NULL UNIQUE
                ) STRICT;
                CREATE TRIGGER clinical_metadata_no_delete BEFORE DELETE ON clinical_metadata
                BEGIN SELECT RAISE(ABORT,'clinical metadata may not be deleted'); END;
                CREATE TRIGGER clinical_trust_no_unseal BEFORE UPDATE OF value ON clinical_metadata
                WHEN OLD.key='review_trust_sealed' AND OLD.value='1' AND NEW.value<>'1'
                BEGIN SELECT RAISE(ABORT,'clinical reviewer trust cannot be unsealed'); END;
                CREATE TRIGGER clinical_metadata_restricted_update
                BEFORE UPDATE ON clinical_metadata
                WHEN OLD.key NOT IN ('audit_tip','review_trust_sealed')
                BEGIN SELECT RAISE(ABORT,'immutable clinical metadata may not be updated'); END;
                CREATE TRIGGER clinical_reviewer_no_insert_after_seal
                BEFORE INSERT ON clinical_reviewers
                WHEN (SELECT value FROM clinical_metadata WHERE key='review_trust_sealed') <> '0'
                BEGIN SELECT RAISE(ABORT,'clinical reviewer trust is sealed'); END;
                """
            )
            for table in CLINICAL_APPEND_ONLY_TABLES:
                connection.execute(clinical_append_only_trigger_sql(table, "no_update"))
                connection.execute(clinical_append_only_trigger_sql(table, "no_delete"))
            policy_sha256 = hashlib.sha256(
                canonical_json({"gates": list(CLINICAL_REQUIRED_GATES), "version": CLINICAL_POLICY_VERSION})
            ).hexdigest()
            schema_fingerprint = clinical_schema_fingerprint(connection)
            with immediate_transaction(connection):
                connection.executemany(
                    "INSERT INTO clinical_metadata(key,value) VALUES (?,?)",
                    (
                        ("schema_version", CLINICAL_SCHEMA_VERSION),
                        ("initialization_complete", "1"),
                        ("installation_id", str(uuid.uuid4())),
                        ("audit_tip", ZERO_HASH),
                        ("review_trust_sealed", "0"),
                        ("policy_sha256", policy_sha256),
                        ("schema_fingerprint", schema_fingerprint),
                    ),
                )
                clinical_append_audit(
                    connection,
                    "CLINICAL_STATE_INITIALIZED",
                    None,
                    "local-operator",
                    {
                        "clinical_mode": "UNCONFIGURED",
                        "device_control": False,
                        "direct_device_transport": False,
                        "fhir_output": "OFFLINE_ARTIFACT_ONLY",
                        "profile_id": CLINICAL_PROFILE_ID,
                        "schema_fingerprint": schema_fingerprint,
                        "schema_version": CLINICAL_SCHEMA_VERSION,
                    },
                )
            clinical_verify_ledger(connection)
            checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint is None or int(checkpoint[0]) != 0 or int(checkpoint[1]) != int(checkpoint[2]):
                raise IntegrityFailure("clinical initialization WAL checkpoint did not complete")
            connection.close()
            connection = None
            _apply_owner_only_acl(staged)
            try:
                os.link(staged, database)
            except FileExistsError as exc:
                raise ControlError(
                    "CLINICAL_ALREADY_INITIALIZED", "clinical database already exists", EXIT_CONFLICT
                ) from exc
            except OSError as exc:
                raise ControlError(
                    "CLINICAL_STATE_PUBLISH_FAILED",
                    f"unable to atomically publish clinical state: {exc}",
                    EXIT_INPUT,
                ) from exc
        finally:
            if connection is not None:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                connection.close()
            for temporary in (staged, *clinical_database_sidecars(staged)):
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
    return {
        "acl": acl,
        "clinical_database": str(database),
        "clinical_use_authorized": False,
        "device_control": False,
        "direct_device_transport": False,
        "export_authorization_evidence": "DETACHED_ED25519_MANIFEST_REQUIRES_TRUSTED_KEY",
        "fhir_output": "OFFLINE_ARTIFACT_ONLY",
        "ok": True,
        "operation": "clinical-init",
        "profile_id": CLINICAL_PROFILE_ID,
    }


def validate_clinical_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not CLINICAL_ID_PATTERN.fullmatch(value):
        raise ControlError("INVALID_CLINICAL_ID", f"{field} is invalid", EXIT_INPUT)
    return value


def validate_small_text(value: Any, field: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ControlError("INVALID_CLINICAL_FIELD", f"{field} is missing or too long", EXIT_INPUT)
    if any(ord(character) < 32 for character in value):
        raise ControlError("INVALID_CLINICAL_FIELD", f"{field} contains control characters", EXIT_INPUT)
    return value


FHIR_REFERENCE_PATTERN = re.compile(
    r"^(Patient|ServiceRequest|Specimen)/[A-Za-z0-9](?:[A-Za-z0-9.-]{0,63})$"
)


def validate_fhir_reference(value: Any, resource_type: str, field: str) -> str:
    reference = validate_small_text(value, field, 80)
    match = FHIR_REFERENCE_PATTERN.fullmatch(reference)
    if match is None or match.group(1) != resource_type:
        raise ControlError(
            "INVALID_FHIR_REFERENCE",
            f"{field} must be exactly {resource_type}/ plus a valid FHIR id",
            EXIT_INPUT,
        )
    return reference


def load_assay_map(path: Path, mode: str = "MOCK") -> dict[str, dict[str, str]]:
    value = load_json_file(path)
    if not isinstance(value, dict) or not value:
        raise ControlError("INVALID_ASSAY_MAP", "assay map must be a nonempty object", EXIT_INPUT)
    result: dict[str, dict[str, str]] = {}
    for source_code, mapped in value.items():
        code = validate_small_text(source_code, "assay source code", 128)
        mapped = require_exact_keys(mapped, {"display", "local_system"}, "assay mapping")
        display = validate_small_text(mapped["display"], "assay display", 256)
        local_system = validate_small_text(mapped["local_system"], "assay local system", 512)
        if not re.match(r"^(?:https?://|urn:)", local_system):
            raise ControlError("INVALID_ASSAY_MAP", "local_system must be an http(s) or urn URI", EXIT_INPUT)
        if mode == "MOCK":
            if (
                "SYNTH-" not in code.upper()
                or "SYNTH" not in display.upper()
                or not local_system.startswith("urn:synthetic:")
            ):
                raise ControlError(
                    "NON_SYNTHETIC_ASSAY_MAP_DENIED",
                    "MOCK assay mappings must be explicitly synthetic",
                    EXIT_DENIED,
                )
        result[code] = {"display": display, "local_system": local_system}
    if mode == "MOCK" and result != CLINICAL_SYNTHETIC_ASSAY_MAP:
        raise ControlError(
            "UNPINNED_SYNTHETIC_ASSAY_MAP_DENIED",
            "MOCK assay map must equal the built-in synthetic assay map exactly",
            EXIT_DENIED,
        )
    return result


def clinical_add_source(
    paths: StatePaths,
    source_id: str,
    mode: str,
    profile_id: str,
    sender_application: str,
    sender_facility: str,
    assay_map_path: Path,
) -> dict[str, Any]:
    source_id = validate_clinical_id(source_id, "source_id")
    if mode not in CLINICAL_MODES:
        raise ControlError(
            "INVALID_CLINICAL_MODE",
            "only MOCK and deidentified LIVE_SHADOW modes are supported",
            EXIT_INPUT,
        )
    expected_profile_id = CLINICAL_PROFILE_IDS[mode]
    if profile_id != expected_profile_id:
        raise ControlError(
            "UNSUPPORTED_SOURCE_PROFILE",
            f"profile for {mode} must be exactly {expected_profile_id}",
            EXIT_DENIED,
        )
    sender_application = validate_small_text(sender_application, "sender_application", 128)
    sender_facility = validate_small_text(sender_facility, "sender_facility", 128)
    if mode == "MOCK":
        if source_id != CLINICAL_SYNTHETIC_SOURCE_ID:
            raise ControlError(
                "UNSUPPORTED_SYNTHETIC_SOURCE",
                f"MOCK source_id must be exactly {CLINICAL_SYNTHETIC_SOURCE_ID}",
                EXIT_DENIED,
            )
        if (
            sender_application != CLINICAL_SYNTHETIC_SENDER_APPLICATION
            or sender_facility != CLINICAL_SYNTHETIC_SENDER_FACILITY
        ):
            raise ControlError(
                "NON_SYNTHETIC_SOURCE_DENIED",
                "MOCK sender application and facility must match the fixed synthetic profile",
                EXIT_DENIED,
            )
    assay_map = load_assay_map(assay_map_path, mode)
    source_assertion = CLINICAL_LIVE_SOURCE_ASSERTION
    if mode == "MOCK":
        source_assertion = "synthetic_external_file_not_device_authenticated"
    config = {
        "assay_map": assay_map,
        "device_control": False,
        "direct_device_transport": False,
        "mode": mode,
        "real_phi_authorized": False,
        "profile_id": profile_id,
        "sender_application": sender_application,
        "sender_facility": sender_facility,
        "source_assertion": source_assertion,
        "source_id": source_id,
        "site_validated": False,
    }
    config_json = canonical_json(config).decode("utf-8")
    config_sha256 = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    with clinical_connect(paths) as connection:
        with immediate_transaction(connection):
            clinical_verify_ledger(connection)
            connection.execute(
                """
                INSERT INTO clinical_sources(
                    source_id,mode,profile_id,sender_application,sender_facility,assay_map_json,
                    config_json,config_sha256,active,created_at
                ) VALUES (?,?,?,?,?,?,?,?,1,?)
                """,
                (
                    source_id,
                    mode,
                    profile_id,
                    sender_application,
                    sender_facility,
                    canonical_json(assay_map).decode("utf-8"),
                    config_json,
                    config_sha256,
                    format_time(utc_now()),
                ),
            )
            clinical_append_audit(
                connection,
                "CLINICAL_SOURCE_ADDED",
                None,
                "local-operator",
                {
                    "config_sha256": config_sha256,
                    "mode": mode,
                    "profile_id": profile_id,
                    "source_id": source_id,
                },
            )
    return {
        "config_sha256": config_sha256,
        "device_control": False,
        "direct_device_transport": False,
        "mode": mode,
        "ok": True,
        "operation": "clinical-source-add",
        "profile_id": profile_id,
        "real_phi_authorized": False,
        "site_validated": False,
        "source_id": source_id,
    }


def clinical_add_reviewer_raw(
    paths: StatePaths,
    reviewer_id: str,
    public_key: Ed25519PublicKey,
    *,
    actor: str = "local-operator",
) -> dict[str, Any]:
    require_cryptography()
    reviewer_id = validate_operator(reviewer_id)
    fingerprint = public_fingerprint(public_key)
    encoded = b64url_encode(public_key_raw(public_key))
    with clinical_connect(paths) as connection:
        with immediate_transaction(connection):
            clinical_verify_ledger(connection)
            sealed = connection.execute(
                "SELECT value FROM clinical_metadata WHERE key='review_trust_sealed'"
            ).fetchone()
            if sealed is None or str(sealed["value"]) != "0":
                raise ControlError("CLINICAL_TRUST_SEALED", "reviewer trust is sealed", EXIT_DENIED)
            connection.execute(
                "INSERT INTO clinical_reviewers VALUES (?,?,?,?,?)",
                (reviewer_id, encoded, fingerprint, 1, format_time(utc_now())),
            )
            clinical_append_audit(
                connection,
                "CLINICAL_REVIEWER_ADDED",
                None,
                actor,
                {"fingerprint": fingerprint, "reviewer_id": reviewer_id},
            )
    return {
        "fingerprint": fingerprint,
        "ok": True,
        "operation": "clinical-reviewer-add",
        "reviewer_id": reviewer_id,
    }


def clinical_add_reviewer(paths: StatePaths, reviewer_id: str, public_path: Path) -> dict[str, Any]:
    return clinical_add_reviewer_raw(paths, reviewer_id, load_public_key(public_path))


def clinical_seal_reviewer_trust(paths: StatePaths) -> dict[str, Any]:
    idempotent = False
    with clinical_connect(paths) as connection:
        with immediate_transaction(connection):
            clinical_verify_ledger(connection)
            count = int(connection.execute(
                "SELECT COUNT(*) FROM clinical_reviewers WHERE enabled=1"
            ).fetchone()[0])
            if count < 1:
                raise ControlError("CLINICAL_REVIEWER_REQUIRED", "at least one reviewer is required", EXIT_CONFLICT)
            sealed = connection.execute(
                "SELECT value FROM clinical_metadata WHERE key='review_trust_sealed'"
            ).fetchone()
            if sealed is None or str(sealed["value"]) not in {"0", "1"}:
                raise IntegrityFailure("clinical reviewer trust-seal state is invalid")
            if str(sealed["value"]) == "1":
                idempotent = True
            else:
                changed = connection.execute(
                    "UPDATE clinical_metadata SET value='1' "
                    "WHERE key='review_trust_sealed' AND value='0'"
                )
                if changed.rowcount != 1:
                    raise IntegrityFailure("clinical reviewer trust seal did not change exactly once")
                clinical_append_audit(
                    connection,
                    "CLINICAL_REVIEW_TRUST_SEALED",
                    None,
                    "local-operator",
                    {"enabled_reviewers": count},
                )
    return {
        "enabled_reviewers": count,
        "idempotent": idempotent,
        "ok": True,
        "operation": "clinical-trust-seal",
        "review_trust_sealed": True,
    }


def hl7_field(segment: str, number: int, separator: str) -> str:
    parts = segment.split(separator)
    if parts[0] == "MSH":
        if number == 1:
            return separator
        index = number - 1
    else:
        index = number
    return parts[index] if index < len(parts) else ""


def hl7_component(value: str, number: int) -> str:
    parts = value.split("^")
    index = number - 1
    return parts[index] if index < len(parts) else ""


def parse_roche_liat_hl7(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > CLINICAL_MAX_HL7_BYTES:
        raise ControlError("INVALID_HL7_SIZE", "HL7 input is empty or exceeds 1 MiB", EXIT_INPUT)
    framed = raw.startswith(b"\x0b")
    if framed:
        if not raw.endswith(b"\x1c\r"):
            raise ControlError("INVALID_MLLP_FRAME", "MLLP input is not terminated by FS CR", EXIT_INPUT)
        payload = raw[1:-2]
    else:
        payload = raw
    if any((byte < 32 and byte != 13) or byte == 127 for byte in payload):
        raise ControlError(
            "INVALID_HL7_CONTROL",
            "HL7 payload contains a forbidden control byte",
            EXIT_INPUT,
        )
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ControlError("INVALID_HL7_UTF8", "HL7 input is not valid UTF-8", EXIT_INPUT) from exc
    segments = [segment for segment in text.split("\r") if segment]
    if not segments or len(segments) > CLINICAL_MAX_SEGMENTS:
        raise ControlError("INVALID_HL7_SEGMENTS", "HL7 segment count is invalid", EXIT_INPUT)
    if any(len(segment) > 16384 for segment in segments):
        raise ControlError("INVALID_HL7_SEGMENT", "an HL7 segment exceeds 16384 characters", EXIT_INPUT)
    if len(segments) < 6:
        raise ControlError("INVALID_HL7_STRUCTURE", "HL7 result has too few segments", EXIT_INPUT)
    msh = segments[0]
    if len(msh) < 8 or not msh.startswith("MSH"):
        raise ControlError("INVALID_HL7_MSH", "first segment must be MSH", EXIT_INPUT)
    separator = msh[3]
    if separator != "|" or hl7_field(msh, 2, separator) != "^~\\&":
        raise ControlError("INVALID_HL7_DELIMITERS", "unsupported HL7 delimiters", EXIT_INPUT)
    segment_types = [segment.split(separator, 1)[0] for segment in segments]
    if segment_types.count("MSH") != 1 or segment_types[:5] != ["MSH", "PID", "ORC", "OBR", "NTE"]:
        raise ControlError(
            "INVALID_HL7_STRUCTURE",
            "Roche result structure must begin MSH, PID, ORC, OBR, NTE",
            EXIT_INPUT,
        )
    if any(kind not in {"OBX", "NTE"} for kind in segment_types[5:]):
        raise ControlError(
            "INVALID_HL7_STRUCTURE",
            "only OBX and observation NTE segments may follow the result NTE",
            EXIT_INPUT,
        )
    for index in range(5, len(segment_types)):
        if segment_types[index] == "NTE" and segment_types[index - 1] != "OBX":
            raise ControlError(
                "INVALID_HL7_STRUCTURE",
                "an observation NTE must immediately follow an OBX",
                EXIT_INPUT,
            )
    message_type = hl7_field(msh, 9, separator)
    message_components = message_type.split("^")
    control_id = validate_small_text(hl7_field(msh, 10, separator), "MSH-10", 199)
    sender_application = validate_small_text(hl7_field(msh, 3, separator), "MSH-3", 128)
    sender_facility = validate_small_text(hl7_field(msh, 4, separator), "MSH-4", 128)
    hl7_version = validate_small_text(hl7_field(msh, 12, separator), "MSH-12", 32)
    character_set = hl7_field(msh, 18, separator) or "ASCII"
    pid_segments = [segment for segment in segments if segment.startswith("PID" + separator)]
    orc_segments = [segment for segment in segments if segment.startswith("ORC" + separator)]
    obr_segments = [segment for segment in segments if segment.startswith("OBR" + separator)]
    obx_segments = [segment for segment in segments if segment.startswith("OBX" + separator)]
    nte_segments = [segment for segment in segments if segment.startswith("NTE" + separator)]
    if (
        len(pid_segments) != 1
        or len(orc_segments) != 1
        or len(obr_segments) != 1
        or not obx_segments
    ):
        raise ControlError(
            "INVALID_HL7_STRUCTURE",
            "exactly one PID/ORC/OBR and at least one OBX are required",
            EXIT_INPUT,
        )
    if len(obx_segments) > CLINICAL_MAX_OBSERVATIONS:
        raise ControlError("TOO_MANY_OBSERVATIONS", "HL7 input exceeds 256 OBX segments", EXIT_INPUT)
    pid = pid_segments[0]
    orc = orc_segments[0]
    obr = obr_segments[0]
    source_subject_token = validate_small_text(hl7_field(pid, 3, separator), "PID-3", 256)
    source_order_token = hl7_field(obr, 2, separator) or hl7_field(obr, 3, separator)
    source_order_token = validate_small_text(source_order_token, "OBR-2/3", 256)
    source_report_id = hl7_field(obr, 3, separator) or control_id
    source_report_id = validate_small_text(source_report_id, "source report id", 256)
    assay_code = validate_small_text(hl7_component(hl7_field(obr, 4, separator), 1), "OBR-4", 128)
    source_status = validate_small_text(hl7_field(obr, 25, separator), "OBR-25", 32)
    observations: list[dict[str, str]] = []
    for index, segment in enumerate(obx_segments, 1):
        identifier = hl7_field(segment, 3, separator)
        units = hl7_field(segment, 6, separator)
        observations.append(
            {
                "local_code": validate_small_text(hl7_component(identifier, 1), "OBX-3 code", 128),
                "local_display": hl7_component(identifier, 2),
                "local_system": hl7_component(identifier, 3),
                "ordinal": str(index),
                "raw_value_lexeme": hl7_field(segment, 5, separator),
                "source_status": hl7_field(segment, 11, separator),
                "unit_code": hl7_component(units, 1),
                "unit_display": hl7_component(units, 2),
                "value_type": validate_small_text(hl7_field(segment, 2, separator), "OBX-2", 8),
            }
        )
    total_note_characters = sum(len(hl7_field(segment, 3, separator)) for segment in nte_segments)
    if total_note_characters > 16384:
        raise ControlError("HL7_NOTES_TOO_LARGE", "aggregate NTE content exceeds 16384 characters", EXIT_INPUT)
    return {
        "assay_code": assay_code,
        "character_set": character_set,
        "effective_time": hl7_field(obr, 7, separator),
        "hl7_version": hl7_version,
        "issued_time": hl7_field(obr, 22, separator),
        "manual_approver": hl7_field(obr, 32, separator),
        "message_control_id": control_id,
        "message_type": message_type,
        "message_type_components": message_components,
        "notes": [hl7_field(segment, 3, separator) for segment in nte_segments],
        "observations": observations,
        "order_control": hl7_field(orc, 1, separator),
        "processing_id": hl7_field(msh, 11, separator),
        "sender_application": sender_application,
        "sender_facility": sender_facility,
        "source_order_token": source_order_token,
        "source_report_id": source_report_id,
        "source_status": source_status,
        "source_subject_token": source_subject_token,
        "specimen_action_code": hl7_field(obr, 11, separator),
        "technician": hl7_field(obr, 34, separator),
        "transport_framing": "MLLP" if framed else "UNFRAMED_FILE",
    }


def require_synthetic_mock_message(raw: bytes, parsed: Mapping[str, Any]) -> None:
    """Require the one fixed synthetic fixture profile before storing raw bytes."""
    def deny(reason_code: str) -> None:
        raise ControlError(
            "NON_SYNTHETIC_CONTENT_DENIED",
            f"MOCK input does not match the fixed synthetic fixture ({reason_code})",
            EXIT_DENIED,
        )

    def raw_field(parts: Sequence[str], index: int) -> str:
        return parts[index] if index < len(parts) else ""

    payload = raw[1:-2] if raw.startswith(b"\x0b") else raw
    segments = [segment for segment in payload.decode("utf-8").split("\r") if segment]
    expected_segment_types = ["MSH", "PID", "ORC", "OBR", "NTE", "OBX", "NTE", "OBX", "OBX"]
    if [segment.split("|", 1)[0] for segment in segments] != expected_segment_types:
        deny("SEGMENT_PROFILE_MISMATCH")
    allowed_nonempty_fields = {
        "MSH": {1, 2, 3, 6, 8, 9, 10, 11, 17},
        "PID": {3},
        "ORC": {1},
        "OBR": {1, 2, 3, 4, 7, 11, 22, 25, 32, 34},
        "NTE": {1, 2, 3},
        "OBX": {1, 2, 3, 5, 6, 11},
    }
    for segment in segments:
        fields = segment.split("|")
        allowed = allowed_nonempty_fields.get(fields[0], set())
        if any(value and index not in allowed for index, value in enumerate(fields[1:], 1)):
            deny("UNEXPECTED_POPULATED_FIELD")

    identifier_checks = (
        (str(parsed["message_control_id"]), CLINICAL_SYNTHETIC_MESSAGE_ID_PATTERN),
        (str(parsed["source_subject_token"]), CLINICAL_SYNTHETIC_SUBJECT_ID_PATTERN),
        (str(parsed["source_order_token"]), CLINICAL_SYNTHETIC_ORDER_ID_PATTERN),
        (str(parsed["source_report_id"]), CLINICAL_SYNTHETIC_REPORT_ID_PATTERN),
    )
    if any(pattern.fullmatch(value) is None for value, pattern in identifier_checks):
        deny("IDENTIFIER_PROFILE_MISMATCH")
    if str(parsed["assay_code"]) not in CLINICAL_SYNTHETIC_ASSAY_CODES:
        deny("ASSAY_CODE_PROFILE_MISMATCH")
    if (
        parsed["sender_application"] != CLINICAL_SYNTHETIC_SENDER_APPLICATION
        or parsed["sender_facility"] != CLINICAL_SYNTHETIC_SENDER_FACILITY
    ):
        deny("SENDER_PROFILE_MISMATCH")

    msh_fields = segments[0].split("|")
    expected_msh_fields = {
        1: "^~\\&",
        2: CLINICAL_SYNTHETIC_SENDER_APPLICATION,
        3: CLINICAL_SYNTHETIC_SENDER_FACILITY,
        6: "20260813050000-0400",
        8: "ORU^R30^ORU_R30",
        9: str(parsed["message_control_id"]),
        10: "P",
        11: "2.5",
        17: "UNICODE UTF-8",
    }
    if any(raw_field(msh_fields, index) != value for index, value in expected_msh_fields.items()):
        deny("MSH_PROFILE_MISMATCH")
    if segments[1] != "PID|||" + str(parsed["source_subject_token"]):
        deny("PID_PROFILE_MISMATCH")
    if segments[2] != "ORC|NW":
        deny("ORC_PROFILE_MISMATCH")

    obr_fields = segments[3].split("|")
    expected_obr_fields = {
        1: "1",
        2: str(parsed["source_order_token"]),
        3: str(parsed["source_report_id"]),
        4: (
            str(parsed["assay_code"])
            + "^Synthetic influenza assay^urn:synthetic:assay"
        ),
        7: "20260813045900-0400",
        11: "O",
        22: "20260813050000-0400",
        25: str(parsed["source_status"]),
        32: "SYNTH-APPROVER",
        34: "SYNTH-TECH",
    }
    if any(raw_field(obr_fields, index) != value for index, value in expected_obr_fields.items()):
        deny("OBR_PROFILE_MISMATCH")
    if parsed["source_status"] not in {"F", "P"}:
        deny("REPORT_STATUS_PROFILE_MISMATCH")

    metadata_note = (
        "Run=SYNTH-00001;Device=SYNTH-DEVICE;Version=3.5.0-SYNTH;"
        "Tube=SYNTH-TUBE;TubeExp=2099-01-01;TubeLot=SYNTH-LOT"
    )
    if segments[4] != "NTE|1|L|" + metadata_note:
        deny("RESULT_NOTE_PROFILE_MISMATCH")
    if segments[6] != "NTE|1|L|SYNTHETIC DATA - NOT FOR CLINICAL USE":
        deny("OBSERVATION_NOTE_PROFILE_MISMATCH")

    raw_observations = [segments[5], segments[7], segments[8]]
    if len(parsed["observations"]) != len(CLINICAL_SYNTHETIC_OBSERVATION_PROFILE):
        deny("OBX_COUNT_PROFILE_MISMATCH")
    observation_fields: list[list[str]] = []
    for raw_observation, expected in zip(
        raw_observations, CLINICAL_SYNTHETIC_OBSERVATION_PROFILE, strict=True
    ):
        fields = raw_observation.split("|")
        observation_fields.append(fields)
        if len(fields) != 12:
            deny("OBX_FIELD_COUNT_PROFILE_MISMATCH")
        if raw_field(fields, 1) != expected["set_id"]:
            deny("OBX_SET_ID_PROFILE_MISMATCH")
        if raw_field(fields, 2) != expected["value_type"]:
            deny("OBX_VALUE_TYPE_PROFILE_MISMATCH")
        if raw_field(fields, 3) != expected["identifier_raw"]:
            deny("OBX_IDENTIFIER_PROFILE_MISMATCH")
        if raw_field(fields, 6) != expected["units_raw"]:
            deny("OBX_UNITS_PROFILE_MISMATCH")
        if raw_field(fields, 11) != parsed["source_status"]:
            deny("OBX_STATUS_PROFILE_MISMATCH")

    allowed_interpretations = {
        "Aborted", "Detected", "Indeterminate", "Invalid", "Not detected"
    }
    if raw_field(observation_fields[0], 5) != "0":
        deny("OBX_CONTROL_VALUE_PROFILE_MISMATCH")
    qualitative_value = raw_field(observation_fields[1], 5)
    if qualitative_value not in allowed_interpretations:
        deny("OBX_QUALITATIVE_VALUE_PROFILE_MISMATCH")
    expected_ct = "28.75" if qualitative_value == "Detected" else ""
    if raw_field(observation_fields[2], 5) != expected_ct:
        deny("OBX_CT_VALUE_PROFILE_MISMATCH")


def require_deidentified_live_shadow_message(
    raw: bytes,
    parsed: Mapping[str, Any],
) -> None:
    """Reject likely PHI before raw LIVE_SHADOW bytes can be persisted.

    This is a deliberately narrow admission profile, not a deidentification
    service or HIPAA determination. A site-owned upstream process must replace
    subject, order, report, patient, service-request, and specimen identifiers.
    """

    def deny(reason_code: str) -> None:
        raise ControlError(
            "LIVE_SHADOW_DEIDENTIFICATION_REQUIRED",
            f"LIVE_SHADOW input failed the deidentified admission profile ({reason_code})",
            EXIT_DENIED,
        )

    payload = raw[1:-2] if raw.startswith(b"\x0b") else raw
    segments = [segment for segment in payload.decode("utf-8").split("\r") if segment]
    allowed_nonempty_fields = {
        "MSH": {1, 2, 3, 6, 8, 9, 10, 11, 17},
        "PID": {3},
        "ORC": {1},
        "OBR": {1, 2, 3, 4, 7, 11, 22, 25},
        "NTE": {1, 2, 3},
        "OBX": {1, 2, 3, 5, 6, 11},
    }
    for segment in segments:
        fields = segment.split("|")
        allowed = allowed_nonempty_fields.get(fields[0], set())
        if any(value and index not in allowed for index, value in enumerate(fields[1:], 1)):
            deny("UNEXPECTED_POPULATED_FIELD")

    for field in ("source_subject_token", "source_order_token", "source_report_id"):
        if CLINICAL_DEIDENTIFIED_TOKEN_PATTERN.fullmatch(str(parsed[field])) is None:
            deny(f"{field.upper()}_NOT_PSEUDONYMOUS")

    allowed_note_keys = {"Device", "Run", "Tube", "TubeExp", "TubeLot", "Version"}
    safe_value = re.compile(r"^[A-Za-z0-9_.:/+-]{1,128}$")
    for note in parsed["notes"]:
        text = str(note)
        if not text or text == "DEIDENTIFIED LIVE SHADOW":
            continue
        fields = text.split(";")
        pairs: dict[str, str] = {}
        for item in fields:
            if item.count("=") != 1:
                deny("FREE_TEXT_NOTE_DENIED")
            key, value = item.split("=", 1)
            if key not in allowed_note_keys or safe_value.fullmatch(value) is None:
                deny("UNSAFE_NOTE_METADATA")
            pairs[key] = value
        if not pairs or "Run" not in pairs:
            deny("UNRECOGNIZED_NOTE_METADATA")


def load_clinical_binding(path: Path, mode: str) -> dict[str, Any]:
    binding = require_exact_keys(
        load_json_file(path),
        {
            "deidentified", "order_reference", "patient_reference", "recipient_id",
            "source_order_token", "source_subject_token", "specimen_reference",
            "supersedes_result_id", "synthetic",
        },
        "clinical binding",
    )
    for field in ("recipient_id", "source_order_token", "source_subject_token"):
        binding[field] = validate_small_text(binding[field], field, 256)
    binding["patient_reference"] = validate_fhir_reference(
        binding["patient_reference"], "Patient", "patient_reference"
    )
    binding["order_reference"] = validate_fhir_reference(
        binding["order_reference"], "ServiceRequest", "order_reference"
    )
    binding["specimen_reference"] = validate_fhir_reference(
        binding["specimen_reference"], "Specimen", "specimen_reference"
    )
    if type(binding["synthetic"]) is not bool or type(binding["deidentified"]) is not bool:
        raise ControlError("INVALID_BINDING_FLAGS", "synthetic/deidentified must be booleans", EXIT_INPUT)
    supersedes = binding["supersedes_result_id"]
    if supersedes is not None:
        binding["supersedes_result_id"] = validate_clinical_id(supersedes, "supersedes_result_id")
    if mode == "MOCK":
        if not binding["synthetic"] or not binding["deidentified"]:
            raise ControlError(
                "REAL_PHI_NOT_AUTHORIZED",
                "MOCK ingestion requires synthetic=true and deidentified=true",
                EXIT_DENIED,
            )
        synthetic_patterns = (
            (binding["patient_reference"], re.compile(r"^Patient/SYNTH-PATIENT-[0-9]{3}$")),
            (binding["order_reference"], re.compile(r"^ServiceRequest/SYNTH-ORDER-[0-9]{3}$")),
            (binding["specimen_reference"], re.compile(r"^Specimen/SYNTH-SPECIMEN-[0-9]{3}$")),
            (binding["source_subject_token"], CLINICAL_SYNTHETIC_SUBJECT_ID_PATTERN),
            (binding["source_order_token"], CLINICAL_SYNTHETIC_ORDER_ID_PATTERN),
        )
        if (
            binding["recipient_id"] != "SYNTH-OFFLINE-RECIPIENT"
            or any(pattern.fullmatch(value) is None for value, pattern in synthetic_patterns)
        ):
            raise ControlError(
                "REAL_PHI_NOT_AUTHORIZED",
                "MOCK binding identifiers must match the fixed numeric synthetic fixture grammar",
                EXIT_DENIED,
            )
    elif mode == "LIVE_SHADOW":
        if binding["synthetic"] or not binding["deidentified"]:
            raise ControlError(
                "LIVE_BINDING_FLAG_MISMATCH",
                "LIVE_SHADOW requires synthetic=false and deidentified=true",
                EXIT_DENIED,
            )
        if any(
            pattern.fullmatch(str(binding[field])) is None
            for field, pattern in CLINICAL_DEIDENTIFIED_REFERENCE_PATTERNS.items()
        ):
            raise ControlError(
                "LIVE_SHADOW_REFERENCE_NOT_PSEUDONYMOUS",
                "LIVE_SHADOW FHIR references must use DEID-prefixed pseudonymous ids",
                EXIT_DENIED,
            )
        if any(
            CLINICAL_DEIDENTIFIED_TOKEN_PATTERN.fullmatch(str(binding[field])) is None
            for field in ("source_order_token", "source_subject_token")
        ):
            raise ControlError(
                "LIVE_SHADOW_SOURCE_TOKEN_NOT_PSEUDONYMOUS",
                "LIVE_SHADOW source subject/order tokens must be DEID-prefixed",
                EXIT_DENIED,
            )
    return dict(binding)


def clinical_policy_sha256() -> str:
    return hashlib.sha256(
        canonical_json({"gates": list(CLINICAL_REQUIRED_GATES), "version": CLINICAL_POLICY_VERSION})
    ).hexdigest()


def clinical_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    note_text = observation["note_text"] if "note_text" in observation.keys() else ""
    return {
        "local_code": str(observation["local_code"]),
        "local_display": str(observation["local_display"]),
        "local_system": str(observation["local_system"]),
        "note_text": str(note_text),
        "ordinal": int(observation["ordinal"]),
        "raw_value_lexeme": str(observation["raw_value_lexeme"]),
        "source_status": str(observation["source_status"]),
        "unit_code": str(observation["unit_code"]),
        "unit_display": str(observation["unit_display"]),
        "value_type": str(observation["value_type"]),
    }


def clinical_observation_set_sha256(observations: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(
        (clinical_observation_payload(observation) for observation in observations),
        key=lambda item: item["ordinal"],
    )
    return hashlib.sha256(canonical_json(ordered)).hexdigest()


def clinical_current_state(connection: sqlite3.Connection, result_id: str) -> str:
    row = connection.execute(
        """
        SELECT to_state FROM clinical_transition_events
        WHERE result_id=? ORDER BY sequence DESC LIMIT 1
        """,
        (result_id,),
    ).fetchone()
    return "NONE" if row is None else str(row["to_state"])


def clinical_transition(
    connection: sqlite3.Connection,
    result_id: str,
    to_state: str,
    actor_kind: str,
    actor_identifier: str,
    reason_code: str,
) -> None:
    from_state = clinical_current_state(connection, result_id)
    if not LoopKernelStateTransition.allowed(from_state, to_state):
        raise ControlError(
            "INVALID_CLINICAL_TRANSITION",
            f"clinical transition {from_state}->{to_state} is not permitted",
            EXIT_DENIED,
        )
    connection.execute(
        """
        INSERT INTO clinical_transition_events(
            result_id,from_state,to_state,actor_kind,actor_identifier,reason_code,created_at
        ) VALUES (?,?,?,?,?,?,?)
        """,
        (
            result_id,
            from_state,
            to_state,
            actor_kind,
            actor_identifier,
            reason_code,
            format_time(utc_now()),
        ),
    )
    clinical_append_audit(
        connection,
        "CLINICAL_STATE_TRANSITION",
        result_id,
        actor_identifier,
        {
            "actor_kind": actor_kind,
            "from_state": from_state,
            "reason_code": reason_code,
            "to_state": to_state,
        },
    )


def clinical_verify_ledger(connection: sqlite3.Connection) -> dict[str, Any]:
    own_transaction = not connection.in_transaction
    if own_transaction:
        connection.execute("BEGIN")
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise IntegrityFailure("SQLite integrity_check failed")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise IntegrityFailure("SQLite foreign_key_check failed")
        schema = connection.execute(
            "SELECT value FROM clinical_metadata WHERE key='schema_version'"
        ).fetchone()
        if schema is None or str(schema["value"]) != CLINICAL_SCHEMA_VERSION:
            raise IntegrityFailure("clinical schema version is missing or unsupported")
        complete = connection.execute(
            "SELECT value FROM clinical_metadata WHERE key='initialization_complete'"
        ).fetchone()
        if complete is None or str(complete["value"]) != "1":
            raise IntegrityFailure("clinical initialization-completion marker is invalid")
        fingerprint = connection.execute(
            "SELECT value FROM clinical_metadata WHERE key='schema_fingerprint'"
        ).fetchone()
        observed_fingerprint = clinical_schema_fingerprint(connection)
        if fingerprint is None or not secrets.compare_digest(
            str(fingerprint["value"]), observed_fingerprint
        ):
            raise IntegrityFailure("clinical schema or trigger fingerprint changed")
        policy = connection.execute(
            "SELECT value FROM clinical_metadata WHERE key='policy_sha256'"
        ).fetchone()
        if policy is None or str(policy["value"]) != clinical_policy_sha256():
            raise IntegrityFailure("clinical policy hash is missing or changed")
        previous_hash = ZERO_HASH
        expected_sequence = 1
        event_count = 0
        audit_bindings: dict[str, list[dict[str, Any]]] = {}
        for row in connection.execute("SELECT * FROM clinical_audit_events ORDER BY sequence"):
            if int(row["sequence"]) != expected_sequence or str(row["previous_hash"]) != previous_hash:
                raise IntegrityFailure("clinical audit sequence or previous hash is invalid")
            payload = parse_json_bytes(str(row["payload_json"]).encode("utf-8"))
            body = clinical_audit_body(
                expected_sequence,
                str(row["timestamp"]),
                str(row["event_type"]),
                row["result_id"],
                str(row["actor"]),
                payload,
            )
            expected_hash = hashlib.sha256(
                bytes.fromhex(previous_hash) + canonical_json(body)
            ).hexdigest()
            if not secrets.compare_digest(expected_hash, str(row["event_hash"])):
                raise IntegrityFailure("clinical audit event hash is invalid")
            audit_bindings.setdefault(str(row["event_type"]), []).append(
                {
                    "actor": str(row["actor"]),
                    "payload": payload,
                    "result_id": row["result_id"],
                }
            )
            previous_hash = expected_hash
            expected_sequence += 1
            event_count += 1
        tip = connection.execute(
            "SELECT value FROM clinical_metadata WHERE key='audit_tip'"
        ).fetchone()
        if tip is None or str(tip["value"]) != previous_hash:
            raise IntegrityFailure("clinical audit tip does not match the verified chain")
        for row in connection.execute("SELECT config_json,config_sha256 FROM clinical_sources"):
            expected = hashlib.sha256(str(row["config_json"]).encode("utf-8")).hexdigest()
            if not secrets.compare_digest(expected, str(row["config_sha256"])):
                raise IntegrityFailure("clinical source configuration hash is invalid")
        for row in connection.execute("SELECT raw_b64,raw_sha256 FROM clinical_messages"):
            raw = b64url_decode(row["raw_b64"])
            expected = hashlib.sha256(raw).hexdigest()
            if not secrets.compare_digest(expected, str(row["raw_sha256"])):
                raise IntegrityFailure("clinical raw-message hash is invalid")
        for row in connection.execute("SELECT * FROM clinical_results"):
            canonical_hash = hashlib.sha256(str(row["canonical_json"]).encode("utf-8")).hexdigest()
            binding_hash = hashlib.sha256(str(row["binding_json"]).encode("utf-8")).hexdigest()
            if not secrets.compare_digest(canonical_hash, str(row["canonical_sha256"])):
                raise IntegrityFailure("clinical canonical-result hash is invalid")
            if not secrets.compare_digest(binding_hash, str(row["binding_sha256"])):
                raise IntegrityFailure("clinical binding hash is invalid")
            canonical = parse_json_bytes(str(row["canonical_json"]).encode("utf-8"))
            canonical_version = canonical.get("result_version")
            stored_version = row["result_version"]
            version_matches = (
                type(canonical_version) is int
                and canonical_version >= 1
                and canonical_version == int(stored_version)
            )
            if not version_matches:
                raise IntegrityFailure("clinical result version differs from its canonical result")
            expected_observation_hash = str(canonical.get("observation_set_sha256", ""))
            observation_rows = connection.execute(
                "SELECT * FROM clinical_observations WHERE result_id=? ORDER BY ordinal",
                (str(row["result_id"]),),
            ).fetchall()
            if len(observation_rows) != int(canonical.get("observation_count", -1)):
                raise IntegrityFailure("clinical observation count differs from canonical result")
            if not secrets.compare_digest(
                clinical_observation_set_sha256(observation_rows), expected_observation_hash
            ):
                raise IntegrityFailure("clinical observations differ from canonical result")
        states: dict[str, str] = {}
        for row in connection.execute("SELECT * FROM clinical_transition_events ORDER BY sequence"):
            result_id = str(row["result_id"])
            expected_from = states.get(result_id, "NONE")
            if str(row["from_state"]) != expected_from:
                raise IntegrityFailure("clinical transition chain has an invalid from_state")
            to_state = str(row["to_state"])
            if not LoopKernelStateTransition.allowed(expected_from, to_state):
                raise IntegrityFailure("clinical transition chain contains a forbidden edge")
            states[result_id] = to_state
        for row in connection.execute(
            """
            SELECT a.*,r.public_key_b64
            FROM clinical_review_attestations a
            JOIN clinical_reviewers r ON r.reviewer_id=a.reviewer_id
            """
        ):
            envelope_json = str(row["envelope_json"])
            expected = hashlib.sha256(envelope_json.encode("utf-8")).hexdigest()
            if not secrets.compare_digest(expected, str(row["envelope_sha256"])):
                raise IntegrityFailure("clinical review-envelope hash is invalid")
            envelope = validate_clinical_review_envelope(
                parse_json_bytes(envelope_json.encode("utf-8")), check_time=False
            )
            if (
                str(envelope["review_id"]) != str(row["review_id"])
                or str(envelope["result_id"]) != str(row["result_id"])
                or str(envelope["reviewer_id"]) != str(row["reviewer_id"])
                or str(envelope["authorization"]["signature"]) != str(row["signature_b64"])
                or str(envelope["recipient_id"]) != str(row["recipient_id"])
            ):
                raise IntegrityFailure("clinical review row is not bound to its signed envelope")
            try:
                Ed25519PublicKey.from_public_bytes(
                    b64url_decode(row["public_key_b64"], exact_bytes=32)
                ).verify(
                    b64url_decode(row["signature_b64"], exact_bytes=64),
                    clinical_review_signature_message(envelope),
                )
            except InvalidSignature as exc:
                raise IntegrityFailure("clinical review signature is invalid") from exc
            reviewed_result = connection.execute(
                """
                SELECT cr.*,cm.source_id
                FROM clinical_results cr
                JOIN clinical_messages cm ON cm.message_id=cr.message_id
                WHERE cr.result_id=?
                """,
                (str(row["result_id"]),),
            ).fetchone()
            reviewed_source = connection.execute(
                "SELECT * FROM clinical_sources WHERE source_id=?",
                (str(reviewed_result["source_id"]),),
            ).fetchone() if reviewed_result is not None else None
            reviewed_observations = connection.execute(
                "SELECT * FROM clinical_observations WHERE result_id=? ORDER BY ordinal",
                (str(row["result_id"]),),
            ).fetchall()
            if reviewed_result is None or reviewed_source is None or not reviewed_observations:
                raise IntegrityFailure("clinical review candidate inputs are missing")
            reviewed_candidate = build_offline_fhir_bundle(
                reviewed_result,
                reviewed_observations,
                reviewed_source,
                clinical_predecessor_artifact_sha256(connection, reviewed_result),
            )
            reviewed_candidate_hash = hashlib.sha256(
                pretty_json_bytes(reviewed_candidate)
            ).hexdigest()
            installation = connection.execute(
                "SELECT value FROM clinical_metadata WHERE key='installation_id'"
            ).fetchone()
            if (
                installation is None
                or str(envelope["installation_id"]) != str(installation["value"])
                or str(envelope["candidate_artifact_sha256"]) != reviewed_candidate_hash
            ):
                raise IntegrityFailure("clinical review does not bind the current exact candidate")
        required_gate_set = set(CLINICAL_REQUIRED_GATES)
        for row in connection.execute("SELECT result_id FROM clinical_results"):
            result_id = str(row["result_id"])
            gate_rows = connection.execute(
                "SELECT gate_code,outcome,policy_sha256 FROM clinical_policy_events WHERE result_id=?",
                (result_id,),
            ).fetchall()
            if {str(gate["gate_code"]) for gate in gate_rows} != required_gate_set:
                raise IntegrityFailure("clinical result does not have exactly the required policy gates")
            if any(str(gate["policy_sha256"]) != clinical_policy_sha256() for gate in gate_rows):
                raise IntegrityFailure("clinical policy event is bound to the wrong policy version")
        for row in connection.execute(
            "SELECT result_id,supersedes_result_id,result_version FROM clinical_results"
        ):
            version = int(row["result_version"])
            predecessor_id = row["supersedes_result_id"]
            if predecessor_id is None:
                if version != 1:
                    raise IntegrityFailure("a root clinical result does not have version 1")
                continue
            lineage_gate = connection.execute(
                "SELECT outcome FROM clinical_policy_events "
                "WHERE result_id=? AND gate_code='SUPERSESSION_LINEAGE_VALID'",
                (str(row["result_id"]),),
            ).fetchone()
            predecessor = connection.execute(
                "SELECT result_version FROM clinical_results WHERE result_id=?",
                (str(predecessor_id),),
            ).fetchone()
            if lineage_gate is None:
                raise IntegrityFailure("a correction candidate is missing its lineage gate")
            if predecessor is None or version != int(predecessor["result_version"]) + 1:
                raise IntegrityFailure("a correction candidate has the wrong lineage version")
        for claim in connection.execute(
            """
            SELECT c.predecessor_result_id,c.successor_result_id,r.supersedes_result_id
            FROM clinical_supersession_claims c
            JOIN clinical_results r ON r.result_id=c.successor_result_id
            """
        ):
            predecessor = str(claim["predecessor_result_id"])
            successor = str(claim["successor_result_id"])
            lineage_gate = connection.execute(
                "SELECT outcome FROM clinical_policy_events "
                "WHERE result_id=? AND gate_code='SUPERSESSION_LINEAGE_VALID'",
                (successor,),
            ).fetchone()
            if (
                str(claim["supersedes_result_id"]) != predecessor
                or lineage_gate is None
                or str(lineage_gate["outcome"]) != "PASS"
                or states.get(successor) not in {
                    "AUTHORIZED_FOR_EXPORT", "ARTIFACT_CREATED", "SUPERSEDED"
                }
            ):
                raise IntegrityFailure("clinical supersession claim is not a valid active lineage")
        for row in connection.execute(
            "SELECT result_id,supersedes_result_id FROM clinical_results "
            "WHERE supersedes_result_id IS NOT NULL"
        ):
            successor = str(row["result_id"])
            predecessor = str(row["supersedes_result_id"])
            gate = connection.execute(
                "SELECT outcome FROM clinical_policy_events "
                "WHERE result_id=? AND gate_code='SUPERSESSION_LINEAGE_VALID'",
                (successor,),
            ).fetchone()
            matching_claim = connection.execute(
                "SELECT 1 FROM clinical_supersession_claims "
                "WHERE predecessor_result_id=? AND successor_result_id=?",
                (predecessor, successor),
            ).fetchone()
            claim_required = states.get(successor) in {
                "AUTHORIZED_FOR_EXPORT", "ARTIFACT_CREATED", "SUPERSEDED"
            }
            if (
                gate is None
                or (matching_claim is not None and str(gate["outcome"]) != "PASS")
                or claim_required != (matching_claim is not None)
            ):
                raise IntegrityFailure("clinical supersession gate and active claim disagree")
        for row in connection.execute(
            "SELECT artifact_json,artifact_sha256,authorization_manifest_json,"
            "authorization_manifest_sha256 FROM clinical_exports"
        ):
            expected = hashlib.sha256(str(row["artifact_json"]).encode("utf-8")).hexdigest()
            manifest_expected = hashlib.sha256(
                str(row["authorization_manifest_json"]).encode("utf-8")
            ).hexdigest()
            if not secrets.compare_digest(expected, str(row["artifact_sha256"])):
                raise IntegrityFailure("clinical export artifact hash is invalid")
            if not secrets.compare_digest(
                manifest_expected, str(row["authorization_manifest_sha256"])
            ):
                raise IntegrityFailure("clinical export authorization-manifest hash is invalid")
            try:
                validate_clinical_export_authorization_manifest(
                    parse_json_bytes(
                        str(row["authorization_manifest_json"]).encode("utf-8")
                    )
                )
            except ControlError as exc:
                raise IntegrityFailure(
                    "clinical export authorization manifest is invalid"
                ) from exc
        for row in connection.execute(
            """
            SELECT e.*,a.result_id AS reviewed_result_id,a.envelope_json,
                   a.envelope_sha256,a.applied_at,r.public_key_b64,r.fingerprint,
                   m.value AS installation_id
            FROM clinical_exports e
            JOIN clinical_review_attestations a ON a.review_id=e.review_id
            JOIN clinical_reviewers r ON r.reviewer_id=a.reviewer_id
            JOIN clinical_metadata m ON m.key='installation_id'
            """
        ):
            envelope = parse_json_bytes(str(row["envelope_json"]).encode("utf-8"))
            try:
                manifest = validate_clinical_export_authorization_manifest(
                    parse_json_bytes(
                        str(row["authorization_manifest_json"]).encode("utf-8")
                    )
                )
            except ControlError as exc:
                raise IntegrityFailure(
                    "clinical export authorization manifest is invalid"
                ) from exc
            if (
                str(row["result_id"]) != str(row["reviewed_result_id"])
                or str(envelope["candidate_artifact_sha256"]) != str(row["artifact_sha256"])
                or str(manifest["artifact"]["sha256"]) != str(row["artifact_sha256"])
                or str(manifest["export"]["export_id"]) != str(row["export_id"])
                or str(manifest["export"]["result_id"]) != str(row["result_id"])
                or str(manifest["export"]["created_at"]) != str(row["created_at"])
                or str(manifest["authorization"]["applied_at"]) != str(row["applied_at"])
                or str(manifest["authorization"]["envelope_sha256"])
                != str(row["envelope_sha256"])
                or canonical_json(manifest["authorization"]["envelope"]).decode("utf-8")
                != str(row["envelope_json"])
                or str(manifest["authorization"]["reviewer_public_key_b64"])
                != str(row["public_key_b64"])
                or str(manifest["authorization"]["reviewer_fingerprint"])
                != str(row["fingerprint"])
                or str(manifest["installation_id"]) != str(row["installation_id"])
            ):
                raise IntegrityFailure(
                    "clinical export or authorization manifest is not bound to its signed review"
                )
        expected_triggers = {
            f"{table}_{operation}"
            for table in CLINICAL_APPEND_ONLY_TABLES
            for operation in ("no_update", "no_delete")
        } | {
            "clinical_metadata_restricted_update",
            "clinical_reviewer_no_insert_after_seal",
            "clinical_trust_no_unseal",
        }
        actual_triggers = {
            str(row["name"]): str(row["sql"] or "")
            for row in connection.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger'")
        }
        if not expected_triggers.issubset(actual_triggers):
            raise IntegrityFailure("required clinical append-only triggers are missing")
        for table in CLINICAL_APPEND_ONLY_TABLES:
            for operation in ("no_update", "no_delete"):
                actual = normalize_schema_sql(actual_triggers[f"{table}_{operation}"])
                expected = normalize_schema_sql(
                    clinical_append_only_trigger_sql(table, operation)
                )
                if actual != expected:
                    raise IntegrityFailure("a clinical append-only trigger has unsafe SQL")
        sealed = connection.execute(
            "SELECT value FROM clinical_metadata WHERE key='review_trust_sealed'"
        ).fetchone()
        if sealed is None or str(sealed["value"]) not in {"0", "1"}:
            raise IntegrityFailure("clinical reviewer trust-seal state is invalid")

        def require_audit_semantics(
            event_type: str,
            expected: Sequence[Mapping[str, Any]],
            *,
            bind_actor: bool = True,
        ) -> None:
            def normalized(item: Mapping[str, Any]) -> bytes:
                value = dict(item)
                if not bind_actor:
                    value.pop("actor", None)
                return canonical_json(value)

            actual_values = sorted(
                normalized(item) for item in audit_bindings.get(event_type, [])
            )
            expected_values = sorted(normalized(item) for item in expected)
            if actual_values != expected_values:
                raise IntegrityFailure(
                    f"{event_type} audit events are not one-to-one semantically bound to rows"
                )

        initialization_expected = [
            {
                "actor": "local-operator",
                "payload": {
                    "clinical_mode": "UNCONFIGURED",
                    "device_control": False,
                    "direct_device_transport": False,
                    "fhir_output": "OFFLINE_ARTIFACT_ONLY",
                    "profile_id": CLINICAL_PROFILE_ID,
                    "schema_fingerprint": observed_fingerprint,
                    "schema_version": CLINICAL_SCHEMA_VERSION,
                },
                "result_id": None,
            }
        ]
        require_audit_semantics("CLINICAL_STATE_INITIALIZED", initialization_expected)

        source_expected = [
            {
                "actor": "local-operator",
                "payload": {
                    "config_sha256": str(row["config_sha256"]),
                    "mode": str(row["mode"]),
                    "profile_id": str(row["profile_id"]),
                    "source_id": str(row["source_id"]),
                },
                "result_id": None,
            }
            for row in connection.execute(
                "SELECT source_id,mode,profile_id,config_sha256 FROM clinical_sources"
            )
        ]
        require_audit_semantics("CLINICAL_SOURCE_ADDED", source_expected)

        reviewer_expected = [
            {
                "payload": {
                    "fingerprint": str(row["fingerprint"]),
                    "reviewer_id": str(row["reviewer_id"]),
                },
                "result_id": None,
            }
            for row in connection.execute(
                "SELECT reviewer_id,fingerprint FROM clinical_reviewers"
            )
        ]
        require_audit_semantics(
            "CLINICAL_REVIEWER_ADDED", reviewer_expected, bind_actor=False
        )
        enabled_reviewer_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM clinical_reviewers WHERE enabled=1"
            ).fetchone()[0]
        )
        seal_expected: list[dict[str, Any]] = []
        if str(sealed["value"]) == "1":
            seal_expected.append(
                {
                    "actor": "local-operator",
                    "payload": {"enabled_reviewers": enabled_reviewer_count},
                    "result_id": None,
                }
            )
        require_audit_semantics("CLINICAL_REVIEW_TRUST_SEALED", seal_expected)

        result_expected = [
            {
                "actor": "clinical-ingest",
                "payload": {
                    "binding_sha256": str(row["binding_sha256"]),
                    "canonical_sha256": str(row["canonical_sha256"]),
                    "message_id": str(row["message_id"]),
                    "mode": str(row["mode"]),
                    "raw_sha256": str(row["raw_sha256"]),
                    "source_id": str(row["source_id"]),
                },
                "result_id": str(row["result_id"]),
            }
            for row in connection.execute(
                """
                SELECT r.result_id,r.message_id,r.binding_sha256,r.canonical_sha256,
                       m.raw_sha256,m.source_id,s.mode
                FROM clinical_results r
                JOIN clinical_messages m ON m.message_id=r.message_id
                JOIN clinical_sources s ON s.source_id=m.source_id
                """
            )
        ]
        require_audit_semantics("CLINICAL_RESULT_INGESTED", result_expected)

        attempt_expected: list[dict[str, Any]] = []
        for row in connection.execute("SELECT * FROM clinical_ingest_attempts"):
            existing_result = connection.execute(
                "SELECT result_id FROM clinical_results WHERE message_id=?",
                (row["existing_message_id"],),
            ).fetchone()
            if row["existing_message_id"] is not None and existing_result is None:
                raise IntegrityFailure("clinical ingest attempt names no existing result")
            attempt_expected.append(
                {
                    "actor": "clinical-ingest",
                    "payload": {
                        "attempt_id": str(row["attempt_id"]),
                        "attempted_binding_sha256": str(row["attempted_binding_sha256"]),
                        "attempted_raw_sha256": str(row["attempted_sha256"]),
                        "outcome": str(row["outcome"]),
                        "source_id": str(row["source_id"]),
                    },
                    "result_id": (
                        str(existing_result["result_id"])
                        if existing_result is not None
                        else None
                    ),
                }
            )
        require_audit_semantics("CLINICAL_INGEST_RETRY", attempt_expected)

        transition_expected = [
            {
                "actor": str(row["actor_identifier"]),
                "payload": {
                    "actor_kind": str(row["actor_kind"]),
                    "from_state": str(row["from_state"]),
                    "reason_code": str(row["reason_code"]),
                    "to_state": str(row["to_state"]),
                },
                "result_id": str(row["result_id"]),
            }
            for row in connection.execute("SELECT * FROM clinical_transition_events")
        ]
        require_audit_semantics("CLINICAL_STATE_TRANSITION", transition_expected)

        review_expected: list[dict[str, Any]] = []
        for row in connection.execute("SELECT * FROM clinical_review_attestations"):
            envelope = parse_json_bytes(str(row["envelope_json"]).encode("utf-8"))
            review_expected.append(
                {
                    "actor": str(row["reviewer_id"]),
                    "payload": {
                        "decision": str(envelope["decision"]),
                        "envelope_sha256": str(row["envelope_sha256"]),
                        "recipient_id": str(row["recipient_id"]),
                        "review_id": str(row["review_id"]),
                    },
                    "result_id": str(row["result_id"]),
                }
            )
        require_audit_semantics("CLINICAL_REVIEW_APPLIED", review_expected)

        claim_expected: list[dict[str, Any]] = []
        for row in connection.execute(
            """
            SELECT c.predecessor_result_id,c.successor_result_id,a.reviewer_id
            FROM clinical_supersession_claims c
            LEFT JOIN clinical_review_attestations a ON a.result_id=c.successor_result_id
            """
        ):
            if row["reviewer_id"] is None:
                raise IntegrityFailure("clinical supersession claim has no applied review")
            claim_expected.append(
                {
                    "actor": str(row["reviewer_id"]),
                    "payload": {
                        "predecessor_result_id": str(row["predecessor_result_id"])
                    },
                    "result_id": str(row["successor_result_id"]),
                }
            )
        require_audit_semantics("CLINICAL_SUPERSESSION_CLAIMED", claim_expected)

        export_expected = [
            {
                "actor": str(row["reviewer_id"]),
                "payload": {
                    "artifact_sha256": str(row["artifact_sha256"]),
                    "authorization_manifest_sha256": str(
                        row["authorization_manifest_sha256"]
                    ),
                    "delivery_claimed": False,
                    "export_id": str(row["export_id"]),
                    "format": str(row["format"]),
                },
                "result_id": str(row["result_id"]),
            }
            for row in connection.execute(
                """
                SELECT e.*,a.reviewer_id
                FROM clinical_exports e
                JOIN clinical_review_attestations a ON a.review_id=e.review_id
                """
            )
        ]
        require_audit_semantics("CLINICAL_FHIR_ARTIFACT_CREATED", export_expected)

        allowed_event_types = {
            "CLINICAL_STATE_INITIALIZED",
            "CLINICAL_SOURCE_ADDED",
            "CLINICAL_REVIEWER_ADDED",
            "CLINICAL_REVIEW_TRUST_SEALED",
            "CLINICAL_RESULT_INGESTED",
            "CLINICAL_INGEST_RETRY",
            "CLINICAL_STATE_TRANSITION",
            "CLINICAL_REVIEW_APPLIED",
            "CLINICAL_SUPERSESSION_CLAIMED",
            "CLINICAL_FHIR_ARTIFACT_CREATED",
        }
        if set(audit_bindings) - allowed_event_types:
            raise IntegrityFailure("clinical audit contains an unsupported event type")
        if int(connection.execute("SELECT COUNT(*) FROM clinical_messages").fetchone()[0]) != int(
            connection.execute("SELECT COUNT(*) FROM clinical_results").fetchone()[0]
        ):
            raise IntegrityFailure("clinical message/result cardinality is invalid")
        result = {
            "audit_events_verified": event_count,
            "audit_tip": previous_hash,
            "integrity": "VERIFIED_LOCAL_CLINICAL_HASH_CHAIN",
            "result_versions": len(states),
        }
        if own_transaction:
            connection.execute("COMMIT")
        return result
    except BaseException:
        if own_transaction and connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


class EnrichedContextGenerator:
    """Copies observed/configured facts; it never creates clinical facts."""

    @staticmethod
    def generate(
        source: Mapping[str, Any],
        parsed: Mapping[str, Any],
        binding: Mapping[str, Any],
        raw_sha256: str,
        mapping: Mapping[str, Any],
    ) -> dict[str, Any]:
        source_config = parse_json_bytes(str(source["config_json"]).encode("utf-8"))
        return {
            "assay_mapping": mapping,
            "binding": dict(binding),
            "clinical_inference_performed": False,
            "device_control": False,
            "direct_device_transport": False,
            "profile_id": str(source["profile_id"]),
            "raw_sha256": raw_sha256,
            "source_assertion": str(source_config["source_assertion"]),
            "source_id": str(source["source_id"]),
            "source_result": dict(parsed),
        }


def make_clinical_check(
    gate_code: str,
    passed: bool,
    pass_reason: str,
    fail_reason: str,
    evidence: Any,
) -> ClinicalPolicyCheck:
    return ClinicalPolicyCheck(
        gate_code=gate_code,
        outcome="PASS" if passed else "FAIL",
        reason_code=pass_reason if passed else fail_reason,
        evidence_sha256=hashlib.sha256(canonical_json(evidence)).hexdigest(),
    )


def clinical_ingest(
    paths: StatePaths,
    source_id: str,
    hl7_path: Path,
    binding_path: Path,
) -> dict[str, Any]:
    source_id = validate_clinical_id(source_id, "source_id")
    if not hl7_path.is_file() or _is_reparse_point(hl7_path):
        raise ControlError("UNSAFE_HL7_INPUT", "HL7 input must be a regular non-reparse file", EXIT_INPUT)
    try:
        raw = hl7_path.read_bytes()
    except OSError as exc:
        raise ControlError("HL7_READ_FAILED", f"unable to read HL7 input: {exc}", EXIT_INPUT) from exc
    parsed = parse_roche_liat_hl7(raw)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    idempotent_result: dict[str, Any] | None = None
    conflict = False
    with clinical_connect(paths) as connection:
        with immediate_transaction(connection):
            clinical_verify_ledger(connection)
            source = connection.execute(
                "SELECT * FROM clinical_sources WHERE source_id=? AND active=1", (source_id,)
            ).fetchone()
            if source is None:
                raise ControlError("CLINICAL_SOURCE_NOT_FOUND", "active clinical source not found", EXIT_CONFLICT)
            source_mode = str(source["mode"])
            if source_mode == "MOCK":
                require_synthetic_mock_message(raw, parsed)
            elif source_mode == "LIVE_SHADOW":
                require_deidentified_live_shadow_message(raw, parsed)
            binding = load_clinical_binding(binding_path, str(source["mode"]))
            binding_json = canonical_json(binding).decode("utf-8")
            binding_sha256 = hashlib.sha256(binding_json.encode("utf-8")).hexdigest()
            existing = connection.execute(
                "SELECT * FROM clinical_messages WHERE source_id=? AND message_control_id=?",
                (source_id, parsed["message_control_id"]),
            ).fetchone()
            if existing is not None:
                existing_result = connection.execute(
                    "SELECT result_id,binding_sha256 FROM clinical_results WHERE message_id=?",
                    (str(existing["message_id"]),),
                ).fetchone()
                same = bool(
                    existing_result is not None
                    and secrets.compare_digest(str(existing["raw_sha256"]), raw_sha256)
                    and secrets.compare_digest(
                        str(existing_result["binding_sha256"]), binding_sha256
                    )
                )
                attempt_id = str(uuid.uuid4())
                outcome = "IDEMPOTENT_RETRY" if same else "CONTROL_ID_CONTENT_CONFLICT"
                connection.execute(
                    "INSERT INTO clinical_ingest_attempts VALUES (?,?,?,?,?,?,?,?)",
                    (
                        attempt_id,
                        source_id,
                        parsed["message_control_id"],
                        raw_sha256,
                        binding_sha256,
                        outcome,
                        str(existing["message_id"]),
                        format_time(utc_now()),
                    ),
                )
                result_id = str(existing_result["result_id"]) if existing_result else None
                clinical_append_audit(
                    connection,
                    "CLINICAL_INGEST_RETRY",
                    result_id,
                    "clinical-ingest",
                    {
                        "attempt_id": attempt_id,
                        "attempted_raw_sha256": raw_sha256,
                        "attempted_binding_sha256": binding_sha256,
                        "outcome": outcome,
                        "source_id": source_id,
                    },
                )
                if same and result_id is not None:
                    idempotent_result = {
                        "idempotent": True,
                        "message_id": str(existing["message_id"]),
                        "ok": True,
                        "operation": "clinical-ingest",
                        "raw_sha256": raw_sha256,
                        "result_id": result_id,
                        "state": clinical_current_state(connection, result_id),
                    }
                else:
                    conflict = True
            else:
                assay_map = parse_json_bytes(str(source["assay_map_json"]).encode("utf-8"))
                mapped = assay_map.get(parsed["assay_code"]) if isinstance(assay_map, dict) else None
                if not isinstance(mapped, dict):
                    mapped = {"display": "UNMAPPED", "local_system": "urn:unmapped"}
                mapping_sha256 = hashlib.sha256(
                    str(source["assay_map_json"]).encode("utf-8")
                ).hexdigest()
                profile_sha256 = hashlib.sha256(str(source["profile_id"]).encode("utf-8")).hexdigest()
                consistency = CrossStepConsistency.evaluate(parsed, binding)
                supersession_valid = False
                supersedes = binding["supersedes_result_id"]
                prior: sqlite3.Row | None = None
                if supersedes is not None:
                    existing_claim = connection.execute(
                        "SELECT successor_result_id FROM clinical_supersession_claims "
                        "WHERE predecessor_result_id=?",
                        (supersedes,),
                    ).fetchone()
                    prior = connection.execute(
                        """
                        SELECT r.*,m.source_id
                        FROM clinical_results r
                        JOIN clinical_messages m ON m.message_id=r.message_id
                        WHERE r.result_id=?
                        """,
                        (supersedes,),
                    ).fetchone()
                    if prior is None:
                        raise ControlError(
                            "SUPERSESSION_PREDECESSOR_NOT_FOUND",
                            "the named predecessor result does not exist",
                            EXIT_CONFLICT,
                        )
                    if prior is not None and existing_claim is None:
                        prior_binding = parse_json_bytes(str(prior["binding_json"]).encode("utf-8"))
                        supersession_valid = bool(
                            parsed["source_status"] == "F"
                            and str(prior["assay_code"]) == parsed["assay_code"]
                            and str(prior["source_id"]) == source_id
                            and str(prior["source_report_id"]) == parsed["source_report_id"]
                            and str(prior["source_subject_token"]) == parsed["source_subject_token"]
                            and str(prior["source_order_token"]) == parsed["source_order_token"]
                            and prior_binding["patient_reference"] == binding["patient_reference"]
                            and prior_binding["order_reference"] == binding["order_reference"]
                            and prior_binding["specimen_reference"] == binding["specimen_reference"]
                            and clinical_current_state(connection, supersedes) == "ARTIFACT_CREATED"
                        )
                else:
                    supersession_valid = parsed["source_status"] == "F"
                result_version = (
                    1 if supersedes is None else int(prior["result_version"]) + 1
                )
                source_config_valid = secrets.compare_digest(
                    hashlib.sha256(str(source["config_json"]).encode("utf-8")).hexdigest(),
                    str(source["config_sha256"]),
                )
                source_mode = str(source["mode"])
                message_components = list(parsed["message_type_components"])
                message_allowed = message_components[:3] == ["ORU", "R30", "ORU_R30"]
                charset_allowed = str(parsed["character_set"]).upper() == "UNICODE UTF-8"
                profile_syntax_valid = (
                    parsed["processing_id"] == "P"
                    and parsed["order_control"] == "NW"
                    and parsed["specimen_action_code"] == "O"
                )
                observation_structure_valid = all(
                    observation["local_code"]
                    and observation["source_status"] == "F"
                    and observation["value_type"] in {"NM", "ST"}
                    for observation in parsed["observations"]
                )
                interpretation_values = [
                    observation["raw_value_lexeme"]
                    for observation in parsed["observations"]
                    if observation["value_type"] == "ST"
                ]
                allowed_interpretations = {
                    "Aborted", "Detected", "Indeterminate", "Invalid", "Not detected"
                }
                observation_schema_valid = bool(
                    observation_structure_valid
                    and len(parsed["observations"]) == 3
                    and [observation["value_type"] for observation in parsed["observations"]]
                    == ["NM", "ST", "NM"]
                    and len(interpretation_values) == 1
                    and interpretation_values[0] in allowed_interpretations
                    and all(
                        re.fullmatch(r"(?:|[+-]?(?:\d+(?:\.\d*)?|\.\d+))", observation["raw_value_lexeme"])
                        for observation in parsed["observations"]
                        if observation["value_type"] == "NM"
                    )
                )
                if source_mode != "MOCK":
                    observation_schema_valid = bool(
                        observation_structure_valid
                        and len(parsed["observations"]) >= 1
                        and all(
                            len(str(observation["raw_value_lexeme"])) <= 4096
                            for observation in parsed["observations"]
                        )
                    )
                expected_profile_id = CLINICAL_PROFILE_IDS[source_mode]
                # The pinned Roche HIF is based on HL7 2.5.1/IHE LAB-32 but
                # specifies the literal MSH-12 value as 2.5.
                expected_hl7_version = "2.5"
                checks = [
                    make_clinical_check(
                        "SOURCE_PROFILE_ID_MATCH", str(source["profile_id"]) == expected_profile_id,
                        "PROFILE_MATCH", "PROFILE_MISMATCH", str(source["profile_id"]),
                    ),
                    make_clinical_check(
                        "SOURCE_SENDER_CONFIG_MATCH",
                        parsed["sender_application"] == str(source["sender_application"])
                        and parsed["sender_facility"] == str(source["sender_facility"]),
                        "SENDER_MATCH", "SENDER_MISMATCH",
                        {"application": parsed["sender_application"], "facility": parsed["sender_facility"]},
                    ),
                    make_clinical_check(
                        "SOURCE_CONFIG_SELF_HASH_VALID", source_config_valid,
                        "CONFIG_HASH_VALID", "CONFIG_HASH_INVALID", str(source["config_sha256"]),
                    ),
                    make_clinical_check(
                        "RAW_HASH_RECOMPUTED", True, "RAW_HASH_VERIFIED", "RAW_HASH_INVALID", raw_sha256,
                    ),
                    make_clinical_check(
                        "MESSAGE_CONTROL_ID_NOT_PREVIOUSLY_SEEN", True, "CONTROL_ID_UNIQUE", "CONTROL_ID_REUSED",
                        parsed["message_control_id"],
                    ),
                    make_clinical_check(
                        "MESSAGE_TYPE_SHAPE_ALLOWED", message_allowed, "MESSAGE_TYPE_ALLOWED", "MESSAGE_TYPE_DENIED",
                        parsed["message_type"],
                    ),
                    make_clinical_check(
                        "HL7_VERSION_FIELD_ALLOWED", parsed["hl7_version"] == expected_hl7_version,
                        "HL7_VERSION_ALLOWED", "HL7_VERSION_DENIED", parsed["hl7_version"],
                    ),
                    make_clinical_check(
                        "CHARSET_FIELD_ALLOWED", charset_allowed, "CHARSET_ALLOWED", "CHARSET_DENIED",
                        parsed["character_set"],
                    ),
                    make_clinical_check(
                        "BOUNDED_MESSAGE_SYNTAX_VALID", profile_syntax_valid,
                        "MESSAGE_SYNTAX_VALID", "MESSAGE_SYNTAX_INVALID",
                        {
                            "order_control": parsed["order_control"],
                            "processing_id": parsed["processing_id"],
                            "segments_bounded": True,
                            "specimen_action_code": parsed["specimen_action_code"],
                            "transport_framing": parsed["transport_framing"],
                        },
                    ),
                    make_clinical_check(
                        "ASSAY_MAP_MATCH", parsed["assay_code"] in assay_map,
                        "ASSAY_EXACTLY_MAPPED", "ASSAY_UNMAPPED", parsed["assay_code"],
                    ),
                    make_clinical_check(
                        "OBSERVATION_SCHEMA_VALID", observation_schema_valid,
                        "OBSERVATIONS_VALID", "OBSERVATIONS_INVALID", len(parsed["observations"]),
                    ),
                    make_clinical_check(
                        "SOURCE_RESULT_STATUS_CAPTURED", bool(parsed["source_status"]),
                        "SOURCE_STATUS_PRESERVED", "SOURCE_STATUS_MISSING", parsed["source_status"],
                    ),
                    make_clinical_check(
                        "FINAL_STATUS_REQUIRED_FOR_EXPORT", parsed["source_status"] == "F",
                        "FINAL_RESULT", "NONFINAL_RESULT", parsed["source_status"],
                    ),
                    make_clinical_check(
                        "SPECIAL_VALUE_LEXEMES_PRESERVED", True,
                        "SPECIAL_STATES_PRESERVED", "SPECIAL_STATE_CHANGED",
                        [observation["raw_value_lexeme"] for observation in parsed["observations"]],
                    ),
                    make_clinical_check(
                        "EMPTY_VALUE_PRESERVED", True,
                        "EMPTY_VALUES_PRESERVED", "EMPTY_VALUE_COERCED",
                        [
                            observation["raw_value_lexeme"]
                            for observation in parsed["observations"]
                            if observation["value_type"] == "NM"
                        ],
                    ),
                    make_clinical_check(
                        "SUBJECT_BINDING_SELF_CONSISTENT",
                        consistency["source_subject_bound"] and consistency["patient_reference_bound"],
                        "SUBJECT_BINDING_SELF_CONSISTENT", "SUBJECT_BINDING_MISMATCH",
                        binding["patient_reference"],
                    ),
                    make_clinical_check(
                        "ORDER_BINDING_SELF_CONSISTENT",
                        consistency["source_order_bound"] and consistency["order_reference_bound"],
                        "ORDER_BINDING_SELF_CONSISTENT", "ORDER_BINDING_MISMATCH",
                        binding["order_reference"],
                    ),
                    make_clinical_check(
                        "SPECIMEN_REFERENCE_PRESENT", consistency["specimen_reference_bound"],
                        "SPECIMEN_REFERENCE_PRESENT", "SPECIMEN_REFERENCE_MISSING",
                        binding["specimen_reference"],
                    ),
                    make_clinical_check(
                        "RECIPIENT_ID_PRESENT", consistency["recipient_bound"],
                        "RECIPIENT_ID_PRESENT", "RECIPIENT_ID_MISSING",
                        binding["recipient_id"],
                    ),
                    make_clinical_check(
                        "LOCAL_PROVENANCE_FIELDS_PRESENT", bool(parsed["message_control_id"] and source_id),
                        "LOCAL_PROVENANCE_FIELDS_PRESENT", "LOCAL_PROVENANCE_FIELDS_MISSING",
                        {"message_control_id": parsed["message_control_id"], "source_id": source_id},
                    ),
                    make_clinical_check(
                        "LOCAL_POLICY_HASH_BOUND", clinical_policy_sha256() == str(connection.execute(
                            "SELECT value FROM clinical_metadata WHERE key='policy_sha256'"
                        ).fetchone()["value"]),
                        "LOCAL_POLICY_HASH_BOUND", "LOCAL_POLICY_HASH_MISMATCH", clinical_policy_sha256(),
                    ),
                    make_clinical_check(
                        "LOCAL_MAPPING_HASH_BOUND", bool(mapping_sha256),
                        "LOCAL_MAPPING_HASH_BOUND", "LOCAL_MAPPING_HASH_MISSING", mapping_sha256,
                    ),
                    make_clinical_check(
                        "SUPERSESSION_LINEAGE_VALID", supersession_valid,
                        "SUPERSESSION_VALID", "SUPERSESSION_INVALID", supersedes or "NONE",
                    ),
                ]
                required_codes = set(CLINICAL_REQUIRED_GATES)
                present_codes = {check.gate_code for check in checks}
                if required_codes != present_codes:
                    raise IntegrityFailure("clinical policy implementation does not match required gates")
                all_pass = all(check.outcome == "PASS" for check in checks)
                result_id = str(uuid.uuid4())
                message_id = str(uuid.uuid4())
                context = EnrichedContextGenerator.generate(source, parsed, binding, raw_sha256, mapped)
                observation_payloads: list[dict[str, Any]] = []
                notes = " | ".join(parsed["notes"])
                for observation in parsed["observations"]:
                    payload = dict(observation)
                    payload["note_text"] = notes if int(observation["ordinal"]) == 1 else ""
                    observation_payloads.append(clinical_observation_payload(payload))
                canonical = {
                    "context": context,
                    "observation_count": len(observation_payloads),
                    "observation_set_sha256": clinical_observation_set_sha256(observation_payloads),
                    "result_id": result_id,
                    "result_version": result_version,
                    "validation_layers": list(EntropyDepthAllocator(3).allocate(("transport", "semantic"))),
                }
                canonical_json_text = canonical_json(canonical).decode("utf-8")
                canonical_sha256 = hashlib.sha256(canonical_json_text.encode("utf-8")).hexdigest()
                received_at = format_time(utc_now())
                connection.execute(
                    "INSERT INTO clinical_messages VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        message_id,
                        source_id,
                        received_at,
                        raw_sha256,
                        b64url_encode(raw),
                        parsed["message_control_id"],
                        parsed["message_type"],
                        parsed["hl7_version"],
                        parsed["character_set"],
                        "ACCEPTED" if all_pass else "QUARANTINED",
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO clinical_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        result_id,
                        message_id,
                        supersedes,
                        result_version,
                        parsed["source_report_id"],
                        parsed["source_subject_token"],
                        parsed["source_order_token"],
                        parsed["assay_code"],
                        parsed["source_status"],
                        canonical_json_text,
                        canonical_sha256,
                        profile_sha256,
                        mapping_sha256,
                        binding_json,
                        binding_sha256,
                        received_at,
                    ),
                )
                for observation in observation_payloads:
                    connection.execute(
                        "INSERT INTO clinical_observations VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            result_id,
                            int(observation["ordinal"]),
                            observation["value_type"],
                            observation["local_code"],
                            observation["local_display"],
                            observation["local_system"],
                            observation["raw_value_lexeme"],
                            observation["unit_code"],
                            observation["unit_display"],
                            observation["source_status"],
                            observation["note_text"],
                        ),
                    )
                policy_sha256 = clinical_policy_sha256()
                for check in checks:
                    connection.execute(
                        "INSERT INTO clinical_policy_events VALUES (?,?,?,?,?,?,?,?)",
                        (
                            str(uuid.uuid4()),
                            result_id,
                            check.gate_code,
                            check.outcome,
                            check.reason_code,
                            check.evidence_sha256,
                            policy_sha256,
                            format_time(utc_now()),
                        ),
                    )
                clinical_append_audit(
                    connection,
                    "CLINICAL_RESULT_INGESTED",
                    result_id,
                    "clinical-ingest",
                    {
                        "binding_sha256": binding_sha256,
                        "canonical_sha256": canonical_sha256,
                        "message_id": message_id,
                        "mode": str(source["mode"]),
                        "raw_sha256": raw_sha256,
                        "source_id": source_id,
                    },
                )
                clinical_transition(connection, result_id, "RECEIVED", "SERVICE", "clinical-ingest", "RAW_DURABLY_STORED")
                clinical_transition(connection, result_id, "PARSED", "SERVICE", "clinical-parser", "PROFILE_PARSED")
                if not all_pass:
                    failed = sorted(check.gate_code for check in checks if check.outcome != "PASS")
                    clinical_transition(
                        connection,
                        result_id,
                        "QUARANTINED",
                        "SERVICE",
                        "clinical-policy",
                        "POLICY_FAILED:" + ",".join(failed),
                    )
                else:
                    clinical_transition(connection, result_id, "VALIDATED", "SERVICE", "clinical-policy", "ALL_GATES_PASS")
                    clinical_transition(
                        connection,
                        result_id,
                        "PENDING_REVIEW",
                        "SERVICE",
                        "clinical-policy",
                        "SYNTHETIC_REVIEWER_KEY_AUTHORIZATION_REQUIRED",
                    )
                idempotent_result = {
                    "failed_gates": sorted(check.gate_code for check in checks if check.outcome != "PASS"),
                    "idempotent": False,
                    "message_id": message_id,
                    "mode": str(source["mode"]),
                    "ok": True,
                    "operation": "clinical-ingest",
                    "raw_sha256": raw_sha256,
                    "result_id": result_id,
                    "state": clinical_current_state(connection, result_id),
                    "truth_boundary": "synthetic_external_file_not_device_authenticated",
                }
    if conflict:
        raise ControlError(
            "MESSAGE_CONTROL_CONFLICT",
            "the same source/message-control ID was observed with different raw bytes or binding",
            EXIT_DENIED,
        )
    if idempotent_result is None:
        raise IntegrityFailure("clinical ingestion ended without a durable result")
    return idempotent_result


def clinical_status(paths: StatePaths, result_id: str) -> dict[str, Any]:
    result_id = validate_clinical_id(result_id, "result_id")
    with clinical_connect(paths) as connection:
        ledger = clinical_verify_ledger(connection)
        row = connection.execute(
            """
            SELECT r.*,m.source_id,s.mode,s.profile_id
            FROM clinical_results r
            JOIN clinical_messages m ON m.message_id=r.message_id
            JOIN clinical_sources s ON s.source_id=m.source_id
            WHERE r.result_id=?
            """,
            (result_id,),
        ).fetchone()
        if row is None:
            raise ControlError("CLINICAL_RESULT_NOT_FOUND", "clinical result not found", EXIT_CONFLICT)
        gates = connection.execute(
            "SELECT outcome,COUNT(*) AS count FROM clinical_policy_events WHERE result_id=? GROUP BY outcome",
            (result_id,),
        ).fetchall()
        gate_counts = {str(gate["outcome"]): int(gate["count"]) for gate in gates}
        state = clinical_current_state(connection, result_id)
        successor = connection.execute(
            "SELECT successor_result_id FROM clinical_supersession_claims "
            "WHERE predecessor_result_id=?",
            (result_id,),
        ).fetchone()
        successor_result_id = (
            str(successor["successor_result_id"]) if successor is not None else None
        )
    return {
        "audit_integrity": ledger["integrity"],
        "binding_sha256": str(row["binding_sha256"]),
        "canonical_sha256": str(row["canonical_sha256"]),
        "clinical_use_authorized": False,
        "device_control": False,
        "direct_device_transport": False,
        "fhir_output": "OFFLINE_ARTIFACT_ONLY",
        "gate_counts": gate_counts,
        "mode": str(row["mode"]),
        "ok": True,
        "operation": "clinical-status",
        "profile_id": str(row["profile_id"]),
        "raw_result_content_in_status": False,
        "result_id": result_id,
        "result_version": int(row["result_version"]),
        "source_id": str(row["source_id"]),
        "state": state,
        "successor_authorized": successor_result_id is not None,
        "successor_result_id": successor_result_id,
        "supersedes_result_id": (
            str(row["supersedes_result_id"])
            if row["supersedes_result_id"] is not None
            else None
        ),
    }


def clinical_capabilities() -> dict[str, Any]:
    return {
        "automatic_clinical_approval": False,
        "clinical_modes": ["LIVE_SHADOW", "MOCK"],
        "clinical_use_authorized": False,
        "device_commands": [],
        "device_control": False,
        "direct_device_transport": False,
        "export_authorization_evidence": "DETACHED_ED25519_MANIFEST_REQUIRES_TRUSTED_KEY",
        "fhir_output": "OFFLINE_ARTIFACT_ONLY",
        "ingest_sources": [
            "deidentified_live_shadow_file",
            "strict_synthetic_hl7_file",
        ],
        "live_listener": False,
        "network_client": False,
        "ok": True,
        "operation": "clinical-capabilities",
        "profile_ids": dict(CLINICAL_PROFILE_IDS),
        "real_phi_authorized": False,
        "site_validated": False,
        "transport_bridge": "SEPARATE_PROCESS_REQUIRED",
    }


def unsigned_clinical_review(envelope: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(envelope))
    result["authorization"]["signature"] = ""
    return result


def clinical_review_signature_message(envelope: Mapping[str, Any]) -> bytes:
    return CLINICAL_REVIEW_DOMAIN + canonical_json(unsigned_clinical_review(envelope))


def validate_clinical_review_envelope(
    envelope: Any,
    *,
    check_time: bool = True,
) -> dict[str, Any]:
    value = require_exact_keys(
        envelope,
        {
            "action", "authorization", "binding_sha256", "candidate_artifact_sha256",
            "canonical_sha256", "decision", "expires_at", "installation_id", "issued_at",
            "mapping_sha256", "nonce", "policy_sha256",
            "profile_sha256", "raw_sha256", "recipient_id", "result_id", "review_id",
            "reviewer_id", "schema", "transform_version",
        },
        "clinical review envelope",
    )
    if value["schema"] != CLINICAL_REVIEW_SCHEMA or value["action"] != CLINICAL_REVIEW_ACTION:
        raise ControlError("INVALID_CLINICAL_REVIEW", "clinical review schema/action is invalid", EXIT_INPUT)
    if value["decision"] != "AUTHORIZE_OFFLINE_FHIR_EXPORT":
        raise ControlError("INVALID_CLINICAL_REVIEW", "only offline FHIR export may be authorized", EXIT_INPUT)
    validate_clinical_id(value["result_id"], "result_id")
    validate_operator(value["reviewer_id"])
    validate_small_text(value["recipient_id"], "recipient_id", 256)
    validate_small_text(value["installation_id"], "installation_id", 128)
    if value["transform_version"] != CLINICAL_TRANSFORM_VERSION:
        raise ControlError("INVALID_CLINICAL_REVIEW", "transform version is unsupported", EXIT_INPUT)
    try:
        uuid.UUID(str(value["review_id"]))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ControlError("INVALID_CLINICAL_REVIEW", "review_id must be a UUID", EXIT_INPUT) from exc
    for field in (
        "binding_sha256", "candidate_artifact_sha256", "canonical_sha256", "mapping_sha256",
        "policy_sha256", "profile_sha256", "raw_sha256",
    ):
        if not isinstance(value[field], str) or not re.fullmatch(r"[0-9a-f]{64}", value[field]):
            raise ControlError("INVALID_CLINICAL_REVIEW", f"{field} must be SHA-256 hex", EXIT_INPUT)
    b64url_decode(value["nonce"], minimum_bytes=32)
    authorization = require_exact_keys(
        value["authorization"], {"scheme", "signature"}, "clinical review authorization"
    )
    if authorization["scheme"] != SIGNATURE_SCHEME or not isinstance(authorization["signature"], str):
        raise ControlError("INVALID_CLINICAL_REVIEW", "review authorization is invalid", EXIT_INPUT)
    if authorization["signature"]:
        b64url_decode(authorization["signature"], exact_bytes=64)
    issued_at = parse_time(value["issued_at"], "issued_at")
    expires_at = parse_time(value["expires_at"], "expires_at")
    if expires_at <= issued_at or (expires_at - issued_at).total_seconds() > MAX_TTL_SECONDS:
        raise ControlError("INVALID_CLINICAL_REVIEW", "review TTL must be 1..300 seconds", EXIT_INPUT)
    if check_time:
        now = utc_now()
        if issued_at > now + timedelta(seconds=MAX_CLOCK_SKEW_SECONDS):
            raise ControlError("CLINICAL_REVIEW_FROM_FUTURE", "review is dated too far in the future", EXIT_DENIED)
        if expires_at <= now:
            raise ControlError("CLINICAL_REVIEW_EXPIRED", "clinical review has expired", EXIT_DENIED)
    return dict(value)


def build_clinical_review(
    paths: StatePaths,
    result_id: str,
    reviewer_id: str,
    ttl_seconds: int,
) -> dict[str, Any]:
    result_id = validate_clinical_id(result_id, "result_id")
    reviewer_id = validate_operator(reviewer_id)
    if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= MAX_TTL_SECONDS:
        raise ControlError("INVALID_TTL", "TTL must be an integer from 1 through 300", EXIT_INPUT)
    with clinical_connect(paths) as connection:
        clinical_verify_ledger(connection)
        row = connection.execute(
            """
            SELECT r.*,m.raw_sha256,m.source_id,s.mode
            FROM clinical_results r
            JOIN clinical_messages m ON m.message_id=r.message_id
            JOIN clinical_sources s ON s.source_id=m.source_id
            WHERE r.result_id=?
            """,
            (result_id,),
        ).fetchone()
        if row is None:
            raise ControlError("CLINICAL_RESULT_NOT_FOUND", "clinical result not found", EXIT_CONFLICT)
        if str(row["mode"]) not in CLINICAL_MODES:
            raise ControlError(
                "CLINICAL_RESULT_NOT_REVIEWABLE",
                "result source mode is not reviewable",
                EXIT_DENIED,
            )
        if clinical_current_state(connection, result_id) != "PENDING_REVIEW":
            raise ControlError("RESULT_NOT_REVIEWABLE", "result is not pending reviewer-key authorization", EXIT_DENIED)
        failed = int(connection.execute(
            "SELECT COUNT(*) FROM clinical_policy_events WHERE result_id=? AND outcome<>'PASS'",
            (result_id,),
        ).fetchone()[0])
        gate_count = int(connection.execute(
            "SELECT COUNT(*) FROM clinical_policy_events WHERE result_id=?", (result_id,)
        ).fetchone()[0])
        if failed or gate_count != len(CLINICAL_REQUIRED_GATES):
            raise ControlError("CLINICAL_POLICY_NOT_PASSING", "all required gates must pass", EXIT_DENIED)
        sealed = connection.execute(
            "SELECT value FROM clinical_metadata WHERE key='review_trust_sealed'"
        ).fetchone()
        reviewer = connection.execute(
            "SELECT 1 FROM clinical_reviewers WHERE reviewer_id=? AND enabled=1", (reviewer_id,)
        ).fetchone()
        if sealed is None or str(sealed["value"]) != "1" or reviewer is None:
            raise ControlError("CLINICAL_REVIEWER_NOT_TRUSTED", "reviewer trust is unavailable", EXIT_DENIED)
        binding = parse_json_bytes(str(row["binding_json"]).encode("utf-8"))
        source_row = connection.execute(
            "SELECT * FROM clinical_sources WHERE source_id=(SELECT source_id FROM clinical_messages WHERE message_id=?)",
            (str(row["message_id"]),),
        ).fetchone()
        observation_rows = connection.execute(
            "SELECT * FROM clinical_observations WHERE result_id=? ORDER BY ordinal", (result_id,)
        ).fetchall()
        installation = connection.execute(
            "SELECT value FROM clinical_metadata WHERE key='installation_id'"
        ).fetchone()
        if source_row is None or installation is None or not observation_rows:
            raise IntegrityFailure("candidate artifact inputs are incomplete")
        candidate = build_offline_fhir_bundle(
            row,
            observation_rows,
            source_row,
            clinical_predecessor_artifact_sha256(connection, row),
        )
        candidate_sha256 = hashlib.sha256(pretty_json_bytes(candidate)).hexdigest()
    now = utc_now()
    envelope = {
        "action": CLINICAL_REVIEW_ACTION,
        "authorization": {"scheme": SIGNATURE_SCHEME, "signature": ""},
        "binding_sha256": str(row["binding_sha256"]),
        "candidate_artifact_sha256": candidate_sha256,
        "canonical_sha256": str(row["canonical_sha256"]),
        "decision": "AUTHORIZE_OFFLINE_FHIR_EXPORT",
        "expires_at": format_time(now + timedelta(seconds=ttl_seconds)),
        "installation_id": str(installation["value"]),
        "issued_at": format_time(now),
        "mapping_sha256": str(row["mapping_sha256"]),
        "nonce": b64url_encode(secrets.token_bytes(32)),
        "policy_sha256": clinical_policy_sha256(),
        "profile_sha256": str(row["profile_sha256"]),
        "raw_sha256": str(row["raw_sha256"]),
        "recipient_id": str(binding["recipient_id"]),
        "result_id": result_id,
        "review_id": str(uuid.uuid4()),
        "reviewer_id": reviewer_id,
        "schema": CLINICAL_REVIEW_SCHEMA,
        "transform_version": CLINICAL_TRANSFORM_VERSION,
    }
    validate_clinical_review_envelope(envelope)
    return envelope


def sign_clinical_review(
    paths: StatePaths,
    envelope: dict[str, Any],
    private_key: Ed25519PrivateKey,
) -> dict[str, Any]:
    envelope = validate_clinical_review_envelope(envelope)
    if envelope["authorization"]["signature"]:
        raise ControlError("CLINICAL_REVIEW_ALREADY_SIGNED", "review is already signed", EXIT_CONFLICT)
    reviewer_id = str(envelope["reviewer_id"])
    with clinical_connect(paths) as connection:
        clinical_verify_ledger(connection)
        sealed = connection.execute(
            "SELECT value FROM clinical_metadata WHERE key='review_trust_sealed'"
        ).fetchone()
        row = connection.execute(
            "SELECT public_key_b64 FROM clinical_reviewers WHERE reviewer_id=? AND enabled=1",
            (reviewer_id,),
        ).fetchone()
    if sealed is None or str(sealed["value"]) != "1" or row is None:
        raise ControlError("CLINICAL_REVIEWER_NOT_TRUSTED", "reviewer is not trusted", EXIT_DENIED)
    if not secrets.compare_digest(
        b64url_decode(row["public_key_b64"], exact_bytes=32),
        public_key_raw(private_key.public_key()),
    ):
        raise ControlError("PRIVATE_KEY_MISMATCH", "private key does not match reviewer", EXIT_DENIED)
    result = copy.deepcopy(envelope)
    result["authorization"]["signature"] = b64url_encode(
        private_key.sign(clinical_review_signature_message(envelope))
    )
    validate_clinical_review_envelope(result)
    return result


def apply_clinical_review(paths: StatePaths, envelope: dict[str, Any]) -> dict[str, Any]:
    envelope = validate_clinical_review_envelope(envelope)
    signature = str(envelope["authorization"]["signature"])
    if not signature:
        raise ControlError("CLINICAL_REVIEW_UNSIGNED", "clinical review is unsigned", EXIT_DENIED)
    result_id = str(envelope["result_id"])
    reviewer_id = str(envelope["reviewer_id"])
    applied_at = format_time(utc_now())
    envelope_json = canonical_json(envelope).decode("utf-8")
    envelope_sha256 = hashlib.sha256(envelope_json.encode("utf-8")).hexdigest()
    with clinical_connect(paths) as connection:
        with immediate_transaction(connection):
            clinical_verify_ledger(connection)
            reviewer = connection.execute(
                "SELECT public_key_b64 FROM clinical_reviewers WHERE reviewer_id=? AND enabled=1",
                (reviewer_id,),
            ).fetchone()
            if reviewer is None:
                raise ControlError("CLINICAL_REVIEWER_NOT_TRUSTED", "reviewer is not trusted", EXIT_DENIED)
            sealed = connection.execute(
                "SELECT value FROM clinical_metadata WHERE key='review_trust_sealed'"
            ).fetchone()
            if sealed is None or str(sealed["value"]) != "1":
                raise ControlError(
                    "CLINICAL_REVIEW_TRUST_NOT_SEALED",
                    "reviewer trust must be sealed before applying authorization",
                    EXIT_DENIED,
                )
            public_key = Ed25519PublicKey.from_public_bytes(
                b64url_decode(reviewer["public_key_b64"], exact_bytes=32)
            )
            try:
                public_key.verify(
                    b64url_decode(signature, exact_bytes=64),
                    clinical_review_signature_message(envelope),
                )
            except InvalidSignature as exc:
                raise ControlError("CLINICAL_REVIEW_SIGNATURE_INVALID", "review signature is invalid", EXIT_DENIED) from exc
            row = connection.execute(
                """
                SELECT r.*,m.raw_sha256,m.source_id,s.mode
                FROM clinical_results r
                JOIN clinical_messages m ON m.message_id=r.message_id
                JOIN clinical_sources s ON s.source_id=m.source_id
                WHERE r.result_id=?
                """,
                (result_id,),
            ).fetchone()
            if row is None or str(row["mode"]) not in CLINICAL_MODES:
                raise ControlError(
                    "CLINICAL_RESULT_NOT_EXPORTABLE",
                    "result is absent or its source mode is unsupported",
                    EXIT_DENIED,
                )
            binding = parse_json_bytes(str(row["binding_json"]).encode("utf-8"))
            review_observations = connection.execute(
                "SELECT * FROM clinical_observations WHERE result_id=? ORDER BY ordinal",
                (result_id,),
            ).fetchall()
            review_source = connection.execute(
                "SELECT * FROM clinical_sources WHERE source_id=?",
                (str(row["source_id"]),),
            ).fetchone()
            if review_source is None or not review_observations:
                raise IntegrityFailure("clinical review candidate inputs are incomplete")
            review_candidate = build_offline_fhir_bundle(
                row,
                review_observations,
                review_source,
                clinical_predecessor_artifact_sha256(connection, row),
            )
            expected = {
                "binding_sha256": str(row["binding_sha256"]),
                "candidate_artifact_sha256": hashlib.sha256(
                    pretty_json_bytes(review_candidate)
                ).hexdigest(),
                "canonical_sha256": str(row["canonical_sha256"]),
                "mapping_sha256": str(row["mapping_sha256"]),
                "policy_sha256": clinical_policy_sha256(),
                "profile_sha256": str(row["profile_sha256"]),
                "raw_sha256": str(row["raw_sha256"]),
                "recipient_id": str(binding["recipient_id"]),
                "installation_id": str(connection.execute(
                    "SELECT value FROM clinical_metadata WHERE key='installation_id'"
                ).fetchone()["value"]),
            }
            for field, expected_value in expected.items():
                if not secrets.compare_digest(str(envelope[field]), expected_value):
                    raise ControlError("CLINICAL_REVIEW_STALE", f"review field {field} is stale", EXIT_DENIED)
            if clinical_current_state(connection, result_id) != "PENDING_REVIEW":
                raise ControlError("RESULT_NOT_REVIEWABLE", "result is not pending reviewer-key authorization", EXIT_DENIED)
            failed = int(connection.execute(
                "SELECT COUNT(*) FROM clinical_policy_events WHERE result_id=? AND outcome<>'PASS'",
                (result_id,),
            ).fetchone()[0])
            gate_count = int(connection.execute(
                "SELECT COUNT(*) FROM clinical_policy_events WHERE result_id=?", (result_id,)
            ).fetchone()[0])
            if failed or gate_count != len(CLINICAL_REQUIRED_GATES):
                raise ControlError("CLINICAL_POLICY_NOT_PASSING", "policy gates changed or failed", EXIT_DENIED)
            supersedes = row["supersedes_result_id"]
            if supersedes is not None:
                try:
                    connection.execute(
                        "INSERT INTO clinical_supersession_claims VALUES (?,?,?)",
                        (str(supersedes), result_id, applied_at),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ControlError(
                        "SUPERSESSION_ALREADY_AUTHORIZED",
                        "another correction already holds reviewer-key authorization for this predecessor",
                        EXIT_CONFLICT,
                    ) from exc
                clinical_append_audit(
                    connection,
                    "CLINICAL_SUPERSESSION_CLAIMED",
                    result_id,
                    reviewer_id,
                    {"predecessor_result_id": str(supersedes)},
                )
            connection.execute(
                "INSERT INTO clinical_review_attestations VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    str(envelope["review_id"]),
                    result_id,
                    reviewer_id,
                    str(envelope["recipient_id"]),
                    str(envelope["nonce"]),
                    envelope_json,
                    envelope_sha256,
                    signature,
                    applied_at,
                ),
            )
            clinical_append_audit(
                connection,
                "CLINICAL_REVIEW_APPLIED",
                result_id,
                reviewer_id,
                {
                    "decision": envelope["decision"],
                    "envelope_sha256": envelope_sha256,
                    "recipient_id": envelope["recipient_id"],
                    "review_id": envelope["review_id"],
                },
            )
            clinical_transition(
                connection,
                result_id,
                "AUTHORIZED_FOR_EXPORT",
                "REVIEWER_KEY",
                reviewer_id,
                "SIGNED_EXACT_CANDIDATE_REVIEWER_KEY_AUTHORIZATION",
            )
    return {
        "clinical_delivery": False,
        "decision": "AUTHORIZE_OFFLINE_FHIR_EXPORT",
        "ok": True,
        "operation": "clinical-review-apply",
        "result_id": result_id,
        "review_id": str(envelope["review_id"]),
        "state": "AUTHORIZED_FOR_EXPORT",
    }


def hl7_time_to_fhir(value: str) -> str | None:
    if not value:
        return None
    match = re.fullmatch(
        r"(\d{4})(\d{2})(\d{2})(?:(\d{2})(\d{2})(\d{2}))?(?:\.(\d{1,6}))?([+-]\d{4})?",
        value,
    )
    if match is None:
        return None
    year, month, day, hour, minute, second, fraction, offset = match.groups()
    if hour is None:
        try:
            datetime(int(year), int(month), int(day))
        except ValueError:
            return None
        return f"{year}-{month}-{day}"
    if not offset:
        return None
    offset_hour = int(offset[1:3])
    offset_minute = int(offset[3:5])
    if offset_hour > 14 or offset_minute > 59 or (offset_hour == 14 and offset_minute != 0):
        return None
    try:
        datetime(
            int(year), int(month), int(day), int(hour), int(minute), int(second),
            int((fraction or "0").ljust(6, "0")),
        )
    except ValueError:
        return None
    suffix = f"{offset[:3]}:{offset[3:]}"
    timestamp = f"{year}-{month}-{day}T{hour}:{minute}:{second}"
    if fraction:
        timestamp += "." + fraction
    return timestamp + suffix


def fhir_status(source_status: str) -> str:
    return {
        "C": "corrected",
        "F": "final",
        "P": "preliminary",
        "X": "cancelled",
    }.get(source_status, "unknown")


def clinical_predecessor_artifact_sha256(
    connection: sqlite3.Connection,
    result_row: Mapping[str, Any],
) -> str | None:
    predecessor_id = result_row["supersedes_result_id"]
    if predecessor_id is None:
        return None
    predecessor = connection.execute(
        "SELECT artifact_sha256 FROM clinical_exports WHERE result_id=?",
        (str(predecessor_id),),
    ).fetchone()
    if predecessor is None or not re.fullmatch(r"[0-9a-f]{64}", str(predecessor["artifact_sha256"])):
        raise IntegrityFailure("correction is missing its immediate predecessor artifact hash")
    return str(predecessor["artifact_sha256"])


def build_offline_fhir_bundle(
    result_row: sqlite3.Row,
    observations: Sequence[sqlite3.Row],
    source_row: sqlite3.Row,
    predecessor_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    canonical = parse_json_bytes(str(result_row["canonical_json"]).encode("utf-8"))
    binding = parse_json_bytes(str(result_row["binding_json"]).encode("utf-8"))
    parsed = canonical["context"]["source_result"]
    mapping = canonical["context"]["assay_mapping"]
    result_id = str(result_row["result_id"])
    source_mode = str(source_row["mode"])
    if source_mode not in CLINICAL_MODES:
        raise IntegrityFailure("FHIR candidate source mode is unsupported")
    truth_code = "synthetic" if source_mode == "MOCK" else "deidentified-live-shadow"
    transform_label = (
        "Synthetic offline transform software"
        if source_mode == "MOCK"
        else "Deidentified live-shadow offline transform software"
    )
    offline_base = "https://offline.invalid/fhir/"
    report_id = "report-" + result_id
    report_reference = offline_base + "DiagnosticReport/" + report_id
    observation_resources: list[dict[str, Any]] = []
    observation_references: list[dict[str, str]] = []
    for observation in observations:
        ordinal = int(observation["ordinal"])
        observation_id = f"obs-{result_id}-{ordinal}"
        coding: dict[str, Any] = {"code": str(observation["local_code"])}
        if observation["local_display"]:
            coding["display"] = str(observation["local_display"])
        source_system = str(observation["local_system"])
        if re.match(r"^(?:https?://|urn:)", source_system):
            coding["system"] = source_system
        resource: dict[str, Any] = {
            "basedOn": [{"reference": str(binding["order_reference"])}],
            "category": [
                {
                    "coding": [
                        {
                            "code": "laboratory",
                            "display": "Laboratory",
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        }
                    ]
                }
            ],
            "code": {"coding": [coding]},
            "id": observation_id,
            "identifier": [
                {
                    "system": "urn:owned-agent-control:clinical-observation",
                    "value": f"{result_id}:{ordinal}",
                }
            ],
            "meta": {
                "tag": [
                    {"code": truth_code, "system": "urn:owned-agent-control:truth-boundary"},
                    {"code": "offline-draft", "system": "urn:owned-agent-control:truth-boundary"},
                ]
            },
            "resourceType": "Observation",
            "specimen": {"reference": str(binding["specimen_reference"])},
            "status": fhir_status(str(observation["source_status"] or result_row["source_status"])),
            "subject": {"reference": str(binding["patient_reference"])},
        }
        raw_value = str(observation["raw_value_lexeme"])
        if raw_value:
            resource["valueString"] = raw_value
        else:
            resource["dataAbsentReason"] = {
                "coding": [
                    {
                        "code": "unknown",
                        "display": "Unknown",
                        "system": "http://terminology.hl7.org/CodeSystem/data-absent-reason",
                    }
                ],
                "text": "Source value was empty; no zero was inferred.",
            }
        effective = hl7_time_to_fhir(str(parsed["effective_time"]))
        if effective:
            resource["effectiveDateTime"] = effective
        units = (str(observation["unit_code"]), str(observation["unit_display"]))
        if source_mode == "MOCK" and units not in {("", ""), ("0", "")}:
            raise IntegrityFailure("synthetic observation units differ from the fixed profile")
        notes: list[dict[str, str]] = []
        if observation["note_text"]:
            notes.append({"text": str(observation["note_text"])})
        if notes:
            resource["note"] = notes
        observation_resources.append(resource)
        observation_references.append(
            {"reference": offline_base + "Observation/" + observation_id}
        )
    report: dict[str, Any] = {
        "basedOn": [{"reference": str(binding["order_reference"])}],
        "category": [
            {
                "coding": [
                    {
                        "code": "LAB",
                        "display": "Laboratory",
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "code": str(result_row["assay_code"]),
                    "display": str(mapping["display"]),
                    "system": str(mapping["local_system"]),
                }
            ]
        },
        "id": report_id,
        "identifier": [
            {
                "system": "urn:owned-agent-control:source-report",
                "value": str(result_row["source_report_id"]),
            },
            {
                "system": "urn:owned-agent-control:result-version",
                "value": str(result_row["result_version"]),
            },
        ],
        "meta": {
            "tag": [
                {"code": truth_code, "system": "urn:owned-agent-control:truth-boundary"},
                {"code": "offline-draft", "system": "urn:owned-agent-control:truth-boundary"},
                {"code": str(source_row["profile_id"]), "system": "urn:owned-agent-control:source-profile"},
                {
                    "code": "source-hl7-status-" + str(result_row["source_status"]),
                    "system": "urn:owned-agent-control:source-result-status",
                },
            ]
        },
        "resourceType": "DiagnosticReport",
        "result": observation_references,
        "specimen": [{"reference": str(binding["specimen_reference"])}],
        "status": (
            "corrected"
            if result_row["supersedes_result_id"] is not None
            else fhir_status(str(result_row["source_status"]))
        ),
        "subject": {"reference": str(binding["patient_reference"])},
    }
    effective = hl7_time_to_fhir(str(parsed["effective_time"]))
    issued = hl7_time_to_fhir(str(parsed["issued_time"]))
    if effective:
        report["effectiveDateTime"] = effective
    if issued and "T" in issued:
        report["issued"] = issued
    provenance_id = "prov-" + result_id
    provenance_targets = [{"reference": report_reference}] + observation_references
    provenance_entities = [
        {
            "role": "source",
            "what": {
                "identifier": {
                    "system": "urn:sha256",
                    "value": str(canonical["context"]["raw_sha256"]),
                }
            },
        }
    ]
    if result_row["supersedes_result_id"]:
        if predecessor_artifact_sha256 is None or not re.fullmatch(
            r"[0-9a-f]{64}", predecessor_artifact_sha256
        ):
            raise IntegrityFailure("correction provenance lacks the predecessor artifact hash")
        report["meta"]["tag"].append({
            "code": truth_code + "-correction",
            "system": "urn:owned-agent-control:truth-boundary",
        })
        provenance_entities.append(
            {
                "role": "revision",
                "what": {
                    "identifier": {
                        "system": "urn:sha256",
                        "value": predecessor_artifact_sha256,
                    },
                    "reference": (
                        offline_base
                        + "DiagnosticReport/report-"
                        + str(result_row["supersedes_result_id"])
                    )
                },
            }
        )
    provenance = {
        "activity": {
            "text": (
                f"Local {truth_code} input normalization recorded at ingest; "
                "not clinical verification or delivery"
            )
        },
        "agent": [
            {
                "type": {"text": transform_label},
                "who": {
                    "identifier": {
                        "system": "urn:owned-agent-control:transform-version",
                        "value": CLINICAL_TRANSFORM_VERSION,
                    }
                },
            }
        ],
        "entity": provenance_entities,
        "id": provenance_id,
        "recorded": str(result_row["created_at"]),
        "resourceType": "Provenance",
        "target": provenance_targets,
    }
    entries = [
        {"fullUrl": report_reference, "resource": report},
        *[
            {
                "fullUrl": offline_base + "Observation/" + str(resource["id"]),
                "resource": resource,
            }
            for resource in observation_resources
        ],
        {
            "fullUrl": offline_base + "Provenance/" + provenance_id,
            "resource": provenance,
        },
    ]
    return {
        "entry": entries,
        "id": "bundle-" + result_id,
        "identifier": {
            "system": "urn:owned-agent-control:offline-fhir-bundle",
            "value": result_id,
        },
        "meta": {
            "tag": [
                {"code": truth_code, "system": "urn:owned-agent-control:truth-boundary"},
                {"code": "not-delivered", "system": "urn:owned-agent-control:truth-boundary"},
            ]
        },
        "resourceType": "Bundle",
        "timestamp": str(result_row["created_at"]),
        "type": "collection",
    }


def build_clinical_export_authorization_manifest(
    *,
    artifact_sha256: str,
    created_at: str,
    export_id: str,
    installation_id: str,
    result_id: str,
    review_row: Mapping[str, Any],
    reviewer_row: Mapping[str, Any],
    source_mode: str,
) -> dict[str, Any]:
    envelope = validate_clinical_review_envelope(
        parse_json_bytes(str(review_row["envelope_json"]).encode("utf-8")),
        check_time=False,
    )
    if not str(envelope["authorization"]["signature"]):
        raise IntegrityFailure("export authorization envelope is unsigned")
    manifest = {
        "artifact": {
            "format": "FHIR_R4_JSON_OFFLINE_COLLECTION",
            "sha256": artifact_sha256,
        },
        "authorization": {
            "applied_at": str(review_row["applied_at"]),
            "database_trust_scope": (
                "local_ledger_required_to_prove_sealed_reviewer_trust_at_apply"
            ),
            "envelope": envelope,
            "envelope_sha256": str(review_row["envelope_sha256"]),
            "key_proof_scope": (
                "ed25519_key_possession_not_human_identity_licensure_"
                "clinical_verification_or_delivery"
            ),
            "reviewer_fingerprint": str(reviewer_row["fingerprint"]),
            "reviewer_public_key_b64": str(reviewer_row["public_key_b64"]),
        },
        "clinical_delivery": False,
        "export": {
            "created_at": created_at,
            "export_id": export_id,
            "result_id": result_id,
        },
        "installation_id": installation_id,
        "ledger_correlation_scope": (
            "export_id_and_apply_export_timestamps_require_local_ledger"
        ),
        "schema": CLINICAL_EXPORT_AUTHORIZATION_SCHEMA,
        "source_mode": source_mode,
        "truth_boundary": CLINICAL_EXPORT_TRUTH_BOUNDARIES[source_mode],
    }
    validate_clinical_export_authorization_manifest(manifest)
    return manifest


def validate_clinical_export_authorization_manifest(value: Any) -> dict[str, Any]:
    manifest = require_exact_keys(
        value,
        {
            "artifact", "authorization", "clinical_delivery", "export",
            "installation_id", "ledger_correlation_scope", "schema", "source_mode",
            "truth_boundary",
        },
        "clinical export authorization manifest",
    )
    if manifest["schema"] != CLINICAL_EXPORT_AUTHORIZATION_SCHEMA:
        raise ControlError(
            "INVALID_EXPORT_AUTHORIZATION", "export authorization schema is unsupported", EXIT_INPUT
        )
    source_mode = str(manifest["source_mode"])
    if source_mode not in CLINICAL_MODES:
        raise ControlError(
            "INVALID_EXPORT_AUTHORIZATION", "export source mode is unsupported", EXIT_INPUT
        )
    if manifest["clinical_delivery"] is not False or manifest["truth_boundary"] != (
        CLINICAL_EXPORT_TRUTH_BOUNDARIES[source_mode]
    ) or manifest["ledger_correlation_scope"] != (
        "export_id_and_apply_export_timestamps_require_local_ledger"
    ):
        raise ControlError(
            "INVALID_EXPORT_AUTHORIZATION", "export authorization overstates its truth boundary", EXIT_INPUT
        )
    artifact = require_exact_keys(
        manifest["artifact"], {"format", "sha256"}, "authorized artifact"
    )
    if artifact["format"] != "FHIR_R4_JSON_OFFLINE_COLLECTION" or not isinstance(
        artifact["sha256"], str
    ) or not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]):
        raise ControlError(
            "INVALID_EXPORT_AUTHORIZATION", "authorized artifact identity is invalid", EXIT_INPUT
        )
    export = require_exact_keys(
        manifest["export"], {"created_at", "export_id", "result_id"}, "authorized export"
    )
    validate_clinical_id(export["result_id"], "result_id")
    try:
        uuid.UUID(str(export["export_id"]))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ControlError(
            "INVALID_EXPORT_AUTHORIZATION", "export_id must be a UUID", EXIT_INPUT
        ) from exc
    parse_time(export["created_at"], "created_at")
    validate_small_text(manifest["installation_id"], "installation_id", 128)
    authorization = require_exact_keys(
        manifest["authorization"],
        {
            "applied_at", "database_trust_scope", "envelope", "envelope_sha256",
            "key_proof_scope", "reviewer_fingerprint", "reviewer_public_key_b64",
        },
        "export reviewer-key authorization",
    )
    parse_time(authorization["applied_at"], "applied_at")
    if authorization["database_trust_scope"] != (
        "local_ledger_required_to_prove_sealed_reviewer_trust_at_apply"
    ) or authorization["key_proof_scope"] != (
        "ed25519_key_possession_not_human_identity_licensure_"
        "clinical_verification_or_delivery"
    ):
        raise ControlError(
            "INVALID_EXPORT_AUTHORIZATION", "authorization scope labels are invalid", EXIT_INPUT
        )
    envelope = validate_clinical_review_envelope(authorization["envelope"], check_time=False)
    envelope_json = canonical_json(envelope).decode("utf-8")
    envelope_sha256 = hashlib.sha256(envelope_json.encode("utf-8")).hexdigest()
    if (
        not secrets.compare_digest(envelope_sha256, str(authorization["envelope_sha256"]))
        or not secrets.compare_digest(str(artifact["sha256"]), str(envelope["candidate_artifact_sha256"]))
        or str(export["result_id"]) != str(envelope["result_id"])
        or str(manifest["installation_id"]) != str(envelope["installation_id"])
    ):
        raise IntegrityFailure("export authorization is not bound to the signed artifact envelope")
    public_key_bytes = b64url_decode(authorization["reviewer_public_key_b64"], exact_bytes=32)
    expected_fingerprint = hashlib.sha256(public_key_bytes).hexdigest()
    if not secrets.compare_digest(expected_fingerprint, str(authorization["reviewer_fingerprint"])):
        raise IntegrityFailure("export authorization reviewer fingerprint is invalid")
    signature = str(envelope["authorization"]["signature"])
    if not signature:
        raise IntegrityFailure("export authorization envelope has no signature")
    try:
        Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(
            b64url_decode(signature, exact_bytes=64),
            clinical_review_signature_message(envelope),
        )
    except InvalidSignature as exc:
        raise IntegrityFailure("export authorization reviewer signature is invalid") from exc
    return dict(manifest)


def verify_clinical_export_authorization(
    artifact_path: Path,
    manifest_path: Path,
    trusted_reviewer_key_path: Path,
) -> dict[str, Any]:
    for path, label, maximum in (
        (artifact_path, "offline FHIR artifact", 16 * 1024 * 1024),
        (manifest_path, "export authorization manifest", MAX_JSON_BYTES),
    ):
        try:
            safe_regular_file = path.is_file() and not _is_reparse_point(path)
            size = path.stat().st_size if safe_regular_file else 0
        except OSError as exc:
            raise ControlError(
                "EXPORT_EVIDENCE_READ_FAILED", f"unable to inspect {label}: {exc}", EXIT_INPUT
            ) from exc
        if not safe_regular_file:
            raise ControlError("UNSAFE_EXPORT_EVIDENCE", f"{label} is not a safe regular file", EXIT_INPUT)
        if size > maximum:
            raise ControlError("EXPORT_EVIDENCE_TOO_LARGE", f"{label} exceeds its size limit", EXIT_INPUT)
    try:
        artifact_bytes = artifact_path.read_bytes()
    except OSError as exc:
        raise ControlError(
            "EXPORT_EVIDENCE_READ_FAILED",
            f"unable to read offline FHIR artifact: {exc}",
            EXIT_INPUT,
        ) from exc
    manifest = validate_clinical_export_authorization_manifest(load_json_file(manifest_path))
    observed_artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if not secrets.compare_digest(
        observed_artifact_sha256, str(manifest["artifact"]["sha256"])
    ):
        raise IntegrityFailure("offline FHIR bytes differ from the signed authorization")
    trusted_key = load_public_key(trusted_reviewer_key_path)
    trusted_raw = public_key_raw(trusted_key)
    embedded_raw = b64url_decode(
        manifest["authorization"]["reviewer_public_key_b64"], exact_bytes=32
    )
    if not secrets.compare_digest(trusted_raw, embedded_raw):
        raise ControlError(
            "UNTRUSTED_EXPORT_REVIEWER_KEY",
            "embedded reviewer key does not match the supplied trust anchor",
            EXIT_DENIED,
        )
    return {
        "artifact_exactly_bound": True,
        "artifact_sha256": observed_artifact_sha256,
        "clinical_delivery": False,
        "key_proof_scope": manifest["authorization"]["key_proof_scope"],
        "ledger_correlation_verified": False,
        "ok": True,
        "operation": "clinical-export-verify",
        "result_id": manifest["export"]["result_id"],
        "reviewer_fingerprint": manifest["authorization"]["reviewer_fingerprint"],
        "signature_cryptographically_valid": True,
        "source_mode": manifest["source_mode"],
        "trusted_reviewer_key_matched": True,
        "truth_boundary": manifest["truth_boundary"],
    }


def clinical_export_fhir(paths: StatePaths, result_id: str, output_path: Path) -> dict[str, Any]:
    result_id = validate_clinical_id(result_id, "result_id")
    artifact_text = ""
    artifact_sha256 = ""
    authorization_manifest_text = ""
    authorization_manifest_sha256 = ""
    idempotent = False
    reported_state = ""
    with clinical_connect(paths) as connection:
        with immediate_transaction(connection):
            clinical_verify_ledger(connection)
            result_row = connection.execute(
                """
                SELECT r.*,m.raw_sha256,m.source_id
                FROM clinical_results r JOIN clinical_messages m ON m.message_id=r.message_id
                WHERE r.result_id=?
                """,
                (result_id,),
            ).fetchone()
            if result_row is None:
                raise ControlError("CLINICAL_RESULT_NOT_FOUND", "clinical result not found", EXIT_CONFLICT)
            source_row = connection.execute(
                "SELECT * FROM clinical_sources WHERE source_id=?", (str(result_row["source_id"]),)
            ).fetchone()
            if source_row is None or str(source_row["mode"]) not in CLINICAL_MODES:
                raise ControlError(
                    "CLINICAL_EXPORT_MODE_DENIED",
                    "result source mode is not eligible for offline artifact export",
                    EXIT_DENIED,
                )
            state = clinical_current_state(connection, result_id)
            existing = connection.execute(
                "SELECT * FROM clinical_exports WHERE result_id=?", (result_id,)
            ).fetchone()
            if state in {"ARTIFACT_CREATED", "SUPERSEDED"} and existing is not None:
                artifact_text = str(existing["artifact_json"])
                artifact_sha256 = str(existing["artifact_sha256"])
                authorization_manifest_text = str(existing["authorization_manifest_json"])
                authorization_manifest_sha256 = str(
                    existing["authorization_manifest_sha256"]
                )
                idempotent = True
                reported_state = state
            else:
                if state != "AUTHORIZED_FOR_EXPORT":
                    raise ControlError(
                        "RESULT_NOT_AUTHORIZED_FOR_EXPORT",
                        "signed reviewer-key authorization is required",
                        EXIT_DENIED,
                    )
                review_row = connection.execute(
                    "SELECT * FROM clinical_review_attestations WHERE result_id=?", (result_id,)
                ).fetchone()
                observations = connection.execute(
                    "SELECT * FROM clinical_observations WHERE result_id=? ORDER BY ordinal", (result_id,)
                ).fetchall()
                if review_row is None or not observations:
                    raise IntegrityFailure("authorized result is missing review or observations")
                bundle = build_offline_fhir_bundle(
                    result_row,
                    observations,
                    source_row,
                    clinical_predecessor_artifact_sha256(connection, result_row),
                )
                artifact_text = pretty_json_bytes(bundle).decode("utf-8")
                artifact_sha256 = hashlib.sha256(artifact_text.encode("utf-8")).hexdigest()
                signed_envelope = parse_json_bytes(str(review_row["envelope_json"]).encode("utf-8"))
                if not secrets.compare_digest(
                    artifact_sha256, str(signed_envelope["candidate_artifact_sha256"])
                ):
                    raise IntegrityFailure("export bytes differ from the signed candidate artifact")
                export_id = str(uuid.uuid4())
                created_at = format_time(utc_now())
                reviewer_row = connection.execute(
                    "SELECT public_key_b64,fingerprint FROM clinical_reviewers WHERE reviewer_id=?",
                    (str(review_row["reviewer_id"]),),
                ).fetchone()
                installation = connection.execute(
                    "SELECT value FROM clinical_metadata WHERE key='installation_id'"
                ).fetchone()
                if reviewer_row is None or installation is None:
                    raise IntegrityFailure("export authorization trust inputs are missing")
                authorization_manifest = build_clinical_export_authorization_manifest(
                    artifact_sha256=artifact_sha256,
                    created_at=created_at,
                    export_id=export_id,
                    installation_id=str(installation["value"]),
                    result_id=result_id,
                    review_row=review_row,
                    reviewer_row=reviewer_row,
                    source_mode=str(source_row["mode"]),
                )
                authorization_manifest_text = pretty_json_bytes(
                    authorization_manifest
                ).decode("utf-8")
                authorization_manifest_sha256 = hashlib.sha256(
                    authorization_manifest_text.encode("utf-8")
                ).hexdigest()
                connection.execute(
                    "INSERT INTO clinical_exports VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        export_id,
                        result_id,
                        str(review_row["review_id"]),
                        "FHIR_R4_JSON_OFFLINE_COLLECTION",
                        artifact_text,
                        artifact_sha256,
                        authorization_manifest_text,
                        authorization_manifest_sha256,
                        created_at,
                    ),
                )
                clinical_append_audit(
                    connection,
                    "CLINICAL_FHIR_ARTIFACT_CREATED",
                    result_id,
                    str(review_row["reviewer_id"]),
                    {
                        "artifact_sha256": artifact_sha256,
                        "authorization_manifest_sha256": authorization_manifest_sha256,
                        "delivery_claimed": False,
                        "export_id": export_id,
                        "format": "FHIR_R4_JSON_OFFLINE_COLLECTION",
                    },
                )
                clinical_transition(
                    connection,
                    result_id,
                    "ARTIFACT_CREATED",
                    "SERVICE",
                    "clinical-fhir-export",
                    "OFFLINE_ARTIFACT_COMMITTED_NOT_DELIVERED",
                )
                reported_state = "ARTIFACT_CREATED"
                supersedes = result_row["supersedes_result_id"]
                if supersedes is not None:
                    clinical_transition(
                        connection,
                        str(supersedes),
                        "SUPERSEDED",
                        "SERVICE",
                        "clinical-fhir-export",
                        "LINKED_CORRECTION_ARTIFACT_COMMITTED",
                    )
    written = clinical_write_output(
        paths,
        output_path,
        artifact_text.encode("utf-8"),
        "offline FHIR output",
    )
    authorization_manifest_path = Path(
        str(Path(os.path.abspath(output_path.expanduser())))
        + CLINICAL_EXPORT_AUTHORIZATION_SUFFIX
    )
    written_manifest = clinical_write_output(
        paths,
        authorization_manifest_path,
        authorization_manifest_text.encode("utf-8"),
        "offline FHIR authorization manifest",
    )
    return {
        "artifact_sha256": artifact_sha256,
        "authorization_manifest_out": str(written_manifest),
        "authorization_manifest_sha256": authorization_manifest_sha256,
        "clinical_delivery": False,
        "fhir_version": "R4",
        "idempotent": idempotent,
        "ok": True,
        "operation": "clinical-export-fhir",
        "out": str(written),
        "result_id": result_id,
        "source_mode": str(source_row["mode"]),
        "state": reported_state,
        "truth_boundary": CLINICAL_EXPORT_TRUTH_BOUNDARIES[str(source_row["mode"])],
    }


def synthetic_hl7_message(
    message_control_id: str,
    order_id: str,
    report_id: str,
    subject_id: str,
    *,
    assay_code: str = "SYNTH-FLU",
    report_status: str = "F",
    qualitative_value: str = "Detected",
) -> bytes:
    identifiers = (
        (message_control_id, CLINICAL_SYNTHETIC_MESSAGE_ID_PATTERN),
        (order_id, CLINICAL_SYNTHETIC_ORDER_ID_PATTERN),
        (report_id, CLINICAL_SYNTHETIC_REPORT_ID_PATTERN),
        (subject_id, CLINICAL_SYNTHETIC_SUBJECT_ID_PATTERN),
    )
    if any(pattern.fullmatch(value) is None for value, pattern in identifiers):
        raise ControlError(
            "INVALID_SYNTHETIC_FIXTURE",
            "synthetic fixture identifiers must use the fixed three-digit grammar",
            EXIT_INPUT,
        )
    if assay_code not in CLINICAL_SYNTHETIC_ASSAY_CODES:
        raise ControlError(
            "INVALID_SYNTHETIC_FIXTURE",
            "synthetic fixture assay code is not an allowed test code",
            EXIT_INPUT,
        )
    if report_status not in {"F", "P"}:
        raise ControlError(
            "INVALID_SYNTHETIC_FIXTURE",
            "synthetic fixture report status must be F or P",
            EXIT_INPUT,
        )
    if qualitative_value not in {
        "Aborted", "Detected", "Indeterminate", "Invalid", "Not detected"
    }:
        raise ControlError(
            "INVALID_SYNTHETIC_FIXTURE",
            "synthetic fixture qualitative value is not allowed",
            EXIT_INPUT,
        )
    msh = ["MSH"] + [""] * 17
    msh[1] = "^~\\&"
    msh[2] = "LIAT-SIM"
    msh[3] = "LAB-SIM"
    msh[6] = "20260813050000-0400"
    msh[8] = "ORU^R30^ORU_R30"
    msh[9] = message_control_id
    msh[10] = "P"
    msh[11] = "2.5"
    msh[17] = "UNICODE UTF-8"
    pid = ["PID"] + [""] * 3
    pid[3] = subject_id
    orc = ["ORC", "NW"]
    obr = ["OBR"] + [""] * 34
    obr[1] = "1"
    obr[2] = order_id
    obr[3] = report_id
    obr[4] = assay_code + "^Synthetic influenza assay^urn:synthetic:assay"
    obr[7] = "20260813045900-0400"
    obr[11] = "O"
    obr[22] = "20260813050000-0400"
    obr[25] = report_status
    obr[32] = "SYNTH-APPROVER"
    obr[34] = "SYNTH-TECH"
    obx_numeric = ["OBX"] + [""] * 11
    obx_numeric[1] = "1"
    obx_numeric[2] = "NM"
    obx_numeric[3] = "SYNTH-FLUA^Synthetic influenza A^urn:synthetic:observation"
    obx_numeric[5] = "0"
    obx_numeric[6] = "0"
    obx_numeric[11] = report_status
    obx_interpretation = ["OBX"] + [""] * 11
    obx_interpretation[1] = "2"
    obx_interpretation[2] = "ST"
    obx_interpretation[3] = "SYNTH-FLUA^Synthetic influenza A^urn:synthetic:observation"
    obx_interpretation[5] = qualitative_value
    obx_interpretation[11] = report_status
    obx_ct = ["OBX"] + [""] * 11
    obx_ct[1] = "3"
    obx_ct[2] = "NM"
    obx_ct[3] = "SYNTH-FLUA-CT^Synthetic influenza A Ct^urn:synthetic:observation^S_OTHER^Synthetic Supplemental^IHE LPOCT"
    obx_ct[5] = "28.75" if qualitative_value == "Detected" else ""
    obx_ct[11] = report_status
    result_nte = [
        "NTE",
        "1",
        "L",
        "Run=SYNTH-00001;Device=SYNTH-DEVICE;Version=3.5.0-SYNTH;"
        "Tube=SYNTH-TUBE;TubeExp=2099-01-01;TubeLot=SYNTH-LOT",
    ]
    observation_nte = ["NTE", "1", "L", "SYNTHETIC DATA - NOT FOR CLINICAL USE"]
    payload = "\r".join(
        "|".join(segment)
        for segment in (
            msh,
            pid,
            orc,
            obr,
            result_nte,
            obx_numeric,
            observation_nte,
            obx_interpretation,
            obx_ct,
        )
    ) + "\r"
    return b"\x0b" + payload.encode("utf-8") + b"\x1c\r"


def synthetic_binding(
    subject_id: str,
    order_id: str,
    suffix: str,
    *,
    supersedes_result_id: str | None = None,
) -> dict[str, Any]:
    if (
        CLINICAL_SYNTHETIC_SUBJECT_ID_PATTERN.fullmatch(subject_id) is None
        or CLINICAL_SYNTHETIC_ORDER_ID_PATTERN.fullmatch(order_id) is None
        or re.fullmatch(r"[0-9]{3}", suffix) is None
    ):
        raise ControlError(
            "INVALID_SYNTHETIC_FIXTURE",
            "synthetic binding values must use the fixed three-digit grammar",
            EXIT_INPUT,
        )
    return {
        "deidentified": True,
        "order_reference": f"ServiceRequest/SYNTH-ORDER-{suffix}",
        "patient_reference": f"Patient/SYNTH-PATIENT-{suffix}",
        "recipient_id": "SYNTH-OFFLINE-RECIPIENT",
        "source_order_token": order_id,
        "source_subject_token": subject_id,
        "specimen_reference": f"Specimen/SYNTH-SPECIMEN-{suffix}",
        "supersedes_result_id": supersedes_result_id,
        "synthetic": True,
    }


def clinical_self_test() -> dict[str, Any]:
    require_cryptography()
    checks: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(
        prefix="owned-agent-clinical-self-test-", ignore_cleanup_errors=True
    ) as directory:
        paths = state_paths(Path(directory).resolve())
        clinical_initialize(paths)
        assay_map_path = paths.root / "assay-map.json"
        atomic_write(
            assay_map_path,
            (
                pretty_json(
                    {
                        "SYNTH-FLU": {
                            "display": "Synthetic influenza assay",
                            "local_system": "urn:synthetic:assay",
                        }
                    }
                )
                + "\n"
            ).encode("utf-8"),
        )
        clinical_add_source(
            paths,
            "synthetic-liat",
            "MOCK",
            CLINICAL_PROFILE_ID,
            "LIAT-SIM",
            "LAB-SIM",
            assay_map_path,
        )
        reviewer_key = Ed25519PrivateKey.generate()
        reviewer_public_path = paths.root / "synthetic-reviewer-public.pem"
        atomic_write(
            reviewer_public_path,
            reviewer_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ),
        )
        clinical_add_reviewer_raw(paths, "synthetic_reviewer", reviewer_key.public_key(), actor="self-test")
        clinical_seal_reviewer_trust(paths)

        hl7_path = paths.root / "synthetic-result.hl7"
        binding_path = paths.root / "synthetic-binding.json"
        raw = synthetic_hl7_message(
            "SYNTH-MSG-001", "SYNTH-SOURCE-ORDER-001", "SYNTH-REPORT-001", "SYNTH-SUBJECT-001"
        )
        atomic_write(hl7_path, raw)
        atomic_write(
            binding_path,
            (
                pretty_json(
                    synthetic_binding("SYNTH-SUBJECT-001", "SYNTH-SOURCE-ORDER-001", "001")
                )
                + "\n"
            ).encode("utf-8"),
        )
        ingested = clinical_ingest(paths, "synthetic-liat", hl7_path, binding_path)
        checks["ingest_state"] = ingested["state"]
        retry = clinical_ingest(paths, "synthetic-liat", hl7_path, binding_path)
        checks["idempotent_retry"] = retry["idempotent"]

        conflict_path = paths.root / "synthetic-conflict.hl7"
        atomic_write(
            conflict_path,
            synthetic_hl7_message(
                "SYNTH-MSG-001",
                "SYNTH-SOURCE-ORDER-001",
                "SYNTH-REPORT-001",
                "SYNTH-SUBJECT-001",
                qualitative_value="Not detected",
            ),
        )
        try:
            clinical_ingest(paths, "synthetic-liat", conflict_path, binding_path)
        except ControlError as exc:
            checks["control_id_conflict"] = exc.code
        else:
            raise ControlError("CLINICAL_SELF_TEST_FAILED", "control-ID conflict was accepted", EXIT_INTERNAL)

        request = build_clinical_review(
            paths, str(ingested["result_id"]), "synthetic_reviewer", MAX_TTL_SECONDS
        )
        signed = sign_clinical_review(paths, request, reviewer_key)
        applied = apply_clinical_review(paths, signed)
        checks["review_state"] = applied["state"]
        export_path = paths.root / "synthetic-fhir.json"
        exported = clinical_export_fhir(paths, str(ingested["result_id"]), export_path)
        checks["export_state"] = exported["state"]
        verified_export = verify_clinical_export_authorization(
            export_path,
            Path(str(exported["authorization_manifest_out"])),
            reviewer_public_path,
        )
        checks["export_authorization_verified"] = bool(
            verified_export["artifact_exactly_bound"]
            and verified_export["signature_cryptographically_valid"]
            and verified_export["trusted_reviewer_key_matched"]
        )
        bundle = load_json_file(export_path)
        checks["fhir_resource_type"] = bundle.get("resourceType") if isinstance(bundle, dict) else None
        checks["fhir_truth_tag"] = (
            bundle.get("meta", {}).get("tag", [])[1].get("code")
            if isinstance(bundle, dict)
            else None
        )

        unknown_path = paths.root / "synthetic-unknown.hl7"
        unknown_binding_path = paths.root / "synthetic-unknown-binding.json"
        atomic_write(
            unknown_path,
            synthetic_hl7_message(
                "SYNTH-MSG-003",
                "SYNTH-SOURCE-ORDER-003",
                "SYNTH-REPORT-003",
                "SYNTH-SUBJECT-003",
                assay_code="SYNTH-UNMAPPED",
            ),
        )
        atomic_write(
            unknown_binding_path,
            (
                pretty_json(
                    synthetic_binding(
                        "SYNTH-SUBJECT-003", "SYNTH-SOURCE-ORDER-003", "003"
                    )
                )
                + "\n"
            ).encode("utf-8"),
        )
        unknown = clinical_ingest(paths, "synthetic-liat", unknown_path, unknown_binding_path)
        checks["unknown_assay_state"] = unknown["state"]
        checks["unknown_assay_gate"] = "ASSAY_MAP_MATCH" in unknown["failed_gates"]

        corrected_path = paths.root / "synthetic-corrected.hl7"
        corrected_binding_path = paths.root / "synthetic-corrected-binding.json"
        atomic_write(
            corrected_path,
            synthetic_hl7_message(
                "SYNTH-MSG-002",
                "SYNTH-SOURCE-ORDER-001",
                "SYNTH-REPORT-001",
                "SYNTH-SUBJECT-001",
                report_status="F",
                qualitative_value="Not detected",
            ),
        )
        atomic_write(
            corrected_binding_path,
            (
                pretty_json(
                    synthetic_binding(
                        "SYNTH-SUBJECT-001",
                        "SYNTH-SOURCE-ORDER-001",
                        "001",
                        supersedes_result_id=str(ingested["result_id"]),
                    )
                )
                + "\n"
            ).encode("utf-8"),
        )
        corrected = clinical_ingest(
            paths, "synthetic-liat", corrected_path, corrected_binding_path
        )
        corrected_request = build_clinical_review(
            paths, str(corrected["result_id"]), "synthetic_reviewer", MAX_TTL_SECONDS
        )
        corrected_signed = sign_clinical_review(paths, corrected_request, reviewer_key)
        apply_clinical_review(paths, corrected_signed)
        clinical_export_fhir(
            paths, str(corrected["result_id"]), paths.root / "synthetic-corrected-fhir.json"
        )
        corrected_status = clinical_status(paths, str(corrected["result_id"]))
        original_status = clinical_status(paths, str(ingested["result_id"]))
        checks["corrected_state"] = corrected_status["state"]
        checks["original_superseded"] = original_status["state"]

        with clinical_connect(paths) as connection:
            ledger = clinical_verify_ledger(connection)
        checks["ledger_integrity"] = ledger["integrity"]

        expected = {
            "control_id_conflict": "MESSAGE_CONTROL_CONFLICT",
            "corrected_state": "ARTIFACT_CREATED",
            "export_state": "ARTIFACT_CREATED",
            "export_authorization_verified": True,
            "fhir_resource_type": "Bundle",
            "fhir_truth_tag": "not-delivered",
            "idempotent_retry": True,
            "ingest_state": "PENDING_REVIEW",
            "ledger_integrity": "VERIFIED_LOCAL_CLINICAL_HASH_CHAIN",
            "original_superseded": "SUPERSEDED",
            "review_state": "AUTHORIZED_FOR_EXPORT",
            "unknown_assay_gate": True,
            "unknown_assay_state": "QUARANTINED",
        }
        for name, expected_value in expected.items():
            if checks.get(name) != expected_value:
                raise ControlError(
                    "CLINICAL_SELF_TEST_FAILED",
                    f"{name} expected {expected_value!r}, observed {checks.get(name)!r}",
                    EXIT_INTERNAL,
                )
    return {
        "checks": checks,
        "clinical_use_authorized": False,
        "device_control": False,
        "direct_device_transport": False,
        "ok": True,
        "operation": "clinical-self-test",
        "operation_status": "VERIFIED_SYNTHETIC_OFFLINE_CLINICAL_PATH",
        "truth_boundary": "synthetic_offline_artifact_not_clinical_delivery",
    }


def require_state(args: argparse.Namespace) -> StatePaths:
    value = getattr(args, "state_dir", None)
    if not value:
        raise ControlError("STATE_DIR_REQUIRED", "--state-dir is required", EXIT_INPUT)
    return state_paths(value)


def build_parser(*, include_internal: bool = False) -> argparse.ArgumentParser:
    parser = StrictArgumentParser(
        description=(
            "Real stop-only control for local Windows process trees launched by this supervisor. "
            "No provider, network-policy, or remote-credential enforcement is claimed."
        )
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {PROGRAM_VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def state_command(name: str, help_text: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name, help=help_text)
        child.add_argument("--state-dir", required=True, type=Path)
        return child

    state_command("init", "initialize protected controller state")

    keygen_parser = subparsers.add_parser("keygen", help="create an encrypted Ed25519 operator key")
    keygen_parser.add_argument("--operator", required=True)
    keygen_parser.add_argument("--private-key", required=True, type=Path)
    keygen_parser.add_argument("--public-key", required=True, type=Path)
    keygen_parser.add_argument("--passphrase-env")

    operator_parser = state_command("operator-add", "trust one Ed25519 operator public key")
    operator_parser.add_argument("--operator", required=True)
    operator_parser.add_argument("--public-key", required=True, type=Path)

    state_command("trust-seal", "irreversibly close operator registration for this state root")

    register_parser = state_command("register", "register an explicit argv-list target")
    register_parser.add_argument("--target", required=True)
    register_parser.add_argument("--cwd", required=True, type=Path)
    register_parser.add_argument("command_argv", nargs=argparse.REMAINDER)

    demo_parser = state_command("register-demo", "register the harmless built-in parent/child target")
    demo_parser.add_argument("--target", default="owned-agent:demo")

    start_parser = state_command("start", "start one registered target under a Job Object")
    start_parser.add_argument("--target", required=True)
    start_parser.add_argument("--timeout", type=float, default=15.0)

    status_parser = state_command("status", "show durable and last-observed target state")
    status_parser.add_argument("--target", required=True)

    request_parser = state_command("request-new", "create an unsigned isolation request")
    request_parser.add_argument("--target", required=True)
    request_parser.add_argument("--ttl", type=int, default=120)
    request_parser.add_argument("--out", required=True, type=Path)

    sign_parser = state_command("request-sign", "add exactly one Ed25519 operator signature")
    sign_parser.add_argument("--request", required=True, type=Path)
    sign_parser.add_argument("--operator", required=True)
    sign_parser.add_argument("--private-key", required=True, type=Path)
    sign_parser.add_argument("--out", required=True, type=Path)
    sign_parser.add_argument("--passphrase-env")

    verify_parser = state_command("request-verify", "verify authorization without consuming it")
    verify_parser.add_argument("--request", required=True, type=Path)

    apply_parser = state_command("apply-isolation", "consume authorization and enforce isolation")
    apply_parser.add_argument("--request", required=True, type=Path)
    apply_parser.add_argument("--timeout", type=float, default=15.0)

    credential_parser = state_command(
        "credential-check", "validate a supervisor-local generation token read from stdin"
    )
    credential_parser.add_argument("--target", required=True)

    state_command("audit-verify", "verify the local hash-chained audit ledger")

    state_command(
        "clinical-init",
        "initialize the MOCK and deidentified LIVE_SHADOW offline result scaffold",
    )

    source_parser = state_command(
        "clinical-source-add",
        "add a clinical source profile for mocked or live transport ingestion",
    )
    source_parser.add_argument("--source-id", required=True)
    source_parser.add_argument("--mode", required=True, choices=sorted(CLINICAL_MODES))
    source_parser.add_argument("--profile-id", default=CLINICAL_PROFILE_ID)
    source_parser.add_argument("--sender-application", required=True)
    source_parser.add_argument("--sender-facility", required=True)
    source_parser.add_argument("--assay-map", required=True, type=Path)

    reviewer_parser = state_command(
        "clinical-reviewer-add",
        "trust one clinical-review Ed25519 public key in a separate signature domain",
    )
    reviewer_parser.add_argument("--reviewer", required=True)
    reviewer_parser.add_argument("--public-key", required=True, type=Path)

    state_command(
        "clinical-trust-seal",
        "irreversibly close clinical reviewer registration for this state root",
    )

    ingest_parser = state_command(
        "clinical-ingest",
        "ingest an HL7 result payload (MOCK mode enforces strict synthetic fixture shape)",
    )
    ingest_parser.add_argument("--source-id", required=True)
    ingest_parser.add_argument("--hl7-file", required=True, type=Path)
    ingest_parser.add_argument("--binding", required=True, type=Path)

    clinical_status_parser = state_command(
        "clinical-status",
        "show redacted hashes, gates, and lifecycle state for one result version",
    )
    clinical_status_parser.add_argument("--result-id", required=True)

    review_new_parser = state_command(
        "clinical-review-new",
        "create an unsigned exact-version authorization for a not-delivered FHIR candidate",
    )
    review_new_parser.add_argument("--result-id", required=True)
    review_new_parser.add_argument("--reviewer", required=True)
    review_new_parser.add_argument("--ttl", type=int, default=120)
    review_new_parser.add_argument("--out", required=True, type=Path)

    review_sign_parser = state_command(
        "clinical-review-sign",
        "sign one offline-export review with a dedicated reviewer key",
    )
    review_sign_parser.add_argument("--request", required=True, type=Path)
    review_sign_parser.add_argument("--private-key", required=True, type=Path)
    review_sign_parser.add_argument("--out", required=True, type=Path)
    review_sign_parser.add_argument("--passphrase-env")

    review_apply_parser = state_command(
        "clinical-review-apply",
        "verify and consume a signed review; no external result delivery occurs",
    )
    review_apply_parser.add_argument("--request", required=True, type=Path)

    export_parser = state_command(
        "clinical-export-fhir",
        "copy out the exact signed, not-delivered FHIR R4 candidate",
    )
    export_parser.add_argument("--result-id", required=True)
    export_parser.add_argument("--out", required=True, type=Path)

    export_verify_parser = subparsers.add_parser(
        "clinical-export-verify",
        help="verify exact FHIR bytes and detached reviewer-key evidence against a trusted public key",
    )
    export_verify_parser.add_argument("--artifact", required=True, type=Path)
    export_verify_parser.add_argument(
        "--authorization-manifest", required=True, type=Path
    )
    export_verify_parser.add_argument(
        "--trusted-reviewer-key", required=True, type=Path
    )

    state_command("clinical-ledger-verify", "verify the local append-only clinical hash chain")
    subparsers.add_parser(
        "clinical-capabilities",
        help="show core, bridge, PHI, site-validation, and clinical-use boundaries",
    )
    subparsers.add_parser(
        "clinical-self-test",
        help="run a disposable synthetic ingest/review/export/correction test",
    )

    subparsers.add_parser("self-test", help="run a disposable real parent/child isolation test")

    if include_internal:
        supervisor_parser = subparsers.add_parser("_supervisor", help=argparse.SUPPRESS)
        supervisor_parser.add_argument("--state-dir", required=True, type=Path)
        supervisor_parser.add_argument("--run-id", required=True)

        demo_target_parser = subparsers.add_parser("_demo-target", help=argparse.SUPPRESS)
        demo_target_parser.add_argument("--state-dir", required=True, type=Path)
        demo_target_parser.add_argument("--heartbeat", required=True, type=Path)
        demo_target_parser.add_argument("--child-record", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argument_list = list(argv) if argv is not None else sys.argv[1:]
    include_internal = bool(
        argument_list and argument_list[0] in {"_supervisor", "_demo-target"}
    )
    args = build_parser(include_internal=include_internal).parse_args(argument_list)
    try:
        if args.command == "init":
            emit(initialize_state(require_state(args)))
            return EXIT_OK
        if args.command == "keygen":
            passphrase = read_passphrase(args.passphrase_env, confirm=True)
            emit(
                generate_operator_keypair(
                    args.operator, args.private_key, args.public_key, passphrase
                )
            )
            return EXIT_OK
        if args.command == "operator-add":
            emit(add_operator(require_state(args), args.operator, args.public_key))
            return EXIT_OK
        if args.command == "trust-seal":
            emit(seal_trust_store(require_state(args)))
            return EXIT_OK
        if args.command == "register":
            command = list(args.command_argv)
            if command and command[0] == "--":
                command = command[1:]
            emit(register_agent(require_state(args), args.target, command, args.cwd))
            return EXIT_OK
        if args.command == "register-demo":
            emit(register_demo(require_state(args), args.target))
            return EXIT_OK
        if args.command == "start":
            emit(start_agent(require_state(args), args.target, args.timeout))
            return EXIT_OK
        if args.command == "status":
            emit(target_status(require_state(args), args.target))
            return EXIT_OK
        if args.command == "request-new":
            request = build_request(require_state(args), args.target, args.ttl)
            atomic_write(args.out, (pretty_json(request) + "\n").encode("utf-8"))
            emit(
                {
                    "ok": True,
                    "operation": "request-new",
                    "out": str(args.out.resolve()),
                    "request_id": request["request_id"],
                    "signatures": 0,
                    "target": request["target"],
                }
            )
            return EXIT_OK
        if args.command == "request-sign":
            request = load_json_file(args.request)
            if not isinstance(request, dict):
                raise ControlError("INVALID_ENVELOPE", "request must be a JSON object", EXIT_INPUT)
            passphrase = read_passphrase(args.passphrase_env, confirm=False)
            private_key = load_private_key(args.private_key, passphrase)
            signed = sign_request_with_key(
                require_state(args), request, args.operator, private_key
            )
            atomic_write(args.out, (pretty_json(signed) + "\n").encode("utf-8"))
            emit(
                {
                    "ok": True,
                    "operation": "request-sign",
                    "operator_id": args.operator,
                    "out": str(args.out.resolve()),
                    "request_id": signed["request_id"],
                    "signature_count": len(signed["authorization"]["signatures"]),
                }
            )
            return EXIT_OK
        if args.command == "request-verify":
            request = load_json_file(args.request)
            if not isinstance(request, dict):
                raise ControlError("INVALID_ENVELOPE", "request must be a JSON object", EXIT_INPUT)
            result = verify_request(require_state(args), request)
            result.update({"ok": True, "operation": "request-verify"})
            emit(result)
            return EXIT_OK
        if args.command == "apply-isolation":
            request = load_json_file(args.request)
            if not isinstance(request, dict):
                raise ControlError("INVALID_ENVELOPE", "request must be a JSON object", EXIT_INPUT)
            emit(apply_isolation(require_state(args), request, args.timeout))
            return EXIT_OK
        if args.command == "credential-check":
            token = sys.stdin.readline(4097).rstrip("\r\n")
            result = credential_check(require_state(args), args.target, token)
            emit(result)
            return EXIT_OK if result["credential_valid"] else EXIT_DENIED
        if args.command == "clinical-init":
            emit(clinical_initialize(require_state(args)))
            return EXIT_OK
        if args.command == "clinical-source-add":
            emit(
                clinical_add_source(
                    require_state(args),
                    args.source_id,
                    args.mode,
                    args.profile_id,
                    args.sender_application,
                    args.sender_facility,
                    args.assay_map,
                )
            )
            return EXIT_OK
        if args.command == "clinical-reviewer-add":
            emit(
                clinical_add_reviewer(
                    require_state(args), args.reviewer, args.public_key
                )
            )
            return EXIT_OK
        if args.command == "clinical-trust-seal":
            emit(clinical_seal_reviewer_trust(require_state(args)))
            return EXIT_OK
        if args.command == "clinical-ingest":
            emit(
                clinical_ingest(
                    require_state(args), args.source_id, args.hl7_file, args.binding
                )
            )
            return EXIT_OK
        if args.command == "clinical-status":
            emit(clinical_status(require_state(args), args.result_id))
            return EXIT_OK
        if args.command == "clinical-review-new":
            clinical_paths = require_state(args)
            review = build_clinical_review(
                clinical_paths, args.result_id, args.reviewer, args.ttl
            )
            review_out = clinical_write_output(
                clinical_paths,
                args.out,
                (pretty_json(review) + "\n").encode("utf-8"),
                "unsigned clinical review",
            )
            emit(
                {
                    "ok": True,
                    "operation": "clinical-review-new",
                    "out": str(review_out),
                    "result_id": review["result_id"],
                    "review_id": review["review_id"],
                }
            )
            return EXIT_OK
        if args.command == "clinical-review-sign":
            clinical_paths = require_state(args)
            review = load_json_file(args.request)
            if not isinstance(review, dict):
                raise ControlError(
                    "INVALID_CLINICAL_REVIEW", "review request must be a JSON object", EXIT_INPUT
                )
            passphrase = read_passphrase(args.passphrase_env, confirm=False)
            private_key = load_private_key(args.private_key, passphrase)
            signed = sign_clinical_review(clinical_paths, review, private_key)
            signed_out = clinical_write_output(
                clinical_paths,
                args.out,
                (pretty_json(signed) + "\n").encode("utf-8"),
                "signed clinical review",
            )
            emit(
                {
                    "ok": True,
                    "operation": "clinical-review-sign",
                    "out": str(signed_out),
                    "result_id": signed["result_id"],
                    "review_id": signed["review_id"],
                }
            )
            return EXIT_OK
        if args.command == "clinical-review-apply":
            review = load_json_file(args.request)
            if not isinstance(review, dict):
                raise ControlError(
                    "INVALID_CLINICAL_REVIEW", "signed review must be a JSON object", EXIT_INPUT
                )
            emit(apply_clinical_review(require_state(args), review))
            return EXIT_OK
        if args.command == "clinical-export-fhir":
            emit(
                clinical_export_fhir(
                    require_state(args), args.result_id, args.out
                )
            )
            return EXIT_OK
        if args.command == "clinical-export-verify":
            emit(
                verify_clinical_export_authorization(
                    args.artifact,
                    args.authorization_manifest,
                    args.trusted_reviewer_key,
                )
            )
            return EXIT_OK
        if args.command == "clinical-ledger-verify":
            with clinical_connect(require_state(args)) as connection:
                result = clinical_verify_ledger(connection)
            result["ok"] = True
            result["operation"] = "clinical-ledger-verify"
            emit(result)
            return EXIT_OK
        if args.command == "clinical-capabilities":
            emit(clinical_capabilities())
            return EXIT_OK
        if args.command == "clinical-self-test":
            emit(clinical_self_test())
            return EXIT_OK
        if args.command == "audit-verify":
            with connect(require_state(args)) as connection:
                result = verify_audit(connection)
            result["operation"] = "audit-verify"
            emit(result)
            return EXIT_OK
        if args.command == "self-test":
            emit(self_test())
            return EXIT_OK
        if args.command == "_supervisor":
            return run_supervisor(state_paths(args.state_dir.resolve()), args.run_id)
        if args.command == "_demo-target":
            return run_demo_target(
                state_paths(args.state_dir.resolve()),
                args.heartbeat.resolve(),
                args.child_record.resolve(),
            )
        raise ControlError("UNKNOWN_COMMAND", "command was not handled", EXIT_INPUT)
    except ControlError as exc:
        emit_error(exc.code, exc.message, exc.exit_code)
    except KeyboardInterrupt:
        emit_error("INTERRUPTED", "operation interrupted", EXIT_INTERNAL)
    except BaseException as exc:
        emit_error("INTERNAL_ERROR", repr(exc), EXIT_INTERNAL)
    return EXIT_INTERNAL


if __name__ == "__main__":
    raise SystemExit(main())
