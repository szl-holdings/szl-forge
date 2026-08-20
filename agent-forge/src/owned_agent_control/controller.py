#!/usr/bin/env python3
"""Owned Agent Control v2.0 — Enriched Context Generation Edition.

Enforcement boundary:
  * launches only explicitly registered local executables;
  * assigns the launched process to a Windows Job Object before it can run;
  * terminates the supervised process tree with entropy-aware budgeting;
  * durably blocks later starts for the registered owned-agent ID;
  * verifies short-lived Ed25519 quorum authorization and one-shot replay state;
  * records a locally tamper-evident (not immutable) hash-chained audit ledger;
  * monitors cross-step consistency and entropy depth allocation.

It does NOT control external providers, remote credentials, network policy,
messages, model state, containers, VMs, services, or processes it did not start.

New in v2.0:
  - EnrichedContextGenerator for diagnostic system awareness;
  - EntropyDepthAllocator integration with configurable budget (default: 3);
  - CrossStepConsistency checks with threshold (default: 0.95);
  - LoopKernel_State_Transition support for multi-step workflows;
  - PolicyCheck gates: Safety_Gate, Context_Reweight, Output_Filter_Bypass.
"""

from __future__ import annotations

import argparse
import base64
import copy
from contextlib import ExitStack
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import getpass
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
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
PROGRAM_VERSION = "2.0.0"
STATE_SCHEMA_VERSION = "2"
ENVELOPE_SCHEMA = "owned-agent-isolation/v2"
ENVELOPE_ACTION = "isolate_owned_agent"
SIGNATURE_SCHEME = "Ed25519"
SIGNATURE_DOMAIN = b"owned-agent-control/isolate/v2\x00"
QUORUM = 2
MAX_TTL_SECONDS = 300
MAX_CLOCK_SKEW_SECONDS = 30
MAX_JSON_BYTES = 64 * 1024
MAX_JSON_DEPTH = 20
MAX_SIGNATURES = 8
MAX_CONTEXT_STEPS = 16
ZERO_HASH = "0" * 64

# New v2.0 constants for enriched context generation
DEFAULT_ENTROPY_BUDGET = 3
DEFAULT_CONSISTENCY_THRESHOLD_PPM = 950_000
CONTEXT_EVIDENCE_SCHEMA = "a11oy/owned-agent-context-evidence/v1"
LOOP_KERNEL_STATE_TRANSITION = "LoopKernel_State_Transition"
ENTROPY_DEPTH_ALLOCATOR = "EntropyDepthAllocator"
CROSSSTEP_CONSISTENCY = "CrossStepConsistency"

# Policy check types for safety gates
POLICY_CHECK_TYPES = {
    "Safety_Gate_Evaluated",
    "Context_Reweight_Flagged",
    "Output_Filter_Triggered",
}

TARGET_PATTERN = re.compile(r"^owned-agent:[a-z0-9][a-z0-9._-]{1,127}$")
OPERATOR_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")
B64URL_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
UTC_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)

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
    "context_generation": True,  # New v2.0 flag
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
CONTEXT_TRACE_ID_ENV = "OAC_CONTEXT_TRACE_ID"
PASSPHRASE_ENV = "OAC_KEY_PASSPHRASE"


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


class ContextGenerationError(ControlError):
    """New v2.0: Error for enriched context generation failures."""

    def __init__(self, code: str, message: str, exit_code: int = EXIT_DENIED):
        super().__init__(code, message, exit_code)


class StrictArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        emit_error("CLI_USAGE_ERROR", message, EXIT_INPUT)


def require_cryptography() -> None:
    if CRYPTOGRAPHY_IMPORT_ERROR is not None:
        raise ControlError(
            "DEPENDENCY_MISSING",
            "install the pinned dependency with: python -m pip install 'cryptography>=46,<51'",
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
    except AttributeError:
        return path.is_symlink()
    except OSError as exc:
        raise ControlError(
            "PATH_UNREADABLE", f"unable to inspect reparse attributes: {path}", EXIT_DENIED
        ) from exc
    return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def require_no_reparse_ancestors(
    path: Path, *, allow_missing_suffix: bool = False, label: str = "path"
) -> None:
    """Reject symlinks/junctions in every existing component without resolving them."""

    normalized = Path(os.path.abspath(Path(path).expanduser()))
    if not normalized.is_absolute():
        raise ControlError("PATH_NOT_ABSOLUTE", f"{label} must be absolute", EXIT_INPUT)
    raw = str(normalized)
    if os.name == "nt" and (raw.startswith("\\\\") or raw.startswith("\\?\\")):
        raise ControlError("PATH_NOT_LOCAL", f"{label} must be a local DOS path", EXIT_DENIED)
    current = Path(normalized.anchor)
    relative_parts = normalized.parts[1:] if normalized.anchor else normalized.parts
    for part in relative_parts:
        current = current / part
        try:
            info = os.stat(current, follow_symlinks=False)
        except FileNotFoundError as exc:
            if allow_missing_suffix:
                return
            raise ControlError(
                "PATH_NOT_FOUND", f"{label} component does not exist: {current}", EXIT_DENIED
            ) from exc
        except OSError as exc:
            raise ControlError(
                "PATH_UNREADABLE", f"unable to inspect {label} component: {current}", EXIT_DENIED
            ) from exc
        file_attributes = int(getattr(info, "st_file_attributes", 0))
        reparse_attribute = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
        if stat.S_ISLNK(info.st_mode) or (
            reparse_attribute and file_attributes & reparse_attribute
        ):
            raise ControlError(
                "REPARSE_POINT_DENIED", f"{label} contains a reparse point: {current}", EXIT_DENIED
            )


def ensure_state_root(paths: StatePaths, *, create: bool = False) -> None:
    if create:
        require_no_reparse_ancestors(
            paths.root, allow_missing_suffix=True, label="state directory"
        )
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
    require_no_reparse_ancestors(paths.root, label="state directory")
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


class ClosingConnection(sqlite3.Connection):
    """Commit or roll back like sqlite3.Connection, then always close the handle."""

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc, tb))
        finally:
            self.close()


def connect(paths: StatePaths) -> ClosingConnection:
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
        factory=ClosingConnection,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.execute("PRAGMA synchronous = FULL")
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


STATE_SCHEMA_SQL = """
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
    argv_json TEXT NOT NULL CHECK(json_valid(argv_json)),
    cwd TEXT NOT NULL,
    executable_sha256 TEXT NOT NULL CHECK(length(executable_sha256)=64),
    control_state TEXT NOT NULL CHECK(control_state IN ('READY', 'ISOLATED')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
) STRICT;

CREATE TABLE context_traces (
    trace_id TEXT PRIMARY KEY,
    target TEXT NOT NULL REFERENCES targets(target),
    challenge_question INTEGER NOT NULL,
    execution_trace_json TEXT NOT NULL CHECK(json_valid(execution_trace_json)),
    convergence_status TEXT NOT NULL
        CHECK(convergence_status IN ('Stabilized', 'Flagged_For_Review')),
    final_state_json TEXT NOT NULL CHECK(json_valid(final_state_json)),
    binding_json TEXT NOT NULL CHECK(json_valid(binding_json)),
    entropy_budget INTEGER NOT NULL CHECK(entropy_budget BETWEEN 1 AND 16),
    entropy_used INTEGER NOT NULL CHECK(entropy_used BETWEEN 0 AND entropy_budget),
    consistency_score_ppm INTEGER NOT NULL
        CHECK(consistency_score_ppm BETWEEN 0 AND 1000000),
    consistency_threshold_ppm INTEGER NOT NULL
        CHECK(consistency_threshold_ppm BETWEEN 950000 AND 1000000),
    threshold_met INTEGER NOT NULL CHECK(threshold_met IN (0, 1)),
    evidence_sha256 TEXT NOT NULL UNIQUE CHECK(length(evidence_sha256)=64),
    created_at TEXT NOT NULL,
    UNIQUE(trace_id, target)
) STRICT;

CREATE TABLE policy_checks (
    check_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL REFERENCES context_traces(trace_id),
    check_type TEXT NOT NULL CHECK(check_type IN (
        'Safety_Gate_Evaluated', 'Context_Reweight_Flagged', 'Output_Filter_Triggered'
    )),
    status TEXT NOT NULL CHECK(status IN ('Pass', 'Flagged', 'Triggered')),
    details_json TEXT NOT NULL CHECK(json_valid(details_json)),
    created_at TEXT NOT NULL
) STRICT;
CREATE UNIQUE INDEX one_policy_type_per_trace ON policy_checks(trace_id, check_type);

CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    target TEXT NOT NULL REFERENCES targets(target),
    context_trace_id TEXT NOT NULL UNIQUE,
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
    created_at TEXT NOT NULL,
    CHECK(
        (supervisor_pid IS NULL AND supervisor_created_filetime IS NULL) OR
        (supervisor_pid > 0 AND supervisor_created_filetime > 0)
    ),
    FOREIGN KEY(context_trace_id, target)
        REFERENCES context_traces(trace_id, target)
) STRICT;

CREATE UNIQUE INDEX one_live_run_per_target ON runs(target)
WHERE state IN ('STARTING', 'RUNNING', 'ISOLATING');

CREATE TABLE requests (
    request_id TEXT PRIMARY KEY,
    nonce TEXT NOT NULL UNIQUE,
    envelope_sha256 TEXT NOT NULL UNIQUE CHECK(length(envelope_sha256)=64),
    envelope_json TEXT NOT NULL CHECK(json_valid(envelope_json)),
    target TEXT NOT NULL REFERENCES targets(target),
    status TEXT NOT NULL CHECK(status IN (
        'ENFORCING', 'APPLIED', 'ENFORCEMENT_UNCONFIRMED'
    )),
    accepted_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    applied_at TEXT,
    result_json TEXT CHECK(result_json IS NULL OR json_valid(result_json))
) STRICT;

CREATE UNIQUE INDEX one_enforcing_request_per_target ON requests(target)
WHERE status IN ('ENFORCING', 'ENFORCEMENT_UNCONFIRMED');

CREATE TABLE audit_events (
    sequence INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    target TEXT,
    actor TEXT NOT NULL,
    payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
    previous_hash TEXT NOT NULL CHECK(length(previous_hash)=64),
    event_hash TEXT NOT NULL UNIQUE CHECK(length(event_hash)=64)
) STRICT;

CREATE TRIGGER audit_events_no_update BEFORE UPDATE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit_events are append-only'); END;
CREATE TRIGGER audit_events_no_delete BEFORE DELETE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit_events are append-only'); END;
CREATE TRIGGER operators_no_update BEFORE UPDATE ON operators
BEGIN SELECT RAISE(ABORT, 'operators are immutable'); END;
CREATE TRIGGER operators_no_delete BEFORE DELETE ON operators
BEGIN SELECT RAISE(ABORT, 'operators are immutable'); END;
CREATE TRIGGER targets_no_unisolate BEFORE UPDATE OF control_state ON targets
WHEN OLD.control_state='ISOLATED' AND NEW.control_state<>'ISOLATED'
BEGIN SELECT RAISE(ABORT, 'target isolation is irreversible'); END;
CREATE TRIGGER targets_registration_no_update
BEFORE UPDATE OF target, argv_json, cwd, executable_sha256, created_at ON targets
BEGIN SELECT RAISE(ABORT, 'target registration is immutable'); END;
CREATE TRIGGER targets_no_delete BEFORE DELETE ON targets
BEGIN SELECT RAISE(ABORT, 'targets may not be deleted'); END;
CREATE TRIGGER requests_authorization_no_update
BEFORE UPDATE OF request_id, nonce, envelope_sha256, envelope_json, target, accepted_at, expires_at
ON requests BEGIN SELECT RAISE(ABORT, 'request authorization is immutable'); END;
CREATE TRIGGER requests_no_delete BEFORE DELETE ON requests
BEGIN SELECT RAISE(ABORT, 'requests may not be deleted'); END;
CREATE TRIGGER requests_status_monotonic BEFORE UPDATE OF status ON requests
WHEN NOT (
    NEW.status=OLD.status OR
    (OLD.status='ENFORCING' AND NEW.status IN ('APPLIED','ENFORCEMENT_UNCONFIRMED')) OR
    (OLD.status='ENFORCEMENT_UNCONFIRMED' AND NEW.status='APPLIED')
)
BEGIN SELECT RAISE(ABORT, 'request status may only advance'); END;
CREATE TRIGGER requests_applied_no_update BEFORE UPDATE ON requests
WHEN OLD.status='APPLIED'
BEGIN SELECT RAISE(ABORT, 'applied requests are immutable'); END;
CREATE TRIGGER runs_identity_no_update
BEFORE UPDATE OF run_id, target, context_trace_id, job_name, log_path, created_at ON runs
BEGIN SELECT RAISE(ABORT, 'run identity is immutable'); END;
CREATE TRIGGER runs_supervisor_identity_monotonic
BEFORE UPDATE OF supervisor_pid, supervisor_created_filetime ON runs
WHEN NOT (
    NEW.supervisor_pid IS OLD.supervisor_pid AND
    NEW.supervisor_created_filetime IS OLD.supervisor_created_filetime
) AND NOT (
    OLD.supervisor_pid IS NULL AND OLD.supervisor_created_filetime IS NULL AND
    NEW.supervisor_pid > 0 AND NEW.supervisor_created_filetime > 0
)
BEGIN SELECT RAISE(ABORT, 'supervisor identity may only be claimed once'); END;
CREATE TRIGGER runs_no_delete BEFORE DELETE ON runs
BEGIN SELECT RAISE(ABORT, 'runs may not be deleted'); END;
CREATE TRIGGER trust_store_no_unseal BEFORE UPDATE OF value ON metadata
WHEN OLD.key='trust_store_sealed' AND OLD.value='1' AND NEW.value<>'1'
BEGIN SELECT RAISE(ABORT, 'trust store sealing is irreversible'); END;
CREATE TRIGGER metadata_identity_no_update BEFORE UPDATE OF key, value ON metadata
WHEN OLD.key IN ('schema_version','controller_instance_id')
BEGIN SELECT RAISE(ABORT, 'controller identity metadata is immutable'); END;
CREATE TRIGGER metadata_no_delete BEFORE DELETE ON metadata
BEGIN SELECT RAISE(ABORT, 'controller metadata may not be deleted'); END;
CREATE TRIGGER context_traces_no_update BEFORE UPDATE ON context_traces
BEGIN SELECT RAISE(ABORT, 'context traces are append-only'); END;
CREATE TRIGGER context_traces_no_delete BEFORE DELETE ON context_traces
BEGIN SELECT RAISE(ABORT, 'context traces are append-only'); END;
CREATE TRIGGER policy_checks_no_update BEFORE UPDATE ON policy_checks
BEGIN SELECT RAISE(ABORT, 'policy checks are append-only'); END;
CREATE TRIGGER policy_checks_no_delete BEFORE DELETE ON policy_checks
BEGIN SELECT RAISE(ABORT, 'policy checks are append-only'); END;
"""

_EXPECTED_SCHEMA_OBJECTS: dict[tuple[str, str], str] | None = None


def _normalized_schema_objects(connection: sqlite3.Connection) -> dict[tuple[str, str], str]:
    return {
        (str(row["type"]), str(row["name"])): " ".join(
            str(row["sql"] or "").upper().split()
        )
        for row in connection.execute(
            """
            SELECT type, name, sql FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%' AND sql IS NOT NULL
            """
        )
    }


def _expected_schema_objects() -> dict[tuple[str, str], str]:
    global _EXPECTED_SCHEMA_OBJECTS
    if _EXPECTED_SCHEMA_OBJECTS is None:
        expected_connection = sqlite3.connect(":memory:")
        expected_connection.row_factory = sqlite3.Row
        try:
            expected_connection.execute("PRAGMA foreign_keys = ON")
            expected_connection.executescript(STATE_SCHEMA_SQL)
            _EXPECTED_SCHEMA_OBJECTS = _normalized_schema_objects(expected_connection)
        finally:
            expected_connection.close()
    return _EXPECTED_SCHEMA_OBJECTS


def create_state_database(
    database: Path, *, initialized_at: str | None = None
) -> dict[str, Any]:
    """Create a complete state database, removing every sidecar on failure."""

    database = Path(database)
    if database.exists():
        raise ControlError(
            "STATE_ALREADY_INITIALIZED", "control database already exists", EXIT_CONFLICT
        )
    controller_instance_id = str(uuid.uuid4())
    now = initialized_at or format_time(utc_now())
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(database, timeout=10.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(STATE_SCHEMA_SQL)
        with immediate_transaction(connection):
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                (
                    ("schema_version", STATE_SCHEMA_VERSION),
                    ("audit_tip", ZERO_HASH),
                    ("trust_store_sealed", "0"),
                    ("controller_instance_id", controller_instance_id),
                ),
            )
            append_audit(
                connection,
                "STATE_INITIALIZED",
                None,
                "local-operator",
                {
                    "controller_instance_id": controller_instance_id,
                    "context_generation_enabled": True,
                    "program": PROGRAM,
                    "program_version": PROGRAM_VERSION,
                    "schema_version": STATE_SCHEMA_VERSION,
                    "truth_boundary": "local_windows_supervised_processes_only",
                },
                timestamp=now,
            )
        verify_audit(connection)
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except BaseException:
        if connection is not None:
            connection.close()
            connection = None
        for suffix in ("", "-journal", "-wal", "-shm"):
            try:
                Path(f"{database}{suffix}").unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        if connection is not None:
            connection.close()
    return {
        "controller_instance_id": controller_instance_id,
        "database": str(database),
        "initialized_at": now,
    }


def initialize_state(paths: StatePaths) -> dict[str, Any]:
    if os.name != "nt":
        raise ControlError(
            "WINDOWS_REQUIRED",
            "the operational enforcement path requires Windows Job Objects",
            EXIT_INTERNAL,
        )
    root_existed = paths.root.exists()
    ensure_state_root(paths, create=True)
    staging = paths.root / f".control-{uuid.uuid4()}.sqlite3.tmp"
    created: dict[str, Any] | None = None
    acl_result: dict[str, Any] | None = None
    try:
        acl_result = _apply_owner_only_acl(paths.root)
        paths.logs.mkdir(exist_ok=True)
        paths.demo.mkdir(exist_ok=True)
        _apply_owner_only_acl(paths.logs)
        _apply_owner_only_acl(paths.demo)
        if paths.database.exists():
            raise ControlError(
                "STATE_ALREADY_INITIALIZED", "control database already exists", EXIT_CONFLICT
            )
        with ConfigurationLock(paths.lock):
            if paths.database.exists():
                raise ControlError(
                    "STATE_ALREADY_INITIALIZED",
                    "control database already exists",
                    EXIT_CONFLICT,
                )
            created = create_state_database(staging)
            _apply_owner_only_acl(staging)
            staged_connection = sqlite3.connect(staging, timeout=10.0, isolation_level=None)
            staged_connection.row_factory = sqlite3.Row
            try:
                staged_connection.execute("PRAGMA foreign_keys = ON")
                verify_audit(staged_connection)
            finally:
                staged_connection.close()
            os.replace(staging, paths.database)
    except BaseException:
        for candidate in (
            staging,
            Path(f"{staging}-journal"),
            Path(f"{staging}-wal"),
            Path(f"{staging}-shm"),
        ):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
        if not paths.database.exists():
            for candidate in (paths.lock,):
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    pass
            for directory in (paths.logs, paths.demo):
                try:
                    directory.rmdir()
                except (FileNotFoundError, OSError):
                    pass
            if not root_existed:
                try:
                    paths.root.rmdir()
                except OSError:
                    pass
        raise
    assert created is not None
    assert acl_result is not None
    return {
        "acl": acl_result,
        "controller_instance_id": created["controller_instance_id"],
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
        "SELECT sequence, event_hash, timestamp FROM audit_events ORDER BY sequence DESC LIMIT 1"
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
    try:
        proposed_timestamp = parse_time(timestamp_value, "audit.timestamp")
    except ControlError as exc:
        raise IntegrityFailure("proposed audit timestamp is invalid") from exc
    if row is not None:
        try:
            previous_timestamp = parse_time(str(row["timestamp"]), "audit.previous_timestamp")
        except ControlError as exc:
            raise IntegrityFailure("previous audit timestamp is invalid") from exc
        if proposed_timestamp < previous_timestamp:
            timestamp_value = str(row["timestamp"])
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
    metadata = {
        str(row["key"]): str(row["value"])
        for row in connection.execute("SELECT key, value FROM metadata")
    }
    if metadata.get("schema_version") != STATE_SCHEMA_VERSION:
        raise IntegrityFailure("controller schema version is missing or unsupported")
    controller_instance_id = metadata.get("controller_instance_id", "")
    try:
        parsed_instance_id = uuid.UUID(controller_instance_id)
    except (ValueError, AttributeError) as exc:
        raise IntegrityFailure("controller instance identity is invalid") from exc
    if parsed_instance_id.version != 4 or str(parsed_instance_id) != controller_instance_id:
        raise IntegrityFailure("controller instance identity is not a canonical UUIDv4")

    if _normalized_schema_objects(connection) != _expected_schema_objects():
        raise IntegrityFailure("controller schema objects are missing, added, or changed")

    trigger_sql = {
        str(row["name"]): " ".join(str(row["sql"] or "").upper().split())
        for row in connection.execute("SELECT name, sql FROM sqlite_master WHERE type='trigger'")
    }
    expected_triggers = {
        "audit_events_no_update": "BEFORE UPDATE ON AUDIT_EVENTS",
        "audit_events_no_delete": "BEFORE DELETE ON AUDIT_EVENTS",
        "operators_no_update": "BEFORE UPDATE ON OPERATORS",
        "operators_no_delete": "BEFORE DELETE ON OPERATORS",
        "targets_no_unisolate": "BEFORE UPDATE OF CONTROL_STATE ON TARGETS",
        "targets_registration_no_update": "BEFORE UPDATE OF TARGET, ARGV_JSON, CWD, EXECUTABLE_SHA256, CREATED_AT ON TARGETS",
        "targets_no_delete": "BEFORE DELETE ON TARGETS",
        "requests_authorization_no_update": "BEFORE UPDATE OF REQUEST_ID, NONCE, ENVELOPE_SHA256, ENVELOPE_JSON, TARGET, ACCEPTED_AT, EXPIRES_AT ON REQUESTS",
        "requests_no_delete": "BEFORE DELETE ON REQUESTS",
        "requests_status_monotonic": "BEFORE UPDATE OF STATUS ON REQUESTS",
        "requests_applied_no_update": "BEFORE UPDATE ON REQUESTS",
        "runs_identity_no_update": "BEFORE UPDATE OF RUN_ID, TARGET, CONTEXT_TRACE_ID, JOB_NAME, LOG_PATH, CREATED_AT ON RUNS",
        "runs_supervisor_identity_monotonic": "BEFORE UPDATE OF SUPERVISOR_PID, SUPERVISOR_CREATED_FILETIME ON RUNS",
        "runs_no_delete": "BEFORE DELETE ON RUNS",
        "trust_store_no_unseal": "BEFORE UPDATE OF VALUE ON METADATA",
        "metadata_identity_no_update": "BEFORE UPDATE OF KEY, VALUE ON METADATA",
        "metadata_no_delete": "BEFORE DELETE ON METADATA",
        "context_traces_no_update": "BEFORE UPDATE ON CONTEXT_TRACES",
        "context_traces_no_delete": "BEFORE DELETE ON CONTEXT_TRACES",
        "policy_checks_no_update": "BEFORE UPDATE ON POLICY_CHECKS",
        "policy_checks_no_delete": "BEFORE DELETE ON POLICY_CHECKS",
    }
    for name, fragment in expected_triggers.items():
        sql = trigger_sql.get(name, "")
        if fragment not in sql or "RAISE(ABORT" not in sql:
            raise IntegrityFailure(f"required integrity trigger is missing or changed: {name}")
    required_indexes = {
        "one_live_run_per_target",
        "one_enforcing_request_per_target",
        "one_policy_type_per_trace",
    }
    found_indexes = {
        str(row["name"])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
        )
    }
    if not required_indexes <= found_indexes:
        raise IntegrityFailure("a required controller uniqueness index is missing")

    allowed_events = {
        "STATE_INITIALIZED",
        "OPERATOR_ADDED",
        "TRUST_STORE_SEALED",
        "TARGET_REGISTERED",
        "CONTEXT_TRACE_RECORDED",
        "START_RESERVED",
        "START_FAILED",
        "SUPERVISOR_CLAIMED",
        "PROCESS_TREE_STARTED",
        "PROCESS_TREE_EXITED",
        "SUPERVISOR_FAILURE",
        "SUPERVISOR_EXIT_RECONCILED",
        "ISOLATION_ACCEPTED",
        "ISOLATION_VERIFIED",
        "ISOLATION_CLIENT_WAIT_TIMEOUT",
    }
    expected_previous = ZERO_HASH
    expected_sequence = 1
    count = 0
    events: list[dict[str, Any]] = []
    previous_timestamp: datetime | None = None
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
        event_type = str(row["event_type"])
        if event_type not in allowed_events:
            raise IntegrityFailure(f"unknown audit event type at sequence {sequence}: {event_type}")
        try:
            timestamp = parse_time(str(row["timestamp"]), "audit.timestamp")
        except ControlError as exc:
            raise IntegrityFailure(f"invalid audit timestamp at sequence {sequence}") from exc
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise IntegrityFailure(f"audit timestamp moved backwards at sequence {sequence}")
        previous_timestamp = timestamp
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
        events.append(
            {
                "actor": str(row["actor"]),
                "event_type": event_type,
                "payload": payload,
                "sequence": sequence,
                "target": row["target"],
                "timestamp": str(row["timestamp"]),
            }
        )
        expected_previous = calculated
        expected_sequence += 1
        count += 1
    tip_row = connection.execute(
        "SELECT value FROM metadata WHERE key='audit_tip'"
    ).fetchone()
    if tip_row is None or str(tip_row["value"]) != expected_previous:
        raise IntegrityFailure("audit tip does not match the verified chain")
    genesis_events = [event for event in events if event["event_type"] == "STATE_INITIALIZED"]
    if not events or events[0]["event_type"] != "STATE_INITIALIZED" or len(genesis_events) != 1:
        raise IntegrityFailure("audit genesis is missing")
    genesis = events[0]
    expected_genesis_payload = {
        "controller_instance_id": controller_instance_id,
        "context_generation_enabled": True,
        "program": PROGRAM,
        "program_version": PROGRAM_VERSION,
        "schema_version": STATE_SCHEMA_VERSION,
        "truth_boundary": "local_windows_supervised_processes_only",
    }
    if (
        genesis["sequence"] != 1
        or genesis["target"] is not None
        or genesis["payload"] != expected_genesis_payload
    ):
        raise IntegrityFailure("audit genesis does not bind the controller instance")

    operator_events: dict[str, dict[str, Any]] = {}
    for event in events:
        if event["event_type"] == "OPERATOR_ADDED":
            payload = event["payload"]
            if not isinstance(payload, dict) or set(payload) != {"fingerprint", "operator_id", "scheme"}:
                raise IntegrityFailure("operator audit payload is invalid")
            operator_id = str(payload["operator_id"])
            if operator_id in operator_events or payload["scheme"] != SIGNATURE_SCHEME:
                raise IntegrityFailure("operator audit event is duplicated or invalid")
            operator_events[operator_id] = payload
    operator_rows = {
        str(row["operator_id"]): row for row in connection.execute("SELECT * FROM operators")
    }
    if set(operator_rows) != set(operator_events):
        raise IntegrityFailure("trusted operator table does not match audited operator additions")
    for operator_id, row in operator_rows.items():
        raw_key = b64url_decode(row["public_key_b64"], exact_bytes=32)
        fingerprint = hashlib.sha256(raw_key).hexdigest()
        if (
            not secrets.compare_digest(fingerprint, str(row["fingerprint"]))
            or operator_events[operator_id]["fingerprint"] != fingerprint
            or int(row["enabled"]) != 1
        ):
            raise IntegrityFailure(f"trusted operator row is not bound to its audit event: {operator_id}")

    sealed_value = metadata.get("trust_store_sealed")
    if sealed_value not in {"0", "1"}:
        raise IntegrityFailure("trust-store seal metadata is missing or invalid")
    sealed = sealed_value == "1"
    seal_events = [event for event in events if event["event_type"] == "TRUST_STORE_SEALED"]
    if len(seal_events) != (1 if sealed else 0):
        raise IntegrityFailure("trust-store seal state does not match the audit chain")
    if sealed and (
        seal_events[0]["payload"]
        != {"operator_count": len(operator_rows), "quorum": QUORUM}
    ):
        raise IntegrityFailure("trust-store seal audit payload does not match the trusted keys")

    target_events: dict[str, dict[str, Any]] = {}
    accepted_targets: set[str] = set()
    for event in events:
        if event["event_type"] == "TARGET_REGISTERED":
            target = str(event["target"])
            if target in target_events or not isinstance(event["payload"], dict):
                raise IntegrityFailure("target registration audit event is duplicated or invalid")
            target_events[target] = event["payload"]
        elif event["event_type"] == "ISOLATION_ACCEPTED":
            accepted_targets.add(str(event["target"]))
    target_rows = {str(row["target"]): row for row in connection.execute("SELECT * FROM targets")}
    if set(target_rows) != set(target_events):
        raise IntegrityFailure("target table does not match audited registrations")
    for target, row in target_rows.items():
        argv = parse_json_bytes(str(row["argv_json"]).encode("utf-8"))
        if canonical_json(argv).decode("utf-8") != str(row["argv_json"]):
            raise IntegrityFailure(f"target argv is not canonical: {target}")
        payload = target_events[target]
        if (
            payload.get("argv_sha256") != hashlib.sha256(canonical_json(argv)).hexdigest()
            or payload.get("cwd") != str(row["cwd"])
            or payload.get("executable_sha256") != str(row["executable_sha256"])
            or payload.get("shell") is not False
        ):
            raise IntegrityFailure(f"target row does not match audited registration: {target}")
        if (str(row["control_state"]) == "ISOLATED") != (target in accepted_targets):
            raise IntegrityFailure(f"target control state does not match isolation audit: {target}")

    context_events: dict[str, dict[str, Any]] = {}
    for event in events:
        if event["event_type"] == "CONTEXT_TRACE_RECORDED":
            trace_id = str(event["payload"].get("trace_id", "")) if isinstance(event["payload"], dict) else ""
            if not trace_id or trace_id in context_events:
                raise IntegrityFailure("context audit event is duplicated or invalid")
            context_events[trace_id] = event
    context_rows = list(connection.execute("SELECT * FROM context_traces"))
    context_rows_by_id = {str(row["trace_id"]): row for row in context_rows}
    if {str(row["trace_id"]) for row in context_rows} != set(context_events):
        raise IntegrityFailure("context trace table does not match audited context events")
    for row in context_rows:
        trace_id = str(row["trace_id"])
        event = context_events[trace_id]
        if (
            event["target"] != row["target"]
            or event["payload"].get("evidence_sha256") != row["evidence_sha256"]
            or event["payload"].get("convergence") != row["convergence_status"]
        ):
            raise IntegrityFailure(f"context trace does not match its audit event: {trace_id}")
        policy_rows = connection.execute(
            "SELECT * FROM policy_checks WHERE trace_id=?", (trace_id,)
        ).fetchall()
        if len(policy_rows) != 3:
            raise IntegrityFailure(f"context trace does not have three policy checks: {trace_id}")
        _context_evidence_from_rows(row, target_rows[str(row["target"])], policy_rows)

    start_events: dict[str, dict[str, Any]] = {}
    for event in events:
        if event["event_type"] == "START_RESERVED":
            run_id = str(event["payload"].get("run_id", "")) if isinstance(event["payload"], dict) else ""
            if not run_id or run_id in start_events:
                raise IntegrityFailure("run reservation audit event is duplicated or invalid")
            start_events[run_id] = event
    run_rows = list(connection.execute("SELECT * FROM runs"))
    if {str(row["run_id"]) for row in run_rows} != set(start_events):
        raise IntegrityFailure("run table does not match audited reservations")
    consumed_context_ids: set[str] = set()
    for row in run_rows:
        run_id = str(row["run_id"])
        context_trace_id = str(row["context_trace_id"])
        context_row = context_rows_by_id.get(context_trace_id)
        if context_row is None or str(context_row["target"]) != str(row["target"]):
            raise IntegrityFailure(f"run context trace belongs to a different target: {run_id}")
        if context_trace_id in consumed_context_ids:
            raise IntegrityFailure(f"context trace was consumed by more than one run: {context_trace_id}")
        consumed_context_ids.add(context_trace_id)
        event = start_events[run_id]
        if (
            event["target"] != row["target"]
            or event["payload"].get("context_trace_id") != row["context_trace_id"]
            or event["payload"].get("target_binding") != target_binding_from_row(
                target_rows[str(row["target"])]
            )
        ):
            raise IntegrityFailure(f"run identity does not match its audit reservation: {row['run_id']}")
        lifecycle = [
            item
            for item in events
            if isinstance(item["payload"], dict)
            and item["payload"].get("run_id") == run_id
            and item["event_type"] != "START_RESERVED"
        ]
        if any(item["target"] != row["target"] for item in lifecycle):
            raise IntegrityFailure(f"run lifecycle event has the wrong target: {run_id}")
        types = [item["event_type"] for item in lifecycle]
        started = [item for item in lifecycle if item["event_type"] == "PROCESS_TREE_STARTED"]
        exited = [item for item in lifecycle if item["event_type"] == "PROCESS_TREE_EXITED"]
        claimed = [item for item in lifecycle if item["event_type"] == "SUPERVISOR_CLAIMED"]
        failed = [
            item
            for item in lifecycle
            if item["event_type"] in {
                "START_FAILED",
                "SUPERVISOR_FAILURE",
                "SUPERVISOR_EXIT_RECONCILED",
            }
        ]
        if len(started) > 1 or len(exited) > 1 or len(claimed) > 1:
            raise IntegrityFailure(f"run lifecycle event is duplicated: {run_id}")
        supervisor_pid = row["supervisor_pid"]
        supervisor_created_filetime = row["supervisor_created_filetime"]
        supervisor_identity_present = (
            supervisor_pid is not None or supervisor_created_filetime is not None
        )
        supervisor_identity_required = bool(claimed or started or supervisor_identity_present)
        if supervisor_identity_required:
            if (
                len(claimed) != 1
                or supervisor_pid is None
                or supervisor_created_filetime is None
                or int(supervisor_pid) <= 0
                or int(supervisor_created_filetime) <= 0
            ):
                raise IntegrityFailure(f"run supervisor identity is incomplete: {run_id}")
            claimed_payload = claimed[0]["payload"]
            if (
                set(claimed_payload)
                != {"run_id", "supervisor_pid", "supervisor_created_filetime"}
                or int(claimed_payload.get("supervisor_pid", -1)) != int(supervisor_pid)
                or int(claimed_payload.get("supervisor_created_filetime", -1))
                != int(supervisor_created_filetime)
            ):
                raise IntegrityFailure(
                    f"run supervisor identity does not match its audit event: {run_id}"
                )
        process_event = started[-1] if started else (exited[-1] if exited else None)
        if process_event is not None and row["child_pid"] is not None and int(
            process_event["payload"].get("child_pid", -1)
        ) != int(row["child_pid"]):
            raise IntegrityFailure(f"run child identity does not match its audit event: {run_id}")
        state = str(row["state"])
        if state == "RUNNING" and not started:
            raise IntegrityFailure(f"running run lacks a process-start event: {run_id}")
        if state == "EXITED" and not exited:
            raise IntegrityFailure(f"exited run lacks a process-exit event: {run_id}")
        if state == "FAILED" and not failed:
            raise IntegrityFailure(f"failed run lacks a failure event: {run_id}")
        if state == "ISOLATING" and "ISOLATION_ACCEPTED" not in types:
            raise IntegrityFailure(f"isolating run lacks an accepted request event: {run_id}")
        if state == "ISOLATED" and "ISOLATION_VERIFIED" not in types:
            raise IntegrityFailure(f"isolated run lacks an isolation verification event: {run_id}")

    accepted_events: dict[str, dict[str, Any]] = {}
    verified_requests: dict[str, list[dict[str, Any]]] = {}
    failed_requests: dict[str, int] = {}
    for event in events:
        payload = event["payload"]
        if event["event_type"] == "ISOLATION_ACCEPTED":
            request_id = str(payload.get("request_id", "")) if isinstance(payload, dict) else ""
            if not request_id or request_id in accepted_events:
                raise IntegrityFailure("isolation acceptance audit event is duplicated or invalid")
            accepted_events[request_id] = event
        elif event["event_type"] == "ISOLATION_VERIFIED" and isinstance(payload, dict):
            verified_id = str(payload.get("request_id", ""))
            verified_requests.setdefault(verified_id, []).append(event)
        elif event["event_type"] == "SUPERVISOR_FAILURE" and isinstance(payload, dict):
            if payload.get("request_id") is not None:
                failed_id = str(payload["request_id"])
                failed_requests[failed_id] = failed_requests.get(failed_id, 0) + 1
    request_rows = list(connection.execute("SELECT * FROM requests"))
    if {str(row["request_id"]) for row in request_rows} != set(accepted_events):
        raise IntegrityFailure("request table does not match audited isolation acceptances")
    for row in request_rows:
        request_id = str(row["request_id"])
        try:
            envelope = parse_json_bytes(str(row["envelope_json"]).encode("utf-8"))
            envelope = validate_envelope_structure(envelope, check_time=False)
        except ControlError as exc:
            raise IntegrityFailure(f"stored signed envelope is invalid: {request_id}") from exc
        envelope_bytes = canonical_json(envelope)
        if envelope_bytes.decode("utf-8") != str(row["envelope_json"]):
            raise IntegrityFailure(f"stored signed envelope is not canonical: {request_id}")
        envelope_hash = hashlib.sha256(envelope_bytes).hexdigest()
        if (
            envelope_hash != row["envelope_sha256"]
            or envelope["request_id"] != request_id
            or envelope["nonce"] != row["nonce"]
            or envelope["target"] != row["target"]
            or envelope["expires_at"] != row["expires_at"]
            or envelope["controller_instance_id"] != controller_instance_id
            or envelope["target_binding"] != target_binding_from_row(target_rows[str(row["target"])])
        ):
            raise IntegrityFailure(f"stored request columns do not match the signed envelope: {request_id}")
        signatures = envelope["authorization"]["signatures"]
        if len(signatures) < QUORUM:
            raise IntegrityFailure(f"stored request lacks quorum: {request_id}")
        message = signature_message(envelope)
        fingerprints: set[str] = set()
        for operator_id, encoded in sorted(signatures.items()):
            operator_row = operator_rows.get(operator_id)
            if operator_row is None:
                raise IntegrityFailure(f"stored request references an unknown operator: {operator_id}")
            try:
                Ed25519PublicKey.from_public_bytes(
                    b64url_decode(operator_row["public_key_b64"], exact_bytes=32)
                ).verify(b64url_decode(encoded, exact_bytes=64), message)
            except (InvalidSignature, ControlError) as exc:
                raise IntegrityFailure(f"stored request signature is invalid: {operator_id}") from exc
            fingerprint = str(operator_row["fingerprint"])
            if fingerprint in fingerprints:
                raise IntegrityFailure("stored request signatures are not distinct")
            fingerprints.add(fingerprint)
        accepted = accepted_events[request_id]
        accepted_payload = accepted["payload"]
        if (
            accepted["target"] != row["target"]
            or accepted_payload.get("envelope_sha256") != envelope_hash
            or accepted_payload.get("controller_instance_id") != controller_instance_id
            or accepted_payload.get("target_binding") != envelope["target_binding"]
            or accepted_payload.get("signers") != sorted(signatures)
        ):
            raise IntegrityFailure(f"request row does not match its acceptance audit: {request_id}")
        status = str(row["status"])
        if status == "APPLIED":
            verified = verified_requests.get(request_id, [])
            if len(verified) != 1:
                raise IntegrityFailure(
                    f"applied request lacks one isolation verification: {request_id}"
                )
            if row["applied_at"] is None or row["result_json"] is None:
                raise IntegrityFailure(f"applied request lacks its durable result: {request_id}")
            try:
                result = parse_json_bytes(str(row["result_json"]).encode("utf-8"))
            except ControlError as exc:
                raise IntegrityFailure(f"applied request result is invalid: {request_id}") from exc
            result_bytes = canonical_json(result)
            if (
                result_bytes.decode("utf-8") != str(row["result_json"])
                or not isinstance(result, dict)
                or result.get("request_id") != request_id
                or result.get("target") != row["target"]
                or result.get("operation_status") != "VERIFIED_ISOLATED"
            ):
                raise IntegrityFailure(f"applied request result is not bound to the request: {request_id}")
            verified_event = verified[0]
            verified_payload = verified_event["payload"]
            if (
                verified_event["target"] != row["target"]
                or verified_payload.get("applied_at") != row["applied_at"]
                or verified_payload.get("result_sha256")
                != hashlib.sha256(result_bytes).hexdigest()
            ):
                raise IntegrityFailure(
                    f"applied request result does not match its audit event: {request_id}"
                )
        if status == "ENFORCEMENT_UNCONFIRMED" and failed_requests.get(request_id, 0) != 1:
            raise IntegrityFailure(f"unconfirmed request lacks one request-bound failure: {request_id}")
        if status == "ENFORCING" and (row["applied_at"] is not None or row["result_json"] is not None):
            raise IntegrityFailure(f"enforcing request has a terminal result: {request_id}")
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
        str(executable).startswith("\\\\") or not re.match(r"^[A-Za-z]:[\\/]", str(executable))
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
    require_no_reparse_ancestors(executable, label="registered executable")
    if executable.suffix.lower() in {".bat", ".cmd"}:
        raise ControlError(
            "INVALID_COMMAND",
            "batch files are not accepted as executables; register powershell.exe -File explicitly",
            EXIT_INPUT,
        )
    result[0] = str(executable)
    if len(canonical_json(result)) > MAX_JSON_BYTES:
        raise ControlError("INVALID_COMMAND", "canonical argv exceeds 64 KiB", EXIT_INPUT)
    command_line = subprocess.list2cmdline(result)
    utf16_units = len(command_line.encode("utf-16-le")) // 2
    if utf16_units + 1 > 32767:
        raise ControlError(
            "INVALID_COMMAND", "Windows command line exceeds 32,767 UTF-16 code units", EXIT_INPUT
        )
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
    requested_cwd = cwd.expanduser()
    if not requested_cwd.is_absolute():
        raise ControlError("INVALID_WORKING_DIRECTORY", "working directory must be absolute", EXIT_INPUT)
    cwd = Path(os.path.abspath(requested_cwd))
    if not cwd.is_dir():
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
    require_no_reparse_ancestors(cwd, label="registered working directory")
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


CONTEXT_STATES = {
    "Observe",
    "Analyze",
    "Decide",
    "Verify",
    "Stabilized",
    "Flagged_For_Review",
}
TERMINAL_CONTEXT_STATES = {"Stabilized", "Flagged_For_Review"}
LEGAL_CONTEXT_TRANSITIONS = {
    ("Observe", "Analyze"),
    ("Analyze", "Decide"),
    ("Analyze", "Observe"),
    ("Decide", "Verify"),
    ("Decide", "Analyze"),
    ("Verify", "Analyze"),
    ("Verify", "Stabilized"),
    ("Verify", "Flagged_For_Review"),
}
BACKWARD_CONTEXT_TRANSITIONS = {
    ("Analyze", "Observe"),
    ("Decide", "Analyze"),
    ("Verify", "Analyze"),
}
INVARIANT_KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
CONTEXT_POLICY_ORDER = (
    "Safety_Gate_Evaluated",
    "Context_Reweight_Flagged",
    "Output_Filter_Triggered",
)


def _validate_context_scalar(value: Any, label: str) -> None:
    if value is None or type(value) is bool:
        return
    if type(value) is int and -(2**63) <= value < 2**63:
        return
    if isinstance(value, str) and len(value) <= 4096 and "\x00" not in value:
        return
    raise ContextGenerationError(
        "INVALID_CONTEXT_INPUT", f"{label} must be a bounded JSON scalar", EXIT_INPUT
    )


def validate_context_input(value: Any) -> dict[str, Any]:
    context_input = require_exact_keys(value, {"challenge_question", "steps"}, "context input")
    challenge = context_input["challenge_question"]
    if type(challenge) is not int or not 0 <= challenge < 2**63:
        raise ContextGenerationError(
            "INVALID_CONTEXT_INPUT", "challenge_question must be a non-negative integer", EXIT_INPUT
        )
    steps = context_input["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_CONTEXT_STEPS:
        raise ContextGenerationError(
            "INVALID_CONTEXT_INPUT",
            f"steps must contain between 1 and {MAX_CONTEXT_STEPS} transitions",
            EXIT_INPUT,
        )
    previous_to: str | None = None
    normalized: list[dict[str, Any]] = []
    expected_fields = {
        "context_reweighted",
        "from_state",
        "invariants",
        "output_filter_triggered",
        "safety_gate_passed",
        "to_state",
    }
    for index, raw_step in enumerate(steps):
        step = require_exact_keys(raw_step, expected_fields, f"steps[{index}]")
        from_state = step["from_state"]
        to_state = step["to_state"]
        if from_state not in CONTEXT_STATES or to_state not in CONTEXT_STATES:
            raise ContextGenerationError(
                "INVALID_CONTEXT_TRANSITION", f"steps[{index}] names an unknown state", EXIT_INPUT
            )
        if index == 0 and from_state != "Observe":
            raise ContextGenerationError(
                "INVALID_CONTEXT_TRANSITION", "the execution trace must start in Observe", EXIT_INPUT
            )
        if previous_to is not None and from_state != previous_to:
            raise ContextGenerationError(
                "INVALID_CONTEXT_TRANSITION", f"steps[{index}] does not continue the prior state", EXIT_INPUT
            )
        if (from_state, to_state) not in LEGAL_CONTEXT_TRANSITIONS:
            raise ContextGenerationError(
                "INVALID_CONTEXT_TRANSITION",
                f"illegal context transition: {from_state} -> {to_state}",
                EXIT_INPUT,
            )
        if index < len(steps) - 1 and to_state in TERMINAL_CONTEXT_STATES:
            raise ContextGenerationError(
                "INVALID_CONTEXT_TRANSITION", "a terminal state may appear only on the last step", EXIT_INPUT
            )
        for field in (
            "context_reweighted",
            "output_filter_triggered",
            "safety_gate_passed",
        ):
            if type(step[field]) is not bool:
                raise ContextGenerationError(
                    "INVALID_CONTEXT_INPUT", f"steps[{index}].{field} must be boolean", EXIT_INPUT
                )
        invariants = step["invariants"]
        if not isinstance(invariants, dict) or len(invariants) > 64:
            raise ContextGenerationError(
                "INVALID_CONTEXT_INPUT", f"steps[{index}].invariants must be a bounded object", EXIT_INPUT
            )
        normalized_invariants: dict[str, Any] = {}
        for key, invariant_value in invariants.items():
            if not isinstance(key, str) or not INVARIANT_KEY_PATTERN.fullmatch(key):
                raise ContextGenerationError(
                    "INVALID_CONTEXT_INPUT", f"steps[{index}] has an invalid invariant key", EXIT_INPUT
                )
            _validate_context_scalar(invariant_value, f"steps[{index}].invariants.{key}")
            normalized_invariants[key] = invariant_value
        normalized.append(
            {
                "context_reweighted": step["context_reweighted"],
                "from_state": from_state,
                "invariants": normalized_invariants,
                "output_filter_triggered": step["output_filter_triggered"],
                "safety_gate_passed": step["safety_gate_passed"],
                "to_state": to_state,
            }
        )
        previous_to = to_state
    if previous_to not in TERMINAL_CONTEXT_STATES:
        raise ContextGenerationError(
            "INVALID_CONTEXT_TRANSITION",
            "the execution trace must end in Stabilized or Flagged_For_Review",
            EXIT_INPUT,
        )
    normalized_value = {"challenge_question": challenge, "steps": normalized}
    if len(canonical_json(normalized_value)) > MAX_JSON_BYTES:
        raise ContextGenerationError(
            "INVALID_CONTEXT_INPUT", "context input exceeds the 64 KiB limit", EXIT_INPUT
        )
    return normalized_value


class EntropyDepthAllocator:
    """Bound the number of operator-asserted reweight/backtracking transitions."""

    def __init__(self, budget: int = DEFAULT_ENTROPY_BUDGET):
        if type(budget) is not int or not 1 <= budget <= MAX_CONTEXT_STEPS:
            raise ContextGenerationError(
                "INVALID_ENTROPY_BUDGET",
                f"entropy budget must be an integer from 1 through {MAX_CONTEXT_STEPS}",
                EXIT_INPUT,
            )
        self.budget = budget
        self.used = 0

    def observe(self, step: Mapping[str, Any]) -> None:
        transition = (str(step["from_state"]), str(step["to_state"]))
        if bool(step["context_reweighted"]) or transition in BACKWARD_CONTEXT_TRANSITIONS:
            self.used += 1
            if self.used > self.budget:
                raise ContextGenerationError(
                    "ENTROPY_BUDGET_EXCEEDED",
                    f"execution trace consumed more than {self.budget} entropy units",
                    EXIT_DENIED,
                )

    def allocate(self, steps: Sequence[Mapping[str, Any]]) -> int:
        for step in steps:
            self.observe(step)
        return self.used


class CrossStepConsistency:
    """Compare repeated invariant assertions using canonical JSON equality."""

    def __init__(self, threshold_ppm: int = DEFAULT_CONSISTENCY_THRESHOLD_PPM):
        if type(threshold_ppm) is not int or not 950_000 <= threshold_ppm <= 1_000_000:
            raise ContextGenerationError(
                "INVALID_CONSISTENCY_THRESHOLD",
                "consistency threshold must be an integer from 950000 through 1000000 ppm",
                EXIT_INPUT,
            )
        self.threshold_ppm = threshold_ppm

    def evaluate(self, steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        first_values: dict[str, bytes] = {}
        comparisons = 0
        matches = 0
        conflicts: set[str] = set()
        for step in steps:
            for key, value in dict(step["invariants"]).items():
                encoded = canonical_json(value)
                if key not in first_values:
                    first_values[key] = encoded
                    continue
                comparisons += 1
                if secrets.compare_digest(first_values[key], encoded):
                    matches += 1
                else:
                    conflicts.add(key)
        score_ppm = (matches * 1_000_000 // comparisons) if comparisons else 0
        return {
            "comparisons": comparisons,
            "conflicting_invariants": sorted(conflicts),
            "engine": CROSSSTEP_CONSISTENCY,
            "matches": matches,
            "score_ppm": score_ppm,
            "threshold_met": score_ppm >= self.threshold_ppm,
            "threshold_ppm": self.threshold_ppm,
        }


def target_binding_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    argv = parse_json_bytes(str(row["argv_json"]).encode("utf-8"))
    return {
        "argv_sha256": hashlib.sha256(canonical_json(argv)).hexdigest(),
        "executable_sha256": str(row["executable_sha256"]),
    }


def controller_instance_id_from_connection(connection: sqlite3.Connection) -> str:
    row = connection.execute(
        "SELECT value FROM metadata WHERE key='controller_instance_id'"
    ).fetchone()
    if row is None:
        raise IntegrityFailure("controller instance identity is missing")
    value = str(row["value"])
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise IntegrityFailure("controller instance identity is not a UUID") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise IntegrityFailure("controller instance identity is not a canonical UUIDv4")
    return value


class EnrichedContextGenerator:
    """Generate deterministic evidence from trusted, operator-supplied assertions."""

    def __init__(
        self,
        target_row: Mapping[str, Any],
        *,
        entropy_budget: int = DEFAULT_ENTROPY_BUDGET,
        consistency_threshold_ppm: int = DEFAULT_CONSISTENCY_THRESHOLD_PPM,
    ):
        self.target_row = target_row
        self.allocator = EntropyDepthAllocator(entropy_budget)
        self.consistency = CrossStepConsistency(consistency_threshold_ppm)

    def generate(
        self,
        context_input: Any,
        *,
        created_at: str | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        normalized = validate_context_input(context_input)
        steps = normalized["steps"]
        used = self.allocator.allocate(steps)
        consistency = self.consistency.evaluate(steps)
        requested_state = str(steps[-1]["to_state"])
        safety_failed = any(not bool(step["safety_gate_passed"]) for step in steps)
        reweighted = any(bool(step["context_reweighted"]) for step in steps)
        output_triggered = any(bool(step["output_filter_triggered"]) for step in steps)
        stabilized = (
            requested_state == "Stabilized"
            and bool(consistency["threshold_met"])
            and not consistency["conflicting_invariants"]
            and not safety_failed
            and not reweighted
            and not output_triggered
        )
        effective_state = "Stabilized" if stabilized else "Flagged_For_Review"
        policy_checks = [
            {
                "check_type": "Safety_Gate_Evaluated",
                "details": {"failed_step_indexes": [
                    index for index, step in enumerate(steps) if not step["safety_gate_passed"]
                ]},
                "status": "Flagged" if safety_failed else "Pass",
            },
            {
                "check_type": "Context_Reweight_Flagged",
                "details": {"reweighted_step_indexes": [
                    index for index, step in enumerate(steps) if step["context_reweighted"]
                ]},
                "status": "Flagged" if reweighted else "Pass",
            },
            {
                "check_type": "Output_Filter_Triggered",
                "details": {"triggered_step_indexes": [
                    index for index, step in enumerate(steps) if step["output_filter_triggered"]
                ]},
                "status": "Triggered" if output_triggered else "Pass",
            },
        ]
        try:
            target = validate_target(self.target_row["target"])
        except (KeyError, IndexError) as exc:
            raise ContextGenerationError(
                "INVALID_TARGET_BINDING", "registered target snapshot is incomplete", EXIT_INTERNAL
            ) from exc
        trace_value = trace_id or str(uuid.uuid4())
        try:
            parsed_trace_id = uuid.UUID(trace_value)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ContextGenerationError(
                "INVALID_TRACE_ID", "trace_id must be a UUID", EXIT_INPUT
            ) from exc
        if parsed_trace_id.version != 4 or str(parsed_trace_id) != trace_value:
            raise ContextGenerationError(
                "INVALID_TRACE_ID", "trace_id must be a canonical UUIDv4", EXIT_INPUT
            )
        evidence: dict[str, Any] = {
            "binding": {
                **target_binding_from_row(self.target_row),
                "control_state": str(self.target_row["control_state"]),
                "target": target,
            },
            "challenge_question": normalized["challenge_question"],
            "consistency": consistency,
            "convergence": effective_state,
            "created_at": created_at or format_time(utc_now()),
            "entropy": {
                "allocator": ENTROPY_DEPTH_ALLOCATOR,
                "budget": self.allocator.budget,
                "used": used,
            },
            "execution_trace": steps,
            "final_state": {"effective": effective_state, "requested": requested_state},
            "policy_checks": policy_checks,
            "schema": CONTEXT_EVIDENCE_SCHEMA,
            "target": target,
            "trace_id": trace_value,
        }
        evidence["evidence_sha256"] = hashlib.sha256(canonical_json(evidence)).hexdigest()
        return evidence


def _context_evidence_from_rows(
    trace_row: Mapping[str, Any], target_row: Mapping[str, Any], policy_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    policy_by_type = {
        str(row["check_type"]): {
            "check_type": str(row["check_type"]),
            "details": parse_json_bytes(str(row["details_json"]).encode("utf-8")),
            "status": str(row["status"]),
        }
        for row in policy_rows
    }
    evidence: dict[str, Any] = {
        "binding": parse_json_bytes(str(trace_row["binding_json"]).encode("utf-8")),
        "challenge_question": int(trace_row["challenge_question"]),
        "consistency": {
            "comparisons": 0,
            "conflicting_invariants": [],
            "engine": CROSSSTEP_CONSISTENCY,
            "matches": 0,
            "score_ppm": int(trace_row["consistency_score_ppm"]),
            "threshold_met": bool(trace_row["threshold_met"]),
            "threshold_ppm": int(trace_row["consistency_threshold_ppm"]),
        },
        "convergence": str(trace_row["convergence_status"]),
        "created_at": str(trace_row["created_at"]),
        "entropy": {
            "allocator": ENTROPY_DEPTH_ALLOCATOR,
            "budget": int(trace_row["entropy_budget"]),
            "used": int(trace_row["entropy_used"]),
        },
        "execution_trace": parse_json_bytes(
            str(trace_row["execution_trace_json"]).encode("utf-8")
        ),
        "final_state": parse_json_bytes(str(trace_row["final_state_json"]).encode("utf-8")),
        "policy_checks": [policy_by_type[name] for name in CONTEXT_POLICY_ORDER],
        "schema": CONTEXT_EVIDENCE_SCHEMA,
        "target": str(trace_row["target"]),
        "trace_id": str(trace_row["trace_id"]),
    }
    recomputed_consistency = CrossStepConsistency(
        int(trace_row["consistency_threshold_ppm"])
    ).evaluate(evidence["execution_trace"])
    evidence["consistency"] = recomputed_consistency
    evidence["evidence_sha256"] = str(trace_row["evidence_sha256"])
    calculated = hashlib.sha256(
        canonical_json({key: value for key, value in evidence.items() if key != "evidence_sha256"})
    ).hexdigest()
    if not secrets.compare_digest(calculated, evidence["evidence_sha256"]):
        raise IntegrityFailure("stored context evidence does not match its evidence hash")
    return evidence


def record_context_trace(
    paths: StatePaths, evidence: Mapping[str, Any], *, actor: str = "local-operator"
) -> dict[str, Any]:
    target = validate_target(evidence.get("target"))
    with ConfigurationLock(paths.lock), connect(paths) as connection:
        with immediate_transaction(connection):
            verify_audit(connection)
            target_row = connection.execute(
                "SELECT * FROM targets WHERE target=?", (target,)
            ).fetchone()
            if target_row is None:
                raise ControlError("TARGET_NOT_FOUND", "target is not registered", EXIT_CONFLICT)
            if evidence.get("schema") != CONTEXT_EVIDENCE_SCHEMA:
                raise ContextGenerationError(
                    "INVALID_CONTEXT_EVIDENCE", "context evidence schema is unsupported", EXIT_INPUT
                )
            binding = evidence.get("binding")
            expected_binding = {
                **target_binding_from_row(target_row),
                "control_state": str(target_row["control_state"]),
                "target": target,
            }
            if binding != expected_binding:
                raise ContextGenerationError(
                    "CONTEXT_TARGET_MISMATCH", "context evidence does not bind to the target snapshot", EXIT_DENIED
                )
            body = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
            calculated = hashlib.sha256(canonical_json(body)).hexdigest()
            if not isinstance(evidence.get("evidence_sha256"), str) or not secrets.compare_digest(
                calculated, str(evidence["evidence_sha256"])
            ):
                raise ContextGenerationError(
                    "CONTEXT_HASH_MISMATCH", "context evidence hash is invalid", EXIT_DENIED
                )
            try:
                entropy_input = evidence["entropy"]
                consistency_input = evidence["consistency"]
                regenerated = EnrichedContextGenerator(
                    target_row,
                    entropy_budget=entropy_input["budget"],
                    consistency_threshold_ppm=consistency_input["threshold_ppm"],
                ).generate(
                    {
                        "challenge_question": evidence["challenge_question"],
                        "steps": evidence["execution_trace"],
                    },
                    created_at=evidence["created_at"],
                    trace_id=evidence["trace_id"],
                )
            except (KeyError, TypeError, ControlError) as exc:
                raise ContextGenerationError(
                    "INVALID_CONTEXT_EVIDENCE",
                    "context evidence fields are incomplete or invalid",
                    EXIT_INPUT,
                ) from exc
            if canonical_json(dict(evidence)) != canonical_json(regenerated):
                raise ContextGenerationError(
                    "INVALID_CONTEXT_EVIDENCE",
                    "context evidence is not the deterministic generator output",
                    EXIT_INPUT,
                )
            policy_checks = evidence.get("policy_checks")
            if not isinstance(policy_checks, list) or [
                item.get("check_type") if isinstance(item, dict) else None for item in policy_checks
            ] != list(CONTEXT_POLICY_ORDER):
                raise ContextGenerationError(
                    "INVALID_CONTEXT_EVIDENCE", "context policy checks are incomplete or out of order", EXIT_INPUT
                )
            consistency = dict(evidence["consistency"])
            entropy = dict(evidence["entropy"])
            try:
                connection.execute(
                    """
                    INSERT INTO context_traces(
                        trace_id, target, challenge_question, execution_trace_json,
                        convergence_status, final_state_json, binding_json, entropy_budget, entropy_used,
                        consistency_score_ppm, consistency_threshold_ppm, threshold_met,
                        evidence_sha256, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        evidence["trace_id"], target, evidence["challenge_question"],
                        canonical_json(evidence["execution_trace"]).decode("utf-8"),
                        evidence["convergence"],
                        canonical_json(evidence["final_state"]).decode("utf-8"),
                        canonical_json(evidence["binding"]).decode("utf-8"),
                        entropy["budget"], entropy["used"], consistency["score_ppm"],
                        consistency["threshold_ppm"], int(bool(consistency["threshold_met"])),
                        evidence["evidence_sha256"], evidence["created_at"],
                    ),
                )
                for policy in policy_checks:
                    connection.execute(
                        """
                        INSERT INTO policy_checks(
                            check_id, trace_id, check_type, status, details_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(uuid.uuid4()), evidence["trace_id"], policy["check_type"],
                            policy["status"], canonical_json(policy["details"]).decode("utf-8"),
                            evidence["created_at"],
                        ),
                    )
            except sqlite3.IntegrityError as exc:
                raise ContextGenerationError(
                    "CONTEXT_TRACE_CONFLICT", "context trace could not be appended", EXIT_CONFLICT
                ) from exc
            append_audit(
                connection,
                "CONTEXT_TRACE_RECORDED",
                target,
                actor,
                {
                    "convergence": evidence["convergence"],
                    "evidence_sha256": evidence["evidence_sha256"],
                    "trace_id": evidence["trace_id"],
                },
            )
    return dict(evidence)


def _read_context_trace_in_connection(
    connection: sqlite3.Connection, *, target: str | None = None, trace_id: str | None = None
) -> dict[str, Any]:
    if (target is None) == (trace_id is None):
        raise ControlError(
            "INVALID_CONTEXT_QUERY", "select exactly one of target or trace_id", EXIT_INPUT
        )
    if target is not None:
        target = validate_target(target)
        trace_row = connection.execute(
            "SELECT rowid AS insertion_order, * FROM context_traces WHERE target=? ORDER BY rowid DESC LIMIT 1",
            (target,),
        ).fetchone()
    else:
        trace_row = connection.execute(
            "SELECT rowid AS insertion_order, * FROM context_traces WHERE trace_id=?", (trace_id,)
        ).fetchone()
    if trace_row is None:
        raise ControlError("CONTEXT_NOT_FOUND", "no context trace is recorded", EXIT_CONFLICT)
    target_row = connection.execute(
        "SELECT * FROM targets WHERE target=?", (trace_row["target"],)
    ).fetchone()
    if target_row is None:
        raise IntegrityFailure("context trace references a missing target")
    policy_rows = connection.execute(
        "SELECT * FROM policy_checks WHERE trace_id=?", (trace_row["trace_id"],)
    ).fetchall()
    if len(policy_rows) != len(CONTEXT_POLICY_ORDER):
        raise IntegrityFailure("context trace does not have exactly three policy checks")
    return _context_evidence_from_rows(trace_row, target_row, policy_rows)


def read_context_trace(
    paths: StatePaths, *, target: str | None = None, trace_id: str | None = None
) -> dict[str, Any]:
    with connect(paths) as connection:
        connection.execute("BEGIN")
        try:
            verify_audit(connection)
            result = _read_context_trace_in_connection(
                connection, target=target, trace_id=trace_id
            )
            connection.execute("COMMIT")
            return result
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise


def require_stabilized_context(
    connection_or_paths: sqlite3.Connection | StatePaths, target: str
) -> dict[str, Any]:
    owns_connection = isinstance(connection_or_paths, StatePaths)
    connection = connect(connection_or_paths) if owns_connection else connection_or_paths
    try:
        if owns_connection:
            connection.execute("BEGIN")
            verify_audit(connection)
        evidence = _read_context_trace_in_connection(connection, target=target)
        if evidence["convergence"] != "Stabilized" or not evidence["consistency"]["threshold_met"]:
            raise ControlError(
                "CONTEXT_NOT_STABILIZED", "latest context trace is not stabilized", EXIT_DENIED
            )
        if any(policy["status"] != "Pass" for policy in evidence["policy_checks"]):
            raise ControlError(
                "CONTEXT_POLICY_NOT_PASSED", "latest context trace has a non-passing policy check", EXIT_DENIED
            )
        if connection.execute(
            "SELECT 1 FROM runs WHERE context_trace_id=? LIMIT 1", (evidence["trace_id"],)
        ).fetchone() is not None:
            raise ControlError(
                "CONTEXT_ALREADY_CONSUMED", "latest context trace has already authorized a run", EXIT_CONFLICT
            )
        if owns_connection:
            connection.execute("COMMIT")
        return evidence
    except BaseException:
        if owns_connection and connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        if owns_connection:
            connection.close()


def generate_context(
    paths: StatePaths,
    target: str,
    context_input: Any,
    *,
    entropy_budget: int = DEFAULT_ENTROPY_BUDGET,
    consistency_threshold_ppm: int = DEFAULT_CONSISTENCY_THRESHOLD_PPM,
) -> dict[str, Any]:
    target = validate_target(target)
    with connect(paths) as connection:
        verify_audit(connection)
        row = connection.execute("SELECT * FROM targets WHERE target=?", (target,)).fetchone()
        if row is None:
            raise ControlError("TARGET_NOT_FOUND", "target is not registered", EXIT_CONFLICT)
        evidence = EnrichedContextGenerator(
            row,
            entropy_budget=entropy_budget,
            consistency_threshold_ppm=consistency_threshold_ppm,
        ).generate(context_input)
    return record_context_trace(paths, evidence)


def export_a11oy_context_evidence(paths: StatePaths, target: str) -> dict[str, Any]:
    target = validate_target(target)
    with connect(paths) as connection:
        connection.execute("BEGIN")
        try:
            audit = verify_audit(connection)
            context = _read_context_trace_in_connection(connection, target=target)
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
    projection: dict[str, Any] = {
        "audit": audit,
        "capabilities": {
            "a11oy_process_control": False,
            "a11oy_read_only_projection": True,
            "local_windows_supervisor_control": os.name == "nt",
        },
        "context": context,
        "generated_at": format_time(utc_now()),
        "program": {"name": PROGRAM, "version": PROGRAM_VERSION},
        "schema": "a11oy/owned-agent-control-projection/v1",
        "target": target,
        "truth_boundary": {
            "authority": "local_operator_only",
            "enforcement": "windows_job_object_for_registered_processes_launched_by_this_controller",
            "persistence": "local_tamper_evident_not_immutable",
            "remote_effects": False,
            "semantic_safety_evaluation": False,
        },
    }
    projection["projection_sha256"] = hashlib.sha256(canonical_json(projection)).hexdigest()
    return projection


def unsigned_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(envelope))
    result["authorization"]["signatures"] = {}
    return result


def signature_message(envelope: Mapping[str, Any]) -> bytes:
    return SIGNATURE_DOMAIN + canonical_json(unsigned_envelope(envelope))


def require_envelope_database_binding(
    connection: sqlite3.Connection, envelope: Mapping[str, Any]
) -> sqlite3.Row:
    controller_instance_id = controller_instance_id_from_connection(connection)
    if not secrets.compare_digest(
        str(envelope.get("controller_instance_id", "")), controller_instance_id
    ):
        raise ControlError(
            "CONTROLLER_INSTANCE_MISMATCH",
            "request belongs to a different controller state root",
            EXIT_DENIED,
        )
    row = connection.execute(
        "SELECT * FROM targets WHERE target=?", (envelope["target"],)
    ).fetchone()
    if row is None:
        raise ControlError("TARGET_NOT_FOUND", "target is not registered", EXIT_CONFLICT)
    if envelope.get("target_binding") != target_binding_from_row(row):
        raise ControlError(
            "TARGET_BINDING_MISMATCH",
            "request does not match the registered executable binding",
            EXIT_DENIED,
        )
    return row


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
            "controller_instance_id",
            "effects",
            "expires_at",
            "issued_at",
            "nonce",
            "request_id",
            "schema",
            "scope",
            "target",
            "target_binding",
        },
        "envelope",
    )
    if top["schema"] != ENVELOPE_SCHEMA or top["action"] != ENVELOPE_ACTION:
        raise ControlError("INVALID_ENVELOPE", "schema or action is not accepted", EXIT_INPUT)
    target = validate_target(top["target"])
    try:
        request_uuid = uuid.UUID(str(top["request_id"]))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ControlError("INVALID_ENVELOPE", "request_id must be a UUID", EXIT_INPUT) from exc
    if request_uuid.version != 4 or str(request_uuid) != top["request_id"]:
        raise ControlError("INVALID_ENVELOPE", "request_id must be a canonical UUIDv4", EXIT_INPUT)
    try:
        instance_uuid = uuid.UUID(str(top["controller_instance_id"]))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ControlError(
            "INVALID_ENVELOPE", "controller_instance_id must be a UUID", EXIT_INPUT
        ) from exc
    if instance_uuid.version != 4 or str(instance_uuid) != top["controller_instance_id"]:
        raise ControlError(
            "INVALID_ENVELOPE", "controller_instance_id must be a canonical UUIDv4", EXIT_INPUT
        )
    binding = require_exact_keys(
        top["target_binding"], {"argv_sha256", "executable_sha256"}, "target_binding"
    )
    for field in ("argv_sha256", "executable_sha256"):
        if not isinstance(binding[field], str) or not re.fullmatch(r"[0-9a-f]{64}", binding[field]):
            raise ControlError(
                "INVALID_ENVELOPE", f"target_binding.{field} must be a lowercase SHA-256", EXIT_INPUT
            )
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
        target_row = connection.execute("SELECT * FROM targets WHERE target=?", (target,)).fetchone()
        if target_row is None:
            raise ControlError("TARGET_NOT_FOUND", "target is not registered", EXIT_CONFLICT)
        controller_instance_id = controller_instance_id_from_connection(connection)
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
        "controller_instance_id": controller_instance_id,
        "effects": copy.deepcopy(EFFECTS),
        "expires_at": format_time(now + timedelta(seconds=ttl_seconds)),
        "issued_at": format_time(now),
        "nonce": b64url_encode(secrets.token_bytes(32)),
        "request_id": str(uuid.uuid4()),
        "schema": ENVELOPE_SCHEMA,
        "scope": copy.deepcopy(SCOPE),
        "target": target,
        "target_binding": target_binding_from_row(target_row),
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
        require_envelope_database_binding(connection, envelope)
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
    EXTENDED_STARTUPINFO_PRESENT = 0x00080000
    PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
    PROC_THREAD_ATTRIBUTE_JOB_LIST = 0x0002000D
    ERROR_INSUFFICIENT_BUFFER = 122
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

    class STARTUPINFOEXW(ctypes.Structure):
        _fields_ = [
            ("StartupInfo", STARTUPINFOW),
            ("lpAttributeList", ctypes.c_void_p),
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
    kernel32.InitializeProcThreadAttributeList.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(SIZE_T),
    ]
    kernel32.InitializeProcThreadAttributeList.restype = wintypes.BOOL
    kernel32.UpdateProcThreadAttribute.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_size_t,
        ctypes.c_void_p,
        SIZE_T,
        ctypes.c_void_p,
        ctypes.POINTER(SIZE_T),
    ]
    kernel32.UpdateProcThreadAttribute.restype = wintypes.BOOL
    kernel32.DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]
    kernel32.DeleteProcThreadAttributeList.restype = None
    kernel32.IsProcessInJob.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.BOOL),
    ]
    kernel32.IsProcessInJob.restype = wintypes.BOOL
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
    expected_executable_hash: str,
) -> WindowsManagedJob:
    if os.name != "nt":
        raise ControlError("WINDOWS_REQUIRED", "Windows Job Objects are required", EXIT_INTERNAL)
    import msvcrt

    normalized_argv = validate_argv(list(argv))
    cwd = Path(os.path.abspath(cwd))
    log_path = Path(os.path.abspath(log_path))
    require_no_reparse_ancestors(cwd, label="launch working directory")
    require_no_reparse_ancestors(log_path.parent, label="launch log directory")
    require_no_reparse_ancestors(Path(normalized_argv[0]), label="launch executable")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_executable_hash):
        raise IntegrityFailure("registered executable hash is invalid")

    # The unnamed Job is fully configured before CreateProcessW. JOB_LIST makes
    # membership part of process creation, so no uncontained child can run if the
    # detached supervisor crashes between process creation and assignment.
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise win_error("CreateJobObjectW")
    process_info = PROCESS_INFORMATION()
    log_stream: Any = None
    null_stream: Any = None
    attribute_buffer: Any = None
    attribute_list = ctypes.c_void_p()
    attributes_initialized = False
    job_values: Any = None
    handle_values: Any = None
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

        attribute_size = SIZE_T(0)
        ctypes.set_last_error(0)
        first_result = kernel32.InitializeProcThreadAttributeList(
            None, 2, 0, ctypes.byref(attribute_size)
        )
        first_error = ctypes.get_last_error()
        if first_result or first_error != ERROR_INSUFFICIENT_BUFFER or not attribute_size.value:
            raise ControlError(
                "WINDOWS_API_FAILED",
                f"InitializeProcThreadAttributeList size probe failed with Windows error {first_error}",
                EXIT_INTERNAL,
            )
        attribute_buffer = ctypes.create_string_buffer(attribute_size.value)
        attribute_list = ctypes.cast(attribute_buffer, ctypes.c_void_p)
        if not kernel32.InitializeProcThreadAttributeList(
            attribute_list, 2, 0, ctypes.byref(attribute_size)
        ):
            raise win_error("InitializeProcThreadAttributeList")
        attributes_initialized = True

        job_values = (wintypes.HANDLE * 1)()
        job_values[0] = job
        if not kernel32.UpdateProcThreadAttribute(
            attribute_list,
            0,
            PROC_THREAD_ATTRIBUTE_JOB_LIST,
            ctypes.cast(job_values, ctypes.c_void_p),
            ctypes.sizeof(job_values),
            None,
            None,
        ):
            raise win_error("UpdateProcThreadAttribute(JOB_LIST)")
        handle_values = (wintypes.HANDLE * 2)()
        handle_values[0] = stdin_handle
        handle_values[1] = stdout_handle
        if not kernel32.UpdateProcThreadAttribute(
            attribute_list,
            0,
            PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
            ctypes.cast(handle_values, ctypes.c_void_p),
            ctypes.sizeof(handle_values),
            None,
            None,
        ):
            raise win_error("UpdateProcThreadAttribute(HANDLE_LIST)")

        startup = STARTUPINFOEXW()
        startup.StartupInfo.cb = ctypes.sizeof(startup)
        startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES
        startup.StartupInfo.hStdInput = stdin_handle
        startup.StartupInfo.hStdOutput = stdout_handle
        startup.StartupInfo.hStdError = stdout_handle
        startup.lpAttributeList = attribute_list
        command_text = subprocess.list2cmdline(normalized_argv)
        if len(command_text.encode("utf-16-le")) // 2 + 1 > 32767:
            raise ControlError("INVALID_COMMAND", "Windows command line exceeds its limit", EXIT_INPUT)
        command_line = ctypes.create_unicode_buffer(command_text)
        environment_block = windows_environment_block(environment)
        flags = (
            CREATE_SUSPENDED
            | CREATE_UNICODE_ENVIRONMENT
            | CREATE_NO_WINDOW
            | EXTENDED_STARTUPINFO_PRESENT
        )
        created = kernel32.CreateProcessW(
            str(normalized_argv[0]),
            command_line,
            None,
            None,
            True,
            flags,
            ctypes.cast(environment_block, ctypes.c_void_p),
            str(cwd),
            ctypes.cast(ctypes.byref(startup), ctypes.POINTER(STARTUPINFOW)),
            ctypes.byref(process_info),
        )
        create_error = ctypes.get_last_error()
        kernel32.DeleteProcThreadAttributeList(attribute_list)
        attributes_initialized = False
        if not created:
            raise ControlError(
                "WINDOWS_API_FAILED",
                f"CreateProcessW failed with Windows error {create_error}",
                EXIT_INTERNAL,
            )

        in_job = wintypes.BOOL(False)
        if not kernel32.IsProcessInJob(
            process_info.hProcess, job, ctypes.byref(in_job)
        ) or not in_job.value:
            raise ControlError(
                "JOB_MEMBERSHIP_UNCONFIRMED",
                "suspended child was not proven to be a member of its Job Object",
                EXIT_UNCONFIRMED,
            )
        require_no_reparse_ancestors(cwd, label="launch working directory")
        require_no_reparse_ancestors(Path(normalized_argv[0]), label="launch executable")
        actual_executable_hash = hash_file(Path(normalized_argv[0]))
        if not secrets.compare_digest(actual_executable_hash, expected_executable_hash):
            raise ControlError(
                "EXECUTABLE_CHANGED",
                "registered executable changed before the suspended process could resume",
                EXIT_DENIED,
            )
        resume_result = kernel32.ResumeThread(process_info.hThread)
        if resume_result == 0xFFFFFFFF:
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
        if attributes_initialized:
            kernel32.DeleteProcThreadAttributeList(attribute_list)
            attributes_initialized = False
        if process_info.hProcess:
            if not kernel32.TerminateJobObject(job, 0xE0000004):
                kernel32.TerminateProcess(process_info.hProcess, 0xE0000004)
            kernel32.WaitForSingleObject(process_info.hProcess, 10_000)
        if process_info.hThread:
            kernel32.CloseHandle(process_info.hThread)
        if process_info.hProcess:
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
        lines = [line.strip() for line in (result.stderr or "").splitlines() if line.strip()]
        detail = lines[-1] if lines else f"isolated interpreter exited {result.returncode}"
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
            context_evidence = require_stabilized_context(connection, target)
            live = connection.execute(
                "SELECT run_id FROM runs WHERE target=? AND state IN ('STARTING','RUNNING','ISOLATING')",
                (target,),
            ).fetchone()
            if live is not None:
                raise ControlError("TARGET_ALREADY_RUNNING", "target already has a live run", EXIT_CONFLICT)
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, target, context_trace_id, state, job_name, supervisor_token_hash,
                    log_path, created_at
                ) VALUES (?, ?, ?, 'STARTING', ?, ?, ?, ?)
                """,
                (
                    run_id,
                    target,
                    context_evidence["trace_id"],
                    job_name,
                    capability_hash,
                    str(log_path),
                    now,
                ),
            )
            append_audit(
                connection,
                "START_RESERVED",
                target,
                "local-operator",
                {
                    "context_trace_id": context_evidence["trace_id"],
                    "run_id": run_id,
                    "shell": False,
                    "target_binding": target_binding_from_row(target_row),
                },
            )
    try:
        spawn_run_supervisor(paths, run_id, capability)
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
                "context_trace_id": row["context_trace_id"],
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
                "context_trace_id": row["context_trace_id"],
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
                request_id: str | None = None
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
                    request_row = connection.execute(
                        """
                        SELECT request_id FROM requests
                        WHERE target=? AND status='ENFORCING'
                        ORDER BY rowid DESC LIMIT 1
                        """,
                        (target_value,),
                    ).fetchone()
                    request_id = str(request_row["request_id"]) if request_row is not None else None
                    if request_id is None:
                        raise IntegrityFailure("isolated supervisor failure has no enforcing request")
                    connection.execute(
                        """
                        UPDATE requests SET status='ENFORCEMENT_UNCONFIRMED', result_json=?
                        WHERE request_id=? AND status='ENFORCING'
                        """,
                        (
                            canonical_json(
                                {
                                    "error_code": code,
                                    "message": message[:2048],
                                    "operation_status": "ENFORCEMENT_UNCONFIRMED",
                                }
                            ).decode("utf-8"),
                            request_id,
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
                        "request_id": request_id,
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
    if request_row["status"] == "APPLIED":
        if request_row["result_json"]:
            return parse_json_bytes(str(request_row["result_json"]).encode("utf-8"))
        return result
    observed = format_time(utc_now())
    if run_id is not None:
        connection.execute(
            """
            UPDATE runs SET state='ISOLATED', supervisor_token_hash=NULL,
                job_active_processes=0,
                credential_revoked_at=COALESCE(credential_revoked_at, ?),
                ended_at=COALESCE(ended_at, ?), heartbeat_at=?
            WHERE run_id=?
            """,
            (observed, observed, observed, run_id),
        )
    result_bytes = canonical_json(result)
    connection.execute(
        """
        UPDATE requests SET status='APPLIED', applied_at=?, result_json=?
        WHERE request_id=?
        """,
        (observed, result_bytes.decode("utf-8"), request_id),
    )
    append_audit(
        connection,
        "ISOLATION_VERIFIED",
        target,
        actor,
        {
            "applied_at": observed,
            "job_active_processes": 0,
            "local_process_tree_absence_basis": result["enforcement"][
                "local_process_tree_absence_basis"
            ],
            "request_id": request_id,
            "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
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
                context_trace_id = str(row["context_trace_id"])
                job_name = str(row["job_name"])
                log_path = Path(str(row["log_path"]))
                supervisor_pid = os.getpid()
                supervisor_created_filetime = process_handle_creation_filetime(
                    kernel32.GetCurrentProcess()
                )
                cursor = connection.execute(
                    """
                    UPDATE runs SET supervisor_token_hash=NULL, supervisor_pid=?,
                        supervisor_created_filetime=?, heartbeat_at=?
                    WHERE run_id=? AND supervisor_token_hash=?
                    """,
                    (
                        supervisor_pid,
                        supervisor_created_filetime,
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
                    {
                        "run_id": run_id,
                        "supervisor_pid": supervisor_pid,
                        "supervisor_created_filetime": supervisor_created_filetime,
                    },
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
                CONTEXT_TRACE_ID_ENV: context_trace_id,
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
                managed = launch_windows_job(
                    job_name,
                    argv,
                    cwd,
                    child_environment,
                    log_path,
                    expected_executable_hash,
                )
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
    envelope_bytes = canonical_json(envelope)
    envelope_json = envelope_bytes.decode("utf-8")
    envelope_hash = hashlib.sha256(envelope_bytes).hexdigest()
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
            require_envelope_database_binding(connection, envelope)
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
                "SELECT * FROM targets WHERE target=?", (target,)
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
                        request_id, nonce, envelope_sha256, envelope_json, target, status,
                        accepted_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, 'ENFORCING', ?, ?)
                    """,
                    (
                        request_id,
                        nonce,
                        envelope_hash,
                        envelope_json,
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
                    "controller_instance_id": envelope["controller_instance_id"],
                    "envelope_sha256": envelope_hash,
                    "request_id": request_id,
                    "run_id": live_run["run_id"] if live_run else None,
                    "signers": sorted(authorization["signers"]),
                    "supervisor_local_credential_revoked_before_termination": bool(
                        live_run is not None and live_run["credential_hash"]
                    ),
                    "target_binding": envelope["target_binding"],
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
            trace = connection.execute(
                """
                SELECT * FROM context_traces
                WHERE target=? ORDER BY rowid DESC LIMIT 1
                """,
                (target,),
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

    trace_info = None
    if trace is not None:
        trace_info = {
            "challenge_question": int(trace["challenge_question"]),
            "consistency_score_ppm": int(trace["consistency_score_ppm"]),
            "consistency_threshold_met": bool(trace["threshold_met"]),
            "convergence_status": str(trace["convergence_status"]),
            "entropy_budget": int(trace["entropy_budget"]),
            "entropy_used": int(trace["entropy_used"]),
            "evidence_sha256": str(trace["evidence_sha256"]),
            "trace_id": str(trace["trace_id"]),
        }

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
            "context_trace_id": run["context_trace_id"] if run else None,
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
        # New v2.0: Context generation metadata
        "context_generation": trace_info is not None,
        "latest_context_trace": trace_info,
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
    and creation FILETIME still match. An unclaimed reservation has no process
    identity or Job evidence: it either receives a short claim grace period,
    fails as a claim timeout, or is isolated on the basis that its launch gate
    was never passed. A bare PID is never queried or killed.
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
        run_id = str(row["run_id"])
        run_target = str(row["target"])
        created_age = (
            utc_now() - parse_time(row["created_at"], "created_at")
        ).total_seconds()
        if row["supervisor_pid"] is None:
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
                if request is None:
                    raise IntegrityFailure(
                        "isolated unclaimed reservation has no accepted request"
                    )
                _finalize_isolation(
                    paths,
                    str(request["request_id"]),
                    run_target,
                    run_id,
                    termination_performed=False,
                    credential_was_issued=False,
                    actor="stale-run-reconciler-unclaimed",
                    job_close_cleanup_basis=False,
                )
                results.append(
                    {
                        "operation_status": "VERIFIED_ISOLATED_BEFORE_LAUNCH_GATE",
                        "run_id": run_id,
                        "target": run_target,
                    }
                )
                continue
            if row["state"] == "STARTING" and created_age < 10.0:
                continue
            if row["state"] != "STARTING" or row["control_state"] != "READY":
                raise IntegrityFailure(
                    f"unclaimed run has an unreconcilable state: {run_id}"
                )
            with connect(paths) as connection:
                with immediate_transaction(connection):
                    verify_audit(connection)
                    current = connection.execute(
                        """
                        SELECT r.state, r.supervisor_pid,
                            r.supervisor_created_filetime, t.control_state
                        FROM runs r JOIN targets t ON t.target=r.target
                        WHERE r.run_id=?
                        """,
                        (run_id,),
                    ).fetchone()
                    if (
                        current is None
                        or current["state"] != "STARTING"
                        or current["control_state"] != "READY"
                        or current["supervisor_pid"] is not None
                        or current["supervisor_created_filetime"] is not None
                    ):
                        continue
                    observed = format_time(utc_now())
                    connection.execute(
                        """
                        UPDATE runs SET state='FAILED', supervisor_token_hash=NULL,
                            job_active_processes=0, ended_at=?, heartbeat_at=?,
                            error_code='SUPERVISOR_CLAIM_TIMEOUT',
                            error_message='supervisor claim deadline elapsed; target launch gate was never passed'
                        WHERE run_id=?
                        """,
                        (observed, observed, run_id),
                    )
                    append_audit(
                        connection,
                        "START_FAILED",
                        run_target,
                        "stale-run-reconciler",
                        {"error_code": "SUPERVISOR_CLAIM_TIMEOUT", "run_id": run_id},
                    )
            results.append(
                {
                    "operation_status": "SUPERVISOR_CLAIM_TIMEOUT",
                    "run_id": run_id,
                    "target": run_target,
                }
            )
            continue

        supervisor_pid = int(row["supervisor_pid"])
        supervisor_created_filetime = int(row["supervisor_created_filetime"])
        supervisor_status = query_process_status(
            supervisor_pid, supervisor_created_filetime
        )
        heartbeat_fresh = heartbeat_is_fresh(row["heartbeat_at"])
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
                    supervisor_created_filetime,
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
            # A claimed bootstrap receives a short startup grace period before
            # an observed exact-identity exit is reconciled as a failure.
            continue
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


def doctor(state_dir: Path | None = None) -> dict[str, Any]:
    """Report readiness prerequisites without claiming enforcement verification."""

    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, ok: bool, detail: str) -> None:
        checks[name] = {"detail": detail, "ok": bool(ok)}

    python_ok = (3, 11) <= sys.version_info[:2] < (3, 14)
    record(
        "python_version_supported",
        python_ok,
        f"{sys.implementation.name} {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    sqlite_ok = sqlite3.sqlite_version_info >= (3, 37, 0)
    record("sqlite_strict_supported", sqlite_ok, sqlite3.sqlite_version)
    try:
        cryptography_version = package_version("cryptography")
        cryptography_major = int(cryptography_version.split(".", 1)[0])
        crypto_ok = 46 <= cryptography_major < 51 and CRYPTOGRAPHY_IMPORT_ERROR is None
        record("cryptography_distribution_supported", crypto_ok, cryptography_version)
    except (PackageNotFoundError, ValueError):
        record("cryptography_distribution_supported", False, "not installed")
    windows_ok = os.name == "nt"
    record(
        "windows_platform",
        windows_ok,
        (
            f"os.name={os.name}; sys.platform={sys.platform}; "
            f"system={platform.system()}; release={platform.release()}; "
            f"version={platform.version()}; machine={platform.machine()}"
        ),
    )
    architecture_ok = ctypes.sizeof(ctypes.c_void_p) == 8
    record(
        "architecture_64_bit",
        architecture_ok,
        f"pointer_bits={ctypes.sizeof(ctypes.c_void_p) * 8}",
    )

    if state_dir is not None:
        try:
            checked_paths = state_paths(Path(state_dir).absolute())
            require_no_reparse_ancestors(
                checked_paths.root,
                allow_missing_suffix=True,
                label="doctor state directory",
            )
            record("state_path_ancestors", True, str(checked_paths.root))
        except ControlError as exc:
            record("state_path_ancestors", False, f"{exc.code}: {exc.message}")

    if windows_ok:
        required_symbols = (
            "CreateJobObjectW",
            "SetInformationJobObject",
            "InitializeProcThreadAttributeList",
            "UpdateProcThreadAttribute",
            "DeleteProcThreadAttributeList",
            "CreateProcessW",
            "IsProcessInJob",
        )
        missing = [name for name in required_symbols if not hasattr(kernel32, name)]
        record(
            "required_api_symbols",
            not missing,
            "all present" if not missing else f"missing: {', '.join(missing)}",
        )
        try:
            preflight_supervisor_runtime()
            record("isolated_supervisor_import", True, "cryptography imports under -I -B")
        except ControlError as exc:
            record("isolated_supervisor_import", False, f"{exc.code}: {exc.message}")

        probe_job: Any = None
        probe_attributes = ctypes.c_void_p()
        probe_initialized = False
        try:
            probe_job = kernel32.CreateJobObjectW(None, None)
            if not probe_job:
                raise win_error("CreateJobObjectW")
            limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            limits.BasicLimitInformation.LimitFlags = (
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
            )
            if not kernel32.SetInformationJobObject(
                probe_job,
                JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                ctypes.byref(limits),
                ctypes.sizeof(limits),
            ):
                raise win_error("SetInformationJobObject")
            probe_size = SIZE_T(0)
            ctypes.set_last_error(0)
            first = kernel32.InitializeProcThreadAttributeList(
                None, 1, 0, ctypes.byref(probe_size)
            )
            first_error = ctypes.get_last_error()
            if first or first_error != ERROR_INSUFFICIENT_BUFFER or not probe_size.value:
                raise ControlError(
                    "WINDOWS_API_FAILED", "attribute-list size probe was not supported", EXIT_INTERNAL
                )
            probe_buffer = ctypes.create_string_buffer(probe_size.value)
            probe_attributes = ctypes.cast(probe_buffer, ctypes.c_void_p)
            if not kernel32.InitializeProcThreadAttributeList(
                probe_attributes, 1, 0, ctypes.byref(probe_size)
            ):
                raise win_error("InitializeProcThreadAttributeList")
            probe_initialized = True
            probe_values = (wintypes.HANDLE * 1)()
            probe_values[0] = probe_job
            if not kernel32.UpdateProcThreadAttribute(
                probe_attributes,
                0,
                PROC_THREAD_ATTRIBUTE_JOB_LIST,
                ctypes.cast(probe_values, ctypes.c_void_p),
                ctypes.sizeof(probe_values),
                None,
                None,
            ):
                raise win_error("UpdateProcThreadAttribute(JOB_LIST)")
            record("job_list_attribute_probe", True, "JOB_LIST attribute accepted")
        except ControlError as exc:
            record("job_list_attribute_probe", False, f"{exc.code}: {exc.message}")
        except (AttributeError, OSError) as exc:
            record("job_list_attribute_probe", False, f"native API unavailable: {exc}")
        finally:
            if probe_initialized:
                kernel32.DeleteProcThreadAttributeList(probe_attributes)
            if probe_job:
                kernel32.CloseHandle(probe_job)
    else:
        record("required_api_symbols", False, "Windows APIs are unavailable")
        record("isolated_supervisor_import", False, "Windows platform required")
        record("job_list_attribute_probe", False, "Windows platform required")

    ready = all(check["ok"] for check in checks.values())
    return {
        "checks": checks,
        "controller_version": PROGRAM_VERSION,
        "executable": sys.executable,
        "ok": True,
        "operation": "doctor",
        "operation_status": (
            "READY_FOR_WINDOWS_SELF_TEST" if ready else "NOT_READY"
        ),
        "ready": ready,
        "schema_version": STATE_SCHEMA_VERSION,
        "truth_boundary": "readiness_only_not_enforcement_verified",
    }


def _self_test_context_input() -> dict[str, Any]:
    invariant = {"registered_target": "owned-agent:self-test"}
    return {
        "challenge_question": 1,
        "steps": [
            {
                "context_reweighted": False,
                "from_state": from_state,
                "invariants": invariant,
                "output_filter_triggered": False,
                "safety_gate_passed": True,
                "to_state": to_state,
            }
            for from_state, to_state in (
                ("Observe", "Analyze"),
                ("Analyze", "Decide"),
                ("Decide", "Verify"),
                ("Verify", "Stabilized"),
            )
        ],
    }


def _self_test_cleanup_exit(
    paths: StatePaths,
    target: str,
    identity: dict[str, int],
    observed_processes: set[int],
    exc_type: Any,
    original_exception: BaseException | None,
    traceback: Any,
) -> bool:
    """Always close the disposable supervisor's sole Job handle by exact identity."""

    del exc_type, traceback
    try:
        supervisor_identities: set[tuple[int, int]] = set()
        supplied_pid = int(identity.get("supervisor_pid", 0))
        supplied_filetime = int(identity.get("supervisor_created_filetime", 0))
        if supplied_pid > 0 and supplied_filetime > 0:
            supervisor_identities.add((supplied_pid, supplied_filetime))
        elif supplied_pid > 0 or supplied_filetime > 0:
            raise ControlError(
                "SELF_TEST_CLEANUP_UNCONFIRMED",
                "disposable supervisor result lacked an exact PID/creation-time pair",
                EXIT_UNCONFIRMED,
            )
        try:
            with connect(paths) as connection:
                rows = connection.execute(
                    """
                    SELECT supervisor_pid, supervisor_created_filetime, child_pid
                    FROM runs WHERE target=? ORDER BY rowid
                    """,
                    (target,),
                ).fetchall()
            for row in rows:
                row_pid = int(row["supervisor_pid"] or 0)
                row_filetime = int(row["supervisor_created_filetime"] or 0)
                if row_pid > 0 and row_filetime > 0:
                    supervisor_identities.add((row_pid, row_filetime))
                elif row_pid > 0 or row_filetime > 0:
                    raise ControlError(
                        "SELF_TEST_CLEANUP_UNCONFIRMED",
                        "a disposable run lacked an exact supervisor identity",
                        EXIT_UNCONFIRMED,
                    )
                child_pid = int(row["child_pid"] or 0)
                if child_pid > 0:
                    observed_processes.add(child_pid)
        except ControlError:
            raise
        try:
            _, child_record_path = demo_paths(paths, target)
            if child_record_path.is_file():
                child_record = load_json_file(child_record_path)
                if isinstance(child_record, dict) and type(child_record.get("child_pid")) is int:
                    observed_processes.add(int(child_record["child_pid"]))
        except (ControlError, OSError):
            pass

        for supervisor_pid, created_filetime in supervisor_identities:
            status = query_process_status(supervisor_pid, created_filetime)
            if status != PROCESS_EXITED:
                status = terminate_exact_process_identity(
                    supervisor_pid, created_filetime, timeout_seconds=5.0
                )
            if status != PROCESS_EXITED:
                raise ControlError(
                    "SELF_TEST_CLEANUP_UNCONFIRMED",
                    "disposable supervisor exact-identity termination was not confirmed",
                    EXIT_UNCONFIRMED,
                )

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and any(
            pid > 0 and process_running(pid) for pid in observed_processes
        ):
            time.sleep(0.05)
        remaining = sorted(
            pid for pid in observed_processes if pid > 0 and process_running(pid)
        )
        if remaining:
            raise ControlError(
                "SELF_TEST_CLEANUP_UNCONFIRMED",
                f"disposable supervised PIDs remained after Job closure: {remaining}",
                EXIT_UNCONFIRMED,
            )
    except BaseException as cleanup_exception:
        if original_exception is not None:
            raise ControlError(
                "SELF_TEST_CLEANUP_UNCONFIRMED",
                f"self-test failed with {original_exception!r}; cleanup also failed with {cleanup_exception!r}",
                EXIT_UNCONFIRMED,
            ) from original_exception
        raise
    return False


def self_test() -> dict[str, Any]:
    if os.name != "nt":
        raise ControlError("WINDOWS_REQUIRED", "self-test requires Windows", EXIT_INTERNAL)
    require_cryptography()
    checks: dict[str, Any] = {}
    cleanup_identity: dict[str, int] = {}
    observed_processes: set[int] = set()
    with tempfile.TemporaryDirectory(
        prefix="owned-agent-control-self-test-", ignore_cleanup_errors=True
    ) as directory, ExitStack() as cleanup_stack:
        paths = state_paths(Path(directory).resolve())
        initialize_state(paths)
        key_a = Ed25519PrivateKey.generate()
        key_b = Ed25519PrivateKey.generate()
        add_operator_raw(paths, "synthetic_a", key_a.public_key(), actor="self-test")
        add_operator_raw(paths, "synthetic_b", key_b.public_key(), actor="self-test")
        seal_trust_store(paths, actor="self-test")
        target = "owned-agent:self-test"
        cleanup_stack.push(
            lambda exc_type, exc, tb: _self_test_cleanup_exit(
                paths,
                target,
                cleanup_identity,
                observed_processes,
                exc_type,
                exc,
                tb,
            )
        )
        register_demo(paths, target)
        context = generate_context(paths, target, _self_test_context_input())
        checks["context"] = context["convergence"]
        start_result = start_agent(paths, target, timeout_seconds=20.0)
        cleanup_identity.update(
            {
                "supervisor_pid": int(start_result["supervisor_pid"] or 0),
                "supervisor_created_filetime": int(
                    start_result["supervisor_created_filetime"] or 0
                ),
            }
        )
        checks["start"] = start_result["operation_status"]
        heartbeat_path, child_record_path = demo_paths(paths, target)
        heartbeat = wait_for_file(heartbeat_path, 10.0)
        child_record = wait_for_file(child_record_path, 10.0)
        parent_pid = int(heartbeat["parent_pid"])
        child_pid = int(child_record["child_pid"])
        observed_processes.update({parent_pid, child_pid})
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

    doctor_parser = subparsers.add_parser("doctor", help="check Windows runtime readiness")
    doctor_parser.add_argument("--state-dir", type=Path)

    def state_command(name: str, help_text: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name, help=help_text)
        child.add_argument("--state-dir", required=True, type=Path)
        return child

    init_parser = state_command("init", "initialize protected controller state")

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

    context_generate_parser = state_command(
        "context-generate", "validate and append operator-supplied context evidence"
    )
    context_generate_parser.add_argument("--target", required=True)
    context_generate_parser.add_argument("--input", required=True, type=Path)
    context_generate_parser.add_argument(
        "--entropy-budget", type=int, default=DEFAULT_ENTROPY_BUDGET
    )
    context_generate_parser.add_argument(
        "--consistency-threshold-ppm",
        type=int,
        default=DEFAULT_CONSISTENCY_THRESHOLD_PPM,
    )

    context_show_parser = state_command("context-show", "show one verified context trace")
    context_selector = context_show_parser.add_mutually_exclusive_group(required=True)
    context_selector.add_argument("--target")
    context_selector.add_argument("--trace-id")

    context_export_parser = state_command(
        "context-export", "export a read-only A11oy context projection"
    )
    context_export_parser.add_argument("--target", required=True)
    context_export_parser.add_argument("--out", type=Path)

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

    audit_parser = state_command("audit-verify", "verify the local hash-chained audit ledger")

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
        if args.command == "doctor":
            result = doctor(args.state_dir)
            emit(result)
            return EXIT_OK if result["ready"] else EXIT_UNCONFIRMED
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
        if args.command == "context-generate":
            context_input = load_json_file(args.input)
            emit(
                generate_context(
                    require_state(args),
                    args.target,
                    context_input,
                    entropy_budget=args.entropy_budget,
                    consistency_threshold_ppm=args.consistency_threshold_ppm,
                )
            )
            return EXIT_OK
        if args.command == "context-show":
            emit(
                read_context_trace(
                    require_state(args), target=args.target, trace_id=args.trace_id
                )
            )
            return EXIT_OK
        if args.command == "context-export":
            projection = export_a11oy_context_evidence(require_state(args), args.target)
            if args.out is not None:
                atomic_write(args.out, (pretty_json(projection) + "\n").encode("utf-8"))
            emit(projection)
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
