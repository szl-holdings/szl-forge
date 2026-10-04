#!/usr/bin/env python3
"""Source-bound Windows host memory lease for the isolated v3 training worker.

This module is deliberately independent of GPU sampling and model imports.
A native sample is an observation, never a capacity guarantee or release gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
from typing import Any

MINIMUM_AVAILABLE_PHYSICAL_BYTES = 8 * 1024**3
MINIMUM_AVAILABLE_COMMIT_BYTES = 8 * 1024**3
SAMPLE_INTERVAL_SECONDS = 2
MAXIMUM_LEASE_GAP_NS = 8_000_000_000
MAXIMUM_NATIVE_UTC_SKEW_SECONDS = 5
MAXIMUM_SAMPLES = 5500
SAMPLE_SCHEMA = "szl.windows-host-memory-sample/v1"
HEX32 = re.compile(r"^[0-9a-f]{32}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAMPLE_KEYS = {
    "schema",
    "runId",
    "sourceRevision",
    "samplerSourceSha256",
    "observedAt",
    "sequence",
    "nativeMonotonicTicks",
    "nativeMonotonicFrequency",
    "availablePhysicalBytes",
    "totalPhysicalBytes",
    "committedBytes",
    "commitLimitBytes",
    "availableCommitBytes",
}
FAILURE_CODES = {
    "HOST_MEMORY_PRESSURE",
    "HOST_MEMORY_SAMPLE_INVALID",
    "HOST_MEMORY_LEASE_EXPIRED",
    "HOST_MEMORY_WATCHER_EXITED",
    "HOST_MEMORY_HELPER_UNAVAILABLE",
    "HOST_MEMORY_EVIDENCE_FAILED",
    "HOST_MEMORY_IDENTITY_MISMATCH",
    "HOST_MEMORY_STOP_UNCONFIRMED",
    "HOST_MEMORY_ABORTED",
}


class HostMemoryError(RuntimeError):
    """A retained bounded reason; never echoes arbitrary native output."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def exact_integer(value: Any, *, positive: bool = False) -> bool:
    return type(value) is int and (1 if positive else 0) <= value <= 2**63 - 1


@dataclass
class MemoryLease:
    run_id: str
    source_revision: str
    sampler_sha256: str
    failure: str | None = None
    samples: list[dict[str, Any]] = field(default_factory=list)
    last_received_ns: int | None = None

    def __post_init__(self) -> None:
        if not (
            isinstance(self.run_id, str)
            and HEX32.fullmatch(self.run_id)
            and isinstance(self.source_revision, str)
            and HEX40.fullmatch(self.source_revision)
            and isinstance(self.sampler_sha256, str)
            and HEX64.fullmatch(self.sampler_sha256)
        ):
            raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")

    def fail(self, reason: str) -> None:
        if reason not in FAILURE_CODES:
            reason = "HOST_MEMORY_SAMPLE_INVALID"
        if self.failure is None:
            self.failure = reason

    def deny(self, reason: str) -> None:
        self.fail(reason)
        raise HostMemoryError(self.failure)

    def assert_fresh(self, now_ns: int) -> None:
        if self.failure:
            raise HostMemoryError(self.failure)
        if (
            self.last_received_ns is None
            or not exact_integer(now_ns)
            or now_ns < self.last_received_ns
            or now_ns - self.last_received_ns > MAXIMUM_LEASE_GAP_NS
        ):
            self.deny("HOST_MEMORY_LEASE_EXPIRED")

    def observe(self, row: Any, *, received_ns: int, now: datetime) -> None:
        if self.failure:
            raise HostMemoryError(self.failure)
        if not isinstance(row, dict) or set(row) != SAMPLE_KEYS:
            self.deny("HOST_MEMORY_SAMPLE_INVALID")
        if (
            row["schema"] != SAMPLE_SCHEMA
            or row["runId"] != self.run_id
            or row["sourceRevision"] != self.source_revision
            or row["samplerSourceSha256"] != self.sampler_sha256
        ):
            self.deny("HOST_MEMORY_IDENTITY_MISMATCH")
        if not isinstance(row["observedAt"], str) or now.tzinfo is None:
            self.deny("HOST_MEMORY_SAMPLE_INVALID")
        try:
            stamp = datetime.fromisoformat(row["observedAt"].replace("Z", "+00:00"))
        except ValueError:
            self.deny("HOST_MEMORY_SAMPLE_INVALID")
        if (
            stamp.tzinfo is None
            or stamp.utcoffset().total_seconds() != 0
            or abs((now - stamp).total_seconds()) > MAXIMUM_NATIVE_UTC_SKEW_SECONDS
        ):
            self.deny("HOST_MEMORY_SAMPLE_INVALID")
        integers = SAMPLE_KEYS - {
            "schema",
            "runId",
            "sourceRevision",
            "samplerSourceSha256",
            "observedAt",
        }
        if not exact_integer(received_ns) or any(
            not exact_integer(row[key]) for key in integers
        ):
            self.deny("HOST_MEMORY_SAMPLE_INVALID")
        if (
            row["nativeMonotonicFrequency"] <= 0
            or row["totalPhysicalBytes"] <= 0
            or row["availablePhysicalBytes"] > row["totalPhysicalBytes"]
            or row["commitLimitBytes"] <= 0
            or row["committedBytes"] > row["commitLimitBytes"]
            or row["availableCommitBytes"]
            != row["commitLimitBytes"] - row["committedBytes"]
        ):
            self.deny("HOST_MEMORY_SAMPLE_INVALID")
        if not self.samples:
            if row["sequence"] != 0:
                self.deny("HOST_MEMORY_SAMPLE_INVALID")
        else:
            previous = self.samples[-1]["native"]
            self.assert_fresh(received_ns)
            if (
                received_ns <= self.last_received_ns
                or row["sequence"] != previous["sequence"] + 1
                or row["nativeMonotonicFrequency"]
                != previous["nativeMonotonicFrequency"]
                or row["nativeMonotonicTicks"] <= previous["nativeMonotonicTicks"]
                or (row["nativeMonotonicTicks"] - previous["nativeMonotonicTicks"])
                / row["nativeMonotonicFrequency"]
                > MAXIMUM_LEASE_GAP_NS / 1e9
            ):
                self.deny("HOST_MEMORY_SAMPLE_INVALID")
        if len(self.samples) >= MAXIMUM_SAMPLES:
            self.deny("HOST_MEMORY_EVIDENCE_FAILED")
        self.samples.append(
            {
                "native": dict(row),
                "receivedMonotonicNs": received_ns,
                "receivedAt": now.isoformat(),
            }
        )
        self.last_received_ns = received_ns
        if (
            row["availablePhysicalBytes"] < MINIMUM_AVAILABLE_PHYSICAL_BYTES
            or row["availableCommitBytes"] < MINIMUM_AVAILABLE_COMMIT_BYTES
        ):
            self.deny("HOST_MEMORY_PRESSURE")

    def public(self) -> dict[str, Any]:
        return {
            "schema": "szl.windows-host-memory-lease/v1",
            "runId": self.run_id,
            "sourceRevision": self.source_revision,
            "samplerSourceSha256": self.sampler_sha256,
            "state": "FAILED"
            if self.failure
            else "HEALTHY"
            if self.samples
            else "UNAVAILABLE",
            "failure": self.failure,
            "minimumAvailablePhysicalBytes": MINIMUM_AVAILABLE_PHYSICAL_BYTES,
            "minimumAvailableCommitBytes": MINIMUM_AVAILABLE_COMMIT_BYTES,
            "sampleIntervalSeconds": SAMPLE_INTERVAL_SECONDS,
            "maximumLeaseGapSeconds": MAXIMUM_LEASE_GAP_NS / 1e9,
            "sampleCount": len(self.samples),
            "lastSample": self.samples[-1] if self.samples else None,
            "samplesSha256": digest(self.samples),
            "fullRunCapacityProven": False,
            "qualificationEligible": False,
            "publicationEligible": False,
            "autonomyEligible": False,
        }


HERE = Path(__file__).resolve().parent
RUNS_ROOT = Path("/home/rosie/szl-runs/receiptagent-v3-supervised")
POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
SYSTEMCTL = "/usr/bin/systemctl"
MAXIMUM_ARTIFACT_BYTES = 8 * 1024 * 1024
ADMISSION_TIMEOUT_SECONDS = 15
CONTROL_TIMEOUT_SECONDS = 2
STOP_TIMEOUT_SECONDS = 20
TERMINAL_SCHEMA = "szl.windows-host-memory-terminal/v1"
UNIT_FIELDS = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "InvocationID",
    "ControlGroup",
    "MainPID",
    "ExecMainPID",
    "ExecMainStatus",
    "Result",
    "BindsTo",
    "After",
    "KillMode",
    "SendSIGKILL",
    "RemainAfterExit",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def decode_object(raw: bytes) -> dict[str, Any]:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate key")
            value[key] = item
        return value

    def reject(_):
        raise ValueError("nonfinite number")

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=reject
        )
        if not isinstance(value, dict):
            raise ValueError("object required")
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise HostMemoryError("HOST_MEMORY_SAMPLE_INVALID") from exc


def read_regular(path: Path, maximum: int = MAXIMUM_ARTIFACT_BYTES) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum
        ):
            raise HostMemoryError("HOST_MEMORY_EVIDENCE_FAILED")
        with os.fdopen(os.dup(fd), "rb") as stream:
            raw = stream.read(maximum + 1)
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if len(raw) > maximum or any(
            getattr(before, f) != getattr(after, f) for f in fields
        ):
            raise HostMemoryError("HOST_MEMORY_EVIDENCE_FAILED")
        return raw
    finally:
        os.close(fd)


def write_json(path: Path, value: dict[str, Any], *, replace: bool = False) -> None:
    data = canonical_bytes(value) + b"\n"
    if len(data) > MAXIMUM_ARTIFACT_BYTES:
        raise HostMemoryError("HOST_MEMORY_EVIDENCE_FAILED")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path, follow_symlinks=False)
            temporary.unlink()
        directory_fd = os.open(
            path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


class Evidence:
    """Supervisor-owned files are outside the worker filesystem namespace."""

    def __init__(self, reports: Path, run_id: str, source_revision: str):
        self.reports, self.run_id, self.source_revision = (
            reports,
            run_id,
            source_revision,
        )
        if reports.is_symlink() or reports.resolve(strict=True) != reports.absolute():
            raise HostMemoryError("HOST_MEMORY_EVIDENCE_FAILED")
        self.lease_path = reports / "host-memory-lease.json"
        self.terminal_path = reports / "host-memory-terminal.json"
        self.abort_path = reports / "host-memory-abort.json"
        self.request_path = reports / "host-memory-finalize.json"

    def lease(self, lease: MemoryLease) -> None:
        # Each observed frame is durably retained before publishing its usable lease.
        sequence = len(lease.samples) - 1
        if sequence >= 0:
            path = self.reports / f"host-memory-sample-{sequence:04d}.json"
            try:
                write_json(path, lease.samples[-1])
            except FileExistsError:
                if decode_object(read_regular(path)) != lease.samples[-1]:
                    raise HostMemoryError("HOST_MEMORY_EVIDENCE_FAILED")
        write_json(self.lease_path, lease.public(), replace=True)

    def abort(self, reason: str) -> None:
        reason = reason if reason in FAILURE_CODES else "HOST_MEMORY_ABORTED"
        try:
            write_json(
                self.abort_path,
                {
                    "runId": self.run_id,
                    "sourceRevision": self.source_revision,
                    "reason": reason,
                    "observedAt": utc_now().isoformat(),
                },
            )
        except FileExistsError:
            pass  # The earlier sticky veto remains authoritative.

    def require_no_abort(self) -> None:
        # lstat also catches a dangling symlink; an unreadable abort is still a veto.
        try:
            self.abort_path.lstat()
        except FileNotFoundError:
            return
        reason = "HOST_MEMORY_ABORTED"
        try:
            value = decode_object(read_regular(self.abort_path, 4096))
            if (
                value.get("runId") == self.run_id
                and value.get("sourceRevision") == self.source_revision
                and isinstance(value.get("reason"), str)
                and value["reason"] in FAILURE_CODES
            ):
                reason = value["reason"]
        except (OSError, HostMemoryError):
            pass
        raise HostMemoryError(reason)

    def terminal(self, receipt: dict[str, Any]) -> None:
        self.require_no_abort() if receipt.get("state") == "FINAL_HEALTHY" else None
        value = {**receipt, "reportSha256": digest(receipt)}
        write_json(self.terminal_path, value)


def unit_names(run_id: str) -> dict[str, str]:
    if not isinstance(run_id, str) or not HEX32.fullmatch(run_id):
        raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")
    return {
        kind: f"szl-ra3-{kind}-{run_id}.service"
        for kind in ("supervisor", "worker", "host-memory")
    }


def validated_interop_socket(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"/run/WSL/[0-9]+_interop", value
    ):
        raise HostMemoryError("HOST_MEMORY_HELPER_UNAVAILABLE")
    path = Path(value)
    info = path.lstat()
    if not stat.S_ISSOCK(info.st_mode) or path.resolve(strict=True) != path:
        raise HostMemoryError("HOST_MEMORY_HELPER_UNAVAILABLE")
    if info.st_uid not in {0, os.getuid()}:
        raise HostMemoryError("HOST_MEMORY_HELPER_UNAVAILABLE")
    return value


class UnitControl:
    """Only exact same-run units and their retained invocation IDs can be stopped."""

    def __init__(self, run_id: str):
        self.names = unit_names(run_id)
        self.worker_identity: dict[str, str] | None = None
        self.identity: dict[str, Any] = {}

    def properties(self, name: str) -> dict[str, str]:
        if name not in self.names.values():
            raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")
        result = subprocess.run(
            [SYSTEMCTL, "--user", "show", name, "--property=" + ",".join(UNIT_FIELDS)],
            capture_output=True,
            timeout=CONTROL_TIMEOUT_SECONDS,
            check=False,
        )
        if len(result.stdout) > 16384 or len(result.stderr) > 16384:
            raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")
        try:
            values = dict(
                line.split("=", 1)
                for line in result.stdout.decode("utf-8").splitlines()
            )
        except (ValueError, UnicodeError) as exc:
            raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH") from exc
        if values.get("LoadState") == "not-found" and not values.get("InvocationID"):
            return values
        if result.returncode or values.get("Id") != name:
            raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")
        return values

    @staticmethod
    def identify(properties: dict[str, str]) -> dict[str, str]:
        invocation = properties.get("InvocationID", "")
        group = properties.get("ControlGroup", "")
        name = properties.get("Id", "")
        path = PurePosixPath(group)
        if (
            not HEX32.fullmatch(invocation)
            or not group.startswith("/")
            or ".." in path.parts
            or path.name != name
        ):
            raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")
        return {"unit": name, "invocationId": invocation, "controlGroup": group}

    def bind(self) -> None:
        outer = self.properties(self.names["supervisor"])
        helper = self.properties(self.names["host-memory"])
        if (
            outer.get("ActiveState") != "active"
            or helper.get("ActiveState") != "active"
        ):
            raise HostMemoryError("HOST_MEMORY_HELPER_UNAVAILABLE")
        if (
            self.names["supervisor"] not in helper.get("BindsTo", "").split()
            or self.names["supervisor"] not in helper.get("After", "").split()
            or helper.get("KillMode") != "control-group"
            or helper.get("SendSIGKILL") != "yes"
            or helper.get("RemainAfterExit") != "yes"
        ):
            raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")
        self.identity = {
            "supervisor": self.identify(outer),
            "helper": self.identify(helper),
        }

    def require_current_helper(self) -> None:
        helper = self.properties(self.names["host-memory"])
        unified = [
            line[3:]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        ]
        if (
            helper.get("MainPID") != str(os.getpid())
            or len(unified) != 1
            or unified[0] != self.identity["helper"]["controlGroup"]
        ):
            raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")

    def observe_worker(self) -> dict[str, str] | None:
        value = self.properties(self.names["worker"])
        if value.get("LoadState") == "not-found" and self.worker_identity is None:
            return None
        identity = self.identify(value)
        dependencies = {self.names["supervisor"], self.names["host-memory"]}
        if (
            not dependencies <= set(value.get("BindsTo", "").split())
            or not dependencies <= set(value.get("After", "").split())
            or value.get("KillMode") != "control-group"
            or value.get("SendSIGKILL") != "yes"
            or (self.worker_identity is not None and identity != self.worker_identity)
        ):
            raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")
        self.worker_identity = identity
        return value

    @staticmethod
    def empty(identity: dict[str, str]) -> bool:
        path = Path("/sys/fs/cgroup") / identity["controlGroup"].lstrip("/")
        try:
            raw = read_regular(path / "cgroup.events", 4096).decode("ascii")
        except FileNotFoundError:
            # A retained exact unit with its former cgroup removed has no descendants.
            return not path.exists()
        fields = dict(line.split() for line in raw.splitlines())
        return fields.get("populated") == "0"

    def completed_worker(self) -> dict[str, str]:
        value = self.observe_worker()
        if (
            value is None
            or self.worker_identity is None
            or value.get("MainPID") != "0"
            or value.get("ExecMainStatus") != "0"
            or value.get("Result") != "success"
            or value.get("SubState") != "exited"
            or not self.empty(self.worker_identity)
        ):
            raise HostMemoryError("HOST_MEMORY_STOP_UNCONFIRMED")
        return self.worker_identity

    def stop_worker(self) -> dict[str, Any]:
        outcome = {
            "workerUnit": self.names["worker"],
            "cgroupEmptyConfirmed": False,
            "identityMatched": False,
            "stopRequested": False,
        }
        try:
            value = self.observe_worker()
            if value is None:
                return {
                    **outcome,
                    "workerNotCreated": True,
                    "cgroupEmptyConfirmed": True,
                }
            outcome["identityMatched"] = True
            outcome["stopRequested"] = True
            try:
                result = subprocess.run(
                    [SYSTEMCTL, "--user", "stop", self.names["worker"]],
                    timeout=STOP_TIMEOUT_SECONDS,
                    capture_output=True,
                    check=False,
                )
                if result.returncode:
                    raise subprocess.TimeoutExpired(
                        "exact worker stop unsuccessful", STOP_TIMEOUT_SECONDS
                    )
            except subprocess.TimeoutExpired:
                # Recheck invocation before escalation; never signal a replacement unit.
                self.observe_worker()
                subprocess.run(
                    [
                        SYSTEMCTL,
                        "--user",
                        "kill",
                        "--signal=KILL",
                        self.names["worker"],
                    ],
                    timeout=CONTROL_TIMEOUT_SECONDS,
                    capture_output=True,
                    check=False,
                )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                current = self.properties(self.names["worker"])
                if current.get("InvocationID") not in {
                    "",
                    self.worker_identity["invocationId"],
                }:
                    raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")
                if self.empty(self.worker_identity):
                    outcome["cgroupEmptyConfirmed"] = True
                    return outcome
                time.sleep(0.05)
        except (OSError, ValueError, subprocess.SubprocessError, HostMemoryError):
            pass
        return outcome

    def stop_outer(self) -> None:
        value = self.identify(self.properties(self.names["supervisor"]))
        if value != self.identity.get("supervisor"):
            raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")
        subprocess.run(
            [SYSTEMCTL, "--user", "--no-block", "stop", self.names["supervisor"]],
            timeout=CONTROL_TIMEOUT_SECONDS,
            capture_output=True,
            check=False,
        )


class GuardSession:
    def __init__(
        self,
        lease: MemoryLease,
        evidence: Evidence,
        units: UnitControl,
        guard_source_sha256: str | None = None,
    ):
        self.lease, self.evidence, self.units = lease, evidence, units
        self.guard_source_sha256 = guard_source_sha256
        self.finished = False

    def receipt(
        self, state: str, *, now_ns: int, request=None, stop=None
    ) -> dict[str, Any]:
        return {
            "schema": TERMINAL_SCHEMA,
            "state": state,
            "runId": self.lease.run_id,
            "guardSourceSha256": self.guard_source_sha256,
            "sourceRevision": self.lease.source_revision,
            "lease": self.lease.public(),
            "samples": self.lease.samples,
            "units": self.units.identity,
            "workerIdentity": self.units.worker_identity,
            "finalizeRequest": request,
            "observedAt": utc_now().isoformat(),
            "observedMonotonicNs": now_ns,
            "workerCgroupEmptyConfirmed": state == "FINAL_HEALTHY",
            "stop": stop,
            "securityBoundary": "COOPERATIVE_SAME_ACCOUNT",
            "integrityDigestIsAuthentication": False,
            "fullRunCapacityProven": False,
            "qualificationEligible": False,
            "publicationEligible": False,
            "autonomyEligible": False,
        }

    def fail(self, reason: str) -> None:
        self.lease.fail(reason)
        # Persist the veto and native witness BEFORE requesting exact-worker shutdown.
        for operation in (
            lambda: self.evidence.abort(self.lease.failure),
            lambda: self.evidence.lease(self.lease),
        ):
            try:
                operation()
            except (OSError, HostMemoryError):
                pass  # Loss of evidence cannot postpone containment.
        stop = self.units.stop_worker()
        self.finished = True
        try:
            self.evidence.terminal(
                self.receipt("FAILED", now_ns=time.monotonic_ns(), stop=stop)
            )
        except (OSError, HostMemoryError):
            pass
        if not stop.get("cgroupEmptyConfirmed"):
            self.units.stop_outer()

    def observe(self, row: dict[str, Any], *, now_ns: int, now: datetime) -> None:
        try:
            self.lease.observe(row, received_ns=now_ns, now=now)
        finally:
            self.evidence.lease(self.lease)
        self.evidence.require_no_abort()
        self.units.observe_worker()

    def finalize_if_requested(self, now_ns: int) -> bool:
        if not self.evidence.request_path.exists():
            return False
        request = decode_object(read_regular(self.evidence.request_path, 4096))
        validate_finalize_request(
            request, self.lease.run_id, self.lease.source_revision
        )
        self.evidence.require_no_abort()
        if (
            self.lease.last_received_ns is None
            or self.lease.last_received_ns < request["requestedMonotonicNs"]
        ):
            return False  # Wait for a fresh native observation AFTER the completion request.
        self.lease.assert_fresh(now_ns)
        if (
            self.lease.samples[0]["receivedMonotonicNs"]
            > request["workerStartedMonotonicNs"]
        ):
            raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")
        self.units.completed_worker()
        self.evidence.terminal(
            self.receipt("FINAL_HEALTHY", now_ns=now_ns, request=request)
        )
        self.finished = True
        return True


def validate_finalize_request(
    value: dict[str, Any], run_id: str, source_revision: str
) -> None:
    if (
        set(value)
        != {
            "runId",
            "sourceRevision",
            "requestedMonotonicNs",
            "workerStartedMonotonicNs",
            "workerEndedMonotonicNs",
        }
        or value["runId"] != run_id
        or value["sourceRevision"] != source_revision
        or any(not exact_integer(value[k]) for k in value if k.endswith("MonotonicNs"))
        or not value["workerStartedMonotonicNs"]
        <= value["workerEndedMonotonicNs"]
        <= value["requestedMonotonicNs"]
    ):
        raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")


def verify_terminal_receipt(
    receipt: dict[str, Any],
    *,
    run_id: str,
    source_revision: str,
    sampler_sha256: str,
    guard_sha256: str,
    worker_cgroup: str,
    worker_started_ns: int,
    worker_ended_ns: int,
) -> None:
    """Replay all native and WSL sample bindings; a digest alone is insufficient."""
    keys = {
        "schema",
        "state",
        "runId",
        "sourceRevision",
        "guardSourceSha256",
        "lease",
        "samples",
        "units",
        "workerIdentity",
        "finalizeRequest",
        "observedAt",
        "observedMonotonicNs",
        "workerCgroupEmptyConfirmed",
        "stop",
        "securityBoundary",
        "integrityDigestIsAuthentication",
        "fullRunCapacityProven",
        "qualificationEligible",
        "publicationEligible",
        "autonomyEligible",
        "reportSha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != keys:
        raise HostMemoryError("HOST_MEMORY_EVIDENCE_FAILED")
    unsigned = {key: value for key, value in receipt.items() if key != "reportSha256"}
    expected = {
        "schema": TERMINAL_SCHEMA,
        "state": "FINAL_HEALTHY",
        "runId": run_id,
        "sourceRevision": source_revision,
        "guardSourceSha256": guard_sha256,
        "workerCgroupEmptyConfirmed": True,
        "stop": None,
        "securityBoundary": "COOPERATIVE_SAME_ACCOUNT",
        "integrityDigestIsAuthentication": False,
        "fullRunCapacityProven": False,
        "qualificationEligible": False,
        "publicationEligible": False,
        "autonomyEligible": False,
    }
    if (
        any(
            type(receipt[k]) is not type(v) or receipt[k] != v
            for k, v in expected.items()
        )
        or digest(unsigned) != receipt["reportSha256"]
    ):
        raise HostMemoryError("HOST_MEMORY_EVIDENCE_FAILED")
    names = unit_names(run_id)
    units = receipt["units"]
    if not isinstance(units, dict) or set(units) != {"supervisor", "helper"}:
        raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")
    for label, expected_unit in (
        ("supervisor", names["supervisor"]),
        ("helper", names["host-memory"]),
        ("worker", names["worker"]),
    ):
        identity = receipt["workerIdentity"] if label == "worker" else units[label]
        if (
            not isinstance(identity, dict)
            or set(identity) != {"unit", "invocationId", "controlGroup"}
            or identity["unit"] != expected_unit
            or not isinstance(identity["invocationId"], str)
            or not HEX32.fullmatch(identity["invocationId"])
            or not isinstance(identity["controlGroup"], str)
        ):
            raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")
        UnitControl.identify(
            {
                "Id": identity["unit"],
                "InvocationID": identity["invocationId"],
                "ControlGroup": identity["controlGroup"],
            }
        )
    if receipt["workerIdentity"]["controlGroup"] != worker_cgroup:
        raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")
    request = receipt["finalizeRequest"]
    if not isinstance(request, dict):
        raise HostMemoryError("HOST_MEMORY_EVIDENCE_FAILED")
    validate_finalize_request(request, run_id, source_revision)
    if (
        request["workerStartedMonotonicNs"] != worker_started_ns
        or request["workerEndedMonotonicNs"] != worker_ended_ns
    ):
        raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")
    samples = receipt["samples"]
    if not isinstance(samples, list) or not 2 <= len(samples) <= MAXIMUM_SAMPLES:
        raise HostMemoryError("HOST_MEMORY_EVIDENCE_FAILED")
    lease = MemoryLease(run_id, source_revision, sampler_sha256)
    try:
        for observation in samples:
            if not isinstance(observation, dict) or set(observation) != {
                "native",
                "receivedMonotonicNs",
                "receivedAt",
            }:
                raise HostMemoryError("HOST_MEMORY_EVIDENCE_FAILED")
            now = datetime.fromisoformat(observation["receivedAt"])
            lease.observe(
                observation["native"],
                received_ns=observation["receivedMonotonicNs"],
                now=now,
            )
        terminal_time = datetime.fromisoformat(receipt["observedAt"])
        if (
            terminal_time.tzinfo is None
            or terminal_time.utcoffset().total_seconds() != 0
        ):
            raise ValueError("UTC required")
    except (TypeError, ValueError, OverflowError) as exc:
        raise HostMemoryError("HOST_MEMORY_EVIDENCE_FAILED") from exc
    lease.assert_fresh(receipt["observedMonotonicNs"])
    if (
        lease.public() != receipt["lease"]
        or samples[0]["receivedMonotonicNs"] > worker_started_ns
        or lease.last_received_ns < request["requestedMonotonicNs"]
        or abs(
            (
                terminal_time - datetime.fromisoformat(samples[-1]["receivedAt"])
            ).total_seconds()
        )
        > 8
    ):
        raise HostMemoryError("HOST_MEMORY_EVIDENCE_FAILED")


def verify_terminal_binding(
    binding: Any,
    *,
    reports: Path,
    run_id: str,
    source_revision: str,
    sampler_sha256: str,
    guard_sha256: str,
    launch: dict[str, Any],
) -> dict[str, Any]:
    evidence = Evidence(reports, run_id, source_revision)
    evidence.require_no_abort()
    if (
        not isinstance(binding, dict)
        or set(binding)
        != {"relativePath", "fileSha256", "bytes", "state", "helperExitConfirmed"}
        or binding["relativePath"] != "reports/host-memory-terminal.json"
        or binding["helperExitConfirmed"] is not True
        or binding["state"] != "FINAL_HEALTHY"
        or not exact_integer(binding["bytes"], positive=True)
    ):
        raise HostMemoryError("HOST_MEMORY_EVIDENCE_FAILED")
    raw = read_regular(evidence.terminal_path)
    if (
        len(raw) != binding["bytes"]
        or hashlib.sha256(raw).hexdigest() != binding["fileSha256"]
    ):
        raise HostMemoryError("HOST_MEMORY_EVIDENCE_FAILED")
    receipt = decode_object(raw)
    request = decode_object(read_regular(evidence.request_path, 4096))
    if receipt.get("finalizeRequest") != request:
        raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")
    verify_terminal_receipt(
        receipt,
        run_id=run_id,
        source_revision=source_revision,
        sampler_sha256=sampler_sha256,
        guard_sha256=guard_sha256,
        worker_cgroup=launch["workerControlGroup"],
        worker_started_ns=launch["startedMonotonicNs"],
        worker_ended_ns=launch["endedMonotonicNs"],
    )
    evidence.require_no_abort()  # A late abort remains a veto even after a healthy seal.
    return {
        "terminalFileSha256": binding["fileSha256"],
        "state": "FINAL_HEALTHY",
        "terminalReportSha256": receipt["reportSha256"],
        "sampleCount": receipt["lease"]["sampleCount"],
        "samplesSha256": receipt["lease"]["samplesSha256"],
        "fullRunCapacityProven": False,
    }


def validate_report_bundle(
    supervisor: dict[str, Any], receipt: dict[str, Any], source_revision: str
) -> str:
    """Authentication/release replays the same native evidence before binding it."""
    try:
        binding = supervisor["hostMemoryGuard"]
        if (
            not isinstance(binding, dict)
            or set(binding)
            != {"relativePath", "fileSha256", "bytes", "state", "helperExitConfirmed"}
            or binding["relativePath"] != "reports/host-memory-terminal.json"
            or binding["helperExitConfirmed"] is not True
            or binding["state"] != "FINAL_HEALTHY"
            or not exact_integer(binding["bytes"], positive=True)
        ):
            raise HostMemoryError("HOST_MEMORY_EVIDENCE_FAILED")
        raw = canonical_bytes(receipt) + b"\n"
        if (
            binding["bytes"] != len(raw)
            or binding["fileSha256"] != hashlib.sha256(raw).hexdigest()
        ):
            raise HostMemoryError("HOST_MEMORY_EVIDENCE_FAILED")
        components = supervisor["source"]["components"]
        sampler_sha = components["windows_host_memory_sampler.ps1"]["sha256"]
        guard_sha = components["host_memory_guard.py"]["sha256"]
        if not isinstance(guard_sha, str) or not HEX64.fullmatch(guard_sha):
            raise HostMemoryError("HOST_MEMORY_EVIDENCE_FAILED")
        launch = supervisor["launch"]
        verify_terminal_receipt(
            receipt,
            run_id=supervisor["runId"],
            source_revision=source_revision,
            sampler_sha256=sampler_sha,
            guard_sha256=guard_sha,
            worker_cgroup=launch["workerControlGroup"],
            worker_started_ns=launch["startedMonotonicNs"],
            worker_ended_ns=launch["endedMonotonicNs"],
        )
        return receipt["reportSha256"]
    except (KeyError, TypeError, ValueError) as exc:
        raise HostMemoryError("HOST_MEMORY_EVIDENCE_FAILED") from exc


class GuardClient:
    """The GPU supervisor consults leases; systemd independently binds the worker."""

    def __init__(self, *, policy, attempt, source):
        self.policy, self.attempt, self.source = policy, attempt, source
        self.evidence = Evidence(attempt.reports, attempt.run_id, source.revision)
        self.units = UnitControl(attempt.run_id)
        self.started = False

    def start(self) -> dict[str, Any]:
        interop = validated_interop_socket(os.environ.get("WSL_INTEROP", ""))
        if not Path(POWERSHELL).is_file():
            raise HostMemoryError("HOST_MEMORY_HELPER_UNAVAILABLE")
        command = [
            self.policy["systemd_run_executable"],
            "--user",
            "--unit=" + self.units.names["host-memory"],
            "--service-type=exec",
            "--property=BindsTo=" + self.units.names["supervisor"],
            "--property=After=" + self.units.names["supervisor"],
            "--property=KillMode=control-group",
            "--property=SendSIGKILL=yes",
            "--property=RemainAfterExit=yes",
            "--property=TimeoutStopSec=45s",
            "--property=UMask=0077",
            "--property=NoNewPrivileges=yes",
            "--property=ProtectControlGroups=yes",
            "--property=LimitCORE=0",
            "--property=OOMPolicy=stop",
            "--",
            "/usr/bin/env",
            "-i",
            "HOME=/home/rosie",
            "USER=rosie",
            "LOGNAME=rosie",
            "PATH=/usr/bin:/bin",
            "LANG=C.UTF-8",
            "LC_ALL=C.UTF-8",
            "XDG_RUNTIME_DIR=/run/user/1000",
            "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus",
            "WSL_INTEROP=" + interop,
            "PYTHONNOUSERSITE=1",
            "PYTHONDONTWRITEBYTECODE=1",
            self.policy["python_executable"],
            "-I",
            "-B",
            str(HERE / "host_memory_guard.py"),
            "--source-commit",
            self.source.revision,
            "--run-id",
            self.attempt.run_id,
        ]
        self.started = True  # A partial launch still requires exact cleanup.
        subprocess.run(
            command, timeout=CONTROL_TIMEOUT_SECONDS, capture_output=True, check=True
        )
        deadline = time.monotonic() + ADMISSION_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            self.evidence.require_no_abort()
            if self.evidence.lease_path.exists():
                lease = self.assert_healthy()
                self.units.bind()
                return {
                    "state": "ADMITTED_PROVISIONAL",
                    "lease": lease,
                    "units": self.units.identity,
                    "fullRunCapacityProven": False,
                }
            if self.evidence.terminal_path.exists():
                raise HostMemoryError("HOST_MEMORY_HELPER_UNAVAILABLE")
            time.sleep(0.05)
        raise HostMemoryError("HOST_MEMORY_HELPER_UNAVAILABLE")

    def assert_healthy(self) -> dict[str, Any]:
        self.evidence.require_no_abort()
        try:
            lease = decode_object(read_regular(self.evidence.lease_path, 16384))
        except OSError as exc:
            raise HostMemoryError("HOST_MEMORY_EVIDENCE_FAILED") from exc
        expected = MemoryLease(
            self.attempt.run_id,
            self.source.revision,
            self.source.component("windows_host_memory_sampler.ps1").sha256,
        ).public()
        variable = {"state", "failure", "sampleCount", "lastSample", "samplesSha256"}
        if (
            set(lease) != set(expected)
            or any(
                type(lease[k]) is not type(v) or lease[k] != v
                for k, v in expected.items()
                if k not in variable
            )
            or lease["state"] != "HEALTHY"
            or lease["failure"] is not None
            or not exact_integer(lease["sampleCount"], positive=True)
            or lease["sampleCount"] > MAXIMUM_SAMPLES
            or not isinstance(lease["samplesSha256"], str)
            or not HEX64.fullmatch(lease["samplesSha256"])
            or not isinstance(lease["lastSample"], dict)
        ):
            reason = lease.get("failure")
            raise HostMemoryError(
                reason
                if isinstance(reason, str) and reason in FAILURE_CODES
                else "HOST_MEMORY_EVIDENCE_FAILED"
            )
        sample = lease["lastSample"]
        if (
            set(sample) != {"native", "receivedMonotonicNs", "receivedAt"}
            or not exact_integer(sample["receivedMonotonicNs"])
            or not 0
            <= time.monotonic_ns() - sample["receivedMonotonicNs"]
            <= MAXIMUM_LEASE_GAP_NS
        ):
            raise HostMemoryError("HOST_MEMORY_LEASE_EXPIRED")
        try:
            native = dict(sample["native"])
            if native["sequence"] != lease["sampleCount"] - 1:
                raise HostMemoryError("HOST_MEMORY_SAMPLE_INVALID")
            native["sequence"] = (
                0  # Validate this last frame against a single-frame lease.
            )
            one = MemoryLease(
                self.attempt.run_id,
                self.source.revision,
                self.source.component("windows_host_memory_sampler.ps1").sha256,
            )
            one.observe(
                native,
                received_ns=sample["receivedMonotonicNs"],
                now=datetime.fromisoformat(sample["receivedAt"]),
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise HostMemoryError("HOST_MEMORY_SAMPLE_INVALID") from exc
        return lease

    def finalize(self, launch: dict[str, Any]) -> dict[str, Any]:
        self.assert_healthy()
        request = {
            "runId": self.attempt.run_id,
            "sourceRevision": self.source.revision,
            "requestedMonotonicNs": time.monotonic_ns(),
            "workerStartedMonotonicNs": launch["startedMonotonicNs"],
            "workerEndedMonotonicNs": launch["endedMonotonicNs"],
        }
        validate_finalize_request(request, self.attempt.run_id, self.source.revision)
        write_json(self.evidence.request_path, request)
        deadline = time.monotonic() + MAXIMUM_LEASE_GAP_NS / 1e9
        while time.monotonic() < deadline:
            self.evidence.require_no_abort()
            if self.evidence.terminal_path.exists():
                # Wait for the exact helper to exit successfully. This closes the
                # write-once link/fsync window and rejects a crash after a healthy seal.
                helper = self.units.properties(self.units.names["host-memory"])
                if (
                    helper.get("InvocationID")
                    != self.units.identity["helper"]["invocationId"]
                ):
                    raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")
                if helper.get("MainPID") != "0" and helper.get("ActiveState") in {
                    "active",
                    "activating",
                    "deactivating",
                }:
                    time.sleep(0.05)
                    continue
                if (
                    helper.get("LoadState") != "loaded"
                    or helper.get("Result") != "success"
                    or helper.get("ExecMainStatus") != "0"
                    or helper.get("MainPID") != "0"
                    or helper.get("ActiveState") != "active"
                    or helper.get("SubState") != "exited"
                ):
                    raise HostMemoryError("HOST_MEMORY_WATCHER_EXITED")
                raw = read_regular(self.evidence.terminal_path)
                recorded = decode_object(raw)
                if (
                    recorded.get("units") != self.units.identity
                    or recorded.get("finalizeRequest") != request
                ):
                    raise HostMemoryError("HOST_MEMORY_IDENTITY_MISMATCH")
                binding = {
                    "relativePath": "reports/host-memory-terminal.json",
                    "fileSha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                    "state": "FINAL_HEALTHY",
                    "helperExitConfirmed": True,
                }
                verify_terminal_binding(
                    binding,
                    reports=self.attempt.reports,
                    run_id=self.attempt.run_id,
                    source_revision=self.source.revision,
                    sampler_sha256=self.source.component(
                        "windows_host_memory_sampler.ps1"
                    ).sha256,
                    guard_sha256=self.source.component("host_memory_guard.py").sha256,
                    launch=launch,
                )
                return binding
            self.assert_healthy()
            time.sleep(0.05)
        raise HostMemoryError("HOST_MEMORY_LEASE_EXPIRED")

    def abort(self, reason: str) -> dict[str, Any]:
        try:
            self.evidence.abort(reason)
        finally:
            if self.started:
                # The helper stops its exact worker before exiting. BindsTo is a second path.
                try:
                    subprocess.run(
                        [SYSTEMCTL, "--user", "stop", self.units.names["host-memory"]],
                        timeout=45,
                        capture_output=True,
                        check=False,
                    )
                except (OSError, subprocess.SubprocessError):
                    pass
        return {
            "state": "ABORTED_NOT_ELIGIBLE",
            "failure": reason,
            "terminalReceiptPresent": self.evidence.terminal_path.is_file(),
        }


def native_command(source, run_id: str) -> list[str]:
    sampler = source.component("windows_host_memory_sampler.ps1")
    MemoryLease(run_id, source.revision, sampler.sha256)
    script = "& {\n" + sampler.source_bytes.decode("utf-8") + "\n} "
    script += f"-RunId {run_id} -SourceRevision {source.revision} -SamplerSourceSha256 {sampler.sha256}"
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    if len(encoded) > 30000:
        raise HostMemoryError("HOST_MEMORY_HELPER_UNAVAILABLE")
    return [
        POWERSHELL,
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        encoded,
    ]


def run_guard(source, run_id: str) -> int:
    evidence = Evidence(RUNS_ROOT / run_id / "reports", run_id, source.revision)
    units = UnitControl(run_id)
    lease = MemoryLease(
        run_id,
        source.revision,
        source.component("windows_host_memory_sampler.ps1").sha256,
    )
    session = GuardSession(
        lease, evidence, units, source.component("host_memory_guard.py").sha256
    )
    process = None

    def interrupted(_signum, _frame):
        raise HostMemoryError("HOST_MEMORY_ABORTED")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        units.bind()
        units.require_current_helper()
        validated_interop_socket(os.environ.get("WSL_INTEROP", ""))
        process = subprocess.Popen(
            native_command(source, run_id),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        os.set_blocking(process.stdout.fileno(), False)
        buffer = bytearray()
        started = time.monotonic()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while not session.finished:
                evidence.require_no_abort()
                if process.poll() is not None:
                    raise HostMemoryError("HOST_MEMORY_WATCHER_EXITED")
                if lease.samples:
                    lease.assert_fresh(time.monotonic_ns())
                elif time.monotonic() - started > ADMISSION_TIMEOUT_SECONDS:
                    raise HostMemoryError("HOST_MEMORY_LEASE_EXPIRED")
                for key, _events in selector.select(timeout=0.1):
                    chunk = os.read(key.fd, 4096)
                    if not chunk:
                        raise HostMemoryError("HOST_MEMORY_WATCHER_EXITED")
                    buffer.extend(chunk)
                    if len(buffer) > 8192:
                        raise HostMemoryError("HOST_MEMORY_SAMPLE_INVALID")
                    while b"\n" in buffer:
                        row, _, rest = buffer.partition(b"\n")
                        buffer = bytearray(rest)
                        session.observe(
                            decode_object(row),
                            now_ns=time.monotonic_ns(),
                            now=utc_now(),
                        )
                session.finalize_if_requested(time.monotonic_ns())
        return 0
    except (
        Exception,
        KeyboardInterrupt,
    ) as exc:  # bounded terminal path, never print provider text
        reason = (
            str(exc)
            if isinstance(exc, HostMemoryError)
            else "HOST_MEMORY_HELPER_UNAVAILABLE"
        )
        try:
            session.fail(reason)
        except (OSError, HostMemoryError, subprocess.SubprocessError):
            pass  # Exiting this BindsTo service independently stops the worker.
        return 81
    finally:
        if process is not None:
            try:
                process.stdin.close()  # The native sampler exits when its controlling pipe closes.
                process.wait(timeout=3)
            except (OSError, subprocess.SubprocessError):
                process.kill()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            finally:
                process.stdout.close()


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if (
        len(args) != 4
        or args[0] != "--source-commit"
        or args[2] != "--run-id"
        or not HEX40.fullmatch(args[1])
        or not HEX32.fullmatch(args[3])
    ):
        raise SystemExit("exact source commit and run ID are required")
    spec = importlib.util.spec_from_file_location(
        "szl_ra3_memory_bootstrap", HERE / "supervisor_bootstrap.py"
    )
    bootstrap = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bootstrap
    spec.loader.exec_module(bootstrap)
    # The trusted supervisor already verified this entry point. Rebind every source
    # component before starting native observations; no model or held-out import.
    source = bootstrap.verify_exact_source_before_import(args[1])
    return run_guard(source, args[3])


if __name__ == "__main__":
    raise SystemExit(main())
