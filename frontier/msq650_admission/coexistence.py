# SPDX-License-Identifier: Apache-2.0
"""Pure pre-launch consistency checks; never a shared-GPU isolation permit.

Inputs are bounded canonical JSON bytes. No hardware read, process operation,
filesystem/network access, signature verification or worker release occurs here.
Capacity and identity observations are caller-supplied, hence MODELED, not fresh
measurements established by this module. A matching hash is not signer trust.
The legacy exclusive-GPU profile is deliberately neither imported nor changed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any

CANDIDATE = "szl-msq650-pilot-v1"
MANIFEST_SHA256 = "e10294badec5f4291e5321c96a925d8629896306da24e77e5d406b3614e6389c"
MAX_BYTES = 32768
MAX_AGE_SECONDS = 30
MAX_LEASE_SECONDS = 1200
MAX_INT = (1 << 63) - 1
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")


class ObservationError(ValueError):
    pass


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ObservationError(reason)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _number(_):
    raise ObservationError("FLOAT_OR_NONFINITE_FORBIDDEN")


def _parse(raw):
    require(type(raw) is bytes and 0 < len(raw) <= MAX_BYTES, "INPUT_TYPE_OR_SIZE")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                       parse_float=_number, parse_constant=_number)
    require(canonical(value) == raw, "NONCANONICAL_JSON")
    return value


def _fields(value, expected):
    require(type(value) is dict and set(value) == set(expected.split()), "FIELD_SET_INVALID")


def _int(value, low=0, high=MAX_INT):
    require(type(value) is int and low <= value <= high, "INTEGER_OUT_OF_BOUNDS")


def _uuid(value):
    require(type(value) is str and _UUID.fullmatch(value) is not None, "UUID_INVALID")


def _hash(value):
    require(type(value) is str and _HEX.fullmatch(value) is not None, "HASH_INVALID")


def _identity(value):
    _fields(value, "hostname boot_id gpu_uuid pid start_ticks executable_sha256 model_sha256")
    require(type(value["hostname"]) is str and re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,127}", value["hostname"]) is not None, "HOSTNAME_INVALID")
    _uuid(value["boot_id"])
    require(type(value["gpu_uuid"]) is str and value["gpu_uuid"].startswith("GPU-"), "GPU_UUID_INVALID")
    _uuid(value["gpu_uuid"][4:])
    _int(value["pid"], 1, (1 << 31) - 1)
    _int(value["start_ticks"], 1)
    _hash(value["executable_sha256"])
    _hash(value["model_sha256"])


def _plan(value):
    _fields(value, "schema candidate_id manifest_sha256 campaign_id lease_id not_before_epoch expires_at_epoch incumbent resources worker_stop_scope")
    require(value["schema"] == "szl.msq650.coexistence-plan/v1" and
            value["candidate_id"] == CANDIDATE and value["manifest_sha256"] == MANIFEST_SHA256, "PLAN_PROFILE_MISMATCH")
    _uuid(value["campaign_id"])
    _uuid(value["lease_id"])
    _int(value["not_before_epoch"], 1)
    _int(value["expires_at_epoch"], 1)
    require(0 < value["expires_at_epoch"] - value["not_before_epoch"] <= MAX_LEASE_SECONDS, "LEASE_LENGTH_INVALID")
    require(value["worker_stop_scope"] == "OWNED_WORKER_SESSION_ONLY", "INCUMBENT_CONTROL_FORBIDDEN")
    _identity(value["incumbent"])
    resources = value["resources"]
    _fields(resources, "worker_gpu_memory_bytes incumbent_extra_gpu_reserve_bytes gpu_safety_bytes worker_host_memory_bytes host_safety_bytes worker_disk_bytes disk_safety_bytes worker_cpu_threads")
    for key, number in resources.items():
        _int(number, 1, 1024 if key == "worker_cpu_threads" else 1 << 40)


def _snapshot(value, plan, now_epoch):
    _fields(value, "schema observed_epoch incumbent compute_processes gpu_total_bytes gpu_free_bytes host_available_bytes disk_free_bytes cpu_threads_available health")
    require(value["schema"] == "szl.msq650.prelaunch-snapshot/v1", "SNAPSHOT_PROFILE_MISMATCH")
    _int(value["observed_epoch"], 1)
    require(0 <= now_epoch - value["observed_epoch"] <= MAX_AGE_SECONDS, "SNAPSHOT_STALE_OR_FUTURE")
    require(plan["not_before_epoch"] <= value["observed_epoch"] < plan["expires_at_epoch"], "SNAPSHOT_OUTSIDE_LEASE")
    _identity(value["incumbent"])
    require(value["incumbent"] == plan["incumbent"], "INCUMBENT_IDENTITY_CHANGED")
    processes = value["compute_processes"]
    require(type(processes) is list and len(processes) == 1, "COMPETING_OR_MISSING_GPU_PROCESS")
    _fields(processes[0], "pid start_ticks")
    _int(processes[0]["pid"], 1, (1 << 31) - 1)
    _int(processes[0]["start_ticks"], 1)
    require(processes[0] == {key: plan["incumbent"][key] for key in ("pid", "start_ticks")}, "GPU_PROCESS_IDENTITY_CHANGED")
    for key in ("gpu_total_bytes", "gpu_free_bytes", "host_available_bytes", "disk_free_bytes", "cpu_threads_available"):
        _int(value[key], 0, 1024 if key == "cpu_threads_available" else 1 << 50)
    require(0 < value["gpu_free_bytes"] <= value["gpu_total_bytes"], "GPU_MEMORY_INCONSISTENT")
    _fields(value["health"], "ready active_requests")
    require(value["health"]["ready"] is True, "INCUMBENT_NOT_HEALTHY")
    _int(value["health"]["active_requests"], 0, 1 << 31)
    require(value["health"]["active_requests"] == 0, "INCUMBENT_BUSY")
    r = plan["resources"]
    require(r["worker_gpu_memory_bytes"] + r["incumbent_extra_gpu_reserve_bytes"] + r["gpu_safety_bytes"] <= value["gpu_free_bytes"], "GPU_HEADROOM_INSUFFICIENT")
    require(r["worker_host_memory_bytes"] + r["host_safety_bytes"] <= value["host_available_bytes"], "HOST_HEADROOM_INSUFFICIENT")
    require(r["worker_disk_bytes"] + r["disk_safety_bytes"] <= value["disk_free_bytes"], "DISK_HEADROOM_INSUFFICIENT")
    require(r["worker_cpu_threads"] < value["cpu_threads_available"], "INCUMBENT_CPU_RESERVE_MISSING")


@dataclass(frozen=True)
class CoexistenceObservation:
    plan_consistent: bool
    reasons: tuple[str, ...]
    input_sha256: tuple[str, ...]
    evidence_class: str = field(default="MODELED", init=False)
    state: str = field(default="BLOCKED", init=False)
    isolation_backend: str = field(default="UNAVAILABLE", init=False)
    training_admitted: bool = field(default=False, init=False)
    execution_authority: bool = field(default=False, init=False)
    publication_eligible: bool = field(default=False, init=False)

    def __bool__(self):
        raise TypeError("COEXISTENCE_OBSERVATION_IS_NOT_AUTHORIZATION")


def observe_coexistence(plan_raw: bytes, before_raw: bytes, after_raw: bytes, *,
                        expected_plan_sha256: str, now_epoch: int) -> CoexistenceObservation:
    """Compare a pinned plan with two pre-launch snapshots; never admit a job.

    The expected plan hash and clock must ultimately come from trusted owner /
    observer channels; this pure function cannot establish their authenticity.
    The plan pins the incumbent host/boot/GPU/PID/start and executable/model
    bytes. Snapshots are pre-launch only, so exactly the incumbent is allowed.
    A future owned worker and durable single-use reservation are NOT implemented.
    """
    hashes = ()
    try:
        plan, before, after = (_parse(raw) for raw in (plan_raw, before_raw, after_raw))
        hashes = tuple(hashlib.sha256(raw).hexdigest() for raw in (plan_raw, before_raw, after_raw))
        _hash(expected_plan_sha256)
        require(hashes[0] == expected_plan_sha256, "PLAN_PIN_MISMATCH")
        _int(now_epoch, 1)
        _plan(plan)
        require(plan["not_before_epoch"] <= now_epoch < plan["expires_at_epoch"], "LEASE_NOT_CURRENT")
        _snapshot(before, plan, now_epoch)
        _snapshot(after, plan, now_epoch)
        require(before["observed_epoch"] < after["observed_epoch"], "SNAPSHOTS_NOT_ORDERED")
        require(before["gpu_total_bytes"] == after["gpu_total_bytes"], "DEVICE_CAPACITY_CHANGED")
        return CoexistenceObservation(True, ("ISOLATION_BACKEND_UNAVAILABLE", "OBSERVATION_AUTHENTICITY_UNESTABLISHED",
            "OWNER_SIGNING_POLICY_UNAVAILABLE", "DURABLE_SINGLE_USE_RESERVATION_UNAVAILABLE"), hashes)
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError) as exc:
        reason = str(exc) if isinstance(exc, ObservationError) else "INVALID_INPUT"
        return CoexistenceObservation(False, (reason, "ISOLATION_BACKEND_UNAVAILABLE"), hashes)
