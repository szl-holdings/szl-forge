#!/usr/bin/env python3
"""Supervise one ReceiptAgent v3 training worker inside a systemd cgroup.

This process observes temperature, deadline, process containment, logs, and
output bytes.  It never turns those observations into a model-quality,
receipt, publication, deployment, runtime-health, or autonomy claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RELATIVE = "frontier/qwen35-receiptagent-v3"
GPU_UUID_PATTERN = re.compile(r"GPU-[A-Za-z0-9-]{16,96}")
MAX_GPU_NAME_CHARACTERS = 128
MAX_GPU_MEMORY_MIB = 10_000_000
MAX_GPU_TEMPERATURE_C = 200
MAX_TELEMETRY_SAMPLE_JSON_BYTES = 512
MAX_NON_TELEMETRY_REPORT_BYTES = 2 * 1024 * 1024


def load_sibling(filename: str, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, HERE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load required sibling module {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


trainer = load_sibling("train_candidate.py", "szl_ra3_train_candidate")
supervisor_validation = load_sibling(
    "supervisor_validation.py", "szl_ra3_supervisor_validation"
)


RUN_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
SUPERVISOR_UNIT_PATTERN = re.compile(r"szl-ra3-supervisor-([0-9a-f]{32})")
EXPECTED_POLICY = {
    "schema": "szl.receiptagent-v3-supervision-policy/v1",
    "runs_root": "/home/rosie/szl-runs/receiptagent-v3-supervised",
    "python_executable": "/home/rosie/.venvs/szl-unsloth/bin/python",
    "nvidia_smi_executable": "/usr/lib/wsl/lib/nvidia-smi",
    "systemd_run_executable": "/usr/bin/systemd-run",
    "systemctl_executable": "/usr/bin/systemctl",
    "cgroup_root": "/sys/fs/cgroup",
    "required_containment": "SYSTEMD_USER_SERVICE_CGROUP_V2",
    "security_boundary": "COOPERATIVE_SAME_ACCOUNT",
    "thermal_sample_interval_seconds": 2.0,
    "telemetry_timeout_seconds": 5.0,
    "maximum_telemetry_gap_seconds": 8.0,
    "smoke_wall_timeout_seconds": 1200.0,
    "full_wall_timeout_seconds": 10800.0,
    "termination_grace_seconds": 10.0,
    "kill_confirmation_seconds": 10.0,
    "maximum_log_bytes_per_stream": 67_108_864,
    "evidence_reserve_bytes": 8_388_608,
}
WORKER_ENVIRONMENT = {
    "USER": "rosie",
    "LOGNAME": "rosie",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib",
    "HF_HOME": "/home/rosie/.cache/huggingface",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
TERMINAL_EXIT_CODES = {
    "SUCCESS": 0,
    "PRECONDITION_DENIED": 70,
    "THERMAL_POLICY_VIOLATION": 71,
    "WALL_TIMEOUT": 72,
    "WORKER_EXIT_FAILURE": 73,
    "TELEMETRY_UNAVAILABLE": 74,
    "TERMINATION_UNCONFIRMED": 75,
    "EVIDENCE_DURABILITY_FAILED": 76,
    "WORKER_REPORT_INVALID": 77,
    "LOG_QUOTA_EXCEEDED": 78,
    "CONTAINMENT_UNAVAILABLE": 79,
}
OBSERVED_TRIGGER_CAUSES = {
    "THERMAL_POLICY_VIOLATION",
    "WALL_TIMEOUT",
    "TELEMETRY_UNAVAILABLE",
    "TERMINATION_UNCONFIRMED",
    "LOG_QUOTA_EXCEEDED",
    "CONTAINMENT_UNAVAILABLE",
}


class SupervisionError(RuntimeError):
    """A supervision precondition, observation, or evidence gate failed."""


class DuplicateKeyError(SupervisionError):
    """A JSON object contained a duplicate key."""


@dataclass(frozen=True)
class Attempt:
    run_id: str
    root: Path
    payload: Path
    logs: Path
    reports: Path
    runtime_cache: Path
    reserve: Path


@dataclass(frozen=True)
class TelemetrySample:
    observed_monotonic_ns: int
    observed_at: str
    gpu_uuid: str
    name: str
    temperature_c: int
    free_mib: int
    total_mib: int

    def public(self, started_ns: int) -> dict[str, Any]:
        return {
            "offsetSeconds": round((self.observed_monotonic_ns - started_ns) / 1e9, 6),
            "observedAt": self.observed_at,
            "gpuUuid": self.gpu_uuid,
            "temperatureC": self.temperature_c,
            "freeMiB": self.free_mib,
            "totalMiB": self.total_mib,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def maximum_telemetry_samples(policy: dict[str, Any]) -> int:
    """Bound initial, periodic, and terminal samples for the longest run."""

    longest_seconds = max(
        policy["smoke_wall_timeout_seconds"], policy["full_wall_timeout_seconds"]
    )
    interval_seconds = policy["thermal_sample_interval_seconds"]
    return math.ceil(longest_seconds / interval_seconds) + 2


def minimum_evidence_reserve_bytes(policy: dict[str, Any]) -> int:
    """Return the conservative maximum report bytes the reserve must cover."""

    return (
        maximum_telemetry_samples(policy) * MAX_TELEMETRY_SAMPLE_JSON_BYTES
        + MAX_NON_TELEMETRY_REPORT_BYTES
    )


def bind_committed_component(source_commit: str, filename: str) -> str:
    """Bind one executed sibling to the exact bytes at the requested commit."""

    committed = trainer.committed_bytes(source_commit, f"{RELATIVE}/{filename}")
    if HERE.joinpath(filename).read_bytes() != committed:
        raise SupervisionError(f"{filename} worktree bytes differ from exact source")
    return sha256_bytes(committed)


def merge_terminal_observation(
    current_cause: str,
    current_error: str | None,
    observation: tuple[str, str],
) -> tuple[str, str]:
    """Preserve an earlier run trigger while appending a terminal observation."""

    observed_cause, observed_error = observation
    primary = (
        current_cause if current_cause in OBSERVED_TRIGGER_CAUSES else observed_cause
    )
    combined = (
        f"{current_error}; terminal={observed_error}"
        if current_error
        else observed_error
    )
    return primary, combined


def hash_file(path: Path, maximum_bytes: int = 512 * 1024 * 1024) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SupervisionError(
                f"evidence is not a single-link regular file: {path.name}"
            )
        if metadata.st_size > maximum_bytes:
            raise SupervisionError(f"evidence exceeds its byte ceiling: {path.name}")
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > maximum_bytes:
                raise SupervisionError(
                    f"evidence grew past its byte ceiling: {path.name}"
                )
            digest.update(block)
        if total != metadata.st_size:
            raise SupervisionError(f"evidence changed while hashing: {path.name}")
        return {"bytes": total, "sha256": digest.hexdigest()}
    finally:
        os.close(descriptor)


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def strict_json_file(
    path: Path, maximum_bytes: int = 2 * 1024 * 1024
) -> dict[str, Any]:
    metadata = hash_file(path, maximum_bytes)
    raw = path.read_bytes()
    if len(raw) != metadata["bytes"] or b"\x00" in raw:
        raise SupervisionError("worker report bytes changed or contain NUL")
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SupervisionError(f"non-finite JSON number: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupervisionError("worker report is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SupervisionError("worker report must be one JSON object")
    return value


def validate_policy(candidate: dict[str, Any]) -> dict[str, Any]:
    policy = candidate.get("supervision_policy")
    if policy != EXPECTED_POLICY:
        raise SupervisionError(
            "committed supervision policy differs from the fixed contract"
        )
    if policy["evidence_reserve_bytes"] < minimum_evidence_reserve_bytes(policy):
        raise SupervisionError("evidence reserve is smaller than the bounded report")
    recipe = candidate.get("training_recipe", {})
    if recipe.get("maximum_gpu_temperature_c") != 80:
        raise SupervisionError("fixed maximum GPU temperature must remain 80 C")
    if recipe.get("minimum_free_gpu_gib") != 4.0:
        raise SupervisionError("fixed minimum free GPU memory must remain 4 GiB")
    return policy


def path_has_no_symlink(path: Path) -> None:
    if not path.is_absolute():
        raise SupervisionError("path must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise SupervisionError(f"symlink path component is forbidden: {current}")


def validate_runs_root(path: Path) -> Path:
    if str(path) != EXPECTED_POLICY["runs_root"]:
        raise SupervisionError("runs root differs from committed policy")
    path_has_no_symlink(path)
    resolved = path.resolve(strict=True)
    if not str(resolved).startswith("/home/rosie/") or str(resolved).startswith(
        "/mnt/"
    ):
        raise SupervisionError(
            "runs root must be on the trusted WSL-native home filesystem"
        )
    metadata = resolved.stat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise SupervisionError("runs root must be an owner-controlled directory")
    if metadata.st_mode & 0o022:
        raise SupervisionError("runs root cannot be group- or world-writable")
    return resolved


def fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_exclusive_file(path: Path, mode: int = 0o600) -> int:
    return os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )


def write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("write returned no progress")
        view = view[written:]


def publish_once(directory: Path, name: str, data: bytes) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,127}", name):
        raise SupervisionError("unsafe evidence filename")
    temp = directory / f".{name}.{secrets.token_hex(16)}.tmp"
    final = directory / name
    descriptor = create_exclusive_file(temp)
    try:
        write_all(descriptor, data)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)
    try:
        os.link(temp, final, follow_symlinks=False)
        fsync_directory(directory)
    finally:
        try:
            temp.unlink()
            fsync_directory(directory)
        except FileNotFoundError:
            pass
    return {"path": str(final), "bytes": len(data), "sha256": sha256_bytes(data)}


def admit_attempt(runs_root: Path, run_id: str, reserve_bytes: int) -> Attempt:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise SupervisionError("run ID must be exactly 32 lowercase hex characters")
    root_fd = os.open(
        runs_root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.mkdir(run_id, mode=0o700, dir_fd=root_fd)
        os.fsync(root_fd)
    finally:
        os.close(root_fd)
    root = runs_root / run_id
    for name in ("payload", "logs", "reports", "runtime-cache"):
        os.mkdir(root / name, mode=0o700)
    reserve = root / ".evidence-reserve"
    descriptor = create_exclusive_file(reserve)
    try:
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(descriptor, 0, reserve_bytes)
        else:
            os.ftruncate(descriptor, reserve_bytes)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    runtime_cache = root / "runtime-cache"
    for name in ("home", "xdg", "torch", "unsloth", "triton", "numba", "cuda"):
        os.mkdir(runtime_cache / name, mode=0o700)
    fsync_directory(runtime_cache)
    fsync_directory(root)
    return Attempt(
        run_id,
        root,
        root / "payload",
        root / "logs",
        root / "reports",
        runtime_cache,
        reserve,
    )


def release_evidence_reserve(attempt: Attempt) -> None:
    try:
        attempt.reserve.unlink()
    except FileNotFoundError:
        return
    fsync_directory(attempt.root)


def systemctl(
    policy: dict[str, Any], *args: str, timeout: float = 20.0
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [policy["systemctl_executable"], "--user", *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def unit_properties(policy: dict[str, Any], unit: str) -> dict[str, str]:
    names = (
        "ActiveState",
        "SubState",
        "MainPID",
        "ExecMainPID",
        "ExecMainStatus",
        "Result",
        "ControlGroup",
        "RemainAfterExit",
        "KillMode",
        "SendSIGKILL",
        "BindsTo",
        "NoNewPrivileges",
        "ProtectControlGroups",
        "ProtectSystem",
        "ProtectHome",
        "PrivateTmp",
        "PrivateNetwork",
        "RestrictSUIDSGID",
        "RestrictNamespaces",
    )
    command = ["show", unit, "--no-pager"]
    for name in names:
        command.append(f"--property={name}")
    result = systemctl(policy, *command)
    properties: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            properties[key] = value
    return properties


def current_cgroup() -> str:
    lines = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    unified = [line.split("::", 1)[1] for line in lines if line.startswith("0::")]
    if len(unified) != 1:
        raise SupervisionError("current process is not in one cgroup-v2 hierarchy")
    return unified[0]


def verify_supervisor_unit(policy: dict[str, Any], unit: str) -> dict[str, Any]:
    expected_cgroup_root = Path(policy["cgroup_root"])
    if (
        os.name != "posix"
        or not (expected_cgroup_root / "cgroup.controllers").is_file()
    ):
        raise SupervisionError("cgroup v2 is unavailable")
    properties = unit_properties(policy, f"{unit}.service")
    required = {
        "MainPID": str(os.getpid()),
        "KillMode": "control-group",
        "SendSIGKILL": "yes",
        "NoNewPrivileges": "yes",
        "ProtectControlGroups": "yes",
        "PrivateTmp": "yes",
        "RestrictSUIDSGID": "yes",
    }
    for key, expected in required.items():
        if properties.get(key) != expected:
            raise SupervisionError(
                f"supervisor systemd property {key} is not {expected}"
            )
    cgroup = properties.get("ControlGroup", "")
    if not cgroup or current_cgroup() != cgroup:
        raise SupervisionError("supervisor cgroup identity differs from systemd")
    cgroup_path = expected_cgroup_root / cgroup.lstrip("/")
    if not cgroup_path.is_dir() or os.getpid() not in {
        int(value) for value in (cgroup_path / "cgroup.procs").read_text().split()
    }:
        raise SupervisionError("supervisor is absent from its reported cgroup")
    return {"unit": f"{unit}.service", "controlGroup": cgroup, **required}


def sample_gpu(policy: dict[str, Any]) -> TelemetrySample:
    result = subprocess.run(
        [
            policy["nvidia_smi_executable"],
            "--query-gpu=uuid,name,temperature.gpu,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        timeout=policy["telemetry_timeout_seconds"],
    )
    try:
        lines = [
            line.strip()
            for line in result.stdout.decode("utf-8").splitlines()
            if line.strip()
        ]
    except UnicodeDecodeError as exc:
        raise SupervisionError("GPU telemetry is not UTF-8") from exc
    if len(lines) != 1:
        raise SupervisionError("GPU telemetry requires exactly one visible GPU")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 5 or GPU_UUID_PATTERN.fullmatch(fields[0]) is None:
        raise SupervisionError("GPU telemetry identity is malformed")
    if (
        not fields[1]
        or len(fields[1]) > MAX_GPU_NAME_CHARACTERS
        or not fields[1].isprintable()
    ):
        raise SupervisionError("GPU telemetry name is malformed")
    try:
        temperature, free_mib, total_mib = map(int, fields[2:])
    except ValueError as exc:
        raise SupervisionError("GPU telemetry values are non-numeric") from exc
    if not 0 <= temperature <= MAX_GPU_TEMPERATURE_C:
        raise SupervisionError("GPU telemetry temperature is out of bounds")
    if not 0 <= free_mib <= total_mib <= MAX_GPU_MEMORY_MIB:
        raise SupervisionError("GPU telemetry memory is out of bounds")
    return TelemetrySample(
        time.monotonic_ns(),
        utc_now(),
        fields[0],
        fields[1],
        temperature,
        free_mib,
        total_mib,
    )


def worker_environment(attempt: Attempt) -> dict[str, str]:
    cache = attempt.runtime_cache
    return {
        **WORKER_ENVIRONMENT,
        "HOME": str(cache / "home"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "TORCH_HOME": str(cache / "torch"),
        "UNSLOTH_COMPILE_LOCATION": str(cache / "unsloth"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        "CUDA_CACHE_PATH": str(cache / "cuda"),
    }


def worker_environment_digest(environment: dict[str, str]) -> str:
    return sha256_json(environment)


def create_log(path: Path) -> None:
    descriptor = create_exclusive_file(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def launch_worker_unit(
    policy: dict[str, Any],
    *,
    outer_unit: str,
    worker_unit: str,
    worker_argv: Sequence[str],
    worker_environment_values: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, str]:
    env_command = ["/usr/bin/env", "-i"]
    env_command.extend(
        f"{key}={value}" for key, value in sorted(worker_environment_values.items())
    )
    command = [
        policy["systemd_run_executable"],
        "--user",
        f"--unit={worker_unit}",
        "--service-type=exec",
        f"--working-directory={ROOT}",
        "--property=KillMode=control-group",
        "--property=SendSIGKILL=yes",
        "--property=RemainAfterExit=yes",
        f"--property=TimeoutStopSec={policy['termination_grace_seconds']}s",
        f"--property=BindsTo={outer_unit}.service",
        f"--property=After={outer_unit}.service",
        "--property=NoNewPrivileges=yes",
        "--property=ProtectControlGroups=yes",
        "--property=ProtectSystem=strict",
        "--property=ProtectHome=read-only",
        "--property=PrivateTmp=yes",
        "--property=PrivateNetwork=yes",
        "--property=RestrictSUIDSGID=yes",
        "--property=RestrictNamespaces=yes",
        "--property=ProtectKernelTunables=yes",
        "--property=ProtectKernelModules=yes",
        "--property=ProtectKernelLogs=yes",
        "--property=ProtectClock=yes",
        "--property=ProtectHostname=yes",
        "--property=RestrictAddressFamilies=AF_UNIX",
        "--property=CapabilityBoundingSet=",
        "--property=AmbientCapabilities=",
        f"--property=ReadOnlyPaths={ROOT} /home/rosie/.cache/huggingface",
        f"--property=ReadWritePaths={stdout_path.parent.parent / 'payload'} {stdout_path.parent.parent / 'runtime-cache'}",
        f"--property=InaccessiblePaths={stdout_path.parent} {stdout_path.parent.parent / 'reports'} /run/user/1000/bus",
        "--property=UMask=0077",
        "--property=LimitCORE=0",
        "--property=KeyringMode=private",
        "--property=LockPersonality=yes",
        "--property=RestrictRealtime=yes",
        "--property=SystemCallArchitectures=native",
        "--property=TasksMax=512",
        "--property=OOMPolicy=stop",
        f"--property=StandardOutput=append:{stdout_path}",
        f"--property=StandardError=append:{stderr_path}",
        "--",
        *env_command,
        *worker_argv,
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    properties = unit_properties(policy, f"{worker_unit}.service")
    required = {
        "KillMode": "control-group",
        "SendSIGKILL": "yes",
        "RemainAfterExit": "yes",
        "NoNewPrivileges": "yes",
        "ProtectControlGroups": "yes",
        "ProtectSystem": "strict",
        "ProtectHome": "read-only",
        "PrivateTmp": "yes",
        "PrivateNetwork": "yes",
        "RestrictSUIDSGID": "yes",
        "RestrictNamespaces": "yes",
    }
    for key, expected in required.items():
        if properties.get(key) != expected:
            raise SupervisionError(f"worker systemd property {key} is not {expected}")
    if f"{outer_unit}.service" not in properties.get("BindsTo", "").split():
        raise SupervisionError("worker unit is not bound to the supervisor unit")
    cgroup = properties.get("ControlGroup", "")
    if not cgroup:
        raise SupervisionError("worker unit lacks a cgroup")
    main_pid = int(properties.get("MainPID", "0") or "0")
    if main_pid > 0:
        cgroup_procs = Path(policy["cgroup_root"]) / cgroup.lstrip("/") / "cgroup.procs"
        try:
            observed_pids = {
                int(value) for value in cgroup_procs.read_text(encoding="utf-8").split()
            }
        except (FileNotFoundError, ValueError) as exc:
            refreshed = unit_properties(policy, f"{worker_unit}.service")
            if not retained_terminal_identity(refreshed):
                raise SupervisionError(
                    "worker cgroup membership is unavailable"
                ) from exc
            properties = refreshed
        else:
            if main_pid in observed_pids:
                return properties
            refreshed = unit_properties(policy, f"{worker_unit}.service")
            if not retained_terminal_identity(refreshed):
                raise SupervisionError("worker main PID is absent from its cgroup")
            properties = refreshed
    elif not retained_terminal_identity(properties):
        raise SupervisionError(
            "worker unit lacks a live or retained main-process identity"
        )
    return properties


def unit_terminal(properties: dict[str, str]) -> bool:
    return properties.get("ActiveState") in {"inactive", "failed"} or properties.get(
        "SubState"
    ) in {"dead", "exited", "failed", "failed-to-start"}


def retained_terminal_identity(properties: dict[str, str]) -> bool:
    """Recognize a fast-exited RemainAfterExit service with retained status."""

    try:
        exec_main_pid = int(properties.get("ExecMainPID", "0") or "0")
        int(properties.get("ExecMainStatus", ""))
    except ValueError:
        return False
    return (
        properties.get("RemainAfterExit") == "yes"
        and exec_main_pid > 0
        and unit_terminal(properties)
        and bool(properties.get("Result"))
    )


def retained_worker_properties(
    policy: dict[str, Any], worker_unit: str
) -> dict[str, str]:
    """Re-read and validate the retained process result after worker exit/stop."""

    properties = unit_properties(policy, f"{worker_unit}.service")
    try:
        exec_main_pid = int(properties.get("ExecMainPID", "0") or "0")
        int(properties.get("ExecMainStatus", ""))
    except ValueError as exc:
        raise SupervisionError("retained worker status is malformed") from exc
    if exec_main_pid <= 0 or not properties.get("Result"):
        raise SupervisionError("retained worker status is incomplete")
    return properties


def cgroup_empty(policy: dict[str, Any], cgroup: str) -> bool:
    path = Path(policy["cgroup_root"]) / cgroup.lstrip("/")
    if not path.exists():
        return True
    events = path / "cgroup.events"
    if events.is_file():
        values = dict(
            line.split(maxsplit=1)
            for line in events.read_text(encoding="utf-8").splitlines()
            if " " in line
        )
        return values.get("populated") == "0"
    return not (path / "cgroup.procs").read_text(encoding="utf-8").strip()


def stop_worker_unit(policy: dict[str, Any], unit: str, cgroup: str) -> dict[str, Any]:
    requested_at = utc_now()
    error: str | None = None
    try:
        systemctl(
            policy,
            "stop",
            f"{unit}.service",
            timeout=policy["termination_grace_seconds"]
            + policy["kill_confirmation_seconds"]
            + 5,
        )
    except Exception as exc:  # noqa: BLE001 - cleanup must continue
        error = trainer.sanitized_error(exc)
        try:
            systemctl(policy, "kill", "--signal=KILL", f"{unit}.service", timeout=10)
        except Exception as kill_exc:  # noqa: BLE001
            error = f"{error}; kill={trainer.sanitized_error(kill_exc)}"
    deadline = time.monotonic() + policy["kill_confirmation_seconds"]
    while time.monotonic() < deadline and not cgroup_empty(policy, cgroup):
        time.sleep(0.1)
    empty = cgroup_empty(policy, cgroup)
    return {
        "requestedAt": requested_at,
        "systemdStopRequested": True,
        "killMode": "control-group",
        "sendSigkill": True,
        "cgroupEmptyConfirmed": empty,
        "error": error,
    }


def bounded_log_metadata(path: Path, maximum_bytes: int) -> dict[str, Any]:
    return hash_file(path, maximum_bytes)


def fsync_payload(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise SupervisionError("worker payload contains a symlink")
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise SupervisionError("worker payload contains a nonregular file")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif not path.is_dir():
            raise SupervisionError("worker payload contains an unsupported entry")
    for path in sorted([p for p in root.rglob("*") if p.is_dir()], reverse=True):
        fsync_directory(path)
    fsync_directory(root)


def expected_worker_argv(
    policy: dict[str, Any], source_commit: str, run_kind: str, attempt: Attempt
) -> list[str]:
    return [
        policy["python_executable"],
        "-I",
        "-B",
        str(HERE / "train_candidate.py"),
        "--source-commit",
        source_commit,
        "--run-kind",
        run_kind,
        "--output-dir",
        str(attempt.payload),
        "--supervisor-run-id",
        attempt.run_id,
    ]


def initial_temperature_gate(sample: TelemetrySample, recipe: dict[str, Any]) -> None:
    if sample.temperature_c > recipe["maximum_gpu_temperature_c"]:
        raise SupervisionError(
            f"initial GPU temperature {sample.temperature_c} C exceeds "
            f"{recipe['maximum_gpu_temperature_c']} C"
        )
    if sample.free_mib < int(float(recipe["minimum_free_gpu_gib"]) * 1024):
        raise SupervisionError(
            "initial free GPU memory is below the fixed policy floor"
        )


def sample_trigger(
    sample: TelemetrySample,
    *,
    expected_gpu_uuid: str,
    maximum_temperature_c: int,
) -> tuple[str, str] | None:
    if sample.gpu_uuid != expected_gpu_uuid:
        return "TELEMETRY_UNAVAILABLE", "GPU UUID changed during the run"
    if sample.temperature_c > maximum_temperature_c:
        return (
            "THERMAL_POLICY_VIOLATION",
            f"GPU temperature {sample.temperature_c} C exceeded fixed "
            f"{maximum_temperature_c} C",
        )
    return None


def verify_worker_report(
    *,
    report: dict[str, Any],
    candidate: dict[str, Any],
    source_commit: str,
    run_kind: str,
    run_id: str,
    worker_sha: str,
    policy_sha: str,
    gpu_uuid: str,
    adapter_dir: Path,
) -> dict[str, Any]:
    expected_bundle, _ = trainer.curriculum(source_commit)
    first_sha, first_files = trainer.hash_adapter(adapter_dir)
    second_sha, second_files = trainer.hash_adapter(adapter_dir)
    if (first_sha, first_files) != (second_sha, second_files):
        raise SupervisionError("adapter bytes changed during independent verification")
    validated = supervisor_validation.validate_successful_report(
        report,
        candidate=candidate,
        expected_source_revision=source_commit,
        expected_source_bundle=expected_bundle,
        expected_supervisor_run_id=run_id,
        expected_gpu_uuid=gpu_uuid,
        expected_worker_source_sha256=worker_sha,
        expected_adapter_aggregate_sha256=first_sha,
        expected_adapter_files=first_files,
        expected_run_kind=run_kind,
    )
    return {
        "childReportSha256": validated.report_sha256,
        "observationState": validated.observation_state,
        "localEvaluationInputBindingSatisfied": (
            validated.local_evaluation_input_binding_satisfied
        ),
        "adapter": {
            "aggregateSha256": first_sha,
            "matchesTrainingReport": True,
            "safeTensorsParsed": True,
            "allowlistedFilesOnly": True,
            "symlinksAbsent": True,
            "files": first_files,
        },
    }


def terminal_report_base(
    *,
    attempt: Attempt,
    source: dict[str, Any],
    run_kind: str,
    policy_sha: str,
    supervisor_sha: str,
    worker_sha: str,
    validator_sha: str,
    candidate_sha: str,
    interpreter: dict[str, Any],
    worker_environment_sha: str,
    outer_containment: dict[str, Any],
    admission_sha: str,
) -> dict[str, Any]:
    return {
        "schema": "szl.frontier-training-supervisor/v1",
        "candidateId": "SZL-ReceiptAgent-Qwen3.5-0.8B-v3",
        "runId": attempt.run_id,
        "runKind": run_kind.upper(),
        "observedAt": utc_now(),
        "source": source,
        "identities": {
            "supervisionPolicySha256": policy_sha,
            "supervisorSourceSha256": supervisor_sha,
            "workerSourceSha256": worker_sha,
            "validatorSourceSha256": validator_sha,
            "candidateSourceSha256": candidate_sha,
            "pythonExecutable": interpreter,
            "workerEnvironmentSha256": worker_environment_sha,
            "admissionRecordSha256": admission_sha,
        },
        "containment": outer_containment,
        "securityBoundary": "COOPERATIVE_SAME_ACCOUNT",
        "integrityDigestIsAuthentication": False,
        "authenticatedSupervisorEnvelopePresent": False,
        "qualificationEligible": False,
        "receiptEligible": False,
        "publicationEligible": False,
        "runtimeWitnessPresent": False,
        "autonomyEligible": False,
        "evaluationPerformed": False,
        "comparisonCriteriaSatisfied": False,
    }


def render_report(report: dict[str, Any]) -> bytes:
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    report["reportSha256"] = sha256_json(unsigned)
    return (canonical_json(report) + "\n").encode("utf-8")


def strict_args(argv: Sequence[str]) -> argparse.Namespace:
    required = ("--source-commit", "--run-kind", "--unit-name")
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-kind", choices=("smoke", "full"), required=True)
    parser.add_argument("--unit-name", required=True)
    if list(argv) == ["--help"]:
        return parser.parse_args(list(argv))
    if len(argv) != 6 or any(argv.count(option) != 1 for option in required):
        raise SystemExit(
            "exactly one --source-commit, --run-kind, and --unit-name are required"
        )
    args = parser.parse_args(list(argv))
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("--source-commit must be exactly 40 lowercase hex characters")
    match = SUPERVISOR_UNIT_PATTERN.fullmatch(args.unit_name)
    if not match:
        parser.error("--unit-name is not a ReceiptAgent v3 supervisor unit")
    args.run_id = match.group(1)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = strict_args(sys.argv[1:] if argv is None else argv)
    attempt: Attempt | None = None
    worker_launched = False
    worker_unit = f"szl-ra3-worker-{args.run_id}"
    terminal_cause = "PRECONDITION_DENIED"
    report: dict[str, Any] | None = None
    exit_code = TERMINAL_EXIT_CODES[terminal_cause]
    try:
        source = trainer.fresh_exact_source(args.source_commit)
        candidate_bytes = trainer.committed_bytes(
            args.source_commit, f"{RELATIVE}/candidate.json"
        )
        candidate = json.loads(candidate_bytes)
        policy = validate_policy(candidate)
        policy_sha = sha256_json(policy)
        supervisor_sha = bind_committed_component(
            args.source_commit, "supervise_training.py"
        )
        worker_sha = bind_committed_component(args.source_commit, "train_candidate.py")
        validator_sha = bind_committed_component(
            args.source_commit, "supervisor_validation.py"
        )
        candidate_sha = sha256_bytes(candidate_bytes)
        python_path = Path(policy["python_executable"])
        if Path(sys.executable).resolve() != python_path.resolve(strict=True):
            raise SupervisionError(
                "supervisor interpreter differs from committed policy"
            )
        interpreter = {
            "path": str(python_path),
            "resolvedPath": str(python_path.resolve(strict=True)),
            **hash_file(python_path.resolve(strict=True), 64 * 1024 * 1024),
        }
        outer_containment = verify_supervisor_unit(policy, args.unit_name)
        runs_root = validate_runs_root(Path(policy["runs_root"]))
        attempt = admit_attempt(
            runs_root, args.run_id, policy["evidence_reserve_bytes"]
        )
        exact_worker_environment = worker_environment(attempt)
        worker_environment_sha = worker_environment_digest(exact_worker_environment)
        stdout_path = attempt.logs / "worker.stdout"
        stderr_path = attempt.logs / "worker.stderr"
        create_log(stdout_path)
        create_log(stderr_path)
        started_ns = time.monotonic_ns()
        initial_sample: TelemetrySample | None = None
        initial_error: str | None = None
        try:
            initial_sample = sample_gpu(policy)
        except Exception as exc:  # noqa: BLE001 - terminal evidence records bounded failure
            initial_error = trainer.sanitized_error(exc)
        worker_argv = expected_worker_argv(
            policy, args.source_commit, args.run_kind, attempt
        )
        admission = {
            "schema": "szl.frontier-training-supervisor-admission/v1",
            "state": "PREPARED",
            "runId": attempt.run_id,
            "runKind": args.run_kind.upper(),
            "preparedAt": utc_now(),
            "source": source,
            "supervisorPid": os.getpid(),
            "supervisorUnit": args.unit_name + ".service",
            "workerUnit": worker_unit + ".service",
            "supervisionPolicySha256": policy_sha,
            "supervisorSourceSha256": supervisor_sha,
            "workerSourceSha256": worker_sha,
            "validatorSourceSha256": validator_sha,
            "candidateSourceSha256": candidate_sha,
            "pythonExecutable": interpreter,
            "workerArgvSha256": sha256_json(worker_argv),
            "workerEnvironmentSha256": worker_environment_sha,
            "initialGpuTelemetry": initial_sample.public(started_ns)
            if initial_sample
            else None,
            "initialGpuTelemetryError": initial_error,
            "outputIdentity": {
                "device": attempt.root.stat().st_dev,
                "inode": attempt.root.stat().st_ino,
                "wasExclusivelyCreated": True,
                "writeOnceByProtocol": True,
            },
            "nonce": secrets.token_hex(32),
        }
        admission_bytes = (
            json.dumps(admission, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        admission_artifact = publish_once(
            attempt.reports, "admission.json", admission_bytes
        )
        report = terminal_report_base(
            attempt=attempt,
            source=source,
            run_kind=args.run_kind,
            policy_sha=policy_sha,
            supervisor_sha=supervisor_sha,
            worker_sha=worker_sha,
            validator_sha=validator_sha,
            candidate_sha=candidate_sha,
            interpreter=interpreter,
            worker_environment_sha=worker_environment_sha,
            outer_containment=outer_containment,
            admission_sha=admission_artifact["sha256"],
        )
        samples: list[TelemetrySample] = []
        if initial_sample is not None:
            samples.append(initial_sample)
        if initial_error is not None:
            terminal_cause = "TELEMETRY_UNAVAILABLE"
            raise SupervisionError(initial_error)
        assert initial_sample is not None
        recipe = candidate["training_recipe"]
        if initial_sample.temperature_c > recipe["maximum_gpu_temperature_c"]:
            terminal_cause = "THERMAL_POLICY_VIOLATION"
        initial_temperature_gate(initial_sample, recipe)

        launch_ns = time.monotonic_ns()
        wall_timeout = policy[
            "smoke_wall_timeout_seconds"
            if args.run_kind == "smoke"
            else "full_wall_timeout_seconds"
        ]
        deadline_ns = launch_ns + int(wall_timeout * 1e9)
        terminal_cause = "CONTAINMENT_UNAVAILABLE"
        worker_launched = True
        worker_properties = launch_worker_unit(
            policy,
            outer_unit=args.unit_name,
            worker_unit=worker_unit,
            worker_argv=worker_argv,
            worker_environment_values=exact_worker_environment,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        terminal_cause = "WORKER_EXIT_FAILURE"
        worker_cgroup = worker_properties["ControlGroup"]
        next_sample_ns = launch_ns + int(
            policy["thermal_sample_interval_seconds"] * 1e9
        )
        last_valid_ns = initial_sample.observed_monotonic_ns
        trigger_error: str | None = None
        final_worker_properties = worker_properties
        while True:
            now_ns = time.monotonic_ns()
            if now_ns >= deadline_ns:
                terminal_cause = "WALL_TIMEOUT"
                trigger_error = "fixed wall-time deadline expired"
                break
            if now_ns - last_valid_ns > int(
                policy["maximum_telemetry_gap_seconds"] * 1e9
            ):
                terminal_cause = "TELEMETRY_UNAVAILABLE"
                trigger_error = "GPU telemetry exceeded the maximum allowed gap"
                break
            if now_ns >= next_sample_ns:
                try:
                    sample = sample_gpu(policy)
                except Exception as exc:  # noqa: BLE001
                    terminal_cause = "TELEMETRY_UNAVAILABLE"
                    trigger_error = trainer.sanitized_error(exc)
                    break
                samples.append(sample)
                last_valid_ns = sample.observed_monotonic_ns
                violation = sample_trigger(
                    sample,
                    expected_gpu_uuid=initial_sample.gpu_uuid,
                    maximum_temperature_c=recipe["maximum_gpu_temperature_c"],
                )
                if violation is not None:
                    terminal_cause, trigger_error = violation
                    break
                while next_sample_ns <= sample.observed_monotonic_ns:
                    next_sample_ns += int(
                        policy["thermal_sample_interval_seconds"] * 1e9
                    )
            for log_path in (stdout_path, stderr_path):
                if log_path.stat().st_size > policy["maximum_log_bytes_per_stream"]:
                    terminal_cause = "LOG_QUOTA_EXCEEDED"
                    trigger_error = f"{log_path.name} exceeded the fixed log quota"
                    break
            if trigger_error is not None:
                break
            try:
                final_worker_properties = unit_properties(
                    policy, f"{worker_unit}.service"
                )
            except Exception as exc:  # noqa: BLE001
                terminal_cause = "CONTAINMENT_UNAVAILABLE"
                trigger_error = trainer.sanitized_error(exc)
                break
            if unit_terminal(final_worker_properties):
                break
            sleep_seconds = min(
                0.2,
                max(0.01, (next_sample_ns - time.monotonic_ns()) / 1e9),
                max(0.01, (deadline_ns - time.monotonic_ns()) / 1e9),
            )
            time.sleep(sleep_seconds)

        termination: dict[str, Any] | None = None
        if trigger_error is not None:
            termination = stop_worker_unit(policy, worker_unit, worker_cgroup)
            if not termination["cgroupEmptyConfirmed"]:
                terminal_cause = "TERMINATION_UNCONFIRMED"
        else:
            confirmation_deadline = (
                time.monotonic() + policy["kill_confirmation_seconds"]
            )
            while time.monotonic() < confirmation_deadline and not cgroup_empty(
                policy, worker_cgroup
            ):
                time.sleep(0.1)
            if not cgroup_empty(policy, worker_cgroup):
                terminal_cause = "TERMINATION_UNCONFIRMED"
                trigger_error = (
                    "worker cgroup remained populated after main-process exit"
                )
                termination = stop_worker_unit(policy, worker_unit, worker_cgroup)

        try:
            final_worker_properties = retained_worker_properties(policy, worker_unit)
        except Exception as exc:  # noqa: BLE001 - retain the first terminal trigger
            refresh_error = trainer.sanitized_error(exc)
            final_worker_properties = {
                "ExecMainStatus": "255",
                "Result": "retained-status-unavailable",
            }
            trigger_error = (
                f"{trigger_error}; retained-status={refresh_error}"
                if trigger_error
                else f"retained-status={refresh_error}"
            )
            if terminal_cause not in OBSERVED_TRIGGER_CAUSES:
                terminal_cause = "CONTAINMENT_UNAVAILABLE"

        try:
            terminal_sample = sample_gpu(policy)
            samples.append(terminal_sample)
            terminal_violation = sample_trigger(
                terminal_sample,
                expected_gpu_uuid=initial_sample.gpu_uuid,
                maximum_temperature_c=recipe["maximum_gpu_temperature_c"],
            )
            if terminal_violation is not None:
                terminal_cause, trigger_error = merge_terminal_observation(
                    terminal_cause, trigger_error, terminal_violation
                )
        except Exception as exc:  # noqa: BLE001
            terminal_cause, trigger_error = merge_terminal_observation(
                terminal_cause,
                trigger_error,
                ("TELEMETRY_UNAVAILABLE", trainer.sanitized_error(exc)),
            )

        worker_status = int(
            final_worker_properties.get("ExecMainStatus", "255") or "255"
        )
        worker_result = final_worker_properties.get("Result", "unknown")
        report["launch"] = {
            "workerUnit": worker_unit + ".service",
            "workerControlGroup": worker_cgroup,
            "workerArgvSha256": sha256_json(worker_argv),
            "startedAt": admission["preparedAt"],
            "endedAt": utc_now(),
            "durationSeconds": round((time.monotonic_ns() - launch_ns) / 1e9, 6),
            "wallTimeoutSeconds": wall_timeout,
            "workerExitStatus": worker_status,
            "workerResult": worker_result,
            "triggerError": trigger_error,
            "termination": termination,
            "cgroupEmptyConfirmed": cgroup_empty(policy, worker_cgroup),
        }
        report["telemetry"] = {
            "source": "INDEPENDENT_SUPERVISOR_FIXED_NVIDIA_SMI",
            "gpuUuid": initial_sample.gpu_uuid,
            "maximumTemperaturePolicyC": recipe["maximum_gpu_temperature_c"],
            "sampleIntervalSeconds": policy["thermal_sample_interval_seconds"],
            "maximumTelemetryGapSeconds": policy["maximum_telemetry_gap_seconds"],
            "samples": [sample.public(launch_ns) for sample in samples],
            "maximumObservedTemperatureC": max(
                sample.temperature_c for sample in samples
            ),
        }
        report["logs"] = {
            "stdout": bounded_log_metadata(
                stdout_path, policy["maximum_log_bytes_per_stream"]
            ),
            "stderr": bounded_log_metadata(
                stderr_path, policy["maximum_log_bytes_per_stream"]
            ),
            "storedLogContentsReviewedForSecrets": False,
        }

        if terminal_cause in {
            "THERMAL_POLICY_VIOLATION",
            "WALL_TIMEOUT",
            "TELEMETRY_UNAVAILABLE",
            "TERMINATION_UNCONFIRMED",
            "LOG_QUOTA_EXCEEDED",
            "CONTAINMENT_UNAVAILABLE",
        }:
            report["state"] = "SUPERVISOR_TERMINATED_RUN_NO_COMPLETION_CLAIM"
            report["primaryCause"] = terminal_cause
            report["workerPayloadDisposition"] = "UNTRUSTED_PARTIAL_NOT_REUSABLE"
            exit_code = TERMINAL_EXIT_CODES[terminal_cause]
        elif worker_status != 0 or worker_result != "success":
            terminal_cause = "WORKER_EXIT_FAILURE"
            report["state"] = "SUPERVISOR_OBSERVED_RUN_FAILURE"
            report["primaryCause"] = terminal_cause
            report["workerPayloadDisposition"] = "UNTRUSTED_PARTIAL_NOT_REUSABLE"
            exit_code = TERMINAL_EXIT_CODES[terminal_cause]
        else:
            terminal_cause = "WORKER_REPORT_INVALID"
            fsync_payload(attempt.payload)
            worker_report_path = attempt.payload / "training-report.json"
            worker_report = strict_json_file(worker_report_path)
            validation = verify_worker_report(
                report=worker_report,
                candidate=candidate,
                source_commit=args.source_commit,
                run_kind=args.run_kind,
                run_id=attempt.run_id,
                worker_sha=worker_sha,
                policy_sha=policy_sha,
                gpu_uuid=initial_sample.gpu_uuid,
                adapter_dir=attempt.payload / "adapter",
            )
            worker_report_file = hash_file(worker_report_path, 2 * 1024 * 1024)
            report["trainingReport"] = {
                "relativePath": "payload/training-report.json",
                "fileSha256": worker_report_file["sha256"],
                "bytes": worker_report_file["bytes"],
                "canonicalReportSha256": validation["childReportSha256"],
                "state": worker_report["state"],
                "provenance": "CHILD_REPORTED_UNATTESTED",
            }
            report["bindings"] = {
                "strictChildReportSchemaAndSemanticsValidated": True,
                "sourceBundleIndependentlyRecomputed": True,
                "adapterIndependentlyHashedTwice": True,
                "adapterMatchesTrainingReport": True,
                "runSourceGpuRecipeRuntimeAndPolicyBound": True,
                "childPromotionBoundariesRemainFalse": True,
            }
            report["adapter"] = validation["adapter"]
            if args.run_kind == "smoke":
                report["state"] = validation["observationState"]
                report["localEvaluationInputBindingSatisfied"] = validation[
                    "localEvaluationInputBindingSatisfied"
                ]
            else:
                report["state"] = validation["observationState"]
                report["localEvaluationInputBindingSatisfied"] = validation[
                    "localEvaluationInputBindingSatisfied"
                ]
            report["primaryCause"] = "SUCCESS"
            report["workerPayloadDisposition"] = "BOUND_UNATTESTED"
            exit_code = 0
        report["claimBoundary"] = (
            "The supervisor observed process containment, telemetry, exit state, and bytes. "
            "It did not prove optimizer semantics, useful learning, model quality, evaluation, "
            "receipt eligibility, publication, deployment, runtime health, or autonomy."
        )
    except Exception as exc:  # noqa: BLE001 - one fail-closed terminal path
        if worker_launched and attempt is not None:
            try:
                policy = validate_policy(candidate)
                properties = unit_properties(policy, f"{worker_unit}.service")
                cleanup = stop_worker_unit(
                    policy, worker_unit, properties.get("ControlGroup", "")
                )
                if not cleanup["cgroupEmptyConfirmed"]:
                    terminal_cause = "TERMINATION_UNCONFIRMED"
            except Exception:  # noqa: BLE001 - original bounded failure remains primary
                terminal_cause = "TERMINATION_UNCONFIRMED"
        exit_code = TERMINAL_EXIT_CODES.get(terminal_cause, 70)
        if report is not None:
            report["state"] = (
                "SUPERVISOR_CHILD_EXITED_WITHOUT_VALID_REPORT"
                if terminal_cause == "WORKER_REPORT_INVALID"
                else "SUPERVISOR_TERMINATED_RUN_NO_COMPLETION_CLAIM"
            )
            report["primaryCause"] = terminal_cause
            report["fatal"] = trainer.sanitized_error(exc)
            report["workerPayloadDisposition"] = "UNTRUSTED_PARTIAL_NOT_REUSABLE"
            report["localEvaluationInputBindingSatisfied"] = False

    if attempt is not None and report is None:
        emergency_report = {
            "schema": "szl.frontier-training-supervisor/v1",
            "candidateId": "SZL-ReceiptAgent-Qwen3.5-0.8B-v3",
            "runId": attempt.run_id,
            "runKind": args.run_kind.upper(),
            "state": "EVIDENCE_DURABILITY_FAILED",
            "primaryCause": terminal_cause,
            "observedAt": utc_now(),
            "integrityDigestIsAuthentication": False,
            "authenticatedSupervisorEnvelopePresent": False,
            "qualificationEligible": False,
            "receiptEligible": False,
            "publicationEligible": False,
            "autonomyEligible": False,
        }
        try:
            release_evidence_reserve(attempt)
            rendered = render_report(emergency_report)
            artifact = publish_once(attempt.reports, "supervisor-report.json", rendered)
            print(rendered.decode("utf-8"), end="")
            print(f"supervisorReportPath={artifact['path']}")
        except Exception as exc:  # noqa: BLE001 - last-resort bounded console evidence
            print(
                json.dumps(
                    {
                        **emergency_report,
                        "fatal": trainer.sanitized_error(exc),
                    },
                    sort_keys=True,
                )
            )
        return TERMINAL_EXIT_CODES["EVIDENCE_DURABILITY_FAILED"]
    if attempt is None or report is None:
        print(
            json.dumps(
                {
                    "schema": "szl.frontier-training-supervisor/v1",
                    "state": "PRECONDITION_DENIED_NO_ATTEMPT_ADMITTED",
                    "primaryCause": terminal_cause,
                    "receiptEligible": False,
                    "publicationEligible": False,
                    "autonomyEligible": False,
                },
                sort_keys=True,
            )
        )
        return exit_code
    try:
        release_evidence_reserve(attempt)
        rendered = render_report(report)
        artifact = publish_once(attempt.reports, "supervisor-report.json", rendered)
        print(rendered.decode("utf-8"), end="")
        print(f"supervisorReportPath={artifact['path']}")
    except Exception as exc:  # noqa: BLE001 - success is withheld if evidence is not durable
        print(
            json.dumps(
                {
                    "schema": "szl.frontier-training-supervisor/v1",
                    "state": "EVIDENCE_DURABILITY_FAILED",
                    "fatal": trainer.sanitized_error(exc),
                    "receiptEligible": False,
                    "publicationEligible": False,
                    "autonomyEligible": False,
                },
                sort_keys=True,
            )
        )
        return TERMINAL_EXIT_CODES["EVIDENCE_DURABILITY_FAILED"]
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
