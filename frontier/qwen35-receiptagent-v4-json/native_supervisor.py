#!/usr/bin/env python3
"""Fail-closed Linux v4 smoke supervision, never model-release authority.

Only the fixed source-bound worker may be dispatched. Kernel containment and
fresh resource observations are separate from a child report. This module
imports no model framework and does not stop or share another GPU workload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import selectors
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass

HERE = Path(__file__).resolve().parent
RELEASE_MARKER = b"SZL-RA4-CONTAINMENT-VERIFIED\n"
MAX_CONTROL_BYTES = 16384
MAX_TREE_ENTRIES = 4096
SYSTEMD_RUN = Path("/usr/bin/systemd-run")
SYSTEMCTL = Path("/usr/bin/systemctl")
NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
CGROUP_ROOT = Path("/sys/fs/cgroup")
SYSTEMD_REQUIRED_PROPERTIES = {
    "Delegate": "no", "ProtectControlGroups": "yes", "NoNewPrivileges": "yes",
    "ProtectSystem": "strict", "ProtectHome": "tmpfs", "PrivateNetwork": "yes",
    "PrivateTmp": "yes", "RestrictSUIDSGID": "yes", "ProtectKernelTunables": "yes",
    "ProtectKernelModules": "yes", "ProtectKernelLogs": "yes", "KillMode": "control-group",
    "ReadOnlyPaths": "/tmp /var/tmp /dev/shm",
}
CONTAINMENT_BOUNDARY = "TRUSTED_SOURCE_BOUND_PROCESS_NOT_HOSTILE_SAME_UID_SANDBOX"


def containment_policy() -> dict:
    return {"schema": "szl.receiptagent.v4-native-containment-policy/v1",
            "systemd_run": str(SYSTEMD_RUN), "systemctl": str(SYSTEMCTL),
            "nvidia_smi": str(NVIDIA_SMI), "cgroup_root": str(CGROUP_ROOT),
            "properties": dict(SYSTEMD_REQUIRED_PROPERTIES), "tasks_max": 128,
            "release_marker_sha256": hashlib.sha256(RELEASE_MARKER).hexdigest(),
            "security_boundary": CONTAINMENT_BOUNDARY, "gpu_sharing": False,
            "read_only_binds": "SOURCE_MODEL_TOKENIZER_NEMO_PYTHON_PREFIX_AND_BOUND_PUBLIC_FILES",
            "writable_binds": "EXACT_OUTPUT_DIRECTORY_ONLY", "shell": False,
            "environment": "FIXED_ENV_I_NO_INHERITANCE"}


def containment_policy_sha256() -> str:
    return hashlib.sha256(canonical(containment_policy())).hexdigest()


class SupervisorError(RuntimeError):
    """Public, value-free failure code; no exception text or secret echo."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class SupervisorInterrupted(BaseException):
    """A direct stop reaches owned cleanup, not an ordinary inner retry."""


def _stop_requested(signum, _frame) -> None:
    for name in (signal.SIGINT, signal.SIGTERM):
        signal.signal(name, signal.SIG_IGN)
    raise SupervisorInterrupted()


def require(condition: bool, code: str) -> None:
    if condition is not True:
        raise SupervisorError(code)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _write_all(fd: int, data: bytes) -> None:
    position = 0
    while position < len(data):
        written = os.write(fd, data[position:])
        require(written > 0, "SHORT_DURABLE_WRITE")
        position += written


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                 | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def private_directory(path: Path) -> Path:
    """Require an existing explicit, unaliased owner-private task directory."""
    require(path.is_absolute(), "ABSOLUTE_DIRECTORY_REQUIRED")
    for component in (path, *path.parents):
        meta = component.lstat()
        require(not stat.S_ISLNK(meta.st_mode)
                and not getattr(meta, "st_file_attributes", 0) & 0x400,
                "DIRECTORY_ALIAS_FORBIDDEN")
    meta = path.stat()
    require(stat.S_ISDIR(meta.st_mode), "DIRECTORY_REQUIRED")
    if os.name == "posix":
        require(meta.st_uid == os.getuid() and not meta.st_mode & 0o077,
                "PRIVATE_DIRECTORY_REQUIRED")
    return path


def exclusive_durable(path: Path, data: bytes) -> dict:
    """No overwrite or retry cleanup: an indeterminate write stays consumed."""
    private_directory(path.parent)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_directory(path.parent)
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "durability": "MEASURED", "authentication": "UNAVAILABLE"}


def consume_nonce(namespace: Path, nonce: str, admission_sha256: str) -> dict:
    require(type(nonce) is str and re.fullmatch(r"[0-9a-f]{64}", nonce) is not None,
            "NONCE_INVALID")
    require(type(admission_sha256) is str
            and re.fullmatch(r"[0-9a-f]{64}", admission_sha256) is not None,
            "ADMISSION_HASH_INVALID")
    record = canonical({"schema": "szl.receiptagent.v4-nonce-consumption/v1",
                        "attempt_nonce": nonce, "admission_sha256": admission_sha256,
                        "evidence_class": "MEASURED", "dispatch_observed": False})
    try:
        return exclusive_durable(namespace / (nonce + ".json"), record)
    except FileExistsError as exc:
        raise SupervisorError("NONCE_ALREADY_CONSUMED") from exc


class ReceiptChain:
    """Append-only, fsynced byte-integrity chain, explicitly not a signature."""

    def __init__(self, directory: Path):
        self.directory = private_directory(directory)
        self.fd = os.open(directory / "supervisor-chain.jsonl",
                          os.O_WRONLY | os.O_CREAT | os.O_EXCL
                          | getattr(os, "O_NOFOLLOW", 0), 0o600)
        self.sequence = 0
        self.tip = "0" * 64
        fsync_directory(directory)

    def append(self, event: dict) -> dict:
        require(type(event) is dict, "CHAIN_EVENT_INVALID")
        body = {"schema": "szl.receiptagent.v4-supervisor-chain/v1",
                "sequence": self.sequence, "previous_sha256": self.tip,
                "event": event, "authenticated": False}
        digest = hashlib.sha256(canonical(body)).hexdigest()
        record = {**body, "record_sha256": digest}
        raw = canonical(record)
        require(len(raw) <= 32768, "CHAIN_EVENT_TOO_LARGE")
        _write_all(self.fd, raw + b"\n")
        os.fsync(self.fd)
        self.sequence += 1
        self.tip = digest
        return record

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def minimal_environment(home: Path, *, gpu_uuid: str | None = None) -> dict[str, str]:
    """Never inherit tokens, SSH, agent, Python, loader or tracking variables."""
    result = {"HOME": str(home), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
              "TMPDIR": str(home),
              "PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1",
              "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
              "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
              "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
              "WANDB_DISABLED": "true", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    if gpu_uuid is not None:
        require(re.fullmatch(r"GPU-[A-Za-z0-9-]{16,96}", gpu_uuid) is not None,
                "GPU_UUID_INVALID")
        result["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    return result


def bounded_command(argv: tuple[str, ...], *, timeout: float = 2.0,
                    maximum: int = MAX_CONTROL_BYTES, manager: bool = False,
                    pending_unit: str | None = None) -> bytes:
    """Fixed control-plane commands only; terminate only this owned process."""
    require(type(argv) is tuple and len(argv) > 0 and Path(argv[0]).is_absolute(),
            "CONTROL_ARGV_INVALID")
    require(type(maximum) is int and 1 <= maximum <= 65536
            and type(timeout) in (int, float) and 0 < timeout <= 15,
            "CONTROL_BOUND_INVALID")
    process = subprocess.Popen(argv, shell=False, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=(_manager_environment() if manager else
                                    minimal_environment(Path("/nonexistent"))),
                               start_new_session=True, close_fds=True)
    selector = selectors.DefaultSelector()
    chunks = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout
    try:
        for name in chunks:
            stream = getattr(process, name)
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while selector.get_map():
            require(time.monotonic() < deadline, "CONTROL_TIMEOUT")
            for key, _ in selector.select(min(0.05, max(0, deadline - time.monotonic()))):
                data = os.read(key.fileobj.fileno(), min(4096, maximum + 1))
                if not data:
                    selector.unregister(key.fileobj)
                else:
                    chunks[key.data].extend(data)
                    require(sum(map(len, chunks.values())) <= maximum,
                            "CONTROL_OUTPUT_CAP")
        remaining = max(0.001, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise SupervisorError("CONTROL_TIMEOUT") from exc
        if pending_unit is not None and process.returncode in (0, 1):
            require(re.fullmatch(r"szl-ra4-[0-9a-f]{64}\.service", pending_unit) is not None,
                    "UNIT_ID_INVALID")
            stdout = bytes(chunks["stdout"])
            stderr = bytes(chunks["stderr"])
            known_absent = (b"LoadState=not-found" in stdout.splitlines()
                            and all(line == b"LoadState=not-found" or b"=" in line
                                    for line in stdout.splitlines())
                            and stderr in (b"", ("Unit " + pending_unit + " could not be found.\n").encode()))
            if known_absent:
                return b"LoadState=not-found\n"
        require(process.returncode == 0, "CONTROL_EXIT_FAILURE")
        require(not chunks["stderr"], "CONTROL_STDERR")
        return bytes(chunks["stdout"])
    finally:
        selector.close()
        if process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        for name in chunks:
            getattr(process, name).close()


@dataclass(frozen=True)
class GpuSample:
    observed_monotonic: float
    uuid: str
    total_bytes: int
    used_bytes: int
    temperature_millicelsius: int


def parse_gpu_sample(raw: bytes, expected_uuid: str, observed: float) -> GpuSample:
    require(type(raw) is bytes and len(raw) <= 512, "GPU_TELEMETRY_INVALID")
    try:
        fields = raw.decode("ascii").strip().split(",")
    except UnicodeError as exc:
        raise SupervisorError("GPU_TELEMETRY_INVALID") from exc
    fields = [field.strip() for field in fields]
    require(len(fields) == 4 and fields[0] == expected_uuid, "GPU_IDENTITY_MISMATCH")
    require(all(re.fullmatch(r"[0-9]{1,8}", item) is not None for item in fields[1:]),
            "GPU_TELEMETRY_INVALID")
    total, used, temperature = map(int, fields[1:])
    require(0 < total <= 10_000_000 and 0 <= used <= total and 0 <= temperature <= 200,
            "GPU_TELEMETRY_INVALID")
    return GpuSample(observed, fields[0], total * 1048576, used * 1048576,
                     temperature * 1000)


def sample_gpu(executable: Path, gpu_uuid: str) -> GpuSample:
    raw = bounded_command((str(executable), "--id=" + gpu_uuid,
                           "--query-gpu=uuid,memory.total,memory.used,temperature.gpu",
                           "--format=csv,noheader,nounits"), maximum=512)
    return parse_gpu_sample(raw, gpu_uuid, time.monotonic())


def assert_gpu_exclusive(executable: Path) -> None:
    raw = bounded_command((str(executable), "--query-compute-apps=pid,gpu_uuid",
                           "--format=csv,noheader,nounits"), maximum=4096)
    # No sharing exception and no stop/kill operation: Bonsai makes this deny.
    require(raw.strip() == b"", "GPU_ALREADY_HAS_COMPUTE_PROCESS")


def check_sample(sample: GpuSample, *, previous: float, now: float,
                 max_gap: int, max_gpu_bytes: int, max_temperature: int) -> None:
    require(previous <= sample.observed_monotonic <= now
            and now - previous <= max_gap, "TELEMETRY_GAP")
    require(sample.used_bytes <= max_gpu_bytes, "GPU_MEMORY_CAP")
    require(sample.temperature_millicelsius < max_temperature, "GPU_TEMPERATURE_CAP")


def tree_bytes(directory: Path, *, cap: int) -> int:
    """Bound entries/bytes without following links, devices or hardlinks."""
    total = 0
    entries = 0
    pending = [directory]
    while pending:
        current = pending.pop()
        with os.scandir(current) as children:
            for child in children:
                entries += 1
                require(entries <= MAX_TREE_ENTRIES, "TREE_ENTRY_CAP")
                # Windows DirEntry.stat can omit nlink; the same bounded
                # lstat operation is used on both platforms, never relax nlink.
                meta = Path(child.path).lstat()
                require(not stat.S_ISLNK(meta.st_mode), "TREE_ALIAS_FORBIDDEN")
                if stat.S_ISDIR(meta.st_mode):
                    pending.append(Path(child.path))
                else:
                    require(stat.S_ISREG(meta.st_mode) and meta.st_nlink == 1,
                            "TREE_SPECIAL_OR_ALIAS_FORBIDDEN")
                    total += meta.st_size
                    require(total <= cap, "TREE_BYTE_CAP")
    return total


def _bounded_file(path: Path, maximum: int = MAX_CONTROL_BYTES) -> bytes:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        meta = os.fstat(fd)
        require(stat.S_ISREG(meta.st_mode), "CONTROL_FILE_INVALID")
        raw = os.read(fd, maximum + 1)
        require(len(raw) <= maximum, "CONTROL_FILE_CAP")
        return raw
    finally:
        os.close(fd)


def _manager_environment() -> dict[str, str]:
    env = minimal_environment(Path("/nonexistent"))
    uid = os.getuid()
    env.update({"XDG_RUNTIME_DIR": f"/run/user/{uid}",
                "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{uid}/bus"})
    return env


def unit_properties(unit: str) -> dict[str, str] | None:
    require(re.fullmatch(r"szl-ra4-[0-9a-f]{64}\.service", unit) is not None,
            "UNIT_ID_INVALID")
    wanted = (*SYSTEMD_REQUIRED_PROPERTIES, "ControlGroup", "MainPID", "MemoryMax",
              "MemorySwapMax", "TasksMax", "ActiveState", "SubState", "Result",
              "ExecMainStatus", "ExecMainCode", "Description", "LoadState")
    raw = bounded_command((str(SYSTEMCTL), "--user", "show", unit,
                           "--property=" + ",".join(wanted)), timeout=2, manager=True,
                          pending_unit=unit)
    if raw == b"LoadState=not-found\n":
        return None
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as exc:
        raise SupervisorError("UNIT_PROPERTY_INVALID") from exc
    properties = {}
    for line in lines:
        require("=" in line, "UNIT_PROPERTY_INVALID")
        key, value = line.split("=", 1)
        require(key in wanted and key not in properties, "UNIT_PROPERTY_INVALID")
        properties[key] = value
    require(set(properties) == set(wanted) and properties["LoadState"] == "loaded",
            "UNIT_PROPERTY_INCOMPLETE")
    return properties


def _control_command(argv: tuple[str, ...], *, timeout: float = 2) -> bytes:
    # bounded_command has a separate explicit manager-mode flag rather than
    # inheriting the caller's ambient bus or SSH environment.
    return bounded_command(argv, timeout=timeout, manager=True)


def verify_containment(unit: str, properties: dict[str, str], bounds: dict) -> tuple[Path, int]:
    require(sys.platform == "linux", "LINUX_REQUIRED")
    for key, value in SYSTEMD_REQUIRED_PROPERTIES.items():
        require(properties.get(key) == value, "CONTAINMENT_PROPERTY_MISMATCH")
    require(properties["MemoryMax"] == str(bounds["host_memory_bytes"])
            and properties["MemorySwapMax"] == "0"
            and properties["TasksMax"] == "128", "CONTAINMENT_BOUND_MISMATCH")
    control_group = properties["ControlGroup"]
    require(control_group.startswith("/") and ".." not in Path(control_group).parts
            and Path(control_group).name == unit, "CGROUP_IDENTITY_INVALID")
    cgroup = CGROUP_ROOT / control_group.lstrip("/")
    for path in (cgroup, *cgroup.parents):
        if path == CGROUP_ROOT.parent:
            break
        require(not stat.S_ISLNK(path.lstat().st_mode), "CGROUP_ALIAS_FORBIDDEN")
    mounts = _bounded_file(Path("/proc/self/mountinfo"), 65536).decode("ascii")
    require(any(" - cgroup2 " in line and line.split(" ")[4] == str(CGROUP_ROOT)
                for line in mounts.splitlines()), "CGROUP_V2_UNAVAILABLE")
    require(_bounded_file(cgroup / "memory.max").strip() == str(bounds["host_memory_bytes"]).encode()
            and _bounded_file(cgroup / "memory.swap.max").strip() == b"0"
            and _bounded_file(cgroup / "pids.max").strip() == b"128",
            "CGROUP_KERNEL_BOUND_MISMATCH")
    pid_text = properties["MainPID"]
    require(re.fullmatch(r"[1-9][0-9]{0,9}", pid_text) is not None,
            "WORKER_MAIN_PID_UNAVAILABLE")
    pid = int(pid_text)
    require(_bounded_file(Path(f"/proc/{pid}/cgroup"), 4096).strip()
            == ("0::" + control_group).encode(), "WORKER_CGROUP_MISMATCH")
    return cgroup, pid


def cgroup_empty(cgroup: Path) -> bool:
    # cgroup.events covers all descendants, unlike cgroup.procs alone.
    try:
        raw = _bounded_file(cgroup / "cgroup.events", 1024)
    except FileNotFoundError:
        # Only called with the previously kernel-verified owned cgroup. cgroup2
        # cannot remove a populated group; successful systemd cleanup can remove
        # this exact directory before the waiting client exits. This is a
        # trusted-manager deletion observation, not hostile-same-UID isolation.
        require(not cgroup.exists(), "CGROUP_EVENTS_UNAVAILABLE")
        return True
    fields = {}
    for line in raw.decode("ascii").splitlines():
        pair = line.split()
        require(len(pair) == 2 and pair[0] not in fields, "CGROUP_EVENTS_INVALID")
        fields[pair[0]] = pair[1]
    require(fields.get("populated") in {"0", "1"}, "CGROUP_EVENTS_INVALID")
    return fields["populated"] == "0"


def drain_owned_unit(unit: str, cgroup: Path, *, kill_seconds: int,
                     drain_seconds: int) -> bool:
    """Only an exact verified unit is signaled; never any preexisting workload."""
    if cgroup_empty(cgroup):
        return True
    for sig, wait in (("TERM", kill_seconds), ("KILL", drain_seconds)):
        _control_command((str(SYSTEMCTL), "--user", "kill", "--kill-whom=all",
                          "--signal=" + sig, unit))
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if cgroup_empty(cgroup):
                return True
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    return cgroup_empty(cgroup)


def _bind_paths(run) -> tuple[Path, ...]:
    # The shared source closure binds only reviewed modules + train-only input,
    # not a whole checkout, .git configuration or held-out curriculum files.
    read_only = list(run.readonly_paths)
    unique = sorted({str(path): path for path in read_only}.values(), key=str)
    require(all(path.is_absolute() and not any(char in str(path) for char in ":\n\r ")
                for path in unique), "SYSTEMD_BIND_PATH_INVALID")
    return tuple(unique)


def worker_argv(run, *, description: str) -> tuple[str, ...]:
    """No caller argv, code string, callback, shell or ambient worker env."""
    output = run.path("output_dir")
    unit = "szl-ra4-" + run.nonce + ".service"
    bounds = run.payload["bounds"]
    require(re.fullmatch(r"szl-ra4-owned-[0-9a-f]{64}", description) is not None,
            "OWNERSHIP_DESCRIPTION_INVALID")
    options = [str(SYSTEMD_RUN), "--user", "--wait", "--pipe", "--quiet",
               "--service-type=exec", "--unit=" + unit, "--description=" + description]
    for key, value in SYSTEMD_REQUIRED_PROPERTIES.items():
        options.append("--property=" + key + "=" + value)
    options.extend(("--property=MemoryMax=" + str(bounds["host_memory_bytes"]),
                    "--property=MemorySwapMax=0", "--property=TasksMax=128",
                    "--property=RuntimeMaxSec=" + str(bounds["wall_seconds"]),
                    "--property=TimeoutStopSec=" + str(bounds["kill_seconds"]),
                    "--property=InaccessiblePaths=/run/user /run/credentials /root",
                    "--property=ReadWritePaths=" + str(output),
                    "--property=WorkingDirectory=" + str(output),
                    "--property=UMask=0077"))
    for path in _bind_paths(run):
        options.append("--property=BindReadOnlyPaths=" + str(path))
    options.append("--property=BindPaths=" + str(output))
    # The manager itself may have a rich environment. env -i clears it before
    # Python, so sanitized Popen env alone is not mistaken for worker isolation.
    options.extend(("--", "/usr/bin/env", "-i"))
    env = minimal_environment(output, gpu_uuid=run.payload["runner"]["device_id"])
    options.extend(key + "=" + value for key, value in sorted(env.items()))
    options.extend((str(run.worker_python), "-I", "-B", str(run.worker_path),
                    "--bound-run", str(run.request_path), "--owner-policy-sha256",
                    run.owner_policy_sha256, "--release-stdin"))
    return tuple(options)


def qualification_matches(run) -> None:
    """Qualification is independently pinned by shared loader, never a callback."""
    record = run.qualification
    require(record.get("containment_policy_sha256") == containment_policy_sha256()
            == run.payload["runner"]["containment_policy_sha256"],
            "CONTAINMENT_POLICY_BINDING_MISMATCH")
    require(record.get("containment_boundary") == CONTAINMENT_BOUNDARY
            and record.get("systemd_properties") == SYSTEMD_REQUIRED_PROPERTIES
            and all(record.get(name) is True for name in (
                "kernel_barrier_observed", "worker_cgroup_escape_denied",
                "worker_environment_cleared", "worker_home_credentials_hidden",
                "worker_network_denied")),
            "CONTAINMENT_QUALIFICATION_INCOMPLETE")
    executables = record.get("control_executable_sha256")
    required = {str(path) for path in (SYSTEMD_RUN, SYSTEMCTL, NVIDIA_SMI, Path("/usr/bin/env"))}
    require(type(executables) is dict and set(executables) == required,
            "CONTROL_EXECUTABLE_QUALIFICATION_UNAVAILABLE")
    for name, digest in executables.items():
        require(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                "CONTROL_EXECUTABLE_QUALIFICATION_INVALID")
        fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            meta = os.fstat(fd)
            require(stat.S_ISREG(meta.st_mode) and meta.st_size <= 32 * 1024**2,
                    "CONTROL_EXECUTABLE_INVALID")
            hasher = hashlib.sha256()
            count = 0
            while True:
                block = os.read(fd, 1024 * 1024)
                if not block:
                    break
                count += len(block)
                require(count <= 32 * 1024**2, "CONTROL_EXECUTABLE_CAP")
                hasher.update(block)
            require(count == meta.st_size and hasher.hexdigest() == digest,
                    "CONTROL_EXECUTABLE_HASH_MISMATCH")
        finally:
            os.close(fd)
    rate = record.get("runner_rate_minor_units_per_hour")
    source = record.get("cost_rate_source")
    require(type(rate) is int and 0 <= rate <= 1_000_000_000
            and type(source) is str and 0 < len(source) <= 512,
            "COST_QUALIFICATION_UNAVAILABLE")
    bounds = run.payload["bounds"]
    charged_seconds = sum(bounds[name] for name in ("wall_seconds", "kill_seconds", "drain_seconds")) + 10
    estimated = (rate * charged_seconds + 3599) // 3600
    require(estimated <= bounds["cost_cap_minor_units"], "COST_CAP_EXCEEDED")


def validate_bounds(bounds: dict) -> None:
    maxima = {"wall_seconds": 1200, "kill_seconds": 30, "drain_seconds": 30,
              "telemetry_max_gap_seconds": 5, "sequence_tokens": 2048,
              "total_tokens": 400000, "host_memory_bytes": 64 * 1024**3,
              "gpu_memory_bytes": 40 * 1024**3, "temperature_limit_millicelsius": 85000,
              "log_cap_bytes": 8 * 1024**2, "output_cap_bytes": 1024**3,
              "artifact_cap_bytes": 512 * 1024**2, "cost_cap_minor_units": 200}
    for name, maximum in maxima.items():
        minimum = 0 if name == "cost_cap_minor_units" else 1
        require(type(bounds.get(name)) is int and minimum <= bounds[name] <= maximum,
                "SMOKE_BOUND_EXCEEDS_HARD_LIMIT")
    require(type(bounds.get("disk_reserve_bytes")) is int
            and type(bounds.get("evidence_reserve_bytes")) is int
            and 10 * 1024**3 <= bounds["disk_reserve_bytes"] <= 1024**4
            and 1024**3 <= bounds["evidence_reserve_bytes"] <= 1024**4,
            "SMOKE_RESERVE_BELOW_MINIMUM")


def _check_disk(run, *, reserve_present: bool) -> None:
    bounds = run.payload["bounds"]
    stats = os.statvfs(run.path("run_dir").parent)
    available = stats.f_bavail * stats.f_frsize
    required = bounds["disk_reserve_bytes"] + bounds["output_cap_bytes"] + bounds["log_cap_bytes"]
    if not reserve_present:
        required += bounds["evidence_reserve_bytes"]
    require(available >= required, "DISK_RESERVE_UNAVAILABLE")


def _allocate_reserve(path: Path, size: int) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        require(hasattr(os, "posix_fallocate"), "DURABLE_RESERVE_ALLOCATION_UNAVAILABLE")
        os.posix_fallocate(fd, 0, size)
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_directory(path.parent)


def _check_foreign_gpu_processes(cgroup: Path, device_uuid: str) -> None:
    raw = bounded_command((str(NVIDIA_SMI), "--query-compute-apps=pid,gpu_uuid",
                           "--format=csv,noheader,nounits"), maximum=4096)
    for line in raw.decode("ascii").splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        require(len(fields) == 2 and fields[1] == device_uuid
                and re.fullmatch(r"[1-9][0-9]{0,9}", fields[0]) is not None,
                "FOREIGN_GPU_COMPUTE_PROCESS")
        expected = "0::/" + str(cgroup.relative_to(CGROUP_ROOT))
        try:
            membership = _bounded_file(Path("/proc") / fields[0] / "cgroup", 4096).decode("ascii").strip()
        except FileNotFoundError as exc:
            raise SupervisorError("GPU_PROCESS_IDENTITY_UNAVAILABLE") from exc
        require(membership == expected or membership.startswith(expected + "/"),
                "FOREIGN_GPU_COMPUTE_PROCESS")


def _worker_observation(run, *, runtime) -> dict:
    """Independently bind bytes; child metrics remain REPORTED, never replay."""
    output = run.path("output_dir")
    # Owner/admission JSON intentionally permits no floats. Only this scoped
    # child-result reader accepts finite metric floats; the closed check below
    # prevents that narrow parser from widening launch authority inputs.
    report = runtime.read_worker_result(output / "worker-result.json", maximum=65536)
    expected = {"schema": "szl.receiptagent.v4-native-worker-result/v1",
                "state": "SMOKE_COMPLETED_NOT_RELEASED", "evidence_class": "REPORTED",
                "admission_nonce": run.nonce, "rows": 360, "optimizer_updates": 1,
                "seed": run.payload["curriculum"]["seed"],
                "train_sha256": run.payload["curriculum"]["train_sha256"],
                "manifest_sha256": run.payload["curriculum"]["manifest_sha256"],
                "base_id": run.payload["base"]["repository"],
                "base_revision": run.payload["base"]["revision"],
                "source_revision": run.payload["source"]["revision"],
                "token_ledger_sha256": run.payload["tokens"]["tensor_ledger_sha256"],
                "row_order_sha256": run.payload["bounds"]["row_order_sha256"],
                "device_uuid": run.payload["runner"]["device_id"],
                "energy_joules": "UNAVAILABLE", "model_evaluation": "NOT_RUN",
                "independent_replay": "UNAVAILABLE", "training_eligible": False,
                "publication_eligible": False, "execution_authority": False}
    require(all(type(report.get(key)) is type(value) and report.get(key) == value
                for key, value in expected.items()), "WORKER_REPORT_BINDING_MISMATCH")
    require(set(report) == set(expected) | {"artifact_sha256", "supervised_tokens",
                "token_weighted_loss", "gradient_norm_before_clip"}, "WORKER_REPORT_FIELDS_INVALID")
    require(type(report["supervised_tokens"]) is int
            and 360 <= report["supervised_tokens"] <= run.payload["bounds"]["total_tokens"],
            "WORKER_REPORTED_TOKEN_COUNT_INVALID")
    for name in ("token_weighted_loss", "gradient_norm_before_clip"):
        value = report[name]
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                "WORKER_REPORTED_METRIC_INVALID")
    hashes = report.get("artifact_sha256")
    names = {"adapter/adapter_model.safetensors", "adapter/adapter_config.json"}
    require(type(hashes) is dict and set(hashes) == names, "WORKER_ARTIFACT_SET_INVALID")
    observed = {}
    for name in sorted(names):
        digest = runtime.sha256_file(output / name,
                                     max_bytes=run.payload["bounds"]["artifact_cap_bytes"])
        require(digest == hashes[name], "WORKER_ARTIFACT_HASH_MISMATCH")
        observed[name] = digest
    # Repeat after all files so an artifact mutation cannot hide behind report
    # validation. A shared UID remains a trusted-source, not hostile, boundary.
    for name, digest in observed.items():
        require(runtime.sha256_file(output / name,
                    max_bytes=run.payload["bounds"]["artifact_cap_bytes"]) == digest,
                "WORKER_ARTIFACT_CHANGED")
    tree_bytes(output / "adapter", cap=run.payload["bounds"]["artifact_cap_bytes"])
    return {"worker_report_sha256": runtime.sha256_file(output / "worker-result.json", max_bytes=65536),
            "artifact_sha256": observed, "metrics_provenance": "CHILD_REPORTED_UNATTESTED",
            "optimizer_semantics_independently_replayed": False}


def supervise(run) -> dict:
    """One admitted fixed worker; current source policy never reaches here.

    There is no test callback or alternate command injection interface. CPU
    fixtures exercise smaller primitives, not fabricated launch authority.
    """
    import native_runtime as runtime
    require(sys.platform == "linux", "LINUX_REQUIRED")
    runtime.revalidate_bound_run(run, role="supervisor")
    validate_bounds(run.payload["bounds"])
    qualification_matches(run)
    run_dir = run.path("run_dir")
    output = run.path("output_dir")
    require(run_dir.name == run.nonce and output == run_dir / "output",
            "PRIVATE_ATTEMPT_LAYOUT_MISMATCH")
    parent = private_directory(run_dir.parent)
    namespace = private_directory(parent / ".v4-nonces")
    require(not run_dir.exists(), "ATTEMPT_ALREADY_EXISTS")
    assert_gpu_exclusive(NVIDIA_SMI)
    bounds = run.payload["bounds"]
    initial = sample_gpu(NVIDIA_SMI, run.payload["runner"]["device_id"])
    check_sample(initial, previous=initial.observed_monotonic, now=time.monotonic(),
                 max_gap=bounds["telemetry_max_gap_seconds"],
                 max_gpu_bytes=bounds["gpu_memory_bytes"],
                 max_temperature=bounds["temperature_limit_millicelsius"])
    _check_disk(run, reserve_present=False)
    admission_sha256 = runtime.sha256_file(run.path("admission"), max_bytes=65536)
    consume_nonce(namespace, run.nonce, admission_sha256)
    # Once consumed, every failure leaves the nonce consumed. No retry cleanup.
    os.mkdir(run_dir, 0o700)
    fsync_directory(parent)
    os.mkdir(output, 0o700)
    fsync_directory(run_dir)
    chain = ReceiptChain(run_dir)
    reserve = run_dir / ".evidence-reserve"
    process = None
    cgroup = None
    unit = "szl-ra4-" + run.nonce + ".service"
    released = False
    owned_unit = False
    drained = False
    report = _terminal("BLOCKED", "PREDISPATCH_FAILURE")
    log_fds = {}
    selector = selectors.DefaultSelector()
    started = time.monotonic()
    last_sample = initial.observed_monotonic
    try:
        _allocate_reserve(reserve, bounds["evidence_reserve_bytes"])
        chain.append({"state": "NONCE_CONSUMED", "admission_sha256": admission_sha256})
        runtime.revalidate_bound_run(run, role="supervisor")
        # An existing unit is not ours even when the nonce namespace was empty.
        require(unit_properties(unit) is None, "UNIT_ALREADY_EXISTS")
        description = "szl-ra4-owned-" + secrets.token_hex(32)
        process = subprocess.Popen(worker_argv(run, description=description), shell=False, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env=_manager_environment(), close_fds=True,
                                   start_new_session=True)
        for name in ("stdout", "stderr"):
            path = run_dir / (name + ".log")
            log_fds[name] = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                                    | getattr(os, "O_NOFOLLOW", 0), 0o600)
            stream = getattr(process, name)
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        launch_deadline = min(started + bounds["wall_seconds"], time.monotonic() + 10)
        while time.monotonic() < launch_deadline:
            require(process.poll() is None, "WORKER_EXITED_BEFORE_CONTAINMENT")
            properties = unit_properties(unit)
            if properties is not None and properties["MainPID"] != "0":
                require(properties["Description"] == description,
                        "UNIT_OWNERSHIP_UNCONFIRMED")
                cgroup, _ = verify_containment(unit, properties, bounds)
                owned_unit = True
                break
            time.sleep(0.05)
        require(cgroup is not None, "CONTAINMENT_STARTUP_UNAVAILABLE")
        runtime.revalidate_bound_run(run, role="supervisor")
        assert_gpu_exclusive(NVIDIA_SMI)
        chain.append({"state": "CONTAINMENT_MEASURED", "unit": unit,
                      "policy_sha256": containment_policy_sha256()})
        process.stdin.write(RELEASE_MARKER)
        process.stdin.flush()
        process.stdin.close()
        released = True
        sizes = {"stdout": 0, "stderr": 0}
        while process.poll() is None or selector.get_map():
            require(time.monotonic() - started <= bounds["wall_seconds"], "WALL_CAP")
            for key, _ in selector.select(0.05):
                data = os.read(key.fileobj.fileno(), 4096)
                if not data:
                    selector.unregister(key.fileobj)
                else:
                    sizes[key.data] += len(data)
                    require(sum(sizes.values()) <= bounds["log_cap_bytes"], "LOG_CAP")
                    _write_all(log_fds[key.data], data)
            now = time.monotonic()
            if now - last_sample >= min(1.0, bounds["telemetry_max_gap_seconds"] / 3):
                sample = sample_gpu(NVIDIA_SMI, run.payload["runner"]["device_id"])
                check_sample(sample, previous=last_sample, now=time.monotonic(),
                             max_gap=bounds["telemetry_max_gap_seconds"],
                             max_gpu_bytes=bounds["gpu_memory_bytes"],
                             max_temperature=bounds["temperature_limit_millicelsius"])
                last_sample = sample.observed_monotonic
                memory = _bounded_file(cgroup / "memory.current", 64).strip()
                require(memory.isdigit() and int(memory) <= bounds["host_memory_bytes"],
                        "HOST_MEMORY_CAP")
                _check_foreign_gpu_processes(cgroup, sample.uuid)
                _check_disk(run, reserve_present=True)
                tree_bytes(output, cap=bounds["output_cap_bytes"])
                chain.append({"state": "RESOURCE_SAMPLE", "gpu_memory_bytes": sample.used_bytes,
                              "gpu_temperature_millicelsius": sample.temperature_millicelsius,
                              "host_memory_bytes": int(memory)})
        require(process.wait(timeout=1) == 0, "WORKER_EXIT_FAILURE")
        drained = drain_owned_unit(unit, cgroup, kill_seconds=bounds["kill_seconds"],
                                   drain_seconds=bounds["drain_seconds"])
        require(drained, "OWNED_CGROUP_DRAIN_UNCONFIRMED")
        final = sample_gpu(NVIDIA_SMI, run.payload["runner"]["device_id"])
        check_sample(final, previous=last_sample, now=time.monotonic(),
                     max_gap=bounds["telemetry_max_gap_seconds"],
                     max_gpu_bytes=bounds["gpu_memory_bytes"],
                     max_temperature=bounds["temperature_limit_millicelsius"])
        properties = unit_properties(unit)
        require(properties is not None and properties["Result"] == "success"
                and properties["ExecMainStatus"] == "0",
                "UNIT_TERMINAL_FAILURE")
        runtime.revalidate_bound_run(run, role="supervisor")
        tree_bytes(output, cap=bounds["output_cap_bytes"])
        observation = _worker_observation(run, runtime=runtime)
        report = {**_terminal("SMOKE_OBSERVED_NOT_RELEASED", "WORKER_BYTES_BOUND"),
                  "training": "REPORTED_SINGLE_SMOKE_UPDATE", "worker": observation}
    except BaseException as exc:
        # SIGINT/SystemExit and ordinary faults all proceed through owned drain.
        report = _terminal("BLOCKED", exc.code if isinstance(exc, SupervisorError)
                           else "SUPERVISOR_STOP_REQUESTED" if isinstance(exc, SupervisorInterrupted)
                           else "SUPERVISOR_FAILURE")
        if process is not None and cgroup is not None and owned_unit:
            try:
                drained = drain_owned_unit(unit, cgroup, kill_seconds=bounds["kill_seconds"],
                                           drain_seconds=bounds["drain_seconds"])
            except Exception:
                drained = False
        # An unsuccessful start does not establish ownership. In particular,
        # never signal a same-name preexisting unit after systemd-run failure.
        if process is not None and not drained:
            report["reason"] = "OWNED_CGROUP_DRAIN_UNCONFIRMED"
    finally:
        selector.close()
        if process is not None:
            if process.poll() is None:
                # Only our own wrapper session; never the preexisting Bonsai PID.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=min(2, bounds["drain_seconds"]))
                except subprocess.TimeoutExpired:
                    report["reason"] = "WRAPPER_WAIT_UNCONFIRMED"
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        for fd in log_fds.values():
            os.fsync(fd)
            os.close(fd)
        report.update({"admission_nonce": run.nonce, "worker_released": released,
                       "owned_cgroup_drained": drained, "containment_boundary": CONTAINMENT_BOUNDARY,
                       "integrity_digest_is_authentication": False, "authenticated": False})
        if released and report["training"] == "NOT_RUN":
            report["training"] = "UNKNOWN"
        try:
            if reserve.exists():
                reserve.unlink()  # Only the exact disposable reserve created above.
                fsync_directory(run_dir)
            chain.append({"state": "TERMINAL", "report": report})
            report["chain_tip_sha256"] = chain.tip
            exclusive_durable(run_dir / "supervisor-result.json", canonical(report))
        finally:
            chain.close()
    return report


def _terminal(state: str, code: str) -> dict:
    return {"schema": "szl.receiptagent.v4-native-supervisor/v1",
            "state": state, "reason": code, "evidence_class": "REPORTED",
            "training": "NOT_RUN", "evaluation": "NOT_RUN",
            "energy_joules": "UNAVAILABLE", "independent_replay": "UNAVAILABLE",
            "training_eligible": False, "publication_eligible": False,
            "execution_authority": False, "operational": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--check-readiness", action="store_true")
    operation.add_argument("--supervise-smoke", action="store_true")
    parser.add_argument("--bound-run", "--config", dest="config", type=Path, required=True)
    parser.add_argument("--owner-policy-sha256", required=True)
    args = parser.parse_args(argv)
    # Source policy is checked by the shared loader before nonce, process,
    # telemetry or framework import. Dispatch wiring follows that shared API.
    previous = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, _stop_requested)
        sys.path.insert(0, str(HERE))
        import native_runtime as runtime
        run = runtime.load_bound_run(args.config, owner_policy_sha256=args.owner_policy_sha256,
                                     role="supervisor")
        if args.check_readiness:
            report = _terminal("BLOCKED", "READINESS_ONLY_NO_DISPATCH")
        else:
            report = supervise(run)
    except Exception as exc:
        code = exc.code if hasattr(exc, "code") else "BOUND_RUN_UNAVAILABLE"
        report = _terminal("BLOCKED", code)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    print(canonical(report).decode("ascii"))
    return 0 if report["state"] == "SMOKE_OBSERVED_NOT_RELEASED" else 70


if __name__ == "__main__":
    raise SystemExit(main())
