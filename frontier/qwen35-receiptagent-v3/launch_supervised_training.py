#!/usr/bin/env python3
"""Launch ReceiptAgent v3 supervision in a dedicated systemd user service."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import posixpath
import re
import secrets
import signal
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RELATIVE = "frontier/qwen35-receiptagent-v3"
RELATIVE_CANDIDATE = f"{RELATIVE}/candidate.json"
SUPERVISOR = HERE / "supervise_training.py"
GIT = "/usr/bin/git"
SYSTEMD_RUN = "/usr/bin/systemd-run"
SYSTEMCTL = "/usr/bin/systemctl"
CLEANUP_TIMEOUT_SECONDS = 100
SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}")
SERVICE_NAME = re.compile(r"szl-ra3-supervisor-[0-9a-f]{32}")
SUPERVISED_EXECUTABLE_COMPONENTS = (
    "launch_supervised_training.py",
    "supervisor_bootstrap.py",
    "supervise_training.py",
    "containment_probe.py",
    "train_candidate.py",
    "supervisor_validation.py",
    "host_memory_guard.py",
    "windows_host_memory_sampler.ps1",
)
MAX_COMPONENT_BYTES = 2 * 1024 * 1024
MAX_SUPERVISOR_REPORT_BYTES = 8 * 1024 * 1024
SUPERVISOR_SUCCESS_STATE_BY_KIND = {
    "smoke": "SUPERVISOR_OBSERVED_SMOKE_OUTPUT_BOUND_NOT_QUALIFIED",
    "full": "SUPERVISOR_OBSERVED_FULL_OUTPUT_BOUND_UNATTESTED",
}
SYSTEMD_PROPERTIES = (
    "KillMode=control-group",
    "SendSIGKILL=yes",
    "TimeoutStopSec=90s",
    "NoNewPrivileges=yes",
    "ProtectControlGroups=yes",
    "PrivateTmp=yes",
    "RestrictSUIDSGID=yes",
    "UMask=0077",
    "LimitCORE=0",
    "KeyringMode=private",
    "LockPersonality=yes",
    "RestrictRealtime=yes",
    "SystemCallArchitectures=native",
    "TasksMax=768",
    "OOMPolicy=stop",
)
SUPERVISOR_ENVIRONMENT = {
    "HOME": "/home/rosie",
    "USER": "rosie",
    "LOGNAME": "rosie",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib",
    "XDG_RUNTIME_DIR": "/run/user/1000",
    "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


class LauncherError(RuntimeError):
    """The committed launcher/runtime contract could not be satisfied."""


class LauncherInterrupted(Exception):
    """The caller requested bounded shutdown of the exact launched unit."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"launcher received signal {signum}")
        self.signum = signum


class SingleUseAction(argparse.Action):
    """Reject an option when it appears more than once."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} must be specified exactly once")
        setattr(namespace, self.dest, values)


def exact_source_commit(value: str) -> str:
    if SOURCE_COMMIT.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "source commit must be exactly 40 lowercase hexadecimal characters"
        )
    return value


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--source-commit",
        action=SingleUseAction,
        type=exact_source_commit,
        required=True,
        default=None,
    )
    parser.add_argument(
        "--run-kind",
        action=SingleUseAction,
        choices=("smoke", "full"),
        required=True,
        default=None,
    )
    return parser


def load_committed_candidate(source_commit: str) -> dict[str, Any]:
    try:
        observed = subprocess.run(
            [GIT, "show", f"{source_commit}:{RELATIVE_CANDIDATE}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            shell=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LauncherError(
            f"could not read candidate.json from the requested commit: {type(exc).__name__}"
        ) from exc
    if observed.returncode != 0:
        raise LauncherError("candidate.json is unavailable at the requested commit")
    try:
        candidate = json.loads(observed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LauncherError("committed candidate.json is not valid UTF-8 JSON") from exc
    if not isinstance(candidate, dict):
        raise LauncherError("committed candidate.json must contain one JSON object")
    return candidate


def _read_regular_file_once(path: Path, maximum_bytes: int) -> bytes:
    """Return stable bytes from one non-symlink, single-link regular file."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LauncherError(
            f"could not open supervised component {path.name}: {type(exc).__name__}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise LauncherError(
                f"supervised component must be a single-link regular file: {path.name}"
            )
        if before.st_size > maximum_bytes:
            raise LauncherError(f"supervised component is too large: {path.name}")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(
                descriptor,
                min(1024 * 1024, maximum_bytes + 1 - total),
            )
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > maximum_bytes:
                raise LauncherError(
                    f"supervised component grew past its byte ceiling: {path.name}"
                )
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or total != before.st_size:
            raise LauncherError(
                f"supervised component changed during its single-open read: {path.name}"
            )
        return b"".join(chunks)
    except OSError as exc:
        raise LauncherError(
            f"could not read supervised component {path.name}: {type(exc).__name__}"
        ) from exc
    finally:
        os.close(descriptor)


def _committed_component_bytes(
    source_commit: str,
    filename: str,
    *,
    repo_root: Path,
) -> bytes:
    try:
        observed = subprocess.run(
            [GIT, "show", f"{source_commit}:{RELATIVE}/{filename}"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            shell=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LauncherError(
            f"could not read committed supervised component {filename}: "
            f"{type(exc).__name__}"
        ) from exc
    if observed.returncode != 0:
        raise LauncherError(
            f"supervised component is unavailable at the requested commit: {filename}"
        )
    if len(observed.stdout) > MAX_COMPONENT_BYTES:
        raise LauncherError(f"committed supervised component is too large: {filename}")
    return observed.stdout


def verify_local_components(
    source_commit: str,
    *,
    component_dir: Path = HERE,
    repo_root: Path = ROOT,
) -> None:
    """Bind every pre-service executable component to the requested commit."""

    if SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise LauncherError("source commit is not exact lowercase hexadecimal")
    for filename in SUPERVISED_EXECUTABLE_COMPONENTS:
        local = _read_regular_file_once(component_dir / filename, MAX_COMPONENT_BYTES)
        committed = _committed_component_bytes(
            source_commit,
            filename,
            repo_root=repo_root,
        )
        if not secrets.compare_digest(local, committed):
            raise LauncherError(
                f"local supervised component differs from requested commit: {filename}"
            )


def committed_python_path(
    candidate: dict[str, Any],
    *,
    observed_executable: str,
) -> str:
    policy = candidate.get("supervision_policy")
    if not isinstance(policy, dict):
        raise LauncherError("committed candidate lacks supervision_policy")
    configured = policy.get("python_executable")
    if not isinstance(configured, str) or not configured.startswith("/"):
        raise LauncherError("committed supervision Python must be an absolute path")
    if posixpath.normpath(configured) != configured:
        raise LauncherError("committed supervision Python path is not normalized")
    if policy.get("systemd_run_executable") != SYSTEMD_RUN:
        raise LauncherError("committed systemd-run path is not /usr/bin/systemd-run")
    if policy.get("systemctl_executable") != SYSTEMCTL:
        raise LauncherError("committed systemctl path is not /usr/bin/systemctl")
    if observed_executable != configured:
        raise LauncherError(
            "launcher must run under the exact Python path committed in candidate.json"
        )
    return configured


def require_local_executables(python_executable: str) -> None:
    for label, path in (
        ("committed Python", Path(python_executable)),
        ("git", Path(GIT)),
        ("systemd-run", Path(SYSTEMD_RUN)),
        ("systemctl", Path(SYSTEMCTL)),
    ):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise LauncherError(f"{label} executable is unavailable at {path}")
def generate_service_name(run_kind: str) -> str:
    if run_kind not in {"smoke", "full"}:
        raise LauncherError("run kind is unsupported")
    token = secrets.token_hex(16)
    name = f"szl-ra3-supervisor-{token}"
    if SERVICE_NAME.fullmatch(name) is None:
        raise LauncherError("generated systemd service name is not safe")
    return name


def attempt_identity(candidate: dict[str, Any], service_name: str) -> tuple[str, str]:
    match = SERVICE_NAME.fullmatch(service_name)
    if match is None:
        raise LauncherError("systemd service name is not safe")
    policy = candidate.get("supervision_policy")
    if not isinstance(policy, dict):
        raise LauncherError("committed candidate lacks supervision_policy")
    runs_root = policy.get("runs_root")
    if (
        not isinstance(runs_root, str)
        or not runs_root.startswith("/")
        or posixpath.normpath(runs_root) != runs_root
        or runs_root == "/"
    ):
        raise LauncherError("committed runs root must be a normalized absolute path")
    run_id = match.group(0).removeprefix("szl-ra3-supervisor-")
    return run_id, posixpath.join(runs_root, run_id)


def emit_launch_identity(service_name: str, run_id: str, attempt_path: str) -> None:
    sys.stdout.write(f"supervisorUnit={service_name}.service\n")
    sys.stdout.write(f"supervisorRunId={run_id}\n")
    sys.stdout.write(f"supervisorAttemptPath={attempt_path}\n")
    sys.stdout.flush()


def _raise_launcher_interrupted(signum: int, _frame: Any) -> None:
    # A repeated terminal signal must not interrupt the bounded systemctl stop
    # which runs as soon as this exception unwinds the blocking wait.
    for managed_signal in (signal.SIGINT, signal.SIGTERM):
        signal.signal(managed_signal, signal.SIG_IGN)
    raise LauncherInterrupted(signum)


def stop_outer_unit(service_name: str) -> str | None:
    if SERVICE_NAME.fullmatch(service_name) is None:
        return "refused unsafe cleanup unit name"
    try:
        stopped = subprocess.run(
            [SYSTEMCTL, "--user", "stop", f"{service_name}.service"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            timeout=CLEANUP_TIMEOUT_SECONDS,
            shell=False,
            env=dict(SUPERVISOR_ENVIRONMENT),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"{type(exc).__name__}"
    # With --collect, a normally completed transient unit can already be gone;
    # systemctl reports that benign state with the LSB "not installed" code.
    if stopped.returncode not in {0, 5}:
        return f"systemctl exit {stopped.returncode}"
    return None


def publish_launcher_observation(reports: Path, observation: dict[str, Any]) -> Path:
    specification = importlib.util.spec_from_file_location(
        "szl_ra3_launcher_bootstrap", HERE / "supervisor_bootstrap.py"
    )
    if specification is None or specification.loader is None:
        raise LauncherError("launcher evidence publisher is unavailable")
    publisher = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = publisher
    specification.loader.exec_module(publisher)
    artifact = publisher.publish_write_once(
        reports,
        "launcher-stop-observation.json",
        (json.dumps(observation, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        ),
    )
    return artifact.path


def classify_supervisor_report(
    path: Path, *, run_id: str, run_kind: str, source_commit: str
) -> str:
    """Require a stable, bounded, exact report before a zero launcher exit."""

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate supervisor report key")
            value[key] = item
        return value

    def reject_constant(_value: str) -> None:
        raise ValueError("nonfinite supervisor report value")

    try:
        raw = _read_regular_file_once(path, MAX_SUPERVISOR_REPORT_BYTES)
        report = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_pairs,
            parse_constant=reject_constant,
        )
    except (LauncherError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return "SUPERVISOR_REPORT_INVALID"
    if not isinstance(report, dict):
        return "SUPERVISOR_REPORT_INVALID"
    unsigned = dict(report)
    declared_digest = unsigned.pop("reportSha256", None)
    if (
        not isinstance(declared_digest, str)
        or hashlib.sha256(
            json.dumps(unsigned, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()
        != declared_digest
        or raw
        != (json.dumps(report, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode(
            "utf-8"
        )
    ):
        return "SUPERVISOR_REPORT_INVALID"
    if (
        report.get("schema") != "szl.frontier-training-supervisor/v1"
        or report.get("runId") != run_id
        or report.get("runKind") != run_kind.upper()
        or not isinstance(report.get("source"), dict)
        or report["source"].get("revision") != source_commit
    ):
        return "SUPERVISOR_REPORT_IDENTITY_MISMATCH"
    if any(
        report.get(field) is not False
        for field in ("receiptEligible", "publicationEligible", "autonomyEligible")
    ):
        return "SUPERVISOR_REPORT_CLAIM_BOUNDARY_INVALID"
    if (
        report.get("primaryCause") == "SUCCESS"
        and report.get("state") == SUPERVISOR_SUCCESS_STATE_BY_KIND[run_kind]
    ):
        return "SUPERVISOR_REPORT_SUCCESS"
    return "SUPERVISOR_REPORTED_NON_SUCCESS"


def record_missing_supervisor_report(
    *,
    service_name: str,
    run_id: str,
    attempt_path: str,
    source_commit: str,
    run_kind: str,
    systemd_exit_code: int | None,
    launcher_signal: int | None,
    outer_stop_error: str | None,
) -> str:
    """Write a separate launcher observation only for this admitted attempt."""

    if (
        SERVICE_NAME.fullmatch(service_name) is None
        or service_name.removeprefix("szl-ra3-supervisor-") != run_id
        or SOURCE_COMMIT.fullmatch(source_commit) is None
        or run_kind not in {"smoke", "full"}
    ):
        raise LauncherError("launcher observation identity is invalid")
    attempt = Path(attempt_path)
    if (
        not attempt.is_absolute()
        or posixpath.normpath(attempt_path) != attempt_path
        or attempt.name != run_id
    ):
        raise LauncherError("launcher observation attempt path is invalid")
    current = Path(attempt.anchor)
    for part in attempt.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            return "NO_ATTEMPT"
        if stat.S_ISLNK(metadata.st_mode):
            raise LauncherError("launcher observation path contains a symlink")
    reports = attempt / "reports"
    try:
        reports_metadata = os.lstat(reports)
    except FileNotFoundError:
        return "NO_REPORTS_DIRECTORY"
    for directory in (attempt.parent, attempt, reports):
        metadata = os.lstat(directory)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o022
        ):
            raise LauncherError("launcher observation directory is not owner controlled")
    if not stat.S_ISDIR(reports_metadata.st_mode):
        raise LauncherError("launcher reports path is not a directory")
    try:
        terminal_metadata = os.lstat(reports / "supervisor-report.json")
    except FileNotFoundError:
        pass
    else:
        if (
            not stat.S_ISREG(terminal_metadata.st_mode)
            or terminal_metadata.st_uid != os.getuid()
            or terminal_metadata.st_nlink != 1
        ):
            raise LauncherError("supervisor report path is not a trusted regular file")
        return classify_supervisor_report(
            reports / "supervisor-report.json",
            run_id=run_id,
            run_kind=run_kind,
            source_commit=source_commit,
        )
    observation = {
        "schema": "szl.frontier-training-launcher-observation/v1",
        "state": "SUPERVISOR_REPORT_ABSENT_AFTER_LAUNCHER_WAIT",
        "observedAt": datetime.now(timezone.utc).isoformat(),
        "runId": run_id,
        "runKind": run_kind.upper(),
        "supervisorUnit": f"{service_name}.service",
        "sourceCommit": source_commit,
        "systemdRunExitCode": systemd_exit_code,
        "launcherSignal": launcher_signal,
        "outerStopError": outer_stop_error,
        "supervisorReportPresent": False,
        "trainingCompletionObserved": False,
        "workerTerminationConfirmedByLauncher": False,
        "integrityDigestIsAuthentication": False,
        "receiptEligible": False,
        "publicationEligible": False,
        "autonomyEligible": False,
    }
    path = publish_launcher_observation(reports, observation)
    sys.stdout.write(f"launcherObservationPath={path}\n")
    sys.stdout.flush()
    return "LAUNCHER_OBSERVATION_WRITTEN"


def invoke_with_cleanup(
    command: Sequence[str],
    *,
    service_name: str,
    run_id: str,
    attempt_path: str,
    source_commit: str,
    run_kind: str,
) -> int:
    previous_handlers: dict[int, Any] = {}
    launched: subprocess.CompletedProcess[Any] | None = None
    interrupted: LauncherInterrupted | None = None
    cleanup_problem: str | None = None
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _raise_launcher_interrupted)
        emit_launch_identity(service_name, run_id, attempt_path)
        launched = subprocess.run(
            list(command),
            cwd=ROOT,
            check=False,
            shell=False,
        )
    except LauncherInterrupted as exc:
        interrupted = exc
    finally:
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        cleanup_problem = stop_outer_unit(service_name)
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)
        if cleanup_problem is not None:
            sys.stderr.write(f"supervisorCleanupWarning={cleanup_problem}\n")
            sys.stderr.flush()
    if interrupted is not None:
        result_code = 128 + interrupted.signum
    elif launched is None:
        raise LauncherError("systemd-run returned no process result")
    else:
        result_code = launched.returncode if 0 <= launched.returncode <= 255 else 1
    try:
        observation_state = record_missing_supervisor_report(
            service_name=service_name,
            run_id=run_id,
            attempt_path=attempt_path,
            source_commit=source_commit,
            run_kind=run_kind,
            systemd_exit_code=launched.returncode if launched is not None else None,
            launcher_signal=interrupted.signum if interrupted is not None else None,
            outer_stop_error=cleanup_problem,
        )
    except Exception as exc:  # noqa: BLE001 - preserve the systemd result
        sys.stderr.write(f"launcherObservationWarning={type(exc).__name__}\n")
        sys.stderr.flush()
        return result_code or 76
    if observation_state != "SUPERVISOR_REPORT_SUCCESS":
        sys.stderr.write(f"launcherSupervisorReportState={observation_state}\n")
        sys.stderr.flush()
    if observation_state != "SUPERVISOR_REPORT_SUCCESS" and result_code == 0:
        return 76
    return result_code


def systemd_command(
    *,
    service_name: str,
    python_executable: str,
    source_commit: str,
    run_kind: str,
    interop_socket: str | None = None,
) -> list[str]:
    if SERVICE_NAME.fullmatch(service_name) is None:
        raise LauncherError("systemd service name is not safe")
    if SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise LauncherError("source commit is not exact lowercase hexadecimal")
    if run_kind not in {"smoke", "full"}:
        raise LauncherError("run kind is unsupported")
    supervisor_environment = dict(SUPERVISOR_ENVIRONMENT)
    if interop_socket is not None:
        if re.fullmatch(r"/run/WSL/[0-9]+_interop", interop_socket) is None:
            raise LauncherError("native Windows interop socket identity is malformed")
        supervisor_environment["WSL_INTEROP"] = interop_socket
    command = [
        SYSTEMD_RUN,
        "--user",
        "--wait",
        "--pipe",
        "--collect",
        f"--unit={service_name}",
        "--service-type=exec",
        f"--working-directory={ROOT}",
    ]
    command.extend(
        f"--property={property_value}" for property_value in SYSTEMD_PROPERTIES
    )
    command.extend(
        [
            "--",
            "/usr/bin/env",
            "-i",
            *(
                f"{key}={value}"
                for key, value in sorted(supervisor_environment.items())
            ),
            python_executable,
            "-I",
            "-B",
            str(SUPERVISOR),
            "--source-commit",
            source_commit,
            "--run-kind",
            run_kind,
            "--unit-name",
            service_name,
        ]
    )
    return command


def require_native_interop() -> str:
    interop_socket = os.environ.get("WSL_INTEROP", "")
    if re.fullmatch(r"/run/WSL/[0-9]+_interop", interop_socket) is None:
        raise LauncherError("native Windows interop is required for the host-memory guard")
    interop_path = Path(interop_socket)
    interop_stat = interop_path.lstat()
    if (not stat.S_ISSOCK(interop_stat.st_mode)
            or interop_path.resolve(strict=True) != interop_path
            or interop_stat.st_uid not in {0, os.getuid()}):
        raise LauncherError("native Windows interop socket is not admissible")
    return interop_socket


def main(argv: Sequence[str] | None = None) -> int:
    parser = argument_parser()
    args = parser.parse_args(argv)
    try:
        candidate = load_committed_candidate(args.source_commit)
        python_executable = committed_python_path(
            candidate,
            observed_executable=sys.executable,
        )
        require_local_executables(python_executable)
        verify_local_components(args.source_commit)
        interop_socket = require_native_interop()
        service_name = generate_service_name(args.run_kind)
        run_id, attempt_path = attempt_identity(candidate, service_name)
        command = systemd_command(
            service_name=service_name,
            python_executable=python_executable,
            source_commit=args.source_commit,
            run_kind=args.run_kind,
            interop_socket=interop_socket,
        )
        return invoke_with_cleanup(
            command,
            service_name=service_name,
            run_id=run_id,
            attempt_path=attempt_path,
            source_commit=args.source_commit,
            run_kind=args.run_kind,
        )
    except LauncherError as exc:
        parser.error(str(exc))
    except OSError as exc:
        parser.error(f"could not invoke /usr/bin/systemd-run: {type(exc).__name__}")
    raise AssertionError("argparse error paths do not return")


if __name__ == "__main__":
    sys.exit(main())
