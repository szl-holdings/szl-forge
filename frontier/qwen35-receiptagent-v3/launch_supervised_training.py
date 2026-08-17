#!/usr/bin/env python3
"""Launch ReceiptAgent v3 supervision in a dedicated systemd user service."""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import secrets
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RELATIVE_CANDIDATE = "frontier/qwen35-receiptagent-v3/candidate.json"
SUPERVISOR = HERE / "supervise_training.py"
GIT = "/usr/bin/git"
SYSTEMD_RUN = "/usr/bin/systemd-run"
SYSTEMCTL = "/usr/bin/systemctl"
CLEANUP_TIMEOUT_SECONDS = 30
SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}")
SERVICE_NAME = re.compile(r"szl-ra3-supervisor-[0-9a-f]{32}")
SYSTEMD_PROPERTIES = (
    "KillMode=control-group",
    "SendSIGKILL=yes",
    "TimeoutStopSec=20s",
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
    if not SUPERVISOR.is_file():
        raise LauncherError(f"supervisor source is unavailable at {SUPERVISOR}")


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


def invoke_with_cleanup(
    command: Sequence[str],
    *,
    service_name: str,
    run_id: str,
    attempt_path: str,
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
        return 128 + interrupted.signum
    if launched is None:
        raise LauncherError("systemd-run returned no process result")
    return launched.returncode if 0 <= launched.returncode <= 255 else 1


def systemd_command(
    *,
    service_name: str,
    python_executable: str,
    source_commit: str,
    run_kind: str,
) -> list[str]:
    if SERVICE_NAME.fullmatch(service_name) is None:
        raise LauncherError("systemd service name is not safe")
    if SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise LauncherError("source commit is not exact lowercase hexadecimal")
    if run_kind not in {"smoke", "full"}:
        raise LauncherError("run kind is unsupported")
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
                for key, value in sorted(SUPERVISOR_ENVIRONMENT.items())
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
        service_name = generate_service_name(args.run_kind)
        run_id, attempt_path = attempt_identity(candidate, service_name)
        command = systemd_command(
            service_name=service_name,
            python_executable=python_executable,
            source_commit=args.source_commit,
            run_kind=args.run_kind,
        )
        return invoke_with_cleanup(
            command,
            service_name=service_name,
            run_id=run_id,
            attempt_path=attempt_path,
        )
    except LauncherError as exc:
        parser.error(str(exc))
    except OSError as exc:
        parser.error(f"could not invoke /usr/bin/systemd-run: {type(exc).__name__}")
    raise AssertionError("argparse error paths do not return")


if __name__ == "__main__":
    sys.exit(main())
