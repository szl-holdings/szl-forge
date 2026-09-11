"""Job-local environment for the AsyncGRPO LoRA / vLLM evaluation lane.

Creates an isolated directory, writes the recorded TRL / PEFT / vLLM closure,
and optionally installs into a venv under that directory. It never writes
site-packages of the calling interpreter, never invents PEFT/vLLM versions,
and never starts vLLM.
"""
from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import venv
from pathlib import Path

from frontier.asyncgrpo_lora_closure import (
    PEFT_SPEC,
    TRL_GIT_REQUIREMENT,
    VLLM_SPEC,
    recorded_closure_identity,
    recorded_requirements_text,
    spec_satisfied,
)
from frontier.asyncgrpo_lora_contract import (
    EvaluationContractError,
    TRL_REVISION,
    serving_cache_path,
)

SCHEMA = "szl.asyncgrpo-lora-job-local/v1"
INSTALL_TIMEOUT_SECONDS = 180
MAX_PIP_LOG_BYTES = 16384


class JobLocalInstallError(EvaluationContractError):
    """Job-local install refused or failed closed."""


def read_direct_url(package: str, *, environ: dict[str, str] | None = None) -> dict[str, object]:
    found: dict[str, object] = {
        "package": package,
        "present": False,
        "version": None,
        "directUrlRevision": None,
        "pep610": False,
    }
    previous = os.environ.get("PYTHONPATH")
    try:
        if environ is not None and "PYTHONPATH" in environ:
            os.environ["PYTHONPATH"] = environ["PYTHONPATH"]
        distribution = metadata.distribution(package)
    except metadata.PackageNotFoundError:
        return found
    finally:
        if environ is not None:
            if previous is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous
    found["present"] = True
    found["version"] = distribution.version
    document = distribution.read_text("direct_url.json")
    if not document or len(document) > 16384:
        return found
    try:
        payload = json.loads(document)
    except json.JSONDecodeError:
        return found
    vcs = payload.get("vcs_info") if isinstance(payload, dict) else None
    if isinstance(vcs, dict) and type(vcs.get("commit_id")) is str:
        found["directUrlRevision"] = vcs["commit_id"]
        found["pep610"] = True
    return found


def _reject_symlink_tree(root: Path) -> Path:
    if root.is_symlink():
        raise JobLocalInstallError("job-local root cannot be a symlink")
    resolved = root.resolve()
    if resolved != root and root.exists() and root.is_symlink():
        raise JobLocalInstallError("job-local root cannot be a symlink")
    return resolved


def _peft_status(peft: dict[str, object]) -> dict[str, object]:
    version = peft.get("version")
    matches = bool(peft.get("present")) and spec_satisfied(version, PEFT_SPEC) if peft.get("present") else False
    return {
        **peft,
        "spec": PEFT_SPEC,
        "status": "MATCHED" if matches else ("PRESENT_OUT_OF_SPEC" if peft.get("present") else "ABSENT"),
        "matchesRecordedSpec": matches,
        "pinInvented": False,
    }


def _vllm_status(vllm: dict[str, object]) -> dict[str, object]:
    version = vllm.get("version")
    matches = bool(vllm.get("present")) and spec_satisfied(version, VLLM_SPEC) if vllm.get("present") else False
    return {
        **vllm,
        "spec": VLLM_SPEC,
        "status": "MATCHED" if matches else ("PRESENT_OUT_OF_SPEC" if vllm.get("present") else "ABSENT"),
        "matchesRecordedSpec": matches,
        "pinInvented": False,
    }


def describe_installed_closure(*, environ: dict[str, str] | None = None) -> dict[str, object]:
    trl = read_direct_url("trl", environ=environ)
    peft = _peft_status(read_direct_url("peft", environ=environ))
    vllm = _vllm_status(read_direct_url("vllm", environ=environ))
    exact_trl = trl.get("directUrlRevision") == TRL_REVISION
    compatible = bool(
        exact_trl
        and peft.get("matchesRecordedSpec") is True
        and vllm.get("matchesRecordedSpec") is True
    )
    return {
        "schema": SCHEMA,
        "productionRuntime": False,
        "jobLocal": True,
        "recorded": recorded_closure_identity(),
        "trl": {
            **trl,
            "requiredRevision": TRL_REVISION,
            "requirement": TRL_GIT_REQUIREMENT,
            "exact": exact_trl,
        },
        "peft": peft,
        "vllm": vllm,
        "compatibleExactPeftVllmClosure": compatible,
    }


def prepare_job_local(output_dir: Path, *, install: bool = False) -> dict[str, object]:
    """Create the job-local tree. Installs only into output_dir/.venv when asked."""
    if type(install) is not bool:
        raise JobLocalInstallError("install must be an explicit boolean")
    if not isinstance(output_dir, Path):
        raise JobLocalInstallError("output_dir must be a pathlib.Path")
    root = _reject_symlink_tree(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / ".vllm_lora").mkdir(exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    (root / "receipts").mkdir(exist_ok=True)
    serving_cache_path(root, root / ".vllm_lora" / ".keep")
    req_path = root / "job-local-requirements.txt"
    req_path.write_text(recorded_requirements_text(), encoding="utf-8")
    install_report: dict[str, object] = {
        "attempted": False,
        "status": "NOT_ATTEMPTED",
        "venvPython": None,
        "productionInterpreter": False,
    }
    if install:
        install_report = install_job_local(root)
    closure = describe_installed_closure()
    if install and install_report.get("venvPython"):
        venv_site = Path(str(install_report["venvPython"])).resolve().parent.parent / "lib"
        extra = os.pathsep.join(str(path) for path in venv_site.glob("python*/site-packages"))
        if extra:
            closure = describe_installed_closure(environ={"PYTHONPATH": extra})
    closure_path = root / "closure.json"
    payload = {
        **closure,
        "install": install_report,
        "productionRuntime": False,
    }
    closure_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "outputDir": str(root),
        "requirementsPath": str(req_path),
        "closurePath": str(closure_path),
        "closure": payload,
        "install": install_report,
        "productionRuntime": False,
        "venvPython": install_report.get("venvPython"),
    }


def _venv_python(root: Path) -> Path:
    unix = root / ".venv" / "bin" / "python"
    windows = root / ".venv" / "Scripts" / "python.exe"
    if unix.exists():
        return unix
    if windows.exists():
        return windows
    raise JobLocalInstallError("job-local venv python is missing")


def install_job_local(output_dir: Path) -> dict[str, object]:
    """Create output_dir/.venv and pip-install the recorded closure into it.

    Refuses the calling interpreter. A failed pip run is recorded, not raised
    as success. GPU wheels are not invented; a CUDA-less host is allowed to
    fail vLLM installation.
    """
    root = _reject_symlink_tree(output_dir)
    if root == Path(sys.prefix).resolve() or root == Path(sys.exec_prefix).resolve():
        raise JobLocalInstallError("refusing to install into the production prefix")
    venv_dir = root / ".venv"
    if venv_dir.exists() and venv_dir.is_symlink():
        raise JobLocalInstallError("job-local venv cannot be a symlink")
    builder = venv.EnvBuilder(with_pip=True, clear=False, symlinks=False)
    builder.create(venv_dir)
    python = _venv_python(root)
    if python.resolve() == Path(sys.executable).resolve():
        raise JobLocalInstallError("venv python resolved to the calling interpreter")
    req = root / "job-local-requirements.txt"
    command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "-r",
        str(req),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=INSTALL_TIMEOUT_SECONDS,
            check=False,
            cwd=str(root),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "attempted": True,
            "status": "FAILED",
            "reason": "pip_unavailable_or_timed_out",
            "venvPython": str(python),
            "productionInterpreter": False,
            "returncode": None,
            "detail": type(exc).__name__,
        }
    log = (completed.stdout or b"") + (completed.stderr or b"")
    truncated = log[:MAX_PIP_LOG_BYTES]
    status = "INSTALLED" if completed.returncode == 0 else "FAILED"
    return {
        "attempted": True,
        "status": status,
        "reason": None if completed.returncode == 0 else "pip_install_failed",
        "venvPython": str(python),
        "productionInterpreter": False,
        "returncode": int(completed.returncode),
        "logBytes": len(log),
        "logHeadSha16": __import__("hashlib").sha256(truncated).hexdigest()[:16],
    }
