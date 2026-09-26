"""Observe binary-only Omni prerequisites; never install the upstream runtime.

SPDX-License-Identifier: Apache-2.0
The explicit networked entry point extends the existing wheel-acquisition owner.
A resolver plan is neither artifact verification nor runtime qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PIP_VERSION = "26.2.1"
PREREQUISITE = "openai-whisper>=20250625"
MAX_REPORT = 8 * 1024 * 1024
MAX_WORKSPACE = 1024 * 1024 * 1024
TIMEOUT = 180
OWNED = (
    "tools/evaluate_vllm_omni_prerequisites.py",
    "tests/test_vllm_omni_prerequisites.py",
    ".github/workflows/vllm-omni-prerequisites.yml",
    "frontier/vllm-omni-029rc1/PREREQUISITES.md",
    "frontier/vllm-omni-029rc1/resolver-requirements.txt",
)
DENIED = {
    "dependencyClosureCaptured": False,
    "transitiveArtifactBytesVerified": False,
    "baseVllmQualified": False,
    "packageInstalled": False,
    "runtimeExecuted": False,
    "modelWeightsAcquired": False,
    "productionServingAuthorized": False,
    "hubPublicationAuthorized": False,
}


class PrerequisiteError(ValueError):
    """Incomplete or inconsistent evidence is not a successful resolution."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PrerequisiteError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def decode_report(raw: bytes) -> dict:
    require(len(raw) <= MAX_REPORT, "resolver report exceeds bound")
    value = json.loads(raw, object_pairs_hook=unique_pairs)
    require(type(value) is dict, "resolver report must be an object")
    return value


def validate_resolution(data: dict, root_name: str, root_identity: dict | None = None) -> dict:
    """Validate pip's reported plan, not downloaded transitive package bytes."""
    require(data.get("version") == "1", "unsupported pip report schema")
    require(data.get("pip_version") == PIP_VERSION, "resolver version drift")
    require(type(data.get("environment")) is dict, "missing resolver environment")
    rows = data.get("install")
    require(type(rows) is list and 0 < len(rows) <= 512, "invalid package inventory")
    packages = []
    names = set()
    for row in rows:
        require(type(row) is dict and row.get("is_yanked") is False, "yanked or malformed selection")
        meta, download = row.get("metadata"), row.get("download_info")
        require(type(meta) is dict and type(download) is dict, "missing selection metadata")
        name, version = meta.get("name"), meta.get("version")
        require(type(name) is str and re.fullmatch(r"[A-Za-z0-9_.-]+", name) is not None, "invalid package name")
        require(type(version) is str and 0 < len(version) <= 100, "missing package version")
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        require(normalized not in names, "duplicate selected package")
        names.add(normalized)
        url = download.get("url")
        require(type(url) is str and len(url) <= 4096, "invalid package URL")
        parsed = urlsplit(url)
        require(
            parsed.scheme == "https" and parsed.hostname == "files.pythonhosted.org"
            and parsed.port in (None, 443) and not parsed.username and not parsed.password
            and parsed.path.endswith(".whl") and not parsed.query and not parsed.fragment,
            "selection is not an admitted public wheel",
        )
        archive = download.get("archive_info")
        hashes = archive.get("hashes") if type(archive) is dict else None
        sha = hashes.get("sha256") if type(hashes) is dict else None
        require(type(sha) is str and re.fullmatch(r"[0-9a-f]{64}", sha) is not None, "missing artifact digest")
        packages.append({"name": normalized, "version": version, "url": url,
                         "reportedSha256": sha, "independentlyVerified": False})
    require(root_name in names, "required root absent from resolver plan")
    root = next(item for item in packages if item["name"] == root_name)
    if root_name == "openai-whisper":
        require(re.fullmatch(r"[0-9]{8}", root["version"]) is not None
                and int(root["version"]) >= 20250625, "Whisper version violates prerequisite")
    if root_name == "vllm-omni":
        require(type(root_identity) is dict, "missing pinned Omni identity")
        for key in ("version", "url", "reportedSha256"):
            require(root[key] == root_identity.get(key), "resolved Omni artifact drift")
    return {"schema": "pip-reported-binary-plan/v1", "packages": packages,
            "environment": data["environment"], "evidenceClass": "RESOLVER_REPORTED_NOT_BYTE_VERIFIED"}


def command(requirement: str, output: Path, wheel_reference: str) -> list[str]:
    require(requirement in (PREREQUISITE, wheel_reference), "unadmitted resolver input")
    return [sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check",
            "--no-input", "--no-cache-dir", "install", "--dry-run", "--ignore-installed",
            "--only-binary=:all:", "--index-url", "https://pypi.org/simple",
            "--timeout", "20", "--retries", "0", "--report", str(output), requirement]


def child_environment(workspace: Path) -> dict[str, str]:
    # Do not pass provider tokens, proxy configuration, PYTHONPATH or pip overrides.
    return {"PATH": os.defpath, "HOME": str(workspace), "TMPDIR": str(workspace),
            "PIP_CONFIG_FILE": os.devnull, "PYTHONNOUSERSITE": "1",
            "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"}


def execute(argv: list[str], workspace: Path, log: Path) -> dict:
    """Bound the owned child/process group. Linux-only, no shell evaluation."""
    started = time.monotonic()
    stop_reason = None
    with log.open("xb") as stream:
        process = subprocess.Popen(argv, cwd=workspace, env=child_environment(workspace),
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started > TIMEOUT:
                    stop_reason = "deadline"
                elif log.stat().st_size > MAX_REPORT:
                    stop_reason = "log_bound"
                elif sum(p.stat().st_size for p in workspace.rglob("*") if p.is_file()) > MAX_WORKSPACE:
                    stop_reason = "workspace_bound"
                if stop_reason is not None:
                    break
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    require(sum(p.stat().st_size for p in workspace.rglob("*") if p.is_file()) <= MAX_WORKSPACE,
            "resolver workspace exceeds bound")
    require(log.stat().st_size <= MAX_REPORT, "resolver log exceeds bound")
    raw = log.read_bytes()
    return {"returnCode": process.returncode, "stoppedBy": stop_reason,
            "logSha256": digest(raw), "logBytes": len(raw)}


def classify(observation: dict, has_report: bool) -> str:
    require(type(observation.get("returnCode")) is int, "invalid resolver exit code")
    if observation.get("stoppedBy") is not None:
        return "INCOMPLETE"
    if observation["returnCode"] != 0:
        return "BLOCKED_OR_UNAVAILABLE"
    require(has_report, "successful resolver omitted its report")
    return "RESOLVER_COMPLETED"


def source_identity() -> tuple[str, dict[str, str]]:
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                              capture_output=True, timeout=20).stdout
    source = git("rev-parse", "HEAD").decode().strip()
    require(re.fullmatch(r"[0-9a-f]{40}", source) is not None, "invalid executing source")
    files = {}
    for name in (*OWNED, "tools/evaluate_vllm_omni_wheel.py"):
        raw = git("show", source + ":" + name)
        require((ROOT / name).read_bytes() == raw, "dirty executing source: " + name)
        files[name] = digest(raw)
    return source, files


def run(output: Path) -> dict:
    result = {"schema": "szl.forge.omni-binary-prerequisites/v1",
              "observedAt": datetime.now(timezone.utc).isoformat(),
              "sourceRepository": "szl-holdings/szl-forge", "sourceRevision": None,
              "python": platform.python_version(), "platform": platform.platform(),
              "scope": "published Omni wheel requirements, no base-vLLM or GPU qualification",
              "state": "INCOMPLETE", "disposition": "HOLD", "phases": [], **DENIED}
    try:
        result["sourceRevision"], result["sourceFiles"] = source_identity()
        require(sys.platform == "linux", "this resolver lane is qualified for Linux execution only")
        require(sys.prefix != sys.base_prefix, "execute in the isolated resolver virtual environment")
        result["installedResolverVersion"] = importlib.metadata.version("pip")
        require(result["installedResolverVersion"] == PIP_VERSION, "unexpected installed pip")
        # Reuse the existing byte/source inspector; never reimplement its pins.
        from tools import evaluate_vllm_omni_wheel as acquisition
        acquired = acquisition.run()
        result["acquisition"] = acquired
        require(acquired.get("state") == "ACQUISITION_VERIFIED_RUNTIME_NOT_RUN", "upstream acquisition did not pass")
        require(PREREQUISITE in acquired["wheel"]["requiresDist"], "pinned prerequisite missing")
        wheel_ref = acquisition.WHEEL_URL + "#sha256=" + acquisition.WHEEL_SHA256
        # The known source-build prerequisite is tested first to avoid downloading
        # a large graph when binary-only installation is already impossible.
        phases = (("whisper-prerequisite", PREREQUISITE, "openai-whisper"),
                  ("omni-distribution", wheel_ref, "vllm-omni"))
        with tempfile.TemporaryDirectory(prefix="szl-omni-resolve-") as temporary:
            workspace = Path(temporary)
            for label, requirement, name in phases:
                report_path = workspace / (label + ".json")
                log_path = output / (label + ".log")
                argv = command(requirement, report_path, wheel_ref)
                observation = execute(argv, workspace, log_path)
                phase = {"name": label, "requirement": requirement, "command": argv,
                         **observation, "state": classify(observation, report_path.is_file())}
                result["phases"].append(phase)
                if phase["state"] != "RESOLVER_COMPLETED":
                    result["state"] = phase["state"]
                    break
                require(report_path.stat().st_size <= MAX_REPORT, "oversized report")
                raw = report_path.read_bytes()
                phase["plan"] = validate_resolution(decode_report(raw), name, {
                    "version": "0.29.0rc1", "url": acquisition.WHEEL_URL,
                    "reportedSha256": acquisition.WHEEL_SHA256,
                })
                phase["reportSha256"] = digest(raw)
                with (output / (label + "-pip-report.json")).open("xb") as retained:
                    retained.write(raw)
            else:
                result["state"] = "OMNI_BINARY_PLAN_REPORTED_NOT_INSTALLED"
        source, files = source_identity()
        require(source == result["sourceRevision"] and files == result["sourceFiles"], "source moved during preflight")
    except (PrerequisiteError, OSError, ValueError, subprocess.SubprocessError) as error:
        result["state"] = "INCOMPLETE"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observe", action="store_true")
    args = parser.parse_args()
    require(args.observe, "networked resolution requires explicit --observe")
    output = ROOT / "reports" / "omni-prerequisites"
    output.mkdir(parents=True, exist_ok=False)
    result = run(output)
    with (output / "observation.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"state": result["state"], "disposition": result["disposition"]}))
    return 0 if result["state"] == "OMNI_BINARY_PLAN_REPORTED_NOT_INSTALLED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
