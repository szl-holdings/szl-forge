"""Exercise only installed wheel modules/assets from a fresh outside-source cwd."""
from importlib import metadata
from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import build_opener, ProxyHandler, Request

import oac_clinical_resources
import oac_operational_health
import oac_stack_api
import owned_agent_clinical_control


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


@contextmanager
def temporary_workspace():
    temporary = tempfile.TemporaryDirectory(prefix="oac-installed-app-")
    try:
        yield temporary.name
    finally:
        # Windows can briefly retain process/log handles after confirmed Job
        # termination. Cleanup must succeed; retry only that bounded condition.
        deadline = time.monotonic() + 5
        while True:
            try:
                temporary.cleanup()
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)


def main():
    distribution = metadata.distribution("szl-oac-clinical-gateway")
    require(distribution.version == "2.5.1", "unexpected installed distribution")
    installation = Path(distribution.locate_file("")).resolve()
    for module in (oac_clinical_resources, oac_operational_health, oac_stack_api, owned_agent_clinical_control):
        require(Path(module.__file__).resolve().is_relative_to(installation), "test imported source instead of installed module")
    with temporary_workspace() as temp:
        root = Path(temp).resolve()
        target = root / "workspace"
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        command = [sys.executable, "-I", "-B", "-m", "oac_clinical_resources", "--directory", str(target)]
        prepared = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, timeout=20)
        require(prepared.returncode == 0, f"workspace preparation failed: {prepared.stderr}")
        receipt = json.loads(prepared.stdout)
        require(receipt["state"] == "PREPARED_SYNTHETIC_SHADOW_WORKSPACE", "preparation state mismatch")
        require(all(receipt[k] is False for k in ("clinical_use_authorized", "real_phi_authorized", "site_validated", "device_connection_started")), "preparation claimed authority")
        repeated = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, timeout=20)
        require(repeated.returncode == 2, "existing workspace was not rejected")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        token = secrets.token_urlsafe(48)
        origin = f"http://127.0.0.1:{port}"
        environment["OAC_API_KEY"] = token
        environment["OAC_ALLOWED_ORIGINS"] = origin
        server_command = [sys.executable, "-I", "-B", "-m", "oac_stack_api", "--port", str(port),
                          "--data-root", str(target), "--state-dir", "state"]
        if os.name == "nt":
            # Windows venv python.exe is a launcher; killing it alone leaves the
            # child API running. Assign before resume and confirm tree teardown.
            server = owned_agent_clinical_control.launch_windows_job(
                "portable-wheel-verification", server_command, root, environment, root / "server.log")
        else:
            server = subprocess.Popen(server_command, cwd=root, env=environment,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        opener = build_opener(ProxyHandler({}))

        def request(path, *, body=None, authorized=True):
            headers = {"Origin": origin}
            if authorized:
                headers["Authorization"] = "Bearer " + token
            if body is not None:
                headers["Content-Type"] = "application/json"
            req = Request(origin + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
            with opener.open(req, timeout=3) as response:
                return response.headers.get_content_type(), response.read()

        try:
            deadline = time.monotonic() + 15
            while True:
                exited = server.root_exited() if os.name == "nt" else server.poll() is not None
                require(not exited, "installed API process exited before readiness")
                try:
                    _, raw = request("/api/health", authorized=False)
                    break
                except (URLError, TimeoutError):
                    require(time.monotonic() < deadline, "installed API readiness timed out")
                    time.sleep(0.1)
            health = json.loads(raw)
            for key in ("clinical_use_authorized", "real_phi_authorized", "site_validated"):
                require(health.get(key) is False, "health claimed clinical authority")
            content_type, html = request("/", authorized=False)
            require(content_type == "text/html" and html == (target / "clinical-gateway/frontend/index.html").read_bytes(), "installed UI bytes mismatch")
            try:
                request("/api/operational-health", authorized=False)
            except HTTPError as exc:
                require(exc.code == 401, "advisory missing-auth status mismatch")
            else:
                raise RuntimeError("advisory unexpectedly allowed unauthenticated access")
            _, raw = request("/api/operational-health")
            require(json.loads(raw)["configured"] is True, "packaged advisory was not loaded")
            example = json.loads((target / "clinical-gateway/operational-model/example-input.json").read_text())
            _, raw = request("/api/operational-health/score", body=example)
            advisory = json.loads(raw)["advisory"]
            authority = advisory["authority"]
            require(set(authority) == {"acknowledgement", "clinical_decision", "device_control",
                                       "result_interpretation", "result_release"}
                    and all(value is False for value in authority.values()), "advisory authority contract mismatch")
            non_operational = {"features": dict(example["features"], patient_id="forbidden")}
            try:
                request("/api/operational-health/score", body=non_operational)
            except HTTPError as exc:
                require(exc.code == 400, "non-operational field rejection mismatch")
            else:
                raise RuntimeError("advisory allowed a non-operational field")
            result = {"state": "INSTALLED_WHEEL_WORKSPACE_API_VERIFIED", "distribution_version": distribution.version,
                      "asset_count": len(receipt["files"]), "manifest_sha256": receipt["manifest_sha256"],
                      "ui": "BYTE_MATCHED", "advisory": "AUTHENTICATED_SYNTHETIC_ONLY",
                      "clinical_use_authorized": False, "real_phi_authorized": False, "site_validated": False}
        finally:
            if os.name == "nt":
                try:
                    require(server.terminate_and_confirm(timeout_seconds=5) == 0, "installed API job descendants remain alive")
                finally:
                    server.close()
            else:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)
                if server.stderr:
                    server.stderr.close()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
