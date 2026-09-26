"""Verify installed wheel UI/API/receipt restart behavior using synthetic data only.

Run with ``python -I -B verify_installed.py`` after installing the wheel. No model,
cloud, arbitrary HTML execution, or real customer information is involved.
"""

import argparse
import base64
from contextlib import contextmanager
import hashlib
import http.client
from importlib import metadata, resources
import json
from pathlib import Path
import platform
import re
import secrets
import sys
import sysconfig
import tempfile
import threading
import uuid
from unittest.mock import patch
import zipfile

import szl_access
from szl_access import api, benchmark, engine, model


SOURCE = '<form><label>Email</label><input id="email" type="email"></form>'
EXPECTED = '<form><label for="email">Email</label><input id="email" type="email"></form>'


def require(condition, description):
    if not condition:
        raise AssertionError(description)


@contextmanager
def serving(directory, key):
    server = api.AccessServer(0, directory, key)
    thread = threading.Thread(target=server.serve_forever, name="szl-access-installed-verifier")
    try:
        thread.start()
        yield server
    finally:
        if thread.is_alive():
            server.shutdown()
            thread.join(timeout=5)
        server.server_close()
        require(not thread.is_alive(), "verification server thread did not stop")


def request(server, key, path, *, body=None, authenticated=True, headers=None):
    actual = {"Authorization": "Bearer " + key} if authenticated else {}
    actual.update(headers or {})
    encoded = None
    if body is not None:
        encoded = json.dumps(body).encode("utf-8")
        actual["Content-Type"] = "application/json"
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request("POST" if body is not None else "GET", path, body=encoded, headers=actual)
        response = connection.getresponse()
        raw = response.read(131073)
        require(len(raw) <= 131072, "response exceeds verification size bound")
        return response.status, raw, dict(response.getheaders())
    finally:
        connection.close()


def run_verification():
    require(sys.flags.isolated == 1, "run the verifier with Python -I")
    require(sys.prefix != sys.base_prefix, "use a dedicated installed-package virtual environment")
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    observed_modules = (szl_access, api, benchmark, engine, model)
    require(all(Path(module.__file__).resolve().is_relative_to(purelib) for module in observed_modules),
            "all application modules must come from installed site-packages")
    require(metadata.version("szl-access-lab") == szl_access.__version__, "installed version mismatch")
    require(not Path.cwd().resolve().is_relative_to(Path(__file__).resolve().parents[1]),
            "run verification from outside the access-lab checkout")
    ui = resources.files("szl_access").joinpath("static/index.html").read_bytes()
    require(b"SZL Access" in ui and len(ui) > 1000, "installed UI resource missing or incomplete")
    require(b"\r\n" not in ui, "packaged UI must use LF line endings for stable CSP content hashes")
    checks = {"installed_package_origin": True, "installed_ui_resource": True}
    key = secrets.token_urlsafe(48)
    body = {"source": SOURCE, "mode": "deterministic", "request_id": str(uuid.uuid4())}

    with tempfile.TemporaryDirectory(prefix="szl-access-installed-state-") as temporary:
        state = Path(temporary) / "state"
        with patch.object(model, "propose", side_effect=AssertionError("No model calls allowed in installed verification")) as no_model:
            with serving(state, key) as server:
                status, raw, headers = request(server, key, "/", authenticated=False)
                require(status == 200 and raw == ui, "public UI must equal installed resource bytes")
                csp = headers.get("Content-Security-Policy", "")
                require("'unsafe-inline'" not in csp and "frame-ancestors 'none'" in csp, "UI CSP scope failed")
                for tag in (b"script", b"style"):
                    blocks = re.findall(b"<" + tag + rb"\b[^>]*>(.*?)</" + tag + b">", ui, re.S)
                    require(bool(blocks), "UI needs an actual packaged script and stylesheet")
                    for block in blocks:
                        token = "'sha256-" + base64.b64encode(hashlib.sha256(block).digest()).decode() + "'"
                        require(token in csp, "CSP hash does not cover exact installed inline bytes")
                checks["ui_exact_bytes_and_csp"] = True
                status, raw, _ = request(server, key, "/api/health", authenticated=False)
                health = json.loads(raw)
                require(status == 200 and health["model"]["enabled"] is False, "public health must truthfully disable model")
                for route in ("/api/examples", "/api/runs"):
                    require(request(server, key, route, authenticated=False)[0] == 401, "private GET accepted without authentication")
                require(request(server, key, "/api/runs", body=body, authenticated=False)[0] == 401, "mutation accepted without authentication")
                require(request(server, key, "/api/runs", body=body, headers={"Origin": "https://example.invalid"})[0] == 403, "foreign origin accepted")
                require(request(server, key, "/api/runs", body=body, headers={"Host": "example.invalid"})[0] == 403, "foreign Host accepted")
                status, raw, _ = request(server, key, "/api/examples")
                require(status == 200 and len(json.loads(raw)["examples"]) == 5, "installed synthetic examples missing")
                checks["public_private_auth_and_origin_scopes"] = True

                status, raw, _ = request(server, key, "/api/runs", body=body)
                repaired = json.loads(raw)
                require(status == 200 and repaired["state"] == "VERIFIED_SCOPED_REPAIR", "synthetic repair failed")
                require(repaired["output_html"] == EXPECTED, "repair changed bytes outside expected association")
                require(repaired["source_sha256"] == hashlib.sha256(SOURCE.encode()).hexdigest(), "source hash mismatch")
                require(repaired["output_sha256"] == hashlib.sha256(EXPECTED.encode()).hexdigest(), "output hash mismatch")
                unsigned = dict(repaired)
                receipt_hash = unsigned.pop("receipt_sha256")
                require(receipt_hash == api.digest(unsigned), "operational receipt hash mismatch")
                require(repaired["model"]["enabled"] is False and repaired["model"]["trained_for_accessibility"] is False,
                        "deterministic repair must not claim trained-model execution")
                checks["exact_scoped_repair_and_receipt"] = True
                summary_route = "/api/runs/" + repaired["run_id"] + "/summary"
                require(request(server, key, summary_route, authenticated=False)[0] == 401,
                        "receipt summary accepted without authentication")
                status, raw, summary_headers = request(server, key, summary_route)
                summary = json.loads(raw)
                require(status == 200 and summary["contains_source"] is False, "source-free summary export failed")
                require(summary["receipt_sha256"] == repaired["receipt_sha256"], "summary references another receipt")
                require(summary["export_sha256"] == api.digest({k: v for k, v in summary.items() if k != "export_sha256"}),
                        "summary canonical hash mismatch")
                require("output_html" not in summary and "applied_bindings" not in summary and "model" not in summary,
                        "summary contains prohibited detailed evidence")
                require(summary_headers.get("Content-Disposition") == f'attachment; filename="szl-access-{repaired["run_id"]}-summary.json"',
                        "summary download metadata missing")
                checks["authenticated_source_free_summary_export"] = True
                require(json.loads(request(server, key, "/api/runs", body=body)[1]) == repaired, "idempotent replay diverged")
                require(len(json.loads(request(server, key, "/api/runs")[1])["runs"]) == 1, "replay created another run")
                require(request(server, key, "/api/runs", body={**body, "source": SOURCE + " "})[0] == 409, "conflicting replay accepted")
                require(request(server, key, "/api/runs", body={**body, "mode": "ollama", "request_id": str(uuid.uuid4())})[0] == 409,
                        "disabled model request was not refused")
                checks["idempotent_replay_conflict_and_model_disabled"] = True
                unsafe = '<script>alert("synthetic")</script>'
                status, raw, _ = request(server, key, "/api/runs", body={**body, "source": unsafe, "request_id": str(uuid.uuid4())})
                refused = json.loads(raw)
                require(status == 200 and refused["state"] == "REVIEW_REQUIRED" and refused["output_html"] == unsafe,
                        "unsupported source was changed or claimed repaired")
                checks["unsupported_source_held_unchanged"] = True
            with serving(state, key) as restarted:
                status, raw, _ = request(restarted, key, "/api/runs/" + repaired["run_id"])
                require(status == 200 and json.loads(raw) == repaired, "saved receipt did not survive restart")
                require(json.loads(request(restarted, key, "/api/runs", body=body)[1]) == repaired, "replay after restart diverged")
                require(len(json.loads(request(restarted, key, "/api/runs")[1])["runs"]) == 2, "restart changed completed history")
                checks["durable_receipt_restart_and_replay"] = True
            no_model.assert_not_called()
            checks["no_model_calls"] = True
        checks["servers_stopped"] = True
    require(not Path(temporary).exists(), "temporary synthetic state did not clean up")
    checks["temporary_state_removed"] = True
    return {
        "schema": "szl-access-installed-application-v1",
        "status": "PASS",
        "version": szl_access.__version__,
        "python": platform.python_version(),
        "platform": platform.system(),
        "package_origin": "verified_dedicated_venv_site_packages",
        "ui_sha256": hashlib.sha256(ui).hexdigest(),
        "source_fixture_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
        "output_fixture_sha256": hashlib.sha256(EXPECTED.encode()).hexdigest(),
        "checks": checks,
        "limitations": [
            "Synthetic installed UI resource/API contract, not a witnessed browser or assistive-technology evaluation.",
            "No specialist weights, model execution, training, cloud jobs, patient data, or clinical authority.",
            "Local operational receipt integrity is not an independent signature or accessibility certificate.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--source-sha")
    args = parser.parse_args()
    if args.source_sha is not None and re.fullmatch(r"[0-9a-f]{40}", args.source_sha) is None:
        parser.error("--source-sha must be a full lowercase Git commit SHA")
    result = run_verification()
    if args.wheel:
        observed_files = {
            "szl_access/__init__.py": Path(szl_access.__file__).read_bytes(),
            "szl_access/api.py": Path(api.__file__).read_bytes(),
            "szl_access/benchmark.py": Path(benchmark.__file__).read_bytes(),
            "szl_access/engine.py": Path(engine.__file__).read_bytes(),
            "szl_access/model.py": Path(model.__file__).read_bytes(),
            "szl_access/static/index.html": resources.files("szl_access").joinpath("static/index.html").read_bytes(),
        }
        with zipfile.ZipFile(args.wheel) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), "wheel contains duplicate archive entries")
            require({name for name in names if name.startswith("szl_access/")} == set(observed_files),
                    "wheel application resource set differs from installed verifier contract")
            for name, raw in observed_files.items():
                require(archive.read(name) == raw, "wheel bytes differ from installed application: " + name)
        result["checks"]["provided_wheel_matches_installed_modules_and_ui"] = True
        result["wheel_sha256"] = hashlib.sha256(args.wheel.read_bytes()).hexdigest()
    if args.source_sha:
        result["declared_source_commit"] = args.source_sha
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
