"""Offline tests for typed credentials and non-publishing preflight evidence."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import invariants_keyless_credentials as credentials

ENV = {
    "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": credentials.REPOSITORY,
    "GITHUB_REF": "refs/heads/main", "GITHUB_WORKFLOW_REF": credentials.WORKFLOW_REF,
    "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_SHA": "b" * 40,
}
MODEL = "hf_model_fixture"
KERNEL = "hf_kernel_fixture"


class CredentialTests(unittest.TestCase):
    def test_closed_resources_and_honest_typed_access_validation(self):
        api = mock.Mock()
        api.list_repo_refs.return_value = SimpleNamespace(branches=[
            SimpleNamespace(name="main", target_commit="c"*40),
            SimpleNamespace(name="v1", target_commit="d"*40),
        ])
        supplier = mock.Mock(side_effect=[MODEL, KERNEL])
        pair = credentials.acquire_pair(environment=ENV, supplier=supplier, api=api)
        self.assertEqual([call.args[0] for call in supplier.call_args_list],
                         [credentials.TARGET, "kernels/" + credentials.TARGET])
        self.assertEqual(pair.model, MODEL)
        self.assertEqual(pair.kernel, KERNEL)
        self.assertNotIn(MODEL, repr(pair))
        self.assertNotIn(KERNEL, repr(pair))
        self.assertEqual(api.auth_check.call_args_list, [
            mock.call(repo_id=credentials.TARGET, repo_type="model", token=MODEL, write=True),
        ])
        api.list_repo_refs.assert_called_once_with(
            credentials.TARGET, repo_type="kernel", token=KERNEL)

    def test_untrusted_context_never_requests_a_token(self):
        for key in ENV:
            with self.subTest(key=key):
                env = {**ENV, key: "wrong"}
                supplier = mock.Mock()
                with self.assertRaises(credentials.KeylessCredentialError):
                    credentials.acquire_pair(environment=env, supplier=supplier, api=mock.Mock())
                supplier.assert_not_called()

    def test_missing_kernel_binding_does_not_return_model_only_pair(self):
        supplier = mock.Mock(side_effect=[MODEL, credentials.KeylessCredentialError("REJECTED")])
        with self.assertRaises(credentials.KeylessCredentialError):
            credentials.acquire_pair(environment={**ENV, "HF_TOKEN": "hf_legacy_fixture"},
                                     supplier=supplier, api=mock.Mock())
        self.assertEqual(supplier.call_count, 2)

    def test_denied_write_has_fixed_nonsecret_error(self):
        api = mock.Mock()
        api.auth_check.side_effect = RuntimeError("private-response-" + MODEL)
        with self.assertRaises(credentials.KeylessCredentialError) as caught:
            credentials.acquire_pair(environment=ENV, supplier=lambda *_: MODEL, api=api)
        self.assertNotIn(MODEL, str(caught.exception))
        self.assertEqual(str(caught.exception), "TARGET_ACCESS_VALIDATION_FAILED")

    def test_exchange_removes_ambient_credentials_and_uses_ephemeral_home(self):
        env = {**ENV, "PATH": os.environ.get("PATH", ""), "HF_TOKEN": MODEL,
               "HF_TOKEN_PATH": "/sensitive", "HF_OIDC_ID_TOKEN": "stale-jwt",
               "HF_HOME": "/cached", "HF_ENDPOINT": "https://evil.invalid",
               "HUGGING_FACE_HUB_TOKEN": MODEL, "HUGGINGFACE_TOKEN": MODEL}
        with mock.patch.object(credentials.subprocess, "run",
                return_value=SimpleNamespace(returncode=0, stdout=KERNEL+"\n", stderr="")) as run:
            result = credentials.exchange("kernels/" + credentials.TARGET, env)
        child = run.call_args.kwargs["env"]
        for key in ("HF_TOKEN", "HF_TOKEN_PATH", "HF_OIDC_ID_TOKEN", "HF_ENDPOINT",
                    "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            self.assertNotIn(key, child)
        self.assertEqual(result, KERNEL)
        self.assertFalse(Path(child["HF_HOME"]).exists())
        self.assertEqual(run.call_args.args[0], ["hf", "auth", "token"])
        self.assertEqual(run.call_args.kwargs["timeout"], 60)

    def test_exchange_rejects_mixed_output_and_error_text(self):
        for output in ("", MODEL+"\n"+KERNEL, "warning\n"+MODEL, "not-a-token"):
            with self.subTest(output=output), mock.patch.object(credentials.subprocess, "run",
                    return_value=SimpleNamespace(returncode=0, stdout=output, stderr="")):
                with self.assertRaises(credentials.KeylessCredentialError):
                    credentials.exchange(credentials.TARGET, ENV)
        with mock.patch.object(credentials.subprocess, "run",
                return_value=SimpleNamespace(returncode=1, stdout="", stderr=MODEL)):
            with self.assertRaisesRegex(credentials.KeylessCredentialError, "^OIDC_EXCHANGE_REJECTED$"):
                credentials.exchange(credentials.TARGET, ENV)

    def test_undeclared_resource_is_rejected_before_cli(self):
        with mock.patch.object(credentials.subprocess, "run") as run:
            with self.assertRaises(credentials.KeylessCredentialError):
                credentials.exchange("SZLHOLDINGS/another-repo", ENV)
            run.assert_not_called()

    def test_preflight_replaces_stale_success_with_denial_without_secret(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text('{"status":"BOTH_TARGETS_WRITE_VALIDATED"}')
            out = io.StringIO()
            with mock.patch.object(credentials, "acquire_pair", side_effect=RuntimeError(MODEL)), \
                    contextlib.redirect_stdout(out):
                self.assertEqual(credentials.preflight(path), 1)
            report = json.loads(path.read_text())
            self.assertEqual(report["status"], "KEYLESS_CREDENTIALS_UNAVAILABLE")
            self.assertFalse(report["publication_verified"])
            self.assertNotIn(MODEL, path.read_text()+out.getvalue())

    def test_atomic_write_failure_leaves_no_temporary_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "receipt.json"
            with mock.patch.object(credentials.os, "replace", side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    credentials.atomic_report(path, {"status": "CHECKING"})
            self.assertEqual(list(root.iterdir()), [])


    def test_real_sdk_transport_does_not_call_unsupported_kernel_auth_check(self):
        # Other suites install process-global SDK stubs. Run real SDK transport
        # in an isolated interpreter, not against those stubs or a skipped test.
        script = r"""
import sys
import unittest
from unittest import mock
sys.path.insert(0, sys.argv[1])
import invariants_keyless_credentials as credentials
import httpx
from huggingface_hub import HfApi
import huggingface_hub.hf_api as hf_api
case = unittest.TestCase()
model, kernel = "hf_model_fixture", "hf_kernel_fixture"
environment = {
    "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": credentials.REPOSITORY,
    "GITHUB_REF": "refs/heads/main", "GITHUB_WORKFLOW_REF": credentials.WORKFLOW_REF,
    "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_SHA": "b" * 40,
}
requests = []
def handle(request):
    requests.append(request)
    case.assertEqual(request.url.host, "huggingface.co")
    if request.url.path == "/api/models/SZLHOLDINGS/szl-invariants/auth-check/write":
        case.assertEqual(request.headers["authorization"], "Bearer " + model)
        return httpx.Response(200, json={})
    case.assertEqual(request.url.path, "/api/kernels/SZLHOLDINGS/szl-invariants/refs")
    case.assertEqual(request.headers["authorization"], "Bearer " + kernel)
    return httpx.Response(200, json={"branches": [
        {"name": "main", "ref": "refs/heads/main", "targetCommit": "c" * 40},
        {"name": "v1", "ref": "refs/heads/v1", "targetCommit": "d" * 40},
    ], "tags": [], "converts": []})
with httpx.Client(transport=httpx.MockTransport(handle)) as client, \
        mock.patch.object(hf_api, "get_session", return_value=client), \
        mock.patch("socket.socket.connect", side_effect=AssertionError("network forbidden")):
    pair = credentials.acquire_pair(environment=environment,
        supplier=mock.Mock(side_effect=[model, kernel]),
        api=HfApi(endpoint="https://huggingface.co", token=False))
case.assertEqual(pair.model, model)
case.assertEqual(pair.kernel, kernel)
case.assertEqual(len(requests), 2)
print("SDK_TRANSPORT_VERIFIED_OFFLINE")
"""
        command = [sys.executable, "-I"]
        if sys.flags.optimize:
            command.append("-O")
        command.extend(["-c", script, str(Path(__file__).resolve().parents[1] / "tools")])
        with tempfile.TemporaryDirectory(prefix="sdk-transport-") as home:
            environment = {key: os.environ[key] for key in
                           ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")
                           if key in os.environ}
            environment.update({"HF_HOME": home, "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"})
            # Deliberately contaminate the parent to prove interpreter isolation.
            with mock.patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace()}):
                result = credentials.subprocess.run(
                    command, env=environment, capture_output=True, text=True,
                    check=False, timeout=30,
                )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("SDK_TRANSPORT_VERIFIED_OFFLINE", result.stdout)

    def test_incomplete_or_invalid_kernel_refs_refuse_pair(self):
        for branches in ([], [SimpleNamespace(name="main", target_commit="c"*40)],
                         [SimpleNamespace(name="main", target_commit="c"*40),
                          SimpleNamespace(name="v1", target_commit="main")]):
            api = mock.Mock()
            api.list_repo_refs.return_value = SimpleNamespace(branches=branches)
            with self.subTest(branches=branches), self.assertRaisesRegex(
                    credentials.KeylessCredentialError, "INVALID_KERNEL_REFS"):
                credentials.acquire_pair(environment=ENV,
                    supplier=mock.Mock(side_effect=[MODEL, KERNEL]), api=api)

    def test_preflight_explicitly_disclaims_independent_kernel_write_check(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(credentials, "acquire_pair", return_value=
                                  credentials.InvariantsCredentials(MODEL, KERNEL)), \
                contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp) / "preflight.json"
            self.assertEqual(credentials.preflight(path), 0)
            report = json.loads(path.read_text())
        self.assertEqual(report["status"], "BOTH_REPO_SCOPED_CREDENTIALS_VALIDATED")
        self.assertEqual(report["kernel_access_check"], "READ_REFS_ONLY")
        self.assertFalse(report["kernel_write_independently_verified"])


if __name__ == "__main__":
    unittest.main()
