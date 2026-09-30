"""Offline contracts for the dedicated HF census OIDC identity."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from tools import hf_census_keyless_credentials as credentials


SOURCE = "a" * 40
USER_TOKEN = "hf_oauth_fixture-only-read-token"
REPO_TOKEN = "hf_jwt_fixture-repo-write-token"
ENV = {
    "GITHUB_ACTIONS": "true",
    "GITHUB_REPOSITORY": credentials.REPOSITORY,
    "GITHUB_REF": "refs/heads/main",
    "GITHUB_WORKFLOW_REF": credentials.WORKFLOW_REF,
    "GITHUB_EVENT_NAME": "workflow_dispatch",
    "GITHUB_SHA": SOURCE,
    "ACTIONS_ID_TOKEN_REQUEST_URL": "https://actions.example.invalid/token",
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "fixture-github-oidc-request-token",
    "PATH": os.environ.get("PATH", ""),
}


class ContextTests(unittest.TestCase):
    def test_source_verifier_accepts_the_clean_canonical_head(self):
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        credentials.verify_source(head)

    def test_exact_protected_workflow_context_is_required(self):
        credentials.require_workflow_context(ENV, SOURCE)
        for key in (
            "GITHUB_ACTIONS",
            "GITHUB_REPOSITORY",
            "GITHUB_REF",
            "GITHUB_WORKFLOW_REF",
            "GITHUB_EVENT_NAME",
            "GITHUB_SHA",
        ):
            with self.subTest(key=key):
                changed = {**ENV, key: "wrong"}
                with self.assertRaises(credentials.CensusCredentialError):
                    credentials.require_workflow_context(changed, SOURCE)
        for key in (
            "ACTIONS_ID_TOKEN_REQUEST_URL",
            "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
        ):
            with self.subTest(key=key):
                missing = {name: value for name, value in ENV.items() if name != key}
                with self.assertRaises(credentials.CensusCredentialError):
                    credentials.require_workflow_context(missing, SOURCE)

    def test_binding_is_exact_user_resource_without_inference_or_wildcards(self):
        binding = credentials.binding_receipt("REQUIRED_UNVERIFIED")
        self.assertEqual(binding["resource"], "betterwithage")
        self.assertNotIn("/", binding["resource"])
        self.assertEqual(
            binding["claims"],
            {
                "repository": "szl-holdings/szl-forge",
                "branch": "main",
                "workflow": "public-estate-file-census.yml",
            },
        )
        self.assertFalse(binding["allow_inference_providers"])
        self.assertNotIn("*", repr(binding))


class ExchangeTests(unittest.TestCase):
    def test_exchange_uses_isolated_home_and_strips_ambient_hf_credentials(self):
        contaminated = {
            **ENV,
            "HF_TOKEN": "hf_static-secret",
            "HF_HOME": "cached-home",
            "HF_ENDPOINT": "https://evil.invalid",
            "HF_OIDC_RESOURCE": "wrong-resource",
            "HF_OIDC_ID_TOKEN": "stale-id-token",
            "HUGGINGFACE_TOKEN": "hf_other-secret",
            "HUGGING_FACE_HUB_TOKEN": "hf_other-secret",
            "GITHUB_TOKEN": "github-repository-token",
            "GH_TOKEN": "github-cli-token",
        }
        with mock.patch.object(
            credentials.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout=USER_TOKEN + "\n", stderr=""),
        ) as run:
            token = credentials.exchange(contaminated, SOURCE)

        self.assertEqual(token, USER_TOKEN)
        child = run.call_args.kwargs["env"]
        self.assertEqual(child["HF_OIDC_RESOURCE"], "betterwithage")
        self.assertEqual(child["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")
        for key in (
            "HF_TOKEN",
            "HF_ENDPOINT",
            "HF_OIDC_ID_TOKEN",
            "HUGGINGFACE_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "GITHUB_TOKEN",
            "GH_TOKEN",
        ):
            self.assertNotIn(key, child)
        self.assertFalse(Path(child["HF_HOME"]).exists())
        self.assertEqual(run.call_args.args[0], ["hf", "auth", "token"])
        self.assertEqual(run.call_args.kwargs["timeout"], 60)

    def test_repo_write_token_mixed_output_and_failures_are_rejected(self):
        outcomes = (
            SimpleNamespace(returncode=0, stdout=REPO_TOKEN + "\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="warning\n" + USER_TOKEN, stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=1, stdout="", stderr=USER_TOKEN),
        )
        for outcome in outcomes:
            with self.subTest(outcome=outcome), mock.patch.object(
                credentials.subprocess, "run", return_value=outcome
            ), self.assertRaises(credentials.CensusCredentialError):
                credentials.exchange(ENV, SOURCE)


class ObserverTests(unittest.TestCase):
    def test_observer_receives_only_user_token_and_no_oidc_minting_authority(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"status": "FILE_METADATA_OBSERVED_NOT_QUALIFIED", "populations": {}}
            ),
            stderr="",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "observation.json"

            def complete(*_args, **_kwargs):
                output.write_text('{}\n', encoding="utf-8")
                return completed

            with mock.patch.object(
                credentials.subprocess, "run", side_effect=complete
            ) as run:
                result = credentials.run_observer(
                    USER_TOKEN,
                    SOURCE,
                    output,
                    ENV,
                )

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.status, "FILE_METADATA_OBSERVED_NOT_QUALIFIED")
        child = run.call_args.kwargs["env"]
        self.assertEqual(child["HF_CENSUS_TOKEN"], USER_TOKEN)
        for key in (
            "GITHUB_TOKEN",
            "GH_TOKEN",
            "ACTIONS_ID_TOKEN_REQUEST_URL",
            "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
            "HF_TOKEN",
            "HF_OIDC_RESOURCE",
        ):
            self.assertNotIn(key, child)
        self.assertTrue(run.call_args.kwargs["capture_output"])
        self.assertEqual(run.call_args.kwargs["timeout"], credentials.OBSERVER_TIMEOUT)

    def test_unrecognized_observer_output_is_fail_closed(self):
        completed = SimpleNamespace(returncode=0, stdout=USER_TOKEN, stderr="")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            credentials.subprocess, "run", return_value=completed
        ), self.assertRaises(credentials.CensusCredentialError):
            credentials.run_observer(
                USER_TOKEN,
                SOURCE,
                Path(directory) / "observation.json",
                ENV,
            )


class ReceiptTests(unittest.TestCase):
    def test_missing_binding_retains_fixed_nonsecret_failure_receipt(self):
        secret = "hf_oauth_never-retain-this"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "credential.json"
            output = root / "observation.json"
            result = credentials.execute(
                source_revision=SOURCE,
                output=output,
                report_path=report,
                environment=ENV,
                source_verifier=lambda _revision: None,
                supplier=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError(secret)
                ),
            )
            raw = report.read_text(encoding="utf-8")
            payload = json.loads(raw)

        self.assertEqual(result, 2)
        self.assertEqual(payload["status"], "OIDC_IDENTITY_UNAVAILABLE")
        self.assertEqual(payload["binding"]["state"], "REQUIRED_UNVERIFIED")
        self.assertEqual(payload["repository_mutation"], "NOT_ATTEMPTED")
        self.assertEqual(payload["provider_mutation"], "NOT_ATTEMPTED")
        self.assertFalse(payload["credential"]["logged"])
        self.assertFalse(payload["credential"]["persisted"])
        self.assertFalse(payload["credential"]["cached"])
        self.assertNotIn(secret, raw)
        self.assertFalse(output.exists())

    def test_success_receipt_contains_scope_but_not_credential(self):
        observed = credentials.ObserverResult(
            exit_code=2, status="PARTIAL_OR_UNAVAILABLE"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "credential.json"
            result = credentials.execute(
                source_revision=SOURCE,
                output=root / "observation.json",
                report_path=report,
                environment=ENV,
                source_verifier=lambda _revision: None,
                supplier=lambda *_args, **_kwargs: USER_TOKEN,
                observer=lambda *_args, **_kwargs: observed,
            )
            raw = report.read_text(encoding="utf-8")
            payload = json.loads(raw)

        self.assertEqual(result, 2)
        self.assertEqual(payload["status"], "CREDENTIAL_VALIDATED_OBSERVATION_PARTIAL")
        self.assertEqual(payload["binding"]["state"], "EXCHANGE_VALIDATED")
        self.assertEqual(payload["observation"]["status"], "PARTIAL_OR_UNAVAILABLE")
        self.assertEqual(payload["credential"]["provider_scope"], "gated-repos")
        self.assertTrue(payload["credential"]["read_only"])
        self.assertFalse(payload["credential"]["private_repositories"])
        self.assertFalse(payload["credential"]["write"])
        self.assertNotIn(USER_TOKEN, raw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
