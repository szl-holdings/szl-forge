"""Offline adversarial tests for the closed two-resource OIDC authority."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import kernels_keyless_credentials as credentials


MODEL = "hf_jwt_model_fixture"
KERNEL = "hf_jwt_kernel_fixture"
ENV = {
    "GITHUB_ACTIONS": "true",
    "GITHUB_REPOSITORY": credentials.REPOSITORY,
    "GITHUB_REF": "refs/heads/main",
    "GITHUB_REF_PROTECTED": "true",
    "GITHUB_WORKFLOW_REF": credentials.WORKFLOW_REF,
    "GITHUB_EVENT_NAME": "workflow_dispatch",
    "GITHUB_SHA": "b" * 40,
    "ACTIONS_ID_TOKEN_REQUEST_URL": (
        "https://pipelines.actions.githubusercontent.com/oidc/token"
    ),
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "github-oidc-request-fixture",
}


class KernelKeylessCredentialTests(unittest.TestCase):
    def test_exact_resources_use_distinct_typed_grants(self) -> None:
        supplier = mock.Mock(side_effect=[MODEL, KERNEL])
        with mock.patch("huggingface_hub.HfApi") as api:
            pair = credentials.acquire_pair(environment=ENV, supplier=supplier)
        self.assertEqual(
            [call.args[0] for call in supplier.call_args_list],
            [credentials.MODEL_RESOURCE, credentials.KERNEL_RESOURCE],
        )
        self.assertEqual(pair.model, MODEL)
        self.assertEqual(pair.kernel, KERNEL)
        self.assertNotIn(MODEL, repr(pair))
        self.assertNotIn(KERNEL, repr(pair))
        api.assert_not_called()

    def test_fork_pr_mutable_ref_and_other_workflow_never_exchange(self) -> None:
        mutations = {
            "GITHUB_REPOSITORY": "attacker/fork",
            "GITHUB_REF": "refs/pull/7/merge",
            "GITHUB_REF_PROTECTED": "false",
            "GITHUB_WORKFLOW_REF": credentials.WORKFLOW_REF.replace(
                credentials.WORKFLOW_FILENAME, "attacker.yml"
            ),
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_SHA": "main",
            "GITHUB_HEAD_REF": "attacker-branch",
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                environment = {**ENV, key: value}
                supplier = mock.Mock()
                with self.assertRaises(credentials.KeylessCredentialError):
                    credentials.acquire_pair(
                        environment=environment, supplier=supplier
                    )
                supplier.assert_not_called()

    def test_only_official_github_oidc_transport_is_admitted(self) -> None:
        for url in (
            "http://pipelines.actions.githubusercontent.com/oidc/token",
            "https://actions.githubusercontent.com.evil.invalid/token",
            "https://example.invalid/token",
            "not-a-url",
            "https://pipelines.actions.githubusercontent.com",
        ):
            with self.subTest(url=url), self.assertRaisesRegex(
                credentials.KeylessCredentialError, "INVALID_GITHUB_OIDC_ENDPOINT"
            ):
                credentials.require_workflow_context({
                    **ENV, "ACTIONS_ID_TOKEN_REQUEST_URL": url
                })
        with self.assertRaisesRegex(
            credentials.KeylessCredentialError,
            "MISSING_GITHUB_OIDC_REQUEST_TOKEN",
        ):
            credentials.require_workflow_context({
                **ENV, "ACTIONS_ID_TOKEN_REQUEST_TOKEN": ""
            })

    def test_exchange_uses_empty_home_and_drops_all_ambient_credentials(self) -> None:
        environment = {
            **ENV,
            "PATH": os.environ.get("PATH", ""),
            "HF_TOKEN": MODEL,
            "HF_ORG_TOKEN": MODEL,
            "HF_OIDC_ID_TOKEN": "stale-subject-token",
            "HF_HOME": "/cached",
            "HUGGING_FACE_HUB_TOKEN": MODEL,
            "GITHUB_TOKEN": "github-secret",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
        }
        with mock.patch.object(
            credentials.subprocess,
            "run",
            return_value=SimpleNamespace(
                returncode=0, stdout=KERNEL + "\n", stderr=""
            ),
        ) as run:
            observed = credentials.exchange(credentials.KERNEL_RESOURCE, environment)
        self.assertEqual(observed, KERNEL)
        child = run.call_args.kwargs["env"]
        for key in (
            "HF_TOKEN", "HF_ORG_TOKEN", "HF_OIDC_ID_TOKEN",
            "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY",
        ):
            self.assertNotIn(key, child)
        self.assertEqual(child["HF_OIDC_RESOURCE"], credentials.KERNEL_RESOURCE)
        self.assertEqual(child["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")
        self.assertEqual(child["GITHUB_ACTIONS"], "true")
        from huggingface_hub._oidc import Provider, detect_provider

        with mock.patch.dict(os.environ, child, clear=True):
            self.assertEqual(detect_provider(), Provider.GITHUB)
        self.assertFalse(Path(child["HF_HOME"]).exists())
        self.assertEqual(run.call_args.args[0], ["hf", "auth", "token"])
        self.assertEqual(run.call_args.kwargs["timeout"], 60)

    def test_pair_acquisition_rejects_ambient_long_lived_credentials(self) -> None:
        for key in credentials.FORBIDDEN_AMBIENT_CREDENTIALS:
            with self.subTest(key=key):
                supplier = mock.Mock()
                with self.assertRaisesRegex(
                    credentials.KeylessCredentialError,
                    "AMBIENT_HUB_CREDENTIAL_REJECTED",
                ):
                    credentials.acquire_pair(
                        environment={**ENV, key: "secret-fixture"},
                        supplier=supplier,
                    )
                supplier.assert_not_called()

    def test_exchange_never_surfaces_provider_output_or_accepts_mixed_stdout(self) -> None:
        for output in ("", MODEL + "\n" + KERNEL, "warning\n" + MODEL, "hf_bad"):
            with self.subTest(output=output), mock.patch.object(
                credentials.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout=output, stderr=""),
            ):
                with self.assertRaises(credentials.KeylessCredentialError) as caught:
                    credentials.exchange(credentials.MODEL_RESOURCE, ENV)
                self.assertNotIn(MODEL, str(caught.exception))
                self.assertNotIn(KERNEL, str(caught.exception))
        with mock.patch.object(
            credentials.subprocess,
            "run",
            return_value=SimpleNamespace(
                returncode=1, stdout=MODEL, stderr="provider-" + KERNEL
            ),
        ):
            with self.assertRaisesRegex(
                credentials.KeylessCredentialError, "^OIDC_EXCHANGE_REJECTED$"
            ):
                credentials.exchange(credentials.MODEL_RESOURCE, ENV)

    def test_undeclared_resource_is_rejected_before_cli(self) -> None:
        with mock.patch.object(credentials.subprocess, "run") as run:
            with self.assertRaisesRegex(
                credentials.KeylessCredentialError, "UNDECLARED_OIDC_RESOURCE"
            ):
                credentials.exchange("SZLHOLDINGS/another-repo", ENV)
            run.assert_not_called()

    def test_same_token_cannot_cross_the_model_kernel_boundary(self) -> None:
        with self.assertRaisesRegex(
            credentials.KeylessCredentialError, "CROSS_TARGET_TOKEN_REUSE_REJECTED"
        ):
            credentials.acquire_pair(
                environment=ENV,
                supplier=mock.Mock(side_effect=[MODEL, MODEL]),
            )

    def test_both_exact_exchanges_must_succeed_before_pair_return(self) -> None:
        supplier = mock.Mock(side_effect=[
            MODEL,
            credentials.KeylessCredentialError("OIDC_EXCHANGE_REJECTED"),
        ])
        with self.assertRaisesRegex(
            credentials.KeylessCredentialError, "^OIDC_EXCHANGE_REJECTED$"
        ):
            credentials.acquire_pair(environment=ENV, supplier=supplier)
        self.assertEqual(
            [call.args[0] for call in supplier.call_args_list],
            [credentials.MODEL_RESOURCE, credentials.KERNEL_RESOURCE],
        )

    def test_repo_scoped_grants_do_not_call_user_token_auth_check(self) -> None:
        with mock.patch("huggingface_hub.HfApi") as api:
            api.return_value.auth_check.side_effect = RuntimeError(
                "the user-token endpoint must remain outside the grant boundary"
            )
            pair = credentials.acquire_pair(
                environment=ENV,
                supplier=mock.Mock(side_effect=[MODEL, KERNEL]),
            )
        self.assertEqual((pair.model, pair.kernel), (MODEL, KERNEL))
        api.assert_not_called()

    def test_authority_evidence_is_closed_and_contains_no_credential(self) -> None:
        evidence = credentials.authority_evidence(ENV)
        self.assertEqual(evidence["issuer"], credentials.OIDC_ISSUER)
        self.assertEqual(evidence["audience"], credentials.OIDC_AUDIENCE)
        self.assertEqual(evidence["resources"], {
            "legacy_model": credentials.MODEL_RESOURCE,
            "first_class_kernel": credentials.KERNEL_RESOURCE,
        })
        self.assertEqual(evidence["publisher_revision"], ENV["GITHUB_SHA"])
        self.assertTrue(evidence["distinct_target_credentials"])
        self.assertFalse(evidence["persistent_hub_secret_used"])
        serialized = repr(evidence)
        self.assertNotIn(MODEL, serialized)
        self.assertNotIn(KERNEL, serialized)


if __name__ == "__main__":
    unittest.main()

