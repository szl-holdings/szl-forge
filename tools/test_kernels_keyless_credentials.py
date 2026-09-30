"""Offline adversarial tests for the closed two-resource OIDC authority."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import httpx
from huggingface_hub.errors import HfHubHTTPError

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


def valid_api() -> mock.Mock:
    api = mock.Mock()
    api.list_repo_refs.return_value = SimpleNamespace(branches=[
        SimpleNamespace(name="main", target_commit="c" * 40),
        SimpleNamespace(name="v1", target_commit="d" * 40),
    ])
    return api


class KernelKeylessCredentialTests(unittest.TestCase):
    def test_exact_resources_use_distinct_typed_grants(self) -> None:
        api = valid_api()
        supplier = mock.Mock(side_effect=[MODEL, KERNEL])
        pair = credentials.acquire_pair(environment=ENV, supplier=supplier, api=api)
        self.assertEqual(
            [call.args[0] for call in supplier.call_args_list],
            [credentials.MODEL_RESOURCE, credentials.KERNEL_RESOURCE],
        )
        self.assertEqual(pair.model, MODEL)
        self.assertEqual(pair.kernel, KERNEL)
        self.assertNotIn(MODEL, repr(pair))
        self.assertNotIn(KERNEL, repr(pair))
        api.auth_check.assert_called_once_with(
            repo_id=credentials.TARGET,
            repo_type="model",
            token=MODEL,
            write=True,
        )
        api.list_repo_refs.assert_called_once_with(
            credentials.TARGET, repo_type="kernel", token=KERNEL
        )

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
                        environment=environment, supplier=supplier, api=mock.Mock()
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
                        api=valid_api(),
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
        api = valid_api()
        with self.assertRaisesRegex(
            credentials.KeylessCredentialError, "CROSS_TARGET_TOKEN_REUSE_REJECTED"
        ):
            credentials.acquire_pair(
                environment=ENV,
                supplier=mock.Mock(side_effect=[MODEL, MODEL]),
                api=api,
            )
        api.auth_check.assert_not_called()
        api.list_repo_refs.assert_not_called()

    def test_incomplete_or_nonimmutable_kernel_refs_refuse_the_pair(self) -> None:
        cases = (
            [],
            [SimpleNamespace(name="main", target_commit="c" * 40)],
            [
                SimpleNamespace(name="main", target_commit="c" * 40),
                SimpleNamespace(name="v1", target_commit="main"),
            ],
        )
        for branches in cases:
            api = valid_api()
            api.list_repo_refs.return_value = SimpleNamespace(branches=branches)
            with self.subTest(branches=branches), self.assertRaisesRegex(
                credentials.KeylessCredentialError, "INVALID_KERNEL_REFS"
            ):
                credentials.acquire_pair(
                    environment=ENV,
                    supplier=mock.Mock(side_effect=[MODEL, KERNEL]),
                    api=api,
                )

    def test_model_and_kernel_access_errors_have_distinct_fixed_codes(self) -> None:
        for target, method, code in (
            ("model", "auth_check", "MODEL_WRITE_ACCESS_VALIDATION_FAILED"),
            ("kernel", "list_repo_refs", "KERNEL_REFS_VALIDATION_FAILED"),
        ):
            for status in (401, 403, 404, None):
                with self.subTest(target=target, status=status):
                    api = valid_api()
                    if status is None:
                        error = RuntimeError("private-response-" + MODEL + KERNEL)
                    else:
                        request = httpx.Request(
                            "GET", "https://huggingface.co/api/private",
                            headers={"Authorization": "Bearer " + MODEL + KERNEL},
                        )
                        error = httpx.HTTPStatusError(
                            "private-response-" + MODEL + KERNEL,
                            request=request,
                            response=httpx.Response(status, request=request),
                        )
                    getattr(api, method).side_effect = error
                    with self.assertRaises(credentials.KeylessCredentialError) as caught:
                        credentials.acquire_pair(
                            environment=ENV,
                            supplier=mock.Mock(side_effect=[MODEL, KERNEL]),
                            api=api,
                        )
                    self.assertEqual(caught.exception.args, (code,))
                    self.assertTrue(caught.exception.__suppress_context__)
                    self.assertNotIn("private-response", repr(caught.exception))
                    self.assertNotIn(MODEL, repr(caught.exception))
                    self.assertNotIn(KERNEL, repr(caught.exception))
                    if target == "model":
                        api.list_repo_refs.assert_not_called()
                    else:
                        api.auth_check.assert_called_once()

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

    def test_access_errors_retain_only_a_typed_bounded_http_status(self) -> None:
        for method, code in (
            ("auth_check", "MODEL_WRITE_ACCESS_VALIDATION_FAILED"),
            ("list_repo_refs", "KERNEL_REFS_VALIDATION_FAILED"),
        ):
            for status in (400, 401, 403, 404, 429, 500, 599, True, "403", 403.0, 399, 600):
                with self.subTest(method=method, status=status):
                    response = httpx.Response(
                        403,
                        request=httpx.Request("GET", "https://huggingface.co/private", headers={
                            "Authorization": "Bearer " + MODEL,
                        }),
                        content=("private-body-" + KERNEL).encode(),
                        headers={"x-request-id": "private-request-id"},
                    )
                    exc = HfHubHTTPError("private-message-" + MODEL, response=response)
                    response.status_code = status
                    api = valid_api()
                    getattr(api, method).side_effect = exc
                    with self.assertRaises(credentials.KeylessCredentialError) as caught:
                        credentials.acquire_pair(
                            environment=ENV,
                            supplier=mock.Mock(side_effect=[MODEL, KERNEL]),
                            api=api,
                        )
                    self.assertEqual(caught.exception.args, (code,))
                    expected = status if type(status) is int and 400 <= status <= 599 else None
                    self.assertEqual(caught.exception.http_status, expected)
                    serialized = repr(caught.exception) + repr(vars(caught.exception))
                    for secret in (MODEL, KERNEL, "private-body", "private-message", "private-request-id"):
                        self.assertNotIn(secret, serialized)
                    if method == "auth_check":
                        api.list_repo_refs.assert_not_called()

    def test_access_errors_do_not_trust_generic_or_custom_response_objects(self) -> None:
        class CustomResponse(httpx.Response):
            pass

        class ResponsePropertyError(RuntimeError):
            @property
            def response(self) -> object:
                raise AssertionError("untrusted response property must not be read")

        response = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/private"))
        generic = RuntimeError(MODEL)
        generic.response = response
        fake_response = HfHubHTTPError(MODEL, response=response)
        fake_response.response = SimpleNamespace(status_code=403)
        custom_response = HfHubHTTPError(MODEL, response=CustomResponse(
            403, request=httpx.Request("GET", "https://huggingface.co/private"),
        ))
        for exc in (generic, fake_response, custom_response, ResponsePropertyError(MODEL)):
            with self.subTest(error_type=type(exc).__name__):
                api = valid_api()
                api.auth_check.side_effect = exc
                with self.assertRaises(credentials.KeylessCredentialError) as caught:
                    credentials.acquire_pair(
                        environment=ENV,
                        supplier=mock.Mock(side_effect=[MODEL, KERNEL]),
                        api=api,
                    )
                self.assertEqual(caught.exception.args, ("MODEL_WRITE_ACCESS_VALIDATION_FAILED",))
                self.assertIsNone(caught.exception.http_status)


if __name__ == "__main__":
    unittest.main()

