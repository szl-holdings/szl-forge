from __future__ import annotations

import unittest
from dataclasses import replace
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

from huggingface_hub import constants

import hf_kernel_migration_guard as guard


class FakeApi:
    def __init__(self) -> None:
        self.sha = "a" * 40
        self.branches = {
            "main": "b" * 40,
            "v1": "c" * 40,
        }
        self.repo_types: list[str] = []

    def repo_info(self, repo_id: str, *, repo_type: str, **_: object) -> SimpleNamespace:
        self.repo_types.append(repo_type)
        if repo_id != "SZLHOLDINGS/szl-kernels":
            raise AssertionError(repo_id)
        return SimpleNamespace(sha=self.sha)

    def list_repo_refs(self, repo_id: str, *, repo_type: str, **_: object) -> SimpleNamespace:
        self.repo_types.append(repo_type)
        if repo_id != "SZLHOLDINGS/szl-kernels":
            raise AssertionError(repo_id)
        return SimpleNamespace(
            branches=[
                SimpleNamespace(name=name, target_commit=sha)
                for name, sha in self.branches.items()
            ]
        )


class KernelMigrationGuardTests(unittest.TestCase):
    def test_first_class_repo_type_is_used_everywhere(self) -> None:
        api = FakeApi()
        evidence = guard.verify_first_class_kernel(
            "SZLHOLDINGS/szl-kernels",
            api=api,
            client_version="0.16.1",
        )
        self.assertEqual(evidence.repo_type, "kernel")
        self.assertEqual(api.repo_types, ["kernel", "kernel"])
        self.assertEqual(evidence.branch_revisions["v1"], "c" * 40)

    def test_legacy_model_repo_type_fails_closed(self) -> None:
        with self.assertRaisesRegex(guard.KernelMigrationError, "legacy model-type"):
            guard.require_first_class_repo_type("model")

    def test_client_version_drift_fails_closed(self) -> None:
        with self.assertRaisesRegex(guard.KernelMigrationError, "client drifted"):
            guard.require_supported_client("0.16.0")

    def test_missing_v1_branch_is_rejected(self) -> None:
        api = FakeApi()
        del api.branches["v1"]
        with self.assertRaisesRegex(guard.KernelMigrationError, "branches missing"):
            guard.verify_first_class_kernel(
                "SZLHOLDINGS/szl-kernels",
                api=api,
                client_version="0.16.1",
            )

    def test_runtime_load_uses_exact_v1_commit(self) -> None:
        api = FakeApi()
        evidence = guard.verify_first_class_kernel(
            "SZLHOLDINGS/szl-kernels",
            api=api,
            client_version="0.16.1",
        )
        calls: list[tuple[str, dict[str, object]]] = []

        def get_kernel(repo_id: str, **kwargs: object) -> object:
            calls.append((repo_id, kwargs))
            return object()

        result = guard.verify_runtime_load(
            evidence,
            backend="cpu",
            get_kernel_fn=get_kernel,
        )
        self.assertFalse(result.runtime_loaded)
        self.assertIsNone(result.runtime_revision)
        self.assertIsNone(result.installed_client_version)
        self.assertEqual(result.runtime_evidence_kind, "INJECTED_TEST_DOUBLE")
        self.assertEqual(calls[0][0], "SZLHOLDINGS/szl-kernels")
        self.assertEqual(calls[0][1]["revision"], "c" * 40)
        self.assertEqual(calls[0][1]["backend"], "cpu")
        self.assertIs(calls[0][1]["trust_remote_code"], False)


class RuntimeEvidenceTests(unittest.TestCase):
    """All imports here are test doubles; none of these are live kernel results."""

    def setUp(self):
        self.evidence = guard.verify_first_class_kernel(
            "SZLHOLDINGS/szl-kernels", api=FakeApi(), client_version="0.16.1"
        )

    def test_declared_version_cannot_hide_missing_installation(self):
        with mock.patch.object(guard, "installed_kernels_version", side_effect=guard.KernelMigrationError("not installed")) as read:
            with self.assertRaisesRegex(guard.KernelMigrationError, "not installed"):
                guard.verify_runtime_load(self.evidence)
            read.assert_called_once_with()

    def test_declared_version_cannot_hide_wrong_installation(self):
        with mock.patch.object(guard, "installed_kernels_version", return_value="0.16.0") as read:
            with self.assertRaisesRegex(guard.KernelMigrationError, "client drifted"):
                guard.verify_runtime_load(self.evidence)
            read.assert_called_once_with()

    def test_bad_input_rejected_before_loader(self):
        for evidence in (
            replace(self.evidence, repo_type="model"),
            replace(self.evidence, repo_id="https://example.test/a/b"),
            replace(self.evidence, repo_id="SZLHOLDINGS/x\n"),
            replace(self.evidence, repo_revision="main"),
            replace(self.evidence, branch_revisions={}),
            replace(self.evidence, branch_revisions={"main": "a" * 40, "v1": "main"}),
            replace(self.evidence, branch_revisions={"main": "a" * 40, "v1": True}),
            replace(self.evidence, branch_revisions=[]),
            replace(self.evidence, client_version="0.16.0"),
        ):
            loader = mock.Mock()
            with self.subTest(evidence=evidence), self.assertRaises(guard.KernelMigrationError):
                guard.verify_runtime_load(evidence, get_kernel_fn=loader)
            loader.assert_not_called()

    def test_backend_is_not_coerced(self):
        for backend in (None, True, "", "auto", "cuda:0", [], {}):
            loader = mock.Mock()
            with self.subTest(backend=backend), self.assertRaises(guard.KernelMigrationError):
                guard.verify_runtime_load(self.evidence, backend=backend, get_kernel_fn=loader)
            loader.assert_not_called()

    def test_noncallable_double_rejected(self):
        with self.assertRaises(guard.KernelMigrationError):
            guard.verify_runtime_load(self.evidence, get_kernel_fn=False)

    def test_mutated_branch_rechecked_at_use(self):
        self.evidence.branch_revisions["v1"] = "main"
        loader = mock.Mock()
        with self.assertRaises(guard.KernelMigrationError):
            guard.verify_runtime_load(self.evidence, get_kernel_fn=loader)
        loader.assert_not_called()

    def test_double_mutation_does_not_rewrite_result_pin(self):
        def loader(*args, **kwargs):
            self.evidence.branch_revisions["v1"] = "d" * 40
            return object()
        result = guard.verify_runtime_load(self.evidence, get_kernel_fn=loader)
        self.assertEqual(result.branch_revisions["v1"], "c" * 40)
        self.assertFalse(result.runtime_loaded)

    def test_double_none_is_not_a_success(self):
        with self.assertRaisesRegex(guard.KernelMigrationError, "no module"):
            guard.verify_runtime_load(self.evidence, get_kernel_fn=lambda *a, **kw: None)

    def test_no_double_can_grant_quality_or_production(self):
        result = guard.verify_runtime_load(self.evidence, get_kernel_fn=lambda *a, **kw: object())
        self.assertFalse(result.production_authorization)
        self.assertFalse(result.numerical_equivalence_verified)
        self.assertIsNone(result.installed_client_version)

    def test_real_branch_protocol_with_simulated_loader(self):
        # This exercises real-branch control flow only, NOT the real package.
        module = ModuleType("szl_fixture")
        record = SimpleNamespace(module=module, repo_info=SimpleNamespace(
            repo_id=self.evidence.repo_id, revision="c" * 40))
        fake_package = ModuleType("kernels")
        fake_package.get_kernel = mock.Mock(return_value=module)
        fake_package.get_loaded_kernels = mock.Mock(return_value=[record])
        with mock.patch.object(guard, "installed_kernels_version", return_value="0.16.1"), \
             mock.patch.object(guard, "_require_runtime_environment"), \
             mock.patch.dict(sys.modules, {"kernels": fake_package}):
            result = guard.verify_runtime_load(self.evidence)
        self.assertTrue(result.runtime_loaded)
        self.assertEqual(result.runtime_evidence_kind, "LOADER_IMPORT_ONLY")
        self.assertEqual(result.installed_client_version, "0.16.1")
        self.assertFalse(result.numerical_equivalence_verified)
        self.assertFalse(result.production_authorization)
        fake_package.get_kernel.assert_called_once_with(
            self.evidence.repo_id, revision="c" * 40, backend="cpu", trust_remote_code=False)

    def test_wrong_origin_record_rejected_after_loader(self):
        module = ModuleType("szl_fixture")
        for records in ([], [SimpleNamespace(module=module, repo_info=None)],
                        [SimpleNamespace(module=module, repo_info=SimpleNamespace(
                            repo_id="OTHER/kernel", revision="c" * 40))]):
            fake = ModuleType("kernels")
            fake.get_kernel = mock.Mock(return_value=module)
            fake.get_loaded_kernels = mock.Mock(return_value=records)
            with mock.patch.object(guard, "installed_kernels_version", return_value="0.16.1"), \
                 mock.patch.object(guard, "_require_runtime_environment"), \
                 mock.patch.dict(sys.modules, {"kernels": fake}):
                with self.assertRaisesRegex(guard.KernelMigrationError, "was loaded"):
                    guard.verify_runtime_load(self.evidence)
            fake.get_kernel.assert_called_once()

    def test_trust_denial_is_not_retried_with_true(self):
        fake = ModuleType("kernels")
        fake.get_kernel = mock.Mock(side_effect=ValueError("publisher untrusted"))
        fake.get_loaded_kernels = mock.Mock()
        with mock.patch.object(guard, "installed_kernels_version", return_value="0.16.1"), \
             mock.patch.object(guard, "_require_runtime_environment"), \
             mock.patch.dict(sys.modules, {"kernels": fake}):
            with self.assertRaisesRegex(ValueError, "publisher untrusted"):
                guard.verify_runtime_load(self.evidence)
        fake.get_kernel.assert_called_once()
        self.assertIs(fake.get_kernel.call_args.kwargs["trust_remote_code"], False)
        fake.get_loaded_kernels.assert_not_called()

    def test_local_override_blocked(self):
        with mock.patch.dict(os.environ, {"LOCAL_KERNELS": "SZLHOLDINGS/szl-kernels=/tmp/fixture"}):
            with self.assertRaisesRegex(guard.KernelMigrationError, "LOCAL_KERNELS"):
                guard._require_runtime_environment()

    def test_offline_trust_bypass_blocked(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(constants, "HF_HUB_OFFLINE", True):
            with self.assertRaisesRegex(guard.KernelMigrationError, "offline"):
                guard._require_runtime_environment()

    def test_implicit_credentials_blocked(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(constants, "HF_HUB_OFFLINE", False), \
             mock.patch.object(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", False):
            with self.assertRaisesRegex(guard.KernelMigrationError, "IMPLICIT_TOKEN"):
                guard._require_runtime_environment()

    def test_noncanonical_endpoint_blocked(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(constants, "HF_HUB_OFFLINE", False), \
             mock.patch.object(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", True), \
             mock.patch.object(constants, "ENDPOINT", "https://example.test"):
            with self.assertRaisesRegex(guard.KernelMigrationError, "canonical"):
                guard._require_runtime_environment()

    def test_clean_environment_preflight_only(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(constants, "HF_HUB_OFFLINE", False), \
             mock.patch.object(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", True), \
             mock.patch.object(constants, "ENDPOINT", "https://huggingface.co"):
            guard._require_runtime_environment()


if __name__ == "__main__":
    unittest.main()
