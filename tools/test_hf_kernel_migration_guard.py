from __future__ import annotations

import unittest
from types import SimpleNamespace

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
        self.assertTrue(result.runtime_loaded)
        self.assertEqual(result.runtime_revision, "c" * 40)
        self.assertEqual(calls[0][0], "SZLHOLDINGS/szl-kernels")
        self.assertEqual(calls[0][1]["revision"], "c" * 40)
        self.assertEqual(calls[0][1]["backend"], "cpu")
        self.assertIs(calls[0][1]["trust_remote_code"], True)


if __name__ == "__main__":
    unittest.main()
