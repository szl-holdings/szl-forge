from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import publish_szl_kernels as publisher


class FakeApi:
    model_revision = "d" * 40
    kernel_revision = "e" * 40

    def __init__(self, artifacts: dict[str, Path]) -> None:
        self.files = list(artifacts)
        self.commits: list[dict[str, object]] = []
        self.remote: dict[tuple[str, str], dict[str, bytes]] = {
            (publisher.LEGACY_REPO_TYPE, self.model_revision): {
                relative: path.read_bytes() for relative, path in artifacts.items()
            },
            (publisher.KERNEL_REPO_TYPE, self.kernel_revision): {
                kernel_path: b"previous"
                for kernel_path in publisher.FIRST_CLASS_KERNEL_FILES.values()
            },
        }

    def model_info(
        self,
        repo_id: str,
        files_metadata: bool = False,
        token: str | None = None,
    ) -> SimpleNamespace:
        del repo_id, files_metadata, token
        return SimpleNamespace(
            sha=self.model_revision,
            siblings=[SimpleNamespace(rfilename=path) for path in self.files],
        )

    def repo_info(
        self,
        repo_id: str,
        *,
        repo_type: str,
        revision: str | None = None,
        **_: object,
    ) -> SimpleNamespace:
        del repo_id, revision
        self._assert_kernel(repo_type)
        return SimpleNamespace(sha=self.kernel_revision)

    def list_repo_refs(
        self,
        repo_id: str,
        *,
        repo_type: str,
        **_: object,
    ) -> SimpleNamespace:
        del repo_id
        self._assert_kernel(repo_type)
        return SimpleNamespace(
            branches=[
                SimpleNamespace(name=branch, target_commit=self.kernel_revision)
                for branch in publisher.KERNEL_BRANCHES
            ]
        )

    def list_repo_tree(
        self,
        repo_id: str,
        *,
        repo_type: str,
        **_: object,
    ) -> list[SimpleNamespace]:
        del repo_id
        self._assert_kernel(repo_type)
        return [
            SimpleNamespace(path=path)
            for path in publisher.FIRST_CLASS_KERNEL_FILES.values()
        ]

    def create_commit(
        self,
        repo_id: str,
        operations: list[object],
        *,
        repo_type: str,
        revision: str | None = None,
        parent_commit: str,
        **_: object,
    ) -> SimpleNamespace:
        del repo_id
        self.assert_parent(repo_type, parent_commit)
        self.commits.append({"repo_type": repo_type, "revision": revision})
        oid = f"{len(self.commits)}" * 40
        remote = dict(self.remote[(repo_type, parent_commit)])
        for operation in operations:
            source = operation.path_or_fileobj
            if isinstance(source, (str, Path)):
                payload = Path(source).read_bytes()
            elif isinstance(source, io.BytesIO):
                payload = source.getvalue()
            else:  # pragma: no cover - the publisher only uses paths and BytesIO
                raise AssertionError(type(source))
            remote[operation.path_in_repo] = payload
        self.remote[(repo_type, oid)] = remote
        return SimpleNamespace(oid=oid)

    def download(
        self,
        repo_id: str,
        filename: str,
        *,
        repo_type: str,
        revision: str,
        **_: object,
    ) -> str:
        if repo_id != publisher.EXPECTED_REPO_ID:
            raise AssertionError(repo_id)
        payload = self.remote[(repo_type, revision)][filename]
        destination = self.download_root / repo_type / revision / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return str(destination)

    def assert_parent(self, repo_type: str, parent_commit: str) -> None:
        expected = (
            self.kernel_revision
            if repo_type == publisher.KERNEL_REPO_TYPE
            else self.model_revision
        )
        if parent_commit != expected:
            raise AssertionError((repo_type, parent_commit, expected))

    @staticmethod
    def _assert_kernel(repo_type: str) -> None:
        if repo_type != publisher.KERNEL_REPO_TYPE:
            raise AssertionError(repo_type)


class PublishSzlKernelsTests(unittest.TestCase):
    def test_gateway_installs_hub_client_before_authorization_tests(self) -> None:
        workflow = (
            Path(__file__).parents[1]
            / ".github"
            / "workflows"
            / "publish-szl-kernels.yml"
        ).read_text(encoding="utf-8")
        install = workflow.index("Install trusted gateway test dependency")
        tests = workflow.index("Test trusted gateway contracts")
        dependency = workflow.index('"huggingface-hub==1.26.0"', install)
        self.assertLess(install, dependency)
        self.assertLess(dependency, tests)

    source_revision = "a" * 40
    publisher_revision = "b" * 40

    def _fixture(self, root: Path) -> tuple[Path, dict[str, Path]]:
        (root / "publishing").mkdir()
        artifacts = {
            ".gitattributes": root / ".gitattributes",
            "LICENSE": root / "LICENSE",
            "README.md": root / "README.md",
            "build/torch-universal/szl_kernels/__init__.py": (
                root / "build/torch-universal/szl_kernels/__init__.py"
            ),
            "build/torch-universal/szl_kernels/_chain.py": (
                root / "build/torch-universal/szl_kernels/_chain.py"
            ),
            "build/torch-universal/szl_kernels/_ops.py": (
                root / "build/torch-universal/szl_kernels/_ops.py"
            ),
            "build/torch-universal/szl_kernels/metadata.json": (
                root / "build/torch-universal/szl_kernels/metadata.json"
            ),
            "vectors.npz": root / "vectors.npz",
        }
        for path in artifacts.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        artifacts[".gitattributes"].write_text("*.bin lfs\n", encoding="utf-8")
        artifacts["LICENSE"].write_text("Apache-2.0\n", encoding="utf-8")
        artifacts["README.md"].write_text("kernel\n", encoding="utf-8")
        artifacts["build/torch-universal/szl_kernels/__init__.py"].write_text(
            '__version__ = "0.1.1"\n', encoding="utf-8"
        )
        artifacts["build/torch-universal/szl_kernels/_chain.py"].write_text(
            "GENESIS = '0' * 64\n", encoding="utf-8"
        )
        artifacts["build/torch-universal/szl_kernels/_ops.py"].write_text(
            "def op(): return True\n", encoding="utf-8"
        )
        artifacts["build/torch-universal/szl_kernels/metadata.json"].write_text(
            json.dumps({"name": "szl_kernels", "version": "0.1.1"}),
            encoding="utf-8",
        )
        artifacts["vectors.npz"].write_bytes(b"weights")
        contract = {
            "schema": "szl.kernel-source-binding/v1",
            "repo_id": publisher.EXPECTED_REPO_ID,
            "source_repository": publisher.EXPECTED_SOURCE_REPOSITORY,
            "artifact_files": list(artifacts),
            "expected_artifact_sha256": {
                "vectors.npz": hashlib.sha256(b"weights").hexdigest()
            },
            "claims": {"scope": "test"},
            "limitations": ["test fixture"],
        }
        (root / publisher.CONTRACT_RELATIVE).write_text(
            json.dumps(contract),
            encoding="utf-8",
        )
        authorization = root / "authorization.json"
        authorization.write_text(
            json.dumps(
                {
                    "schema": "szl.kernels-release-authorization/v1",
                    "status": "AUTHORIZED_PROTECTED_MAIN",
                    "source": {
                        "repository": publisher.EXPECTED_SOURCE_REPOSITORY,
                        "revision": self.source_revision,
                        "protected_main": self.source_revision,
                        "signature_verified": True,
                        "checks": [],
                    },
                    "publisher": {
                        "repository": publisher.EXPECTED_PUBLISHER_REPOSITORY,
                        "revision": self.publisher_revision,
                        "protected_main": self.publisher_revision,
                    },
                }
            ),
            encoding="utf-8",
        )
        return authorization, artifacts

    def test_dry_run_uses_authorized_data_and_immutable_publisher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization, artifacts = self._fixture(root)

            api = FakeApi(artifacts)
            api.download_root = root / "downloads"

            identity = publisher.publisher_identity(
                repository=publisher.EXPECTED_PUBLISHER_REPOSITORY,
                revision=self.publisher_revision,
                workflow_ref=(
                    f"{publisher.EXPECTED_PUBLISHER_REPOSITORY}/"
                    ".github/workflows/publish-szl-kernels.yml@refs/heads/main"
                ),
                run_id="123",
                run_attempt="1",
            )
            result = publisher.run(
                source_root=root,
                report_path=root / "report.json",
                authorization_path=authorization,
                source_revision=self.source_revision,
                publisher=identity,
                publish=False,
                token=None,
                api=api,
                download_fn=api.download,
            )
            self.assertEqual(result["status"], "VERIFIED_DRY_RUN")
            self.assertEqual(
                result["targets"]["first_class_kernel"]["mapped_file_count"],
                len(publisher.FIRST_CLASS_KERNEL_FILES),
            )
            self.assertIn(self.publisher_revision, identity["workflow_url"])

    def test_publish_updates_kernel_main_and_v1_then_legacy_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization, artifacts = self._fixture(root)
            api = FakeApi(artifacts)
            api.download_root = root / "downloads"
            identity = publisher.publisher_identity(
                repository=publisher.EXPECTED_PUBLISHER_REPOSITORY,
                revision=self.publisher_revision,
                workflow_ref=(
                    f"{publisher.EXPECTED_PUBLISHER_REPOSITORY}/"
                    ".github/workflows/publish-szl-kernels.yml@refs/heads/main"
                ),
                run_id="123",
                run_attempt="1",
            )
            result = publisher.run(
                source_root=root,
                report_path=root / "report.json",
                authorization_path=authorization,
                source_revision=self.source_revision,
                publisher=identity,
                publish=True,
                token="test-token",
                api=api,
                download_fn=api.download,
            )
            self.assertEqual(
                result["status"], "PUBLISHED_AND_EXACT_READBACK_VERIFIED"
            )
            self.assertEqual(
                api.commits,
                [
                    {"repo_type": "kernel", "revision": "main"},
                    {"repo_type": "kernel", "revision": "v1"},
                    {"repo_type": "model", "revision": None},
                ],
            )
            branches_after = result["targets"]["first_class_kernel"][
                "branches_after"
            ]
            self.assertEqual(set(branches_after), set(publisher.KERNEL_BRANCHES))

    def test_contract_cannot_redirect_publisher_to_another_repo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            contract_path = root / publisher.CONTRACT_RELATIVE
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["repo_id"] = "attacker/target"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(
                publisher.PublicationError,
                "another Hub repository",
            ):
                publisher.load_contract(root)


if __name__ == "__main__":
    unittest.main()
