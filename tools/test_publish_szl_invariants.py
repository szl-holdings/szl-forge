from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import publish_szl_invariants as publisher


class FakeApi:
    def __init__(self) -> None:
        self.heads = {"model": "1" * 40}
        self.kernel_branches = {"main": "2" * 40, "v1": "3" * 40}
        self.files = {"model": {}, "kernel": {}}
        self.commits: list[str] = []
        self.kernel_uploads = 0
        self.head_calls = {"model": 0}
        self.kernel_ref_calls = 0
        self.drift_type: str | None = None
        self.corrupt_path: str | None = None
        self.change_default_head = False

    def model_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        self.head_calls["model"] += 1
        if self.drift_type == "model" and self.head_calls["model"] >= 2:
            return SimpleNamespace(sha="9" * 40)
        return SimpleNamespace(sha=self.heads["model"])

    def repo_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        raise AssertionError("Kernel default head must not be used for v1 publication")

    def list_repo_refs(self, *args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        self.kernel_ref_calls += 1
        branches = dict(self.kernel_branches)
        if self.drift_type == "kernel" and self.kernel_ref_calls >= 2:
            branches["v1"] = "9" * 40
        return SimpleNamespace(
            branches=[
                SimpleNamespace(name=name, target_commit=revision)
                for name, revision in branches.items()
            ]
        )

    def list_repo_tree(self, *args: object, **kwargs: object) -> list[SimpleNamespace]:
        del args
        repo_type = str(kwargs["repo_type"])
        revision = str(kwargs["revision"])
        expected = (
            self.heads["model"]
            if repo_type == "model"
            else self.kernel_branches["v1"]
        )
        if revision != expected:
            raise AssertionError("unexpected immutable tree revision")
        return [SimpleNamespace(path=path) for path in self.files[repo_type]]

    def create_commit(self, **kwargs: object) -> SimpleNamespace:
        repo_type = str(kwargs["repo_type"])
        if repo_type != "model":
            raise AssertionError("Kernel publication must use kernel-builder")
        if kwargs["parent_commit"] != self.heads["model"]:
            raise AssertionError("unexpected parent commit")
        for operation in kwargs["operations"]:
            payload = Path(operation.path_or_fileobj).read_bytes()
            if self.corrupt_path == operation.path_in_repo:
                payload += b"corrupt"
            self.files["model"][operation.path_in_repo] = payload
        revision = "4" * 40
        self.heads["model"] = revision
        self.commits.append("model")
        return SimpleNamespace(oid=revision)

    def upload_kernel(self, staging_root: Path, token: str) -> None:
        if token != "token":
            raise AssertionError("missing fake token")
        self.kernel_uploads += 1
        for path in sorted((staging_root / "build").rglob("*")):
            if not path.is_file():
                continue
            repository_path = path.relative_to(staging_root).as_posix()
            payload = path.read_bytes()
            if self.corrupt_path == repository_path:
                payload += b"corrupt"
            self.files["kernel"][repository_path] = payload
        self.kernel_branches["v1"] = "5" * 40
        if self.change_default_head:
            self.kernel_branches["main"] = "6" * 40


class PublishInvariantsTests(unittest.TestCase):
    source_revision = "a" * 40
    publisher_revision = "b" * 40

    def _fixture(self, root: Path) -> tuple[Path, FakeApi]:
        artifacts = {
            "build/torch-universal/szl_invariants/__init__.py": (
                b'PROVENANCE = {"trained_weights_present": False}\n'
            ),
            "build/torch-universal/szl_invariants/metadata.json": (
                b'{"trained_weights_present": false}\n'
            ),
            "torch-ext/szl_invariants/__init__.py": (
                b'PROVENANCE = {"trained_weights_present": False}\n'
            ),
            "torch-ext/szl_invariants/metadata.json": (
                b'{"trained_weights_present": false}\n'
            ),
        }
        for relative, payload in artifacts.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        contract_path = root / publisher.CONTRACT_RELATIVE
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(
            json.dumps(
                {
                    "schema": "szl.invariants-source-binding/v1",
                    "repo_id": publisher.EXPECTED_REPO_ID,
                    "source_repository": publisher.EXPECTED_SOURCE_REPOSITORY,
                    "artifact_files": sorted(publisher.EXPECTED_ARTIFACT_FILES),
                    "expected_artifact_sha256": {
                        path: hashlib.sha256(payload).hexdigest()
                        for path, payload in artifacts.items()
                    },
                    "publication_targets": [
                        {
                            "repo_type": repo_type,
                            "source_path": source_path,
                            "path_in_repo": path_in_repo,
                        }
                        for repo_type, source_path, path_in_repo in sorted(
                            publisher.EXPECTED_TARGETS
                        )
                    ],
                    "claims": {"trained_weights_present": False},
                    "limitations": ["test fixture"],
                }
            ),
            encoding="utf-8",
        )
        authorization = root / "authorization.json"
        authorization.write_text(
            json.dumps(
                {
                    "schema": "szl.invariants-release-authorization/v1",
                    "status": "AUTHORIZED_PROTECTED_MAIN",
                    "source": {
                        "repository": publisher.EXPECTED_SOURCE_REPOSITORY,
                        "revision": self.source_revision,
                        "protected_main": self.source_revision,
                        "signature_verified": True,
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
        api = FakeApi()
        for repo_type, _source, destination in publisher.EXPECTED_TARGETS:
            api.files[repo_type][destination] = b"old"
        return authorization, api

    @staticmethod
    def _identity() -> dict[str, object]:
        return publisher.publisher_identity(
            repository=publisher.EXPECTED_PUBLISHER_REPOSITORY,
            revision="b" * 40,
            workflow_ref=(
                "szl-holdings/szl-forge/.github/workflows/"
                "publish-szl-invariants.yml@refs/heads/main"
            ),
            run_id="123",
            run_attempt="1",
        )

    @staticmethod
    def _download(api: FakeApi, root: Path):
        def download(
            repo_id: str,
            filename: str,
            *,
            repo_type: str,
            revision: str,
            token: str | None,
        ) -> str:
            del token
            if repo_id != publisher.EXPECTED_REPO_ID:
                raise AssertionError(repo_id)
            expected = (
                api.heads["model"]
                if repo_type == "model"
                else api.kernel_branches["v1"]
            )
            if revision != expected:
                raise AssertionError("unexpected immutable revision")
            target = root / "downloads" / repo_type / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(api.files[repo_type][filename])
            return str(target)

        return download

    def _run(
        self,
        root: Path,
        authorization: Path,
        api: FakeApi,
        *,
        publish: bool,
        token: str | None,
    ) -> dict[str, object]:
        return publisher.run(
            source_root=root,
            report_path=root / "report.json",
            authorization_path=authorization,
            source_revision=self.source_revision,
            publisher=self._identity(),
            publish=publish,
            token=token,
            api=api,
            download_fn=self._download(api, root),
            kernel_upload_fn=api.upload_kernel,
        )

    def test_dry_run_is_credentialless_and_non_mutating(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization, api = self._fixture(root)
            result = self._run(root, authorization, api, publish=False, token=None)
        self.assertEqual(result["status"], "VERIFIED_DRY_RUN")
        self.assertEqual(api.commits, [])
        self.assertEqual(api.kernel_uploads, 0)
        self.assertEqual(result["observed_before"]["kernel"]["branch"], "v1")
        self.assertEqual(
            result["observed_before"]["kernel"]["revision"],
            "3" * 40,
        )

    def test_publish_uses_exact_parents_and_verifies_readback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization, api = self._fixture(root)
            result = self._run(root, authorization, api, publish=True, token="token")
        self.assertEqual(result["status"], "PUBLISHED_AND_EXACT_READBACK_VERIFIED")
        self.assertEqual(api.kernel_uploads, 1)
        self.assertEqual(api.commits, ["model"])
        self.assertEqual(
            result["targets"]["kernel"]["status"],
            "V1_EXACT_READBACK_VERIFIED",
        )
        self.assertEqual(result["targets"]["kernel"]["branch"], "v1")
        self.assertEqual(
            result["targets"]["kernel"]["branches_after"]["main"],
            "2" * 40,
        )
        self.assertEqual(
            result["targets"]["model"]["status"], "EXACT_READBACK_VERIFIED"
        )
        self.assertEqual(
            api.files["kernel"][
                "build/torch-cpu/szl_invariants/metadata.json"
            ],
            b'{"trained_weights_present": false}\n',
        )

    def test_publish_requires_token_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization, api = self._fixture(root)
            with self.assertRaisesRegex(publisher.PublicationError, "HF_TOKEN"):
                self._run(root, authorization, api, publish=True, token=None)
        self.assertEqual(api.commits, [])
        self.assertEqual(api.kernel_uploads, 0)

    def test_local_hash_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization, api = self._fixture(root)
            (root / "torch-ext/szl_invariants/__init__.py").write_bytes(b"changed")
            with self.assertRaisesRegex(publisher.PublicationError, "drifted"):
                self._run(root, authorization, api, publish=False, token=None)

    def test_parent_drift_fails_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization, api = self._fixture(root)
            api.drift_type = "kernel"
            with self.assertRaisesRegex(publisher.PublicationError, "parent changed"):
                self._run(root, authorization, api, publish=True, token="token")
        self.assertEqual(api.commits, [])
        self.assertEqual(api.kernel_uploads, 0)

    def test_cpu_metadata_readback_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization, api = self._fixture(root)
            api.corrupt_path = "build/torch-cpu/szl_invariants/metadata.json"
            with self.assertRaisesRegex(publisher.PublicationError, "readback mismatch"):
                self._run(root, authorization, api, publish=True, token="token")
        self.assertEqual(api.commits, [])
        self.assertEqual(api.kernel_uploads, 1)

    def test_missing_cpu_metadata_is_recorded_then_repaired(self) -> None:
        destination = "build/torch-cpu/szl_invariants/metadata.json"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization, api = self._fixture(root)
            del api.files["kernel"][destination]
            result = self._run(root, authorization, api, publish=True, token="token")
        before = {
            item["path"]: item["status"]
            for item in result["observed_before"]["kernel"]["files"]
        }
        self.assertEqual(before[destination], "MISSING")
        self.assertEqual(
            api.files["kernel"][destination],
            b'{"trained_weights_present": false}\n',
        )

    def test_default_head_change_during_v1_upload_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization, api = self._fixture(root)
            api.change_default_head = True
            with self.assertRaisesRegex(
                publisher.PublicationError,
                "default head changed",
            ):
                self._run(root, authorization, api, publish=True, token="token")
        self.assertEqual(api.commits, [])

    def test_contract_rejects_added_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            contract = root / publisher.CONTRACT_RELATIVE
            payload = json.loads(contract.read_text(encoding="utf-8"))
            payload["artifact_files"].append("README.md")
            contract.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(publisher.PublicationError, "closed publication"):
                publisher.load_contract(root)

    def test_contract_rejects_target_remap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            contract = root / publisher.CONTRACT_RELATIVE
            payload = json.loads(contract.read_text(encoding="utf-8"))
            changed = copy.deepcopy(payload)
            changed["publication_targets"][0]["path_in_repo"] = "README.md"
            contract.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(publisher.PublicationError, "closed destination"):
                publisher.load_contract(root)

    def test_contract_rejects_missing_cpu_metadata_target(self) -> None:
        destination = "build/torch-cpu/szl_invariants/metadata.json"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            contract = root / publisher.CONTRACT_RELATIVE
            payload = json.loads(contract.read_text(encoding="utf-8"))
            payload["publication_targets"] = [
                target
                for target in payload["publication_targets"]
                if target["path_in_repo"] != destination
            ]
            contract.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(publisher.PublicationError, "closed destination"):
                publisher.load_contract(root)

    def test_staging_includes_cpu_metadata_and_no_default_head_card(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            contract = publisher.load_contract(root)
            staging = root / "staging"
            expected = publisher.stage_kernel_targets(root, contract, staging)
            cpu_metadata = "build/torch-cpu/szl_invariants/metadata.json"
            self.assertEqual(
                expected[cpu_metadata],
                b'{"trained_weights_present": false}\n',
            )
            self.assertIn("build/torch-cpu/metadata.json", expected)
            self.assertIn("build/torch-universal/metadata.json", expected)
            self.assertFalse((staging / "build" / "CARD.md").exists())

    def test_builder_requires_exact_official_version_identity(self) -> None:
        version = SimpleNamespace(
            returncode=0,
            stdout="hf-kernel-builder 0.17.0-dev0\n",
            stderr="",
        )
        with patch.object(
            publisher.shutil,
            "which",
            return_value="/trusted/kernel-builder",
        ), patch.object(publisher.subprocess, "run", return_value=version):
            executable = publisher.require_kernel_builder_executable()
        self.assertEqual(executable, "/trusted/kernel-builder")

    def test_builder_upload_targets_kernel_v1(self) -> None:
        commands: list[list[str]] = []

        def run(command: list[str], **kwargs: object) -> SimpleNamespace:
            del kwargs
            commands.append(command)
            output = Path(command[command.index("--output-json") + 1])
            output.write_text(
                json.dumps(
                    {
                        "status": "uploaded",
                        "repo_id": publisher.EXPECTED_REPO_ID,
                        "branch": "v1",
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temporary, patch.object(
            publisher,
            "require_kernel_builder_executable",
            return_value="/trusted/kernel-builder",
        ), patch.object(publisher.subprocess, "run", side_effect=run):
            publisher.upload_first_class_kernel(Path(temporary), "token")

        self.assertEqual(commands[0][0], "/trusted/kernel-builder")
        self.assertEqual(
            commands[0][commands[0].index("--branch") + 1],
            "v1",
        )
        self.assertEqual(
            commands[0][commands[0].index("--repo-type") + 1],
            "kernel",
        )

    def test_workflow_keeps_secret_after_authorization(self) -> None:
        workflow = (
            Path(__file__).parents[1]
            / ".github"
            / "workflows"
            / "publish-szl-invariants.yml"
        ).read_text(encoding="utf-8")
        authorize = workflow.index("Authorize protected source without publisher secret")
        publish = workflow.index("Publish with trusted Forge code and verify exact readback")
        secret = workflow.index("secrets.HF_ORG_TOKEN")
        self.assertLess(authorize, publish)
        self.assertLess(publish, secret)
        self.assertIn("group: publish-szl-invariants", workflow)
        self.assertIn("repository: szl-holdings/szl-invariants", workflow)
        self.assertIn(
            "--rev 633246310320d85def0c67d62c7912fd444a842f",
            workflow,
        )
        self.assertIn(
            'test "$(kernel-builder --version)" = '
            '"hf-kernel-builder 0.17.0-dev0"',
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
