from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import publish_szl_invariants as publisher


class FakeApi:
    def __init__(self) -> None:
        self.heads = {"model": "1" * 40, "kernel": "2" * 40}
        self.files = {"model": {}, "kernel": {}}
        self.commits: list[str] = []
        self.head_calls = {"model": 0, "kernel": 0}
        self.drift_type: str | None = None
        self.corrupt_readback = False

    def _info(self, repo_type: str) -> SimpleNamespace:
        self.head_calls[repo_type] += 1
        if self.drift_type == repo_type and self.head_calls[repo_type] >= 2:
            return SimpleNamespace(sha="9" * 40)
        return SimpleNamespace(sha=self.heads[repo_type])

    def model_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return self._info("model")

    def repo_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return self._info("kernel")

    def create_commit(self, **kwargs: object) -> SimpleNamespace:
        repo_type = str(kwargs["repo_type"])
        if kwargs["parent_commit"] != self.heads[repo_type]:
            raise AssertionError("unexpected parent commit")
        for operation in kwargs["operations"]:
            payload = Path(operation.path_or_fileobj).read_bytes()
            if self.corrupt_readback and not self.commits:
                payload += b"corrupt"
            self.files[repo_type][operation.path_in_repo] = payload
        revision = ("3" if repo_type == "kernel" else "4") * 40
        self.heads[repo_type] = revision
        self.commits.append(repo_type)
        return SimpleNamespace(oid=revision)


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
            if revision != api.heads[repo_type]:
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
        )

    def test_dry_run_is_credentialless_and_non_mutating(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization, api = self._fixture(root)
            result = self._run(root, authorization, api, publish=False, token=None)
        self.assertEqual(result["status"], "VERIFIED_DRY_RUN")
        self.assertEqual(api.commits, [])

    def test_publish_uses_exact_parents_and_verifies_readback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization, api = self._fixture(root)
            result = self._run(root, authorization, api, publish=True, token="token")
        self.assertEqual(result["status"], "PUBLISHED_AND_EXACT_READBACK_VERIFIED")
        self.assertEqual(api.commits, ["kernel", "model"])
        self.assertEqual(
            result["targets"]["kernel"]["status"], "EXACT_READBACK_VERIFIED"
        )
        self.assertEqual(
            result["targets"]["model"]["status"], "EXACT_READBACK_VERIFIED"
        )

    def test_publish_requires_token_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization, api = self._fixture(root)
            with self.assertRaisesRegex(publisher.PublicationError, "HF_TOKEN"):
                self._run(root, authorization, api, publish=True, token=None)
        self.assertEqual(api.commits, [])

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

    def test_readback_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization, api = self._fixture(root)
            api.corrupt_readback = True
            with self.assertRaisesRegex(publisher.PublicationError, "readback mismatch"):
                self._run(root, authorization, api, publish=True, token="token")

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


if __name__ == "__main__":
    unittest.main()
