from __future__ import annotations

import subprocess
import unittest

import httpx
from huggingface_hub.utils import RepositoryNotFoundError
from types import SimpleNamespace
from unittest.mock import Mock, patch

import publish_hf_space as publisher


REVISION = "a" * 40


class SpacePublicationPlanTests(unittest.TestCase):
    def _tracked_files(self, *names: str) -> bytes:
        return b"\0".join(name.encode("utf-8") for name in names) + b"\0"

    def test_docker_plan_requires_dockerfile(self) -> None:
        source = publisher.ROOT / "spaces" / "szl-model-inference-lab"
        output = self._tracked_files(
            "spaces/szl-model-inference-lab/README.md",
        )
        result = subprocess.CompletedProcess([], 0, stdout=output, stderr=b"")
        with patch("publish_hf_space.subprocess.run", return_value=result):
            with self.assertRaisesRegex(
                publisher.PublishError, "Dockerfile"
            ):
                publisher.build_plan(source, "owner/space", REVISION)

    def test_static_plan_requires_index_not_dockerfile(self) -> None:
        source = publisher.ROOT / "spaces" / "szl-forge-lab"
        output = self._tracked_files(
            "spaces/szl-forge-lab/README.md",
            "spaces/szl-forge-lab/index.html",
        )
        result = subprocess.CompletedProcess([], 0, stdout=output, stderr=b"")
        with patch("publish_hf_space.subprocess.run", return_value=result):
            plan = publisher.build_plan(
                source,
                "owner/space",
                REVISION,
                static=True,
            )
        self.assertEqual(
            {"README.md", "index.html"},
            set(plan["files"]),
        )

    def test_static_plan_rejects_missing_index(self) -> None:
        source = publisher.ROOT / "spaces" / "szl-forge-lab"
        output = self._tracked_files(
            "spaces/szl-forge-lab/README.md",
        )
        result = subprocess.CompletedProcess([], 0, stdout=output, stderr=b"")
        with patch("publish_hf_space.subprocess.run", return_value=result):
            with self.assertRaisesRegex(publisher.PublishError, "index.html"):
                publisher.build_plan(
                    source,
                    "owner/space",
                    REVISION,
                    static=True,
                )

    @staticmethod
    def _missing_space() -> RepositoryNotFoundError:
        request = httpx.Request(
            "GET", "https://huggingface.co/api/spaces/owner/space"
        )
        return RepositoryNotFoundError(
            "missing Space",
            response=httpx.Response(404, request=request),
        )

    def test_existing_space_is_write_confirmed_without_recreation(self) -> None:
        api = Mock()
        api.space_info.return_value = SimpleNamespace(
            id="owner/space",
            runtime=SimpleNamespace(volumes=[]),
        )

        result = publisher.ensure_space_repository(api, "owner/space")

        self.assertEqual(
            "EXISTING_AND_WRITE_CONFIRMED", result["state"]
        )
        self.assertFalse(result["created"])
        self.assertEqual("docker", result["sdk"])
        api.create_repo.assert_not_called()
        api.auth_check.assert_called_once_with(
            repo_id="owner/space",
            repo_type="space",
            write=True,
        )

    def test_missing_docker_space_is_created_and_read_back(self) -> None:
        api = Mock()
        api.space_info.side_effect = [
            self._missing_space(),
            SimpleNamespace(
                id="owner/space",
                runtime=SimpleNamespace(volumes=[]),
            ),
        ]

        result = publisher.ensure_space_repository(api, "owner/space")

        self.assertEqual(
            "CREATED_AND_WRITE_CONFIRMED", result["state"]
        )
        self.assertTrue(result["created"])
        self.assertEqual("public", result["visibility_on_create"])
        api.create_repo.assert_called_once_with(
            repo_id="owner/space",
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True,
        )
        api.auth_check.assert_called_once_with(
            repo_id="owner/space",
            repo_type="space",
            write=True,
        )

    def test_missing_static_space_uses_static_sdk(self) -> None:
        api = Mock()
        api.space_info.side_effect = [
            self._missing_space(),
            SimpleNamespace(
                id="owner/space",
                runtime=SimpleNamespace(volumes=[]),
            ),
        ]

        result = publisher.ensure_space_repository(
            api,
            "owner/space",
            static=True,
        )

        self.assertEqual("static", result["sdk"])
        api.create_repo.assert_called_once_with(
            repo_id="owner/space",
            repo_type="space",
            space_sdk="static",
            private=False,
            exist_ok=True,
        )

    def test_legacy_volume_removal_is_observed(self) -> None:
        api = Mock()
        api.space_info.side_effect = [
            SimpleNamespace(
                runtime=SimpleNamespace(
                    volumes=[SimpleNamespace(source="owner/model")]
                )
            ),
            SimpleNamespace(runtime=SimpleNamespace(volumes=[])),
        ]
        result = publisher.clear_legacy_space_volumes(
            api,
            "owner/space",
            wait_seconds=1,
        )
        self.assertEqual("CLEARED_AND_OBSERVED", result["state"])
        self.assertEqual(1, result["before_count"])
        self.assertEqual(0, result["after_count"])
        api.delete_space_volumes.assert_called_once_with(repo_id="owner/space")
        api.get_space_runtime.assert_not_called()

    def test_exact_runtime_wait_can_require_final_zero_volumes(self) -> None:
        api = Mock()
        info = SimpleNamespace(
            sha="b" * 40,
            runtime=SimpleNamespace(stage="RUNNING", volumes=[]),
        )
        api.space_info.return_value = info
        with patch("publish_hf_space.time.sleep") as sleep:
            observed = publisher.wait_for_exact_running_space(
                api,
                "owner/space",
                "b" * 40,
                wait_seconds=1,
                require_zero_volumes=True,
            )
        self.assertIs(info, observed)
        self.assertEqual(2, api.space_info.call_count)
        sleep.assert_called_once_with(10)

    def test_final_volume_reconciliation_restarts_changed_runtime(self) -> None:
        api = Mock()
        info = SimpleNamespace(
            sha="b" * 40,
            runtime=SimpleNamespace(stage="RUNNING", volumes=[]),
        )
        api.space_info.side_effect = [
            SimpleNamespace(
                runtime=SimpleNamespace(
                    volumes=[SimpleNamespace(source="owner/model")]
                )
            ),
            SimpleNamespace(runtime=SimpleNamespace(volumes=[])),
            info,
            info,
        ]
        api.get_space_runtime.side_effect = [
            SimpleNamespace(
                stage="BUILDING",
                raw={"domains": [{"stage": "BUILDING"}]},
            )
        ]
        with patch("publish_hf_space.time.sleep"):
            evidence, observed = publisher.reconcile_final_space_volumes(
                api,
                "owner/space",
                "b" * 40,
                wait_seconds=1,
            )
        self.assertIs(info, observed)
        self.assertTrue(evidence["restart_requested"])
        self.assertTrue(evidence["restart_transition"]["observed"])
        self.assertEqual("BUILDING", evidence["restart_transition"]["runtime_stage"])
        self.assertEqual(0, evidence["final_count"])
        api.delete_space_volumes.assert_called_once_with(repo_id="owner/space")
        api.restart_space.assert_called_once_with(repo_id="owner/space")



class PublishedByteVerificationTests(unittest.TestCase):
    def _files(self, **contents: bytes) -> dict[str, dict[str, object]]:
        return {
            name: {"sha256": publisher.sha256_bytes(data), "size": len(data)}
            for name, data in contents.items()
        }

    def test_payload_files_require_exact_bytes(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "Dockerfile").write_bytes(b"FROM scratch\n")
            files = self._files(Dockerfile=b"FROM scratch\n")
            receipt = publisher.verify_published_bytes(
                files, source, lambda target: b"FROM scratch\n"
            )
            self.assertEqual({}, receipt)
            with self.assertRaisesRegex(publisher.PublishError, "byte mismatch: Dockerfile"):
                publisher.verify_published_bytes(files, source, lambda target: b"FROM other\n")

    def test_gitattributes_accepts_hub_appended_lfs_rules_only(self) -> None:
        import tempfile
        from pathlib import Path

        source_text = b"* -text\nrelease.zip -filter -diff -merge -text\n"
        published = source_text + b"release.zip filter=lfs diff=lfs merge=lfs -text\n"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / ".gitattributes").write_bytes(source_text)
            files = self._files(**{".gitattributes": source_text})
            receipt = publisher.verify_published_bytes(files, source, lambda target: published)
            entry = receipt[".gitattributes"]
            self.assertEqual("SOURCE_LINES_PRESENT_NOT_BYTE_PARITY", entry["verification"])
            self.assertEqual(publisher.sha256_bytes(source_text), entry["source_sha256"])
            self.assertEqual(publisher.sha256_bytes(published), entry["published_sha256"])
            self.assertEqual(
                ["release.zip filter=lfs diff=lfs merge=lfs -text"], entry["hub_appended_lines"]
            )
            with self.assertRaisesRegex(publisher.PublishError, "missing="):
                publisher.verify_published_bytes(files, source, lambda target: b"* -text\n")
            with self.assertRaisesRegex(publisher.PublishError, "foreign="):
                publisher.verify_published_bytes(
                    files, source, lambda target: source_text + b"secret.bin -text\n"
                )


LIVE_GITATTRIBUTES = (
    "* -text\n"
    "*.zip filter=lfs diff=lfs merge=lfs -text\n"
    "release.zip filter=lfs diff=lfs merge=lfs -text\n"
)
REGRESSED_GITATTRIBUTES = "* -text\nrelease.zip -filter -diff -merge -text\n"
FOUNDATION_DIGEST = "869e318dd5f328205dd181ee836ef267bd2ae278f6430a9e8ddc661fbc689d03"


def lfs_pointer(oid: str, size: int) -> bytes:
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {size}\n".encode()


class FrozenLfsArchivePublicationTests(unittest.TestCase):
    ARCHIVE = b"PK\x03\x04 frozen archive bytes"

    def setUp(self) -> None:
        import tempfile
        from pathlib import Path

        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.source = Path(self._directory.name)
        self.digest = publisher.sha256_bytes(self.ARCHIVE)
        (self.source / ".gitattributes").write_text(LIVE_GITATTRIBUTES, encoding="utf-8")
        (self.source / "release.zip").write_bytes(self.ARCHIVE)

    def files(self, *, pointer_oid: str | None = None) -> dict:
        files = {
            ".gitattributes": {"sha256": publisher.sha256_bytes(LIVE_GITATTRIBUTES.encode()),
                               "size": len(LIVE_GITATTRIBUTES)},
            "release.zip": {"sha256": self.digest, "size": len(self.ARCHIVE)},
        }
        if pointer_oid is not None:
            files["release.zip"]["lfs_pointer"] = {"oid": pointer_oid, "size": len(self.ARCHIVE)}
        return files

    def test_live_gitattributes_routes_release_zip_through_lfs(self) -> None:
        self.assertEqual({"filter": "lfs", "diff": "lfs", "merge": "lfs"},
                         publisher.require_lfs_routing(LIVE_GITATTRIBUTES, "release.zip"))
        repository = (publisher.ROOT / "spaces" / "szl-foundation-confirmation" / ".gitattributes")
        self.assertEqual(LIVE_GITATTRIBUTES, repository.read_text(encoding="utf-8"))
        publisher.require_lfs_routing(repository.read_text(encoding="utf-8"), "release.zip")

    def test_gitattributes_that_disable_lfs_are_refused(self) -> None:
        for text in (REGRESSED_GITATTRIBUTES, "* -text\n", LIVE_GITATTRIBUTES + "release.zip -filter\n",
                     LIVE_GITATTRIBUTES + "*.zip binary\n", LIVE_GITATTRIBUTES + "release.zip !filter\n",
                     "*.zip filter=lfs\n"):
            with self.subTest(text=text):
                with self.assertRaisesRegex(publisher.PublishError, "does not route release.zip through LFS"):
                    publisher.require_lfs_routing(text, "release.zip")

    def test_frozen_archive_is_bound_before_upload(self) -> None:
        evidence = publisher.verify_frozen_archives_before(
            self.files(pointer_oid=self.digest), self.source, {"release.zip": self.digest})
        item = evidence["release.zip"]
        self.assertTrue(item["verified_before_upload"])
        self.assertFalse(item["verified_after_publish"])
        self.assertEqual(self.digest, item["lfs_pointer_oid"])

    def test_frozen_archive_failures_before_upload(self) -> None:
        cases = [
            ({"release.zip": "0" * 64}, self.files(), "digest mismatch before upload"),
            ({"release.zip": self.digest}, self.files(pointer_oid="1" * 64), "pointer oid mismatch"),
            ({"absent.zip": self.digest}, self.files(), "not a tracked Space file"),
        ]
        for frozen, files, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(publisher.PublishError, message):
                    publisher.verify_frozen_archives_before(files, self.source, frozen)
        (self.source / ".gitattributes").write_text(REGRESSED_GITATTRIBUTES, encoding="utf-8")
        with self.assertRaisesRegex(publisher.PublishError, "does not route"):
            publisher.verify_frozen_archives_before(self.files(), self.source, {"release.zip": self.digest})
        files = self.files()
        del files[".gitattributes"]
        with self.assertRaisesRegex(publisher.PublishError, "require a tracked Space .gitattributes"):
            publisher.verify_frozen_archives_before(files, self.source, {"release.zip": self.digest})

    def test_frozen_archive_is_reverified_after_publication(self) -> None:
        evidence = publisher.verify_frozen_archives_before(
            self.files(), self.source, {"release.zip": self.digest})
        published = {".gitattributes": LIVE_GITATTRIBUTES.encode(), "release.zip": self.ARCHIVE}
        result = publisher.verify_frozen_archives_after(evidence, published.__getitem__)
        self.assertTrue(result["release.zip"]["verified_after_publish"])
        for broken, message in (
            ({**published, "release.zip": lfs_pointer(self.digest, len(self.ARCHIVE))}, "published frozen archive digest mismatch"),
            ({**published, ".gitattributes": REGRESSED_GITATTRIBUTES.encode()}, "does not route"),
        ):
            with self.subTest(message=message):
                fresh = publisher.verify_frozen_archives_before(
                    self.files(), self.source, {"release.zip": self.digest})
                with self.assertRaisesRegex(publisher.PublishError, message):
                    publisher.verify_frozen_archives_after(fresh, broken.__getitem__)

    def test_frozen_archive_declarations_are_exact(self) -> None:
        self.assertEqual({"release.zip": FOUNDATION_DIGEST},
                         publisher.parse_frozen_archives([f"release.zip={FOUNDATION_DIGEST}"]))
        for value in ("release.zip", f"={FOUNDATION_DIGEST}", "release.zip=abc", f"/release.zip={FOUNDATION_DIGEST}",
                      f"../release.zip={FOUNDATION_DIGEST}", f"release.zip={FOUNDATION_DIGEST.upper()}"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(publisher.PublishError, "invalid --frozen-archive"):
                    publisher.parse_frozen_archives([value])
        with self.assertRaisesRegex(publisher.PublishError, "invalid --frozen-archive"):
            publisher.parse_frozen_archives([f"release.zip={FOUNDATION_DIGEST}"] * 2)

    def test_pointer_is_never_uploaded_in_place_of_the_archive(self) -> None:
        source = publisher.ROOT / "spaces" / "szl-foundation-confirmation"
        ls_files = b"\0".join(path.encode() for path in (
            "spaces/szl-foundation-confirmation/README.md",
            "spaces/szl-foundation-confirmation/Dockerfile",
            "spaces/szl-foundation-confirmation/release.zip",
        )) + b"\0"
        result = subprocess.CompletedProcess([], 0, stdout=ls_files, stderr=b"")
        pointer = lfs_pointer(FOUNDATION_DIGEST, 2893790)
        real_read = publisher.Path.read_bytes

        def read(path):
            return pointer if path.name == "release.zip" else real_read(path)

        with patch("publish_hf_space.subprocess.run", return_value=result), \
                patch.object(publisher.Path, "read_bytes", autospec=True, side_effect=read):
            with self.assertRaisesRegex(publisher.PublishError, "LFS object not materialized"):
                publisher.build_plan(source, "owner/space", REVISION)

    def test_materialized_bytes_must_match_the_committed_pointer(self) -> None:
        source = publisher.ROOT / "spaces" / "szl-foundation-confirmation"
        paths = ["spaces/szl-foundation-confirmation/README.md",
                 "spaces/szl-foundation-confirmation/Dockerfile",
                 "spaces/szl-foundation-confirmation/release.zip"]
        real_read = publisher.Path.read_bytes

        def read(path):
            return self.ARCHIVE if path.name == "release.zip" else real_read(path)

        for oid, expected_error in ((self.digest, None), ("2" * 64, "do not match the committed pointer")):
            committed = lfs_pointer(oid, len(self.ARCHIVE))

            def run(argv, **kwargs):
                if argv[:2] == ["git", "ls-files"]:
                    return subprocess.CompletedProcess(argv, 0, stdout=b"\0".join(p.encode() for p in paths) + b"\0")
                if argv[:2] == ["git", "check-attr"]:
                    out = b"".join(p.encode() + b"\0filter\0" + (b"lfs" if p.endswith(".zip") else b"unspecified") + b"\0"
                                   for p in paths)
                    return subprocess.CompletedProcess(argv, 0, stdout=out)
                if argv[:2] == ["git", "cat-file"]:
                    return subprocess.CompletedProcess(argv, 0, stdout=committed)
                raise AssertionError(argv)

            with self.subTest(oid=oid), patch("publish_hf_space.subprocess.run", side_effect=run), \
                    patch.object(publisher.Path, "read_bytes", autospec=True, side_effect=read):
                if expected_error:
                    with self.assertRaisesRegex(publisher.PublishError, expected_error):
                        publisher.build_plan(source, "owner/space", REVISION)
                else:
                    plan = publisher.build_plan(source, "owner/space", REVISION)
                    self.assertEqual({"oid": oid, "size": len(self.ARCHIVE)}, plan["files"]["release.zip"]["lfs_pointer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
