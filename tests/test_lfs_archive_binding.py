"""LFS pointer binding for frozen archives: accept only oid == frozen digest.

Every branch runs against a real temporary Git repository using plumbing only,
so results do not depend on git-lfs being installed. The real checkout is also
verified: the foundation-confirmation archive must be bound to app.py's
ARCHIVE_SHA256 and its Space .gitattributes must match the live repair.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("lfs_archive_binding_tests", ROOT / "scripts" / "lfs_archive_binding.py")
binding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(binding)

SPACE = "spaces/szl-foundation-confirmation"
ARCHIVE = f"{SPACE}/release.zip"
FROZEN = "869e318dd5f328205dd181ee836ef267bd2ae278f6430a9e8ddc661fbc689d03"
LIVE_GITATTRIBUTES = (
    b"* -text\n"
    b"*.zip filter=lfs diff=lfs merge=lfs -text\n"
    b"release.zip filter=lfs diff=lfs merge=lfs -text\n"
)
REGRESSED_GITATTRIBUTES = b"* -text\nrelease.zip -filter -diff -merge -text\n"
PAYLOAD = b"PK\x03\x04 frozen archive stand-in bytes"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


def pointer(oid: str, size: int) -> bytes:
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {size}\n".encode()


class TemporaryArchiveRepository:
    """A throwaway repository holding ``archive.zip`` under an attributes file."""

    def __init__(self, *, committed: bytes, attributes: bytes = LIVE_GITATTRIBUTES, track: bool = True):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
                    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
                    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_LFS_SKIP_SMUDGE": "1"}
        self.git("init", "-q")
        (self.root / ".gitattributes").write_bytes(attributes)
        (self.root / "archive.zip").write_bytes(committed)
        self.git("add", ".gitattributes")
        if track:
            self.git("add", "archive.zip")
        self.git("commit", "-q", "--no-verify", "-m", "fixture")

    def git(self, *args: str) -> None:
        subprocess.run(["git", "-c", "core.hooksPath=" + os.devnull, *args], cwd=self.root,
                       env=self.env, check=True, capture_output=True, timeout=30)

    def verify(self, expected: str = PAYLOAD_SHA) -> dict:
        return binding.verify_archive(self.root, "archive.zip", expected)

    def close(self) -> None:
        self._directory.cleanup()


class LfsArchiveBindingTests(unittest.TestCase):
    def repository(self, **kwargs) -> TemporaryArchiveRepository:
        repo = TemporaryArchiveRepository(**kwargs)
        self.addCleanup(repo.close)
        return repo

    def assert_code(self, code: str, repo: TemporaryArchiveRepository, expected: str = PAYLOAD_SHA) -> None:
        with self.assertRaises(binding.BindingError) as caught:
            repo.verify(expected)
        self.assertEqual(caught.exception.code, code, str(caught.exception))

    def test_pointer_whose_oid_is_the_frozen_digest_is_accepted(self):
        repo = self.repository(committed=pointer(PAYLOAD_SHA, len(PAYLOAD)))
        evidence = repo.verify()
        self.assertEqual(evidence["worktree"], "LFS_POINTER_NOT_MATERIALIZED")
        self.assertEqual(evidence["pointer_oid"], PAYLOAD_SHA)
        self.assertEqual(evidence["lfs_attributes"], {"filter": "lfs", "diff": "lfs", "merge": "lfs"})

    def test_materialized_bytes_matching_the_pointer_are_accepted(self):
        repo = self.repository(committed=pointer(PAYLOAD_SHA, len(PAYLOAD)))
        (repo.root / "archive.zip").write_bytes(PAYLOAD)
        self.assertEqual(repo.verify()["worktree"], "LFS_MATERIALIZED_BYTES_MATCH")

    def test_pointer_with_a_different_oid_fails_closed(self):
        other = "0" * 64
        repo = self.repository(committed=pointer(other, len(PAYLOAD)))
        self.assert_code("POINTER_OID_MISMATCH", repo)

    def test_raw_binary_whose_digest_differs_fails_closed(self):
        repo = self.repository(committed=pointer(PAYLOAD_SHA, len(PAYLOAD)))
        (repo.root / "archive.zip").write_bytes(PAYLOAD + b"tampered")
        self.assert_code("WORKTREE_DIGEST_MISMATCH", repo)
        # Same digest prefix but truncated/oversized bytes must not pass either.
        (repo.root / "archive.zip").write_bytes(PAYLOAD[:-1])
        self.assert_code("WORKTREE_DIGEST_MISMATCH", repo)

    def test_committed_raw_binary_is_not_an_lfs_binding(self):
        repo = self.repository(committed=PAYLOAD)
        self.assert_code("COMMITTED_NOT_LFS_POINTER", repo)

    def test_missing_archive_fails_closed(self):
        repo = self.repository(committed=pointer(PAYLOAD_SHA, len(PAYLOAD)))
        (repo.root / "archive.zip").unlink()
        self.assert_code("MISSING", repo)

    def test_untracked_archive_fails_closed(self):
        repo = self.repository(committed=pointer(PAYLOAD_SHA, len(PAYLOAD)), track=False)
        self.assert_code("UNTRACKED", repo)

    def test_modified_tracked_archive_fails_closed(self):
        repo = self.repository(committed=pointer(PAYLOAD_SHA, len(PAYLOAD)))
        (repo.root / "archive.zip").write_bytes(pointer("1" * 64, len(PAYLOAD)))
        self.assert_code("WORKTREE_POINTER_MISMATCH", repo)
        repo.git("add", "archive.zip")
        self.assert_code("STAGED_CHANGE", repo)

    def test_gitattributes_disabling_the_lfs_filter_fails_closed(self):
        for attributes in (REGRESSED_GITATTRIBUTES, b"* -text\n", b"*.zip filter=lfs diff=lfs merge=lfs -text\narchive.zip -filter\n",
                           b"archive.zip filter=lfs diff=lfs merge=lfs -text\narchive.zip !filter\n"):
            with self.subTest(attributes=attributes):
                repo = self.repository(committed=pointer(PAYLOAD_SHA, len(PAYLOAD)), attributes=attributes)
                self.assert_code("LFS_FILTER_DISABLED", repo)

    def test_malformed_expected_digest_fails_closed(self):
        repo = self.repository(committed=pointer(PAYLOAD_SHA, len(PAYLOAD)))
        for expected in ("", PAYLOAD_SHA.upper(), PAYLOAD_SHA[:-1]):
            with self.subTest(expected=expected):
                self.assert_code("FROZEN_DIGEST_UNREADABLE", repo, expected)

    def test_pointer_parser_is_exact(self):
        good = pointer(PAYLOAD_SHA, 7)
        self.assertEqual(binding.parse_pointer(good), (PAYLOAD_SHA, 7))
        for bad in (good[:-1], good + b"\n", good.replace(b"sha256:", b"sha1:"), b" " + good,
                    good.upper(), pointer(PAYLOAD_SHA, 7).replace(b"size 7", b"size 07")):
            with self.subTest(bad=bad):
                self.assertIsNone(binding.parse_pointer(bad))

    def test_frozen_digest_is_read_from_the_declaring_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text(f'ARCHIVE_SHA256 = "{PAYLOAD_SHA}"\n', encoding="utf-8")
            self.assertEqual(binding.frozen_digest(root, "app.py", "ARCHIVE_SHA256"), PAYLOAD_SHA)
            for text in ("", f'ARCHIVE_SHA256 = "{PAYLOAD_SHA}"\nARCHIVE_SHA256 = "{PAYLOAD_SHA}"\n',
                         'ARCHIVE_SHA256 = "abc"\n', "ARCHIVE_SHA256 = None\n", "def (:\n"):
                with self.subTest(text=text):
                    (root / "app.py").write_text(text, encoding="utf-8")
                    with self.assertRaises(binding.BindingError) as caught:
                        binding.frozen_digest(root, "app.py", "ARCHIVE_SHA256")
                    self.assertEqual(caught.exception.code, "FROZEN_DIGEST_UNREADABLE")


class RealCheckoutBindingTests(unittest.TestCase):
    def test_space_gitattributes_is_byte_identical_to_the_live_repair(self):
        self.assertEqual((ROOT / SPACE / ".gitattributes").read_bytes(), LIVE_GITATTRIBUTES)

    def test_foundation_archive_is_registered_against_app_digest(self):
        self.assertEqual(binding.BOUND_ARCHIVES[ARCHIVE], (f"{SPACE}/app.py", "ARCHIVE_SHA256"))
        self.assertEqual(binding.frozen_digest(ROOT, f"{SPACE}/app.py", "ARCHIVE_SHA256"), FROZEN)
        self.assertIn(f"Archive SHA-256: `{FROZEN}`", (ROOT / SPACE / "README.md").read_text(encoding="utf-8"))

    def test_real_checkout_binds_pointer_oid_to_the_frozen_digest(self):
        evidence = binding.verify_registered(ROOT)
        self.assertEqual([item["path"] for item in evidence], [ARCHIVE])
        self.assertEqual(evidence[0]["pointer_oid"], FROZEN)
        self.assertEqual(evidence[0]["pointer_size"], 2893790)
        self.assertIn(evidence[0]["worktree"], {"LFS_POINTER_NOT_MATERIALIZED", "LFS_MATERIALIZED_BYTES_MATCH"})

    def test_publisher_workflow_materializes_lfs_and_binds_the_frozen_digest(self):
        import yaml

        workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "foundation-runtime.yml").read_text(encoding="utf-8"))
        declaration = f"--frozen-archive release.zip={FROZEN}"
        for job in ("qualify", "publish"):
            steps = workflow["jobs"][job]["steps"]
            checkout = [step for step in steps if str(step.get("uses", "")).startswith("actions/checkout@")]
            self.assertEqual(len(checkout), 1, job)
            self.assertIs(checkout[0]["with"]["lfs"], True, job)
            publisher = [str(step.get("run", "")) for step in steps if "tools/publish_hf_space.py" in str(step.get("run", ""))]
            self.assertEqual(len(publisher), 1, job)
            self.assertIn(declaration, publisher[0], job)
        qualify_publisher = next(str(step["run"]) for step in workflow["jobs"]["qualify"]["steps"]
                                 if "tools/publish_hf_space.py" in str(step.get("run", "")))
        self.assertNotIn("--publish", qualify_publisher)
        self.assertEqual(workflow["jobs"]["publish"]["if"], "github.ref == 'refs/heads/main' && github.event_name != 'pull_request'")


if __name__ == "__main__":
    unittest.main()
