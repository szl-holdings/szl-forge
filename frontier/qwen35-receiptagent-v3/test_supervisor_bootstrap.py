from __future__ import annotations

import hashlib
import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = pathlib.Path(__file__).resolve().parent
MODULE_PATH = HERE / "supervisor_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("receipt_agent_v3_bootstrap", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load supervisor_bootstrap.py")
bootstrap = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bootstrap
SPEC.loader.exec_module(bootstrap)


SOURCE = "a" * 40
RUN_ID = "b" * 32


class SourceBootstrapTests(unittest.TestCase):
    def component_fixture(
        self,
        root: pathlib.Path,
        *,
        sentinel: pathlib.Path | None = None,
        validator_drift: bool = False,
    ) -> tuple[pathlib.Path, dict[str, bytes]]:
        component_dir = root / "frontier" / "qwen35-receiptagent-v3"
        component_dir.mkdir(parents=True)
        if sentinel is None:
            trainer = b"VALUE = 'trainer'\n"
        else:
            trainer = (
                "from pathlib import Path\n"
                f"Path({str(sentinel)!r}).write_text('executed', encoding='utf-8')\n"
                "VALUE = 'trainer'\n"
            ).encode("utf-8")
        committed = {
            "launch_supervised_training.py": b"LAUNCHER = True\n",
            "supervisor_bootstrap.py": b"BOOTSTRAP = True\n",
            "supervise_training.py": b"SUPERVISOR = True\n",
            "containment_probe.py": b"CONTAINMENT = True\n",
            "train_candidate.py": trainer,
            "supervisor_validation.py": b"VALUE = 'validator'\n",
        }
        for filename, data in committed.items():
            local = data
            if filename == "supervisor_validation.py" and validator_drift:
                local = b"VALUE = 'drifted-validator'\n"
            (component_dir / filename).write_bytes(local)
        return component_dir, committed

    def git_result(
        self,
        committed: dict[str, bytes],
        *,
        branch: str = "main",
        head: str = SOURCE,
        cached_main: str = SOURCE,
        remote_main: str = SOURCE,
        dirty: bytes = b"",
    ):
        def run(_root: pathlib.Path, *arguments: str, timeout: float = 60.0):
            self.assertGreater(timeout, 0)
            if arguments == ("remote", "get-url", "origin"):
                stdout = b"https://github.com/szl-holdings/szl-forge.git\n"
            elif arguments == (
                "ls-remote",
                "--exit-code",
                "origin",
                "refs/heads/main",
            ):
                stdout = f"{remote_main}\trefs/heads/main\n".encode()
            elif arguments == ("rev-parse", "HEAD"):
                stdout = f"{head}\n".encode()
            elif arguments == ("branch", "--show-current"):
                stdout = f"{branch}\n".encode()
            elif arguments == ("rev-parse", "refs/remotes/origin/main"):
                stdout = f"{cached_main}\n".encode()
            elif arguments == ("status", "--porcelain", "--untracked-files=all"):
                stdout = dirty
            elif arguments[0] == "show":
                filename = arguments[1].rsplit("/", 1)[1]
                stdout = committed[filename]
            else:
                self.fail(f"unexpected git arguments: {arguments!r}")
            return subprocess.CompletedProcess(
                [bootstrap.GIT, *arguments],
                0,
                stdout=stdout,
                stderr=b"",
            )

        return run

    def test_all_components_are_verified_before_any_sibling_executes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            sentinel = root / "malicious-import-sentinel"
            component_dir, committed = self.component_fixture(
                root,
                sentinel=sentinel,
                validator_drift=True,
            )
            with mock.patch.object(
                bootstrap,
                "_run_git",
                side_effect=self.git_result(committed),
            ):
                with self.assertRaisesRegex(
                    bootstrap.SourceVerificationError,
                    "supervisor_validation.py",
                ):
                    bootstrap.verify_and_load_siblings(
                        SOURCE,
                        repo_root=root,
                        component_dir=component_dir,
                    )
            self.assertFalse(sentinel.exists())
            self.assertNotIn("szl_ra3_train_candidate", sys.modules)
            self.assertNotIn("szl_ra3_supervisor_validation", sys.modules)

    def test_verified_in_memory_bytes_are_loaded_and_publicly_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            component_dir, committed = self.component_fixture(root)
            previous = {
                name: sys.modules.get(name)
                for name in bootstrap.SIBLING_MODULE_NAMES.values()
            }
            try:
                with mock.patch.object(
                    bootstrap,
                    "_run_git",
                    side_effect=self.git_result(committed),
                ):
                    verified, modules = bootstrap.verify_and_load_siblings(
                        SOURCE,
                        repo_root=root,
                        component_dir=component_dir,
                    )
                self.assertEqual("trainer", modules["train_candidate.py"].VALUE)
                self.assertEqual("validator", modules["supervisor_validation.py"].VALUE)
                evidence = verified.public_evidence()
                self.assertEqual(SOURCE, evidence["revision"])
                self.assertEqual(set(committed), set(evidence["components"]))
                self.assertEqual(
                    hashlib.sha256(committed["train_candidate.py"]).hexdigest(),
                    evidence["components"]["train_candidate.py"]["sha256"],
                )
            finally:
                for name, module in previous.items():
                    if module is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = module

    def test_launcher_and_containment_tampering_fail_before_sibling_imports(self):
        for filename in ("launch_supervised_training.py", "containment_probe.py"):
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as directory:
                    root = pathlib.Path(directory).resolve()
                    component_dir, committed = self.component_fixture(root)
                    (component_dir / filename).write_bytes(b"tampered\n")
                    with mock.patch.object(
                        bootstrap,
                        "_run_git",
                        side_effect=self.git_result(committed),
                    ):
                        with self.assertRaisesRegex(
                            bootstrap.SourceVerificationError,
                            filename,
                        ):
                            bootstrap.verify_and_load_siblings(
                                SOURCE,
                                repo_root=root,
                                component_dir=component_dir,
                            )

    def test_fresh_clean_current_main_is_exactly_required(self):
        cases = (
            ({"branch": "feature"}, "local main"),
            ({"head": "c" * 40}, "HEAD differs"),
            ({"cached_main": "d" * 40}, "cached origin/main differs"),
            ({"remote_main": "e" * 40}, "fresh remote main differs"),
            ({"dirty": b" M changed.py\n"}, "clean worktree"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with tempfile.TemporaryDirectory() as directory:
                    root = pathlib.Path(directory).resolve()
                    component_dir, committed = self.component_fixture(root)
                    with mock.patch.object(
                        bootstrap,
                        "_run_git",
                        side_effect=self.git_result(committed, **overrides),
                    ):
                        with self.assertRaisesRegex(
                            bootstrap.SourceVerificationError,
                            message,
                        ):
                            bootstrap.verify_exact_source_before_import(
                                SOURCE,
                                repo_root=root,
                                component_dir=component_dir,
                            )

    def test_git_executable_is_fixed_and_shell_is_never_used(self):
        completed = subprocess.CompletedProcess(
            [],
            0,
            stdout=b"value\n",
            stderr=b"",
        )
        with mock.patch.object(
            bootstrap.subprocess,
            "run",
            return_value=completed,
        ) as run:
            bootstrap._run_git(pathlib.Path("/fixed/repository"), "rev-parse", "HEAD")
        self.assertEqual(
            ["/usr/bin/git", "rev-parse", "HEAD"],
            run.call_args.args[0],
        )
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertFalse(run.call_args.kwargs["check"])


class StrictJSONTests(unittest.TestCase):
    def test_one_open_binds_parsed_bytes_and_digest(self):
        raw = b'{"outer":{"state":"ok"},"value":1}\n'
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory).resolve() / "report.json"
            path.write_bytes(raw)
            with (
                mock.patch.object(bootstrap.os, "open", wraps=os.open) as opened,
                mock.patch.object(
                    pathlib.Path,
                    "read_bytes",
                    side_effect=AssertionError("path must not be reopened"),
                ),
            ):
                document = bootstrap.read_strict_json_once(path)
        self.assertEqual(1, opened.call_count)
        self.assertEqual(raw, document.raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), document.sha256)
        self.assertEqual("ok", document.value["outer"]["state"])

    def test_duplicate_nonfinite_nul_and_oversize_inputs_fail_closed(self):
        cases = (
            (b'{"state":1,"state":2}', bootstrap.DuplicateJSONKey),
            (b'{"outer":{"state":1,"state":2}}', bootstrap.DuplicateJSONKey),
            (b'{"value":1e999}', bootstrap.StrictJSONError),
            (b'{"value":NaN}', bootstrap.StrictJSONError),
            (b'{"value":"nul\x00byte"}', bootstrap.StrictJSONError),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory).resolve() / "report.json"
            for raw, error in cases:
                with self.subTest(raw=raw):
                    path.write_bytes(raw)
                    with self.assertRaises(error):
                        bootstrap.read_strict_json_once(path)
            path.write_bytes(b'{"tooLarge":true}')
            with self.assertRaises(bootstrap.StrictJSONError):
                bootstrap.read_strict_json_once(path, maximum_bytes=4)

    @unittest.skipUnless(os.name == "posix", "O_NOFOLLOW semantics require POSIX")
    def test_symlink_and_hardlink_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            target = root / "target.json"
            target.write_bytes(b'{"state":"ok"}')
            symlink = root / "symlink.json"
            symlink.symlink_to(target)
            with self.assertRaises((OSError, bootstrap.StrictJSONError)):
                bootstrap.read_strict_json_once(symlink)
            hardlink = root / "hardlink.json"
            os.link(target, hardlink)
            with self.assertRaises(bootstrap.StrictJSONError):
                bootstrap.read_strict_json_once(target)


@unittest.skipUnless(os.name == "posix", "write-once protocol requires POSIX")
class WriteOncePublicationTests(unittest.TestCase):
    def test_success_reaches_explicit_commit_point(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            artifact = bootstrap.publish_write_once(root, "report.json", b"evidence")
            self.assertTrue(artifact.committed)
            self.assertEqual("FINAL_LINK_AND_DIRECTORY_FSYNC", artifact.commit_point)
            self.assertTrue(artifact.cleanup_complete)
            self.assertEqual(b"evidence", artifact.path.read_bytes())
            self.assertEqual([], list(root.glob(".*.tmp")))

    def test_existing_final_and_symlink_are_never_clobbered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            final = root / "report.json"
            final.write_bytes(b"original")
            with self.assertRaises(bootstrap.PublicationNotCommitted):
                bootstrap.publish_write_once(root, "report.json", b"replacement")
            self.assertEqual(b"original", final.read_bytes())
            final.unlink()
            target = root / "target.json"
            target.write_bytes(b"target")
            final.symlink_to(target)
            with self.assertRaises(bootstrap.PublicationNotCommitted):
                bootstrap.publish_write_once(root, "report.json", b"replacement")
            self.assertTrue(final.is_symlink())
            self.assertEqual(b"target", target.read_bytes())

    def test_file_fsync_failure_is_not_committed_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            real_fsync = os.fsync
            calls = 0

            def fail_first(descriptor: int):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OSError("injected file fsync failure")
                return real_fsync(descriptor)

            with mock.patch.object(bootstrap.os, "fsync", side_effect=fail_first):
                with self.assertRaises(bootstrap.PublicationNotCommitted):
                    bootstrap.publish_write_once(root, "report.json", b"evidence")
            self.assertFalse((root / "report.json").exists())
            self.assertEqual([], list(root.glob(".*.tmp")))

    def test_commit_fsync_failure_is_explicitly_indeterminate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            real_fsync = os.fsync
            calls = 0

            def fail_second(descriptor: int):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected commit fsync failure")
                return real_fsync(descriptor)

            with mock.patch.object(bootstrap.os, "fsync", side_effect=fail_second):
                with self.assertRaises(bootstrap.PublicationIndeterminate) as captured:
                    bootstrap.publish_write_once(root, "report.json", b"evidence")
            self.assertEqual(root / "report.json", captured.exception.final_path)
            self.assertTrue(captured.exception.final_path.exists())
            self.assertTrue(captured.exception.temporary_path.exists())

    def test_post_commit_unlink_failure_never_downgrades_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            with mock.patch.object(
                bootstrap.os,
                "unlink",
                side_effect=PermissionError("injected unlink failure"),
            ):
                artifact = bootstrap.publish_write_once(
                    root,
                    "report.json",
                    b"committed",
                )
            self.assertTrue(artifact.committed)
            self.assertFalse(artifact.cleanup_complete)
            self.assertEqual("unlink:PermissionError", artifact.cleanup_error)
            self.assertEqual(b"committed", artifact.path.read_bytes())
            self.assertIsNotNone(artifact.temporary_path)
            self.assertTrue(artifact.temporary_path.exists())

    def test_post_commit_cleanup_fsync_failure_never_downgrades_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            real_fsync = os.fsync
            calls = 0

            def fail_third(descriptor: int):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError("injected cleanup fsync failure")
                return real_fsync(descriptor)

            with mock.patch.object(bootstrap.os, "fsync", side_effect=fail_third):
                artifact = bootstrap.publish_write_once(
                    root,
                    "report.json",
                    b"committed",
                )
            self.assertTrue(artifact.committed)
            self.assertFalse(artifact.cleanup_complete)
            self.assertEqual(
                "cleanup-fsync:OSError",
                artifact.cleanup_error,
            )
            self.assertEqual(b"committed", artifact.path.read_bytes())
            self.assertEqual([], list(root.glob(".*.tmp")))


@unittest.skipUnless(os.name == "posix", "atomic admission requires POSIX")
class AtomicAdmissionTests(unittest.TestCase):
    def test_complete_admission_prepares_exact_paths_and_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_root = pathlib.Path(directory).resolve()
            result = bootstrap.admit_attempt_atomic(runs_root, RUN_ID, 4096)
            self.assertTrue(result.prepared)
            self.assertTrue(result.reserve_allocated)
            self.assertIsNone(result.tombstone)
            self.assertEqual(4096, result.paths.reserve.stat().st_size)
            self.assertTrue(result.paths.payload.is_dir())
            self.assertTrue(result.paths.logs.is_dir())
            self.assertTrue(result.paths.reports.is_dir())
            self.assertTrue((result.paths.runtime_cache / "cuda").is_dir())

    def test_existing_empty_leaf_is_unchanged_and_not_tombstoned(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_root = pathlib.Path(directory).resolve()
            leaf = runs_root / RUN_ID
            leaf.mkdir(mode=0o700)
            before = leaf.stat()
            with self.assertRaises(bootstrap.AdmissionCollision):
                bootstrap.admit_attempt_atomic(runs_root, RUN_ID, 4096)
            after = leaf.stat()
            self.assertEqual(before.st_ino, after.st_ino)
            self.assertEqual([], list(leaf.iterdir()))

    def test_post_leaf_directory_failure_returns_partial_with_tombstone(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_root = pathlib.Path(directory).resolve()
            with mock.patch.object(
                bootstrap,
                "_mkdir_at",
                side_effect=OSError("injected reports failure"),
            ):
                result = bootstrap.admit_attempt_atomic(runs_root, RUN_ID, 4096)
            self.assertFalse(result.prepared)
            self.assertEqual("CREATE_REPORTS_DIRECTORY", result.failed_stage)
            self.assertIsNotNone(result.tombstone)
            self.assertIsNotNone(result.tombstone.artifact)
            document = bootstrap.read_strict_json_once(
                result.paths.reports / "admission-failure.json"
            )
            self.assertEqual("ADMISSION_FAILED_PARTIAL_LEAF", document.value["state"])
            self.assertTrue(document.value["attemptLeafExclusivelyCreated"])
            self.assertFalse(document.value["workerLaunched"])
            self.assertFalse(document.value["publicationEligible"])

    def test_failure_after_reserve_releases_it_for_durable_tombstone(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_root = pathlib.Path(directory).resolve()
            real_mkdir = bootstrap._mkdir_at
            calls = 0

            def fail_payload(directory_fd: int, name: str):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected payload failure")
                return real_mkdir(directory_fd, name)

            with mock.patch.object(
                bootstrap,
                "_mkdir_at",
                side_effect=fail_payload,
            ):
                result = bootstrap.admit_attempt_atomic(runs_root, RUN_ID, 4096)
            self.assertFalse(result.prepared)
            self.assertEqual("CREATE_PAYLOAD_DIRECTORY", result.failed_stage)
            self.assertFalse(result.reserve_allocated)
            self.assertTrue(result.tombstone.reserve_released)
            self.assertFalse(result.paths.reserve.exists())
            self.assertTrue((result.paths.reports / "admission-failure.json").is_file())

    def test_tombstone_publication_failure_is_returned_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_root = pathlib.Path(directory).resolve()
            with (
                mock.patch.object(
                    bootstrap,
                    "_mkdir_at",
                    side_effect=OSError("injected preparation failure"),
                ),
                mock.patch.object(
                    bootstrap,
                    "publish_write_once",
                    side_effect=bootstrap.PublicationNotCommitted(
                        "injected tombstone failure"
                    ),
                ),
            ):
                result = bootstrap.admit_attempt_atomic(runs_root, RUN_ID, 4096)
            self.assertFalse(result.prepared)
            self.assertIsNone(result.tombstone.artifact)
            self.assertEqual("PublicationNotCommitted", result.tombstone.error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
