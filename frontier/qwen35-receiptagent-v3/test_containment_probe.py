from __future__ import annotations

import argparse
import errno
import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


HERE = pathlib.Path(__file__).resolve().parent


def load_probe():
    spec = importlib.util.spec_from_file_location(
        "v3_containment_probe_under_test", HERE / "containment_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = load_probe()


class ContainmentProbeTests(unittest.TestCase):
    def test_expected_unreadability_errors_are_accepted(self):
        for error_number in probe.EXPECTED_UNREADABLE_ERRNOS:
            with self.subTest(error_number=error_number), mock.patch.object(
                probe.os,
                "open",
                side_effect=OSError(error_number, "expected isolation denial"),
            ):
                probe.assert_unreadable(pathlib.Path("/hidden/secret"))

    def test_unexpected_read_errors_fail_closed(self):
        for error_number in (errno.EINTR, errno.EMFILE, errno.EIO, errno.EEXIST):
            with self.subTest(error_number=error_number), mock.patch.object(
                probe.os,
                "open",
                side_effect=OSError(error_number, "unexpected read failure"),
            ):
                with self.assertRaisesRegex(probe.ProbeError, "unexpectedly"):
                    probe.assert_unreadable(pathlib.Path("/hidden/secret"))

    def test_expected_unwritability_errors_are_accepted(self):
        for error_number in probe.EXPECTED_UNWRITABLE_ERRNOS:
            with self.subTest(error_number=error_number), mock.patch.object(
                probe.os,
                "open",
                side_effect=OSError(error_number, "expected isolation denial"),
            ):
                probe.assert_unwritable(pathlib.Path("/readonly/.probe-write"))

    def test_unexpected_write_errors_and_collisions_fail_closed(self):
        for error_number in (errno.EINTR, errno.EMFILE, errno.EIO, errno.EEXIST):
            with self.subTest(error_number=error_number), mock.patch.object(
                probe.os,
                "open",
                side_effect=OSError(error_number, "unexpected write failure"),
            ):
                with self.assertRaisesRegex(probe.ProbeError, "unexpectedly"):
                    probe.assert_unwritable(pathlib.Path("/readonly/.probe-write"))

    def test_readable_forbidden_path_fails_without_reading_content(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = pathlib.Path(directory) / "secret"
            secret.write_bytes(b"never-return-this-content")
            with self.assertRaisesRegex(probe.ProbeError, "remained readable"):
                probe.assert_unreadable(secret)

    def test_exact_training_only_bundle_passes_structural_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            input_dir = root / "input"
            cache_dir = root / "cache"
            venv_dir = root / "venv"
            model_repository = root / "model"
            input_dir.mkdir()
            cache_dir.mkdir()
            (venv_dir / "bin").mkdir(parents=True)
            (venv_dir / "bin" / "python").write_bytes(b"runtime")
            revision = "a" * 40
            (model_repository / "snapshots" / revision).mkdir(parents=True)
            for filename in probe.EXPECTED_INPUT_FILES:
                (input_dir / filename).write_bytes(b"source")
            args = argparse.Namespace(
                input_dir=input_dir,
                cache_dir=cache_dir,
                venv_dir=venv_dir,
                model_repository=model_repository,
                model_revision=revision,
                forbidden_read=[root / "absent-secret"],
                report=cache_dir / "containment-probe.json",
            )
            with (
                mock.patch.object(probe, "assert_unreadable"),
                mock.patch.object(probe, "assert_unwritable"),
            ):
                report = probe.perform_probe(args)
            self.assertEqual("PASS", report["state"])
            self.assertEqual(3, report["forbiddenHostReadTargetCount"])
            self.assertTrue(report["fixedHostDecoysHidden"])
            self.assertTrue(report["rootWriteDenied"])
            self.assertTrue(report["workerMountRootWriteDenied"])
            self.assertFalse(report["secretContentRead"])


    def test_fixed_host_decoys_are_always_checked_without_content_reads(self):
        observed: list[pathlib.Path] = []

        def record(path: pathlib.Path) -> None:
            observed.append(path)

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            input_dir = root / "input"
            cache_dir = root / "cache"
            venv_dir = root / "venv"
            model_repository = root / "model"
            input_dir.mkdir()
            cache_dir.mkdir()
            (venv_dir / "bin").mkdir(parents=True)
            (venv_dir / "bin" / "python").write_bytes(b"runtime")
            revision = "a" * 40
            (model_repository / "snapshots" / revision).mkdir(parents=True)
            for filename in probe.EXPECTED_INPUT_FILES:
                (input_dir / filename).write_bytes(b"source")
            args = argparse.Namespace(
                input_dir=input_dir,
                cache_dir=cache_dir,
                venv_dir=venv_dir,
                model_repository=model_repository,
                model_revision=revision,
                forbidden_read=[root / "credential-canary", root / "dev", root / "test"],
                report=cache_dir / "containment-probe.json",
            )
            with (
                mock.patch.object(probe, "assert_unreadable", side_effect=record),
                mock.patch.object(probe, "assert_unwritable"),
            ):
                report = probe.perform_probe(args)
        self.assertEqual(
            [*args.forbidden_read, *probe.FIXED_HOST_DECOYS], observed
        )
        self.assertEqual(5, report["forbiddenHostReadTargetCount"])

    def test_heldout_or_extra_file_in_input_bundle_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            input_dir = root / "input"
            cache_dir = root / "cache"
            venv_dir = root / "venv"
            model_repository = root / "model"
            input_dir.mkdir()
            cache_dir.mkdir()
            venv_dir.mkdir()
            model_repository.mkdir()
            for filename in probe.EXPECTED_INPUT_FILES:
                (input_dir / filename).write_bytes(b"source")
            (input_dir / "test.jsonl").write_bytes(b"heldout")
            args = argparse.Namespace(
                input_dir=input_dir,
                cache_dir=cache_dir,
                venv_dir=venv_dir,
                model_repository=model_repository,
            )
            with self.assertRaisesRegex(probe.ProbeError, "unapproved"):
                # Only the first structural branch is relevant to this test.
                probe.perform_probe(args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
