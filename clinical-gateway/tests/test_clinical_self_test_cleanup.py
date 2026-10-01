"""Cleanup is required before the synthetic self-test may report success."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import owned_agent_clinical_control as oac


class ClinicalSelfTestCleanupTests(unittest.TestCase):
    def temporary(self, cleanup=None):
        return SimpleNamespace(name="fixture-synthetic-workspace", cleanup=cleanup or Mock())

    def test_strict_cleanup_is_required_and_success_is_verified(self):
        temporary = self.temporary()
        with patch.object(oac.tempfile, "TemporaryDirectory", return_value=temporary) as create, \
             patch.object(oac.os, "lstat", side_effect=FileNotFoundError):
            with oac.clinical_self_test_workspace() as directory:
                self.assertEqual(directory, temporary.name)
            create.assert_called_once_with(prefix="owned-agent-clinical-self-test-")
            temporary.cleanup.assert_called_once_with()

    def test_transient_permission_error_is_retried_without_ignoring_it(self):
        temporary = self.temporary(Mock(side_effect=[PermissionError("fixture lock"), None]))
        with patch.object(oac.tempfile, "TemporaryDirectory", return_value=temporary), \
             patch.object(oac.time, "monotonic", side_effect=[0.0, 0.1]), \
             patch.object(oac.time, "sleep") as sleep, \
             patch.object(oac.os, "lstat", side_effect=FileNotFoundError):
            with oac.clinical_self_test_workspace():
                pass
            self.assertEqual(temporary.cleanup.call_count, 2)
            sleep.assert_called_once_with(0.1)

    def test_persistent_permission_error_fails_with_bounded_attempts(self):
        temporary = self.temporary(Mock(side_effect=PermissionError("fixture lock")))
        with patch.object(oac.tempfile, "TemporaryDirectory", return_value=temporary), \
             patch.object(oac.time, "monotonic", side_effect=[0.0, 1.0, 6.0]), \
             patch.object(oac.time, "sleep") as sleep:
            with self.assertRaises(oac.ControlError) as error:
                with oac.clinical_self_test_workspace():
                    pass
            self.assertEqual(error.exception.code, "CLINICAL_SELF_TEST_CLEANUP_FAILED")
            self.assertEqual(temporary.cleanup.call_count, 2)
            sleep.assert_called_once_with(0.1)

    def test_other_filesystem_error_is_not_retried_or_admitted(self):
        temporary = self.temporary(Mock(side_effect=OSError("fixture failure")))
        with patch.object(oac.tempfile, "TemporaryDirectory", return_value=temporary), \
             patch.object(oac.time, "sleep") as sleep:
            with self.assertRaises(oac.ControlError) as error:
                with oac.clinical_self_test_workspace():
                    pass
            self.assertEqual(error.exception.code, "CLINICAL_SELF_TEST_CLEANUP_FAILED")
            self.assertEqual(temporary.cleanup.call_count, 1)
            sleep.assert_not_called()

    def test_cleanup_return_does_not_prove_the_directory_was_removed(self):
        temporary = self.temporary()
        with patch.object(oac.tempfile, "TemporaryDirectory", return_value=temporary), \
             patch.object(oac.os, "lstat", return_value=object()):
            with self.assertRaises(oac.ControlError) as error:
                with oac.clinical_self_test_workspace():
                    pass
            self.assertEqual(error.exception.code, "CLINICAL_SELF_TEST_CLEANUP_FAILED")

    def test_unreadable_path_is_not_proof_of_removal(self):
        temporary = self.temporary()
        with patch.object(oac.tempfile, "TemporaryDirectory", return_value=temporary), \
             patch.object(oac.os, "lstat", side_effect=PermissionError("fixture unreadable")):
            with self.assertRaises(oac.ControlError) as error:
                with oac.clinical_self_test_workspace():
                    pass
            self.assertEqual(error.exception.code, "CLINICAL_SELF_TEST_CLEANUP_FAILED")

    def test_other_path_readback_error_is_not_proof_of_removal(self):
        temporary = self.temporary()
        with patch.object(oac.tempfile, "TemporaryDirectory", return_value=temporary), \
             patch.object(oac.os, "lstat", side_effect=OSError("fixture unavailable")):
            with self.assertRaises(oac.ControlError) as error:
                with oac.clinical_self_test_workspace():
                    pass
            self.assertEqual(error.exception.code, "CLINICAL_SELF_TEST_CLEANUP_FAILED")

    def test_original_test_failure_is_preserved_when_cleanup_succeeds(self):
        temporary = self.temporary()
        with patch.object(oac.tempfile, "TemporaryDirectory", return_value=temporary), \
             patch.object(oac.os, "lstat", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(ValueError, "original fixture failure"):
                with oac.clinical_self_test_workspace():
                    raise ValueError("original fixture failure")
            temporary.cleanup.assert_called_once_with()

    def test_real_synthetic_lifecycle_removes_state_before_success(self):
        observed = []
        original = oac.clinical_self_test_workspace

        @contextmanager
        def tracked_workspace():
            with original() as directory:
                observed.append(Path(directory))
                yield directory

        with patch.object(oac, "clinical_self_test_workspace", tracked_workspace):
            result = oac.clinical_self_test()
        self.assertIs(result["ok"], True)
        self.assertIs(result["disposable_state_removed"], True)
        self.assertIs(result["clinical_use_authorized"], False)
        self.assertIs(result["direct_device_transport"], False)
        self.assertEqual(len(observed), 1)
        with self.assertRaises(FileNotFoundError):
            observed[0].lstat()


if __name__ == "__main__":
    unittest.main()
