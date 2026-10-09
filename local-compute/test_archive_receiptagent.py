"""Tiny synthetic byte-preservation tests; no model import, data, or training."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import archive_receiptagent as target


class ArchiveRunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.results = self.root / "results"
        self.run = self.results / "synthetic-run"
        self.adapter = self.run / "adapter"
        self.adapter.mkdir(parents=True)
        self.archive = self.root / "private-archive"
        self.archive.mkdir()
        self.request_path = self.root / "private-review.json"
        self.enterContext(patch.object(target, "LOCAL_RESULTS_ROOT", self.results))
        self.enterContext(patch.object(target.shutil, "disk_usage",
                                return_value=Mock(free=4 * 1024**3)))
        (self.adapter / "adapter_model.safetensors").write_bytes(b"synthetic adapter bytes")
        (self.adapter / "adapter_config.json").write_bytes(b'{"synthetic":true}')
        self.write_report_and_request()

    def write_report_and_request(self, *, state="MEASURED_LOCAL_CONTINUATION_COMPLETED",
                                 rights_status="APPROVED_FOR_PRIVATE_ARCHIVE"):
        entries = []
        for path in sorted(self.adapter.iterdir()):
            raw = path.read_bytes()
            entries.append({"path": path.name, "bytes": len(raw), "sha256": target._sha(raw)})
        report = {
            "schema": target.REPORT_SCHEMA, "state": state,
            "durable_report_written": True, "publication_eligible": False,
            "autonomy_eligible": False, "parent_artifacts_unchanged": True,
            "candidate_reloaded_and_generation_tested": False,
            "provider_cost_usd": 0, "candidate_id": self.run.name,
            "source_revision": "a" * 40, "runner_sha256": "b" * 64,
            "base": {"repo_id": target.BASE_REPO,
                     "revision": target.BASE_REVISION,
                     "manifest_sha256": target.BASE_MANIFEST_SHA256},
            "adapter_parent": {"repo_id": target.PARENT_REPO,
                               "revision": target.PARENT_REVISION,
                               "manifest_sha256": target.PARENT_MANIFEST_SHA256,
                               "files": {"adapter_model.safetensors": {
                                   "sha256": target.PARENT_WEIGHT_SHA256}}},
            "curriculum_hashes": {f"synthetic-{i}": f"{i}" * 64 for i in range(6)},
            "candidate_files": entries,
        }
        report["report_sha256"] = target._sha(target._canonical(report))
        report_bytes = (json.dumps(report, indent=2) + "\n").encode()
        (self.run / "training-report.json").write_bytes(report_bytes)
        request = {
            "schema": target.REQUEST_SCHEMA,
            "report_sha256": target._sha(report_bytes),
            "source_revision": report["source_revision"],
            "runner_sha256": report["runner_sha256"],
            "source_provenance_sha256": "c" * 64,
            "rights_review_sha256": "d" * 64,
            "leakage_review_sha256": "e" * 64,
            "owner_release_sha256": "f" * 64,
            "rights_status": rights_status,
            "leakage_status": "PASS",
            "owner_release_status": "RELEASED_FOR_PRIVATE_ARCHIVE",
        }
        self.request_path.write_bytes(target._canonical(request))
        return report, request

    def destination(self, request):
        return (self.archive / "native-receiptagent" /
                ("candidate-" + request["report_sha256"]))

    def test_local_copy_and_receipt_are_verified_but_remote_is_not(self):
        request = target._strict_json(self.request_path.read_bytes(), target.MAX_REQUEST_BYTES)
        receipt = target.archive_run(self.run, self.request_path, self.archive)
        destination = self.destination(request)
        self.assertEqual(receipt["state"], "LOCAL_COPY_VERIFIED_REMOTE_UNVERIFIED")
        self.assertTrue(receipt["local_copy_verified"])
        self.assertFalse(receipt["remote_restore_verified"])
        self.assertFalse(receipt["publication_eligible"])
        self.assertEqual(receipt["checkpoint_completeness"], "NOT_RESTARTABLE")
        self.assertEqual((destination / "adapter" / "adapter_model.safetensors").read_bytes(),
                         (self.adapter / "adapter_model.safetensors").read_bytes())
        self.assertEqual(target._strict_json((destination / "archive-copy-receipt.json").read_bytes(),
                                              target.MAX_REQUEST_BYTES), receipt)
        self.assertFalse(list(destination.parent.glob(".pending-*")))
        with self.assertRaisesRegex(target.GateError, "ARCHIVE_DESTINATION_EXISTS"):
            target.archive_run(self.run, self.request_path, self.archive)

    def test_interrupted_pending_copy_retained_and_retry_publishes_final(self):
        request = target._strict_json(self.request_path.read_bytes(), target.MAX_REQUEST_BYTES)
        destination = self.destination(request)
        original_copy = target._copy_via_plain_stage
        interrupted = False

        def fail_once(source, target_path, item, stage, archive_root):
            nonlocal interrupted
            if source == self.adapter / "adapter_model.safetensors" and not interrupted:
                interrupted = True
                raise OSError("synthetic copy interruption")
            return original_copy(source, target_path, item, stage, archive_root)

        with patch.object(target, "_copy_via_plain_stage", side_effect=fail_once):
            with self.assertRaises(OSError):
                target.archive_run(self.run, self.request_path, self.archive)
        pending = list(destination.parent.glob(".pending-*"))
        self.assertEqual(len(pending), 1)
        self.assertFalse(destination.exists())
        self.assertFalse((pending[0] / "archive-copy-receipt.json").exists())
        receipt = target.archive_run(self.run, self.request_path, self.archive)
        self.assertTrue((destination / "archive-copy-receipt.json").is_file())
        self.assertEqual(receipt["state"], "LOCAL_COPY_VERIFIED_REMOTE_UNVERIFIED")
        self.assertTrue(pending[0].exists())  # Failed evidence is never auto-deleted.

    def test_report_and_candidate_tampering_fail_before_new_destination(self):
        request = target._strict_json(self.request_path.read_bytes(), target.MAX_REQUEST_BYTES)
        (self.run / "training-report.json").write_bytes(b"tampered")
        with self.assertRaisesRegex(target.GateError, "REPORT_BYTES_MISMATCH"):
            target.archive_run(self.run, self.request_path, self.archive)
        self.assertFalse(self.destination(request).exists())
        self.write_report_and_request()
        (self.adapter / "adapter_model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(target.GateError, "FILE_SIZE_MISMATCH"):
            target.archive_run(self.run, self.request_path, self.archive)
        self.assertFalse((self.archive / "native-receiptagent").exists())

    def test_failed_or_unreleased_run_and_extra_candidate_are_refused(self):
        self.write_report_and_request(state="FAILED_CLOSED")
        with self.assertRaisesRegex(target.GateError, "TRAINING_REPORT_NOT_COMPLETED"):
            target.archive_run(self.run, self.request_path, self.archive)
        self.write_report_and_request(rights_status="PENDING")
        with self.assertRaisesRegex(target.GateError, "ARCHIVE_REVIEW_GATE_NOT_RELEASED"):
            target.archive_run(self.run, self.request_path, self.archive)
        self.write_report_and_request()
        (self.adapter / "unexpected.py").write_bytes(b"print('synthetic')")
        with self.assertRaisesRegex(target.GateError, "CANDIDATE_FILE_TYPE_REJECTED"):
            target.archive_run(self.run, self.request_path, self.archive)

    def test_empty_directory_and_sensitive_metadata_are_not_copied(self):
        (self.adapter / "undeclared-folder").mkdir()
        with self.assertRaisesRegex(target.GateError, "CANDIDATE_INVENTORY_MISMATCH"):
            target.archive_run(self.run, self.request_path, self.archive)
        (self.adapter / "undeclared-folder").rmdir()
        (self.adapter / "adapter_config.json").write_bytes(
            b'{"contact":"person@example.com"}')
        self.write_report_and_request()
        with self.assertRaisesRegex(target.GateError,
                                    "CANDIDATE_CONTENT_MAY_CONTAIN_PRIVATE_DATA"):
            target.archive_run(self.run, self.request_path, self.archive)

    def test_unrecognized_report_field_is_not_copied(self):
        report, request = self.write_report_and_request()
        report.pop("report_sha256")
        report["private_notes"] = "opaque-sensitive-value"
        report["report_sha256"] = target._sha(target._canonical(report))
        raw = (json.dumps(report, indent=2) + "\n").encode()
        (self.run / "training-report.json").write_bytes(raw)
        request["report_sha256"] = target._sha(raw)
        self.request_path.write_bytes(target._canonical(request))
        with self.assertRaisesRegex(target.GateError, "TRAINING_REPORT_UNKNOWN_FIELD"):
            target.archive_run(self.run, self.request_path, self.archive)
        self.assertFalse((self.archive / "native-receiptagent").exists())

    def test_unmaterialized_candidate_is_refused_before_payload_read(self):
        original = target._entry_metadata

        def metadata(path):
            attrs, tag = original(path)
            if path == self.adapter / "adapter_model.safetensors":
                return attrs | 0x1000, tag
            return attrs, tag

        with patch.object(target, "_entry_metadata", side_effect=metadata):
            with self.assertRaisesRegex(target.GateError, "PATH_NOT_FULLY_MATERIALIZED"):
                target.archive_run(self.run, self.request_path, self.archive)
        self.assertFalse((self.archive / "native-receiptagent").exists())

    def test_only_ready_cloud_reparse_is_allowed_in_archive_context(self):
        original = target._entry_metadata

        def cloud(path):
            attrs, tag = original(path)
            if path == self.archive:
                return attrs | 0x400, 0x9000701a
            return attrs, tag

        with patch.object(target, "_entry_metadata", side_effect=cloud), \
             patch.object(target, "_placeholder_state", return_value=1):
            target._safe_existing(self.archive, allow_hydrated_cloud=True)
            with self.assertRaisesRegex(target.GateError, "SYMLINK_OR_REPARSE_REJECTED"):
                target._safe_existing(self.archive)
        with patch.object(target, "_entry_metadata", side_effect=lambda p: (
                (original(p)[0] | 0x400, 0xa0000003) if p == self.archive else original(p))), \
             patch.object(target, "_placeholder_state", return_value=1):
            with self.assertRaisesRegex(target.GateError, "SYMLINK_OR_REPARSE_REJECTED"):
                target._safe_existing(self.archive, allow_hydrated_cloud=True)

    def test_candidate_reparse_is_refused_before_copy(self):
        original = target._entry_metadata
        source = self.adapter / "adapter_model.safetensors"

        def reparse(path):
            attrs, tag = original(path)
            if path == source:
                return attrs | 0x400, 0xa0000003  # Junction/name surrogate.
            return attrs, tag

        with patch.object(target, "_entry_metadata", side_effect=reparse):
            with self.assertRaisesRegex(target.GateError, "SYMLINK_OR_REPARSE_REJECTED"):
                target.archive_run(self.run, self.request_path, self.archive)
        self.assertFalse((self.archive / "native-receiptagent").exists())

    def test_mutation_during_stage_never_reaches_archive(self):
        request = target._strict_json(self.request_path.read_bytes(), target.MAX_REQUEST_BYTES)
        source = self.adapter / "adapter_model.safetensors"
        changed = b"person@example.com".ljust(source.stat().st_size, b"x")
        original_copy = target._copy_verified

        def mutate_before_stage_copy(source_path, destination, item, *, destination_cloud):
            if source_path == source and not destination_cloud:
                source.write_bytes(changed)
            return original_copy(source_path, destination, item,
                                 destination_cloud=destination_cloud)

        with patch.object(target, "_copy_verified", side_effect=mutate_before_stage_copy):
            with self.assertRaisesRegex(target.GateError, "CANDIDATE_BYTES_MISMATCH"):
                target.archive_run(self.run, self.request_path, self.archive)
        destination = self.destination(request)
        self.assertFalse((destination / "adapter" / "adapter_model.safetensors").exists())
        self.assertFalse((destination / "archive-copy-receipt.json").exists())

    def test_lexical_root_traversal_is_rejected_before_resolution(self):
        disguised = self.archive / "unused" / ".."
        with self.assertRaisesRegex(target.GateError, "ARCHIVE_PATH_TRAVERSAL_REJECTED"):
            target.archive_run(self.run, self.request_path, disguised)

    def test_source_symlink_is_refused(self):
        source = self.adapter / "adapter_config.json"
        source.unlink()
        try:
            source.symlink_to(self.adapter / "adapter_model.safetensors")
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation privilege unavailable")
        with self.assertRaisesRegex(target.GateError, "SYMLINK_OR_REPARSE_REJECTED"):
            target.archive_run(self.run, self.request_path, self.archive)


if __name__ == "__main__":
    unittest.main()
