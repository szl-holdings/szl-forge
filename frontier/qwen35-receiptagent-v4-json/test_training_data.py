# SPDX-License-Identifier: Apache-2.0
"""SIMULATED pure-converter tests; fixtures are not admitted training evidence.

These tests construct their own rows and snapshots. They never open actual
train/dev/held-out curriculum, call a doctrine kernel, invoke a model or sign.
Hash patches below allow independent unit fixtures only, never a launch path.
"""
from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import hashlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curriculum_admission as admission
import json_contract as contract
import training_data
from test_curriculum_admission import fixture_manifest, fixture_row, serialize


class TrainingMessagesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [fixture_row(family, kind, index)
                    for family in sorted(admission.TRAIN_FAMILIES)
                    for kind in contract.CLAIMS for index in range(30)]
        cls.raw = serialize(cls.rows)
        cls.manifest = fixture_manifest(cls.raw)
        cls.manifest_raw = (contract.canonical_json(cls.manifest) + "\n").encode("utf-8")
        # Larger than contract.MAX_BYTES, like a full 360-receipt report. It is
        # opaque provenance here, not parsed as one contract document or used
        # to substitute for fresh admission/kernel execution.
        cls.report_raw = b'{"SIMULATED_opaque_provenance":"' + b"x" * 220_000 + b'"}\n'

    def snapshot(self, *, manifest_raw=None, train_raw=None, report_raw=None):
        return admission.TrainOnlySnapshot(
            self.manifest_raw if manifest_raw is None else manifest_raw,
            self.raw if train_raw is None else train_raw,
            self.report_raw if report_raw is None else report_raw,
        )

    def convert(self, snapshot, *, expected_manifest_raw=None):
        raw = self.manifest_raw if expected_manifest_raw is None else expected_manifest_raw
        with patch.object(admission, "MANIFEST_SHA256", hashlib.sha256(raw).hexdigest()):
            return training_data.training_messages(snapshot)

    def reject_rebound_fixture(self, raw):
        manifest_raw = (contract.canonical_json(fixture_manifest(raw)) + "\n").encode("utf-8")
        with self.assertRaises(admission.AdmissionError):
            self.convert(self.snapshot(manifest_raw=manifest_raw, train_raw=raw),
                         expected_manifest_raw=manifest_raw)

    def test_exact_strings_and_order_survive_without_source_io_or_gates(self):
        with patch("builtins.open", side_effect=AssertionError("converter opened a path")), \
             patch.object(io, "open", side_effect=AssertionError("converter opened a path")), \
             patch.object(admission, "_read_once", side_effect=AssertionError("source reread")), \
             patch.object(admission, "validate_manifest", side_effect=AssertionError("source reread")), \
             patch.object(admission, "validate_train_bytes", side_effect=AssertionError("schema reread")), \
             patch.object(admission, "run_gates", side_effect=AssertionError("kernel rerun")):
            messages = self.convert(self.snapshot())
        self.assertEqual(len(messages), 360)
        expected = tuple(tuple((message["role"], message["content"])
                               for message in row["messages"]) for row in self.rows)
        self.assertEqual(messages, expected)
        self.assertTrue(all(type(row) is tuple and len(row) == 3 for row in messages))
        self.assertTrue(all(type(message) is tuple for row in messages for message in row))

    def test_snapshot_and_output_do_not_retain_mutable_message_references(self):
        snapshot = self.snapshot()
        messages = self.convert(snapshot)
        with self.assertRaises(FrozenInstanceError):
            snapshot.train_bytes = b"changed"
        with self.assertRaises(TypeError):
            messages[0][0] = ("system", "changed")
        changed = copy.deepcopy(self.rows)
        changed[0]["messages"][2]["content"] = "changed local fixture object"
        self.assertEqual(messages[0][2][1], self.rows[0]["messages"][2]["content"])

    def test_report_is_opaque_provenance_not_a_contract_document_or_permit(self):
        self.assertGreater(len(self.report_raw), contract.MAX_BYTES)
        messages = self.convert(self.snapshot())
        self.assertEqual(len(messages), 360)
        self.assertIs(type(messages), tuple)
        self.assertFalse(hasattr(messages, "training_eligible"))
        self.assertFalse(hasattr(messages, "execution_authority"))

    def test_only_exact_snapshot_type_is_accepted_not_serialized_reports_or_paths(self):
        class SnapshotSubclass(admission.TrainOnlySnapshot):
            pass

        for value in (self.raw, {"train_bytes": self.raw, "state": "VALID"},
                      Path("train.jsonl"), (self.manifest_raw, self.raw, self.report_raw),
                      SnapshotSubclass(self.manifest_raw, self.raw, self.report_raw)):
            with self.subTest(kind=type(value).__name__), self.assertRaises(admission.AdmissionError):
                training_data.training_messages(value)

    def test_buffers_are_nonempty_bounded_exact_bytes(self):
        class ByteSubclass(bytes):
            pass

        for fields in ((b"", self.raw, self.report_raw),
                       (self.manifest_raw, b"", self.report_raw),
                       (self.manifest_raw, self.raw, b""),
                       (b"x" * (contract.MAX_BYTES + 1), self.raw, self.report_raw),
                       (self.manifest_raw, b"x" * (admission.MAX_FILE_BYTES + 1), self.report_raw),
                       (self.manifest_raw, self.raw, b"x" * (admission.MAX_FILE_BYTES + 1)),
                       (ByteSubclass(self.manifest_raw), self.raw, self.report_raw),
                       (self.manifest_raw, bytearray(self.raw), self.report_raw)):
            with self.subTest(sizes=[len(value) for value in fields]), \
                 self.assertRaises(admission.AdmissionError):
                self.convert(admission.TrainOnlySnapshot(*fields))

    def test_manifest_and_train_hash_changes_are_rejected(self):
        for snapshot in (self.snapshot(manifest_raw=self.manifest_raw + b" "),
                         self.snapshot(train_raw=self.raw.replace(b"sample-0", b"sample-X", 1))):
            with self.assertRaises(admission.AdmissionError):
                self.convert(snapshot)

    def test_strict_manifest_even_in_synthetic_hash_rebound_fixtures(self):
        for raw in (b'{"schema":1,"schema":2}\n', b'{"seed":NaN}\n',
                    b'{"invalid":"\xff"}\n', b'[]\n', self.manifest_raw.rstrip(b"\n"),
                    b" " + self.manifest_raw):
            with self.subTest(raw=raw[:40]), self.assertRaises(admission.AdmissionError):
                self.convert(self.snapshot(manifest_raw=raw), expected_manifest_raw=raw)

    def test_manifest_authority_runtime_identity_and_exact_count_types_are_checked(self):
        for key, value in (("signature", "SIGNED"), ("runtime_binding", "READY"),
                           ("training_eligible", True), ("publication_eligible", 0),
                           ("execution_authority", True), ("seed", True),
                           ("candidate_id", "other"), ("contract_source_sha256", "a" * 64),
                           ("unexpected", "extra")):
            manifest = copy.deepcopy(self.manifest)
            manifest[key] = value
            raw = (contract.canonical_json(manifest) + "\n").encode("utf-8")
            with self.subTest(key=key), self.assertRaises(admission.AdmissionError):
                self.convert(self.snapshot(manifest_raw=raw), expected_manifest_raw=raw)
        for field, value in (("rows", True), ("path", "dev.jsonl"),
                             ("kind_counts", {kind: 120.0 for kind in contract.CLAIMS})):
            manifest = copy.deepcopy(self.manifest)
            manifest["splits"]["train"][field] = value
            raw = (contract.canonical_json(manifest) + "\n").encode("utf-8")
            with self.subTest(field=field), self.assertRaises(admission.AdmissionError):
                self.convert(self.snapshot(manifest_raw=raw), expected_manifest_raw=raw)

    def test_train_shape_metadata_roles_and_strings_fail_closed(self):
        mutations = (
            lambda row: row.update(split="dev"),
            lambda row: row.update(family=sorted(admission.HELDOUT_FAMILIES)[0]),
            lambda row: row.update(kind="UNKNOWN"),
            lambda row: row.update(unexpected="extra"),
            lambda row: row["messages"][0].update(content="different system prompt"),
            lambda row: row["messages"][1].update(role="assistant"),
            lambda row: row["messages"][2].update(content=12),
            lambda row: row["messages"][2].update(content=""),
            lambda row: row["messages"][2].update(unexpected="extra"),
            lambda row: row["messages"].pop(),
        )
        for mutate in mutations:
            rows = copy.deepcopy(self.rows)
            mutate(rows[0])
            self.reject_rebound_fixture(serialize(rows))

    def test_row_coverage_uniqueness_and_lf_bytes_fail_closed(self):
        duplicate = copy.deepcopy(self.rows)
        duplicate[1] = duplicate[0]
        changed_kind = copy.deepcopy(self.rows)
        changed_kind[0]["kind"] = "REFUSAL"
        for raw in (serialize(duplicate), serialize(changed_kind), serialize(self.rows[:-1]),
                    self.raw.rstrip(b"\n"), self.raw.replace(b"\n", b"\r\n"),
                    b" " + self.raw, b"\xff" + self.raw[1:]):
            self.reject_rebound_fixture(raw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
