# SPDX-License-Identifier: Apache-2.0
"""Forensic evidence durability only; mocked blobs, no owner data, GPU or APIs."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gguf_forensics as m


def inspected(expected):
    # Synthetic parsed result, not a real model or quality measurement.
    ts = [{"name_sha256": "a" * 64, "shape": [1], "type": "F16",
           "bytes": 2, "data_sha256": "b" * 64}]
    return {"container_sha256": expected, "tensor_set_sha256": m.digest(m.canonical(ts)),
            "tensors": ts, "matches_historical_from_digest": True,
            "metadata_fingerprints": {name: {"sha256": None, "fields": 0}
                for name in ("all", "tokenizer", "chat_template", "architecture")},
            "tensor_values_or_health_checked": False}


class EvidenceTests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / ".ollama" / "models" / "blobs").mkdir(parents=True)
            fake_os = SimpleNamespace(name="nt", environ={"COMPUTERNAME": "BETTERWITHAGE"}, fsync=os.fsync)
            with patch.object(m, "os", fake_os), patch.object(m.Path, "home", return_value=home), \
                 patch.object(m, "require_local_windows_path") as local, \
                 contextlib.redirect_stdout(io.StringIO()):
                yield home, local

    def checkpoints(self, home):
        paths = sorted(home.glob("szl-gguf-forensics-*/checkpoint-*.json"))
        return [json.loads(p.read_text(encoding="utf-8")) for p in paths]

    def test_read_intent_is_persisted_before_first_blob(self):
        with self.fixture_home() as (home, local):
            def interrupted_read(*args):
                rows = self.checkpoints(home)
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[-1]["phase"], "BLOB_READ_INTENT")
                self.assertEqual(rows[-1]["report"]["active_model"], next(iter(m.BLOBS)))
                self.assertEqual(rows[-1]["report"]["expected_historical_blobs"], m.BLOBS)
                self.assertEqual(local.call_count, 2)
                raise KeyboardInterrupt()
            with patch.object(m, "inspect_file", side_effect=interrupted_read):
                with self.assertRaises(KeyboardInterrupt):
                    m.main(["--inspect-blobs"])

    def test_first_result_survives_interrupt_during_second_blob(self):
        names = list(m.BLOBS)
        first = inspected(m.BLOBS[names[0]])
        with self.fixture_home() as (home, _):
            with patch.object(m, "inspect_file", side_effect=[first, KeyboardInterrupt()]) as read:
                with self.assertRaises(KeyboardInterrupt):
                    m.main(["--inspect-blobs"])
            self.assertEqual(read.call_count, 2)
            rows = self.checkpoints(home)
            self.assertEqual(len(rows), 4)
            report = rows[-1]["report"]
            self.assertEqual(report["models"][names[0]]["container_sha256"], m.BLOBS[names[0]])
            self.assertNotIn(names[1], report["models"])
            self.assertEqual(report["active_model"], names[1])
            self.assertEqual(report["state"], "INCOMPLETE")
            self.assertIsNone(report["comparison"])
            self.assertFalse(list(home.glob("szl-gguf-forensics-*/gguf-forensics.json")))

    def test_success_keeps_five_checkpoints_and_original_comparison(self):
        results = [inspected(x) for x in m.BLOBS.values()]
        expected = m.compare(*results)
        with self.fixture_home() as (home, _):
            with patch.object(m, "inspect_file", side_effect=results) as read:
                self.assertEqual(m.main(["--inspect-blobs"]), 0)
            self.assertEqual(read.call_count, 2)
            rows = self.checkpoints(home)
            self.assertEqual([r["sequence"] for r in rows], list(range(5)))
            self.assertEqual([r["phase"] for r in rows], ["INSPECTION_PLANNED", "BLOB_READ_INTENT",
                             "BLOB_RESULT_RECORDED", "BLOB_READ_INTENT", "BLOB_RESULT_RECORDED"])
            final = json.loads(next(home.glob("szl-gguf-forensics-*/gguf-forensics.json")).read_text())
            self.assertEqual(final["comparison"], expected)
            self.assertEqual(final["state"], "FORENSICS_COMPLETED_NOT_MODEL_QUALIFICATION")
            self.assertFalse(final["comparison"]["model_qualified"])
            for row in rows:
                self.assertEqual(row["report"]["state"], "INCOMPLETE")
                self.assertFalse(row["automatic_resume_authorized"])
                self.assertFalse(row["report"]["paired_attempts_touched"])
                self.assertFalse(row["report"]["inference"])
            self.assertFalse(final["automatic_resume_authorized"])

    def test_fsync_failure_before_reads_stops_without_retry(self):
        with self.fixture_home() as (home, _):
            with patch.object(m.os, "fsync", side_effect=OSError("PRIVATE_FSYNC_DETAIL")) as sync, \
                 patch.object(m, "inspect_file") as read:
                with self.assertRaises(OSError):
                    m.main(["--inspect-blobs"])
            self.assertEqual(sync.call_count, 1)
            read.assert_not_called()
            # A failed durability acknowledgement is preserved, never deleted.
            self.assertEqual(len(list(home.glob("szl-gguf-forensics-*/checkpoint-000.json"))), 1)

    def test_failed_first_result_checkpoint_blocks_second_blob(self):
        with self.fixture_home() as (home, _):
            with patch.object(m.os, "fsync", side_effect=[None, None, OSError("disk")]), \
                 patch.object(m, "inspect_file", return_value=inspected(next(iter(m.BLOBS.values())))) as read:
                with self.assertRaises(OSError):
                    m.main(["--inspect-blobs"])
            self.assertEqual(read.call_count, 1)
            self.assertEqual(len(self.checkpoints(home)), 3)
            self.assertFalse(list(home.glob("szl-gguf-forensics-*/gguf-forensics.json")))

    def test_final_failure_preserves_completed_per_blob_evidence(self):
        with self.fixture_home() as (home, _):
            real_write = m.write_evidence
            def fail_final(path, value):
                if path.name == "gguf-forensics.json":
                    raise OSError("disk full")
                return real_write(path, value)
            with patch.object(m, "write_evidence", side_effect=fail_final), \
                 patch.object(m, "inspect_file", side_effect=[inspected(x) for x in m.BLOBS.values()]) as read:
                with self.assertRaises(OSError):
                    m.main(["--inspect-blobs"])
            self.assertEqual(read.call_count, 2)
            rows = self.checkpoints(home)
            self.assertEqual(len(rows[-1]["report"]["models"]), 2)
            self.assertIsNone(rows[-1]["report"]["comparison"])
            self.assertFalse(list(home.glob("szl-gguf-forensics-*/gguf-forensics.json")))

    def test_rejected_input_records_redacted_error_not_quality_or_absence(self):
        with self.fixture_home() as (home, _):
            with patch.object(m, "inspect_file", side_effect=OSError("PRIVATE_SOURCE_PATH")):
                self.assertEqual(m.main(["--inspect-blobs"]), 1)
            rows = self.checkpoints(home)
            self.assertNotIn("PRIVATE_SOURCE_PATH", json.dumps(rows))
            self.assertIsNone(rows[-1]["report"]["comparison"])
            for model in rows[-1]["report"]["models"].values():
                self.assertEqual(model["state"], "UNAVAILABLE_OR_REJECTED")
                self.assertEqual(model["error_code"], "OSError")

    def test_plan_does_not_create_checkpoints_or_touch_paths(self):
        with patch.object(m, "write_evidence") as write, patch.object(m, "inspect_file") as read, \
             patch.object(m.Path, "home") as home, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main([]), 0)
        write.assert_not_called()
        read.assert_not_called()
        home.assert_not_called()

    def test_wrong_host_stops_before_checkpoint_or_filesystem(self):
        fake_os = SimpleNamespace(name="nt", environ={"COMPUTERNAME": "NOT_AUTHORIZED"})
        with patch.object(m, "os", fake_os), patch.object(m, "write_evidence") as write, \
             patch.object(m.Path, "home") as home:
            with self.assertRaisesRegex(m.ForensicsError, "BETTERWITHAGE_WINDOWS_ONLY"):
                m.main(["--inspect-blobs"])
        write.assert_not_called()
        home.assert_not_called()

    def test_exclusive_writer_preserves_existing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"
            path.write_bytes(b"preserve-original")
            with self.assertRaises(FileExistsError):
                m.write_evidence(path, {"new": True})
            self.assertEqual(path.read_bytes(), b"preserve-original")

    def test_invalid_serialization_opens_no_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"
            with self.assertRaises(ValueError):
                m.write_evidence(path, {"bad": float("nan")})
            self.assertFalse(path.exists())

    def test_linked_output_refused_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"
            with patch.object(m, "plain_path", return_value=False):
                with self.assertRaisesRegex(m.ForensicsError, "LINKED_OUTPUT_REFUSED"):
                    m.write_evidence(path, {})
            self.assertFalse(path.exists())

    def test_buffer_flushed_before_fsync(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"
            def sync(fd):
                self.assertEqual(json.loads(path.read_text()), {"value": 1})
                os.fstat(fd)
            with patch.object(m.os, "fsync", side_effect=sync) as flush:
                m.write_evidence(path, {"value": 1})
            flush.assert_called_once()
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_checkpoint_collision_never_truncates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m.checkpoint(root, {"state": "INCOMPLETE"}, 0, "INSPECTION_PLANNED")
            original = (root / "checkpoint-000.json").read_bytes()
            with self.assertRaises(FileExistsError):
                m.checkpoint(root, {"state": "changed"}, 0, "INSPECTION_PLANNED")
            self.assertEqual((root / "checkpoint-000.json").read_bytes(), original)

    def test_unexpected_parser_failure_retains_intent_without_continuing(self):
        with self.fixture_home() as (home, _):
            with patch.object(m, "inspect_file", side_effect=RuntimeError("unexpected")) as read:
                with self.assertRaises(RuntimeError):
                    m.main(["--inspect-blobs"])
            self.assertEqual(read.call_count, 1)
            self.assertEqual(self.checkpoints(home)[-1]["phase"], "BLOB_READ_INTENT")
            self.assertFalse(list(home.glob("szl-gguf-forensics-*/gguf-forensics.json")))


if __name__ == "__main__":
    unittest.main()
