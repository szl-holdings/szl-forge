# SPDX-License-Identifier: Apache-2.0
"""Offline crash/replay regressions; no owner device or live GPU access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
import szl_recovery as m
from test_szl_recovery import FakeClient, healthy_gpu


class OnceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.out = self.home / "szl-recovery-20260912-120000-aabbccdd"
        self.out.mkdir()

    def prior(self, value=None):
        p = self.home / "szl-recovery-20260912-110000-01234567"
        p.mkdir()
        if value is not None:
            m.write_new(p / "recovery-report.json", value)
        return p

    def test_identity_stable_across_output_and_release(self):
        first = m.experiment_identity()
        with patch.object(m, "VERSION", "different-implementation"):
            self.assertEqual(first, m.experiment_identity())
        with patch.dict(m.EXPECTED, {m.RECEIPT: "b" * 64}):
            self.assertNotEqual(first, m.experiment_identity())

    def test_claim_is_fsynced_and_retained(self):
        with patch.object(m.os, "fsync", wraps=m.os.fsync) as sync:
            result = m.claim_paired_attempt(self.home, self.out)
        sync.assert_called_once()
        saved = next((self.home / ".szl-recovery-attempts").glob("*.json"))
        self.assertEqual(json.loads(saved.read_bytes()), result)
        self.assertEqual(result["output_directory"], self.out.name)
        with self.assertRaisesRegex(m.GateError, "ALREADY_CLAIMED"):
            m.claim_paired_attempt(self.home, self.out)

    def test_failed_fsync_leaves_a_blocking_claim(self):
        with patch.object(m.os, "fsync", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                m.claim_paired_attempt(self.home, self.out)
        with self.assertRaisesRegex(m.GateError, "ALREADY_CLAIMED"):
            m.claim_paired_attempt(self.home, self.out)

    def test_corrupt_claim_is_never_reclaimed(self):
        root = self.home / ".szl-recovery-attempts"
        root.mkdir()
        (root / (m.experiment_identity() + ".json")).write_bytes(b"")
        with self.assertRaisesRegex(m.GateError, "ALREADY_CLAIMED"):
            m.claim_paired_attempt(self.home, self.out)

    def test_output_cannot_escape_home(self):
        with self.assertRaisesRegex(m.GateError, "OUTPUT_PATH"):
            m.claim_paired_attempt(self.home, self.home.parent)

    def test_linked_claim_root_refused(self):
        target = self.home / "target"; target.mkdir()
        try:
            (self.home / ".szl-recovery-attempts").symlink_to(target, target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation unavailable to this test identity")
        with self.assertRaisesRegex(m.GateError, "DIRECTORY_LINKED"):
            m.claim_paired_attempt(self.home, self.out)

    def test_prior_partial_or_complete_pair_blocks(self):
        p = self.prior()
        (p / "paired-schema-report.json").write_bytes(b"{}")
        with self.assertRaisesRegex(m.GateError, "PRIOR_PAIRED"):
            m.claim_paired_attempt(self.home, self.out)

    def test_prior_single_check_blocks(self):
        p = self.prior()
        (p / "check-00.json").write_bytes(b"{}")
        with self.assertRaisesRegex(m.GateError, "PRIOR_PAIRED"):
            m.claim_paired_attempt(self.home, self.out)

    def test_prior_unknown_inflight_blocks(self):
        self.prior()
        with self.assertRaisesRegex(m.GateError, "INCOMPLETE_NO_REPLAY"):
            m.claim_paired_attempt(self.home, self.out)

    def test_prior_unrecognized_report_blocks(self):
        self.prior({"state": "looks-fine"})
        with self.assertRaisesRegex(m.GateError, "UNRECOGNIZED"):
            m.claim_paired_attempt(self.home, self.out)

    def test_inspection_history_does_not_consume_attempt(self):
        self.prior({"schema": "szl.laptop-recovery/v1", "version": "1.1.0",
                    "state": "METADATA_RECORDED_NO_INFERENCE_REQUESTED"})
        m.claim_paired_attempt(self.home, self.out)

    def test_known_legacy_busy_gate_does_not_consume_attempt(self):
        self.prior({"schema": "szl.laptop-recovery/v1", "version": "1.0.0",
                    "state": "STOPPED_WITH_EVIDENCE_NO_AUTOMATIC_RETRY",
                    "error_code": "OTHER_GPU_WORK_PRESENT_NO_PROCESS_STOPPED"})
        m.claim_paired_attempt(self.home, self.out)

    def test_legacy_timeout_is_never_replayed(self):
        self.prior({"schema": "szl.laptop-recovery/v1", "version": "1.1.0",
                    "state": "STOPPED_WITH_EVIDENCE_NO_AUTOMATIC_RETRY",
                    "error_code": "TIMEOUT_NO_AUTOMATIC_RETRY"})
        with self.assertRaisesRegex(m.GateError, "UNCERTAIN_NO_REPLAY"):
            m.claim_paired_attempt(self.home, self.out)

    def test_new_preclaim_failure_does_not_consume_attempt(self):
        self.prior({"schema": "szl.laptop-recovery/v1", "version": "1.2.0",
                    "state": "STOPPED_WITH_EVIDENCE_NO_AUTOMATIC_RETRY",
                    "paired_execution": "NOT_CLAIMED"})
        m.claim_paired_attempt(self.home, self.out)

    def test_inspection_label_cannot_hide_existing_pair(self):
        self.prior({"schema": "szl.laptop-recovery/v1", "version": "1.2.0",
                    "state": "METADATA_RECORDED_NO_INFERENCE_REQUESTED",
                    "paired_result": {"scores": {}}})
        with self.assertRaisesRegex(m.GateError, "PRIOR_PAIRED"):
            m.claim_paired_attempt(self.home, self.out)

    def test_bad_response_is_saved_before_assessment(self):
        c = FakeClient()
        original = c.call
        def call(path, payload=None, timeout=15):
            result = original(path, payload, timeout)
            if path == "/api/chat":
                result["message"]["tool_calls"] = [{"name": "not-executed"}]
            return result
        c.call = call
        with self.assertRaisesRegex(m.GateError, "TOOL_CALL"):
            m.paired_check(c, self.out, {}, gpu_probe=healthy_gpu)
        self.assertTrue((self.out / "raw-response-00.json").is_file())
        self.assertEqual(len(c.requests), 1)

    def test_main_timeout_then_rerun_makes_no_second_pair(self):
        # A real main flow with mocked hardware/API; permanent claim survives
        # finally's cooperative-lock removal and report creation.
        self.out.rmdir()
        calls = []
        def timeout(*args):
            self.assertEqual(len(list((self.home / ".szl-recovery-attempts").glob("*.json"))), 1)
            calls.append(1)
            raise TimeoutError("synthetic ambiguous request")
        with ExitStack() as stack:
            stack.enter_context(patch.object(m.platform, "system", return_value="Windows"))
            stack.enter_context(patch.object(m.platform, "node", return_value="BETTERWITHAGE"))
            stack.enter_context(patch.object(m.sys, "version_info", (3, 12, 10)))
            stack.enter_context(patch.object(m.sys, "argv", ["recovery", "--run-paired"]))
            stack.enter_context(patch.object(m, "verify_package"))
            stack.enter_context(patch.object(m.Path, "home", return_value=self.home))
            stack.enter_context(patch.object(m, "verify_evidence", return_value={"fixture": "a"*64}))
            stack.enter_context(patch.object(m, "recipe_presence", return_value={}))
            stack.enter_context(patch.object(m, "gpu_observation", return_value=healthy_gpu()))
            stack.enter_context(patch.object(m, "loaded_models", return_value=[]))
            stack.enter_context(patch.object(m, "tags", return_value={m.RECEIPT: {"digest":m.EXPECTED[m.RECEIPT],"size":10}}))
            client = stack.enter_context(patch.object(m, "LocalClient"))
            client.return_value.call.return_value = {"details":{"format":"gguf"},"capabilities":["completion"]}
            stack.enter_context(patch.object(m, "paired_check", side_effect=timeout))
            self.assertEqual(m.main(), 1)
            self.assertFalse((self.home / ".szl-two-machine-lab.lock").exists())
            self.assertEqual(m.main(), 1)
        self.assertEqual(len(calls), 1)
        reports = [json.loads(p.read_bytes()) for p in self.home.glob("szl-recovery-*/recovery-report.json")]
        self.assertEqual({r["error_code"] for r in reports},
                         {"TIMEOUT_NO_AUTOMATIC_RETRY", "PAIRED_ATTEMPT_ALREADY_CLAIMED_NO_REPLAY"})

if __name__ == "__main__":
    unittest.main()
