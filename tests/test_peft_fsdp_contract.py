"""Schema and supervisor regressions; not substitute distributed execution."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import evaluate_peft_fsdp as fsdp


def rank_fixture(rank):
    return {"rank": rank, "worldSize": 2, "sourceRevision": "a" * 40,
            "upstreamRevision": fsdp.PEFT_REVISION,
            "baseSha256": "a" * 64, "gatheredSha256": "b" * 64, "ungatheredSha256": "c" * 64,
            "ungatheredWarnedAndWritten": True, "ungatheredRejected": True,
            "gatheredStructurePass": True, "peftReloadPass": True, "shardsRestoredExactly": True,
            "adapterBytesUnchanged": True, "maxAbsoluteError": 0.0, "gatheredShapes": copy.deepcopy(fsdp.SHAPES),
            "shardedShapes": {"base_model.model.linear.lora_A.default.weight": [6],
                              "base_model.model.linear.lora_B.default.weight": [0]}}


class FakeProcess:
    def __init__(self, code=None, ignore_terminate=False):
        self.exitcode = code
        self.ignore_terminate = ignore_terminate
        self.killed = False
        self.joined = False

    def is_alive(self):
        return self.exitcode is None

    def terminate(self):
        if not self.ignore_terminate:
            self.exitcode = -15

    def kill(self):
        self.killed = True
        self.exitcode = -9

    def join(self, timeout):
        self.joined = True


class FsdpContractTests(unittest.TestCase):
    def test_two_complete_rows(self):
        fsdp.validate_ranks([rank_fixture(0), rank_fixture(1)], "a" * 40)

    def test_missing_duplicate_non_record_and_bool_rank_rejected(self):
        for rows in ([], [rank_fixture(0)], [rank_fixture(0), rank_fixture(0)], [None, {}],
                     [rank_fixture(False), rank_fixture(1)]):
            with self.subTest(rows=rows):
                with self.assertRaises(fsdp.EvaluationError):
                    fsdp.validate_ranks(rows, "a" * 40)

    def test_every_rank_gate_is_required(self):
        for field in ("ungatheredWarnedAndWritten", "ungatheredRejected", "gatheredStructurePass",
                      "peftReloadPass", "shardsRestoredExactly", "adapterBytesUnchanged"):
            rows = [rank_fixture(0), rank_fixture(1)]
            rows[1][field] = False
            with self.assertRaises(fsdp.EvaluationError):
                fsdp.validate_ranks(rows, "a" * 40)

    def test_gathered_bytes_must_match_across_ranks(self):
        rows = [rank_fixture(0), rank_fixture(1)]
        rows[1]["gatheredSha256"] = "d" * 64
        with self.assertRaisesRegex(fsdp.EvaluationError, "CROSS_RANK_BYTES_DISAGREE"):
            fsdp.validate_ranks(rows, "a" * 40)

    def test_source_parity_shapes_and_digests_are_not_optional(self):
        for field, wrong in (("sourceRevision", "b" * 40), ("upstreamRevision", "0" * 40),
                             ("maxAbsoluteError", float("nan")), ("worldSize", True),
                             ("gatheredShapes", {}), ("baseSha256", "unknown"),
                             ("shardedShapes", {}), ("ungatheredSha256", "b" * 64)):
            rows = [rank_fixture(0), rank_fixture(1)]
            rows[0][field] = wrong
            with self.subTest(field=field):
                with self.assertRaises(fsdp.EvaluationError):
                    fsdp.validate_ranks(rows, "a" * 40)

    def test_nonzero_worker_rejects_and_drains_peer(self):
        children = [FakeProcess(17), FakeProcess()]
        with self.assertRaisesRegex(fsdp.EvaluationError, "WORKER_NONZERO"):
            fsdp.supervise(children, timeout=10, ready=lambda: True)
        self.assertTrue(all(p.joined and not p.is_alive() for p in children))

    def test_cancel_after_ready_rejects_and_drains(self):
        children = [FakeProcess(), FakeProcess()]
        with self.assertRaisesRegex(fsdp.EvaluationError, "GROUP_CANCELLED"):
            fsdp.supervise(children, timeout=10, ready=lambda: True, cancel=True)
        self.assertTrue(all(p.joined and not p.is_alive() for p in children))

    def test_timeout_kills_stubborn_worker(self):
        children = [FakeProcess(ignore_terminate=True), FakeProcess()]
        with patch.object(fsdp.time, "monotonic", side_effect=[0.0, 11.0]):
            with self.assertRaisesRegex(fsdp.EvaluationError, "GROUP_TIMEOUT"):
                fsdp.supervise(children, timeout=10, ready=lambda: False)
        self.assertTrue(children[0].killed)
        self.assertTrue(all(p.joined and not p.is_alive() for p in children))

    def test_success_requires_all_zero_exits(self):
        children = [FakeProcess(0), FakeProcess(0)]
        fsdp.supervise(children, timeout=10, ready=lambda: False)
        self.assertTrue(all(p.joined for p in children))

    def test_workflow_executes_fsdp_and_preserves_old_interop_dependency(self):
        text = (Path(__file__).resolve().parents[1] / ".github/workflows/peft-export-runtime.yml").read_text()
        self.assertIn("python -m tools.evaluate_peft_fsdp", text)
        self.assertIn("'packaging==26.3'", text)
        self.assertIn("'safetensors==0.7.0'", text)
        self.assertNotIn("continue-on-error", text)
        self.assertNotIn("--no-deps", text)


if __name__ == "__main__":
    unittest.main()
