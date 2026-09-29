"""Offline regression coverage for release guards; no model or Docker access.

SPDX-License-Identifier: Apache-2.0
"""
from contextlib import ExitStack
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location("local_model_release_guards", Path(__file__).with_name("lab.py"))
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


class SmokeAbortTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="szl-smoke-guards-")))
        tasks = self.root / "tasks"
        tasks.mkdir()
        self.paths = [tasks / name for name in ("a.json", "b.json", "c.json")]
        for path in self.paths:
            path.write_text("{}", encoding="utf-8")
        self.stack.enter_context(mock.patch.object(lab, "ROOT", self.root))
        self.stack.enter_context(mock.patch.object(lab.sys, "argv", ["lab.py", "smoke"]))
        self.stdout = self.stack.enter_context(mock.patch("sys.stdout", new=io.StringIO()))
        self.stack.enter_context(mock.patch.object(lab.subprocess, "run", side_effect=AssertionError("No processes allowed")))
        self.stack.enter_context(mock.patch.object(lab.subprocess, "Popen", side_effect=AssertionError("No processes allowed")))
        self.stack.enter_context(mock.patch.object(lab, "api", side_effect=AssertionError("No model calls allowed")))
        self.run_task = self.stack.enter_context(mock.patch.object(lab, "run_task"))

    def summary(self):
        return json.loads(self.stdout.getvalue())

    def test_cleanup_failure_stops_before_second_task_and_returns_nonzero(self):
        self.run_task.side_effect = [
            {"passed": False, "error": {"type": "RuntimeError", "message": "Cleanup unconfirmed"},
             "attempts": [{"evaluation": {"passed": False, "reason": "INFRASTRUCTURE_ERROR"}}]},
            AssertionError("Must not start another task after unconfirmed cleanup"),
        ]
        self.assertEqual(lab.main(), 1)
        self.run_task.assert_called_once_with(self.paths[0])
        self.assertEqual(self.summary(), {"passed": 0, "completed": 1, "planned": 3, "aborted": True})

    def test_later_infrastructure_error_preserves_prior_success_and_stops(self):
        self.run_task.side_effect = [
            {"passed": True},
            {"passed": False, "error": {"type": "TimeoutExpired", "message": "Docker unavailable"}},
            AssertionError("Must not start the third task"),
        ]
        self.assertEqual(lab.main(), 1)
        self.assertEqual(self.run_task.call_args_list, [mock.call(path) for path in self.paths[:2]])
        self.assertEqual(self.summary(), {"passed": 1, "completed": 2, "planned": 3, "aborted": True})

    def test_ordinary_candidate_mismatch_does_not_abort_remaining_tasks(self):
        self.run_task.side_effect = [
            {"passed": False, "attempts": [{"evaluation": {"passed": False, "reason": "ANSWER_MISMATCH"}}]},
            {"passed": True}, {"passed": True},
        ]
        self.assertEqual(lab.main(), 1)
        self.assertEqual(self.run_task.call_args_list, [mock.call(path) for path in self.paths])
        self.assertEqual(self.summary(), {"passed": 2, "completed": 3, "planned": 3, "aborted": False})

    def test_only_all_completed_and_passed_tasks_return_success(self):
        self.run_task.return_value = {"passed": True}
        self.assertEqual(lab.main(), 0)
        self.assertEqual(self.run_task.call_args_list, [mock.call(path) for path in self.paths])
        self.assertEqual(self.summary(), {"passed": 3, "completed": 3, "planned": 3, "aborted": False})


class AggregateInputGuardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="szl-input-guards-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "task.json"

    def write_task(self, values):
        task = {"id": "input-size", "instruction": "Return null for every input.",
                "cases": [{"input": value, "expected": None} for value in values]}
        data = lab.canonical(task).encode("utf-8")
        self.assertLessEqual(len(data), 100000, "Fixture must pass the separate task-file byte cap")
        self.path.write_bytes(data)
        return task

    def test_accepts_exact_aggregate_encoded_input_limit(self):
        # A one-string JSON array adds four ASCII bytes: ["..."] .
        values = ["x" * (lab.MAX_OUTPUT - 4)]
        self.assertEqual(len(lab.canonical(values).encode("utf-8")), lab.MAX_OUTPUT)
        task = self.write_task(values)
        self.assertEqual(lab.load_task(self.path)["cases"], task["cases"])

    def test_rejects_one_byte_above_aggregate_input_limit(self):
        self.write_task(["x" * (lab.MAX_OUTPUT - 3)])
        with self.assertRaisesRegex(ValueError, "Aggregate task inputs exceed sandbox input limit"):
            lab.load_task(self.path)

    def test_counts_combined_cases_not_only_each_individual_value(self):
        values = ["x" * 32767, "y" * 32767]
        self.assertTrue(all(len(lab.canonical(value).encode("utf-8")) < lab.MAX_OUTPUT for value in values))
        self.write_task(values)
        with self.assertRaisesRegex(ValueError, "Aggregate task inputs"):
            lab.load_task(self.path)

    def test_counts_utf8_bytes_instead_of_unicode_character_count(self):
        values = ["é" * 34000]
        self.assertLess(len(lab.canonical(values)), lab.MAX_OUTPUT)
        self.assertGreater(len(lab.canonical(values).encode("utf-8")), lab.MAX_OUTPUT)
        self.write_task(values)
        with self.assertRaisesRegex(ValueError, "Aggregate task inputs"):
            lab.load_task(self.path)

    def test_oversized_task_fails_before_generation_or_sandbox_and_retains_receipt(self):
        self.write_task(["x" * 70000])
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(lab, "ROOT", self.root))
            stack.enter_context(mock.patch.object(lab.shutil, "disk_usage", return_value=mock.Mock(free=1024**3)))
            stack.enter_context(mock.patch("sys.stdout", new=io.StringIO()))
            stack.enter_context(mock.patch.object(lab, "preflight_docker"))
            stack.enter_context(mock.patch.object(lab, "check_model", return_value={"manifest_sha256": "a" * 64}))
            api = stack.enter_context(mock.patch.object(lab, "api", side_effect=AssertionError("No model calls allowed")))
            sandbox = stack.enter_context(mock.patch.object(lab, "sandbox", side_effect=AssertionError("No execution allowed")))
            receipt = lab.run_task(self.path)
        self.assertIs(receipt["passed"], False)
        self.assertIs(receipt["generation_started"], False)
        self.assertEqual(receipt["phase"], "load_task")
        self.assertEqual(receipt["attempts"], [])
        self.assertEqual(receipt["error"]["type"], "ValueError")
        self.assertIn("Aggregate task inputs", receipt["error"]["message"])
        api.assert_not_called()
        sandbox.assert_not_called()
        directories = list((self.root / "runs").iterdir())
        self.assertEqual(len(directories), 1)
        self.assertEqual({path.name for path in directories[0].iterdir()}, {"receipt.json", "trajectory.json"})
        self.assertEqual(json.loads((directories[0] / "receipt.json").read_text(encoding="utf-8")), receipt)


class StructuralWireGrammarTests(unittest.TestCase):
    def test_wire_schema_avoids_expanded_repetitions_but_host_limits_remain_enforced(self):
        wire = lab.canonical(lab.SCHEMA)
        for expanded_bound in ("minItems", "maxItems", "minLength", "maxLength"):
            self.assertNotIn(expanded_bound, wire)
        self.assertEqual(set(lab.SCHEMA["required"]), {"code_lines", "explanation"})
        self.assertIs(lab.SCHEMA["additionalProperties"], False)
        valid = ["def solve(value):", "    return value"]
        invalid_documents = [
            {"code_lines": [], "explanation": ""},
            {"code_lines": valid + [""] * lab.MAX_CODE_LINES, "explanation": ""},
            {"code_lines": valid + ["#" * (lab.MAX_LINE_LENGTH + 1)], "explanation": ""},
            {"code_lines": valid + ["#" * lab.MAX_LINE_LENGTH] * 13, "explanation": ""},
            {"code_lines": valid, "explanation": "x" * 4001},
        ]
        for index, document in enumerate(invalid_documents):
            with self.subTest(bound=index), self.assertRaises(ValueError):
                lab.candidate(document)
        self.assertEqual(lab.candidate({"code_lines": valid, "explanation": "Identity"}), "\n".join(valid) + "\n")


if __name__ == "__main__":
    unittest.main()
