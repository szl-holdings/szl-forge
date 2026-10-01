"""Offline run lifecycle tests with actual temporary receipt persistence.

SPDX-License-Identifier: Apache-2.0
These tests mock all model, Docker, and candidate execution boundaries. They do
not establish model quality, GPU operation, or hostile-code isolation.
"""
from contextlib import ExitStack
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location("local_model_run_under_test", Path(__file__).with_name("lab.py"))
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


class RunLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="szl-run-lifecycle-")))
        self.worker = b"# Mocked worker; never executed.\n"
        (self.root / "worker.py").write_bytes(self.worker)
        self.pin = {"model": lab.MODEL, "manifest_sha256": "a" * 64}
        self.task = {
            "id": "identity", "instruction": "Return the input without changing it.",
            "snapshot_sha256": "b" * 64,
            "cases": [
                {"input": {"unicode": "é", "values": [1, True, None]},
                 "expected": {"unicode": "é", "values": [1, True, None]}},
                {"input": "case-marker-not-in-prompt", "expected": "case-marker-not-in-prompt"},
            ],
        }
        self.document = {"code_lines": ["def solve(value):", "    return value"], "explanation": "Identity."}
        self.response = {
            "message": {"role": "assistant", "content": json.dumps(self.document)},
            "done": True, "done_reason": "stop", "eval_count": 12,
            "eval_duration": 12000, "total_duration": 15000,
        }
        self.execution = {
            "failure": None, "exit_code": 0,
            "stdout": json.dumps([case["expected"] for case in self.task["cases"]]),
            "stderr": "", "seconds": 0.1, "container_image": lab.IMAGE,
            "container_name": "szl-local-mocked-test", "cleanup_confirmed": True,
        }
        self.runtime = {"models": [{"name": lab.MODEL, "size_vram": 123}]}
        self.api_inputs = []

        def fake_api(path, body=None, **kwargs):
            # Copy request bytes at the call boundary: run_task mutates its
            # conversation list after the mocked model has returned.
            self.api_inputs.append((path, copy.deepcopy(body), kwargs))
            if path == "/api/chat":
                return copy.deepcopy(self.response)
            if path == "/api/ps":
                return copy.deepcopy(self.runtime)
            raise AssertionError("Unexpected model API call: " + path)

        self.stack.enter_context(mock.patch.object(lab, "ROOT", self.root))
        self.stack.enter_context(mock.patch.object(lab.shutil, "disk_usage", return_value=mock.Mock(free=1024**3)))
        self.stack.enter_context(mock.patch("sys.stdout", new=io.StringIO()))
        # Fail the test if a future refactor bypasses the explicit mocks.
        self.stack.enter_context(mock.patch.object(lab.subprocess, "run", side_effect=AssertionError("No processes allowed")))
        self.stack.enter_context(mock.patch.object(lab.subprocess, "Popen", side_effect=AssertionError("No processes allowed")))
        self.stack.enter_context(mock.patch.object(lab.urllib.request, "build_opener", side_effect=AssertionError("No network allowed")))
        self.preflight = self.stack.enter_context(mock.patch.object(lab, "preflight_docker"))
        self.check_model = self.stack.enter_context(mock.patch.object(lab, "check_model", return_value=self.pin))
        self.load_task = self.stack.enter_context(mock.patch.object(lab, "load_task", return_value=self.task))
        self.api = self.stack.enter_context(mock.patch.object(lab, "api", side_effect=fake_api))
        self.sandbox = self.stack.enter_context(mock.patch.object(lab, "sandbox", return_value=self.execution))

    def persisted(self, returned):
        directories = list((self.root / "runs").iterdir())
        self.assertEqual(len(directories), 1)
        directory = directories[0]
        receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
        trajectory = json.loads((directory / "trajectory.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt, returned)
        self.assertEqual(trajectory["result"], receipt)
        self.assertIs(trajectory["training_eligible"], False)
        self.assertIn("completed_at", receipt)
        self.assertIs(receipt["training_performed"], False)
        self.assertIs(receipt["production_changes"], False)
        self.assertIs(receipt["remote_inference"], False)
        return directory, receipt, trajectory

    def assert_no_generation_artifacts(self, directory):
        self.assertEqual({path.name for path in directory.iterdir()}, {"receipt.json", "trajectory.json"})

    def test_docker_failure_persists_before_model_task_or_generation_access(self):
        self.preflight.side_effect = TimeoutError("Docker did not respond")
        returned = lab.run_task("must-not-read.json")
        directory, receipt, trajectory = self.persisted(returned)
        self.assertIs(receipt["passed"], False)
        self.assertIs(receipt["generation_started"], False)
        self.assertEqual(receipt["phase"], "preflight_docker")
        self.assertEqual(receipt["error"], {"type": "TimeoutError", "message": "Docker did not respond"})
        self.assertEqual(receipt["attempts"], [])
        self.assertEqual(trajectory["messages"], [])
        self.check_model.assert_not_called()
        self.load_task.assert_not_called()
        self.api.assert_not_called()
        self.sandbox.assert_not_called()
        self.assert_no_generation_artifacts(directory)

    def test_model_preflight_failure_persists_without_task_or_generation_access(self):
        self.check_model.side_effect = RuntimeError("Model identity changed")
        returned = lab.run_task("must-not-read.json")
        directory, receipt, trajectory = self.persisted(returned)
        self.preflight.assert_called_once_with()
        self.check_model.assert_called_once_with()
        self.assertIs(receipt["passed"], False)
        self.assertIs(receipt["generation_started"], False)
        self.assertEqual(receipt["phase"], "preflight_model")
        self.assertEqual(receipt["error"]["type"], "RuntimeError")
        self.assertEqual(receipt["attempts"], [])
        self.assertEqual(trajectory["messages"], [])
        self.load_task.assert_not_called()
        self.api.assert_not_called()
        self.sandbox.assert_not_called()
        self.assert_no_generation_artifacts(directory)

    def test_model_identity_is_rechecked_immediately_before_generation(self):
        self.check_model.side_effect = [self.pin, RuntimeError("Model identity changed after preflight")]
        returned = lab.run_task("task.json")
        directory, receipt, _ = self.persisted(returned)
        self.assertEqual(self.check_model.call_count, 2)
        self.load_task.assert_called_once_with("task.json")
        self.assertIs(receipt["passed"], False)
        self.assertIs(receipt["generation_started"], False)
        self.assertEqual(receipt["phase"], "preflight_model")
        self.assertEqual(receipt["model_manifest_sha256"], self.pin["manifest_sha256"])
        self.api.assert_not_called()
        self.sandbox.assert_not_called()
        self.assert_no_generation_artifacts(directory)

    def test_generated_identity_candidate_passes_only_external_scoring_and_persists_bindings(self):
        returned = lab.run_task("task.json")
        directory, receipt, trajectory = self.persisted(returned)
        self.assertIs(receipt["passed"], True)
        self.assertIs(receipt["generation_started"], True)
        self.assertEqual(receipt["phase"], "complete")
        self.assertEqual(receipt["task_id"], self.task["id"])
        self.assertEqual(receipt["task_sha256"], self.task["snapshot_sha256"])
        self.assertEqual(receipt["model_manifest_sha256"], self.pin["manifest_sha256"])
        self.assertEqual(receipt["harness_sha256"], lab.sha(Path(lab.__file__).read_bytes()))
        self.assertEqual(receipt["worker_sha256"], lab.sha(self.worker))
        self.assertEqual(receipt["gpu_runtime_readback"], self.runtime)
        self.assertNotIn("error", receipt)
        self.assertEqual(len(receipt["attempts"]), 1)
        entry = receipt["attempts"][0]
        self.assertEqual(entry["evaluation"], {"passed": True, "correct": 2, "total": 2, "reason": "PASS"})
        code = (directory / "candidate-1" / "solution.py").read_bytes()
        self.assertEqual(code, b"def solve(value):\n    return value\n")
        self.assertEqual(entry["code_sha256"], lab.sha(code))
        self.assertEqual((directory / "candidate-1" / "worker.py").read_bytes(), self.worker)
        self.assertEqual(json.loads((directory / "generation-1.json").read_text(encoding="utf-8")), self.response)
        self.assertEqual(json.loads((directory / "execution-1.json").read_text(encoding="utf-8")), self.execution)
        self.sandbox.assert_called_once_with(directory / "candidate-1", [case["input"] for case in self.task["cases"]])
        self.assertEqual(self.check_model.call_count, 2)
        chat_path, chat_body, _ = self.api_inputs[0]
        self.assertEqual(chat_path, "/api/chat")
        self.assertEqual(chat_body["messages"], [{"role": "system", "content": lab.SYSTEM},
                                                {"role": "user", "content": self.task["instruction"]}])
        self.assertNotIn("case-marker-not-in-prompt", json.dumps(chat_body))
        self.assertEqual(chat_body["model"], lab.MODEL)
        self.assertEqual(chat_body["format"], lab.SCHEMA)
        self.assertIs(type(chat_body["keep_alive"]), int)
        self.assertEqual(chat_body["keep_alive"], 0)
        self.assertEqual([call[0] for call in self.api_inputs], ["/api/chat", "/api/ps"])
        self.assertEqual(len(trajectory["messages"]), 3)

    def test_every_retry_requests_unload_without_changing_generation_or_sandbox_contract(self):
        failed = {**self.execution, "stdout": "[null,null]"}
        self.sandbox.side_effect = [failed, self.execution]
        _, receipt, _ = self.persisted(lab.run_task("task.json", attempts=2))
        self.assertIs(receipt["passed"], True)
        self.assertEqual(len(receipt["attempts"]), 2)
        self.assertEqual(self.check_model.call_count, 3)
        self.assertEqual(self.sandbox.call_count, 2)
        requests = [body for path, body, _ in self.api_inputs if path == "/api/chat"]
        self.assertEqual(len(requests), 2)
        for index, body in enumerate(requests, start=1):
            self.assertIs(type(body["keep_alive"]), int)
            self.assertEqual(body["keep_alive"], 0)
            self.assertEqual(body["model"], lab.MODEL)
            self.assertEqual(body["format"], lab.SCHEMA)
            self.assertIs(body["stream"], False)
            self.assertEqual(body["options"], {"num_ctx": 4096, "num_predict": 1800,
                                              "temperature": 0.2, "seed": 37 + index})
        self.preflight.assert_called_once_with()

    def test_empty_post_run_model_snapshot_does_not_rewrite_external_score(self):
        self.runtime = {"models": []}
        _, receipt, _ = self.persisted(lab.run_task("task.json"))
        self.assertIs(receipt["passed"], True)
        self.assertEqual(receipt["gpu_runtime_readback"], {"models": []})
        self.assertNotIn("telemetry_error", receipt)
        self.assertNotIn("error", receipt)
        self.assertEqual(receipt["attempts"][0]["evaluation"]["reason"], "PASS")

    def test_one_shot_ask_requests_unload_without_docker_or_execution_access(self):
        with mock.patch("sys.argv", ["lab.py", "ask", "Suggest a testable experiment."]), \
                mock.patch("sys.stdout", new=io.StringIO()) as output:
            self.assertEqual(lab.main(), 0)
        self.check_model.assert_called_once_with()
        self.preflight.assert_not_called()
        self.load_task.assert_not_called()
        self.sandbox.assert_not_called()
        self.assertEqual(len(self.api_inputs), 1)
        path, body, _ = self.api_inputs[0]
        self.assertEqual(path, "/api/chat")
        self.assertIs(type(body["keep_alive"]), int)
        self.assertEqual(body["keep_alive"], 0)
        self.assertEqual(body["model"], lab.MODEL)
        self.assertIs(body["stream"], False)
        self.assertEqual(body["options"], {"num_ctx": 4096, "num_predict": 1200, "temperature": 0.4})
        self.assertEqual(body["messages"][-1], {"role": "user", "content": "Suggest a testable experiment."})
        self.assertEqual(output.getvalue(), self.response["message"]["content"] + "\n")

    def test_telemetry_failure_does_not_rewrite_a_passed_external_score(self):
        prior = self.api.side_effect

        def missing_telemetry(path, *args, **kwargs):
            if path == "/api/ps":
                raise TimeoutError("Optional telemetry unavailable")
            return prior(path, *args, **kwargs)

        self.api.side_effect = missing_telemetry
        directory, receipt, _ = self.persisted(lab.run_task("task.json"))
        self.assertIs(receipt["passed"], True)
        self.assertEqual(receipt["phase"], "complete")
        self.assertEqual(receipt["telemetry_error"], {"type": "TimeoutError", "message": "Optional telemetry unavailable"})
        self.assertNotIn("error", receipt)
        self.assertNotIn("gpu_runtime_readback", receipt)
        self.assertIs(receipt["attempts"][0]["evaluation"]["passed"], True)
        self.assertTrue((directory / "execution-1.json").is_file())

    def test_invalid_candidate_cannot_reach_sandbox(self):
        self.response["message"]["content"] = json.dumps({"code_lines": ["```python", "x = 1"], "explanation": "Invalid"})
        directory, receipt, _ = self.persisted(lab.run_task("task.json", attempts=1))
        self.assertIs(receipt["passed"], False)
        self.assertIs(receipt["generation_started"], True)
        self.assertEqual(receipt["phase"], "complete")
        self.assertEqual(receipt["attempts"][0]["evaluation"]["reason"], "INVALID_CANDIDATE")
        self.sandbox.assert_not_called()
        self.assertTrue((directory / "generation-1.json").is_file())
        self.assertFalse((directory / "candidate-1").exists())
        self.assertFalse((directory / "execution-1.json").exists())

    def test_candidate_claims_do_not_override_wrong_external_answer(self):
        self.document["explanation"] = "All tests passed perfectly."
        self.response["message"]["content"] = json.dumps(self.document)
        self.execution["stdout"] = "[null,null]"
        _, receipt, _ = self.persisted(lab.run_task("task.json", attempts=1))
        self.assertIs(receipt["passed"], False)
        self.assertEqual(receipt["attempts"][0]["evaluation"],
                         {"passed": False, "correct": 0, "total": 2, "reason": "ANSWER_MISMATCH"})


    def test_unconfirmed_cleanup_preserves_execution_and_code_then_aborts_without_retry(self):
        self.execution.update(cleanup_confirmed=False, failure="CLEANUP_UNCONFIRMED",
                              execution_failure=None, cleanup_error="Docker cleanup timed out")
        directory, receipt, trajectory = self.persisted(lab.run_task("task.json", attempts=3))
        self.assertIs(receipt["passed"], False)
        self.assertIs(receipt["generation_started"], True)
        self.assertEqual(receipt["phase"], "evaluate")
        self.assertEqual(len(receipt["attempts"]), 1)
        entry = receipt["attempts"][0]
        self.assertEqual(entry["evaluation"]["reason"], "INFRASTRUCTURE_ERROR")
        self.assertIs(entry["evaluation"]["passed"], False)
        self.assertEqual(receipt["error"]["type"], "RuntimeError")
        self.assertIn(self.execution["container_name"], receipt["error"]["message"])
        code = (directory / "candidate-1" / "solution.py").read_bytes()
        self.assertEqual(entry["code_sha256"], lab.sha(code))
        self.assertEqual(json.loads((directory / "execution-1.json").read_text(encoding="utf-8")), self.execution)
        self.sandbox.assert_called_once()
        self.assertEqual([call[0] for call in self.api_inputs], ["/api/chat"])
        self.assertEqual(len(trajectory["messages"]), 3)
        self.assertFalse((directory / "candidate-2").exists())
        self.assertFalse((directory / "generation-2.json").exists())


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.pin = {"model": lab.MODEL, "manifest_sha256": "a" * 64}
        self.check_model = self.stack.enter_context(mock.patch.object(lab, "check_model", return_value=self.pin))
        self.clock = self.stack.enter_context(mock.patch.object(lab.time, "monotonic", return_value=100))
        self.sleep = self.stack.enter_context(mock.patch.object(lab.time, "sleep"))
        self.api = self.stack.enter_context(mock.patch.object(lab, "api", side_effect=AssertionError("No service calls allowed")))

    def test_immediate_valid_pin_returns_without_sleep(self):
        self.assertEqual(lab.wait_ready(), self.pin)
        self.check_model.assert_called_once_with(timeout=2)
        self.sleep.assert_not_called()
        self.api.assert_not_called()

    def test_transient_timeout_is_retried_then_valid_pin_returns(self):
        self.check_model.side_effect = [TimeoutError("Still starting"), self.pin]
        self.assertEqual(lab.wait_ready(timeout=5), self.pin)
        self.assertEqual(self.check_model.call_args_list, [mock.call(timeout=2), mock.call(timeout=2)])
        self.sleep.assert_called_once_with(0.5)
        self.api.assert_not_called()

    def test_identity_mismatch_fails_fast_without_retry(self):
        self.check_model.side_effect = RuntimeError("Model identity mismatch")
        with self.assertRaisesRegex(RuntimeError, "Model identity mismatch"):
            lab.wait_ready(timeout=60)
        self.check_model.assert_called_once_with(timeout=2)
        self.sleep.assert_not_called()
        self.api.assert_not_called()

    def test_invalid_timeout_is_rejected_before_access_or_sleep(self):
        for timeout in (-1, 0, 121):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, "1 to 120 seconds"):
                lab.wait_ready(timeout=timeout)
        self.clock.assert_not_called()
        self.check_model.assert_not_called()
        self.sleep.assert_not_called()
        self.api.assert_not_called()

    def test_deadline_expiry_fails_after_bounded_transient_retry(self):
        self.clock.side_effect = [100, 100, 100, 100.1, 101]
        self.check_model.side_effect = TimeoutError("Still starting")
        with self.assertRaisesRegex(RuntimeError, "readiness deadline exceeded: TimeoutError"):
            lab.wait_ready(timeout=1)
        self.check_model.assert_called_once_with(timeout=1)
        self.sleep.assert_called_once_with(0.5)
        self.api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
