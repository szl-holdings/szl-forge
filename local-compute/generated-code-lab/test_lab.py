"""Offline unit tests for the local development harness; no Docker/model execution.

SPDX-License-Identifier: Apache-2.0
Passing these tests does not establish hostile-code isolation or model quality.
"""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location("local_model_lab_under_test", Path(__file__).with_name("lab.py"))
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


class CandidateTests(unittest.TestCase):
    def test_accepts_complete_source_without_executing_it(self):
        lines = ["raise RuntimeError('must not execute on host')", "def solve(value):", "    return value"]
        self.assertEqual(lab.candidate({"code_lines": lines, "explanation": "identity"}),
                         "\n".join(lines) + "\n")

    def test_rejects_missing_extra_or_legacy_fields(self):
        for document in [None, [], {}, {"code_lines": ["x"]},
                         {"code_lines": ["x"], "explanation": "", "path": "../anything"},
                         {"code": "def solve(value):\n    return value", "explanation": ""},
                         {"code_lines": ["x"], "explanation": []}]:
            with self.subTest(document=document), self.assertRaises(ValueError):
                lab.candidate(document)

    def test_requires_nonempty_list_of_strings(self):
        for lines in [None, [], "def solve(value): return value", ("x",), {}, [None],
                      [0], [True], ["def solve(value):", {"line": "    return value"}]]:
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                lab.candidate({"code_lines": lines, "explanation": ""})

    def test_rejects_actual_line_breaks_inside_items(self):
        for line_break in ["\n", "\r", "\r\n"]:
            lines = ["def solve(value):" + line_break + "    return value"]
            with self.subTest(line_break=repr(line_break)), self.assertRaises(ValueError):
                lab.candidate({"code_lines": lines, "explanation": ""})

    def test_preserves_indentation_empty_lines_and_literal_backslashes(self):
        lines = ["", "def solve(value):", "", 
                 r'    return {"regex": r"\d+\n", "path": r"C:\new\test", "literal": "\\n"}', ""]
        document = {"code_lines": lines, "explanation": "Literal source lines"}
        # Exercise the JSON wire boundary without executing the supplied source.
        decoded = json.loads(json.dumps(document))
        code = lab.candidate(decoded)
        self.assertEqual(code, "\n".join(lines) + "\n")
        self.assertEqual(decoded, document)
        self.assertIn(r'C:\new\test', code)
        self.assertIn(r'"\\n"', code)

    def test_does_not_decode_literal_backslash_n_into_source_newlines(self):
        lines = [r"def solve(value):\n    return value"]
        with self.assertRaises(SyntaxError):
            lab.candidate({"code_lines": lines, "explanation": "Malformed source must not be repaired"})

    def test_rejects_markdown_after_blank_lines(self):
        for prefix in [[], ["", "   "]]:
            lines = prefix + ["```python", "def solve(value):", "    return value", "```"]
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                lab.candidate({"code_lines": lines, "explanation": ""})

    def test_rejects_syntax_error(self):
        with self.assertRaises(SyntaxError):
            lab.candidate({"code_lines": ["def solve(:"], "explanation": ""})

    def test_requires_top_level_synchronous_function(self):
        for code in ["solve = lambda value: value", "async def solve(value):\n    return value",
                     "def outer():\n    def solve(value):\n        return value"]:
            with self.subTest(code=code), self.assertRaises(ValueError):
                lab.candidate({"code_lines": code.split("\n"), "explanation": ""})

    def test_enforces_line_count_and_individual_line_bounds(self):
        valid = ["def solve(value):", "    return value"]
        too_many = valid + [""] * (lab.MAX_CODE_LINES - len(valid) + 1)
        too_long = valid + ["#" + "x" * lab.MAX_LINE_LENGTH]
        for lines in [too_many, too_long]:
            with self.subTest(line_count=len(lines)), self.assertRaises(ValueError):
                lab.candidate({"code_lines": lines, "explanation": ""})

    def test_accepts_exact_line_count_and_individual_line_bounds(self):
        valid = ["def solve(value):", "    return value"]
        for lines in [valid + [""] * (lab.MAX_CODE_LINES - len(valid)),
                      valid + ["#" + "x" * (lab.MAX_LINE_LENGTH - 1)]]:
            with self.subTest(line_count=len(lines)):
                self.assertEqual(lab.candidate({"code_lines": lines, "explanation": ""}),
                                 "\n".join(lines) + "\n")

    def test_enforces_utf8_byte_bound_including_joined_newlines(self):
        lines = ["def solve(value):", "    return value", "# é"]
        code = "\n".join(lines) + "\n"
        document = {"code_lines": lines, "explanation": ""}
        exact_size = len(code.encode("utf-8"))
        with mock.patch.object(lab, "MAX_CODE", exact_size):
            self.assertEqual(lab.candidate(document), code)
        with mock.patch.object(lab, "MAX_CODE", exact_size - 1), self.assertRaises(ValueError):
            lab.candidate(document)

    def test_enforces_explanation_bound(self):
        document = {"code_lines": ["def solve(value):", "    return value"], "explanation": "x" * 4000}
        self.assertTrue(lab.candidate(document))
        document["explanation"] += "x"
        with self.assertRaises(ValueError):
            lab.candidate(document)

    def test_wire_schema_is_structural_and_does_not_expand_large_repetitions(self):
        self.assertEqual(lab.SCHEMA["type"], "object")
        self.assertEqual(set(lab.SCHEMA["properties"]), {"code_lines", "explanation"})
        self.assertEqual(set(lab.SCHEMA["required"]), {"code_lines", "explanation"})
        self.assertIs(lab.SCHEMA["additionalProperties"], False)
        lines = lab.SCHEMA["properties"]["code_lines"]
        self.assertEqual(lines["type"], "array")
        self.assertEqual(lines["items"]["type"], "string")
        for key in ("maxItems", "minItems", "maxLength"):
            self.assertNotIn(key, lines)
            self.assertNotIn(key, lines["items"])
            self.assertNotIn(key, lab.SCHEMA["properties"]["explanation"])


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="szl-lab-test-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "task.json"
        self.task = {"id": "identity-1", "instruction": "Return the input.",
                     "cases": [{"input": {"value": "é"}, "expected": {"value": "é"}}]}

    def write(self, task):
        raw = json.dumps(task, ensure_ascii=False).encode("utf-8")
        self.path.write_bytes(raw)
        return raw

    def test_snapshot_binds_exact_original_bytes(self):
        raw = self.write(self.task)
        loaded = lab.load_task(self.path)
        self.assertEqual(loaded["snapshot_sha256"], lab.sha(raw))
        self.assertEqual(loaded["cases"], self.task["cases"])
        self.assertNotIn("snapshot_sha256", self.task)

    def test_rejects_pathlike_task_identities(self):
        for identity in ["../escape", "a/b", "A", "", "x" * 65, "with space"]:
            task = copy.deepcopy(self.task)
            task["id"] = identity
            self.write(task)
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                lab.load_task(self.path)

    def test_rejects_invalid_instructions(self):
        for instruction in [[], None, "x" * 12001]:
            task = copy.deepcopy(self.task)
            task["instruction"] = instruction
            self.write(task)
            with self.subTest(instruction_type=type(instruction).__name__), self.assertRaises(ValueError):
                lab.load_task(self.path)

    def test_requires_bounded_nonempty_exact_case_contract(self):
        for cases in [[], {}, [{"input": 1}], [{"input": 1, "expected": 1, "extra": True}],
                      [None], [{"input": 0, "expected": 0}] * 101]:
            task = copy.deepcopy(self.task)
            task["cases"] = cases
            self.write(task)
            with self.subTest(case_count=len(cases)), self.assertRaises(ValueError):
                lab.load_task(self.path)

    def test_rejects_nonfinite_json(self):
        self.task["cases"][0]["expected"] = float("nan")
        self.write(self.task)
        with self.assertRaises(ValueError):
            lab.load_task(self.path)

    def test_rejects_oversized_document(self):
        self.path.write_bytes(b" " * 100001)
        with self.assertRaises(ValueError):
            lab.load_task(self.path)


class DockerCommandTests(unittest.TestCase):
    def test_restricted_no_network_no_gpu_readonly_single_mount(self):
        folder = Path(tempfile.gettempdir()) / "candidate with spaces"
        command = lab.docker_command("szl-local-test", folder)
        self.assertEqual(command[:3], ["docker", "--host", "npipe:////./pipe/dockerDesktopLinuxEngine"])
        for flag in ["--rm", "--pull=never", "--network=none", "--read-only", "--cap-drop=ALL",
                     "--security-opt=no-new-privileges", "--user=65534:65534", "--pids-limit=32",
                     "--memory=256m", "--memory-swap=256m", "--cpus=1", "--log-driver=none"]:
            self.assertIn(flag, command)
        self.assertEqual(command.count("--mount"), 1)
        self.assertEqual(command[command.index("--mount") + 1],
                         f"type=bind,source={folder.resolve()},target=/candidate,readonly")
        self.assertEqual(command[command.index("--name") + 1], "szl-local-test")
        self.assertIn(lab.IMAGE, command)
        self.assertTrue(lab.IMAGE.startswith("sha256:"))
        self.assertEqual(command[-4:], ["python", "-I", "-B", "/candidate/worker.py"])
        for token in ["--privileged", "--gpus", "--device", "--env-file", "-e", "-v"]:
            self.assertNotIn(token, command)
        self.assertFalse(any("docker.sock" in part for part in command))

    def test_proxy_configuration_cannot_inject_credentials(self):
        command = lab.docker_command("szl-local-test", Path(tempfile.gettempdir()) / "candidate")
        environment = [command[index + 1] for index, value in enumerate(command) if value == "--env"]
        expected = {key + "=" for key in (
            "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "ALL_PROXY", "NO_PROXY",
            "http_proxy", "https_proxy", "ftp_proxy", "all_proxy", "no_proxy")}
        self.assertEqual(set(environment), expected)
        self.assertEqual(len(environment), len(expected))

    def test_rejects_mount_option_injection(self):
        with self.assertRaises(ValueError):
            lab.docker_command("szl-local-test", Path(tempfile.gettempdir()) / "candidate,readonly=false")

    def test_environment_cannot_override_explicit_local_daemon(self):
        overrides = {"DOCKER_HOST": "tcp://remote.example:2376", "DOCKER_CONTEXT": "paid-cloud",
                     "DOCKER_TLS": "1", "DOCKER_TLS_VERIFY": "1", "DOCKER_CERT_PATH": "elsewhere",
                     "PATH": "retained"}
        with mock.patch.dict(lab.os.environ, overrides, clear=True):
            clean = lab.docker_environment()
        self.assertEqual(clean, {"PATH": "retained"})


class DockerPreflightTests(unittest.TestCase):
    def result(self, stdout, returncode=0):
        return lab.subprocess.CompletedProcess(args=[], returncode=returncode,
                                               stdout=stdout, stderr="")

    def test_accepts_healthy_server_and_exact_pinned_image(self):
        responses = [self.result("28.0.0\n"), self.result(lab.IMAGE + "\n")]
        environment = {"DOCKER_HOST": "tcp://remote.example:2376",
                       "DOCKER_CONTEXT": "paid-cloud", "DOCKER_TLS": "1",
                       "DOCKER_TLS_VERIFY": "1", "DOCKER_CERT_PATH": "elsewhere",
                       "PATH": "retained"}
        with mock.patch.dict(lab.os.environ, environment, clear=True), \
                mock.patch.object(lab.subprocess, "run", side_effect=responses) as run:
            self.assertIsNone(lab.preflight_docker())
        local = ["docker", "--host", "npipe:////./pipe/dockerDesktopLinuxEngine"]
        self.assertEqual(lab.DOCKER, local)
        expected_options = {"capture_output": True, "text": True, "timeout": 12,
                            "env": {"PATH": "retained"}}
        self.assertEqual(run.call_args_list, [
            mock.call([*local, "info", "--format", "{{.ServerVersion}}"], **expected_options),
            mock.call([*local, "image", "inspect", lab.IMAGE, "--format", "{{.Id}}"],
                      **expected_options),
        ])

    def test_refuses_nonzero_exit_even_with_plausible_output(self):
        for stage in (0, 1):
            responses = [self.result("28.0.0"), self.result(lab.IMAGE)]
            responses[stage].returncode = 1
            with self.subTest(stage=stage), \
                    mock.patch.object(lab.subprocess, "run", side_effect=responses) as run, \
                    self.assertRaisesRegex(RuntimeError, "Local Docker preflight failed"):
                lab.preflight_docker()
            self.assertEqual(run.call_count, stage + 1)

    def test_refuses_empty_or_whitespace_server_response(self):
        for stdout in ("", " ", "\r\n\t"):
            with self.subTest(stdout=repr(stdout)), \
                    mock.patch.object(lab.subprocess, "run", return_value=self.result(stdout)) as run, \
                    self.assertRaisesRegex(RuntimeError, "Local Docker preflight failed"):
                lab.preflight_docker()
            self.assertEqual(run.call_count, 1)

    def test_refuses_empty_or_whitespace_image_response(self):
        for stdout in ("", " ", "\r\n\t"):
            responses = [self.result("28.0.0"), self.result(stdout)]
            with self.subTest(stdout=repr(stdout)), \
                    mock.patch.object(lab.subprocess, "run", side_effect=responses) as run, \
                    self.assertRaisesRegex(RuntimeError, "Local Docker preflight failed"):
                lab.preflight_docker()
            self.assertEqual(run.call_count, 2)

    def test_refuses_wrong_or_ambiguous_image_identity(self):
        for identity in ("sha256:" + "0" * 64, lab.IMAGE.removeprefix("sha256:"),
                         lab.IMAGE + "\nsha256:" + "0" * 64):
            responses = [self.result("28.0.0"), self.result(identity)]
            with self.subTest(identity=identity), \
                    mock.patch.object(lab.subprocess, "run", side_effect=responses), \
                    self.assertRaisesRegex(RuntimeError, "Pinned local Docker image identity mismatch"):
                lab.preflight_docker()

    def test_timeout_receipt_precedes_model_access_or_task_generation(self):
        for stage in (0, 1):
            timeout = lab.subprocess.TimeoutExpired(cmd="mocked Docker preflight", timeout=12)
            responses = [timeout] if stage == 0 else [self.result("28.0.0"), timeout]
            with self.subTest(stage=stage), \
                    tempfile.TemporaryDirectory(prefix="szl-preflight-test-") as directory, \
                    mock.patch.object(lab, "ROOT", Path(directory)), \
                    mock.patch.object(lab.shutil, "disk_usage", return_value=mock.Mock(free=1024**3)), \
                    mock.patch.object(lab.subprocess, "run", side_effect=responses) as run, \
                    mock.patch.object(lab, "check_model") as check_model, \
                    mock.patch.object(lab, "load_task") as load_task, \
                    mock.patch.object(lab, "api") as api, \
                    mock.patch.object(lab, "sandbox") as sandbox, \
                    mock.patch.object(lab, "save") as save:
                receipt = lab.run_task("task-must-not-be-read.json")
                self.assertFalse(receipt["passed"])
                self.assertFalse(receipt["generation_started"])
                self.assertEqual(receipt["phase"], "preflight_docker")
                self.assertEqual(receipt["error"]["type"], "TimeoutExpired")
                self.assertEqual(run.call_count, stage + 1)
                check_model.assert_not_called()
                load_task.assert_not_called()
                api.assert_not_called()
                sandbox.assert_not_called()
                self.assertEqual(save.call_count, 2)


class LocalApiTests(unittest.TestCase):
    def test_redirects_are_refused_even_to_other_loopback_ports(self):
        handler = lab.NoRedirect()
        for target in ["https://remote.example/inference", "http://127.0.0.1:11434/api/chat"]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                handler.redirect_request(None, None, 302, "Found", {}, target)

    def test_fixed_loopback_endpoint_proxy_disabled_and_redirect_handler_installed(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"ok":true}'
        opener = mock.MagicMock()
        opener.open.return_value = response
        with mock.patch.object(lab.urllib.request, "build_opener", return_value=opener) as build:
            self.assertEqual(lab.api("/api/chat", {"model": lab.MODEL}), {"ok": True})
        handlers = build.call_args.args
        self.assertTrue(any(isinstance(item, lab.NoRedirect) for item in handlers))
        proxies = [item for item in handlers if isinstance(item, lab.urllib.request.ProxyHandler)]
        self.assertEqual(len(proxies), 1)
        self.assertEqual(proxies[0].proxies, {})
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11439/api/chat")
        self.assertEqual(json.loads(request.data), {"model": lab.MODEL})

    def test_oversized_response_is_rejected(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b" " * (2 * 1024 * 1024 + 1)
        opener = mock.MagicMock()
        opener.open.return_value = response
        with mock.patch.object(lab.urllib.request, "build_opener", return_value=opener), self.assertRaises(ValueError):
            lab.api("/api/tags")


class EvaluateTests(unittest.TestCase):
    def result(self, stdout, **updates):
        return {"failure": None, "exit_code": 0, "stdout": stdout, **updates}

    def test_passes_exact_values_ignoring_object_key_order(self):
        result = lab.evaluate(self.result('[{"b":2,"a":1},null]'),
                              [{"expected": {"a": 1, "b": 2}}, {"expected": None}])
        self.assertEqual(result, {"passed": True, "correct": 2, "total": 2, "reason": "PASS"})

    def test_does_not_confuse_booleans_and_numbers(self):
        result = lab.evaluate(self.result("[true, false]"), [{"expected": 1}, {"expected": 0}])
        self.assertFalse(result["passed"])
        self.assertEqual(result["correct"], 0)

    def test_partial_match_cannot_pass(self):
        result = lab.evaluate(self.result("[1, 9]"), [{"expected": 1}, {"expected": 2}])
        self.assertFalse(result["passed"])
        self.assertEqual(result["correct"], 1)

    def test_order_is_part_of_array_contract(self):
        result = lab.evaluate(self.result("[[2, 1]]"), [{"expected": [1, 2]}])
        self.assertFalse(result["passed"])

    def test_execution_failure_overrides_correct_answer(self):
        for updates in [{"failure": "TIMEOUT"}, {"failure": "OUTPUT_LIMIT"}, {"exit_code": 1}]:
            result = lab.evaluate(self.result("[42]", **updates), [{"expected": 42}])
            self.assertFalse(result["passed"])
            self.assertEqual(result["correct"], 0)

    def test_rejects_nonfinite_malformed_extra_and_wrong_length_output(self):
        for output in ["", "not json", "[1]\nextra", "{}", "[]", "[1,2]", "[NaN]", "[Infinity]"]:
            with self.subTest(output=output):
                result = lab.evaluate(self.result(output), [{"expected": 1}])
                self.assertFalse(result["passed"])
                self.assertEqual(result["reason"], "INVALID_OUTPUT")


class ModelPinTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="szl-model-pin-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.patch = mock.patch.object(lab, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.pin = {"model": lab.MODEL, "manifest_sha256": "a" * 64}
        self.write_pin()

    def write_pin(self):
        (self.root / "model.lock.json").write_text(json.dumps(self.pin), encoding="utf-8")

    def test_accepts_exact_model_and_digest(self):
        with mock.patch.object(lab, "api", return_value={"models": [
            {"name": lab.MODEL, "digest": self.pin["manifest_sha256"]}]}) as api:
            self.assertEqual(lab.check_model(), self.pin)
            api.assert_called_once_with("/api/tags", timeout=10)

    def test_rejects_missing_or_changed_model(self):
        for models in [[], [{"name": "other:4b", "digest": "a" * 64}],
                       [{"name": lab.MODEL, "digest": "b" * 64}]]:
            with self.subTest(models=models), mock.patch.object(lab, "api", return_value={"models": models}):
                with self.assertRaises(RuntimeError):
                    lab.check_model()

    def test_rejects_other_model_in_lock(self):
        self.pin["model"] = "other:4b"
        self.write_pin()
        with mock.patch.object(lab, "api", return_value={"models": [
            {"name": lab.MODEL, "digest": "a" * 64}]}), self.assertRaises(RuntimeError):
            lab.check_model()


if __name__ == "__main__":
    unittest.main()
