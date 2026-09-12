"""Dependency-free admission tests; real PEFT runs in its separate CI lane."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

from tools import evaluate_peft_runtime as runtime


def observation():
    """A schema fixture, NOT an execution receipt."""
    return {
        "state": "CPU_EVALUATION_PASS", "sourceRevision": "a" * 40,
        "upstreamRevision": runtime.PEFT_REVISION,
        "loadedPeftSaveAndLoadGitBlob": runtime.SAVE_AND_LOAD_BLOB,
        "productionAuthorized": False, "publicationAuthorized": False, "automaticPromotion": False,
        "trainingExecuted": False, "pretrainedWeightsDownloaded": False,
        "providerWrites": 0, "productionDisposition": "HOLD",
        "validExport": {"structure": "PASS", "peftReload": "PASS", "wrongBaseNegative": "PASS",
                        "maxAbsoluteError": 0.0, "adapterBytesUnchanged": True},
        "malformedExports": [{"side": side, "shapeCase": case, "explicitStateDict": explicit,
                              "warningObserved": not explicit, "writeCompleted": True,
                              "inspectorReason": "UNGATHERED_AB_SHAPE"}
                             for side in ("lora_A", "lora_B") for case in ("flat", "empty", "scalar")
                             for explicit in (False, True)],
        "auxiliaryVectorHeuristic": "PASS", "doraScope": "UNSUPPORTED",
        "baseIdentity": {"kind": "SYNTHETIC_RECIPE_NOT_HUB_MODEL"},
        "remaining": ["REAL_ZERO3_FSDP_GATHERING", "AUTHENTICATED_EXTERNAL_MODEL_LINEAGE",
                      "IMMUTABLE_PROVIDER_READBACK", "DISTRIBUTED_CANCELLATION_FAILURE_ROLLBACK",
                      "MODEL_QUALITY_QUALIFICATION", "INDEPENDENT_PUBLISHER_ADMISSION"],
    }


class PeftRuntimeContractTests(unittest.TestCase):
    def test_exact_install_provenance(self):
        for name, (version, repo, sha) in runtime.SOURCES.items():
            with self.subTest(name=name):
                document = {"url": f"https://github.com/{repo}.git",
                            "vcs_info": {"vcs": "git", "commit_id": sha}}
                self.assertEqual(runtime.validate_install(name, version, document)["revision"], sha)
                for invalid in (None, {}, {"url": document["url"], "vcs_info": {}},
                                {"url": "https://untrusted.invalid/source", "vcs_info": document["vcs_info"]},
                                {"url": document["url"], "vcs_info": {"vcs": "git", "commit_id": "a" * 40}}):
                    with self.assertRaises(runtime.EvaluationError):
                        runtime.validate_install(name, version, invalid)
                with self.assertRaises(runtime.EvaluationError):
                    runtime.validate_install(name, version + ".different", document)

    def test_requirements_are_immutable_and_do_not_install(self):
        text = runtime.requirements()
        for name, (_, repo, sha) in runtime.SOURCES.items():
            self.assertIn(f"{name} @ git+https://github.com/{repo}.git@{sha}", text)
        self.assertNotIn("@main", text)
        with patch.object(runtime, "check_installations", side_effect=AssertionError("must not install")):
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(runtime.main(["requirements"]), 0)
            self.assertEqual(output.getvalue(), text)

    def test_source_gate_rejects_dirty_or_moved(self):
        with patch.dict(runtime.os.environ, {"SZL_EVAL_SOURCE_SHA": "a" * 40}):
            with patch.object(runtime.subprocess, "check_output", side_effect=["a" * 40, " M tools/x.py"]):
                with self.assertRaisesRegex(runtime.EvaluationError, "DIRTY_FORGE_SOURCE"):
                    runtime.source_revision()
            with patch.object(runtime.subprocess, "check_output", side_effect=["b" * 40, ""]):
                with self.assertRaisesRegex(runtime.EvaluationError, "FORGE_SOURCE_MOVED"):
                    runtime.source_revision()
            with patch.object(runtime.subprocess, "check_output", side_effect=["a" * 40, ""]):
                self.assertEqual(runtime.source_revision(), "a" * 40)

    def test_python_network_trap_and_restoration(self):
        original = socket.create_connection
        with runtime.deny_network():
            with self.assertRaisesRegex(runtime.EvaluationError, "UNEXPECTED_NETWORK_ATTEMPT"):
                socket.create_connection(("127.0.0.1", 1))
        self.assertIs(socket.create_connection, original)

    def test_complete_schema_fixture_keeps_hold(self):
        runtime.validate_observation(observation(), "a" * 40)

    def test_partial_duplicate_and_wrong_warning_are_rejected(self):
        for change in ("missing", "duplicate", "wrong_warning", "write_missing", "wrong_shape"):
            value = observation()
            with self.subTest(change=change):
                if change == "missing":
                    value["malformedExports"].pop()
                elif change == "duplicate":
                    value["malformedExports"][-1] = copy.deepcopy(value["malformedExports"][0])
                elif change == "wrong_warning":
                    value["malformedExports"][0]["warningObserved"] = False
                elif change == "write_missing":
                    value["malformedExports"][0]["writeCompleted"] = False
                else:
                    value["malformedExports"][0]["inspectorReason"] = "PASS"
                with self.assertRaises(runtime.EvaluationError):
                    runtime.validate_observation(value, "a" * 40)

    def test_authority_flags_cannot_be_truthy_or_missing(self):
        for key in ("productionAuthorized", "publicationAuthorized", "automaticPromotion",
                    "trainingExecuted", "pretrainedWeightsDownloaded", "providerWrites"):
            for wrong in (True, None, "false"):
                with self.subTest(key=key, wrong=wrong):
                    value = observation()
                    value[key] = wrong
                    with self.assertRaises(runtime.EvaluationError):
                        runtime.validate_observation(value, "a" * 40)

    def test_revision_and_upstream_bytes_bound(self):
        for key in ("sourceRevision", "upstreamRevision", "loadedPeftSaveAndLoadGitBlob"):
            value = observation()
            value[key] = "0" * 40
            with self.assertRaises(runtime.EvaluationError):
                runtime.validate_observation(value, "a" * 40)

    def test_reload_must_be_exact_finite_and_immutable(self):
        for key, wrong in (("maxAbsoluteError", float("nan")), ("maxAbsoluteError", 0.01),
                           ("maxAbsoluteError", False), ("adapterBytesUnchanged", False),
                           ("wrongBaseNegative", "NOT_RUN"), ("peftReload", "NOT_RUN")):
            value = observation()
            value["validExport"][key] = wrong
            with self.assertRaises(runtime.EvaluationError):
                runtime.validate_observation(value, "a" * 40)

    def test_missing_qualification_bounds_never_pass(self):
        for key, wrong in (("remaining", []), ("doraScope", "PASS"),
                           ("baseIdentity", {"kind": "HUB_MODEL"})):
            value = observation()
            value[key] = wrong
            with self.assertRaises(runtime.EvaluationError):
                runtime.validate_observation(value, "a" * 40)

    def test_failed_execution_is_nonzero_and_redacted(self):
        with patch.object(runtime, "source_revision", return_value="a" * 40):
            with patch.object(runtime, "run_cpu", side_effect=RuntimeError("private-secret-fixture")):
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(runtime.main(["run"]), 2)
                result = json.loads(output.getvalue())
                self.assertEqual(result["state"], "BLOCKED")
                self.assertNotIn("private-secret-fixture", output.getvalue())
                self.assertIs(result["productionAuthorized"], False)

    def test_partial_execution_cannot_exit_successfully(self):
        result = observation()
        result["malformedExports"].pop()
        with patch.object(runtime, "source_revision", return_value="a" * 40):
            with patch.object(runtime, "run_cpu", return_value=result):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(runtime.main(["run"]), 2)

    def test_workflow_has_no_production_authority(self):
        text = (Path(__file__).resolve().parents[1] / ".github/workflows/peft-export-runtime.yml").read_text()
        self.assertNotIn("pull_request_target", text)
        self.assertNotIn("secrets.", text)
        self.assertNotIn("continue-on-error", text)
        self.assertIn("contents: read", text)
        self.assertIn("persist-credentials: false", text)
        self.assertIn("HF_HUB_OFFLINE: \"1\"", text)
        self.assertIn("SZL_EVAL_SOURCE_SHA:", text)
        self.assertIn("python -m tools.evaluate_peft_runtime run", text)
        self.assertIn("if: ${{ always() }}", text)
        self.assertIn("if-no-files-found: error", text)
        self.assertIn('python-version: ["3.11", "3.12"]', text)


if __name__ == "__main__":
    unittest.main()
