"""Mutation evidence must distinguish assertion failures from broken execution."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / "examples/evidence_to_decision/mutation_sweep.py"
SPEC = importlib.util.spec_from_file_location("evidence_mutation_sweep", PATH)
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


def completed(outcome="PASS", tests=2, skipped=0):
    return {"rc": 0 if outcome == "PASS" else 1, "last": outcome, "outcome": outcome,
            "coverage": {"tests": tests, "skipped": skipped, "expected_failures": 0}}


def layers(outcome="PASS"):
    return {key: completed(outcome) for key in ("unit", "base_acceptance", "acceptance")}


class UnitEvidenceTests(unittest.TestCase):
    def check(self, rc, output, expected):
        self.assertEqual(sweep.unit_outcome(rc, output)["outcome"], expected)

    def test_assertion_is_a_kill(self):
        self.check(1, "Ran 2 tests in 0.001s\n\nFAILED (failures=1)", "KILLED")

    def test_error_is_invalid(self):
        self.check(1, "Ran 2 tests in 0.001s\n\nFAILED (errors=1)", "INVALID")

    def test_assertion_plus_error_is_invalid(self):
        self.check(1, "Ran 2 tests in 0.001s\n\nFAILED (failures=1, errors=1)", "INVALID")

    def test_crash_and_zero_tests_are_invalid(self):
        for rc, out in ((1, "ImportError: missing dependency"), (0, "Ran 0 tests in 0s\nOK"),
                        (-9, "Ran 2 tests in 0s\nFAILED (failures=1)")):
            with self.subTest(rc=rc, out=out):
                self.check(rc, out, "INVALID")

    def test_disabled_planted_failure_is_disclosed(self):
        result = sweep.unit_outcome(0, "Ran 3 tests in 0.001s\n\nOK (skipped=1)")
        self.assertEqual(result["outcome"], "PASS")
        self.assertEqual(result["coverage"], {"tests": 3, "skipped": 1, "expected_failures": 0})

    def test_all_skipped_is_invalid(self):
        self.check(0, "Ran 2 tests in 0.001s\n\nOK (skipped=2)", "INVALID")

    def test_real_assertion_and_crash_subprocesses(self):
        for statement, expected in (("self.assertEqual(1, 2)", "KILLED"),
                                    ("raise RuntimeError('broken fixture')", "INVALID")):
            with self.subTest(statement=statement), tempfile.TemporaryDirectory() as root:
                Path(root, "test_case.py").write_text(
                    "import unittest\nclass Case(unittest.TestCase):\n    def test_case(self):\n        "
                    + statement + "\n", encoding="utf-8")
                self.assertEqual(sweep.run(sweep.UNIT, root)["outcome"], expected)

    def test_timeout_is_invalid(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "test_case.py").write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
            result = sweep.run(sweep.UNIT, root, timeout=0.1)
            self.assertEqual((result["rc"], result["outcome"]), ("TIMEOUT", "INVALID"))

    def test_completed_guard_error_is_a_kill(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, sweep.TARGET).write_text("def guard():\n    return {}['unknown']\n", encoding="utf-8")
            Path(root, "test_case.py").write_text(
                "import unittest\nimport eight_case_harness as h\n"
                "class Case(unittest.TestCase):\n    def test_guard(self):\n        h.guard()\n", encoding="utf-8")
            self.assertEqual(sweep.run(sweep.UNIT, root)["outcome"], "KILLED")

    def test_fixture_callback_error_is_not_attributed_to_guard(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, sweep.TARGET).write_text("def guard(callback):\n    callback()\n", encoding="utf-8")
            Path(root, "test_case.py").write_text(
                "import unittest\nimport eight_case_harness as h\n"
                "def broken_fixture():\n    raise RuntimeError('fixture failed')\n"
                "class Case(unittest.TestCase):\n    def test_guard(self):\n        h.guard(broken_fixture)\n",
                encoding="utf-8")
            self.assertEqual(sweep.run(sweep.UNIT, root)["outcome"], "INVALID")

    def test_setup_error_is_invalid_even_inside_target(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, sweep.TARGET).write_text("def guard():\n    return {}['unknown']\n", encoding="utf-8")
            Path(root, "test_case.py").write_text(
                "import unittest\nimport eight_case_harness as h\n"
                "class Case(unittest.TestCase):\n    def setUp(self):\n        h.guard()\n"
                "    def test_guard(self):\n        pass\n", encoding="utf-8")
            self.assertEqual(sweep.run(sweep.UNIT, root)["outcome"], "INVALID")


class AcceptanceEvidenceTests(unittest.TestCase):
    def report(self):
        return {"schema": "szl.e2d.acceptance/v0", "verdict": "ACCEPTANCE_NOT_MET",
                "guard_checks": [{"pass": False, "checks": {"decision": False}}],
                "negative_controls": [{"pass": True, "harness_rc": 3}],
                "guard_cases_MEASURED_against_spy": [{"case_id": "synthetic"}],
                "guard_deterministic": True, "base_eight_case_rc": 0}

    def outcome(self, r, rc=1):
        return sweep.acceptance_outcome(rc, r, "acceptance.py")["outcome"]

    def test_completed_behavioral_failure_is_kill(self):
        self.assertEqual(self.outcome(self.report()), "KILLED")

    def test_caught_guard_exception_is_a_completed_behavioral_failure(self):
        r = self.report()
        r["guard_cases_MEASURED_against_spy"][0]["error"] = "KeyError"
        self.assertEqual(self.outcome(r), "KILLED")
        r = self.report()
        r["guard_checks"][0]["checks"] = {"no_exception": False}
        self.assertEqual(self.outcome(r), "KILLED")

    def test_negative_control_exception_is_invalid(self):
        r = self.report()
        r["negative_controls"][0].update({"pass": False, "harness_rc": "EXC:RuntimeError"})
        self.assertEqual(self.outcome(r), "INVALID")

    def test_incomplete_or_inconsistent_reports_are_invalid(self):
        for key, value in (("schema", "wrong"), ("guard_checks", []),
                           ("guard_deterministic", False), ("base_eight_case_rc", 3),
                           ("verdict", "ACCEPTANCE_MET_ON_SPY")):
            r = self.report()
            r[key] = value
            with self.subTest(key=key):
                self.assertEqual(self.outcome(r), "INVALID")

    def test_bare_exit_one_does_not_count(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "acceptance.py").write_text("raise SystemExit(1)\n", encoding="utf-8")
            self.assertEqual(sweep.run(["acceptance.py"], root)["outcome"], "INVALID")


class SweepGateTests(unittest.TestCase):
    def run_fixture(self, clean, mutant=None):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root, "source")
            source.mkdir()
            Path(source, sweep.TARGET).write_text("if True:  # GUARD:H01 test\n    pass\n", encoding="utf-8")
            out = Path(root, "result.json")
            observations = [clean] if mutant is None else [clean, mutant]
            with patch.object(sweep, "HERE", str(source)), patch.object(sweep, "layers", side_effect=observations) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = sweep.main(["--scratch", root, "--out", str(out)])
                count = run.call_count
            report = json.loads(out.read_text()) if out.exists() else None
            self.assertFalse(any(p.name.startswith("e2d_sweep_") for p in Path(root).iterdir()))
            return rc, report, count

    def test_red_baseline_never_runs_mutants(self):
        clean = layers()
        clean["unit"] = completed("KILLED")
        rc, report, count = self.run_fixture(clean)
        self.assertEqual((rc, report, count), (3, None, 1))

    def test_invalid_baseline_never_runs_mutants(self):
        clean = layers()
        clean["unit"] = {"rc": "TIMEOUT", "outcome": "INVALID"}
        self.assertEqual(self.run_fixture(clean), (3, None, 1))

    def test_valid_behavioral_kills_still_pass(self):
        rc, report, count = self.run_fixture(layers(), layers("KILLED"))
        self.assertEqual((rc, count, report["invalid"]), (0, 2, []))

    def test_timeout_mutant_never_counts_as_kill(self):
        mutant = layers("KILLED")
        mutant["unit"] = {"rc": "TIMEOUT", "outcome": "INVALID"}
        rc, report, _ = self.run_fixture(layers(), mutant)
        self.assertEqual((rc, report["killed"]["unit"], report["invalid"]), (1, 0, ["H01"]))

    def test_changed_count_or_new_skip_is_invalid(self):
        for key, value in (("tests", 1), ("skipped", 1)):
            mutant = layers("KILLED")
            mutant["unit"]["coverage"][key] = value
            with self.subTest(key=key):
                rc, report, _ = self.run_fixture(layers(), mutant)
                self.assertEqual((rc, report["invalid"]), (1, ["H01"]))

    def test_baseline_skips_remain_visible_without_fake_kills(self):
        clean, mutant = layers(), layers("KILLED")
        clean["unit"]["coverage"]["skipped"] = 1
        mutant["unit"]["coverage"]["skipped"] = 1
        rc, report, _ = self.run_fixture(clean, mutant)
        self.assertEqual(rc, 0)
        self.assertEqual(report["clean"]["unit"]["coverage"]["skipped"], 1)


if __name__ == "__main__":
    unittest.main()
