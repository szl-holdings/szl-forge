from pathlib import Path
import tempfile
import unittest

from evaluate_native_smoke import admit_candidate, benchmark, score, training


class NativeEvaluationTests(unittest.TestCase):
    def test_correct_text_without_eos_is_not_a_pass(self):
        self.assertFalse(score(benchmark.CASES[0], "102", False))
        self.assertTrue(score(benchmark.CASES[0], "102", True))
        self.assertFalse(score(benchmark.CASES[0], "wrong", True))

    def test_rewritten_report_cannot_authenticate_itself(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            path.write_text('{"state":"MEASURED_LOCAL_CONTINUATION_COMPLETED"}')
            with self.assertRaisesRegex(training.GateError, "REPORT_NOT_AUTHENTICATED"):
                admit_candidate(Path(temp), path, "a" * 64)
            with self.assertRaisesRegex(training.GateError, "TRUSTED_REPORT_DIGEST_REQUIRED"):
                admit_candidate(Path(temp), path, "")


if __name__ == "__main__":
    unittest.main()
