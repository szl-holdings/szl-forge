from pathlib import Path
import unittest

CONSOLE = Path(__file__).resolve().parents[1] / "frontier" / "sample-console" / "index.html"


class SampleConsoleTests(unittest.TestCase):
    def test_html_metrics_case_is_clickable_and_not_aligned(self):
        text = CONSOLE.read_text(encoding="utf-8")
        self.assertIn("html-metrics", text)
        self.assertIn("HTML 200 metrics", text)
        self.assertIn("numeric_boolean_rejected", text)
        self.assertIn("SAMPLE", text)
        self.assertIn("signer ABSENT", text)
        self.assertIn("szl-maskmod", text)
        self.assertIn("9/12", text)
        self.assertIn("production authorized false", text)
