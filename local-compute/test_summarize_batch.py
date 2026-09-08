import json
from pathlib import Path
import tempfile
import unittest

from summarize_batch import summarize


class SummaryTests(unittest.TestCase):
    def test_unknown_metadata_is_not_exported_or_promoted(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ollama.json"
            path.write_text(json.dumps({"schema": "szl.local-ollama-smoke/v2", "provider_cost_usd": 0,
                "private_debug": "not-for-export", "models": [{"name": "fixture", "digest": "a" * 64,
                "state": "UNAVAILABLE", "passed": 0, "completed": 0, "private_debug": "not-for-export"}]}))
            result = summarize(path, [])
            self.assertNotIn("not-for-export", json.dumps(result))
            self.assertFalse(result["publication_eligible"])
            self.assertEqual(result["ollama"]["models"][0]["state"], "UNAVAILABLE")

    def test_paid_cost_is_not_relabeled_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ollama.json"
            path.write_text(json.dumps({"schema": "szl.local-ollama-smoke/v2", "provider_cost_usd": 5, "models": []}))
            with self.assertRaisesRegex(ValueError, "zero-provider-cost"):
                summarize(path, [])


if __name__ == "__main__":
    unittest.main()
