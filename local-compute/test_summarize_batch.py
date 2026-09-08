import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from summarize_batch import summarize


class SummaryTests(unittest.TestCase):
    def make_linked_reports(self, root):
        ollama = root / "ollama.json"
        ollama.write_text(json.dumps({"schema": "szl.local-ollama-smoke/v2", "provider_cost_usd": 0, "models": []}))
        training = root / "training.json"
        value = {"schema": "szl.native-bf16-continuation/v1", "state": "MEASURED_LOCAL_CONTINUATION_COMPLETED",
                 "provider_cost_usd": 0, "candidate_id": "candidate-a"}
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        value["report_sha256"] = hashlib.sha256(encoded).hexdigest()
        training.write_text(json.dumps(value))
        native = root / "native.json"
        native_value = {"schema": "szl.native-reloaded-smoke/v1", "provider_cost_usd": 0,
                        "candidate_id": "candidate-a", "models": [],
                        "expected_training_report_sha256": hashlib.sha256(training.read_bytes()).hexdigest()}
        native.write_text(json.dumps(native_value))
        return ollama, training, native, native_value

    def test_native_reload_requires_exact_report_and_candidate_link(self):
        with tempfile.TemporaryDirectory() as temp:
            ollama, training, native, value = self.make_linked_reports(Path(temp))
            self.assertEqual(summarize(ollama, [training], native)["native_reload"]["candidate_id"], "candidate-a")
            for paths, change in (([], {}), ([training, training], {}),
                ([training], {"candidate_id": "candidate-b"}),
                ([training], {"expected_training_report_sha256": "a" * 64})):
                native.write_text(json.dumps({**value, **change}))
                with self.subTest(change=change, reports=len(paths)), self.assertRaisesRegex(ValueError, "must bind exactly one"):
                    summarize(ollama, paths, native)

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
