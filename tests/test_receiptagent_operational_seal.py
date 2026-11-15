"""Regression tests for durable one-way evaluation evidence; no model or test data."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "operational/evaluate_receiptagent_v3.py"
spec = importlib.util.spec_from_file_location("receipt_operational_seal", SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ReceiptSealTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.test = self.root / "synthetic-tests.jsonl"
        self.test.write_text(json.dumps({"id": "synthetic-1", "prompt": "Synthetic fixture.",
                                        "must_contain": ["DRAFT"], "must_not_contain": ["EXECUTED"]}) + "\n")
        self.receipt = self.root / "fresh-out/receipt.json"
        self.patch = mock.patch.object(runner, "RECEIPT", self.receipt)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def run_main(self, *args):
        output = io.StringIO()
        with mock.patch("sys.argv", [str(SOURCE), "--test", str(self.test), *args]), contextlib.redirect_stdout(output):
            code = runner.main()
        return code, output.getvalue()

    def test_status_is_read_only_and_does_not_infer_seal_from_file(self):
        code, output = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["status"], "UNKNOWN")
        self.assertIsNone(json.loads(output)["seal_intact"])
        self.assertFalse(self.receipt.parent.exists())

    def test_unseal_survives_default_status_check_byte_for_byte(self):
        self.receipt.parent.mkdir()
        with mock.patch.object(runner, "generate", return_value="DRAFT"):
            self.assertEqual(self.run_main("--unseal")[0], 0)
        before = self.receipt.read_bytes()
        code, output = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(self.receipt.read_bytes(), before)
        self.assertFalse(json.loads(output)["seal_intact"])
        self.assertIn("unsealed_at", json.loads(output))

    def test_unseal_is_persisted_before_model_request(self):
        def generate(*_):
            value = json.loads(self.receipt.read_text())
            self.assertEqual(value["status"], "UNSEALED_EVALUATION_PENDING")
            self.assertIn("unsealed_at", value)
            return "DRAFT"
        with mock.patch.object(runner, "generate", side_effect=generate):
            self.assertEqual(self.run_main("--unseal")[0], 0)

    def test_empty_or_vacuous_tests_never_report_measured(self):
        for content in ["", json.dumps({"id": "case", "prompt": "Hello"}),
                        json.dumps({"id": "case", "prompt": "Hello", "must_contain": [""]})]:
            with self.subTest(content=content):
                self.receipt.unlink(missing_ok=True)
                self.test.write_text(content)
                with mock.patch.object(runner, "generate") as generate:
                    self.assertNotEqual(self.run_main("--unseal")[0], 0)
                    generate.assert_not_called()
                result = json.loads(self.receipt.read_text())
                self.assertEqual(result["status"], "UNKNOWN")
                self.assertIn("unsealed_at", result)

    def test_endpoint_error_preserves_unseal(self):
        with mock.patch.object(runner, "generate", side_effect=RuntimeError("offline")):
            self.assertNotEqual(self.run_main("--unseal")[0], 0)
        result = json.loads(self.receipt.read_text())
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIn("unsealed_at", result)
        self.assertFalse(result["seal_intact"])

    def test_second_unseal_cannot_overwrite_original(self):
        with mock.patch.object(runner, "generate", return_value="DRAFT"):
            self.assertEqual(self.run_main("--unseal")[0], 0)
        before = self.receipt.read_bytes()
        with mock.patch.object(runner, "generate") as generate:
            self.assertNotEqual(self.run_main("--unseal", "--model", "different-candidate")[0], 0)
            generate.assert_not_called()
        self.assertEqual(self.receipt.read_bytes(), before)

    def test_corrupt_prior_receipt_is_not_overwritten(self):
        self.receipt.parent.mkdir()
        self.receipt.write_text("not JSON")
        self.assertNotEqual(self.run_main("--unseal")[0], 0)
        self.assertEqual(self.receipt.read_text(), "not JSON")

    def test_legacy_file_existence_claim_is_not_repeated(self):
        self.receipt.parent.mkdir()
        self.receipt.write_text('{"status":"SEALED","seal_intact":true}')
        before = self.receipt.read_bytes()
        _, output = self.run_main()
        self.assertEqual(json.loads(output)["status"], "UNKNOWN")
        self.assertEqual(self.receipt.read_bytes(), before)

    def test_concurrent_or_interrupted_run_lock_blocks_new_unseal(self):
        self.receipt.parent.mkdir()
        lock = self.receipt.with_suffix(".json.lock")
        lock.write_text("")
        with mock.patch.object(runner, "generate") as generate:
            self.assertNotEqual(self.run_main("--unseal")[0], 0)
            generate.assert_not_called()
        self.assertTrue(lock.exists())
        self.assertFalse(self.receipt.exists())


if __name__ == "__main__":
    unittest.main()
