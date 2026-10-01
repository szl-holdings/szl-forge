"""Regression checks for exact-location runtime evidence and receipt tampering."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("foundation_runtime_witness", Path(__file__).with_name("verify_public_runtime.py"))
witness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(witness)


class Response:
    status = 200

    def __init__(self, location):
        self.location = location

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def geturl(self):
        return self.location

    def read(self, _limit):
        return b'{"ready":true}'


class WitnessBoundaryTests(unittest.TestCase):
    def test_redirect_cannot_be_attributed_to_declared_runtime(self):
        origin = "https://" + witness.HOST
        with patch.object(witness, "urlopen", return_value=Response("https://other.example/readyz")):
            with self.assertRaisesRegex(ValueError, "redirected"):
                witness.fetch(origin, "/readyz")

    def test_exact_endpoint_is_accepted(self):
        origin = "https://" + witness.HOST
        with patch.object(witness, "urlopen", return_value=Response(origin + "/readyz")):
            self.assertEqual(witness.fetch(origin, "/readyz"), (200, {"ready": True}))

    def test_result_tampering_and_wrong_source_are_rejected(self):
        request = {"seed": 1, "index": 2, "family": "shared_bias", "policy": "learned", "model_seed": 17}
        source = "a" * 40
        receipt = {"status": "COMPLETE", "request": request, "request_sha256": witness.digest(request),
                   "result": {"policy": "learned", "history": []}, "archive_sha256": witness.ARCHIVE,
                   "scientific_overall_gate": "FAILED", "unsigned": True, "authenticity_established": False,
                   "id": "b" * 32, "source": {"revision": source},
                   "binding": {"checkpoint_sha256": "c" * 64, "model_fingerprint": "d" * 64}}
        receipt["result_sha256"] = witness.digest(receipt["result"])
        receipt["receipt_sha256"] = witness.digest(receipt)
        witness.verify_receipt(receipt, request, source)
        altered = json.loads(json.dumps(receipt))
        altered["result"]["history"].append({"fabricated": True})
        with self.assertRaises(ValueError):
            witness.verify_receipt(altered, request, source)
        with self.assertRaises(ValueError):
            witness.verify_receipt(receipt, request, "e" * 40)


if __name__ == "__main__":
    unittest.main()
