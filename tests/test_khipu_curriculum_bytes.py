"""Raw checkout-byte guards for existing synthetic Khipu curriculum artifacts.

These checks establish file integrity and JSON parseability only: no training,
inference, model qualification, or scientific correctness is measured here.
"""

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CURRICULUM = ROOT / "khipu"
EXPECTED_SPLITS = {
    "train.jsonl": 15,
    "eval.jsonl": 5,
    "train.abstain.jsonl": 8,
    "adversarial.jsonl": 6,
}
SCHEMA_FILE = "khipu.schema.json"


def raw_sha256(raw):
    """Hash the received bytes, without text decoding or newline conversion."""
    return hashlib.sha256(raw).hexdigest()


class KhipuCurriculumByteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((CURRICULUM / "manifest.json").read_bytes())

    def test_manifest_binds_exact_five_files_and_existing_split_counts(self):
        files = self.manifest["files"]
        self.assertEqual(set(files), set(EXPECTED_SPLITS) | {SCHEMA_FILE})
        for name, expected_rows in EXPECTED_SPLITS.items():
            with self.subTest(file=name):
                self.assertEqual(files[name]["rows"], expected_rows)
        self.assertEqual(files[SCHEMA_FILE]["kind"], "schema")
        self.assertEqual(files[SCHEMA_FILE]["rows"], 0)
        contract = self.manifest["contract"]
        self.assertEqual(contract["outputSchemaFile"], SCHEMA_FILE)
        self.assertEqual(contract["outputSchemaSha256"], files[SCHEMA_FILE]["sha256"])

    def test_every_checkout_file_matches_manifest_raw_byte_sha256(self):
        for name, entry in self.manifest["files"].items():
            with self.subTest(file=name):
                raw = (CURRICULUM / name).read_bytes()
                self.assertEqual(raw_sha256(raw), entry["sha256"],
                                 f"{name}: raw checkout bytes differ from manifest")

    def test_all_five_files_have_lf_newlines_without_carriage_returns(self):
        for name in self.manifest["files"]:
            with self.subTest(file=name):
                raw = (CURRICULUM / name).read_bytes()
                self.assertFalse(b"\r" in raw, f"{name}: CR byte violates LF policy")
                self.assertTrue(raw.endswith(b"\n"), f"{name}: final LF is required")

    def test_split_row_counts_and_each_row_parse_as_json_object(self):
        for name, expected_rows in EXPECTED_SPLITS.items():
            with self.subTest(file=name):
                rows = (CURRICULUM / name).read_bytes().splitlines()
                self.assertEqual(len(rows), expected_rows)
                self.assertTrue(all(rows), f"{name}: blank curriculum row")
                for number, raw in enumerate(rows, start=1):
                    with self.subTest(file=name, row=number):
                        row = json.loads(raw.decode("utf-8"))
                        self.assertIsInstance(row, dict)
                        self.assertIsInstance(row["messages"], list)

    def test_schema_parses_with_existing_profile_and_required_fields(self):
        schema = json.loads((CURRICULUM / SCHEMA_FILE).read_bytes())
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["properties"]["capabilityProfile"]["const"],
                         self.manifest["contract"]["capabilityProfile"])
        self.assertIsInstance(schema["required"], list)
        self.assertFalse(schema["additionalProperties"])

    def test_crlf_fixture_fails_same_hash_even_when_json_values_are_equal(self):
        lf = b'{"scope":"synthetic-byte-fixture"}\n'
        crlf = b'{"scope":"synthetic-byte-fixture"}\r\n'
        expected_lf_sha256 = raw_sha256(lf)
        self.assertEqual(json.loads(lf), json.loads(crlf))
        self.assertEqual(raw_sha256(lf), expected_lf_sha256)
        self.assertNotEqual(raw_sha256(crlf), expected_lf_sha256)


if __name__ == "__main__":
    unittest.main()
