"""Pure-CPU falsifiers for the explicit document-boundary witness."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from training.document_boundary_witness import (
    BoundaryWitnessError,
    audit_document_boundaries,
    read_manifest,
)


def fixture():
    # Both documents deliberately lack EOS: EOS-only truncation is unsafe.
    return {"tokens": [11, 12, 13, 21, 22, 23, 24], "document_lengths": [3, 4],
            "block_size": 2, "eos_token_id": 0, "source_revision": "synthetic-only"}


class BoundaryWitnessTests(unittest.TestCase):
    def test_conserves_all_tokens_and_keeps_documents_separate(self):
        r = audit_document_boundaries(fixture())
        self.assertEqual("METADATA_CONSERVATION_PASSED", r["status"])
        self.assertEqual((7, 7), (r["nonpadding_token_utilization_numerator"], r["nonpadding_token_utilization_denominator"]))
        self.assertEqual([0, 0, 1, 1], [s["document"] for s in r["segments"]])
        self.assertEqual([(0, 2), (2, 3), (3, 5), (5, 7)], [(s["start"], s["stop"]) for s in r["segments"]])
        self.assertEqual([0, 1], r["eos_unsafe_document_indices"])
        self.assertEqual("UNSAFE_FOR_EOS_ONLY_INFERENCE", r["eos_only_inference"])
        self.assertFalse(r["training_consumption_verified"])
        self.assertEqual("NONE", r["training_authority"])

    def test_digest_replay_is_deterministic_and_input_bound(self):
        a = audit_document_boundaries(fixture())
        b = audit_document_boundaries(fixture())
        self.assertEqual(a, b)
        self.assertEqual(64, len(a["receipt_sha256"]))
        mutated = fixture()
        mutated["tokens"][3] = 99
        self.assertNotEqual(a["token_array_sha256"], audit_document_boundaries(mutated)["token_array_sha256"])

    def test_explicit_terminal_eos_is_not_assumed_for_source_custody(self):
        x = {"tokens": [11, 0, 22, 0], "document_lengths": [2, 2], "block_size": 3, "eos_token_id": 0}
        r = audit_document_boundaries(x)
        self.assertEqual("TERMINATORS_OBSERVED_NOT_SOURCE_PROOF", r["eos_only_inference"])
        self.assertFalse(r["source_identity_verified"])

    def test_without_eos_id_do_not_assume_terminators(self):
        x = fixture()
        del x["eos_token_id"]
        self.assertEqual("UNKNOWN_NO_EOS_ID", audit_document_boundaries(x)["eos_only_inference"])

    def test_reject_missing_forged_or_ambiguous_boundaries(self):
        bad = fixture()
        del bad["document_lengths"]
        for x in (bad, {**fixture(), "document_lengths": [7, 1]},
                  {**fixture(), "document_lengths": [3, 3]},
                  {**fixture(), "document_lengths": [3, True, 3]},
                  {**fixture(), "document_lengths": [0, 7]}):
            with self.subTest(data=x), self.assertRaises(BoundaryWitnessError):
                audit_document_boundaries(x)

    def test_reject_invalid_types_fields_and_excessive_segments(self):
        for x in ({**fixture(), "tokens": [11, True, 13, 21, 22, 23, 24]},
                  {**fixture(), "block_size": False},
                  {**fixture(), "block_size": 0},
                  {**fixture(), "eos_token_id": 0.1},
                  {**fixture(), "extra": "not permitted"},
                  {**fixture(), "source_revision": chr(10)},
                  {"tokens": list(range(9000)), "document_lengths": [9000], "block_size": 1}):
            with self.subTest(data=str(x)[:80]), self.assertRaises(BoundaryWitnessError):
                audit_document_boundaries(x)

    def test_duplicate_json_keys_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fixture.json"
            for raw in (b'{"tokens":[],"tokens":[]}', b'{"tokens":[NaN]}'):
                p.write_bytes(raw)
                with self.assertRaises(BoundaryWitnessError):
                    read_manifest(p)

    def test_cli_rejects_oversized_json_integer_with_denial_contract(self):
        # A bounded JSON file may still contain a numeric literal above
        # Python's integer-digit limit; it must not escape as a traceback.
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fixture.json"
            p.write_text('{"tokens":[' + "9" * 5000 +
                         '],"document_lengths":[1],"block_size":1}', encoding="utf-8")
            cmd = [sys.executable, "training/document_boundary_witness.py",
                   "--input", str(p)]
            denied = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(2, denied.returncode, denied.stderr)
            self.assertEqual("", denied.stderr)
            self.assertEqual(
                {"schema": "szl.document-boundary-witness/v1",
                 "status": "DENY", "reason": "INVALID_OR_UNREADABLE_JSON"},
                json.loads(denied.stdout),
            )

    def test_duplicate_keys_preserve_specific_denial_reason(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fixture.json"
            p.write_bytes(b'{"tokens":[],"tokens":[]}')
            with self.assertRaisesRegex(BoundaryWitnessError, "^DUPLICATE_JSON_KEY$"):
                read_manifest(p)

    def test_cli_uses_exit_two_on_denial_and_never_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fixture.json"
            p.write_text(json.dumps(fixture()))
            cmd = [sys.executable, "training/document_boundary_witness.py", "--input", str(p)]
            ok = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(0, ok.returncode, ok.stderr)
            self.assertEqual("METADATA_CONSERVATION_PASSED", json.loads(ok.stdout)["status"])
            p.write_text("{}")
            no = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(2, no.returncode, no.stderr)
            self.assertEqual("DENY", json.loads(no.stdout)["status"])
            self.assertEqual(["fixture.json"], sorted(x.name for x in Path(d).iterdir()))


if __name__ == "__main__":
    unittest.main()
