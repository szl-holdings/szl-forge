from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import evidence_chain as evidence
import qualify_runtime as qualification
import train_candidate as training
import evaluate_candidate as evaluation


HERE = Path(__file__).resolve().parent


class QualificationContractTests(unittest.TestCase):
    def test_candidate_is_pinned_and_fail_closed(self) -> None:
        candidate = qualification.load_candidate(HERE / "candidate.json")
        self.assertEqual(40, len(candidate["canonical_base"]["revision"]))
        self.assertEqual(
            40,
            len(candidate["training_implementation"]["revision"]),
        )
        self.assertIs(candidate["publication_eligible"], False)
        self.assertIs(candidate["autonomy_eligible"], False)

    def test_candidate_rejects_premature_publication(self) -> None:
        candidate = json.loads(
            (HERE / "candidate.json").read_text(encoding="utf-8")
        )
        candidate["publication_eligible"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.json"
            path.write_text(json.dumps(candidate), encoding="utf-8")
            with self.assertRaisesRegex(
                qualification.QualificationError,
                "cannot be publishable",
            ):
                qualification.load_candidate(path)

    def test_report_digest_is_stable(self) -> None:
        first = {"b": 2, "a": [1, True]}
        second = {"a": [1, True], "b": 2}
        self.assertEqual(
            qualification.sha256_json(first),
            qualification.sha256_json(second),
        )

    def test_committed_curriculum_matches_manifest(self) -> None:
        digests, rows = training.curriculum_evidence()
        self.assertEqual(5, len(digests))
        self.assertEqual(31, len(rows))

    def test_vlm_conversion_preserves_roles_and_text(self) -> None:
        converted = training.vlm_conversation(
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        )
        self.assertEqual("user", converted["messages"][0]["role"])
        self.assertEqual(
            [{"type": "text", "text": "answer"}],
            converted["messages"][1]["content"],
        )

    def test_training_source_imports_unsloth_before_trl(self) -> None:
        source = (HERE / "train_candidate.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index("import unsloth"),
            source.index("from trl import SFTConfig"),
        )
        self.assertIn('"eos_token": processor.tokenizer.eos_token', source)
        self.assertIn('default=64', source)

    def test_held_out_curriculum_is_committed_and_pinned(self) -> None:
        digests, schema = evaluation.verify_held_out_curriculum()
        self.assertEqual(3, len(digests))
        self.assertEqual("object", schema["type"])
        self.assertEqual(5, len(evaluation.committed_jsonl("receiptagent/eval.jsonl")))
        self.assertEqual(
            6,
            len(evaluation.committed_jsonl("receiptagent/adversarial.jsonl")),
        )

    def test_refusal_classifier_is_strictly_prefix_based(self) -> None:
        self.assertTrue(evaluation.is_refusal("REFUSE: no authority"))
        self.assertTrue(evaluation.is_refusal("  refuse — no evidence"))
        self.assertFalse(evaluation.is_refusal("I should refuse this"))

    def test_draft_validation_uses_the_repository_schema(self) -> None:
        _, schema = evaluation.verify_held_out_curriculum()
        validator_class = evaluation.validator_for(schema)
        validator = validator_class(schema)
        row = evaluation.committed_jsonl("receiptagent/eval.jsonl")[0]
        expected = next(
            message["content"]
            for message in row["messages"]
            if message["role"] == "assistant"
        )
        valid, error = evaluation.validate_draft(expected, validator)
        self.assertTrue(valid)
        self.assertIsNone(error)
        invalid, invalid_error = evaluation.validate_draft("{}", validator)
        self.assertFalse(invalid)
        self.assertIn("ValidationError", invalid_error)

    def test_receipt_source_bundle_survives_protected_squash(self) -> None:
        receipt = json.loads(
            (
                HERE
                / "receipts"
                / "training_receipt.signed.json"
            ).read_text(encoding="utf-8")
        )
        real_run = evidence.subprocess.run

        def squash_without_intermediate_commit(args, **kwargs):
            if len(args) > 1 and args[1] == "cat-file":
                return evidence.subprocess.CompletedProcess(args, 1, b"", b"missing")
            return real_run(args, **kwargs)

        with mock.patch.object(
            evidence.subprocess,
            "run",
            side_effect=squash_without_intermediate_commit,
        ):
            evidence.verify_source_binding(receipt["payload"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
