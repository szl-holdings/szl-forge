"""Offline toy-row regressions for source-disjoint evaluation preparation."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "operational"))
import external_retrieval_data as data


def squad_rows():
    rows = []
    for group in range(9):
        for item in range(4):
            context = f"Article {group} paragraph {item} has answer Alpha."
            for answerable in (True, False):
                rows.append({
                    "id": f"g{group}-p{item}-a{int(answerable)}", "title": f"Article_{group}",
                    "context": context, "question": f"Question {group} {item} {answerable}?",
                    "answers": {"text": ["Alpha"] if answerable else [],
                                "answer_start": [context.index("Alpha")] if answerable else []},
                })
    return rows


def hotpot_rows():
    return [{"id": f"h{index}", "question": f"Which evidence for {index}?",
             "context": {"title": [f"First {index}", f"Second {index}", f"Distractor {index}"],
                         "sentences": [["Alpha. ", "Beta."], ["Gamma."], ["Unrelated."]]},
             "supporting_facts": {"title": [f"First {index}", f"Second {index}"], "sent_id": [1, 0]}}
            for index in range(4)]


def build(rows=None, hotpot=None):
    return data.build_payload(rows if rows is not None else squad_rows(),
                              hotpot if hotpot is not None else hotpot_rows(),
                              calibration_per_label=2, evaluation_per_label=3, hotpot_limit=2)


class ExternalRetrievalDataTests(unittest.TestCase):
    def test_exact_quotas_all_contexts_and_label_scope(self):
        payload = build()
        self.assertEqual(36, len(payload["documents"]))
        self.assertEqual(4, len(payload["calibration"]))
        self.assertEqual(6, len(payload["evaluation"]))
        self.assertEqual(2, len(payload["hotpot"]))
        self.assertFalse(payload["metadata"]["training_eligible"])
        self.assertFalse(payload["metadata"]["publication_eligible"])
        self.assertEqual("UNKNOWN", payload["metadata"]["pretraining_contamination"])
        self.assertIn("exact original paragraph", payload["metadata"]["human_label_derivation"]["squad_unsupported"])

    def test_input_order_does_not_change_payload(self):
        self.assertEqual(build(), build(list(reversed(squad_rows())), list(reversed(hotpot_rows()))))

    def test_article_groups_and_contexts_are_disjoint_for_selected_queries(self):
        payload = build()
        for field in ("group_id", "context_id", "id"):
            self.assertFalse({row[field] for row in payload["calibration"]} &
                             {row[field] for row in payload["evaluation"]})
        groups = payload["metadata"]["squad_source_groups"]
        self.assertEqual(3, len(groups["calibration_groups"]))
        self.assertEqual(6, len(groups["evaluation_groups"]))

    def test_whitespace_equivalent_context_unions_article_components(self):
        rows = squad_rows()
        duplicate = deepcopy(rows[0])
        duplicate.update(id="bridge", title="Article_1", context=rows[0]["context"].replace(" ", "  "), question="Unique bridge?")
        duplicate["answers"]["answer_start"] = [duplicate["context"].index("Alpha")]
        rows.append(duplicate)
        payload = build(rows)
        self.assertEqual(36, len(payload["documents"]))
        groups = payload["metadata"]["squad_source_groups"]["group_titles"]
        self.assertEqual(8, len(groups))
        self.assertTrue(any({"article 0", "article 1"} <= set(titles) for titles in groups.values()))
        self.assertTrue(any(doc["titles"] == ["Article_0", "Article_1"] for doc in payload["documents"]))

    def test_cross_group_duplicate_question_text_is_dropped_everywhere(self):
        rows = squad_rows()
        for row in rows:
            if row["title"] in {"Article_0", "Article_1"}:
                row["question"] = " SAME   Question "
        payload = build(rows)
        self.assertEqual(16, payload["metadata"]["squad_source_groups"]["excluded_queries"]["duplicate_question_text_crossing_groups"])
        self.assertNotIn(" SAME   Question ", [row["question"] for row in payload["calibration"] + payload["evaluation"]])

    def test_ineligible_questions_leave_corpus_intact(self):
        rows = squad_rows()
        rows[0]["question"], rows[1]["question"] = "", "q" * 513
        payload = build(rows)
        self.assertEqual(36, len(payload["documents"]))
        self.assertEqual(2, payload["metadata"]["squad_source_groups"]["excluded_queries"]["empty_nonstring_or_over_512_character_question"])

    def test_answer_span_must_match_original_not_normalized_context(self):
        rows = squad_rows()
        rows[0]["answers"]["answer_start"] = [0]
        with self.assertRaisesRegex(ValueError, "span mismatch"):
            build(rows)

    def test_malformed_answer_alignment_is_rejected(self):
        rows = squad_rows()
        rows[0]["answers"]["answer_start"] = []
        with self.assertRaisesRegex(ValueError, "aligned lists"):
            build(rows)

    def test_no_quota_fallback(self):
        with self.assertRaisesRegex(ValueError, "no quota fallback"):
            data.build_payload(squad_rows(), hotpot_rows(), calibration_per_label=100)

    def test_hotpot_positive_mapping_preserves_both_documents(self):
        payload = build()
        for case in payload["hotpot"]:
            self.assertEqual(2, len(case["positive_ids"]))
            self.assertTrue(set(case["positive_ids"]) <= {doc["id"] for doc in case["documents"]})
            self.assertNotIn("answerable", case)

    def test_missing_hotpot_support_and_bad_sentence_index_fail(self):
        for change in ("missing", "bad_index"):
            rows = hotpot_rows()
            for row in rows:
                if change == "missing":
                    row["supporting_facts"]["title"][0] = "Absent"
                else:
                    row["supporting_facts"]["sent_id"][0] = 99
            with self.subTest(change=change), self.assertRaises(ValueError):
                build(hotpot=rows)

    def test_payload_tampering_is_rejected(self):
        mutators = [
            lambda p: p["metadata"].update(training_eligible=True),
            lambda p: p["metadata"].update(publication_eligible=True),
            lambda p: p["metadata"].update(pretraining_contamination="CLEAN"),
            lambda p: p["evaluation"][0].update(group_id=p["calibration"][0]["group_id"]),
            lambda p: p["evaluation"][0].update(context_id=p["calibration"][0]["context_id"]),
            lambda p: p["documents"][0].update(text="tampered"),
            lambda p: p["hotpot"][0].update(positive_ids=["absent"]),
            lambda p: p["hotpot"][0].update(answerable=False),
            lambda p: p["calibration"].pop(),
        ]
        for index, mutate in enumerate(mutators):
            payload = build()
            mutate(payload)
            with self.subTest(index=index), self.assertRaises(ValueError):
                data.validate_payload(payload)

    def test_duplicate_source_ids_fail(self):
        rows = squad_rows()
        rows.append(deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate SQuAD id"):
            build(rows)

    def test_fixed_file_hash_rejected_before_parquet_import(self):
        with tempfile.TemporaryDirectory() as directory:
            lab = Path(directory)
            path = lab / data.PINS["squad_v2"]["path"]
            path.parent.mkdir(parents=True)
            path.write_bytes(b"wrong")
            with mock.patch("builtins.__import__", wraps=__import__) as importer:
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    data.prepare_payload(lab)
            self.assertFalse(any(call.args[0].startswith("pyarrow") for call in importer.call_args_list))

    def test_verified_bytes_and_receipt(self):
        content = b"small synthetic payload"
        pin = {"size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
               "path": "toy.bin", "dataset_id": "toy/fixture"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.bin"
            path.write_bytes(content)
            result, receipt = data._verified_file_bytes(path, pin)
        self.assertEqual(content, result)
        self.assertEqual(pin["sha256"], receipt["sha256"])
        self.assertTrue(receipt["verified"])


if __name__ == "__main__":
    unittest.main()
