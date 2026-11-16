"""Adversarial evidence tests for the candidate data selector."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import unicodedata

MODULE = Path(__file__).parents[1] / "tools" / "salad_eval_selection.py"
_spec = importlib.util.spec_from_file_location("salad_eval_selection", MODULE)
selector = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(selector)


def example(*turns):
    return {
        "source": "fixture",
        "messages": [
            {"role": "user" if i % 2 == 0 else "assistant", "content": value}
            for i, value in enumerate(turns)
        ],
    }


def encoded(row, ending=b"\n"):
    return json.dumps(row, ensure_ascii=False).encode("utf-8") + ending


class SelectionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.train = self.root / "train.jsonl"
        self.eval = self.root / "eval.jsonl"

    def configure(self, train_rows, eval_rows, eval_bytes=None):
        self.train.write_bytes(b"".join(encoded(x) for x in train_rows))
        self.eval.write_bytes(
            eval_bytes if eval_bytes is not None else b"".join(encoded(x) for x in eval_rows)
        )
        return {
            "normalization": {
                "id": selector.NORMALIZER_ID,
                "unicode_data_version": unicodedata.unidata_version,
            },
            "dataset": {
                "repo": "test/fixture", "revision": "a" * 40,
                "files": {
                    "train": {"path": "train.jsonl", "sha256": selector.file_hash(self.train), "rows": len(train_rows)},
                    "eval": {"path": "eval.jsonl", "sha256": selector.file_hash(self.eval), "rows": len(eval_rows)},
                },
            },
        }

    def test_late_turn_overlap_is_counted_once_and_answers_do_not_select(self):
        long_prefix = "same beginning " * 1000
        spec = self.configure(
            [example("first shared turn", "answer one", "second shared turn", "answer two")],
            [
                example("fresh first", "fresh answer", "SECOND SHARED TURN", "different target"),
                example("first shared turn", "a", "second shared turn", "b"),
                example("unseen", "answer one"),
                example("unseen", "a different answer"),
                example(long_prefix + "suffix A", "target"),
                example(long_prefix + "suffix B", "target"),
            ],
        )
        selection, manifest = selector.build_selection(spec, self.train, self.eval)
        self.assertEqual(manifest["counts"]["excluded_eval_rows_for_training_turn_overlap"], 2)
        self.assertEqual(manifest["counts"]["excluded_residual_eval_duplicate_rows"], 1)
        self.assertEqual([x["record_index_0based"] for x in selection["selected_eval_rows"]], [2, 4, 5])
        self.assertFalse(any(manifest["retained_intersections"].values()))
        self.assertFalse(selection["selection_policy"]["answer_content_used_for_selection"])

    def test_unicode_and_whitespace_use_complete_normalized_turns(self):
        spec = self.configure(
            [example("Ａlpha \t BETA", "train answer")],
            [example("alpha beta", "different answer"), example("alpha beta plus", "keep")],
        )
        selection, manifest = selector.build_selection(spec, self.train, self.eval)
        self.assertEqual(manifest["counts"]["excluded_eval_rows_for_training_turn_overlap"], 1)
        self.assertEqual(selection["selected_eval_rows"][0]["record_index_0based"], 1)

    def test_physical_lines_record_indices_and_raw_newlines_are_preserved(self):
        first, second = example("unseen first", "answer"), example("unseen second", "answer")
        raw_first = encoded(first, b"\r\n")
        raw_second = encoded(second, b"")
        raw_eval = b"\r\n" + raw_first + b"\n" + raw_second
        spec = self.configure([example("train only", "answer")], [first, second], raw_eval)
        selection, manifest = selector.build_selection(spec, self.train, self.eval)
        ids = selection["selected_eval_rows"]
        self.assertEqual([x["physical_line_1based"] for x in ids], [2, 4])
        self.assertEqual([x["record_index_0based"] for x in ids], [0, 1])
        self.assertEqual(ids[0]["raw_line_sha256"], hashlib.sha256(raw_first).hexdigest())
        output = self.root / "selected.jsonl"
        selector.materialize_eval(self.eval, output, selection, manifest)
        self.assertEqual(output.read_bytes(), raw_first + raw_second)
        self.assertEqual(selector.file_hash(output), manifest["materialized_eval"]["sha256"])
        with self.assertRaises(ValueError):
            selector.materialize_eval(self.eval, output, selection, manifest)
        self.assertEqual(output.read_bytes(), raw_first + raw_second)

    def test_input_hash_count_and_unicode_drift_fail_closed(self):
        original = self.configure([example("train", "answer")], [example("eval", "answer")])
        for field, value in (("sha256", "0" * 64), ("rows", 2)):
            spec = copy.deepcopy(original)
            spec["dataset"]["files"]["train"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                selector.build_selection(spec, self.train, self.eval)
        spec = copy.deepcopy(original)
        spec["normalization"]["unicode_data_version"] = "unknown"
        with self.assertRaises(ValueError):
            selector.build_selection(spec, self.train, self.eval)

    def test_unsupported_messages_and_empty_earlier_targets_are_rejected(self):
        invalid = [
            {"source": "fixture", "messages": [{"role": "assistant", "content": "orphan"}]},
            example("first", "", "second", "final"),
            {"source": "fixture", "messages": [
                {"role": "user", "content": "question", "name": "hidden"},
                {"role": "assistant", "content": "answer"},
            ]},
            {"source": "fixture", "messages": [
                {"role": "tool", "content": "tool output"},
                {"role": "assistant", "content": "answer"},
            ]},
        ]
        for row in invalid:
            with self.subTest(row_index=invalid.index(row)):
                spec = self.configure([row], [example("eval", "answer")])
                with self.assertRaises(ValueError):
                    selector.build_selection(spec, self.train, self.eval)

    def test_selection_ignores_changed_answers_but_binds_raw_row_identity(self):
        spec = self.configure([example("train", "answer")], [example("eval", "target A")])
        selection_a, _ = selector.build_selection(spec, self.train, self.eval)
        spec = self.configure([example("train", "answer")], [example("eval", "target B")])
        selection_b, _ = selector.build_selection(spec, self.train, self.eval)
        a, b = selection_a["selected_eval_rows"][0], selection_b["selected_eval_rows"][0]
        self.assertEqual(a["user_context_sha256"], b["user_context_sha256"])
        self.assertNotEqual(a["raw_line_sha256"], b["raw_line_sha256"])

    def test_empty_tool_call_lists_are_inert_and_active_calls_rejected(self):
        inert = example("unseen", "answer")
        for message in inert["messages"]:
            message["tool_calls"] = []
        spec = self.configure([example("train", "answer")], [inert])
        selection, manifest = selector.build_selection(spec, self.train, self.eval)
        self.assertEqual(manifest["counts"]["selected_eval_rows"], 1)
        active = copy.deepcopy(inert)
        active["messages"][-1]["tool_calls"] = [{"name": "not-supported"}]
        spec = self.configure([example("train", "answer")], [active])
        with self.assertRaisesRegex(ValueError, "exactly empty tool_calls"):
            selector.build_selection(spec, self.train, self.eval)

    def test_existing_receipt_directory_cannot_be_overwritten(self):
        spec = self.configure([example("train", "answer")], [example("eval", "answer")])
        selection, manifest = selector.build_selection(spec, self.train, self.eval)
        output = self.root / "receipt"
        selector.write_receipts(output, selection, manifest)
        old = (output / "selection.json").read_bytes()
        with self.assertRaises(FileExistsError):
            selector.write_receipts(output, selection, manifest)
        self.assertEqual((output / "selection.json").read_bytes(), old)

    def test_selection_manifest_mismatch_fails_before_materialization(self):
        spec = self.configure([example("train", "answer")], [example("eval", "answer")])
        selection, manifest = selector.build_selection(spec, self.train, self.eval)
        altered = copy.deepcopy(selection)
        altered["selected_eval_rows"][0]["raw_line_sha256"] = "0" * 64
        destination = self.root / "materialized.jsonl"
        with self.assertRaisesRegex(ValueError, "not bound"):
            selector.materialize_eval(self.eval, destination, altered, manifest)
        self.assertFalse(destination.exists())
        altered = copy.deepcopy(manifest)
        altered["normalization"]["unicode_data_version"] = "different"
        with self.assertRaisesRegex(ValueError, "normalizer"):
            selector.materialize_eval(self.eval, destination, selection, altered)
        self.assertFalse(destination.exists())

    def test_selected_context_is_checked_against_actual_row(self):
        spec = self.configure([example("train", "answer")], [example("eval", "answer")])
        selection, manifest = selector.build_selection(spec, self.train, self.eval)
        selection["selected_eval_rows"][0]["user_context_sha256"] = "0" * 64
        manifest["selection_canonical_sha256"] = selector.value_hash(selection)
        destination = self.root / "materialized.jsonl"
        with self.assertRaisesRegex(ValueError, "identity changed"):
            selector.materialize_eval(self.eval, destination, selection, manifest)
        self.assertFalse(destination.exists())

    def test_consumed_bytes_are_bound_even_when_input_changes_after_initial_hash(self):
        spec = self.configure([example("train", "answer")], [example("eval", "answer")])
        original = selector.file_hash
        changed = False
        def hash_then_change(path):
            nonlocal changed
            result = original(path)
            if Path(path) == self.train and not changed:
                self.train.write_bytes(encoded(example("changed training input", "answer")))
                changed = True
            return result
        with patch.object(selector, "file_hash", side_effect=hash_then_change):
            with self.assertRaisesRegex(ValueError, "consumed input bytes"):
                selector.build_selection(spec, self.train, self.eval)

    def test_duplicate_selection_identities_and_byte_mismatch_are_rejected(self):
        spec = self.configure([example("train", "answer")], [example("eval one", "a"), example("eval two", "b")])
        selection, manifest = selector.build_selection(spec, self.train, self.eval)
        duplicate = copy.deepcopy(selection)
        duplicate["selected_eval_rows"][1] = copy.deepcopy(duplicate["selected_eval_rows"][0])
        bad_manifest = copy.deepcopy(manifest)
        bad_manifest["selection_canonical_sha256"] = selector.value_hash(duplicate)
        destination = self.root / "materialized.jsonl"
        with self.assertRaisesRegex(ValueError, "unique and in original order"):
            selector.materialize_eval(self.eval, destination, duplicate, bad_manifest)
        self.assertFalse(destination.exists())
        manifest["materialized_eval"]["bytes"] += 1
        with self.assertRaisesRegex(ValueError, "byte count mismatch"):
            selector.materialize_eval(self.eval, destination, selection, manifest)
        self.assertFalse(destination.exists())

    def test_no_rows_remaining_is_not_a_pass(self):
        spec = self.configure([example("same prompt", "a")], [example("same prompt", "b")])
        with self.assertRaisesRegex(ValueError, "no evaluation rows"):
            selector.build_selection(spec, self.train, self.eval)


if __name__ == "__main__":
    unittest.main()
