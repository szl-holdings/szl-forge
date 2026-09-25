"""Literal synthetic tests only; no benchmark inputs or model downloads."""
import copy
import importlib.util
import math
from pathlib import Path
import threading
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "operational" / "external_retrieval_reader.py"
SPEC = importlib.util.spec_from_file_location("external_reader_under_test", SOURCE)
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


def window(*, null=2.0, starts=None, ends=None):
    return {"offsets": [(0, 0), (0, 0), (0, 3), (4, 7), (8, 13), (0, 0)],
            "context_mask": [False, False, True, True, True, False],
            "start_logits": [null / 2, 100.0, 1.0, 5.0, 0.0, 100.0] if starts is None else starts,
            "end_logits": [null / 2, 100.0, 1.0, 5.0, 0.0, 100.0] if ends is None else ends,
            "cls_index": 0}


class DecodeTests(unittest.TestCase):
    def test_literal_span_margin_and_offsets(self):
        result = reader.decode_windows("The red apple", [window()])
        self.assertEqual({"prediction": "red", "margin": 8.0,
                          "start": 4, "end": 7, "windows": 1}, result)

    def test_minimum_null_across_same_context_windows(self):
        first = window(null=20.0)
        second = window(null=-6.0, starts=[-3, 0, 0, 1, 0, 0], ends=[-3, 0, 0, 1, 0, 0])
        result = reader.decode_windows("The red apple", [first, second])
        self.assertEqual("red", result["prediction"])
        self.assertEqual(16.0, result["margin"])
        self.assertEqual(2, result["windows"])

    def test_independent_document_calls_never_share_null(self):
        first = reader.decode_windows("The red apple", [window(null=-20.0)])
        second = reader.decode_windows("The tan apple", [window(null=20.0)])
        again = reader.decode_windows("The red apple", [window(null=-20.0)])
        self.assertEqual(30.0, first["margin"])
        self.assertEqual(-10.0, second["margin"])
        self.assertEqual("tan", second["prediction"])
        self.assertEqual(first, again)

    def test_best_span_can_come_from_another_window(self):
        later = window(starts=[1, 0, 0, 0, 8, 0], ends=[1, 0, 0, 0, 9, 0])
        result = reader.decode_windows("The red apple", [window(), later])
        self.assertEqual("apple", result["prediction"])
        self.assertEqual(15.0, result["margin"])

    def test_equal_scores_choose_earliest_shortest_span(self):
        item = window(starts=[0, 0, 5, 5, 5, 0], ends=[0, 0, 5, 5, 5, 0])
        self.assertEqual("The", reader.decode_windows("The red apple", [item])["prediction"])

    def test_null_index_need_not_be_zero(self):
        item = window()
        item["cls_index"] = 1
        item["start_logits"][0] = item["end_logits"][0] = 40
        item["start_logits"][1] = item["end_logits"][1] = 3
        self.assertEqual(4.0, reader.decode_windows("The red apple", [item])["margin"])

    def test_offsets_are_literal_unicode_character_indices(self):
        item = {"offsets": [(0, 0), (0, 1), (2, 6)], "context_mask": [False, True, True],
                "start_logits": [0, 1, 5], "end_logits": [0, 1, 5], "cls_index": 0}
        context = "\U0001f34e caf\u00e9"
        result = reader.decode_windows(context, [item])
        self.assertEqual("caf\u00e9", result["prediction"])
        self.assertEqual(context[result["start"]:result["end"]], result["prediction"])

    def test_zero_width_tokens_are_not_answer_endpoints(self):
        item = window()
        item["offsets"][3] = (4, 4)
        item["start_logits"][3] = item["end_logits"][3] = 999
        result = reader.decode_windows("The red apple", [item])
        self.assertEqual("The", result["prediction"])

    def test_non_context_offsets_do_not_point_into_context(self):
        item = window()
        item["offsets"][1] = (999, 9999)
        self.assertEqual("red", reader.decode_windows("The red apple", [item])["prediction"])

    def test_non_context_gap_cannot_be_crossed(self):
        item = window(starts=[0, 0, 10, 0, 0, 0], ends=[0, 0, 0, 0, 10, 0])
        item["context_mask"][3] = False
        result = reader.decode_windows("The red apple", [item])
        self.assertEqual("The", result["prediction"])

    def test_reversed_token_span_cannot_be_selected(self):
        item = window(starts=[0, 0, 0, 1, 10, 0], ends=[0, 0, 10, 1, 0, 0])
        result = reader.decode_windows("The red apple", [item])
        self.assertEqual("The", result["prediction"])

    def test_thirty_token_maximum(self):
        context = "x " * 31
        item = {"offsets": [(0, 0)] + [(2 * i, 2 * i + 1) for i in range(31)],
                "context_mask": [False] + [True] * 31,
                "start_logits": [0, 10] + [0] * 30,
                "end_logits": [0] * 31 + [10], "cls_index": 0}
        result = reader.decode_windows(context, [item])
        self.assertLessEqual(len(result["prediction"].split()), 30)
        self.assertNotEqual(context.strip(), result["prediction"])

    def test_top_twenty_is_a_real_candidate_limit(self):
        # Start index 21 has the best compatible end but is outside top-20 starts.
        context = "x " * 22
        item = {"offsets": [(0, 0)] + [(2 * i, 2 * i + 1) for i in range(22)],
                "context_mask": [False] + [True] * 22,
                "start_logits": [0] + [10] * 20 + [9, 8],
                "end_logits": [0] + [0] * 20 + [100, 0], "cls_index": 0}
        result = reader.decode_windows(context, [item])
        self.assertEqual(0, result["start"])
        self.assertEqual(41, result["end"])
        self.assertEqual(110, result["margin"])

    def test_nonfinite_or_nonnumeric_logits_fail_even_outside_context(self):
        for value in (math.nan, math.inf, -math.inf, True, "4", None, 10 ** 1000):
            for name in ("start_logits", "end_logits"):
                item = window()
                item[name][1] = value
                with self.subTest(value=repr(value), name=name), self.assertRaises(ValueError):
                    reader.decode_windows("The red apple", [item])

    def test_score_or_margin_overflow_fails_closed(self):
        for start, end, null in ((1e308, 1e308, 0), (1e308, 0, -1e308)):
            item = window(null=null, starts=[null / 2, 0, 0, start, 0, 0],
                          ends=[null / 2, 0, 0, end, 0, 0])
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                reader.decode_windows("The red apple", [item])
        item = window()
        item["start_logits"][0] = item["end_logits"][0] = 1e308
        with self.assertRaises(ValueError):
            reader.decode_windows("The red apple", [item])

    def test_invalid_context_offsets_raise(self):
        for offset in (None, (), (4,), (7, 4), (-1, 5), (4, 14), (4.0, 7), (True, 7)):
            item = window()
            item["offsets"][3] = offset
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                reader.decode_windows("The red apple", [item])
        item = window()
        item["offsets"][4] = (0, 1)
        with self.assertRaisesRegex(ValueError, "monotonically"):
            reader.decode_windows("The red apple", [item])

    def test_bad_cls_indices_rejected(self):
        for index in (-1, 6, 2, True, 0.0, None):
            item = window()
            item["cls_index"] = index
            with self.subTest(index=index), self.assertRaises(ValueError):
                reader.decode_windows("The red apple", [item])

    def test_no_valid_context_span_raises_instead_of_inventing_score(self):
        for mode in ("masked", "zero_width", "whitespace"):
            item = window()
            if mode == "masked":
                item["context_mask"] = [False] * 6
            elif mode == "zero_width":
                item["offsets"] = [(0, 0)] * 6
            else:
                item["offsets"] = [(3, 4)] * 6
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "no valid"):
                reader.decode_windows("The red apple", [item])

    def test_bad_window_shapes_masks_and_missing_keys(self):
        cases = [None, "bad", {}]
        for key in ("offsets", "context_mask", "start_logits", "end_logits"):
            item = window()
            item[key] = item[key][:-1]
            cases.append(item)
        item = window()
        item["context_mask"][2] = 1
        cases.append(item)
        for item in cases:
            with self.subTest(item=item), self.assertRaises(ValueError):
                reader.decode_windows("The red apple", [item])

    def test_empty_or_overbudget_inputs_fail(self):
        for context in (None, "", " \n\t", "x" * 16001):
            with self.subTest(context=repr(context)[:30]), self.assertRaises(ValueError):
                reader.decode_windows(context, [window()])
        for windows in (None, [], [window()] * 9, "bad"):
            with self.subTest(windows=type(windows)), self.assertRaises(ValueError):
                reader.decode_windows("The red apple", windows)
        self.assertEqual(8, reader.decode_windows("The red apple", [window()] * 8)["windows"])

    def test_decoder_does_not_mutate_inputs(self):
        windows = [window(), window(null=4)]
        original = copy.deepcopy(windows)
        reader.decode_windows("The red apple", windows)
        self.assertEqual(original, windows)


class InputBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.instance = reader.ExtractiveReader.__new__(reader.ExtractiveReader)
        self.instance._lock = threading.Lock()
        self.instance.tokenizer = mock.Mock()

    def test_invalid_text_rejected_before_any_tokenizer_or_model(self):
        for question, context in (("", "apple"), (None, "apple"), ("x" * 513, "apple"),
                                  ("Which?", ""), ("Which?", None), ("Which?", "x" * 16001)):
            with self.subTest(question=repr(question)[:20]), self.assertRaises(ValueError):
                self.instance.predict(question, context)
        self.instance.tokenizer.assert_not_called()

    def test_question_token_budget_rejected_before_pair_encoding(self):
        for tokens in ([], list(range(129))):
            self.instance.tokenizer.reset_mock()
            self.instance.tokenizer.return_value = {"input_ids": tokens}
            with self.subTest(count=len(tokens)), self.assertRaisesRegex(ValueError, "question.*tokens"):
                self.instance.predict("Which fruit?", "The red apple")
            self.assertEqual(1, self.instance.tokenizer.call_count)

    def test_more_than_eight_windows_fails_before_inference(self):
        self.instance.tokenizer.side_effect = [{"input_ids": [3, 4]}, {"input_ids": [[0]] * 9}]
        with self.assertRaisesRegex(ValueError, "more than 8"):
            self.instance.predict("Which fruit?", "The red apple")
        args = self.instance.tokenizer.call_args.kwargs
        self.assertEqual("only_second", args["truncation"])
        self.assertEqual(384, args["max_length"])
        self.assertEqual(128, args["stride"])


if __name__ == "__main__":
    unittest.main()
