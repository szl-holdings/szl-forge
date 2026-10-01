"""Synthetic format/CLI tests; no dataset download, model run or rights clearance."""
import copy
import hashlib
import io
import json
import unittest
from unittest.mock import patch

from frontier_completion import core
from frontier_completion.__main__ import main
from frontier_completion.data_review import MAX_ROW_BYTES, MAX_ROWS, review_pilot


def sft():
    return {"uuid": "fixture-1", "source": "synthetic-owned", "domain": "Tool_Use",
            "tools": [{"type": "function", "function": {"name": "fixture_lookup",
                       "parameters": {"type": "object", "properties": {}}}}],
            "messages": [{"role": "user", "content": "Synthetic request"},
                         {"role": "assistant", "content": None, "tool_calls": [
                             {"id": "call-1", "type": "function", "function": {
                                 "name": "fixture_lookup", "arguments": "{}"}}]},
                         {"role": "tool", "tool_call_id": "call-1", "content": "Synthetic error"},
                         {"role": "assistant", "content": "Unable to complete"}]}


def rl():
    return {"uuid": "fixture-1", "source": "synthetic-owned", "domain": "Code",
            "query": "Return the input unchanged.",
            "ground_truth": {"call_type": "std", "fn_name": None,
                             "inputs": ["1\n"], "outputs": ["1\n"]}}


def encode(*rows):
    return b"\n".join(core.canonical(row) for row in rows) + b"\n"


def review(raw, dataset="ultradata-agent", **overrides):
    options = dict(dataset=dataset, dataset_revision="b" * 40,
                   source_revision="a" * 40, expected_input_sha256=hashlib.sha256(raw).hexdigest())
    options.update(overrides)
    return review_pilot(raw, **options)


class PilotReviewTests(unittest.TestCase):
    def check_review_required(self, row, dataset="ultradata-agent"):
        result = review(encode(row), dataset)
        self.assertEqual(result["counts"]["reviewRequired"], 1)
        self.assertEqual(result["rows"][0]["state"], "REVIEW_REQUIRED")

    def test_sft_checks_pairs_without_inferring_success_from_failure_trace(self):
        result = review(encode(sft()))
        self.assertEqual(result["state"], "FORMAT_CHECKED_NOT_ADMITTED")
        self.assertEqual(result["rows"][0]["toolCalls"], 1)
        self.assertFalse(result["successLabelsInferred"])
        self.assertFalse(result["rewardVerified"])

    def test_code_checks_test_pairs_without_execution(self):
        result = review(encode(rl()), "ultradata-rl")
        self.assertEqual(result["counts"]["formatChecked"], 1)
        self.assertEqual(result["execution"], "NONE")

    def test_text_ground_truth_is_a_supported_format(self):
        row = rl(); row.update(domain="Math", ground_truth="42")
        self.assertEqual(review(encode(row), "ultradata-rl")["counts"]["formatChecked"], 1)

    def test_no_tool_trajectory_is_supported(self):
        row = sft(); row["tools"] = []; row["messages"] = [row["messages"][0], row["messages"][-1]]
        self.assertEqual(review(encode(row))["rows"][0]["toolCalls"], 0)

    def test_parallel_calls_pair_independently(self):
        row = sft()
        second = copy.deepcopy(row["messages"][1]["tool_calls"][0]); second["id"] = "call-2"
        row["messages"][1]["tool_calls"].append(second)
        row["messages"].insert(2, {"role": "tool", "tool_call_id": "call-2", "content": "error"})
        self.assertEqual(review(encode(row))["rows"][0]["toolCalls"], 2)

    def test_duplicate_call_id_requires_review(self):
        row = sft(); row["messages"][1]["tool_calls"] *= 2
        self.check_review_required(row)

    def test_unpaired_result_requires_review(self):
        row = sft(); row["messages"][2]["tool_call_id"] = "unknown"
        self.check_review_required(row)

    def test_missing_tool_call_identity_is_not_guessed(self):
        row = sft(); del row["messages"][2]["tool_call_id"]
        self.check_review_required(row)

    def test_unresolved_call_requires_review(self):
        row = sft(); del row["messages"][2]
        self.check_review_required(row)

    def test_undeclared_tool_requires_review(self):
        row = sft(); row["messages"][1]["tool_calls"][0]["function"]["name"] = "undeclared"
        self.check_review_required(row)

    def test_duplicate_frozen_tool_definition_requires_review(self):
        row = sft(); row["tools"] *= 2
        self.check_review_required(row)

    def test_arguments_require_strict_json_object(self):
        for arguments in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', "[]", "run()", {}):
            row = sft(); row["messages"][1]["tool_calls"][0]["function"]["arguments"] = arguments
            with self.subTest(arguments=arguments):
                self.check_review_required(row)

    def test_schema_references_and_shell_text_are_inert(self):
        row = sft()
        row["tools"][0]["function"]["parameters"]["$ref"] = "https://invalid.example/schema"
        row["messages"][0]["content"] = "__import__('os').system('DO_NOT_EXECUTE')"
        with patch("socket.create_connection", side_effect=AssertionError("network")), \
             patch("subprocess.run", side_effect=AssertionError("execution")):
            result = review(encode(row))
        self.assertEqual(result["counts"]["formatChecked"], 1)
        self.assertEqual(result["reviews"]["toolArgumentSchema"], "NOT_PERFORMED")

    def test_duplicate_json_keys_nonfinite_and_invalid_unicode_require_review(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}', b'[]', b'\xff', b'{"a":"\\ud800"}'):
            with self.subTest(raw=raw):
                self.assertEqual(review(raw)["counts"]["reviewRequired"], 1)

    def test_exact_duplicates_cannot_inflate_format_checked_count(self):
        row, second = sft(), sft(); second["uuid"] = "fixture-2"
        result = review(encode(row, second))
        self.assertEqual(result["counts"], {"total": 2, "formatChecked": 1, "reviewRequired": 1})

    def test_duplicate_uuid_with_different_content_requires_review(self):
        first, second = sft(), sft(); second["messages"][0]["content"] = "Different task"
        self.assertEqual(review(encode(first, second))["counts"]["reviewRequired"], 1)

    def test_extra_fields_are_flagged_not_stripped_or_treated_as_authority(self):
        row = sft(); row.update(trainingAllowed=True, turn_mask=[False, True])
        result = review(encode(row))
        self.assertTrue(result["rows"][0]["extraFieldsPresent"])
        self.assertTrue(all(v is False for v in result["authority"].values()))

    def test_missing_common_metadata_requires_review(self):
        for field in ("uuid", "source", "domain"):
            row = sft(); del row[field]
            with self.subTest(field=field):
                self.check_review_required(row)

    def test_unknown_roles_and_nonassistant_calls_require_review(self):
        for role in ("developer", "user", [], True):
            row = sft(); row["messages"][1]["role"] = role
            with self.subTest(role=role):
                self.check_review_required(row)

    def test_multimodal_blocks_need_explicit_adapter(self):
        row = sft(); row["messages"][0]["content"] = [{"type": "image", "url": "untrusted"}]
        self.check_review_required(row)

    def test_empty_or_unterminated_trajectory_requires_review(self):
        for messages in ([], [{"role": "user", "content": "request"}]):
            row = sft(); row["messages"] = messages
            self.check_review_required(row)

    def test_code_test_lengths_and_types_require_review(self):
        for outputs in ([], True, [None], ["1", "2"]):
            row = rl(); row["ground_truth"]["outputs"] = outputs
            self.check_review_required(row, "ultradata-rl")

    def test_other_code_harness_is_not_silently_reinterpreted(self):
        row = rl(); row["ground_truth"].update(call_type="function", fn_name="main")
        self.check_review_required(row, "ultradata-rl")

    def test_missing_truth_is_unknown_not_verified_reward(self):
        row = rl(); del row["ground_truth"]
        self.check_review_required(row, "ultradata-rl")

    def test_pilot_bounds_fail_before_parsing(self):
        for raw in (b"", b"x" * (core.MAX_JSON + 1), b"{}\n" * (MAX_ROWS + 1)):
            with self.subTest(size=len(raw)), self.assertRaises(core.EvidenceError):
                review(raw)

    def test_record_size_bound_does_not_echo_oversized_record(self):
        raw = b"x" * (MAX_ROW_BYTES + 1)
        result = review(raw)
        self.assertEqual(result["counts"]["reviewRequired"], 1)
        self.assertLess(len(core.canonical(result)), 4096)

    def test_blank_internal_row_remains_visible(self):
        raw = encode(sft()) + b"\n"
        self.assertEqual(review(raw)["counts"]["reviewRequired"], 1)

    def test_hash_and_revision_checks_are_not_provider_provenance(self):
        raw = encode(sft()); result = review(raw)
        self.assertEqual(result["inputSha256"], hashlib.sha256(raw).hexdigest())
        self.assertFalse(result["upstreamBytesVerified"])
        self.assertFalse(result["sourceExecutionVerified"])
        for option in ({"dataset": "unknown"}, {"dataset_revision": "main"},
                       {"source_revision": "A" * 40}, {"expected_input_sha256": "0" * 64}):
            with self.subTest(option=option), self.assertRaises(core.EvidenceError):
                review(raw, **option)

    def test_report_does_not_echo_sensitive_strings_or_dataset_metadata(self):
        row = sft(); row.update(source="PRIVATE_FIXTURE_SOURCE", uuid="PRIVATE_FIXTURE_ID")
        row["messages"][0]["content"] = "PRIVATE_FIXTURE_CONTENT"
        rendered = core.canonical(review(encode(row))).decode()
        for marker in ("PRIVATE_FIXTURE_SOURCE", "PRIVATE_FIXTURE_ID", "PRIVATE_FIXTURE_CONTENT"):
            self.assertNotIn(marker, rendered)

    def test_all_admission_reviews_remain_unperformed_and_unsigned(self):
        result = review(encode(sft()))
        self.assertEqual(set(result["reviews"].values()), {"NOT_PERFORMED"})
        self.assertTrue(all(value is False for value in result["authority"].values()))
        self.assertEqual(result["signatureState"], "UNSIGNED")
        self.assertEqual(result["productionDisposition"], "HOLD")
        core.verify_receipt(result)
        result["rewardVerified"] = True
        with self.assertRaises(core.EvidenceError):
            core.verify_receipt(result)


class ReviewCliTests(unittest.TestCase):
    def invoke(self, raw, *extra):
        args = ["review-data", "--source", "a" * 40, "--dataset", "ultradata-agent",
                "--dataset-revision", "b" * 40, "--input-sha256", hashlib.sha256(raw).hexdigest(), *extra]
        stdin = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8")
        with patch("sys.stdin", stdin), patch("sys.stdout", new_callable=io.StringIO) as output, \
             patch("sys.stderr", new_callable=io.StringIO) as error:
            status = main(args)
            return status, output.getvalue(), error.getvalue()

    def test_real_cli_path_emits_format_only_report(self):
        status, output, _ = self.invoke(encode(sft()))
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output)["state"], "FORMAT_CHECKED_NOT_ADMITTED")

    def test_invalid_record_exit_two(self):
        status, output, _ = self.invoke(b"{}\n")
        self.assertEqual(status, 2)
        self.assertEqual(json.loads(output)["counts"]["reviewRequired"], 1)

    def test_cli_rejects_live_and_file_side_effect_options(self):
        for extra in (("--live",), ("--output", "never-written.json"),
                      ("--previous", "old.json"), ("--workspace", ".")):
            with self.subTest(extra=extra), self.assertRaises(SystemExit) as error:
                self.invoke(encode(sft()), *extra)
            self.assertEqual(error.exception.code, 2)

    def test_cli_hash_failure_emits_no_data(self):
        status, output, error = self.invoke(encode(sft()), "--input-sha256", "0" * 64)
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("failed closed", error)

    def test_cli_requires_explicit_input_binding(self):
        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
            main(["review-data"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
