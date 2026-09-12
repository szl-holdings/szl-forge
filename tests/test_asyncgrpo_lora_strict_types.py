"""Regression coverage for malformed-plan admission; no upstream runtime runs."""
from __future__ import annotations

import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from frontier.asyncgrpo_lora_contract import (
    ALLOWED_MAX_LORA_RANKS,
    DENIED_AUTHORITY,
    EvaluationContractError,
    evaluate_plan,
    required_max_loras,
    serving_cache_path,
)

BOOL_FIELDS = (
    "lora_server_enabled", "shared_storage_verified", "checkpoints_enabled", "adapter_servable",
)
REASON_FOR = dict(zip(BOOL_FIELDS, (
    "vllm_lora_not_enabled", "shared_storage_unverified", "durable_checkpoint_missing",
    "adapter_not_vllm_servable",
)))
BASE = dict(
    lora_rank=1, max_lora_rank=8, max_loras=2, max_staleness=0,
    lora_server_enabled=True, shared_storage_verified=True,
    checkpoints_enabled=True, adapter_servable=True,
)


class IntSubclass(int):
    pass


class TruthinessTrap:
    def __bool__(self):
        raise AssertionError("validator must not invoke untrusted __bool__")

    def __repr__(self):
        raise AssertionError("validator must not expose untrusted values")


class AsyncGrpoStrictTypesTests(unittest.TestCase):
    def test_each_boolean_field_rejects_non_booleans(self):
        for field in BOOL_FIELDS:
            for value in ("false", "true", "0", "", 0, 1, -1, 0.0, 1.0, None, [], [1], {}, {"ok": True}):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(EvaluationContractError):
                        evaluate_plan(**(BASE | {field: value}))

    def test_truthiness_or_representation_is_never_evaluated(self):
        for field in BOOL_FIELDS:
            with self.subTest(field=field), self.assertRaises(EvaluationContractError):
                evaluate_plan(**(BASE | {field: TruthinessTrap()}))

    def test_all_boolean_combinations_preserve_negative_reason_semantics(self):
        for vector in itertools.product((False, True), repeat=4):
            flags = dict(zip(BOOL_FIELDS, vector))
            with self.subTest(flags=flags):
                result = evaluate_plan(**(BASE | flags))
                expected = [REASON_FOR[f] for f in BOOL_FIELDS if flags[f] is False]
                self.assertEqual(result["reasons"], expected)
                self.assertEqual(result["disposition"], "HOLD" if expected else "EVALUATION")
                for key in DENIED_AUTHORITY:
                    self.assertIs(result[key], False)

    def test_rank_capacity_rejects_boolean_float_string_and_containers(self):
        for value in (True, False, 1.0, 8.0, 32.0, "8", None, [], {}, {8}, IntSubclass(8)):
            with self.subTest(type=type(value).__name__), self.assertRaises(EvaluationContractError):
                evaluate_plan(**(BASE | {"max_lora_rank": value}))

    def test_all_integer_fields_require_builtin_integers(self):
        for field in ("lora_rank", "max_lora_rank", "max_loras", "max_staleness"):
            for value in (True, False, 8.0, "8", None, [], {}, IntSubclass(8)):
                with self.subTest(field=field, type=type(value).__name__):
                    with self.assertRaises(EvaluationContractError):
                        evaluate_plan(**(BASE | {field: value}))

    def test_all_admitted_rank_capacities_remain_accepted(self):
        for rank in ALLOWED_MAX_LORA_RANKS:
            with self.subTest(rank=rank):
                result = evaluate_plan(**(BASE | {"lora_rank": rank, "max_lora_rank": rank}))
                self.assertEqual(result["disposition"], "EVALUATION")
                self.assertEqual(result["reasons"], [])
                for key in DENIED_AUTHORITY:
                    self.assertIs(result[key], False)

    def test_rank_and_capacity_boundaries(self):
        for change in ({"lora_rank": 0}, {"max_loras": 0}, {"max_staleness": -1},
                       {"max_lora_rank": 7}, {"lora_rank": 9, "max_lora_rank": 8}):
            with self.subTest(change=change), self.assertRaises(EvaluationContractError):
                evaluate_plan(**(BASE | change))
        self.assertEqual(required_max_loras(0), 2)
        self.assertEqual(required_max_loras(32), 34)
        result = evaluate_plan(**(BASE | {"max_loras": 1}))
        self.assertEqual(result["reasons"], ["insufficient_version_capacity"])
        self.assertEqual(result["disposition"], "HOLD")

    def test_staleness_rejects_boolean_subclasses_and_float(self):
        for value in (True, False, 0.0, "0", None, IntSubclass(0)):
            with self.subTest(type=type(value).__name__), self.assertRaises(EvaluationContractError):
                required_max_loras(value)

    def test_malformed_late_field_is_not_hidden_by_earlier_false(self):
        with self.assertRaises(EvaluationContractError):
            evaluate_plan(**(BASE | {"lora_server_enabled": False, "adapter_servable": "false"}))

    def test_error_does_not_echo_supplied_sensitive_text(self):
        sentinel = "sensitive-value-for-test-only"
        with self.assertRaises(EvaluationContractError) as caught:
            evaluate_plan(**(BASE | {"shared_storage_verified": sentinel}))
        self.assertNotIn(sentinel, str(caught.exception))

    def test_cache_root_and_normalized_root_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for target in (root, root / ".", root / "child" / ".."):
                with self.subTest(target=str(target)), self.assertRaises(EvaluationContractError):
                    serving_cache_path(root, target)

    def test_cache_sibling_with_same_prefix_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            with self.assertRaises(EvaluationContractError):
                serving_cache_path(root, Path(temp) / "run-other" / "cache")

    def test_cache_child_does_not_create_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            target = root / "cache" / "v1"
            self.assertEqual(serving_cache_path(root, target), target.resolve())
            self.assertFalse(root.exists())

    def test_cache_resolution_errors_become_contract_errors(self):
        for error in (OSError("not-readable"), RuntimeError("symlink-loop"), ValueError("bad-path")):
            with patch.object(Path, "resolve", side_effect=error), self.assertRaises(EvaluationContractError):
                serving_cache_path(Path("run"), Path("run/cache"))

    def test_cache_strings_rejected_without_attribute_error(self):
        for root, target in (("run", Path("run/cache")), (Path("run"), "run/cache")):
            with self.assertRaises(EvaluationContractError):
                serving_cache_path(root, target)

    def test_symlink_escape_rejected_on_supported_filesystems(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            outside = Path(temp) / "outside"
            root.mkdir()
            outside.mkdir()
            try:
                (root / "escape").symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlink unavailable on this host: {type(exc).__name__}")
            with self.assertRaises(EvaluationContractError):
                serving_cache_path(root, root / "escape" / "v1")


if __name__ == "__main__":
    unittest.main()
