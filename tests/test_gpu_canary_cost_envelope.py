# SPDX-License-Identifier: Apache-2.0
"""Credential-free regression tests for the source-only cost envelope."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from unittest import TestCase

from training.gpu_canary_cost_envelope import MAX_PLAN_BYTES, SCHEMA, assess, parse_plan_bytes


NOW = datetime(2026, 11, 19, 8, 0, tzinfo=timezone.utc)


def candidate():
    return {
        "schema": SCHEMA,
        "provider": "salad",
        "organization": "alloyscape",
        "project": "default",
        "container_group": "szl-one-gpu-reviewed-canary",
        "student_revision": "a" * 40,
        "dataset_revision": "b" * 40,
        "train_sha256": "c" * 64,
        "holdout_sha256": "d" * 64,
        "worker_image": "ghcr.io/szl/worker@sha256:" + "e" * 64,
        "gpu_count": 1,
        "max_running_seconds": 1800,
        "max_attempts": 1,
        "observed_gpu_rate_usd_hour": "0.44",
        "operator_cost_cap_usd": "0.25",
        "price_observed_utc": "2026-11-19T07:45:00Z",
    }


class GpuCostEnvelopeTests(TestCase):
    def test_one_gpu_cost_is_modeled_not_a_launch(self):
        result = assess(candidate(), now=NOW)
        self.assertEqual(result["verdict"], "COST_ENVELOPE_MODELED_NOT_ADMITTED")
        self.assertEqual(result["modeled_gpu_cost_ceiling_usd"], "0.22")
        self.assertFalse(result["launch_authorized"])
        self.assertFalse(result["rights_verified"])
        self.assertFalse(result["owner_signature_verified"])
        self.assertEqual(result["provider_mutations"], 0)

    def test_two_gpu_scale_and_retries_refused(self):
        plan = candidate()
        plan["gpu_count"] = 2
        plan["max_attempts"] = 2
        result = assess(plan, now=NOW)
        self.assertIn("SINGLE_GPU_ONLY", result["codes"])
        self.assertIn("RETRY_CAP_INVALID", result["codes"])

    def test_boolean_number_not_accepted(self):
        plan = candidate()
        plan["gpu_count"] = True
        plan["max_running_seconds"] = False
        result = assess(plan, now=NOW)
        self.assertIn("SINGLE_GPU_ONLY", result["codes"])
        self.assertIn("RUNTIME_CAP_INVALID", result["codes"])

    def test_not_admitted_when_price_stale_or_future(self):
        plan = candidate()
        plan["price_observed_utc"] = "2026-11-19T07:00:00Z"
        self.assertIn("PROVIDER_PRICE_NOT_FRESH", assess(plan, now=NOW)["codes"])
        plan["price_observed_utc"] = "2026-11-19T08:01:00Z"
        self.assertIn("PROVIDER_PRICE_NOT_FRESH", assess(plan, now=NOW)["codes"])

    def test_budget_overrun_refused(self):
        plan = candidate()
        plan["operator_cost_cap_usd"] = "0.21"
        result = assess(plan, now=NOW)
        self.assertIn("MODELED_COST_EXCEEDS_DECLARED_CAP", result["codes"])
        self.assertFalse(result["launch_authorized"])

    def test_quote_requires_decimal_string_not_json_float(self):
        plan = candidate()
        plan["observed_gpu_rate_usd_hour"] = 0.44
        self.assertIn("PROVIDER_QUOTE_OR_CAP_INVALID", assess(plan, now=NOW)["codes"])
        plan["observed_gpu_rate_usd_hour"] = "1e1000"
        self.assertIn("PROVIDER_QUOTE_OR_CAP_INVALID", assess(plan, now=NOW)["codes"])

    def test_moving_image_and_missing_source_pins_refused(self):
        plan = candidate()
        plan["worker_image"] = "ghcr.io/szl/worker:latest"
        plan["student_revision"] = "main"
        result = assess(plan, now=NOW)
        self.assertIn("WORKER_IMAGE_DIGEST_UNPINNED", result["codes"])
        self.assertIn("STUDENT_REVISION_UNPINNED", result["codes"])

    def test_same_train_holdout_file_hash_refused(self):
        plan = candidate()
        plan["holdout_sha256"] = plan["train_sha256"]
        self.assertIn("TRAIN_HOLDOUT_IDENTICAL", assess(plan, now=NOW)["codes"])

    def test_group_scope_and_runtime_cap_refused(self):
        plan = candidate()
        plan["organization"] = "other"
        plan["container_group"] = "../wrong"
        plan["max_running_seconds"] = 86400
        result = assess(plan, now=NOW)
        self.assertIn("PROVIDER_SCOPE_NOT_CANONICAL", result["codes"])
        self.assertIn("CONTAINER_GROUP_INVALID", result["codes"])
        self.assertIn("RUNTIME_CAP_INVALID", result["codes"])

    def test_bounded_json_roundtrip_keeps_exact_plan(self):
        plan = candidate()
        self.assertEqual(parse_plan_bytes(json.dumps(plan).encode("utf-8")), plan)

    def test_duplicate_top_level_gpu_count_is_rejected(self):
        raw = json.dumps(candidate(), separators=(",", ":")).replace(
            '"gpu_count":1', '"gpu_count":2,"gpu_count":1',
        )
        with self.assertRaisesRegex(ValueError, "DUPLICATE_PLAN_JSON_MEMBER"):
            parse_plan_bytes(raw.encode("utf-8"))

    def test_duplicate_nested_unknown_fields_rejected(self):
        raw = json.dumps(candidate(), separators=(",", ":"))
        raw = raw[:-1] + ',"metadata":{"owner":"first","owner":"second"}}'
        with self.assertRaisesRegex(ValueError, "DUPLICATE_PLAN_JSON_MEMBER"):
            parse_plan_bytes(raw.encode("utf-8"))

    def test_nonfinite_and_fractional_json_numbers_refused(self):
        raw = json.dumps(candidate(), separators=(",", ":"))
        for number in ("NaN", "Infinity", "-Infinity", "1.0", "1e100000"):
            with self.subTest(number=number):
                malformed = raw.replace('"gpu_count":1', '"gpu_count":' + number)
                with self.assertRaisesRegex(ValueError, "UNSUPPORTED_PLAN_JSON_NUMBER"):
                    parse_plan_bytes(malformed.encode("utf-8"))

    def test_empty_oversized_or_nonobject_plan_refused(self):
        for raw in (b"", b" " * (MAX_PLAN_BYTES + 1), b"[]"):
            with self.subTest(size=len(raw)):
                with self.assertRaises(ValueError):
                    parse_plan_bytes(raw)

    def test_invalid_config_and_clock_refused(self):
        result = assess(None, now=NOW)
        self.assertEqual(result["verdict"], "HOLD")
        self.assertFalse(result["launch_authorized"])
        with self.assertRaises(ValueError):
            assess(candidate(), now=datetime(2026, 11, 19, 8))
        self.assertEqual(
            assess(candidate(), now=NOW + timedelta(minutes=45))["verdict"],
            "HOLD",
        )
