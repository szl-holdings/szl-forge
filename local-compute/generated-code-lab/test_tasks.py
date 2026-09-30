"""Validate public development smoke fixtures, without a model or network.

These tests verify fixture consistency, not model ability. These public cases are
not a sealed scientific benchmark. Reference functions are independent of any
model-generated solution and are never production evidence authenticators.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import unittest


TASK_DIR = Path(__file__).resolve().parent / "tasks"


def reference_digest(value):
    report = copy.deepcopy(value)
    report.pop("report_sha256", None)
    canonical = json.dumps(
        report, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def reference_admission(value):
    def valid_revision(candidate):
        return isinstance(candidate, str) and re.fullmatch(r"[0-9a-f]{40}", candidate) is not None

    checks = [
        (valid_revision(value.get("source_sha")), "invalid_source"),
        (valid_revision(value.get("runtime_source_sha")), "invalid_runtime_source"),
        (value.get("source_sha") == value.get("runtime_source_sha"), "source_mismatch"),
        (value.get("artifact_verified") is True, "artifact_unverified"),
        (value.get("publication_status") == "success", "publication_not_success"),
        (value.get("runtime_healthy") is True, "runtime_not_healthy"),
    ]
    for passes, reason in checks:
        if not passes:
            return {"admitted": False, "reason": reason}
    return {"admitted": True, "reason": "admitted"}


def reference_routing(value):
    def nonnegative_number(candidate):
        return (
            type(candidate) in (int, float)
            and candidate >= 0
            and (type(candidate) is int or math.isfinite(candidate))
        )

    limit = value.get("max_latency_ms")
    providers = value.get("providers")
    if not nonnegative_number(limit) or not isinstance(providers, list):
        return None
    candidates = []
    for provider in providers:
        if not isinstance(provider, dict) or provider.get("eligible") is not True:
            continue
        name = provider.get("name")
        price = provider.get("price")
        latency = provider.get("latency_ms")
        if not isinstance(name, str) or not name:
            continue
        if not nonnegative_number(price) or not nonnegative_number(latency):
            continue
        if latency <= limit:
            candidates.append((price, name))
    return min(candidates)[1] if candidates else None


REFERENCES = {
    "canonical_report_digest": reference_digest,
    "evidence_admission": reference_admission,
    "stable_provider_routing": reference_routing,
}


def reject_nonfinite(token):
    raise ValueError("Non-JSON numeric constant: " + token)


class TaskFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = sorted(TASK_DIR.glob("*.json"))
        cls.tasks = [
            json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)
            for path in cls.paths
        ]

    def test_schema_and_complete_task_set(self):
        self.assertEqual(len(self.tasks), 3)
        self.assertEqual({task.get("id") for task in self.tasks}, set(REFERENCES))
        for path, task in zip(self.paths, self.tasks):
            with self.subTest(task=path.name):
                self.assertIsInstance(task, dict)
                self.assertEqual(set(task) - {"starter"}, {"id", "title", "instruction", "cases"})
                self.assertRegex(task["id"], r"\A[a-z][a-z0-9_]*\Z")
                self.assertEqual(task["id"], path.stem)
                for key in ("title", "instruction"):
                    self.assertIsInstance(task[key], str)
                    self.assertTrue(task[key].strip())
                if "starter" in task:
                    self.assertIsInstance(task["starter"], str)
                    compile(task["starter"], path.name, "exec")
                self.assertIsInstance(task["cases"], list)
                self.assertGreaterEqual(len(task["cases"]), 6)
                for case in task["cases"]:
                    self.assertEqual(set(case), {"input", "expected"})
                    self.assertIsInstance(case["input"], dict)
                    json.dumps(case, allow_nan=False)

    def test_every_expected_value_against_independent_reference(self):
        for task in self.tasks:
            reference = REFERENCES[task["id"]]
            for index, case in enumerate(task["cases"]):
                with self.subTest(task=task["id"], case=index):
                    original = copy.deepcopy(case["input"])
                    observed = reference(case["input"])
                    # JSON comparison distinguishes booleans from integers.
                    self.assertEqual(
                        json.dumps(observed, sort_keys=True, allow_nan=False),
                        json.dumps(case["expected"], sort_keys=True, allow_nan=False),
                    )
                    self.assertEqual(case["input"], original)

    def test_digest_self_field_and_nested_field_contract(self):
        self.assertEqual(reference_digest({"report_sha256": "old"}), reference_digest({}))
        self.assertNotEqual(
            reference_digest({"nested": {"report_sha256": "old"}}),
            reference_digest({"nested": {}}),
        )

    def test_routing_order_independence(self):
        task = next(t for t in self.tasks if t["id"] == "stable_provider_routing")
        for case in task["cases"]:
            value = copy.deepcopy(case["input"])
            if isinstance(value.get("providers"), list):
                value["providers"].reverse()
                self.assertEqual(reference_routing(value), case["expected"])


if __name__ == "__main__":
    unittest.main()
