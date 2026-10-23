"""Offline, synthetic fixtures only; no trusted clocks, leases or signatures."""

import copy
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "frontier/msq650_admission/campaign_budget.py"
SPEC = importlib.util.spec_from_file_location("msq650_campaign_budget_test_module", MODULE_PATH)
budget = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = budget
SPEC.loader.exec_module(budget)
PIN = "e10294badec5f4291e5321c96a925d8629896306da24e77e5d406b3614e6389c"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


class CampaignBudgetTests(unittest.TestCase):
    def setUp(self):
        self.genesis = {"campaign_id": "msq650-test", "candidate_sha256": PIN,
            "start_epoch": 1_800_000_000, "deadline_epoch": 1_800_180_000,
            "max_cost_cents": 200_000, "max_wall_seconds": 180_000}
        self.ledger = {"schema": "szl.msq650.campaign-ledger.v1", "genesis": self.genesis, "records": []}
        self.request = {"lease_id": "lease-1", "candidate_sha256": PIN,
            "start_epoch": self.genesis["start_epoch"], "wall_seconds": 3600,
            "rate_cents_per_hour": 57, "overhead_cents": 3}
        self.context = {"expected_campaign_id": "msq650-test", "expected_candidate_sha256": PIN,
            "expected_start_epoch": self.genesis["start_epoch"],
            "expected_genesis_sha256": budget.canonical_sha256(self.genesis),
            "expected_tip_sha256": budget.canonical_sha256(self.genesis), "expected_sequence": 0,
            "current_epoch": self.genesis["start_epoch"], "observed_at_epoch": self.genesis["start_epoch"]}

    def observe(self):
        return budget.observe_reservation(encoded(self.ledger), encoded(self.request), encoded(self.context))

    def append(self, status="reserved"):
        item = dict(self.observe().proposed_record)
        item["status"] = status
        item["sha256"] = budget.canonical_sha256({key: value for key, value in item.items() if key != "sha256"})
        self.ledger["records"].append(item)
        self.context["expected_sequence"] += 1
        self.context["expected_tip_sha256"] = item["sha256"]
        self.request["lease_id"] = "lease-" + str(self.context["expected_sequence"] + 1)
        self.request["start_epoch"] += self.request["wall_seconds"]

    def reject(self):
        with self.assertRaises(budget.BudgetError):
            self.observe()

    def repin(self):
        self.context["expected_genesis_sha256"] = budget.canonical_sha256(self.genesis)
        self.context["expected_tip_sha256"] = budget.canonical_sha256(self.genesis)

    def test_modeled_only_immutable_non_truthy(self):
        result = self.observe()
        self.assertEqual(result.evidence_class, "MODELED")
        self.assertFalse(result.training_admitted)
        self.assertFalse(result.execution_authority)
        self.assertFalse(result.atomic_reservation)
        self.assertFalse(result.externally_authenticated)
        self.assertEqual(result.proposed_record["reserved_cost_cents"], 60)
        with self.assertRaises(TypeError):
            bool(result)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.training_admitted = True
        with self.assertRaises(TypeError):
            result.proposed_record["reserved_cost_cents"] = 0

    def test_exact_maximum_equalities(self):
        self.request.update(wall_seconds=180_000, rate_cents_per_hour=4000, overhead_cents=0)
        record = self.observe().proposed_record
        self.assertEqual(record["cumulative_cost_cents"], 200_000)
        self.assertEqual(record["cumulative_wall_seconds"], 180_000)
        self.assertEqual(record["start_epoch"] + record["wall_seconds"], self.genesis["deadline_epoch"])

    def test_exact_integer_round_up_and_separate_overhead(self):
        self.request.update(wall_seconds=1, rate_cents_per_hour=1, overhead_cents=7)
        self.assertEqual(self.observe().proposed_record["reserved_cost_cents"], 8)
        self.request.update(wall_seconds=3601, rate_cents_per_hour=57, overhead_cents=7)
        self.assertEqual(self.observe().proposed_record["reserved_cost_cents"], 65)

    def test_lower_limits(self):
        self.genesis.update(max_cost_cents=60, max_wall_seconds=3600,
                            deadline_epoch=self.genesis["start_epoch"] + 3600)
        self.repin()
        self.assertEqual(self.observe().proposed_record["cumulative_cost_cents"], 60)
        self.request["overhead_cents"] += 1
        self.reject()

    def test_ceilings_cannot_increase(self):
        for key, value in (("max_cost_cents", 200_001), ("max_wall_seconds", 180_001)):
            with self.subTest(key=key):
                self.setUp()
                self.genesis[key] = value
                self.repin()
                self.reject()

    def test_cap_exceeded_by_one_cent(self):
        self.request.update(wall_seconds=180_000, rate_cents_per_hour=4000, overhead_cents=1)
        self.reject()

    def test_elapsed_campaign_window_counts_idle_gaps(self):
        self.request["start_epoch"] = self.genesis["deadline_epoch"] - 1
        self.request["wall_seconds"] = 2
        self.reject()

    def test_deadline_cannot_be_extended(self):
        self.genesis["deadline_epoch"] += 1
        self.repin()
        self.reject()

    def test_before_campaign_and_expired_campaign(self):
        for now in (self.genesis["start_epoch"] - 1, self.genesis["deadline_epoch"]):
            with self.subTest(now=now):
                self.context["current_epoch"] = now
                self.context["observed_at_epoch"] = now
                self.reject()

    def test_stale_or_future_context(self):
        for delta in (-61, 1):
            with self.subTest(delta=delta):
                self.context["observed_at_epoch"] = self.context["current_epoch"] + delta
                self.reject()
        self.context["observed_at_epoch"] = self.context["current_epoch"] - 60
        self.assertTrue(self.observe().feasible)

    def test_past_request(self):
        self.request["start_epoch"] -= 1
        self.reject()

    def test_sequential_no_overlap(self):
        self.append()
        self.assertEqual(self.observe().proposed_record["cumulative_cost_cents"], 120)
        self.request["start_epoch"] -= 1
        self.reject()

    def test_every_status_consumes_reserved_cost_and_wall(self):
        for status in ("reserved", "failed", "unknown", "completed"):
            with self.subTest(status=status):
                self.setUp()
                self.append(status)
                record = self.observe().proposed_record
                self.assertEqual(record["cumulative_cost_cents"], 120)
                self.assertEqual(record["cumulative_wall_seconds"], 7200)

    def test_duplicate_lease_rejected(self):
        self.append()
        self.request["lease_id"] = "lease-1"
        self.reject()

    def test_candidate_drift_request_and_context(self):
        self.request["candidate_sha256"] = "b" * 64
        self.reject()
        self.setUp()
        self.context["expected_candidate_sha256"] = "b" * 64
        self.reject()

    def test_candidate_drift_in_rehashed_history(self):
        self.append()
        record = self.ledger["records"][0]
        record["candidate_sha256"] = "b" * 64
        record["sha256"] = budget.canonical_sha256({k: v for k, v in record.items() if k != "sha256"})
        self.context["expected_tip_sha256"] = record["sha256"]
        self.reject()

    def test_genesis_commitment_and_external_identity_required(self):
        for key, value in (("expected_genesis_sha256", "0" * 64),
                           ("expected_campaign_id", "other"),
                           ("expected_start_epoch", self.genesis["start_epoch"] + 1)):
            with self.subTest(key=key):
                self.setUp()
                self.context[key] = value
                self.reject()

    def test_tip_and_history_rollback_rejected(self):
        self.append()
        self.ledger["records"] = []
        self.reject()
        self.context["expected_sequence"] = 0
        self.reject()

    def test_tampered_record_and_rehashed_fake_accounting_rejected(self):
        self.append()
        record = self.ledger["records"][0]
        record["reserved_cost_cents"] = 1
        self.reject()
        record["sha256"] = budget.canonical_sha256({k: v for k, v in record.items() if k != "sha256"})
        self.context["expected_tip_sha256"] = record["sha256"]
        self.reject()

    def test_broken_predecessor_rejected_even_if_rehashed(self):
        self.append()
        record = self.ledger["records"][0]
        record["previous_sha256"] = "f" * 64
        record["sha256"] = budget.canonical_sha256({k: v for k, v in record.items() if k != "sha256"})
        self.context["expected_tip_sha256"] = record["sha256"]
        self.reject()

    def test_unknown_fields_and_wrong_types(self):
        for value in (True, 1.5, "3600", None, -1, 0):
            with self.subTest(value=value):
                self.setUp()
                self.request["wall_seconds"] = value
                self.reject()
        self.setUp()
        self.request["authorize"] = True
        self.reject()

    def test_boolean_and_unknown_fields_at_every_schema_level(self):
        for target, key in (("genesis", "max_cost_cents"), ("context", "expected_sequence"),
                            ("request", "overhead_cents")):
            with self.subTest(target=target):
                self.setUp()
                getattr(self, target)[key] = True
                self.reject()
        for target in ("ledger", "genesis", "context", "request"):
            with self.subTest(target=target):
                self.setUp()
                getattr(self, target)["new_authority"] = True
                self.reject()
        self.setUp()
        self.append()
        self.ledger["records"][0]["refund"] = 60
        self.reject()

    def test_negative_rate_or_overhead_cannot_refund(self):
        for key in ("rate_cents_per_hour", "overhead_cents"):
            with self.subTest(key=key):
                self.setUp()
                self.request[key] = -1
                self.reject()

    def test_duplicate_keys_nonfinite_and_wrong_input_type(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'[]', b'{} trailing', b'\xff', "{}"):
            with self.subTest(raw=raw):
                with self.assertRaises(budget.BudgetError):
                    budget.observe_reservation(raw, encoded(self.request), encoded(self.context))

    def test_byte_and_record_bounds(self):
        with self.assertRaises(budget.BudgetError):
            budget.observe_reservation(b" " * (budget.MAX_LEDGER_BYTES + 1), encoded(self.request), encoded(self.context))
        with self.assertRaises(budget.BudgetError):
            budget.observe_reservation(encoded(self.ledger), b" " * (budget.MAX_INPUT_BYTES + 1), encoded(self.context))
        self.ledger["records"] = [{}] * (budget.MAX_RECORDS + 1)
        self.reject()

    def test_no_input_mutation(self):
        snapshot = copy.deepcopy((self.ledger, self.request, self.context))
        self.observe()
        self.assertEqual((self.ledger, self.request, self.context), snapshot)


if __name__ == "__main__":
    unittest.main()
