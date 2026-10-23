# SPDX-License-Identifier: Apache-2.0
"""SIMULATED binding fixtures; mocked corpus success is not candidate replay."""
import copy
from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "frontier/msq650_admission"
SPEC = importlib.util.spec_from_file_location("_msq650_preflight_tests", PACKAGE / "__init__.py",
    submodule_search_locations=[str(PACKAGE)])
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
PREFLIGHT = __import__(SPEC.name + ".preflight", fromlist=["preflight"])
FIXTURE_SPEC = importlib.util.spec_from_file_location("_msq650_shared_fixture",
    ROOT / "tests/test_msq650_coexistence.py")
FIXTURE_MODULE = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(FIXTURE_MODULE)


def fixture():
    plan, before, after = FIXTURE_MODULE.fixture()
    pin = PREFLIGHT.coexistence.MANIFEST_SHA256
    genesis = {"campaign_id": plan["campaign_id"], "candidate_sha256": pin,
        "start_epoch": 1000, "deadline_epoch": 181000, "max_cost_cents": 200000,
        "max_wall_seconds": 180000}
    tip = PREFLIGHT.campaign_budget.canonical_sha256(genesis)
    ledger = {"schema": "szl.msq650.campaign-ledger.v1", "genesis": genesis, "records": []}
    request = {"lease_id": plan["lease_id"], "candidate_sha256": pin, "start_epoch": 1003,
        "wall_seconds": 97, "rate_cents_per_hour": 57, "overhead_cents": 5}
    context = {"expected_campaign_id": plan["campaign_id"], "expected_candidate_sha256": pin,
        "expected_start_epoch": 1000, "expected_genesis_sha256": tip, "expected_tip_sha256": tip,
        "expected_sequence": 0, "current_epoch": 1003, "observed_at_epoch": 1003}
    return plan, before, after, ledger, request, context


def inputs(values):
    plan, before, after, ledger, request, context = values
    raw_plan = PREFLIGHT.coexistence.canonical(plan)
    return {"corpus_buffers": {key: b"SIMULATED" for key in
                ("manifest", "train", "dev", "units", "rights", "acquisition")},
        "plan_bytes": raw_plan, "before_bytes": PREFLIGHT.coexistence.canonical(before),
        "after_bytes": PREFLIGHT.coexistence.canonical(after),
        "expected_plan_sha256": hashlib.sha256(raw_plan).hexdigest(),
        "ledger_bytes": PREFLIGHT.coexistence.canonical(ledger),
        "request_bytes": PREFLIGHT.coexistence.canonical(request),
        "context_bytes": PREFLIGHT.coexistence.canonical(context), "now_epoch": 1003}


class PreflightTests(unittest.TestCase):
    def modeled(self, values):
        observation = PREFLIGHT.corpus_contract.CorpusObservation(True, True, ())
        with patch.object(PREFLIGHT.corpus_contract, "observe_corpus", return_value=observation):
            return PREFLIGHT.observe_preflight(**inputs(values))

    def test_bound_simulation_still_blocked_non_boolean_frozen(self):
        result = self.modeled(fixture())
        self.assertTrue(result.inputs_consistent and result.bindings_consistent)
        self.assertTrue(result.corpus_verified and result.coexistence_consistent and result.budget_feasible)
        self.assertEqual((result.state, result.evidence_class), ("BLOCKED", "MODELED"))
        self.assertFalse(result.training_admitted or result.execution_authority or result.publication_eligible)
        self.assertIn("ISOLATION_BACKEND_UNAVAILABLE", result.reasons)
        with self.assertRaises(TypeError):
            bool(result)
        with self.assertRaises(FrozenInstanceError):
            result.training_admitted = True

    def test_without_corpus_mock_synthetic_candidate_rejected(self):
        result = PREFLIGHT.observe_preflight(**inputs(fixture()))
        self.assertFalse(result.inputs_consistent or result.corpus_verified or result.training_admitted)

    def test_campaign_and_lease_cross_binding(self):
        for key in ("campaign_id", "lease_id"):
            values = fixture()
            values[0][key] = "55555555-5555-5555-5555-555555555555"
            result = self.modeled(values)
            self.assertTrue(result.budget_feasible and result.coexistence_consistent)
            self.assertFalse(result.bindings_consistent)
            self.assertIn("CROSS_CONTRACT_BINDING_MISMATCH", result.reasons)

    def test_candidate_cannot_be_swapped_in_individually_consistent_budget(self):
        values = fixture()
        _, _, _, ledger, request, context = values
        ledger["genesis"]["candidate_sha256"] = "f" * 64
        request["candidate_sha256"] = context["expected_candidate_sha256"] = "f" * 64
        tip = PREFLIGHT.campaign_budget.canonical_sha256(ledger["genesis"])
        context["expected_genesis_sha256"] = context["expected_tip_sha256"] = tip
        result = self.modeled(values)
        self.assertTrue(result.budget_feasible)
        self.assertFalse(result.inputs_consistent)

    def test_entire_remaining_lease_must_be_charged(self):
        for start, wall in ((1003, 96), (1003, 98), (1004, 96)):
            values = fixture()
            values[4].update(start_epoch=start, wall_seconds=wall)
            result = self.modeled(values)
            self.assertTrue(result.budget_feasible)
            self.assertFalse(result.bindings_consistent)

    def test_clock_must_match_all_contracts(self):
        values = fixture()
        values[5]["current_epoch"] = 1002
        values[5]["observed_at_epoch"] = 1002
        result = self.modeled(values)
        self.assertTrue(result.budget_feasible)
        self.assertFalse(result.bindings_consistent)

    def test_budget_failure_remains_blocked(self):
        values = fixture()
        values[4]["overhead_cents"] = 200000
        result = self.modeled(values)
        self.assertFalse(result.inputs_consistent or result.budget_feasible)
        self.assertIn("BUDGET_INVALID_OR_INFEASIBLE", result.reasons)

    def test_identity_failure_remains_blocked(self):
        values = fixture()
        values[2]["incumbent"]["start_ticks"] += 1
        result = self.modeled(values)
        self.assertFalse(result.inputs_consistent or result.coexistence_consistent)

    def test_closed_corpus_field_set_and_no_launcher_argument(self):
        for value in (None, [], {}, {"train": b"bad"}):
            kwargs = inputs(fixture())
            kwargs["corpus_buffers"] = value
            self.assertFalse(PREFLIGHT.observe_preflight(**kwargs).inputs_consistent)
        with self.assertRaises(TypeError):
            PREFLIGHT.observe_preflight(**inputs(fixture()), isolation_qualified=True)

    def test_malformed_budget_and_plan_are_not_exception_authority(self):
        for field in ("plan_bytes", "ledger_bytes", "request_bytes", "context_bytes"):
            kwargs = copy.deepcopy(inputs(fixture()))
            kwargs[field] = b'{"x":0,"x":1}'
            self.assertFalse(PREFLIGHT.observe_preflight(**kwargs).training_admitted)


if __name__ == "__main__":
    unittest.main()
