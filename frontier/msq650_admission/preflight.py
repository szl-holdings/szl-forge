# SPDX-License-Identifier: Apache-2.0
"""Compose bounded byte observations without creating an execution capability.

This is deliberately not a launcher. Even consistent input remains BLOCKED.
Plan, price, clock and custodian expectations are untrusted comparison inputs;
no hash, local fixture or caller assertion can qualify the missing backend.
"""
from dataclasses import dataclass, field
import json

from . import campaign_budget, coexistence, corpus_contract


@dataclass(frozen=True)
class PreflightObservation:
    inputs_consistent: bool
    reasons: tuple[str, ...]
    corpus_verified: bool = False
    coexistence_consistent: bool = False
    budget_feasible: bool = False
    bindings_consistent: bool = False
    evidence_class: str = field(default="MODELED", init=False)
    state: str = field(default="BLOCKED", init=False)
    training_admitted: bool = field(default=False, init=False)
    execution_authority: bool = field(default=False, init=False)
    publication_eligible: bool = field(default=False, init=False)

    def __bool__(self):
        raise TypeError("PREFLIGHT_OBSERVATION_IS_NOT_AUTHORIZATION")


HOLDS = (
    "RIGHTS_AUTHORITY_UNAVAILABLE", "SEALED_EVALUATION_UNAVAILABLE",
    "SEMANTIC_CONTAMINATION_CLEARANCE_UNKNOWN", "BASE_AND_TOKENIZER_IDENTITY_UNAVAILABLE",
    "NATIVE_DEPENDENCY_AND_RUNTIME_UNQUALIFIED", "ISOLATION_BACKEND_UNAVAILABLE",
    "OWNER_SIGNING_POLICY_UNAVAILABLE", "OBSERVATION_AUTHENTICITY_UNESTABLISHED",
    "DURABLE_SINGLE_USE_RESERVATION_UNAVAILABLE", "PRICE_AND_CLOCK_AUTHORITY_UNESTABLISHED",
)


def observe_preflight(*, corpus_buffers, plan_bytes, before_bytes, after_bytes,
                      expected_plan_sha256, ledger_bytes, request_bytes, context_bytes,
                      now_epoch):
    """Validate captured bytes and cross-bind their candidate/campaign/lease.

corpus_buffers must be the six-name dict required by observe_corpus. Inputs are
bytes captured once, not paths or deferred readers. No filesystem/network side
effects or raw data exposure. The budget reservation must cover the *entire*
remaining lease from now to exclusive expiry, including future setup, worker,
termination, drain and receipt finalization; no execution occurs in this slice.
"""
    if type(corpus_buffers) is not dict or set(corpus_buffers) != {
            "manifest", "train", "dev", "units", "rights", "acquisition"}:
        return PreflightObservation(False, ("CORPUS_INPUT_FIELDS",) + HOLDS)
    corpus = corpus_contract.observe_corpus(**corpus_buffers)
    coexist = coexistence.observe_coexistence(plan_bytes, before_bytes, after_bytes,
        expected_plan_sha256=expected_plan_sha256, now_epoch=now_epoch)
    corpus_ok = corpus.candidate_bytes_verified and corpus.structure_verified
    try:
        budget = campaign_budget.observe_reservation(ledger_bytes, request_bytes, context_bytes)
    except campaign_budget.BudgetError:
        return PreflightObservation(False, ("BUDGET_INVALID_OR_INFEASIBLE",) + HOLDS,
            corpus_ok, coexist.plan_consistent)
    if not (corpus_ok and coexist.plan_consistent):
        return PreflightObservation(False, ("CORPUS_OR_COEXISTENCE_INPUT_INVALID",) + HOLDS,
            corpus_ok, coexist.plan_consistent, True)
    # These exact bytes have already passed each closed parser. Re-reading the
    # immutable buffers cannot introduce a path/snapshot time-of-check race.
    plan = json.loads(plan_bytes)
    context = json.loads(context_bytes)
    record = budget.proposed_record
    manifest_pin = next((digest for name, _, digest in corpus.artifact_pins if name == "manifest"), None)
    bound = (
        manifest_pin == coexistence.MANIFEST_SHA256
        and context["expected_candidate_sha256"] == coexistence.MANIFEST_SHA256
        and record["candidate_sha256"] == coexistence.MANIFEST_SHA256
        and context["expected_campaign_id"] == plan["campaign_id"]
        and record["lease_id"] == plan["lease_id"]
        and context["current_epoch"] == now_epoch
        and record["start_epoch"] == now_epoch
        and record["start_epoch"] + record["wall_seconds"] == plan["expires_at_epoch"]
    )
    return PreflightObservation(bound, HOLDS if bound else ("CROSS_CONTRACT_BINDING_MISMATCH",) + HOLDS,
        corpus_ok, True, True, bound)
