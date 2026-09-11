from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "frontier" / "ghlore_030_evaluation.json"
EXACT_REVISION = "87290a46c79e26ebb0d47575263b7ee34c1c1390"
PREDECESSOR = "82fd2b25be9205025afa43c334a578307b95d7b4"


def load_contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_exact_source_is_fixed_and_held() -> None:
    contract = load_contract()
    assert contract["candidate"] == "huggingface/ghlore"
    assert contract["sourceVersion"] == "0.3.0-source"
    assert contract["upstreamRevision"] == EXACT_REVISION
    assert contract["predecessorRevision"] == PREDECESSOR
    assert contract["upstreamRevision"] != contract["predecessorRevision"]
    assert contract["disposition"] == "HOLD"
    assert contract["evaluationOnly"] is True
    assert contract["productionEligible"] is False
    assert contract["automaticPromotionAuthorized"] is False


def test_no_effectors_or_implicit_trust_are_granted() -> None:
    authority = load_contract()["authority"]
    assert authority
    assert all(value is False for value in authority.values())


def test_network_perimeter_is_explicit_and_fail_closed() -> None:
    network = load_contract()["networkContract"]
    assert network["trustNetworkDefault"] is False
    assert network["independentPrivatePerimeterRequiredForEvaluation"] is True
    assert network["lossOfPerimeterMustFailClosed"] is True


def test_retrieval_remains_untrusted_and_advisory() -> None:
    retrieval = load_contract()["retrievalContract"]
    assert retrieval["retrievedProseUntrusted"] is True
    assert retrieval["perLineQuoteBoundaryRequired"] is True
    assert retrieval["citationsRequired"] is True
    assert retrieval["freshnessRequired"] is True
    assert retrieval["currentSourceRevalidationRequiredBeforeChange"] is True
    assert retrieval["inflightClaimsRepositoryScoped"] is True
    assert retrieval["inflightClaimsAdvisoryOnly"] is True
    assert retrieval["truncationMustBeExplicit"] is True


def test_successor_evidence_cannot_be_skipped() -> None:
    evidence = set(load_contract()["requiredEvidence"])
    required = {
        "client-daemon-wire-version-mismatch-refusal",
        "trust-network-default-off-and-perimeter-negative-path",
        "read-only-github-ingestion",
        "per-line-untrusted-quotation-test",
        "repo-scoped-inflight-relationship-test",
        "citation-freshness-and-truncation-test",
        "fixed-query-relevance-baseline-comparison",
        "stale-history-and-prompt-injection-negative-path",
        "retention-deletion-unavailable-backend-credential-loss",
        "rollback-disable",
    }
    assert required.issubset(evidence)
