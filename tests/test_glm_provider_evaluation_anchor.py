from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = (
    ROOT
    / "frontier"
    / "evidence"
    / "glm-5-3-flash-provider-evaluation-2026-09-08.json"
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def load() -> dict:
    return json.loads(ANCHOR.read_text(encoding="utf-8"))


def test_anchor_binds_exact_source_run_artifact_and_hub_commit() -> None:
    data = load()
    assert data["schema"] == "szl.forge.frontier-evaluation-anchor.v1"
    assert HEX40.fullmatch(data["source"]["commit"])
    assert data["source"]["workflow_run_id"] == 34177931732
    assert data["source"]["workflow_job_id"] == 101910979655
    assert HEX64.fullmatch(data["source"]["artifact"]["sha256"])
    projection = data["hugging_face_projection"]
    assert HEX40.fullmatch(projection["commit"])
    assert projection["resolved_commit_verified"] is True
    assert projection["published_files"] == [
        "bundle.json",
        "receipt.json",
        "runner-output.jsonl",
        "summary.json",
    ]


def test_candidate_and_baseline_identities_are_immutable() -> None:
    data = load()
    assert data["candidate"]["model_id"] == "zai-org/GLM-5.3-Flash"
    assert HEX40.fullmatch(data["candidate"]["source_revision"])
    assert data["baseline"]["model_id"] == (
        "SZLHOLDINGS/SZL-Khipu-1.5B-GGUF"
    )
    assert HEX40.fullmatch(data["baseline"]["revision"])
    assert HEX64.fullmatch(data["baseline"]["weight_sha256"])


def test_measurements_are_real_bounded_and_candidate_beats_baseline_on_suite() -> None:
    data = load()
    measured = data["measurements"]
    assert measured["truth_label"] == "MEASURED"
    assert data["suite"]["name"] == "full"
    assert data["suite"]["case_count"] == 4
    assert measured["candidate"]["successful_call_rate"] == 1.0
    assert measured["baseline"]["successful_call_rate"] == 1.0
    assert measured["comparison"]["score_rate_delta"] == 0.5
    assert measured["comparison"]["schema_valid_rate_delta"] == 0.5
    assert 0 < measured["comparison"]["mean_latency_ratio"] < 1
    assert (
        measured["candidate"]["score_rate"]
        > measured["baseline"]["score_rate"]
    )


def test_fallback_is_semantically_safe_without_granting_authority() -> None:
    fallback = load()["fallback"]
    assert fallback["provider_failure_measured"] is True
    assert fallback["baseline_transport_pass"] is True
    assert fallback["baseline_semantic_pass"] is False
    assert fallback["selected_source"] == "DETERMINISTIC_SAFETY_GUARD"
    assert fallback["selected_output"] == {
        "decision": "ESCALATE",
        "answer": "PROVIDER_UNAVAILABLE",
        "evidence_ids": ["F1"],
        "tool_calls": [],
    }
    assert fallback["transport_pass"] is True
    assert fallback["semantic_safety_pass"] is True
    assert fallback["production_fallback_qualified"] is True
    assert fallback["production_authority"] == "NONE"
    assert HEX64.fullmatch(fallback["selected_output_sha256"])
    assert HEX64.fullmatch(fallback["evidence_sha256"])


def test_receipt_is_content_addressed_but_not_misrepresented_as_authenticated() -> None:
    receipt = load()["receipt"]
    for key in (
        "receipt_sha256",
        "receipt_file_sha256",
        "summary_file_sha256",
        "bundle_file_sha256",
    ):
        assert HEX64.fullmatch(receipt[key])
    assert receipt["signature_status"] == "UNSIGNED_HONEST"
    assert receipt["authenticity_not_established"] is True
    assert receipt["violated_invariants"] == []


def test_anchor_never_promotes_the_candidate() -> None:
    data = load()
    assert data["decision"] == "EVIDENCE_COMPLETE_REVIEW_REQUIRED"
    assert data["production_disposition"] == "HOLD"
    assert data["promotion_effect"] == "NONE"
    bounds = set(data["known_bounds"])
    assert (
        "provider_execution_weight_revision_not_attested_by_chat_response"
        in bounds
    )
    assert "integrity_receipt_is_unsigned" in bounds
    assert any("long_context" in value for value in bounds)
    gates = " ".join(data["next_promotion_gates"]).lower()
    assert "signed" in gates or "signature" in gates
    assert "a11oy" in gates
