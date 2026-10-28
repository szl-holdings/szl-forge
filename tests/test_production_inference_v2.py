from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from inference.production import (
    ProductionBoundaryError,
    canonical_sha256,
    load_production_contract,
    production_infer,
    text_sha256,
    verify_external_execution,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "inference" / "production_control_plane.v2.json"
TEXT = "The exact locked formula set contains eight IDs."
TEXT_DIGEST = text_sha256(TEXT)
REV_A = "a" * 40
REV_B = "b" * 40
REV_C = "c" * 40
REV_D = "d" * 40
DIGEST = "e" * 64


def accept_receipt(admission, receipt):
    """Test-local stand-in for the host's receipt signature verifier."""
    return receipt.get("sha256") == DIGEST


def request(**overrides):
    value = {
        "request_id": "req-production-001",
        "prompt": "What is the locked formula set?",
        "principal_id": "principal-001",
        "tenant_id": "tenant-001",
        "policy_revision": REV_D,
        "grounding_required": True,
        "formula_applications": [
            {
                "formula_id": "F1",
                "applicability": "APPLIES",
                "basis_sha256": DIGEST,
                "authorization_basis": True,
            }
        ],
    }
    value.update(overrides)
    return value


def retriever(query, k=6):
    evidence = [
        {
            "node_id": "formula:locked-eight",
            "source": "szl-holdings/lutar-lean",
            "sha256": TEXT_DIGEST,
        }
    ]
    return {
        "schema": "szl.second-brain.hybrid-context/v1",
        "state": "GROUNDED_HANDLES_READY",
        "ready": True,
        "content_access": "HANDLES_ONLY",
        "query_sha256": text_sha256(query),
        "handles": [
            {
                "nodeId": "formula:locked-eight",
                "nodeKind": "INDEX",
                "label": "DECLARED",
                "note": "Locked eight",
            }
        ],
        "evidence": evidence,
        "evidence_set_sha256": canonical_sha256(evidence),
        "ranking_receipt": {"mode": "BM25_ONLY", "selected_count": 1, "k": k},
    }


def hydrator(handles, *, principal_id, tenant_id, policy_revision):
    evidence = [
        {
            "node_id": "formula:locked-eight",
            "source": "szl-holdings/lutar-lean",
            "sha256": TEXT_DIGEST,
        }
    ]
    assert handles[0]["nodeId"] == evidence[0]["node_id"]
    return {
        "schema": "szl.second-brain.authorized-hydration/v1",
        "state": "AUTHORIZED_CONTENT_READY",
        "ready": True,
        "principal_id_sha256": text_sha256(principal_id),
        "tenant_id_sha256": text_sha256(tenant_id),
        "policy_revision": policy_revision,
        "evidence_set_sha256": canonical_sha256(evidence),
        "documents": [
            {
                "node_id": "formula:locked-eight",
                "source": "szl-holdings/lutar-lean",
                "sha256": TEXT_DIGEST,
                "content": TEXT,
            }
        ],
        "content_access": "CONTROLLER_ONLY",
        "raw_graph_nodes_admitted_to_gradients": 0,
    }


class Generator:
    def __init__(self):
        self.calls = 0
        self._identity = {
            "model": {
                "id": "SZLHOLDINGS/test-model",
                "revision": REV_A,
                "adapter_revision": "NONE",
                "tokenizer_revision": REV_B,
                "template_revision": REV_C,
                "quantization_revision": "NONE",
            },
            "runtime": {
                "engine": "test-engine",
                "version": "1.0.0",
                "hardware_fingerprint": "cpu:test",
            },
        }

    def identity(self):
        return copy.deepcopy(self._identity)

    def __call__(self, context):
        self.calls += 1
        assert context["authority"] == "PROPOSAL_ONLY"
        assert context["evidence"][0]["content"] == TEXT
        answer = (
            "The locked set is F1, F4, F7, F11, F12, F18, F19, "
            "and F22 [formula:locked-eight]."
        )
        return {
            "text": answer,
            "output_schema": "szl.answer-with-node-citations/v2",
            "claims": [
                {
                    "label": "MODELED",
                    "statement_sha256": text_sha256(answer),
                    "supporting_node_ids": ["formula:locked-eight"],
                }
            ],
            "citations": ["formula:locked-eight"],
            "identity": self.identity(),
            "metrics": {"tokens_per_second": 1.0},
        }


def witness(envelope):
    decision = "ALLOW"
    reasons = []
    if (
        envelope["stage"] == "PRE_TOOL"
        and envelope["action_admission"]["human_approval"] == "PENDING"
    ):
        decision = "REVIEW"
        reasons = ["A11oy human approval and signed receipt pending"]
    if (
        envelope["stage"] == "POST_TOOL"
        and envelope["postcondition"]["status"] == "FAIL"
    ):
        decision = "REVIEW"
        reasons = ["postcondition failed"]
    return {
        "schema_version": "szl.nemo.decision.v1",
        "decision": decision,
        "violated_rules": [],
        "reasons": reasons,
        "rule_version": "doctrine-v11/E1-E10",
        "input_hash": "sha256:" + canonical_sha256(envelope),
        "receipt_status": "UNSIGNED_HONEST",
    }


def test_contract_is_operational_but_runtime_winner_is_unselected():
    value = load_production_contract(CONTRACT)
    assert value["status"] == "OPERATIONAL_GOVERNED_BOUNDARY"
    assert value["runtime_selection"]["winner"] == "UNSELECTED"
    assert value["formula_authority"]["locked_proven_count"] == 8


def test_happy_path_is_proposal_only_and_does_not_persist_prompt_or_content():
    observed = []
    result = production_infer(
        request(),
        retriever=retriever,
        hydrator=hydrator,
        generator=Generator(),
        witness=witness,
        observer=observed.append,
        contract_path=CONTRACT,
    )
    assert result["state"] == "PROPOSAL"
    assert result["authority_state"] == "NO_ACTION_AUTHORITY"
    assert result["executed"] is False
    assert [item["stage"] for item in result["nemo"]] == [
        "PRE_GENERATION",
        "POST_GENERATION",
    ]
    assert result["receipt"]["signature"]["status"] == "UNSIGNED_LOCAL"
    assert result["anatomy_observation"]["delivery"] == "DELIVERED"
    assert observed[0]["observer_authority"] == "NONE"
    serialized = json.dumps(result, sort_keys=True)
    assert request()["prompt"] not in serialized
    assert TEXT not in serialized
    assert "chain_of_thought" not in serialized


def test_receipt_is_deterministic_for_same_inputs():
    first = production_infer(
        request(),
        retriever=retriever,
        hydrator=hydrator,
        generator=Generator(),
        witness=witness,
        contract_path=CONTRACT,
    )
    second = production_infer(
        request(),
        retriever=retriever,
        hydrator=hydrator,
        generator=Generator(),
        witness=witness,
        contract_path=CONTRACT,
    )
    assert first["receipt"]["receipt_sha256"] == second["receipt"]["receipt_sha256"]


def test_no_grounding_abstains_before_hydration_or_generation():
    generator = Generator()

    def empty_retriever(query, k=6):
        return {
            "ready": False,
            "content_access": "HANDLES_ONLY",
            "handles": [],
            "evidence": [],
            "evidence_set_sha256": canonical_sha256([]),
            "ranking_receipt": {"mode": "BM25_ONLY", "selected_count": 0},
        }

    result = production_infer(
        request(),
        retriever=empty_retriever,
        hydrator=lambda *_args, **_kwargs: pytest.fail("must not hydrate"),
        generator=generator,
        witness=witness,
        contract_path=CONTRACT,
    )
    assert result["state"] == "ABSTAIN"
    assert generator.calls == 0


def test_acl_or_hydration_binding_failure_blocks_before_generation():
    generator = Generator()

    def denied_hydrator(handles, **kwargs):
        value = hydrator(handles, **kwargs)
        value["tenant_id_sha256"] = "0" * 64
        return value

    result = production_infer(
        request(),
        retriever=retriever,
        hydrator=denied_hydrator,
        generator=generator,
        witness=witness,
        contract_path=CONTRACT,
    )
    assert result["state"] == "BLOCKED"
    assert "SECOND_BRAIN_AUTHORIZATION_OR_HYDRATION_INVALID" in result["reason_codes"]
    assert generator.calls == 0


def test_unpinned_identity_blocks_before_generation():
    generator = Generator()
    generator._identity["model"]["tokenizer_revision"] = "main"
    result = production_infer(
        request(),
        retriever=retriever,
        hydrator=hydrator,
        generator=generator,
        witness=witness,
        contract_path=CONTRACT,
    )
    assert result["state"] == "BLOCKED"
    assert "GENERATOR_IDENTITY_INVALID" in result["reason_codes"]
    assert generator.calls == 0


def test_private_reasoning_output_is_blocked_and_not_persisted():
    class LeakingGenerator(Generator):
        def __call__(self, context):
            value = super().__call__(context)
            value["text"] = "<think>private scratchpad</think> final"
            return value

    result = production_infer(
        request(),
        retriever=retriever,
        hydrator=hydrator,
        generator=LeakingGenerator(),
        witness=witness,
        contract_path=CONTRACT,
    )
    assert result["state"] == "BLOCKED"
    assert result["output"] is None
    assert "private scratchpad" not in json.dumps(result)


def test_f23_cannot_be_an_authorization_basis():
    result = production_infer(
        request(
            formula_applications=[
                {
                    "formula_id": "F23",
                    "applicability": "APPLIES",
                    "basis_sha256": DIGEST,
                    "authorization_basis": True,
                }
            ]
        ),
        retriever=retriever,
        hydrator=hydrator,
        generator=Generator(),
        witness=witness,
        contract_path=CONTRACT,
    )
    assert result["state"] == "BLOCKED"
    assert "REQUEST_BOUNDARY_INVALID" in result["reason_codes"]


def test_tool_intent_without_admission_becomes_review_not_execution():
    result = production_infer(
        request(tool_intent={"tool": "repo.write", "target": "example"}),
        retriever=retriever,
        hydrator=hydrator,
        generator=Generator(),
        witness=witness,
        contract_path=CONTRACT,
    )
    assert result["state"] == "REVIEW"
    assert result["authority_state"] == "HUMAN_APPROVAL_AND_SIGNED_RECEIPT_REQUIRED"
    assert result["executed"] is False
    assert [item["stage"] for item in result["nemo"]][-1] == "PRE_TOOL"


def test_signed_a11oy_admission_makes_action_ready_but_forge_still_does_not_execute():
    admission = {
        "authority": "A11OY",
        "human_approval": "APPROVED",
        "signed_receipt_required": True,
    }
    receipt = {"signature_status": "SIGNED_VERIFIED", "sha256": DIGEST}
    result = production_infer(
        request(
            tool_intent={"tool": "repo.write", "target": "example"},
            action_admission=admission,
            action_receipt=receipt,
        ),
        retriever=retriever,
        hydrator=hydrator,
        generator=Generator(),
        witness=witness,
        contract_path=CONTRACT,
        receipt_verifier=accept_receipt,
    )
    assert result["state"] == "ACTION_READY_EXTERNAL_EXECUTION"
    assert result["authority_state"] == "A11OY_SIGNED_ADMISSION_VERIFIED"
    assert result["executed"] is False
    assert result["reason_codes"] == ["FORGE_DID_NOT_EXECUTE_TOOL"]

    verified = verify_external_execution(
        result,
        tool_result_sha256="f" * 64,
        postcondition_status="PASS",
        postcondition_details_sha256="1" * 64,
        action_admission=admission,
        action_receipt=receipt,
        witness=witness,
        receipt_verifier=accept_receipt,
    )
    assert verified["state"] == "VERIFIED"
    assert verified["executed"] is True
    assert verified["authority_state"] == "A11OY_EXECUTED_FORGE_DID_NOT_EXECUTE"


def test_failed_postcondition_routes_to_review():
    admission = {
        "authority": "A11OY",
        "human_approval": "APPROVED",
        "signed_receipt_required": True,
    }
    receipt = {"signature_status": "SIGNED_VERIFIED", "sha256": DIGEST}
    result = production_infer(
        request(
            tool_intent={"tool": "repo.write", "target": "example"},
            action_admission=admission,
            action_receipt=receipt,
        ),
        retriever=retriever,
        hydrator=hydrator,
        generator=Generator(),
        witness=witness,
        contract_path=CONTRACT,
        receipt_verifier=accept_receipt,
    )
    verified = verify_external_execution(
        result,
        tool_result_sha256="f" * 64,
        postcondition_status="FAIL",
        postcondition_details_sha256="1" * 64,
        action_admission=admission,
        action_receipt=receipt,
        witness=witness,
        receipt_verifier=accept_receipt,
    )
    assert verified["state"] == "REVIEW"
    assert verified["postcondition"]["status"] == "FAIL"


def test_action_contract_cannot_change_after_admission():
    admission = {
        "authority": "A11OY",
        "human_approval": "APPROVED",
        "signed_receipt_required": True,
    }
    receipt = {"signature_status": "SIGNED_VERIFIED", "sha256": DIGEST}
    result = production_infer(
        request(
            tool_intent={"tool": "repo.write", "target": "example"},
            action_admission=admission,
            action_receipt=receipt,
        ),
        retriever=retriever,
        hydrator=hydrator,
        generator=Generator(),
        witness=witness,
        contract_path=CONTRACT,
        receipt_verifier=accept_receipt,
    )
    changed = dict(admission, human_approval="PENDING")
    with pytest.raises(ProductionBoundaryError, match="changed after inference"):
        verify_external_execution(
            result,
            tool_result_sha256="f" * 64,
            postcondition_status="PASS",
            postcondition_details_sha256="1" * 64,
            action_admission=changed,
            action_receipt=receipt,
            witness=witness,
            receipt_verifier=accept_receipt,
        )


APPROVED_ADMISSION = {
    "authority": "A11OY",
    "human_approval": "APPROVED",
    "signed_receipt_required": True,
}
FORGED_RECEIPT = {"signature_status": "SIGNED_VERIFIED", "sha256": DIGEST}


def _raising_verifier(admission, receipt):
    raise RuntimeError("verifier unavailable")


def _action_request():
    return request(
        tool_intent={"tool": "repo.write", "target": "example"},
        action_admission=dict(APPROVED_ADMISSION),
        action_receipt=dict(FORGED_RECEIPT),
    )


@pytest.mark.parametrize(
    "verifier",
    [
        None,
        lambda admission, receipt: False,
        lambda admission, receipt: 1,
        lambda admission, receipt: "True",
        _raising_verifier,
    ],
    ids=["no-verifier", "false", "truthy-int", "truthy-str", "raises"],
)
def test_approved_admission_labels_alone_never_reach_action_ready(verifier):
    kwargs = {} if verifier is None else {"receipt_verifier": verifier}
    result = production_infer(
        _action_request(),
        retriever=retriever,
        hydrator=hydrator,
        generator=Generator(),
        witness=witness,
        contract_path=CONTRACT,
        **kwargs,
    )
    assert result["state"] == "BLOCKED"
    assert result["authority_state"] == "NONE"
    assert result["reason_codes"][0] == "A11OY_ACTION_CONTRACT_INVALID"
    assert result["executed"] is False


def test_receipt_verifier_cannot_mutate_the_bound_admission():
    def mutating_verifier(admission, receipt):
        admission["human_approval"] = "PENDING"
        receipt["sha256"] = "0" * 64
        return True

    req = _action_request()
    result = production_infer(
        req,
        retriever=retriever,
        hydrator=hydrator,
        generator=Generator(),
        witness=witness,
        contract_path=CONTRACT,
        receipt_verifier=mutating_verifier,
    )
    assert result["state"] == "ACTION_READY_EXTERNAL_EXECUTION"
    assert result["action_admission_sha256"] == canonical_sha256(APPROVED_ADMISSION)
    assert result["action_receipt_sha256"] == canonical_sha256(FORGED_RECEIPT)


def test_post_tool_verification_requires_independent_receipt_verification():
    result = production_infer(
        _action_request(),
        retriever=retriever,
        hydrator=hydrator,
        generator=Generator(),
        witness=witness,
        contract_path=CONTRACT,
        receipt_verifier=accept_receipt,
    )
    assert result["state"] == "ACTION_READY_EXTERNAL_EXECUTION"
    with pytest.raises(ProductionBoundaryError, match="independently verified"):
        verify_external_execution(
            result,
            tool_result_sha256="f" * 64,
            postcondition_status="PASS",
            postcondition_details_sha256="1" * 64,
            action_admission=dict(APPROVED_ADMISSION),
            action_receipt=dict(FORGED_RECEIPT),
            witness=witness,
        )
