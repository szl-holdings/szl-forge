from __future__ import annotations

import copy
import importlib.metadata
import json

import pytest

pytest.importorskip("second_brain")
pytest.importorskip("szl_nemo")

from inference.production import (
    make_second_brain_hydrator,
    make_second_brain_retriever,
    make_szl_nemo_envelope_witness,
    production_infer,
    text_sha256,
    verify_external_execution,
)

POLICY_REVISION = "a" * 40
MODEL_REVISION = "b" * 40
TOKENIZER_REVISION = "c" * 40
TEMPLATE_REVISION = "d" * 40
SIGNED_RECEIPT_DIGEST = "e" * 64


class IntegrationGenerator:
    def __init__(self) -> None:
        self._identity = {
            "model": {
                "id": "SZLHOLDINGS/integration-proposal-model",
                "revision": MODEL_REVISION,
                "adapter_revision": "NONE",
                "tokenizer_revision": TOKENIZER_REVISION,
                "template_revision": TEMPLATE_REVISION,
                "quantization_revision": "NONE",
            },
            "runtime": {
                "engine": "integration-stub",
                "version": "1.0.0",
                "hardware_fingerprint": "github-actions:cpu",
            },
        }

    def identity(self):
        return copy.deepcopy(self._identity)

    def __call__(self, context):
        assert context["authority"] == "PROPOSAL_ONLY"
        assert context["evidence"]
        node_id = context["evidence"][0]["node_id"]
        answer = f"Lambda remains an open conjecture and advisory only [{node_id}]."
        return {
            "text": answer,
            "output_schema": "szl.answer-with-node-citations/v2",
            "claims": [
                {
                    "label": "REPORTED",
                    "statement_sha256": text_sha256(answer),
                    "supporting_node_ids": [node_id],
                }
            ],
            "citations": [node_id],
            "identity": self.identity(),
            "metrics": {"generator": "deterministic-integration-stub"},
        }


def _runtime():
    assert importlib.metadata.version("szl-nemo") == "0.4.0"
    assert importlib.metadata.version("szl-second-brain") == "1.2.0"
    retriever = make_second_brain_retriever()

    def authorizer(principal_id, tenant_id, policy_revision, node_id, source):
        return bool(
            principal_id == "integration-principal"
            and tenant_id == "integration-tenant"
            and policy_revision == POLICY_REVISION
            and node_id
            and source
        )

    hydrator = make_second_brain_hydrator(authorizer)
    witness = make_szl_nemo_envelope_witness()
    return retriever, hydrator, witness


def _request(**overrides):
    request = {
        "request_id": "runtime-integration-001",
        "prompt": "Explain Lambda uniqueness and its current proof status.",
        "principal_id": "integration-principal",
        "tenant_id": "integration-tenant",
        "policy_revision": POLICY_REVISION,
        "grounding_required": True,
        "formula_applications": [],
    }
    request.update(overrides)
    return request


def test_real_second_brain_and_nemo_produce_a_governed_proposal():
    retriever, hydrator, witness = _runtime()
    result = production_infer(
        _request(),
        retriever=retriever,
        hydrator=hydrator,
        generator=IntegrationGenerator(),
        witness=witness,
    )
    assert result["state"] == "PROPOSAL"
    assert result["executed"] is False
    assert result["evidence_handles"]
    assert [record["decision"] for record in result["nemo"]] == [
        "ALLOW",
        "ALLOW",
    ]
    assert [record["rule_version"] for record in result["nemo"]] == [
        "doctrine-v11/E1-E10",
        "doctrine-v11/E1-E10",
    ]
    serialized = json.dumps(result, sort_keys=True)
    assert _request()["prompt"] not in serialized
    assert "chain_of_thought" not in serialized
    assert result["anatomy_observation"]["event"]["hydrated_content_present"] is False
    assert result["continuation"]["hydrated_content_present"] is False
    assert all("content" not in handle for handle in result["evidence_handles"])
    assert all(
        "content" not in item
        for item in result["continuation"]["evidence"]["items"]
    )


def test_real_nemo_routes_unapproved_tool_intent_to_review():
    retriever, hydrator, witness = _runtime()
    result = production_infer(
        _request(tool_intent={"tool": "repo.write", "target": "example"}),
        retriever=retriever,
        hydrator=hydrator,
        generator=IntegrationGenerator(),
        witness=witness,
    )
    assert result["state"] == "REVIEW"
    assert result["authority_state"] == "HUMAN_APPROVAL_AND_SIGNED_RECEIPT_REQUIRED"
    assert result["executed"] is False
    assert result["nemo"][-1]["stage"] == "PRE_TOOL"
    assert result["nemo"][-1]["decision"] == "REVIEW"


def test_real_nemo_verifies_signed_a11oy_admission_and_postcondition():
    retriever, hydrator, witness = _runtime()
    admission = {
        "authority": "A11OY",
        "human_approval": "APPROVED",
        "signed_receipt_required": True,
    }
    receipt = {
        "signature_status": "SIGNED_VERIFIED",
        "sha256": SIGNED_RECEIPT_DIGEST,
    }
    result = production_infer(
        _request(
            tool_intent={"tool": "repo.write", "target": "example"},
            action_admission=admission,
            action_receipt=receipt,
        ),
        retriever=retriever,
        hydrator=hydrator,
        generator=IntegrationGenerator(),
        witness=witness,
    )
    assert result["state"] == "ACTION_READY_EXTERNAL_EXECUTION"
    assert result["nemo"][-1]["decision"] == "ALLOW"
    assert result["executed"] is False

    verification = verify_external_execution(
        result,
        tool_result_sha256="f" * 64,
        postcondition_status="PASS",
        postcondition_details_sha256="1" * 64,
        action_admission=admission,
        action_receipt=receipt,
        witness=witness,
    )
    assert verification["state"] == "VERIFIED"
    assert verification["nemo"]["decision"] == "ALLOW"
    assert verification["executed"] is True
    assert verification["authority_state"] == "A11OY_EXECUTED_FORGE_DID_NOT_EXECUTE"
