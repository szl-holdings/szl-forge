import copy
import hashlib
import json
from pathlib import Path

import pytest

from inference.governed_inference import canonical_sha256, governed_infer
from inference.validate_control_plane import ContractError, load, validate

CONTRACT = Path("inference/control_plane.v1.json")
TEXT = "The formal locked set has eight formula IDs."
TEXT_SHA = hashlib.sha256(TEXT.encode("utf-8")).hexdigest()
MODEL_REV = "a" * 40


def contract():
    return load(CONTRACT)


def request(**overrides):
    value = {
        "request_id": "req-001",
        "principal_id": "principal-001",
        "tenant_id": "tenant-001",
        "prompt": "What is the locked formula set?",
        "formula_ids": ["F1", "F4", "F7", "F11", "F12", "F18", "F19", "F22"],
        "requires_grounding": True,
    }
    value.update(overrides)
    return value


def retrieval(ready=True, digest=TEXT_SHA):
    handles = [{"nodeId": "formula:locked-eight", "nodeKind": "INDEX", "label": "DECLARED", "note": "Locked eight"}] if ready else []
    evidence = [{"node_id": "formula:locked-eight", "source": "lutar-lean", "sha256": digest}] if ready else []
    return {
        "ready": ready,
        "content_access": "HANDLES_ONLY",
        "handles": handles,
        "evidence": evidence,
        "evidence_set_sha256": canonical_sha256(evidence),
    }


def retriever(query, k=6):
    assert query
    assert 1 <= k <= 12
    return retrieval()


def hydrator(handles, safe_request):
    assert safe_request["tenant_id"] == "tenant-001"
    return [{"node_id": handles[0]["nodeId"], "source": "lutar-lean", "content": TEXT}]


def generator(context):
    assert context["instructions"]["authority"] == "PROPOSAL_ONLY"
    assert context["formula_binding"]["lambda_can_authorize"] is False
    return {
        "text": "The locked set is F1, F4, F7, F11, F12, F18, F19, and F22 [formula:locked-eight].",
        "output_schema": "szl.answer-with-node-citations/v1",
        "model": {"id": "example/model", "revision": MODEL_REV, "adapter_revision": "NONE"},
        "runtime": {"engine": "test-engine", "version": "1.0.0", "hardware_fingerprint": "cpu:test"},
        "metrics": {"tokens_per_second": 1.0},
    }


def witness(stage, payload):
    return {
        "decision": "ALLOW",
        "rule_version": "szl-nemo-test/v1",
        "reason_codes": [],
        "input_sha256": canonical_sha256({"stage": stage, "payload": payload}),
    }


def test_control_plane_contract_validates():
    data = contract()
    assert data["formula_binding"]["locked_proven_count"] == 8
    assert data["runtime_selection"]["winner"] == "UNSELECTED"


def test_formula_set_drift_fails_closed():
    data = contract()
    data["formula_binding"]["locked_proven_ids"] = ["F1", "F11", "F12", "F18", "F19"]
    data["formula_binding"]["locked_proven_count"] = 5
    with pytest.raises(ContractError, match="exact formal eight"):
        validate(data)


def test_callable_formula_namespace_cannot_claim_an_unproved_f_id_mapping():
    data = contract()
    data["formula_binding"]["kernel_source"]["f_id_to_callable_mapping"] = "F1=lambda_aggregate"
    with pytest.raises(ContractError, match="without proof"):
        validate(data)


def test_lambda_cannot_be_promoted_to_authority():
    data = contract()
    data["formula_binding"]["lambda"]["can_authorize"] = True
    with pytest.raises(ContractError, match="Conjecture 1"):
        validate(data)


def test_nemo_cannot_be_reclassified_as_a_generative_model():
    data = contract()
    nemo = next(p for p in data["planes"] if p["id"] == "nemo-witness")
    nemo["generative"] = True
    with pytest.raises(ContractError, match="Nemotron"):
        validate(data)


def test_anatomy_cannot_gain_decision_authority():
    data = contract()
    anatomy = next(p for p in data["planes"] if p["id"] == "anatomy-observer")
    anatomy["authority"] = "ALLOW"
    with pytest.raises(ContractError, match="observer"):
        validate(data)


def test_runtime_winner_cannot_be_declared_without_bakeoff():
    data = contract()
    data["runtime_selection"]["winner"] = "sglang"
    with pytest.raises(ContractError, match="before measured bakeoff"):
        validate(data)


def test_happy_path_is_proposal_only_and_receipted():
    observed = []
    result = governed_infer(
        request(),
        retriever=retriever,
        hydrator=hydrator,
        generator=generator,
        witness=witness,
        observer=observed.append,
    )
    assert result["state"] == "PROPOSAL"
    assert result["authority_state"] == "NO_ACTION_AUTHORITY"
    assert result["executed"] is False
    assert result["receipt"]["signature"]["status"] == "UNSIGNED_LOCAL"
    assert result["receipt"]["signature"]["must_be_signed_before_consequential_action"] is True
    assert result["anatomy_observation"]["delivery"] == "DELIVERED"
    assert observed[0]["observer_authority"] == "NONE"
    assert observed[0]["raw_prompt_present"] is False
    assert observed[0]["private_reasoning_present"] is False
    serialized = json.dumps(result, sort_keys=True)
    assert "What is the locked formula set?" not in serialized
    assert "chain_of_thought" not in serialized


def test_same_inputs_produce_same_receipt_digest():
    first = governed_infer(request(), retriever=retriever, hydrator=hydrator, generator=generator, witness=witness)
    second = governed_infer(request(), retriever=retriever, hydrator=hydrator, generator=generator, witness=witness)
    assert first["receipt"]["receipt_sha256"] == second["receipt"]["receipt_sha256"]


def test_no_grounding_abstains_before_generation():
    calls = {"generator": 0}

    def empty_retriever(query, k=6):
        return retrieval(ready=False)

    def never_generator(context):
        calls["generator"] += 1
        return generator(context)

    result = governed_infer(
        request(),
        retriever=empty_retriever,
        hydrator=lambda handles, safe: [],
        generator=never_generator,
        witness=witness,
    )
    assert result["state"] == "ABSTAIN"
    assert calls["generator"] == 0


def test_hydration_digest_mismatch_blocks_before_generation():
    calls = {"generator": 0}

    def bad_hydrator(handles, safe):
        return [{"node_id": handles[0]["nodeId"], "source": "lutar-lean", "content": "tampered"}]

    def never_generator(context):
        calls["generator"] += 1
        return generator(context)

    result = governed_infer(
        request(),
        retriever=retriever,
        hydrator=bad_hydrator,
        generator=never_generator,
        witness=witness,
    )
    assert result["state"] == "BLOCKED"
    assert "HYDRATION_BOUNDARY_INVALID" in result["reason_codes"]
    assert calls["generator"] == 0


def test_nemo_post_generation_block_suppresses_output():
    def blocking_witness(stage, payload):
        decision = "BLOCK" if stage == "POST_GENERATION" else "ALLOW"
        return {
            "decision": decision,
            "rule_version": "szl-nemo-test/v1",
            "reason_codes": ["R_TEST"] if decision == "BLOCK" else [],
            "input_sha256": canonical_sha256(payload),
        }

    result = governed_infer(
        request(),
        retriever=retriever,
        hydrator=hydrator,
        generator=generator,
        witness=blocking_witness,
    )
    assert result["state"] == "BLOCKED"
    assert result["output"] is None
    assert "NEMO_POST_GENERATION_BLOCK" in result["reason_codes"]


def test_tool_intent_requires_a11oy_and_human_review():
    result = governed_infer(
        request(tool_intent={"tool": "repo.write", "target": "example"}),
        retriever=retriever,
        hydrator=hydrator,
        generator=generator,
        witness=witness,
    )
    assert result["state"] == "REVIEW"
    assert result["authority_state"] == "HUMAN_APPROVAL_REQUIRED"
    assert result["executed"] is False
    assert "TOOL_INTENT_REQUIRES_A11OY_ADMISSION" in result["reason_codes"]


def test_lambda_alone_cannot_authorize_tool_intent():
    result = governed_infer(
        request(formula_ids=["F23"], tool_intent={"tool": "repo.write"}),
        retriever=retriever,
        hydrator=hydrator,
        generator=generator,
        witness=witness,
    )
    assert result["state"] == "BLOCKED"
    assert result["reason_codes"] == ["LAMBDA_CANNOT_AUTHORIZE_ACTION"]


def test_unpinned_model_revision_is_blocked():
    def unpinned_generator(context):
        value = copy.deepcopy(generator(context))
        value["model"]["revision"] = "main"
        return value

    result = governed_infer(
        request(),
        retriever=retriever,
        hydrator=hydrator,
        generator=unpinned_generator,
        witness=witness,
    )
    assert result["state"] == "BLOCKED"
    assert "GENERATOR_BOUNDARY_INVALID" in result["reason_codes"]


def test_private_reasoning_marker_is_blocked_and_not_persisted():
    def leaking_generator(context):
        value = copy.deepcopy(generator(context))
        value["text"] = "<think>private trace</think> final"
        return value

    result = governed_infer(
        request(),
        retriever=retriever,
        hydrator=hydrator,
        generator=leaking_generator,
        witness=witness,
    )
    assert result["state"] == "BLOCKED"
    assert result["output"] is None
    assert "private trace" not in json.dumps(result)


def test_observer_failure_cannot_change_decision():
    def failed_observer(event):
        event["state"] = "ALLOW"
        raise RuntimeError("observer unavailable")

    result = governed_infer(
        request(),
        retriever=retriever,
        hydrator=hydrator,
        generator=generator,
        witness=witness,
        observer=failed_observer,
    )
    assert result["state"] == "PROPOSAL"
    assert result["anatomy_observation"]["delivery"] == "UNAVAILABLE"
