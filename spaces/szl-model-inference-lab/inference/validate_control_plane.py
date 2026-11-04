#!/usr/bin/env python3
"""Fail-closed validator for the SZL Forge inference control-plane contract."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONTRACT = ROOT / "inference" / "control_plane.v1.json"
LOCKED_EIGHT = ("F1", "F4", "F7", "F11", "F12", "F18", "F19", "F22")
REQUIRED_PLANES = {
    "second-brain-evidence",
    "model-execution",
    "nemo-witness",
    "a11oy-action-gate",
    "anatomy-observer",
}
REQUIRED_GATES = {
    "FORMULA_BINDING_VALID",
    "DATA_RIGHTS_AND_LINEAGE_VALID",
    "TRAIN_EVAL_TEST_SEPARATION_VALID",
    "SECOND_BRAIN_GROUNDING_VALID",
    "SECOND_BRAIN_ABSTENTION_VALID",
    "STRUCTURED_OUTPUT_VALID",
    "NEMO_PRE_GENERATION_VALID",
    "NEMO_POST_GENERATION_VALID",
    "TOOL_AUTHORITY_VALID",
    "POSTCONDITION_VALID",
    "QUALITY_NON_REGRESSION_VALID",
    "LATENCY_MEMORY_COST_MEASURED",
    "REPRODUCIBILITY_VALID",
    "SIGNED_RECEIPT_AND_REPLAY_VALID",
}
REQUIRED_TRUTH_LABELS = {
    "MEASURED",
    "REPORTED",
    "MODELED",
    "CONJECTURE",
    "UNKNOWN",
    "UNAVAILABLE",
}
PIN_RE = re.compile(r"^[0-9a-f]{40}$")


class ContractError(ValueError):
    """Raised when the inference contract would permit ambiguous authority."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def contract_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _plane_map(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    planes = document.get("planes")
    if not isinstance(planes, list):
        raise ContractError("planes must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for plane in planes:
        if not isinstance(plane, dict) or not isinstance(plane.get("id"), str):
            raise ContractError("each plane must be an object with an id")
        if plane["id"] in by_id:
            raise ContractError(f"duplicate plane id: {plane['id']}")
        by_id[plane["id"]] = plane
    if set(by_id) != REQUIRED_PLANES:
        raise ContractError(f"plane set drift: {sorted(by_id)}")
    return by_id


def validate(document: dict[str, Any]) -> dict[str, Any]:
    """Validate semantic invariants and return a defensive copy."""
    if not isinstance(document, dict):
        raise ContractError("contract must be a JSON object")
    if document.get("schema") != "szl.forge.inference-control-plane/v1":
        raise ContractError("unsupported inference contract schema")
    if document.get("status") != "EXPERIMENTAL_FAIL_CLOSED":
        raise ContractError("contract must remain experimental and fail closed")

    authority = document.get("authority")
    expected_authority = {
        "lifecycle": "szl-holdings/szl-forge",
        "formal_proof": "szl-holdings/lutar-lean",
        "formula_kernel": "szl-holdings/szl-formulas",
        "retrieval": "szl-holdings/szl-second-brain",
        "witness": "szl-holdings/szl-nemo",
        "observer": "szl-holdings/anatomy",
        "action_admission": "szl-holdings/a11oy",
    }
    if authority != expected_authority:
        raise ContractError("authority map drift")

    formula = document.get("formula_binding")
    if not isinstance(formula, dict):
        raise ContractError("formula_binding must be an object")
    if tuple(formula.get("locked_proven_ids", ())) != LOCKED_EIGHT:
        raise ContractError("locked-proven formula set must be the exact formal eight")
    if formula.get("locked_proven_count") != len(LOCKED_EIGHT):
        raise ContractError("locked-proven count/list mismatch")
    if formula.get("callable_formula_count") != 21:
        raise ContractError("callable formula count drift")
    formal_source = formula.get("formal_source") or {}
    kernel_source = formula.get("kernel_source") or {}
    if formal_source.get("repository") != "szl-holdings/lutar-lean":
        raise ContractError("formal formula authority must be lutar-lean")
    if formal_source.get("count_theorem") != "Lutar.Wave8.AxiomDisclosure.locked_count_eight":
        raise ContractError("locked-count theorem binding drift")
    if not PIN_RE.fullmatch(str(formal_source.get("commit", ""))):
        raise ContractError("formal formula source must be pinned to a full commit")
    if kernel_source.get("repository") != "szl-holdings/szl-formulas":
        raise ContractError("formula kernel authority drift")
    if not PIN_RE.fullmatch(str(kernel_source.get("commit", ""))):
        raise ContractError("formula kernel source must be pinned to a full commit")
    if kernel_source.get("f_id_to_callable_mapping") != "UNKNOWN_NOT_ASSERTED":
        raise ContractError("formal F-IDs must not be mapped onto callable formulas without proof")
    lambda_rule = formula.get("lambda") or {}
    if lambda_rule != {
        "formula_id": "F23",
        "status": "CONJECTURE_1_ADVISORY",
        "can_authorize": False,
        "can_be_sole_allow_basis": False,
    }:
        raise ContractError("Lambda must remain Conjecture 1 and non-authorizing")

    planes = _plane_map(document)
    brain = planes["second-brain-evidence"]
    if brain.get("authority") != "READ_ONLY_EVIDENCE":
        raise ContractError("Second Brain may provide evidence only")
    if brain.get("training_authority") != "NONE":
        raise ContractError("Second Brain may not silently become training authority")
    if brain.get("private_graph_in_gradients") is not False:
        raise ContractError("private graph must not enter gradients")
    if brain.get("content_hydration") != "CONTROLLER_SIDE_REQUIRED":
        raise ContractError("handles-only retrieval requires controller-side hydration")

    model = planes["model-execution"]
    if model.get("authority") != "PROPOSAL_ONLY" or model.get("can_execute_tools") is not False:
        raise ContractError("model execution must remain proposal-only")
    if model.get("must_emit_pinned_identity") is not True:
        raise ContractError("model identity must be revision-pinned")

    nemo = planes["nemo-witness"]
    if nemo.get("artifact_kind") != "SOFTWARE_KERNEL":
        raise ContractError("Nemo must be classified as a software kernel")
    if nemo.get("generative") is not False or nemo.get("not_nemotron") is not True:
        raise ContractError("Nemo witness must not be confused with a generative Nemotron wrapper")
    if set(nemo.get("required_stages", ())) != {
        "PRE_GENERATION",
        "POST_GENERATION",
        "PRE_TOOL",
        "POST_TOOL",
    }:
        raise ContractError("Nemo witness stage coverage drift")

    action_gate = planes["a11oy-action-gate"]
    if action_gate.get("authority") != "ACTION_ADMISSION":
        raise ContractError("only A11oy may admit an action")
    if action_gate.get("signed_receipt_required_before_action") is not True:
        raise ContractError("consequential action requires a signed receipt")

    anatomy = planes["anatomy-observer"]
    if anatomy.get("authority") != "NONE" or anatomy.get("observability_only") is not True:
        raise ContractError("Anatomy must remain an observer")
    if anatomy.get("can_change_decision") is not False:
        raise ContractError("Anatomy may not alter a decision")
    if anatomy.get("receives_private_reasoning") is not False:
        raise ContractError("Anatomy may not receive private reasoning")

    runtime = document.get("runtime_selection") or {}
    if runtime.get("winner") != "UNSELECTED":
        raise ContractError("a runtime winner cannot be declared before measured bakeoff")
    if runtime.get("policy") != "MEASURED_BAKEOFF_REQUIRED":
        raise ContractError("runtime selection policy must require measured bakeoff")
    lanes = runtime.get("lanes")
    if not isinstance(lanes, list):
        raise ContractError("runtime lanes must be a list")
    lane_ids = {lane.get("id") for lane in lanes if isinstance(lane, dict)}
    if lane_ids != {"local-cpu-gguf", "single-node-gpu", "distributed-gpu"}:
        raise ContractError("runtime lane set drift")
    distributed = next(lane for lane in lanes if lane["id"] == "distributed-gpu")
    if distributed.get("admission") != "WORKLOAD_JUSTIFIED_ONLY":
        raise ContractError("distributed serving must be workload-justified")
    metrics = set(runtime.get("selection_metrics", ()))
    for required in {
        "schema_valid_rate",
        "grounding_correct_rate",
        "citation_binding_rate",
        "abstention_correct_rate",
        "tool_authority_pass_rate",
        "time_to_first_token_p95_ms",
        "inter_token_latency_p95_ms",
        "tokens_per_second",
        "peak_memory_bytes",
        "cost_per_successful_governed_request",
    }:
        if required not in metrics:
            raise ContractError(f"missing runtime selection metric: {required}")

    gates = set(document.get("promotion_gates", ()))
    if not REQUIRED_GATES.issubset(gates):
        raise ContractError(f"missing promotion gates: {sorted(REQUIRED_GATES - gates)}")

    privacy = document.get("privacy") or {}
    if privacy.get("persist_private_chain_of_thought") is not False:
        raise ContractError("private chain-of-thought persistence is forbidden")
    if privacy.get("persist_raw_private_graph") is not False:
        raise ContractError("raw private graph persistence is forbidden")
    allowed = set(privacy.get("allowed_persisted_artifacts", ()))
    forbidden_tokens = {"private_chain_of_thought", "raw_private_graph", "hidden_reasoning"}
    if allowed & forbidden_tokens:
        raise ContractError("forbidden private artifacts appear in persistence allow-list")

    if set(document.get("truth_labels", ())) != REQUIRED_TRUTH_LABELS:
        raise ContractError("truth-label vocabulary drift")
    blockers = document.get("known_blockers")
    if not isinstance(blockers, list) or not blockers:
        raise ContractError("known blockers must remain explicit")
    blocker_ids = {b.get("id") for b in blockers if isinstance(b, dict)}
    if "A11OY_FORMULA_REGISTRY_DRIFT" not in blocker_ids:
        raise ContractError("known formula-authority drift must block promotion")
    if any(b.get("blocking") is not True for b in blockers if isinstance(b, dict)):
        raise ContractError("known blockers cannot be silently downgraded")

    return copy.deepcopy(document)


def load(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return validate(document)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    path = Path(args[0]) if args else DEFAULT_CONTRACT
    try:
        document = load(path)
    except (OSError, json.JSONDecodeError, ContractError) as exc:
        print(f"BLOCK: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "schema": document["schema"],
        "status": document["status"],
        "contract_sha256": contract_sha256(document),
        "locked_proven_ids": document["formula_binding"]["locked_proven_ids"],
        "runtime_winner": document["runtime_selection"]["winner"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
