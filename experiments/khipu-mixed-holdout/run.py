#!/usr/bin/env python3
"""Offline verification of a preserved failed Khipu diagnostic.

This verifier reads recorded bytes and independently regrades saved outputs.
It never executes the archived inference runner, imports a model runtime,
downloads weights, writes evidence, or claims an independent inference replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases.json"
SCHEMA = HERE / "evidence" / "plan-schema.json"
RECORDED_SHA256 = {
    "cases.json": "ab19127b68f089e379457e3ea70c14a4c88217136b18fadb1f994542b6ad75dc",
    "preregister.json": "33781211baf1e4fa4446593eb0f0766fa6ece2725c9cfa68729f0fa15b824b5d",
    "evidence/frozen-runner.py.txt": "c2498ec5cdfb63ca625f0a0af52eaed995231d6874711b25b76d55916499504c",
    "evidence/plan-schema.json": "b95f9927366dae7c5d36cfb7de6e229eb605524318ab642a6aa2292a212170d0",
    "evidence/receipt.json": "e3e6ce706c0afe001146e156682e75917fdd9367ef707b69a5835ecbd3125d86",
    "evidence/outputs.jsonl": "190f18728ac0d0cf7401ed2f4553176ffc60b1681abbbe1697c3f499a0f92313",
}


def strict_json(raw: str) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(_value: str) -> None:
        raise ValueError("non-JSON constant")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def load_json(path: Path) -> dict:
    value = strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("recorded JSON document must be an object")
    return value


def grade(raw: str, case: dict, schema: dict) -> dict:
    """Regrade selected structural fields; free-text truth is not verified.

    Unlike the historical runner, reject duplicate keys and non-JSON constants.
    Verification fails if this stricter parse changes any recorded grade.
    """
    from jsonschema import Draft202012Validator

    result = {
        "json_valid": False,
        "schema_valid": False,
        "contract_valid": False,
        "decision_correct": False,
        "citation_correct": False,
        "case_pass": False,
    }
    try:
        plan = strict_json(raw)
    except (ValueError, TypeError):
        return result
    result["json_valid"] = isinstance(plan, dict)
    if not isinstance(plan, dict) or not Draft202012Validator(schema).is_valid(plan):
        return result
    result["schema_valid"] = True
    offered = {candidate["nodeId"] for candidate in case["candidates"]}
    steps = plan.get("steps") or []
    cited = plan.get("citedNodeIds") or []
    cite_steps = {step["nodeId"] for step in steps if step["action"] == "CITE"}
    result["contract_valid"] = (
        plan.get("provenance") == "MODEL_PROPOSED"
        and plan.get("query") == case["query"]
        and plan.get("candidates") == case["candidates"]
        and len(cited) == len(set(cited))
        and set(cited).issubset(offered)
        and all(step["nodeId"] in offered for step in steps)
        and set(cited) == cite_steps
        and (
            (plan.get("decision") == "ABSTAIN" and not steps and not cited and bool(plan.get("abstainReason")))
            or (plan.get("decision") == "NAVIGATE" and bool(cited) and plan.get("abstainReason") is None)
        )
    )
    result["decision_correct"] = plan.get("decision") == case["expected_decision"]
    result["citation_correct"] = set(cited) == set(case["expected_cited_node_ids"])
    result["case_pass"] = (
        result["contract_valid"] and result["decision_correct"] and result["citation_correct"]
    )
    return result


def verify_recorded(directory: Path = HERE) -> dict:
    """Verify exact historical files and grades without executing their source."""
    # Read each file once; grading uses the exact bytes whose digest was checked.
    raw = {}
    for relative, expected in RECORDED_SHA256.items():
        data = (directory / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"recorded byte mismatch: {relative}")
        raw[relative] = data

    frozen = strict_json(raw["preregister.json"].decode("utf-8"))
    receipt = strict_json(raw["evidence/receipt.json"].decode("utf-8"))
    fixture = strict_json(raw["cases.json"].decode("utf-8"))
    schema = strict_json(raw["evidence/plan-schema.json"].decode("utf-8"))
    if frozen["schema"] != "szl.khipu.mixed-holdout.preregister/v1":
        raise ValueError("unexpected historical preregistration")
    if receipt["schema"] != "szl.khipu.mixed-holdout.local-receipt/v1":
        raise ValueError("unexpected historical receipt")
    for field, relative in (
        ("runner_sha256", "evidence/frozen-runner.py.txt"),
        ("plan_schema_sha256", "evidence/plan-schema.json"),
    ):
        if frozen[field] != RECORDED_SHA256[relative] or receipt[field] != frozen[field]:
            raise ValueError(f"historical binding mismatch: {field}")
    if (
        frozen["cases_sha256"] != RECORDED_SHA256["cases.json"]
        or receipt["fixture_sha256"] != frozen["cases_sha256"]
        or receipt["preregister_sha256"] != RECORDED_SHA256["preregister.json"]
        or receipt["outputs_sha256"] != RECORDED_SHA256["evidence/outputs.jsonl"]
    ):
        raise ValueError("historical artifact binding mismatch")
    for field in (
        "source_base_commit", "hub_repo_id", "hub_revision", "source_binding_sha256",
        "training_receipt_file_sha256", "training_receipt_canonical_sha256",
        "expected_weight_sha256", "generation",
    ):
        if receipt[field] != frozen[field]:
            raise ValueError(f"historical metadata mismatch: {field}")
    if (
        frozen["training_eligible"] is not False
        or frozen["promotion_eligible"] is not False
        or fixture["training_eligible"] is not False
        or receipt["promotion_eligible"] is not False
        or receipt["signed"] is not False
        or receipt["independent_replay"] != "NOT_RUN"
    ):
        raise ValueError("historical evidence boundary changed")

    cases = fixture["cases"]
    expected_ids = {
        *(f"NAV-{index:02d}" for index in range(1, 7)),
        *(f"ABS-{index:02d}" for index in range(1, 7)),
    }
    by_id = {case["id"]: case for case in cases}
    rows = [strict_json(line) for line in raw["evidence/outputs.jsonl"].decode("utf-8").splitlines()]
    if (
        len(cases) != 12 or set(by_id) != expected_ids
        or len(rows) != 12 or {row["case_id"] for row in rows} != expected_ids
    ):
        raise ValueError("historical case coverage differs")
    for row in rows:
        case = by_id[row["case_id"]]
        if (
            row["expected_decision"] != case["expected_decision"]
            or row["expected_cited_node_ids"] != case["expected_cited_node_ids"]
        ):
            raise ValueError("recorded case label differs")
        current_grade = grade(row["raw_output"], case, schema)
        if current_grade != row["grade"]:
            raise ValueError(f"recorded grade differs: {row['case_id']}")
    metrics = {
        "total": len(rows),
        "plan_contract_valid": sum(row["grade"]["contract_valid"] for row in rows),
        "navigation_correct": sum(
            row["grade"]["case_pass"] for row in rows if row["expected_decision"] == "NAVIGATE"
        ),
        "navigation_total": sum(row["expected_decision"] == "NAVIGATE" for row in rows),
        "abstention_correct": sum(
            row["grade"]["case_pass"] for row in rows if row["expected_decision"] == "ABSTAIN"
        ),
        "abstention_total": sum(row["expected_decision"] == "ABSTAIN" for row in rows),
    }
    if metrics != receipt["metrics"]:
        raise ValueError("recorded aggregate metrics differ")
    return {
        "status": "RECORDED_EVIDENCE_CONSISTENT",
        "verification_scope": "RECORDED_BYTES_AND_GRADES_ONLY",
        "historical_source_base_commit": receipt["source_base_commit"],
        "historical_metrics": metrics,
        "historical_result": "FAILURE",
        "historical_receipt_sha256": RECORDED_SHA256["evidence/receipt.json"],
        "archived_runner_sha256": RECORDED_SHA256["evidence/frozen-runner.py.txt"],
        "archived_runner_executed": False,
        "model_inference": "NOT_RUN",
        "independent_inference_replay": "NOT_RUN",
        "promotion_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-recorded", action="store_true", help="check saved bytes and grades only")
    parser.parse_args()
    print(json.dumps(verify_recorded(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
