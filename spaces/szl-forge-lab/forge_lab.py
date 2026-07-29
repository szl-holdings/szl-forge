"""Read-only evidence services for the SZL Forge Lab Space.

This module is deliberately standard-library only so the evidence contract can
be tested independently of Gradio. It reads packaged snapshots and never
downloads data, trains a model, promotes an artifact, or mutates external state.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent

ASSETS = {
    "manifest": "run_manifest.json",
    "evaluation": "eval_receipt.json",
    "formulas": "thesis_formula_index.json",
    "sources": "science_source_ledger.json",
    "curriculum": "curriculum.json",
    "training": "training_summary.json",
}

LOCAL_INPUT_MAP = {
    "thesis_formula_index.json": ASSETS["formulas"],
    "science/source_ledger.json": ASSETS["sources"],
    "science/curriculum.json": ASSETS["curriculum"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(name: str, fallback: Any) -> Any:
    """Load a packaged JSON asset, returning a typed diagnostic on failure."""
    path = ROOT / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if isinstance(fallback, dict):
            result = dict(fallback)
            result["_load_error"] = f"{name}: {type(exc).__name__}"
            return result
        return fallback


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_receipt(value: dict[str, Any]) -> bool:
    expected = value.get("receipt_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        return False
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    actual = hashlib.sha256(canonical(unsigned)).hexdigest()
    return actual == expected


def _manifest_body() -> dict[str, Any]:
    manifest = load_json(ASSETS["manifest"], {})
    payload = manifest.get("payload")
    return payload if isinstance(payload, dict) else manifest


def _formula_entries() -> list[dict[str, Any]]:
    value = load_json(ASSETS["formulas"], {})
    entries = value.get("entries", []) if isinstance(value, dict) else value
    return [item for item in entries if isinstance(item, dict)]


def _source_entries() -> list[dict[str, Any]]:
    value = load_json(ASSETS["sources"], {})
    return [item for item in value.get("sources", []) if isinstance(item, dict)]


def _curriculum() -> dict[str, Any]:
    return load_json(ASSETS["curriculum"], {})


def _training_summary() -> dict[str, Any]:
    return load_json(ASSETS["training"], {})


def get_integrity() -> dict[str, Any]:
    """Compare packaged artifacts with hashes declared by the run manifest."""
    body = _manifest_body()
    declared = body.get("inputs", {})
    if not declared and isinstance(body.get("artifacts"), list):
        declared = {
            str(item.get("path", "")): item
            for item in body["artifacts"]
            if isinstance(item, dict)
        }

    checks: list[dict[str, Any]] = []
    for raw_name, metadata in declared.items():
        normalized = str(raw_name).replace("\\", "/")
        metadata = metadata if isinstance(metadata, dict) else {}
        expected = metadata.get("sha256")
        local_name = next(
            (value for suffix, value in LOCAL_INPUT_MAP.items() if normalized.endswith(suffix)),
            None,
        )
        if local_name is None:
            checks.append(
                {
                    "declared_input": normalized,
                    "packaged_asset": None,
                    "state": "NOT_PACKAGED",
                    "expected_sha256": expected,
                    "actual_sha256": None,
                }
            )
            continue

        path = ROOT / local_name
        if not path.is_file():
            state = "MISSING"
            actual = None
        else:
            actual = sha256_file(path)
            state = "VERIFIED" if expected and actual == expected else "DRIFT"
        checks.append(
            {
                "declared_input": normalized,
                "packaged_asset": local_name,
                "state": state,
                "expected_sha256": expected,
                "actual_sha256": actual,
            }
        )

    states = {item["state"] for item in checks}
    if states.intersection({"DRIFT", "MISSING"}):
        overall = "DRIFT"
    elif "NOT_PACKAGED" in states:
        overall = "VERIFIED_WITH_EXCLUSIONS"
    elif checks:
        overall = "VERIFIED"
    else:
        overall = "NO_DECLARED_INPUTS"

    manifest = load_json(ASSETS["manifest"], {})
    manifest_receipt = verify_receipt(manifest) if "receipt_sha256" in manifest else None
    return {
        "schema": "szl.forge.lab-integrity/v1",
        "evidence_state": "MEASURED_LOCAL_HASHES",
        "overall_state": overall,
        "manifest_receipt_valid": manifest_receipt,
        "manifest_signature_state": (
            "HASH_VERIFIED" if manifest_receipt else "UNSIGNED_CONFIGURATION_SNAPSHOT"
        ),
        "checks": checks,
        "limits": [
            "A matching hash confirms byte identity, not scientific correctness.",
            "NOT_PACKAGED inputs are intentionally unavailable through this showcase.",
        ],
    }


def get_evaluation() -> dict[str, Any]:
    receipt = load_json(ASSETS["evaluation"], {})
    payload = receipt.get("payload", {})
    results = payload.get("results", [])
    summary = payload.get("summary", {})
    return {
        "schema": "szl.forge.lab-evaluation/v1",
        "evidence_state": "MEASURED_GOVERNED_STACK",
        "receipt_valid": verify_receipt(receipt),
        "receipt_sha256": receipt.get("receipt_sha256"),
        "created_at": receipt.get("created_at"),
        "scope": "Model-bound governed runtime responses scored against deterministic policy contracts.",
        "not_evidence_of": [
            "trained model quality",
            "raw-model contract compliance",
            "frontier benchmark performance",
            "generalization",
            "mathematical proof",
        ],
        "summary": summary,
        "thresholds": payload.get("thresholds", {}),
        "gate_passed": payload.get("gate_passed") is True,
        "results": [item for item in results if isinstance(item, dict)],
    }


def get_receipt() -> dict[str, Any]:
    receipt = load_json(ASSETS["evaluation"], {})
    return {
        "schema": "szl.forge.lab-receipt-envelope/v1",
        "verification": {
            "receipt_valid": verify_receipt(receipt),
            "algorithm": "SHA-256 over canonical JSON excluding receipt_sha256",
        },
        "receipt_sha256": receipt.get("receipt_sha256"),
        "receipt_json": json.dumps(receipt, sort_keys=True, ensure_ascii=False),
        "transport_note": "Receipt is serialized to prevent client libraries from coercing artifact path fields into file downloads.",
    }


def get_status() -> dict[str, Any]:
    manifest = _manifest_body()
    training = _training_summary()
    evaluation = get_evaluation()
    integrity = get_integrity()
    claims = manifest.get("claims", {})
    training_status = training.get("status", manifest.get("status", "UNKNOWN"))
    return {
        "schema": "szl.forge.lab-status/v1",
        "observed_at": utc_now(),
        "surface": "SZL Forge Lab",
        "transport_state": "REACHABLE",
        "evidence_state": "SNAPSHOT",
        "interface_mode": "READ_ONLY",
        "training": {
            "status": training_status,
            "base_model": training.get("base_model", manifest.get("base_model", "UNKNOWN")),
            "base_revision": training.get("base_revision", manifest.get("base_revision", "UNKNOWN")),
            "weights_measured": training.get("status", "").startswith("COMPLETED"),
            "weights_published": training.get("merged_model", {}).get("published") is True,
            "run": training.get("run", {}),
        },
        "evaluation": {
            "scope": "GOVERNED_STACK_POLICY_CONTRACTS",
            "passed": evaluation.get("summary", {}).get("passed"),
            "total": evaluation.get("summary", {}).get("total"),
            "pass_rate": evaluation.get("summary", {}).get("pass_rate"),
            "receipt_valid": evaluation["receipt_valid"],
            "model_benchmark_claim": False,
            "raw_model_contract": training.get("evaluation", {}).get("raw_model_contract", {}),
            "governed_stack_contract": training.get("evaluation", {}).get("governed_stack_contract", {}),
        },
        "formula_registry": {
            "entries": len(_formula_entries()),
            "proofs_rechecked_by_this_space": False,
        },
        "science_policy": {
            "sources": len(_source_entries()),
            "curriculum_stages": len(_curriculum().get("stages", [])),
            "default_ingestion_policy": load_json(ASSETS["sources"], {})
            .get("policy", {})
            .get("default", "UNKNOWN"),
            "raw_human_neurodata_training": "EXCLUDED",
        },
        "integrity": {
            "state": integrity["overall_state"],
            "manifest_signature_state": integrity["manifest_signature_state"],
        },
        "promotion": {
            "state": training.get("promotion", {}).get("release_decision", "BLOCK"),
            "reason": "Weights are local; independent evaluator, model owner, and security reviewer approvals remain required.",
            "external_mutation_performed": False,
        },
        "limits": [
            "RUNNING or REACHABLE means transport availability only.",
            "This Space performs no training, deployment, promotion, or external mutation.",
            "Registry status is displayed as metadata and is not independently re-proven here.",
        ],
    }


def _bounded_limit(value: Any, maximum: int = 100) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 25
    return max(1, min(parsed, maximum))


def get_formulas(query: str = "", status: str = "ALL", limit: int = 25) -> dict[str, Any]:
    query_norm = str(query or "").strip().casefold()
    status_norm = str(status or "ALL").strip().upper()
    matches = []
    for entry in _formula_entries():
        haystack = " ".join(
            str(entry.get(key, ""))
            for key in ("id", "title", "description", "lean_theorem", "runtime_gate")
        ).casefold()
        if query_norm and query_norm not in haystack:
            continue
        if status_norm != "ALL" and str(entry.get("status", "UNKNOWN")).upper() != status_norm:
            continue
        matches.append(entry)
    capped = _bounded_limit(limit)
    items = [
        {
            "id": item.get("id"),
            "title": item.get("title"),
            "description": item.get("description"),
            "declared_status": item.get("status", "UNKNOWN"),
            "lean_theorem": item.get("lean_theorem"),
            "runtime_gate": item.get("runtime_gate"),
        }
        for item in matches[:capped]
    ]
    return {
        "schema": "szl.forge.lab-formulas/v1",
        "evidence_state": "SNAPSHOT_INDEX_METADATA",
        "verification_scope": "DECLARED_STATUS_NOT_RECHECKED",
        "query": query_norm,
        "status_filter": status_norm,
        "total_matches": len(matches),
        "returned": len(items),
        "items": items,
    }


def get_formula(formula_id: str) -> dict[str, Any]:
    wanted = str(formula_id or "").strip().casefold()
    item = next(
        (entry for entry in _formula_entries() if str(entry.get("id", "")).casefold() == wanted),
        None,
    )
    return {
        "schema": "szl.forge.lab-formula/v1",
        "evidence_state": "SNAPSHOT_INDEX_METADATA",
        "found": item is not None,
        "verification_scope": "DECLARED_STATUS_NOT_RECHECKED",
        "formula": item,
        "limit": "Independent checker execution is outside this Space.",
    }


def get_sources(domain: str = "ALL", decision: str = "ALL", limit: int = 50) -> dict[str, Any]:
    domain_norm = str(domain or "ALL").strip().casefold()
    decision_norm = str(decision or "ALL").strip().upper()
    matches = []
    for source in _source_entries():
        if domain_norm != "all" and str(source.get("domain", "")).casefold() != domain_norm:
            continue
        if decision_norm != "ALL" and str(source.get("decision", "")).upper() != decision_norm:
            continue
        matches.append(source)
    capped = _bounded_limit(limit)
    items = [
        {
            "source_id": item.get("source_id"),
            "name": item.get("name"),
            "domain": item.get("domain"),
            "decision": item.get("decision"),
            "license_expression": item.get("license_expression"),
            "risk": item.get("risk"),
            "canonical_url": item.get("canonical_url"),
        }
        for item in matches[:capped]
    ]
    return {
        "schema": "szl.forge.lab-sources/v1",
        "evidence_state": "SNAPSHOT_POLICY_METADATA",
        "policy_default": load_json(ASSETS["sources"], {}).get("policy", {}).get("default", "UNKNOWN"),
        "domain_filter": domain_norm,
        "decision_filter": decision_norm,
        "total_matches": len(matches),
        "returned": len(items),
        "items": items,
        "limit": "Policy metadata is not legal advice or live license re-verification.",
    }


def get_source(source_id: str) -> dict[str, Any]:
    wanted = str(source_id or "").strip().casefold()
    item = next(
        (source for source in _source_entries() if str(source.get("source_id", "")).casefold() == wanted),
        None,
    )
    return {
        "schema": "szl.forge.lab-source/v1",
        "evidence_state": "SNAPSHOT_POLICY_METADATA",
        "found": item is not None,
        "source": item,
        "limit": "The recorded decision must be rechecked at artifact acquisition time.",
    }


def get_curriculum() -> dict[str, Any]:
    curriculum = _curriculum()
    contract = curriculum.get("global_contract", {})
    stages = [item for item in curriculum.get("stages", []) if isinstance(item, dict)]
    return {
        "schema": "szl.forge.lab-curriculum/v1",
        "evidence_state": "BLUEPRINT_SNAPSHOT",
        "status": curriculum.get("status", "UNKNOWN"),
        "purpose": curriculum.get("purpose"),
        "truth_states": contract.get("truth_states", []),
        "hard_gates": contract.get("hard_gates", []),
        "stage_count": len(stages),
        "stages": [
            {
                "stage_id": item.get("stage_id"),
                "domains": item.get("domains", []),
                "goal": item.get("goal"),
                "task_count": len(item.get("tasks", [])),
                "evaluation_count": len(item.get("evaluations", [])),
                "exit_gate": item.get("exit_gate"),
            }
            for item in stages
        ],
        "training_recipe": curriculum.get("training_recipe", {}),
        "anti_patterns": curriculum.get("anti_patterns", []),
        "limit": "This is a curriculum blueprint; it is not evidence that training occurred.",
    }


def get_curriculum_stage(stage_id: str) -> dict[str, Any]:
    wanted = str(stage_id or "").strip().casefold()
    item = next(
        (
            stage
            for stage in _curriculum().get("stages", [])
            if str(stage.get("stage_id", "")).casefold() == wanted
        ),
        None,
    )
    return {
        "schema": "szl.forge.lab-curriculum-stage/v1",
        "evidence_state": "BLUEPRINT_SNAPSHOT",
        "found": item is not None,
        "stage": item,
        "limit": "Stage definitions are plans and gates, not completed training evidence.",
    }


def integrity_rows() -> list[list[Any]]:
    return [
        [
            item["declared_input"],
            item["packaged_asset"] or "Not packaged",
            item["state"],
            item["expected_sha256"] or "—",
            item["actual_sha256"] or "—",
        ]
        for item in get_integrity()["checks"]
    ]


def evaluation_rows() -> list[list[Any]]:
    return [
        [
            item.get("id"),
            item.get("category"),
            "PASS" if item.get("passed") else "FAIL",
            item.get("response_sha256", "—"),
        ]
        for item in get_evaluation()["results"]
    ]


def formula_rows(query: str = "", status: str = "ALL", limit: int = 25) -> list[list[Any]]:
    return [
        [item["id"], item["title"], item["declared_status"], item.get("lean_theorem") or "—"]
        for item in get_formulas(query, status, limit)["items"]
    ]


def source_rows(domain: str = "ALL", decision: str = "ALL", limit: int = 50) -> list[list[Any]]:
    return [
        [
            item["source_id"],
            item["name"],
            item["domain"],
            item["decision"],
            item["license_expression"],
            item["risk"],
        ]
        for item in get_sources(domain, decision, limit)["items"]
    ]


def curriculum_rows() -> list[list[Any]]:
    return [
        [
            item["stage_id"],
            ", ".join(item["domains"]),
            item["goal"],
            item["task_count"],
            item["evaluation_count"],
        ]
        for item in get_curriculum()["stages"]
    ]


def inventory() -> dict[str, list[str]]:
    formulas = _formula_entries()
    sources = _source_entries()
    curriculum = _curriculum()
    return {
        "formula_statuses": sorted(
            {str(item.get("status", "UNKNOWN")).upper() for item in formulas}
        ),
        "formula_ids": [str(item.get("id")) for item in formulas if item.get("id")],
        "source_domains": sorted({str(item.get("domain")) for item in sources}),
        "source_decisions": sorted({str(item.get("decision")) for item in sources}),
        "source_ids": [str(item.get("source_id")) for item in sources if item.get("source_id")],
        "stage_ids": [
            str(item.get("stage_id"))
            for item in curriculum.get("stages", [])
            if item.get("stage_id")
        ],
    }


def metrics() -> dict[str, Any]:
    status = get_status()
    evaluation = status["evaluation"]
    return {
        "training": status["training"]["status"],
        "formulas": status["formula_registry"]["entries"],
        "sources": status["science_policy"]["sources"],
        "stages": status["science_policy"]["curriculum_stages"],
        "eval_cases": f"{evaluation['passed']}/{evaluation['total']}",
        "integrity": status["integrity"]["state"],
    }


def decision_counts() -> dict[str, int]:
    return dict(Counter(str(item.get("decision", "UNKNOWN")) for item in _source_entries()))
