#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Field-leader bridge. Offline. Not a collector, publisher, or authorizer.

Takes current evaluation fashion (evidence layers, pass@k vs pass^k,
evaluator independence, pacing scorecards, chip-as-predicate) and binds
it to SZL doctrine. Nothing here can emit production_authorization=true.
HTTP 200 is OBSERVED. AGI is false. Unknown is HOLD.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

MAX_JSON = 32 * 1024
MAX_CLAIM = 4096
MAX_ROWS = 64
SCHEMA = "szl.field-leader-bridge/v1"

EVIDENCE_LAYERS = ("STATE", "ARTIFACT", "BEHAVIOR", "NARRATIVE")
LAYER_RANK = {name: index for index, name in enumerate(EVIDENCE_LAYERS)}

CHIP = {
    "LIVE": "forbidden",
    "ADMIT_PREDICATES_ONLY": "admit",
    "BLOCKED": "blocked",
    "HOLD": "hold",
    "REJECTED": "blocked",
    "EVALUATION": "eval",
    "OBSERVED": "info",
    "UNAVAILABLE": "hold",
    "PARTIAL": "hold",
}

PACING_QUESTIONS = (
    "Is any measured subset operating fully autonomously?",
    "Does oversight keep pace with claimed capability?",
    "Can a third party re-read the same receipt without the collector?",
    "Are pass@k, pass^k, and mean@k reported as distinct numbers?",
    "Is HTTP 200 labelled OBSERVED rather than LIVE?",
)

FORBIDDEN = (
    (re.compile(r"\bagi\b", re.I), "FORBIDDEN_PROMOTION_AGI"),
    (re.compile(r"all[_\s-]*done", re.I), "FORBIDDEN_PROMOTION_ALL_DONE"),
    (re.compile(r"fully operational", re.I), "FORBIDDEN_PROMOTION_OPERATIONAL"),
    (re.compile(r"all green", re.I), "FORBIDDEN_PROMOTION_ALL_GREEN"),
    (re.compile(r"production_authori[sz]ed\s*[:=]\s*true", re.I), "FORBIDDEN_PROMOTION_AUTHORITY"),
    (re.compile(r"\blive\b.{0,24}\bproduction\b", re.I), "FORBIDDEN_PROMOTION_LIVE"),
)

SOURCES = (
    {"id": "anthropic-pacing-2026", "take": "Report only metrics a developer can measure today. Autonomy subsets stay explicit."},
    {"id": "traccia-evidence-layers-2026", "take": "State > artifact > behavior > self-report. Grade environment first."},
    {"id": "tau-bench-pass-power-k", "take": "pass@k is capability. pass^k is reliability. Do not collapse."},
    {"id": "evaluator-bench-2026", "take": "Evaluate the evaluator. Independence and publication rights are first-class."},
    {"id": "observable-ai-sre-2026", "take": "Telemetry layers: context, policy, outcome. Connected by one trace, never by vibes."},
    {"id": "hf-kernels-signed-2026", "take": "Kernels are signed Hub artifacts. Untrusted publishers stay EVALUATION."},
    {"id": "linear-stripe-vercel-fashion-2026", "take": "One accent. Chip color is a predicate, not decoration."},
    {"id": "typesafe-system-one", "take": "Typed judgments over generated prose. Code owns the workflow."},
)


class BridgeError(ValueError):
    """Fixed diagnostic. Never a private path or raw dump."""


def need(condition: bool, code: str) -> None:
    if not condition:
        raise BridgeError(code)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def dream_scan(claim: Any) -> list[str]:
    if claim is None:
        return []
    need(type(claim) is str and len(claim) <= MAX_CLAIM, "CLAIM_TYPE")
    return [code for pattern, code in FORBIDDEN if pattern.search(claim)]


def layer_of(value: Any) -> str:
    need(type(value) is str and value in LAYER_RANK, "LAYER")
    return value


def refuse_outrank(claim_layer: Any, evidence_layer: Any) -> dict[str, Any]:
    claim = layer_of(claim_layer)
    evidence = layer_of(evidence_layer)
    if LAYER_RANK[claim] < LAYER_RANK[evidence]:
        raise BridgeError("CLAIM_OUTRANKS_EVIDENCE")
    return {
        "claim_layer": claim,
        "evidence_layer": evidence,
        "admitted": True,
        "meaning": "CLAIM_MAY_NOT_OUTRANK_ITS_EVIDENCE",
    }


def refuse_altk_collapse(payload: Any) -> dict[str, Any]:
    need(type(payload) is dict, "ALTK_OBJECT")
    mean = payload.get("mean_at_k")
    power = payload.get("pass_power_k")
    any_run = payload.get("pass_at_k")
    as_one = payload.get("as_one_score")
    for value in (mean, power, any_run):
        need(value is None or type(value) in (int, float), "ALTK_METRIC")
        if type(value) is float:
            need(math.isfinite(value), "ALTK_METRIC")
    need(as_one is False, "ALTK_COLLAPSE")
    gap = None
    if type(any_run) in (int, float) and type(power) in (int, float):
        gap = float(any_run) - float(power)
    return {
        "mean_at_k": mean,
        "pass_power_k": power,
        "pass_at_k": any_run,
        "consistency_gap": gap,
        "collapsed": False,
        "meaning": "MEAN_AT_K_PASS_POWER_K_PASS_AT_K_ARE_DISTINCT",
    }


def chip_for(fold: Any, evidence_layer: Any) -> dict[str, Any]:
    need(type(fold) is str and fold in CHIP, "FOLD")
    layer = layer_of(evidence_layer)
    kind = CHIP[fold]
    if fold == "ADMIT_PREDICATES_ONLY" and layer != "STATE":
        kind = "eval"
    if kind == "forbidden":
        raise BridgeError("FORBIDDEN_PROMOTION_CHIP")
    return {
        "fold": fold,
        "evidence_layer": layer,
        "chip": kind,
        "live": False,
        "production_authorized": False,
        "meaning": "CHIP_IS_A_PREDICATE_NOT_DECORATION",
    }


def evaluator_independence(payload: Any) -> dict[str, Any]:
    need(type(payload) is dict, "EVAL_OBJECT")
    collector = payload.get("collector_id")
    reader = payload.get("second_reader_id")
    network = payload.get("second_reader_network")
    can_auth = payload.get("second_reader_can_authorize")
    need(type(collector) is str and type(reader) is str, "EVAL_IDS")
    need(network is False, "EVAL_NETWORK")
    need(can_auth is False, "EVAL_AUTHORIZE")
    same = collector == reader
    return {
        "independent": not same,
        "floor": "CONDITIONAL" if same else "CLEAR",
        "collector_id": collector,
        "second_reader_id": reader,
        "meaning": "EVALUATOR_OF_EVALUATORS_NO_SELF_QUALIFY",
    }


def telemetry_layers(payload: Any) -> dict[str, Any]:
    need(type(payload) is dict, "TELEM_OBJECT")
    needed = {"context", "policy", "outcome"}
    need(set(payload) >= needed, "TELEM_KEYS")
    trace = payload.get("trace_id")
    need(type(trace) is str and 0 < len(trace) <= 128, "TELEM_TRACE")
    for key in needed:
        need(type(payload[key]) is str and 0 < len(payload[key]) <= 256, "TELEM_FIELD")
    return {
        "trace_id": trace,
        "layers": ["context", "policy", "outcome"],
        "connected": True,
        "meaning": "THREE_LAYER_TELEMETRY_ONE_TRACE",
    }


def pacing_card(payload: Any) -> dict[str, Any]:
    need(type(payload) is dict, "PACING_OBJECT")
    autonomy = payload.get("autonomy_subset_measured")
    authorized = payload.get("production_authorized")
    hub = payload.get("hub_module_present")
    sovereign = payload.get("sovereign_reachable")
    preflight_ok = payload.get("preflight_not_degraded")
    need(autonomy is False, "PACING_AUTONOMY")
    need(authorized is False, "PACING_AUTHORITY")
    answers = [
        {"question": PACING_QUESTIONS[0], "answer": "NO", "evidence": "agi_claim=false"},
        {"question": PACING_QUESTIONS[1], "answer": "HOLD",
         "evidence": "capability claims outrun closed HOLDs"},
        {"question": PACING_QUESTIONS[2], "answer": "EVALUATION",
         "evidence": "szl-forge#348 second-reader draft"},
        {"question": PACING_QUESTIONS[3], "answer": "YES",
         "evidence": "ALTK refuse-collapse"},
        {"question": PACING_QUESTIONS[4], "answer": "YES",
         "evidence": "HTTP 200 is OBSERVED"},
    ]
    holds = []
    if hub is not True:
        holds.append("hub_module_present")
    if sovereign is not True:
        holds.append("sovereign_reachable")
    if preflight_ok is not True:
        holds.append("preflight_not_degraded")
    return {
        "reportable_today": answers,
        "holds": holds,
        "estate_disposition": "PARTIAL",
        "production_authorized": False,
        "agi_claim": False,
        "meaning": "WHAT_ANY_DEVELOPER_CAN_REPORT_TODAY",
    }


def classify(payload: Any) -> dict[str, Any]:
    need(type(payload) is dict, "PAYLOAD_OBJECT")
    if payload.get("production_authorization") is True:
        raise BridgeError("FORBIDDEN_PROMOTION_AUTHORITY")
    if payload.get("agi_claim") is True:
        raise BridgeError("FORBIDDEN_PROMOTION_AGI")
    dreams = dream_scan(payload.get("claim"))
    if dreams:
        raise BridgeError(dreams[0])
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "state": "EVALUATION",
        "sources": [row["id"] for row in SOURCES],
        "production_authorization": False,
        "runtime_verified": False,
        "agi_claim": False,
        "meaning": "FIELD_LEADER_BRIDGE_NOT_A_RELEASE",
    }
    if "rank" in payload:
        out["rank"] = refuse_outrank(payload["rank"].get("claim_layer"),
                                     payload["rank"].get("evidence_layer"))
    if "altk" in payload:
        out["altk"] = refuse_altk_collapse(payload["altk"])
    if "chip" in payload:
        out["chip"] = chip_for(payload["chip"].get("fold"), payload["chip"].get("evidence_layer"))
    if "evaluator" in payload:
        out["evaluator"] = evaluator_independence(payload["evaluator"])
    if "telemetry" in payload:
        out["telemetry"] = telemetry_layers(payload["telemetry"])
    if "pacing" in payload:
        out["pacing"] = pacing_card(payload["pacing"])
        out["state"] = "PARTIAL"
    return out


def default_bridge() -> dict[str, Any]:
    """This-observer fixture. Not a live qualification of the estate."""
    return classify({
        "claim": "Estate remains PARTIAL. HTTP 200 is OBSERVED.",
        "rank": {"claim_layer": "BEHAVIOR", "evidence_layer": "BEHAVIOR"},
        "altk": {"mean_at_k": 0.75, "pass_power_k": 0.5, "pass_at_k": 1.0, "as_one_score": False},
        "chip": {"fold": "BLOCKED", "evidence_layer": "STATE"},
        "evaluator": {
            "collector_id": "observe_public_estate_files.py",
            "second_reader_id": "szl_payload_contract.py",
            "second_reader_network": False,
            "second_reader_can_authorize": False,
        },
        "telemetry": {
            "trace_id": "console-local",
            "context": "ops-console probe",
            "policy": "doctrine-v11-fail-closed",
            "outcome": "PARTIAL",
        },
        "pacing": {
            "autonomy_subset_measured": False,
            "production_authorized": False,
            "hub_module_present": False,
            "sovereign_reachable": False,
            "preflight_not_degraded": False,
        },
    })
