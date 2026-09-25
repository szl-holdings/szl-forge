#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline predicate-AND and dream-gate. Not a publisher, collector, or authorizer.

Independent predicates combine by AND only. Unknown is HOLD, never fail.
Incomparable populations cannot be summed. ALTK metrics cannot collapse.
Forbidden promotion language is classified, not executed.
This module cannot emit production_authorization=true.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from typing import Any

MAX_JSON = 32 * 1024
MAX_CLAIM = 4096
MAX_PREDICATES = 64

RELEASE_GATES = (
    "source_admitted",
    "package_built",
    "configuration_present",
    "artifact_published",
    "immutable_bytes_verified",
    "runtime_loaded",
    "task_evaluated",
    "ui_exercised",
    "persistence_restart_verified",
    "product_current",
    "proof_current",
    "release_authorized",
)

# This-observer fixture. Not a live probe. Sovereign path 404 this request.
CURRENT_ESTATE_PREDICATES = (
    {"id": "lambda_admit", "value": False, "population": "payload-lambda"},
    {"id": "sovereign_reachable", "value": False, "population": "a11oy-sovereign"},
    {"id": "preflight_not_degraded", "value": False, "population": "a11oy-preflight"},
    {"id": "hub_module_present", "value": False, "population": "huggingface-hub-readback"},
    {"id": "census_content_review", "value": False, "population": "public-census-bytes"},
    {"id": "operator_explicit_approval", "value": False, "population": "owner-approval"},
)

DREAMS = (
    (re.compile(r"\bagi\b", re.I), "FORBIDDEN_PROMOTION_AGI"),
    (re.compile(r"all[_\s-]*done", re.I), "FORBIDDEN_PROMOTION_ALL_DONE"),
    (re.compile(r"fully operational", re.I), "FORBIDDEN_PROMOTION_OPERATIONAL"),
    (re.compile(r"all green", re.I), "FORBIDDEN_PROMOTION_ALL_GREEN"),
    (re.compile(r"production_authori[sz]ed\s*[:=]\s*true", re.I), "FORBIDDEN_PROMOTION_AUTHORITY"),
    (re.compile(r"runtime_verified\s*[:=]\s*true", re.I), "FORBIDDEN_PROMOTION_RUNTIME"),
    (re.compile(r"make it all", re.I), "FORBIDDEN_PROMOTION_SWEEP"),
    (re.compile(r"http\s*200.{0,48}\blive\b", re.I), "FORBIDDEN_PROMOTION_HTTP_LIVE"),
    (re.compile(r"running.{0,40}(runtime[_\s-]?verif|qualified)", re.I), "FORBIDDEN_PROMOTION_RUNNING"),
)


class GateError(ValueError):
    """A fixed diagnostic, never raw input or a private path."""


def need(condition: bool, code: str) -> None:
    if not condition:
        raise GateError(code)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def strict_json(raw: bytes) -> Any:
    need(type(raw) is bytes and 0 < len(raw) <= MAX_JSON, "JSON_SIZE")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            need(key not in out, "DUPLICATE_JSON_KEY")
            out[key] = value
        return out

    def finite(text: str) -> float:
        value = float(text)
        need(math.isfinite(value), "NONFINITE_JSON")
        return value

    def reject(_: str) -> None:
        raise GateError("NONFINITE_JSON")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                          parse_float=finite, parse_constant=reject)
    except GateError:
        raise
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        raise GateError("INVALID_JSON") from None


def dream_scan(claim: Any) -> list[str]:
    if claim is None:
        return []
    need(type(claim) is str and len(claim) <= MAX_CLAIM, "CLAIM_TYPE")
    need(not any(ord(c) < 32 and c not in "\n\t" for c in claim), "CLAIM_CONTROL")
    return [code for pattern, code in DREAMS if pattern.search(claim)]


def and_fold(predicates: Any) -> dict[str, Any]:
    need(type(predicates) is list and 0 < len(predicates) <= MAX_PREDICATES, "PREDICATE_BOUND")
    seen: set[str] = set()
    blocked: list[str] = []
    holds: list[str] = []
    admits: list[str] = []
    for row in predicates:
        need(type(row) is dict, "PREDICATE_OBJECT")
        ident = row.get("id")
        value = row.get("value")
        population = row.get("population")
        need(type(ident) is str and 0 < len(ident) <= 128 and ident not in seen, "PREDICATE_ID")
        need(type(population) is str and 0 < len(population) <= 128, "PREDICATE_POPULATION")
        need(value is None or type(value) is bool, "PREDICATE_VALUE")
        seen.add(ident)
        if value is False:
            blocked.append(ident)
        elif value is None:
            holds.append(ident)
        else:
            admits.append(ident)
    if blocked:
        fold = "BLOCKED"
    elif holds:
        fold = "HOLD"
    else:
        fold = "ADMIT_PREDICATES_ONLY"
    return {
        "fold": fold,
        "admitted": admits,
        "blocked": blocked,
        "holds": holds,
        "count": len(predicates),
        "meaning": "INDEPENDENT_AND_NOT_OR_NOT_AVERAGE_NOT_AUTHORIZATION",
    }


def sum_counts(rows: Any) -> dict[str, Any]:
    need(type(rows) is list and 1 <= len(rows) <= MAX_PREDICATES, "SUM_BOUND")
    populations: list[str] = []
    total = 0
    for row in rows:
        need(type(row) is dict, "SUM_OBJECT")
        population = row.get("population")
        count = row.get("count")
        need(type(population) is str and 0 < len(population) <= 128, "SUM_POPULATION")
        need(type(count) is int and 0 <= count <= 10**12, "SUM_COUNT")
        populations.append(population)
        total += count
    unique = set(populations)
    if len(unique) != 1:
        raise GateError("INCOMPARABLE_POPULATIONS")
    return {"population": populations[0], "count": total, "terms": len(rows)}


def refuse_collapse(payload: Any) -> None:
    need(type(payload) is dict, "COLLAPSE_OBJECT")
    mean = payload.get("mean_at_k")
    power = payload.get("pass_power_k")
    any_run = payload.get("pass_at_k_empirical")
    as_one = payload.get("as_one_score")
    for value in (mean, power, any_run):
        need(value is None or (type(value) is float or type(value) is int), "COLLAPSE_METRIC")
        if type(value) is float:
            need(math.isfinite(value), "COLLAPSE_METRIC")
    need(type(as_one) is bool, "COLLAPSE_FLAG")
    if as_one is True:
        raise GateError("DISTINCT_METRICS")


def sequence_check(gates: Any) -> dict[str, Any]:
    need(type(gates) is dict, "SEQUENCE_OBJECT")
    need(set(gates) == set(RELEASE_GATES), "SEQUENCE_KEYS")
    first_open = None
    skipped = []
    values = []
    for name in RELEASE_GATES:
        value = gates[name]
        need(value is None or type(value) is bool, "SEQUENCE_VALUE")
        values.append({"id": name, "value": value})
        if value is True and first_open is not None:
            skipped.append(name)
        if first_open is None and value is not True:
            first_open = name
    if skipped:
        raise GateError("SEQUENCE_SKIP")
    if first_open == "release_authorized" and gates["release_authorized"] is True:
        raise GateError("SEQUENCE_SKIP")
    if gates["release_authorized"] is True:
        raise GateError("FORBIDDEN_PROMOTION_RELEASE")
    return {
        "first_open": first_open,
        "gates": values,
        "meaning": "CANONICAL_SEQUENCE_NO_SKIP_NO_AUTHORITY",
    }


def classify(payload: Any) -> dict[str, Any]:
    need(type(payload) is dict, "PAYLOAD_OBJECT")
    if payload.get("production_authorization") is True:
        raise GateError("FORBIDDEN_PROMOTION_AUTHORITY")
    if payload.get("runtime_verified") is True:
        raise GateError("FORBIDDEN_PROMOTION_RUNTIME")
    if payload.get("agi_claim") is True:
        raise GateError("FORBIDDEN_PROMOTION_AGI")
    dreams = dream_scan(payload.get("claim"))
    if dreams:
        raise GateError(dreams[0])
    fold = None
    if "predicates" in payload:
        fold = and_fold(payload["predicates"])
    summed = None
    if "sum" in payload:
        summed = sum_counts(payload["sum"])
    if "collapse" in payload:
        refuse_collapse(payload["collapse"])
    sequence = None
    if "sequence" in payload:
        sequence = sequence_check(payload["sequence"])
    state = "EVALUATION"
    if fold is not None:
        state = fold["fold"]
    return {
        "schema": "szl.predicate-and/v1",
        "state": state,
        "predicates": fold,
        "sum": summed,
        "sequence": sequence,
        "dreams": [],
        "diagnostic": None,
        "production_authorization": False,
        "runtime_verified": False,
        "agi_claim": False,
        "meaning": "PREDICATE_AND_FAIL_CLOSED_NOT_A_RELEASE",
    }


def default_estate() -> dict[str, Any]:
    """This-observer fixture. Not a live product probe."""
    return classify({"predicates": [dict(row) for row in CURRENT_ESTATE_PREDICATES]})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("classify", "estate"))
    args = parser.parse_args(argv)
    try:
        if args.action == "estate":
            print(canonical(default_estate()).decode())
            return 2  # retained gaps: estate predicates are blocked
        raw = sys.stdin.buffer.read(MAX_JSON + 1)
        result = classify(strict_json(raw))
        print(canonical(result).decode())
        return 0 if result["state"] == "ADMIT_PREDICATES_ONLY" else 2
    except GateError as exc:
        print(canonical({
            "schema": "szl.predicate-and/v1",
            "state": "REJECTED",
            "diagnostic": str(exc),
            "production_authorization": False,
            "runtime_verified": False,
            "agi_claim": False,
        }).decode(), file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError, OverflowError):
        print('{"state":"REJECTED","diagnostic":"LOCAL_IO_OR_SHAPE_ERROR","production_authorization":false}',
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
