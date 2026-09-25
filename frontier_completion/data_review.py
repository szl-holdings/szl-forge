"""Bounded, inert UltraData pilot inspection; format is not training admission.

Uses existing evidence primitives. No network, model, schema-ref resolution,
embedded-code evaluation, reward execution, or data publication. The two schema
adapters are not a new dataset inventory: canonical intake remains szl-frontier.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from .core import (AUTHORITY_ORDER, DENIED, MAX_JSON, EvidenceError, canonical,
                   digest, parse_json, receipt, require, sha256, text, utc_now)

MAX_ROWS = 256
MAX_ROW_BYTES = 1024 * 1024
DATASETS = {
    "ultradata-agent": "openbmb/UltraData-SFT-Agent-2609",
    "ultradata-rl": "openbmb/UltraData-RL-2609",
}
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]{0,127}\Z")


def _common(row: dict[str, Any]) -> None:
    for key in ("uuid", "source", "domain"):
        text(row.get(key), 512)


def _body(value: Any, *, nullable: bool = False) -> None:
    # Multimodal content blocks need their own reviewed adapter. Do not coerce
    # them to strings, silently discard them, or imply complete corpus coverage.
    require((nullable and value is None) or type(value) is str,
            "unsupported_content_shape")


def _sft(row: dict[str, Any]) -> tuple[int, bool]:
    tools, messages = row.get("tools"), row.get("messages")
    require(type(tools) is list and len(tools) <= 128, "tool_definitions_shape")
    require(type(messages) is list and 1 <= len(messages) <= 1024, "messages_shape")
    names: set[str] = set()
    for tool in tools:
        require(type(tool) is dict and tool.get("type") == "function", "tool_definition_shape")
        function = tool.get("function")
        require(type(function) is dict, "tool_definition_shape")
        name = function.get("name")
        require(type(name) is str and _NAME.fullmatch(name) is not None and name not in names,
                "duplicate_or_invalid_tool_name")
        parameters = function.get("parameters")
        require(type(parameters) is dict and parameters.get("type") == "object",
                "tool_parameters_shape")
        # STRUCTURE ONLY: neither JSON Schema validity nor argument-schema
        # satisfaction is established. In particular $ref is never dereferenced.
        names.add(name)
    pending: set[str] = set()
    seen: set[str] = set()
    calls_count, user_seen = 0, False
    for message in messages:
        require(type(message) is dict, "message_shape")
        role = message.get("role")
        require(type(role) is str and role in {"system", "user", "assistant", "tool"}, "message_role")
        calls = message.get("tool_calls", [])
        require(type(calls) is list and len(calls) <= 128, "tool_calls_shape")
        require(not calls or role == "assistant", "tool_call_wrong_role")
        require("content" in message, "message_content_missing")
        _body(message["content"], nullable=bool(calls))
        if role == "tool":
            identity = message.get("tool_call_id")
            require(type(identity) is str and identity in pending, "unpaired_tool_response")
            pending.remove(identity)
        else:
            require(not pending, "unresolved_tool_calls_before_next_turn")
        user_seen |= role == "user"
        for call in calls:
            require(type(call) is dict and call.get("type") == "function", "tool_call_shape")
            identity, function = call.get("id"), call.get("function")
            text(identity, 256)
            require(identity not in seen, "duplicate_tool_call_id")
            require(type(function) is dict, "tool_call_function_shape")
            name, arguments = function.get("name"), function.get("arguments")
            require(type(name) is str and name in names, "undeclared_tool")
            require(type(arguments) is str and len(arguments) <= 65536, "tool_arguments_shape")
            # Decoding JSON is not execution. Unknown tools stay historical data.
            parse_json(arguments.encode("utf-8"))
            seen.add(identity)
            pending.add(identity)
            calls_count += 1
    require(not pending, "unresolved_tool_calls")
    require(user_seen and messages[-1].get("role") == "assistant", "incomplete_trajectory")
    return calls_count, bool(set(row) - {"uuid", "source", "domain", "messages", "tools"})


def _rl(row: dict[str, Any]) -> tuple[int, bool]:
    _body(row.get("query"))
    require(bool(row["query"].strip()), "empty_query")
    truth = row.get("ground_truth")
    if row["domain"] == "Code":
        require(type(truth) is dict, "code_ground_truth_shape")
        inputs, outputs = truth.get("inputs"), truth.get("outputs")
        require(type(inputs) is list and type(outputs) is list and
                1 <= len(inputs) == len(outputs) <= 4096, "code_test_pairs_shape")
        call_type = truth.get("call_type")
        # The first adapter supports the documented stdin/stdout shape only.
        # Other harness protocols require review rather than guessed execution.
        require(call_type == "std" and truth.get("fn_name") is None, "unsupported_code_call_type")
        require(all(type(v) is str for v in inputs + outputs), "code_test_value_shape")
    else:
        require(type(truth) is str and bool(truth.strip()), "unsupported_ground_truth_shape")
    return 0, bool(set(row) - {"uuid", "source", "domain", "query", "ground_truth"})


def review_pilot(raw: bytes, *, dataset: str, dataset_revision: str,
                 source_revision: str, expected_input_sha256: str) -> dict[str, Any]:
    """Inspect a bounded supplied pilot. Returned rows NEVER contain input text.

    Revisions are operator-declared identities, not provider observations.
    The expected input hash is an integrity assertion, not proof of provenance.
    Exact-content duplicates ignore uuid only; semantic/cross-harness dedup and
    held-out decontamination are explicitly not performed.
    """
    require(type(dataset) is str and dataset in DATASETS, "dataset_not_supported")
    digest(dataset_revision, 40)
    digest(source_revision, 40)
    digest(expected_input_sha256)
    require(type(raw) is bytes and 0 < len(raw) <= MAX_JSON, "pilot_byte_bound")
    actual = hashlib.sha256(raw).hexdigest()
    require(actual == expected_input_sha256, "pilot_hash_mismatch")
    lines = raw.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    require(1 <= len(lines) <= MAX_ROWS, "pilot_row_bound")
    results, uuids, content_ids = [], set(), set()
    reviewer = _sft if dataset == "ultradata-agent" else _rl
    for index, line in enumerate(lines, 1):
        result: dict[str, Any] = {
            "line": index, "rowSha256": hashlib.sha256(line).hexdigest(),
            "state": "REVIEW_REQUIRED", "reasonCode": None,
            "toolCalls": None, "extraFieldsPresent": None,
        }
        try:
            require(0 < len(line) <= MAX_ROW_BYTES, "pilot_row_byte_bound")
            row = parse_json(line)
            _common(row)
            identity = row["uuid"]
            content_id = sha256({k: v for k, v in row.items() if k != "uuid"})
            require(identity not in uuids, "duplicate_uuid")
            require(content_id not in content_ids, "duplicate_record_content")
            # Remember even structurally unsupported rows to retain duplicate findings.
            uuids.add(identity)
            content_ids.add(content_id)
            calls, extras = reviewer(row)
            result.update(state="FORMAT_CHECKED", toolCalls=calls, extraFieldsPresent=extras)
        except (EvidenceError, UnicodeError, OverflowError, RecursionError):
            # Never echo arbitrary exceptions or trajectory contents into reports.
            # A bounded second pass is unnecessary; source-defined codes are enough.
            result["reasonCode"] = "unsupported_or_invalid_record"
        results.append(result)
    checked = sum(row["state"] == "FORMAT_CHECKED" for row in results)
    return receipt({
        "schema": "szl.frontier.ultradata-pilot-review/v1",
        "generatedAt": utc_now(), "sourceRevisionDeclared": source_revision,
        "datasetRepoId": DATASETS[dataset], "datasetRevisionDeclared": dataset_revision,
        "inputSha256": actual, "inputHashMatched": True,
        "upstreamBytesVerified": False, "sourceExecutionVerified": False,
        "rows": results, "counts": {"total": len(results), "formatChecked": checked,
                                     "reviewRequired": len(results) - checked},
        "state": "FORMAT_CHECKED_NOT_ADMITTED" if checked == len(results) else "REVIEW_REQUIRED",
        "authorityOrder": list(AUTHORITY_ORDER), "authority": dict(DENIED),
        "productionDisposition": "HOLD", "signatureState": "UNSIGNED",
        "reviews": {"rights": "NOT_PERFORMED", "privacy": "NOT_PERFORMED",
                    "decontamination": "NOT_PERFORMED", "semanticDedup": "NOT_PERFORMED",
                    "toolArgumentSchema": "NOT_PERFORMED", "outcomeQuality": "NOT_PERFORMED"},
        "execution": "NONE", "rewardVerified": False, "successLabelsInferred": False,
        "bounds": ["Static pilot format inspection, not whole-corpus validation.",
                   "No environment replay, code evaluation or tool execution.",
                   "Frozen tool names are not permissions or available runtime tools.",
                   "Hashes are content commitments, not anonymization or rights clearance.",
                   "Retain original private pilot and failure traces under the owner's data policy."],
    })


def emit_review(raw: bytes, *, dataset: str, dataset_revision: str,
                source_revision: str, expected_input_sha256: str) -> tuple[bytes, int]:
    """Pure CLI adapter; exit zero means format checks only, never admission."""
    report = review_pilot(raw, dataset=dataset, dataset_revision=dataset_revision,
                          source_revision=source_revision, expected_input_sha256=expected_input_sha256)
    return canonical(report) + b"\n", 0 if report["counts"]["reviewRequired"] == 0 else 2
