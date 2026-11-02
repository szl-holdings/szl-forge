#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate an SFT chat JSONL dataset before any training run consumes it.

Fail-closed: any malformed record, missing messages, unknown role, or empty
content makes the whole dataset INVALID and exits 1. Duplicate conversations,
duplicate prompts, and prompt overlap with a declared held-out/reference file
also fail closed. A dataset that merely exists is not a dataset that trains -
this is the guard that was missing when a file picker once handed a package
recipe to the trainer.

Stdlib only. Safe to run anywhere: no torch, no GPU, no network.

  python tools/validate_sft_dataset.py szl_dataset.jsonl --min-examples 8
  python tools/validate_sft_dataset.py train.jsonl --disjoint-from eval.jsonl --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA = "szl.forge-sft-dataset-report/v2"
FINGERPRINT_SCHEMA = "sha256-canonical-chat-input-v1"
ALLOWED_ROLES = ("system", "user", "assistant")
MAX_REPORTED_ERRORS = 25


def _normalize_content(value: str) -> str:
    """Normalize transport-only differences without changing internal text."""
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", value).strip()


def _fingerprint(messages: Sequence[Dict[str, str]], *, inputs_only: bool) -> str:
    selected = [
        {"role": message["role"], "content": _normalize_content(message["content"])}
        for message in messages
        if not inputs_only or message["role"] != "assistant"
    ]
    canonical = json.dumps(
        selected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _read_dataset(path: str, *, require_assistant: bool = True) -> Dict[str, Any]:
    """Read and structurally validate one dataset without exposing its text."""
    errors: List[str] = []
    roles_seen = set()

    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        return {
            "errors": [f"unreadable dataset: {exc}"],
            "raw": None,
            "rows": [],
            "roles_seen": roles_seen,
        }

    try:
        text = raw.decode("utf-8", errors="strict") if raw else ""
    except UnicodeDecodeError as exc:
        return {
            "errors": [f"dataset is not valid UTF-8 ({exc})"],
            "raw": raw,
            "rows": [],
            "roles_seen": roles_seen,
        }

    rows: List[Tuple[int, List[Dict[str, str]]]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: invalid JSON ({exc})")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {lineno}: record is not a JSON object")
            continue
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            errors.append(f"line {lineno}: missing or empty 'messages' list")
            continue
        record_roles = []
        record_ok = True
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                errors.append(f"line {lineno}: message {index} is not an object")
                record_ok = False
                continue
            role = message.get("role")
            content = message.get("content")
            if role not in ALLOWED_ROLES:
                errors.append(
                    f"line {lineno}: message {index} role {role!r} "
                    f"not in {list(ALLOWED_ROLES)}"
                )
                record_ok = False
            if not isinstance(content, str) or not content.strip():
                errors.append(
                    f"line {lineno}: message {index} has empty content"
                )
                record_ok = False
            record_roles.append(role)
        if record_ok:
            missing_user = "user" not in record_roles
            missing_assistant = require_assistant and "assistant" not in record_roles
            if missing_user or missing_assistant:
                requirement = (
                    "at least one user and one assistant message"
                    if require_assistant else "at least one user message"
                )
                errors.append(f"line {lineno}: record needs {requirement}")
                continue
            rows.append((lineno, messages))
            roles_seen.update(r for r in record_roles if r)

    return {
        "errors": errors,
        "raw": raw,
        "rows": rows,
        "roles_seen": roles_seen,
    }


def _deduplicate(rows: Sequence[Tuple[int, List[Dict[str, str]]]]) -> Dict[str, Any]:
    """Reject repeated complete conversations and repeated model inputs."""
    seen_conversations: Dict[str, int] = {}
    seen_inputs: Dict[str, int] = {}
    accepted: List[Tuple[int, str]] = []
    errors: List[str] = []
    duplicate_conversations = 0
    duplicate_inputs = 0

    for lineno, messages in rows:
        conversation = _fingerprint(messages, inputs_only=False)
        model_input = _fingerprint(messages, inputs_only=True)
        prior_conversation = seen_conversations.get(conversation)
        prior_input = seen_inputs.get(model_input)
        if prior_conversation is not None:
            duplicate_conversations += 1
            duplicate_inputs += 1
            errors.append(
                f"line {lineno}: duplicate conversation content "
                f"(same canonical content as line {prior_conversation})"
            )
            continue
        if prior_input is not None:
            duplicate_inputs += 1
            errors.append(
                f"line {lineno}: duplicate model input with a different response "
                f"(same canonical input as line {prior_input})"
            )
            continue
        seen_conversations[conversation] = lineno
        seen_inputs[model_input] = lineno
        accepted.append((lineno, model_input))

    return {
        "accepted": accepted,
        "errors": errors,
        "duplicate_conversations": duplicate_conversations,
        "duplicate_inputs": duplicate_inputs,
        "input_lines": seen_inputs,
    }


def validate_dataset(
    path: str,
    min_examples: int = 1,
    disjoint_from: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Validate one JSONL chat dataset and optional held-out references."""
    parsed = _read_dataset(path)
    errors = list(parsed["errors"])
    deduped = _deduplicate(parsed["rows"])
    errors.extend(deduped["errors"])
    records = len(deduped["accepted"])

    if type(min_examples) is not int or min_examples < 1:
        errors.append("min_examples must be a positive integer")
    elif records < min_examples:
        errors.append(
            f"only {records} valid unique records; --min-examples requires "
            f"{min_examples}"
        )

    cross_dataset_overlaps = 0
    reference_reports: List[Dict[str, Any]] = []
    primary_inputs = {fingerprint: lineno for lineno, fingerprint in deduped["accepted"]}
    for reference_path in disjoint_from or ():
        # Held-out inputs may intentionally omit assistant labels. The overlap
        # contract only needs the exact system/user input surface.
        reference = _read_dataset(reference_path, require_assistant=False)
        reference_deduped = _deduplicate(reference["rows"])
        reference_errors = [*reference["errors"], *reference_deduped["errors"]]
        for error in reference_errors:
            errors.append(f"reference {reference_path}: {error}")
        overlaps = sorted(set(primary_inputs).intersection(reference_deduped["input_lines"]))
        cross_dataset_overlaps += len(overlaps)
        for fingerprint in overlaps:
            errors.append(
                f"line {primary_inputs[fingerprint]}: model input overlaps reference "
                f"{reference_path} line {reference_deduped['input_lines'][fingerprint]}"
            )
        reference_raw = reference["raw"]
        reference_reports.append({
            "path": reference_path,
            "status": "VALID" if not reference_errors else "INVALID",
            "records": len(reference_deduped["accepted"]),
            "dataset_sha256": (
                "sha256:" + hashlib.sha256(reference_raw).hexdigest()
                if reference_raw is not None else None
            ),
            "overlapping_inputs": len(overlaps),
        })

    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "fingerprint_schema": FINGERPRINT_SCHEMA,
        "path": path,
        "status": "VALID" if not errors else "INVALID",
        "records": records,
        "roles_seen": sorted(parsed["roles_seen"]),
        "min_examples_required": min_examples,
        "duplicate_conversations": deduped["duplicate_conversations"],
        "duplicate_inputs": deduped["duplicate_inputs"],
        "cross_dataset_overlaps": cross_dataset_overlaps,
        "reference_datasets": reference_reports,
        "errors": errors[:MAX_REPORTED_ERRORS],
    }
    if parsed["raw"] is not None:
        report["dataset_sha256"] = (
            "sha256:" + hashlib.sha256(parsed["raw"]).hexdigest()
        )
    if len(errors) > MAX_REPORTED_ERRORS:
        report["errors_truncated"] = len(errors) - MAX_REPORTED_ERRORS
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="path to the JSONL chat dataset")
    parser.add_argument("--min-examples", type=int, default=1)
    parser.add_argument(
        "--disjoint-from",
        action="append",
        default=[],
        metavar="REFERENCE_JSONL",
        help=(
            "fail if a canonical system/user input overlaps this held-out or "
            "reference dataset; repeat for multiple references"
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit report JSON")
    args = parser.parse_args(argv)

    report = validate_dataset(
        args.dataset,
        min_examples=args.min_examples,
        disjoint_from=args.disjoint_from,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"dataset: {report['path']}")
        print(f"status: {report['status']}")
        print(f"valid records: {report['records']}")
        if "dataset_sha256" in report:
            print(f"dataset_sha256: {report['dataset_sha256']}")
        print(f"duplicate conversations: {report['duplicate_conversations']}")
        print(f"duplicate inputs: {report['duplicate_inputs']}")
        print(f"cross-dataset overlaps: {report['cross_dataset_overlaps']}")
        for error in report["errors"]:
            print(f"error: {error}")
    return 0 if report["status"] == "VALID" else 1


if __name__ == "__main__":
    sys.exit(main())
