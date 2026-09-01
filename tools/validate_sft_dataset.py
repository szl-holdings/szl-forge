#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate an SFT chat JSONL dataset before any training run consumes it.

Fail-closed: any malformed record, missing messages, unknown role, or empty
content makes the whole dataset INVALID and exits 1. A dataset that merely
exists is not a dataset that trains — this is the guard that was missing
when a file picker once handed a package recipe to the trainer.

Stdlib only. Safe to run anywhere: no torch, no GPU, no network.

  python tools/validate_sft_dataset.py szl_dataset.jsonl --min-examples 8
  python tools/validate_sft_dataset.py szl_dataset.jsonl --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Any, Dict, List, Optional

SCHEMA = "szl.forge-sft-dataset-report/v1"
ALLOWED_ROLES = ("system", "user", "assistant")
MAX_REPORTED_ERRORS = 25


def validate_dataset(path: str, min_examples: int = 1) -> Dict[str, Any]:
    """Validate one JSONL chat dataset; return a structured report."""
    errors: List[str] = []
    records = 0
    roles_seen = set()
    sha256 = hashlib.sha256()

    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        return {
            "schema": SCHEMA,
            "path": path,
            "status": "INVALID",
            "errors": [f"unreadable dataset: {exc}"],
            "records": 0,
        }
    sha256.update(raw)

    text = raw.decode("utf-8", errors="strict") if raw else ""
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
            if "user" not in record_roles or "assistant" not in record_roles:
                errors.append(
                    f"line {lineno}: record needs at least one user and one "
                    "assistant message"
                )
                continue
            records += 1
            roles_seen.update(r for r in record_roles if r)

    if records < min_examples:
        errors.append(
            f"only {records} valid records; --min-examples requires "
            f"{min_examples}"
        )

    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "path": path,
        "status": "VALID" if not errors else "INVALID",
        "records": records,
        "roles_seen": sorted(roles_seen),
        "min_examples_required": min_examples,
        "dataset_sha256": "sha256:" + sha256.hexdigest(),
        "errors": errors[:MAX_REPORTED_ERRORS],
    }
    if len(errors) > MAX_REPORTED_ERRORS:
        report["errors_truncated"] = len(errors) - MAX_REPORTED_ERRORS
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="path to the JSONL chat dataset")
    parser.add_argument("--min-examples", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="emit report JSON")
    args = parser.parse_args(argv)

    report = validate_dataset(args.dataset, min_examples=args.min_examples)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"dataset: {report['path']}")
        print(f"status: {report['status']}")
        print(f"valid records: {report['records']}")
        if "dataset_sha256" in report:
            print(f"dataset_sha256: {report['dataset_sha256']}")
        for error in report["errors"]:
            print(f"error: {error}")
    return 0 if report["status"] == "VALID" else 1


if __name__ == "__main__":
    sys.exit(main())
