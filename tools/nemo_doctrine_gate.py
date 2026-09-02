#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Doctrine-content gate: run SZL training/eval datasets through the
szl-nemo doctrine kernel (rule_check R1-R5) before they shape a model.

The sibling tool validate_sft_dataset.py proves a dataset is WELL-FORMED;
this gate proves its content is DOCTRINE-CONFORMANT. A well-formed dataset
that teaches the model to fabricate is still a bad dataset.

Personas (R3 polarity is persona-bound; choosing one is a doctrine act,
so --persona is required and never defaults):

  wrapper    The SZL-Nemo posture: SZL did NOT fine-tune the weights.
             rule_check R1-R5 applies verbatim; a fine-tune question
             answered without the not-fine-tuned disclosure is a violation.
  finetuned  The SZL-1 / ReceiptAgent posture: SZL DID fine-tune. Universal
             honesty rules (R1 fabrication labels, R2 honest unknown,
             R4 Lambda-not-theorem, R5 trust ceiling) are enforced
             identically. R3 is mirrored: a fine-tune question must be
             answered by AFFIRMING the SZL fine-tune; denying or evading
             it is the violation (R3_persona_finetune_affirm). The
             wrapper-polarity R3 is waived because "not fine-tuned" would
             be a false claim for these lanes.

Every checked record yields a szl.nemo.decision.v1 Decision and the whole
run is bound into a hash-chained szl.nemo.receipt.v1 ledger — deterministic
(no clocks, no RNG), so the same dataset always produces the same tip.

Exit codes: 0 VALID, 1 VIOLATIONS/INVALID, 2 NOT_READY (kernel missing,
unreadable input, bad usage). NOT_READY is never a pass.

  pip install "szl-nemo @ git+https://github.com/szl-holdings/szl-nemo.git@<sha>"
  python tools/nemo_doctrine_gate.py szl_dataset.jsonl --persona finetuned
  python tools/nemo_doctrine_gate.py receiptagent/train.jsonl --persona finetuned \
      --write-receipt receipts/doctrine
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "szl.forge-doctrine-gate/v1"
KERNEL_REPO = "https://github.com/szl-holdings/szl-nemo"
MAX_REPORTED_VIOLATIONS = 25

try:
    from szl_nemo.engine import input_hash
    from szl_nemo.receipt import chain as receipt_chain
    from szl_nemo.rules import FINETUNE_ASK_RE, NOT_FT_RE, rule_check
    from szl_nemo.schema import ALLOW, BLOCK, RULE_VERSION, Decision

    _IMPORT_ERROR: Optional[ImportError] = None
except ImportError as exc:  # pragma: no cover - exercised via NOT_READY test
    _IMPORT_ERROR = exc

# Fine-tuned-persona R3 mirror: the answer must plainly affirm that SZL
# fine-tuned/trained the model. (Denial is caught by NOT_FT_RE.)
AFFIRM_FT_RE = re.compile(
    r"\b(szl (?:fine[- ]?tuned|trained)|we (?:fine[- ]?tuned|trained)|"
    r"fine[- ]?tuned (?:by|on) szl|trained (?:by szl|on szl))\b",
    re.I,
)
R3_PERSONA_RULE = "R3_persona_finetune_affirm"

_PERSONA_REASONS = {
    R3_PERSONA_RULE: (
        "fine-tune provenance question answered without affirming the SZL "
        "fine-tune this persona actually is"
    ),
}
_WRAPPER_REASON = (
    "fine-tune provenance question answered without disclosing that SZL "
    "did NOT fine-tune the weights"
)
_UNIVERSAL_REASONS = {
    "R1_no_fabrication_label": (
        "numeric/benchmark claim lacks an honesty label "
        "(MEASURED/REPORTED/MODELED/HEURISTIC/UNKNOWN/UNAVAILABLE)"
    ),
    "R2_honest_unknown": (
        "benchmark/quality question answered with an invented number "
        "instead of an honest UNKNOWN"
    ),
    "R3_not_finetuned": _WRAPPER_REASON,
    "R4_lambda_not_theorem": (
        "calls Lambda a theorem/proven/certified; Lambda is Conjecture 1 "
        "(open, advisory)"
    ),
    "R5_trust_ceiling": (
        "claims 100%/perfect/fully-trusted; the trust ceiling is 0.97"
    ),
}


def extract_pair(record: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Last user message + last assistant message of a chat record."""
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    prompt = answer = None
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role, content = message.get("role"), message.get("content")
        if not isinstance(content, str):
            continue
        if role == "user" and prompt is None:
            prompt = content
        elif role == "assistant" and answer is None:
            answer = content
        if prompt is not None and answer is not None:
            return prompt, answer
    return None


def check_pair(prompt: str, answer: str, persona: str) -> Tuple[List[str], List[str]]:
    """Return (violated_rule_ids, reasons) under the given persona."""
    _, violated = rule_check(prompt, answer)
    violated = list(violated)
    if persona == "finetuned":
        if "R3_not_finetuned" in violated:
            # Wrapper-polarity R3 would punish the honest answer here.
            violated.remove("R3_not_finetuned")
        if FINETUNE_ASK_RE.search(prompt) and (
            not AFFIRM_FT_RE.search(answer) or NOT_FT_RE.search(answer)
        ):
            violated.append(R3_PERSONA_RULE)
    reasons = [
        (_UNIVERSAL_REASONS.get(rule) or _PERSONA_REASONS.get(rule) or rule)
        for rule in violated
    ]
    return violated, reasons


def gate_dataset(path: str, persona: str) -> Dict[str, Any]:
    """Gate one JSONL chat dataset; return a structured, receipted report."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        return {
            "schema": SCHEMA,
            "path": path,
            "persona": persona,
            "status": "INVALID",
            "errors": [f"unreadable dataset: {exc}"],
        }

    dataset_sha = hashlib.sha256(raw).hexdigest()
    decisions: List[Decision] = []
    violations: List[Dict[str, Any]] = []
    records = 0
    skipped = 0

    for lineno, line in enumerate(raw.decode("utf-8", errors="strict").splitlines(), 1):
        if not line.strip():
            continue
        records += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            violations.append({"line": lineno, "record": None,
                               "violated_rules": ["RECORD_unparseable"],
                               "reasons": [f"invalid JSON: {exc}"]})
            continue
        pair = extract_pair(record)
        if pair is None:
            skipped += 1
            violations.append({"line": lineno, "record": record.get("id"),
                               "violated_rules": ["RECORD_no_chat_pair"],
                               "reasons": ["no user+assistant pair to check"]})
            continue
        prompt, answer = pair
        violated, reasons = check_pair(prompt, answer, persona)
        decisions.append(
            Decision(
                decision=ALLOW if not violated else BLOCK,
                violated_rules=tuple(violated),
                reasons=tuple(reasons),
                rule_version=f"{RULE_VERSION} persona/{persona}",
                input_hash=input_hash(prompt, answer),
            )
        )
        if violated:
            violations.append({
                "line": lineno,
                "record": record.get("id"),
                "violated_rules": violated,
                "reasons": reasons,
            })

    receipts = receipt_chain(decisions) if decisions else []
    tip = receipts[-1]["receipt_sha256"] if receipts else None

    counts: Dict[str, int] = {}
    for violation in violations:
        for rule in violation["violated_rules"]:
            counts[rule] = counts.get(rule, 0) + 1

    return {
        "schema": SCHEMA,
        "path": path,
        "persona": persona,
        "dataset_sha256": "sha256:" + dataset_sha,
        "kernel": {"repo": KERNEL_REPO, "rule_version": RULE_VERSION},
        "status": "VALID" if not violations else "VIOLATIONS",
        "records": records,
        "checked": len(decisions),
        "skipped_no_chat_pair": skipped,
        "violation_counts": counts,
        "violations": violations[:MAX_REPORTED_VIOLATIONS],
        "receipt_schema": "szl.nemo.receipt.v1",
        "receipt_chain_tip": tip,
        "receipt_chain": receipts,
        "honesty": (
            "Decisions are deterministic outputs of the szl-nemo kernel; "
            "receipts are UNSIGNED_HONEST (integrity and ordering checkable, "
            "no signing identity claimed). REGEX ground truth is conservative "
            "and can false-BLOCK negated phrasing; see szl-nemo "
            "test_vectors/README.md."
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Doctrine-content gate over SFT datasets (szl-nemo kernel)."
    )
    parser.add_argument("datasets", nargs="+", help="JSONL chat dataset paths")
    parser.add_argument(
        "--persona",
        required=True,
        choices=("wrapper", "finetuned"),
        help="doctrine posture of the model being trained (R3 polarity)",
    )
    parser.add_argument("--json", action="store_true", help="print full JSON reports")
    parser.add_argument(
        "--write-receipt",
        default=None,
        metavar="DIR",
        help="write each dataset's receipt chain to DIR/<dataset-sha12>.json",
    )
    args = parser.parse_args(argv)

    if _IMPORT_ERROR is not None:
        print(
            f"NOT_READY: szl-nemo doctrine kernel is not importable "
            f"({_IMPORT_ERROR}). Install it pinned, e.g.:\n"
            f"  pip install \"szl-nemo @ git+{KERNEL_REPO}.git@<commit-sha>\"",
            file=sys.stderr,
        )
        return 2

    reports = []
    worst = 0
    for path in args.datasets:
        report = gate_dataset(path, args.persona)
        reports.append(report)
        if report["status"] == "INVALID":
            worst = 1
            print(f"INVALID   {path}: {report['errors']}", file=sys.stderr)
            continue
        if report["status"] != "VALID":
            worst = 1
        tip = report["receipt_chain_tip"] or "n/a"
        print(
            f"{report['status']:<10} {path} "
            f"(records={report['records']} checked={report['checked']} "
            f"violations={sum(report['violation_counts'].values())} tip={tip[:23]}...)"
        )
        for violation in report["violations"]:
            print(
                f"  line {violation['line']}: "
                f"{','.join(violation['violated_rules'])}",
                file=sys.stderr,
            )
        if args.write_receipt:
            _write_receipt(args.write_receipt, report)

    if args.json:
        for report in reports:
            print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return worst


def _write_receipt(directory: str, report: Dict[str, Any]) -> None:
    import os

    os.makedirs(directory, exist_ok=True)
    name = report["dataset_sha256"].replace("sha256:", "")[:12] + ".doctrine-receipts.json"
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report["receipt_chain"], handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print(f"  receipt chain -> {path} ({len(report['receipt_chain'])} receipts)")


if __name__ == "__main__":
    sys.exit(main())
