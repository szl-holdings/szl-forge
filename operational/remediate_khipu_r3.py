#!/usr/bin/env python3
"""Khipu-R3 abstention remediation curriculum builder.

Measured failure on `SZLHOLDINGS/khipu-r3`: grounding 4/5 but abstain 0/6 —
the model retrieves when it must refuse navigation. The fix is curriculum,
not vibes: many near-miss negatives whose surface form looks navigable but
whose provenance is absent, contradicted, or out of scope.

Output: `operational/out/khipu-r3-remediation.jsonl` (SFT rows,
CANDIDATE_REQUIRES_REVIEW). The khipu lanes remain the trainers of record.

Row kinds:
  - abstain-no-source:   query has no bound corpus evidence -> ABSTAIN
  - abstain-near-miss:   evidence exists but does not answer the query -> ABSTAIN
  - navigate-grounded:   evidence directly answers -> navigate + cite source
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
TARGET = OUT / "khipu-r3-remediation.jsonl"

SYSTEM = (
    "You are Khipu, the SZL retrieval navigator. Navigate only when bound "
    "evidence directly answers the query, and cite the source handle. "
    "Otherwise ABSTAIN and name the missing evidence. Near-miss passages "
    "that merely share vocabulary do not count. Proposal-only, always."
)

ABSTAIN_NO_SOURCE = [
    "What is the measured throughput of the szl-router on H100 hardware?",
    "Which receipt proves the energy meter ran on the GPU mesh this week?",
    "What nDCG@10 did chakana score on the frozen slice?",
    "Who approved the promotion of WILLAY to operational status?",
    "What does the sealed ReceiptAgent v3 test set contain?",
    "Show the signed attestation for last night's training job.",
]

NEAR_MISS_PAIRS = [
    (
        "Does the doctrine permit autonomous publication?",
        "Doctrine v11: control before action; evidence after. Governance gates require receipts for governed actions.",
        "ABSTAIN: passage covers governed actions generally; no statement on publication autonomy.",
    ),
    (
        "What is chakana's embedding dimension?",
        "Chakana is the SZLHOLDINGS bridge-embedding lane (NINA, FORGE-class) with an admitted base of Qwen/Qwen3-Embedding-0.6B.",
        "ABSTAIN: passage admits a base model but states no measured output dimension for the SZL cut.",
    ),
    (
        "Is KHIPU-R2 cleared for autonomous use?",
        "KHIPU-R2: adapter AVAILABLE; abstain MEASURED 3/6; grounding 5/5; plan 11/11; publication_eligible false.",
        "ABSTAIN: receipt shows abstain 3/6 — below any autonomy gate; no clearance statement exists.",
    ),
]

GROUNDED_PAIRS = [
    (
        "What must an SZL model do when evidence is unknown?",
        "Doctrine: claims are MEASURED, REPORTED, MODELED, or UNKNOWN. UNKNOWN is stated, never filled in.",
        "NAVIGATE: state UNKNOWN and name the missing evidence. Source: szl doctrine (doctrine-v11).",
    ),
    (
        "Who may fire a training job on owner hardware?",
        "szl-gpu-bridge accepts only signed job specs and returns receipts; unsigned work is refused.",
        "NAVIGATE: only a signed job spec crosses the bridge, and a receipt returns. Source: szl-gpu-bridge.",
    ),
]


def row(kind: str, user: str, assistant: str, passage: str | None) -> dict:
    content = user if passage is None else f"Evidence:\n{passage}\n\nQuery: {user}"
    return {
        "id": hashlib.sha256(content.encode()).hexdigest()[:16],
        "kind": kind,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
            {"role": "assistant", "content": assistant},
        ],
        "admission_status": "CANDIDATE_REQUIRES_REVIEW",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in ABSTAIN_NO_SOURCE:
        rows.append(row("abstain-no-source", q,
                        "ABSTAIN: no bound estate evidence answers this query.", None))
    for q, passage, verdict in NEAR_MISS_PAIRS:
        rows.append(row("abstain-near-miss", q, verdict, passage))
    for q, passage, verdict in GROUNDED_PAIRS:
        rows.append(row("navigate-grounded", q, verdict, passage))

    with TARGET.open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")

    receipt = {
        "kind": "szl-ops-khipu-r3-remediation",
        "rows": len(rows),
        "targets": ["abstain 0/6 on SZLHOLDINGS/khipu-r3"],
        "admission_status": "CANDIDATE_REQUIRES_REVIEW",
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "khipu-r3-remediation-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[remediate-khipu-r3] wrote {TARGET} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
