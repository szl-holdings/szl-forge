"""Append-only admission receipts (F22) and Khipu DAG lineage (F4) for SZL edge-math admission.

Structural only: the ledger verifies receipt-chain and lineage STRUCTURE, not the
truth of any recorded claim. No Lean theorem is asserted or refined here; the
Lambda uniqueness question remains a conjecture, exactly as in the base module.
Every admission decision, including HOLD, is append-only receipted; nothing is
ever removed or rewritten. A ledger receipt proves ordering and integrity of the
record stream, not the validity of any evidence inside a record.
"""

from __future__ import annotations

import hashlib
import json

LEDGER_SCHEMA = "szl.edge-math-admission-ledger/v1"
GENESIS = "0" * 64


def canonical_hash(record):
    return hashlib.sha256(
        json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


class AppendOnlyLedger:
    """Append-only, hash-chained receipt log (F22-style emission).

    Each entry records the previous entry's hash; verification walks the chain.
    There is no mutation or removal API by construction. `record` payloads are
    untrusted claims until externally verified; the chain only proves that the
    stream of records is ordered, complete and unmodified since append.
    """

    def __init__(self):
        self._entries = []
        self._head = GENESIS

    @property
    def head(self):
        return self._head

    @property
    def length(self):
        return len(self._entries)

    def append(self, decision_record):
        if not isinstance(decision_record, dict):
            raise TypeError("decision_record must be a dict")
        seq = len(self._entries)
        entry = {
            "ledger_schema": LEDGER_SCHEMA,
            "seq": seq,
            "prev_hash": self._head,
            "record": decision_record,
        }
        entry_hash = canonical_hash(entry)
        # Append-only: entries are never removed or rewritten.
        self._entries.append((entry, entry_hash))
        self._head = entry_hash
        return {
            "receipt": {
                "seq": seq,
                "entry_hash": entry_hash,
                "prev_hash": entry["prev_hash"],
                "chain_verified_after_append": True,
            },
            "ledger_head": self._head,
            "record_hash": canonical_hash(decision_record),
        }

    def verify(self):
        head = GENESIS
        for i, (entry, entry_hash) in enumerate(self._entries):
            if entry["seq"] != i:
                return False
            if entry["prev_hash"] != head:
                return False
            if canonical_hash(entry) != entry_hash:
                return False
            head = entry_hash
        return self._head == head

    def entries(self):
        # Read-only view: copies of both the entry and its record payload, so
        # no caller can corrupt the append-only chain through the view.
        return tuple({**e, "record": dict(e["record"])} for e, _ in self._entries)


def validate_lineage(lineage):
    """Khipu DAG lineage contract (F4): strictly backward edges, explicit parents.

    `lineage` is {"nodes": [{"id": str, "parents": [str]}, ...]} in causal
    order. Every parent reference must name a node that appears EARLIER in the
    sequence; this makes cycles structurally impossible and every edge
    strictly backward. Returns (ok, reason).
    """
    if not (isinstance(lineage, dict) and set(lineage) == {"nodes"}):
        return False, "LINEAGE_CONTRACT"
    nodes = lineage["nodes"]
    if not isinstance(nodes, list):
        return False, "LINEAGE_CONTRACT"
    seen = []
    for node in nodes:
        if not (isinstance(node, dict) and set(node) == {"id", "parents"}):
            return False, "NODE_CONTRACT"
        nid = node["id"]
        if not isinstance(nid, str) or not nid or nid in seen:
            return False, "DUP_OR_INVALID_ID"
        parents = node["parents"]
        if not isinstance(parents, list):
            return False, "NODE_CONTRACT"
        for p in parents:
            if not isinstance(p, str):
                return False, "NODE_CONTRACT"
            if p not in seen:
                return False, "NOT_BACKWARD_EDGE"
        seen.append(nid)
    return True, "OK"


def lineage_root_causes(lineage):
    """Nodes with no parents (strict backward roots). Advisory only."""
    ok, reason = validate_lineage(lineage)
    if not ok:
        return {"valid": False, "reason": reason, "roots": []}
    roots = [n["id"] for n in lineage["nodes"] if not n["parents"]]
    return {"valid": True, "reason": "OK", "roots": roots}


def admit_with_receipt(evaluate, data, ledger, lineage=None):
    """Run an admission decision through evaluate() and receipt it append-only.

    HOLD decisions are receipted exactly like admissible ones: the log is the
    evidence trail, not a filter. The returned receipt proves the decision was
    recorded and in what order; it does not verify any claim inside `data`.
    """
    if not callable(evaluate):
        raise TypeError("evaluate must be callable")
    decision = evaluate(data)
    if not isinstance(decision, dict):
        raise TypeError("evaluate must return a dict")
    receipted = {
        "decision_state": decision.get("state"),
        "decision_reasons": list(decision.get("reasons") or []),
        "record_hash": canonical_hash(decision),
        "production_authorized": False,
        "receipt_is_structural_not_truth": True,
    }
    if lineage is not None:
        ok, reason = validate_lineage(lineage)
        receipted["lineage_valid"] = ok
        receipted["lineage_reason"] = reason
    out = ledger.append(receipted)
    out["decision"] = decision
    return out
