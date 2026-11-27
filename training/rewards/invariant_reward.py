#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""szl-invariants as an RLVR reward (TRL ``reward_funcs`` compatible).

Task shape: the prompt gives the facts of one or more runs; the model must
emit the ledger export (a JSON list of rows) for them. The reward replays the
eight falsifiable szl-invariants over the emitted rows.

Anti-gaming rules (measured on szl-invariants 0.1.4: an empty export is all
NO_DATA, and a single bare row already HOLDS four structural invariants):

* every completion must be bound to its task through the ``expected_rows``
  column: a list of field subsets, each of which must match a distinct emitted
  row exactly. Missing or unmatched expectations score 0.0;
* any VIOLATED invariant scores 0.0;
* NO_DATA / UNAVAILABLE are never coerced to a pass; they score as not held;
* otherwise reward = (# required invariants that HOLD) / (# required).

Parsing is strict: raw JSON or exactly one fenced ```json block. Malformed
output scores 0.0 rather than raising, because a bad generation is a normal
RL outcome. Malformed *batches* (shape, misaligned columns) raise before any
scoring, because those are trainer wiring bugs.

This module does not sign, publish, train, or certify a model.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

try:  # PyPI ``szl-invariants`` (GitHub szl-holdings/szl-invariants)
    import szl_invariants as _si
except ImportError:  # pragma: no cover - exercised only without the package
    _si = None

MAX_BATCH = 256
MAX_TEXT = 256 * 1024
MAX_ROWS = 256
DEFAULT_REQUIRED = (
    "receipt-chain-continuity",
    "served-run-has-model",
    "signed-columns-atomic",
    "loop-steps-positive",
)
_FENCE = re.compile(r"```[ \t]*(?:json)?[ \t]*\n(.*?)\n?```", re.S)


def _text(comp: Any) -> str:
    if type(comp) is str:
        text = comp
    else:
        if type(comp) in (list, tuple):
            if len(comp) != 1:
                raise ValueError("COMPLETION_SEQUENCE_AMBIGUOUS")
            comp = comp[0]
        if type(comp) is not dict or comp.get("role") != "assistant":
            raise ValueError("COMPLETION_SHAPE_INVALID")
        text = comp.get("content")
        if type(text) is not str:
            raise ValueError("COMPLETION_CONTENT_NOT_TEXT")
    if len(text) > MAX_TEXT or "\x00" in text:
        raise ValueError("COMPLETION_TEXT_INVALID_OR_OVERSIZED")
    return text


def _reject_constant(_name: str) -> Any:
    raise ValueError("NON_FINITE_JSON")


def parse_rows(text: str) -> Optional[List[Dict[str, Any]]]:
    """Return emitted ledger rows, or None when the output is not a clean export."""
    blocks = _FENCE.findall(text)
    if len(blocks) > 1:
        return None
    body = blocks[0] if blocks else text.strip()
    try:
        rows = json.loads(body, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return None
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        return None
    if not all(isinstance(r, dict) for r in rows):
        return None
    return rows


def _matches_expected(rows: List[Dict[str, Any]], expected: Any) -> bool:
    if not isinstance(expected, (list, tuple)) or not expected:
        return False
    used = set()
    for want in expected:
        if not isinstance(want, dict) or not want:
            return False
        hit = next((i for i, row in enumerate(rows)
                    if i not in used and all(k in row and row[k] == v for k, v in want.items())), None)
        if hit is None:
            return False
        used.add(hit)
    return True


class InvariantReward:
    """Callable reward: ``reward(completions, expected_rows=[...], **_)``."""

    __name__ = "szl_invariants_reward"

    def __init__(self, required: Sequence[str] = DEFAULT_REQUIRED, pubkey: Optional[str] = None,
                 kernel: Any = None) -> None:
        self.si = kernel or _si
        if self.si is None:
            raise RuntimeError("szl-invariants is not installed; refusing to score")
        known = set(self.si.INVARIANT_IDS)
        required = tuple(required)
        if not required or len(set(required)) != len(required) or not set(required) <= known:
            raise ValueError("required must be a nonempty, duplicate-free subset of INVARIANT_IDS")
        self.required = required
        self.pubkey = pubkey

    def score_one(self, text: str, expected: Any) -> Dict[str, Any]:
        rows = parse_rows(text)
        if rows is None:
            return {"reward": 0.0, "reason": "UNPARSEABLE"}
        if not _matches_expected(rows, expected):
            return {"reward": 0.0, "reason": "TASK_BINDING_FAILED"}
        report = self.si.run_invariants(rows, samples=None, pubkey=self.pubkey)
        status = {i["id"]: i["status"] for i in report["invariants"]}
        if any(s == "VIOLATED" for s in status.values()):
            return {"reward": 0.0, "reason": "VIOLATED", "status": status}
        held = sum(1 for k in self.required if status.get(k) == "HOLDS")
        return {"reward": held / len(self.required), "reason": "SCORED", "status": status}

    def __call__(self, completions, expected_rows=None, **_kwargs):
        if type(completions) not in (list, tuple) or not 1 <= len(completions) <= MAX_BATCH:
            raise ValueError("COMPLETIONS_MUST_BE_BOUNDED_SEQUENCE")
        if type(expected_rows) not in (list, tuple) or len(expected_rows) != len(completions):
            raise ValueError("EXPECTED_ROWS_MISSING_OR_MISALIGNED")
        texts = [_text(c) for c in completions]  # full-batch validation first
        return [self.score_one(t, e)["reward"] for t, e in zip(texts, expected_rows)]
