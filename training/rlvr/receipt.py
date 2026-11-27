"""Shared RLVR receipt and the TRL GRPO reward-batch shape.

Every verifier returns a reward in {0.0, 1.0} and a JSON receipt. The batch
object is a list of floats, so TRL's GRPOTrainer can consume it, and it
carries the receipts that the trainer itself does not store.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

VERIFIER_VERSION = "1.0.0"


def input_bytes(completion: Any) -> bytes:
    if isinstance(completion, str):
        return completion.encode("utf-8")
    return json.dumps(
        completion, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_hex(completion: Any) -> str:
    return hashlib.sha256(input_bytes(completion)).hexdigest()


def make_receipt(
    verifier: str,
    completion: Any,
    verdict: str,
    reason: str,
    **extra: Any,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "verifier": verifier,
        "version": VERIFIER_VERSION,
        "input_sha256": sha256_hex(completion),
        "verdict": verdict,
        "reason": reason,
    }
    receipt.update(extra)
    return receipt


class VerifierOutcome:
    """One completion: a reward and the receipt that justifies it."""

    def __init__(self, reward: float, receipt: dict[str, Any]) -> None:
        self.reward = float(reward)
        self.receipt = receipt

    def __iter__(self):
        yield self.reward
        yield self.receipt


def passed(verifier: str, completion: Any, reason: str, **extra: Any) -> VerifierOutcome:
    return VerifierOutcome(
        1.0, make_receipt(verifier, completion, "pass", reason, **extra)
    )


def failed(verifier: str, completion: Any, reason: str, **extra: Any) -> VerifierOutcome:
    return VerifierOutcome(
        0.0, make_receipt(verifier, completion, "fail", reason, **extra)
    )


class RewardBatch(list):
    """list[float] for GRPO, with one receipt per completion.

    ``torch.tensor(batch)`` sees the floats. ``batch.receipts`` is the evidence.
    """

    def __init__(self, outcomes: list[VerifierOutcome]) -> None:
        super().__init__(outcome.reward for outcome in outcomes)
        self.receipts = [outcome.receipt for outcome in outcomes]


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content")
        if isinstance(content, str):
            return content
        return json.dumps(completion, sort_keys=True, ensure_ascii=False)
    if isinstance(completion, list):
        for message in reversed(completion):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if message.get("role") == "assistant" and isinstance(content, str):
                return content
        if completion and isinstance(completion[-1], dict):
            content = completion[-1].get("content")
            if isinstance(content, str):
                return content
    raise TypeError("unreadable completion")


def align_column(value: Any, count: int) -> list[Any]:
    """Zip a TRL column when it is a per-completion list of strings or dicts.

    Grids and other nested lists stay scalar: their first element is not a
    string or a dict.
    """
    if (
        isinstance(value, list)
        and len(value) == count
        and count > 0
        and isinstance(value[0], (str, dict))
    ):
        return list(value)
    return [value] * count


def batch_reward(completions: list[Any], score, **kwargs: Any) -> RewardBatch:
    count = len(completions)
    aligned = {key: align_column(value, count) for key, value in kwargs.items()}
    outcomes: list[VerifierOutcome] = []
    for index, completion in enumerate(completions):
        row = {key: values[index] for key, values in aligned.items()}
        try:
            outcomes.append(score(completion, **row))
        except Exception as exc:  # noqa: BLE001 — a reward must not crash GRPO
            outcomes.append(
                failed(
                    getattr(score, "verifier_name", "szl.rlvr"),
                    completion,
                    f"verifier error: {type(exc).__name__}",
                )
            )
    return RewardBatch(outcomes)
