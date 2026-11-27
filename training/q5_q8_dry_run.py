# SPDX-License-Identifier: Apache-2.0
"""CPU dry-run for one Q5–Q8 lane on a tiny random config.

The step is standard-library arithmetic. It does not import a training
stack, read a token, open a socket, or write a checkpoint. Q7 uses a
contrastive step. The causal lanes do not.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from training.q5_q8_lanes import LaneError, load_lane


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _causal(spec: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(spec["seed"])
    width = int(spec["width"])
    vocab = int(spec["vocab"])
    seq = int(spec["seq"])
    steps = int(spec["steps"])
    lr = int(spec["lr_milli"]) / 1000.0
    embedding = [
        [rng.uniform(-0.1, 0.1) for _ in range(width)] for _ in range(vocab)
    ]
    output = [
        [rng.uniform(-0.1, 0.1) for _ in range(vocab)] for _ in range(width)
    ]
    tokens = [rng.randrange(vocab) for _ in range(seq)]
    before = output[0][0]
    losses: list[float] = []
    positions = seq - 1
    for _ in range(steps):
        loss = 0.0
        hidden_grad = [[0.0] * width for _ in range(vocab)]
        output_grad = [[0.0] * vocab for _ in range(width)]
        for position in range(positions):
            hidden = embedding[tokens[position]]
            logits = [
                sum(hidden[index] * output[index][token] for index in range(width))
                for token in range(vocab)
            ]
            shift = max(logits)
            scores = [math.exp(logit - shift) for logit in logits]
            total = sum(scores)
            probs = [score / total for score in scores]
            target = tokens[position + 1]
            loss += -math.log(probs[target])
            for index in range(width):
                for token in range(vocab):
                    grad = probs[token] - (1.0 if token == target else 0.0)
                    output_grad[index][token] += hidden[index] * grad
                    hidden_grad[tokens[position]][index] += output[index][token] * grad
        losses.append(loss / positions)
        for token in range(vocab):
            for index in range(width):
                embedding[token][index] -= lr * hidden_grad[token][index]
        for index in range(width):
            for token in range(vocab):
                output[index][token] -= lr * output_grad[index][token]
    return {
        "kind": "causal",
        "losses": losses,
        "loss_finite": _finite(losses),
        "parameter_moved": output[0][0] != before,
    }


def _contrastive(spec: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(spec["seed"])
    width = int(spec["width"])
    batch = int(spec["batch"])
    steps = int(spec["steps"])
    lr = int(spec["lr_milli"]) / 1000.0
    queries = [
        [rng.uniform(-0.2, 0.2) for _ in range(width)] for _ in range(batch)
    ]
    keys = [[rng.uniform(-0.2, 0.2) for _ in range(width)] for _ in range(batch)]
    before = queries[0][0]
    losses: list[float] = []
    for _ in range(steps):
        scores = []
        for row in range(batch):
            scores.append(
                [
                    sum(queries[row][dim] * keys[col][dim] for dim in range(width))
                    for col in range(batch)
                ]
            )
        query_grad = [[0.0] * width for _ in range(batch)]
        key_grad = [[0.0] * width for _ in range(batch)]
        loss = 0.0
        for row in range(batch):
            shift = max(scores[row])
            scores_exp = [math.exp(value - shift) for value in scores[row]]
            total = sum(scores_exp)
            probs = [value / total for value in scores_exp]
            loss += -math.log(probs[row])
            for col in range(batch):
                grad = probs[col] - (1.0 if col == row else 0.0)
                for dim in range(width):
                    query_grad[row][dim] += grad * keys[col][dim]
                    key_grad[col][dim] += grad * queries[row][dim]
        losses.append(loss / batch)
        for row in range(batch):
            for dim in range(width):
                queries[row][dim] -= lr * query_grad[row][dim]
                keys[row][dim] -= lr * key_grad[row][dim]
    return {
        "kind": "contrastive",
        "losses": losses,
        "loss_finite": _finite(losses),
        "parameter_moved": queries[0][0] != before,
    }


def run_lane(lane_id: str, directory: Path | None = None) -> dict[str, Any]:
    """Take the lane's two CPU steps. Q4 is refused before any arithmetic."""

    if lane_id == "Q4":
        raise LaneError("Q4_NOT_APPROVED")
    lane = load_lane(lane_id, directory)
    spec = lane["dry_run"]
    if spec["kind"] == "causal":
        step = _causal(spec)
    elif spec["kind"] == "contrastive":
        step = _contrastive(spec)
    else:
        raise LaneError("DRY_RUN")
    if not step["loss_finite"] or not step["parameter_moved"]:
        raise LaneError("DRY_RUN_FAILED")
    return {
        "lane": lane_id,
        "kind": step["kind"],
        "steps": spec["steps"],
        "device": "cpu",
        "losses": step["losses"],
        "loss_finite": True,
        "parameter_moved": True,
        "launch_executed": False,
        "hub_write": False,
        "provider_call": False,
        "image_push": False,
        "weights_are_random_stand_in": True,
    }
