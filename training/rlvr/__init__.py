"""Deterministic RLVR verifiers. No model judge, no training, no Hub writes."""

from training.rlvr.lean import lean_reward, score_lean
from training.rlvr.puzzles import (
    certified_puzzle_reward,
    score_certified_puzzle,
    score_sudoku,
    sudoku_reward,
)
from training.rlvr.receipts import receipt_reward, score_receipt

__all__ = [
    "certified_puzzle_reward",
    "lean_reward",
    "receipt_reward",
    "score_certified_puzzle",
    "score_lean",
    "score_receipt",
    "score_sudoku",
    "sudoku_reward",
]
