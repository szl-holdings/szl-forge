"""Exact puzzle rewards for the two Hub dataset shapes. No network, no solution oracle."""
from __future__ import annotations

import json
from pathlib import Path

from training.rlvr.puzzles import (
    CERTIFIED,
    SUDOKU,
    certified_puzzle_reward,
    eval_dsl,
    score_certified_puzzle,
    score_sudoku,
    sudoku_reward,
)

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "rlvr" / "certified_bank_samples.json"
SAMPLES = json.loads(FIXTURE.read_text(encoding="utf-8"))

QUESTION_9 = """### Question: Solve this 9x9 Sudoku. Symbols are 1-9; . marks an empty cell.
1 5 . . 2 9 6 7 .
. . 3 1 . 7 . . 8
7 . 2 . . . . . 4
4 . 7 . . 2 . . .
. . . 3 1 4 9 8 .
. . . . . 6 . . 2
2 . . . . 3 8 4 9
. . 9 4 7 . 2 . 5
6 . . . . . . . .
"""
ANSWER_9 = """1 5 4 8 2 9 6 7 3
9 6 3 1 4 7 5 2 8
7 8 2 6 3 5 1 9 4
4 1 7 9 8 2 3 5 6
5 2 6 3 1 4 9 8 7
3 9 8 7 5 6 4 1 2
2 7 1 5 6 3 8 4 9
8 3 9 4 7 1 2 6 5
6 4 5 2 9 8 7 3 1
"""
ROW_9 = QUESTION_9 + "### Answer:\n" + ANSWER_9


def _banded(size, box_rows, box_cols, symbols):
    grid = []
    for row in range(size):
        grid.append([
            symbols[(box_cols * row + row // box_rows + column) % size]
            for column in range(size)
        ])
    return grid


def _format_grid(grid):
    return "\n".join(" ".join(row) for row in grid) + "\n"


def _question_from(grid, symbols_label, blanks):
    shown = []
    for row_index, row in enumerate(grid):
        cells = []
        for column, symbol in enumerate(row):
            cells.append("." if (row_index, column) in blanks else symbol)
        shown.append(" ".join(cells))
    return (
        f"### Question: Solve this {symbols_label} Sudoku. Symbols are "
        + ("1-9; . marks an empty cell.\n" if symbols_label == "9x9" else "1-9,A-G; . marks an empty cell.\n")
        + "\n".join(shown)
        + "\n"
    )


def test_sudoku_row_and_answer_only_pass():
    full = score_sudoku(ROW_9)
    assert full.reward == 1.0
    assert full.receipt["verifier"] == SUDOKU
    assert full.receipt["verdict"] == "pass"
    assert len(full.receipt["input_sha256"]) == 64
    answer_only = score_sudoku(ANSWER_9, question=QUESTION_9)
    assert answer_only.reward == 1.0


def test_sudoku_rejects_broken_given_duplicate_and_malformed_grids():
    lines = ANSWER_9.strip().splitlines()
    swapped = lines[0].split()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    lines[0] = " ".join(swapped)
    broken = score_sudoku("\n".join(lines) + "\n", question=QUESTION_9)
    assert broken.reward == 0.0
    assert broken.receipt["verdict"] == "fail"

    duplicate = ANSWER_9.replace("1 5 4 8 2 9 6 7 3", "1 5 4 8 2 9 6 7 1", 1)
    assert score_sudoku(duplicate, question=QUESTION_9).reward == 0.0

    ragged = score_sudoku("1 2 3\n", question=QUESTION_9)
    assert ragged.reward == 0.0
    assert ragged.receipt["reason"] == "malformed sudoku grid"

    dotted = score_sudoku(ANSWER_9.replace("3", ".", 1), question=QUESTION_9)
    assert dotted.reward == 0.0


def test_rewritten_question_cannot_replace_the_original_givens():
    forged = QUESTION_9.replace("1 5 .", "5 5 .", 1) + "### Answer:\n" + ANSWER_9
    # Graded alone, the completion's own clue says the first cell is 5 and the
    # answer puts 1 there.
    alone = score_sudoku(forged)
    assert alone.reward == 0.0
    assert alone.receipt["reason"] == "answer breaks a given"
    # The original puzzle still says 1. The rewrite does not replace that clue.
    kept = score_sudoku(forged, question=QUESTION_9)
    assert kept.reward == 1.0


def test_sixteen_square_pass_and_bad_symbol():
    symbols = list("123456789ABCDEFG")
    grid = _banded(16, 4, 4, symbols)
    question = _question_from(grid, "16x16", {(0, 0), (1, 2), (15, 15)})
    outcome = score_sudoku(_format_grid(grid), question=question)
    assert outcome.reward == 1.0
    bad = _format_grid(grid).replace("G", "H", 1)
    rejected = score_sudoku(bad, question=question)
    assert rejected.reward == 0.0


def test_trl_sudoku_batch():
    batch = sudoku_reward(
        [
            [{"role": "assistant", "content": ANSWER_9}],
            [{"role": "assistant", "content": "not a grid"}],
        ],
        question=QUESTION_9,
    )
    assert list(batch) == [1.0, 0.0]
    assert [item["verdict"] for item in batch.receipts] == ["pass", "fail"]


def _without_solution(item):
    cleaned = json.loads(json.dumps(item))
    cleaned.pop("solution", None)
    if isinstance(cleaned.get("spec"), dict):
        cleaned["spec"].pop("solution", None)
    return cleaned


def test_dsl_operators_and_injection():
    houses = {"a": 1, "b": 2, "c": 3}
    assert eval_dsl('pos("a") == 1', houses)
    assert eval_dsl('pos("a") != pos("b")', houses)
    assert eval_dsl('pos("a") < pos("b")', houses)
    assert eval_dsl('pos("a") + 1 == pos("b")', houses)
    assert eval_dsl('Abs(pos("a") - pos("c")) == 2', houses)
    assert eval_dsl('Abs(pos("a") - pos("b")) != 0', houses)
    assert eval_dsl('Or(pos("a") == 9, pos("b") == 2)', houses)
    assert eval_dsl('Or(pos("a") == pos("b"), pos("c") == 3)', houses)
    assert eval_dsl('pos("a") == 1 or True', houses) is False
    assert eval_dsl('__import__("os")', houses) is False


def test_certified_samples_pass_without_the_stored_solution():
    for family in ("zebra", "futoshiki", "killer", "sudoku"):
        item = _without_solution(SAMPLES[family])
        outcome = score_certified_puzzle(SAMPLES[family]["solution"], item=item)
        assert outcome.reward == 1.0, outcome.receipt
        assert outcome.receipt["verifier"] == CERTIFIED
        assert outcome.receipt["verdict"] == "pass"


def test_zebra_house_format_and_clue_failure():
    item = _without_solution(SAMPLES["zebra"])
    houses = {}
    for category, labels in item["spec"]["cats"].items():
        for label in labels:
            house = SAMPLES["zebra"]["solution"][label.lower()]
            houses.setdefault(str(house), {})[category] = label
    outcome = score_certified_puzzle(houses, item=item)
    assert outcome.reward == 1.0

    broken = dict(SAMPLES["zebra"]["solution"])
    broken["red"], broken["blue"] = broken["blue"], broken["red"]
    rejected = score_certified_puzzle(broken, item=item)
    assert rejected.reward == 0.0
    assert rejected.receipt["reason"] == "zebra clue failed"


def test_certified_negatives_are_exact():
    fut = _without_solution(SAMPLES["futoshiki"])
    assert score_certified_puzzle({"grid": True}, item=fut).reward == 0.0
    short = score_certified_puzzle([[1, 2, 3]], item=fut)
    assert short.reward == 0.0
    assert short.receipt["reason"] == "malformed grid"

    mutated = json.loads(json.dumps(fut))
    mutated["clues"] = [clue for clue in mutated["clues"] if clue["type"] == "given"]
    mutated["clues"].append({"type": "lt", "a": [0, 0], "b": [0, 1], "n": 99})
    inequality = score_certified_puzzle(SAMPLES["futoshiki"]["solution"], item=mutated)
    assert inequality.reward == 0.0
    assert inequality.receipt["reason"] == "futoshiki inequality failed"

    killer = _without_solution(SAMPLES["killer"])
    bad_sum = json.loads(json.dumps(killer))
    for clue in bad_sum["clues"]:
        if clue["type"] == "cage":
            clue["sum"] = 999
            break
    cages = score_certified_puzzle(SAMPLES["killer"]["solution"], item=bad_sum)
    assert cages.reward == 0.0
    assert cages.receipt["reason"] == "killer cage failed"

    sudoku = _without_solution(SAMPLES["sudoku"])
    wrong_given = json.loads(json.dumps(sudoku))
    for clue in wrong_given["clues"]:
        if clue["type"] == "given":
            clue["value"] = 1 if clue["value"] != 1 else 2
            break
    given = score_certified_puzzle(SAMPLES["sudoku"]["solution"], item=wrong_given)
    assert given.reward == 0.0
    assert given.receipt["reason"] == "given mismatch"


def test_killer_does_not_require_cages_to_cover_the_grid():
    item = {
        "family": "killer",
        "clues": [{"type": "cage", "cells": [[0, 0], [0, 1]], "sum": 3, "n": 1}],
        "spec": {"family": "killer", "n": 2, "box": [1, 2]},
    }
    assert score_certified_puzzle([[1, 2], [2, 1]], item=item).reward == 1.0
    assert score_certified_puzzle([[1, 2], [2, 2]], item=item).reward == 0.0


def test_certified_batch_and_missing_item():
    item = _without_solution(SAMPLES["zebra"])
    batch = certified_puzzle_reward(
        [SAMPLES["zebra"]["solution"], {"red": 1}],
        item=item,
    )
    assert list(batch) == [1.0, 0.0]
    missing = score_certified_puzzle({"a": 1})
    assert missing.reward == 0.0
    assert missing.receipt["reason"] == "certified item missing"
