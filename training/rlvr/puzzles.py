"""Exact puzzle rewards. No solver, no model judge, no stored-answer shortcut.

Sudoku rows follow ``cram-sudoku-extreme-sft-v1``: a ``text`` field whose
question and answer are space-separated grids. 9x9 uses symbols 1-9.
16x16 uses 1-9 and A-G. ``.`` is an empty question cell.

Certified items follow ``certified-puzzles-frontier-bank-v1``: ``family``,
``clues``, and ``spec`` (``n``, ``box``, ``cats``). The checker uses those
constraints. It does not read ``solution``.
"""
from __future__ import annotations

import json
import re
from typing import Any

from training.rlvr.receipt import RewardBatch, batch_reward, completion_text, failed, passed

SUDOKU = "szl.rlvr.sudoku"
CERTIFIED = "szl.rlvr.certified_puzzle"
_SYMBOLS_9 = "123456789"
_SYMBOLS_16 = "123456789ABCDEFG"
_MAX_CHARS = 200_000


def _plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class _Parse:
    def __init__(self, text: str, env: dict[str, int]) -> None:
        self.text = text
        self.index = 0
        self.env = env

    def skip(self) -> None:
        while self.index < len(self.text) and self.text[self.index].isspace():
            self.index += 1

    def startswith(self, token: str) -> bool:
        self.skip()
        return self.text.startswith(token, self.index)

    def eat(self, token: str) -> None:
        if not self.startswith(token):
            raise ValueError(f"expected {token}")
        self.index += len(token)

    def parse(self) -> bool:
        value = self.comparison()
        self.skip()
        if self.index != len(self.text):
            raise ValueError("trailing dsl")
        if not isinstance(value, bool):
            raise ValueError("dsl is not a predicate")
        return value

    def comparison(self) -> Any:
        left = self.sum()
        for operator in ("==", "!=", "<"):
            if self.startswith(operator):
                self.eat(operator)
                right = self.sum()
                if not _plain_int(left) or not _plain_int(right):
                    raise ValueError("comparison of non-integers")
                if operator == "==":
                    return left == right
                if operator == "!=":
                    return left != right
                return left < right
        return left

    def sum(self) -> Any:
        left = self.unary()
        while True:
            if self.startswith("+"):
                operator = "+"
            elif self.startswith("-"):
                operator = "-"
            else:
                break
            self.eat(operator)
            right = self.unary()
            if not _plain_int(left) or not _plain_int(right):
                raise ValueError("arithmetic of non-integers")
            left = left + right if operator == "+" else left - right
        return left

    def unary(self) -> Any:
        if self.startswith("Abs("):
            self.eat("Abs(")
            inner = self.sum()
            self.eat(")")
            if not _plain_int(inner):
                raise ValueError("Abs of non-integer")
            return abs(inner)
        if self.startswith("Or("):
            self.eat("Or(")
            left = self.comparison()
            self.eat(",")
            right = self.comparison()
            self.eat(")")
            if not isinstance(left, bool) or not isinstance(right, bool):
                raise ValueError("Or of non-predicates")
            return bool(left or right)
        if self.startswith("pos("):
            self.eat('pos("')
            end = self.text.find('"', self.index)
            if end < 0:
                raise ValueError("unterminated pos")
            name = self.text[self.index:end]
            self.index = end + 1
            self.eat(")")
            if name not in self.env:
                raise ValueError("unknown pos")
            return self.env[name]
        if self.startswith("("):
            self.eat("(")
            value = self.comparison()
            self.eat(")")
            return value
        self.skip()
        match = re.match(r"\d+", self.text[self.index:])
        if not match:
            raise ValueError("expected integer")
        self.index += match.end()
        return int(match.group())


def eval_dsl(dsl: str, houses: dict[str, int]) -> bool:
    """Closed zebra DSL. Anything outside the grammar is false."""
    try:
        return _Parse(dsl, houses).parse()
    except (ValueError, KeyError, IndexError, RecursionError):
        return False


def _rows(block: str) -> list[str]:
    return [line.strip() for line in block.splitlines() if line.strip()]


def _cell(token: str, symbols: str) -> str | None:
    if token == ".":
        return "."
    upper = token.upper()
    if upper in symbols and len(token) == 1:
        return upper if symbols is _SYMBOLS_16 else token
    return None


def _parse_block(block: str, symbols: str) -> list[list[str]] | None:
    size = len(symbols)
    rows = _rows(block)
    if len(rows) != size:
        return None
    grid: list[list[str]] = []
    for line in rows:
        parts = line.split()
        if len(parts) != size:
            return None
        cells = [_cell(part, symbols) for part in parts]
        if any(cell is None for cell in cells):
            return None
        grid.append([cell for cell in cells if cell is not None])
    return grid


def parse_sft_text(text: str) -> tuple[list[list[str]], list[list[str]], str] | None:
    if "### Answer:" not in text or "### Question:" not in text:
        return None
    question_part, answer_part = text.split("### Answer:", 1)
    header = question_part.split("### Question:", 1)[1]
    if "16x16" in question_part:
        symbols = _SYMBOLS_16
    elif "9x9" in question_part:
        symbols = _SYMBOLS_9
    else:
        return None
    # Drop the prose line before the grid.
    question_lines = header.splitlines()
    question_body = "\n".join(question_lines[1:]) if len(question_lines) > 1 else ""
    question = _parse_block(question_body, symbols)
    answer = _parse_block(answer_part, symbols)
    if question is None or answer is None:
        return None
    return question, answer, symbols


def _latin(grid: list[list[str]], symbols: str, box: int) -> bool:
    size = len(symbols)
    want = set(symbols)
    if len(grid) != size:
        return False
    for row in grid:
        if len(row) != size or set(row) != want or any(cell == "." for cell in row):
            return False
    for column in range(size):
        if {grid[row][column] for row in range(size)} != want:
            return False
    for brow in range(0, size, box):
        for bcol in range(0, size, box):
            cells = {
                grid[row][column]
                for row in range(brow, brow + box)
                for column in range(bcol, bcol + box)
            }
            if cells != want:
                return False
    return True


def _matches_dots(question: list[list[str]], answer: list[list[str]]) -> bool:
    for row, question_row in enumerate(question):
        for column, given in enumerate(question_row):
            if given != "." and answer[row][column] != given:
                return False
    return True


def _question_and_answer(body: str, ground: str | None) -> tuple[list[list[str]], list[list[str]], str] | None:
    """Clues come from ``ground`` when the caller supplied the original puzzle.

    A completion that restates an easier question cannot replace those clues.
    """
    question_source = ground if isinstance(ground, str) and "### Question:" in ground else body
    if "### Question:" not in question_source:
        return None
    preamble = question_source.split("### Answer:", 1)[0]
    if "16x16" in preamble:
        symbols = _SYMBOLS_16
    elif "9x9" in preamble:
        symbols = _SYMBOLS_9
    else:
        return None
    header = preamble.split("### Question:", 1)[1]
    question_lines = header.splitlines()
    question_body = "\n".join(question_lines[1:]) if len(question_lines) > 1 else ""
    if "### Answer:" in body:
        answer_text = body.split("### Answer:", 1)[1]
    elif ground and "### Question:" in ground:
        answer_text = body
    else:
        return None
    question_grid = _parse_block(question_body, symbols)
    answer_grid = _parse_block(answer_text, symbols)
    if question_grid is None or answer_grid is None:
        return None
    return question_grid, answer_grid, symbols


def score_sudoku(completion: Any, *, question: str | None = None, text: str | None = None, **_ignored: Any):
    try:
        body = completion_text(completion)
    except TypeError:
        return failed(SUDOKU, completion, "unreadable completion")
    if len(body) > _MAX_CHARS:
        return failed(SUDOKU, completion, "completion too large")
    ground = question if isinstance(question, str) else text if isinstance(text, str) else None
    parsed = _question_and_answer(body, ground)
    if parsed is None:
        return failed(SUDOKU, completion, "malformed sudoku grid")
    question_grid, answer_grid, symbols = parsed
    box = 3 if len(symbols) == 9 else 4
    if not _latin(answer_grid, symbols, box):
        return failed(SUDOKU, completion, "answer is not a completed sudoku")
    if not _matches_dots(question_grid, answer_grid):
        return failed(SUDOKU, completion, "answer breaks a given")
    return passed(SUDOKU, completion, f"{len(symbols)}x{len(symbols)} sudoku checks")


def _as_json(completion: Any) -> Any:
    if isinstance(completion, (dict, list)):
        return completion
    text = completion_text(completion).strip()
    if len(text) > _MAX_CHARS:
        raise ValueError("too large")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    return json.loads(text)


def _int_grid_from_row(row: Any, size: int) -> list[int] | None:
    if isinstance(row, str):
        parts = row.split()
        if len(parts) == 1 and len(row) == size and row.isdigit():
            parts = list(row)
        if len(parts) != size:
            return None
        values: list[int] = []
        for part in parts:
            if not part.isdigit():
                return None
            number = int(part)
            if number < 1 or number > size:
                return None
            values.append(number)
        return values
    if isinstance(row, list) and len(row) == size and all(_plain_int(cell) for cell in row):
        if any(cell < 1 or cell > size for cell in row):
            return None
        return list(row)
    return None


def parse_int_grid(plan: Any, size: int) -> list[list[int]] | None:
    if isinstance(plan, dict):
        plan = plan.get("grid")
    if not isinstance(plan, list) or len(plan) != size:
        return None
    grid = [_int_grid_from_row(row, size) for row in plan]
    if any(row is None for row in grid):
        return None
    return [row for row in grid if row is not None]


def _int_latin(grid: list[list[int]], size: int, box: tuple[int, int] | None) -> bool:
    want = set(range(1, size + 1))
    if any(set(row) != want for row in grid):
        return False
    for column in range(size):
        if {grid[row][column] for row in range(size)} != want:
            return False
    if box is None:
        return True
    box_rows, box_cols = box
    if box_rows < 1 or box_cols < 1 or size % box_rows or size % box_cols:
        return False
    for brow in range(0, size, box_rows):
        for bcol in range(0, size, box_cols):
            cells = {
                grid[row][column]
                for row in range(brow, brow + box_rows)
                for column in range(bcol, bcol + box_cols)
            }
            if cells != want:
                return False
    return True


def _clue_list(item: dict[str, Any]) -> list[Any] | None:
    clues = item.get("clues")
    if not isinstance(clues, list) or not clues:
        return None
    return clues


def _zebra_houses(plan: Any, spec: dict[str, Any]) -> dict[str, int] | None:
    categories = spec.get("cats")
    size = spec.get("n")
    if not isinstance(categories, dict) or not categories or not _plain_int(size):
        return None
    labels = {}
    for name, values in categories.items():
        if not isinstance(name, str) or not isinstance(values, list) or not values:
            return None
        for value in values:
            if not isinstance(value, str):
                return None
            labels[value.lower()] = name
    house_keyed = (
        isinstance(plan, dict)
        and bool(plan)
        and "grid" not in plan
        and all(isinstance(value, dict) for value in plan.values())
        and all((isinstance(key, str) and key.isdigit()) or _plain_int(key) for key in plan)
    )
    if house_keyed:
        flat = {}
        for house, mapping in plan.items():
            house_number = int(house)
            for _category, value in mapping.items():
                if not isinstance(value, str):
                    return None
                flat[value.lower()] = house_number
        plan = flat
    if not isinstance(plan, dict):
        return None
    houses: dict[str, int] = {}
    for key, value in plan.items():
        if not isinstance(key, str) or not _plain_int(value):
            return None
        houses[key.lower()] = value
    if set(houses) != set(labels):
        return None
    for values in categories.values():
        seen = [houses[value.lower()] for value in values]
        if any(not 1 <= house <= size for house in seen):
            return None
        if sorted(seen) != list(range(1, size + 1)):
            return None
    return houses


def _check_zebra(item: dict[str, Any], plan: Any) -> str | None:
    spec = item.get("spec")
    if not isinstance(spec, dict):
        return "malformed zebra spec"
    houses = _zebra_houses(plan, spec)
    if houses is None:
        return "malformed zebra answer"
    clues = _clue_list(item)
    if clues is None:
        return "malformed zebra clues"
    for clue in clues:
        if not isinstance(clue, dict) or not isinstance(clue.get("dsl"), str):
            return "malformed zebra clue"
        if not eval_dsl(clue["dsl"], houses):
            return "zebra clue failed"
    return None


def _box(spec: dict[str, Any]) -> tuple[int, int] | None:
    box = spec.get("box")
    if (
        isinstance(box, list)
        and len(box) == 2
        and _plain_int(box[0])
        and _plain_int(box[1])
    ):
        return (box[0], box[1])
    return None


def _givens_and_rest(grid: list[list[int]], clues: list[Any], family: str) -> str | None:
    size = len(grid)
    for clue in clues:
        if not isinstance(clue, dict) or not isinstance(clue.get("type"), str):
            return "malformed clue"
        kind = clue["type"]
        if kind == "given":
            cell = clue.get("cell")
            value = clue.get("value")
            if (
                not isinstance(cell, list)
                or len(cell) != 2
                or not all(_plain_int(part) for part in cell)
                or not _plain_int(value)
            ):
                return "malformed given"
            row, column = cell
            if not (0 <= row < size and 0 <= column < size) or grid[row][column] != value:
                return "given mismatch"
        elif family == "futoshiki" and kind == "lt":
            left, right = clue.get("a"), clue.get("b")
            if not _coord(left, size) or not _coord(right, size):
                return "malformed inequality"
            if not grid[left[0]][left[1]] < grid[right[0]][right[1]]:
                return "futoshiki inequality failed"
        elif family == "killer" and kind == "cage":
            continue
        else:
            return f"unknown clue type {kind}"
    return None


def _coord(cell: Any, size: int) -> bool:
    return (
        isinstance(cell, list)
        and len(cell) == 2
        and all(_plain_int(part) for part in cell)
        and 0 <= cell[0] < size
        and 0 <= cell[1] < size
    )


def _cages(grid: list[list[int]], clues: list[Any]) -> str | None:
    size = len(grid)
    seen: set[tuple[int, int]] = set()
    found = False
    for clue in clues:
        if not isinstance(clue, dict) or clue.get("type") != "cage":
            continue
        found = True
        cells = clue.get("cells")
        total = clue.get("sum")
        if not isinstance(cells, list) or not cells or not _plain_int(total):
            return "malformed cage"
        values = []
        for cell in cells:
            if not _coord(cell, size) or (cell[0], cell[1]) in seen:
                return "malformed cage"
            seen.add((cell[0], cell[1]))
            values.append(grid[cell[0]][cell[1]])
        if len(set(values)) != len(values) or sum(values) != total:
            return "killer cage failed"
    if not found:
        return "killer has no cages"
    return None


def score_certified_puzzle(completion: Any, *, item: dict[str, Any] | None = None, **_ignored: Any):
    if not isinstance(item, dict):
        return failed(CERTIFIED, completion, "certified item missing")
    family = item.get("family")
    try:
        plan = _as_json(completion)
    except Exception:  # noqa: BLE001
        return failed(CERTIFIED, completion, "malformed puzzle answer")
    if family == "zebra":
        reason = _check_zebra(item, plan)
        if reason:
            return failed(CERTIFIED, completion, reason)
        return passed(CERTIFIED, completion, "zebra constraints hold")
    spec = item.get("spec")
    if not isinstance(spec, dict) or spec.get("family") not in (None, family):
        return failed(CERTIFIED, completion, "malformed spec")
    size = spec.get("n")
    if family not in {"sudoku", "futoshiki", "killer"} or not _plain_int(size):
        return failed(CERTIFIED, completion, "unsupported certified family")
    grid = parse_int_grid(plan, size)
    if grid is None:
        return failed(CERTIFIED, completion, "malformed grid")
    box = None if family == "futoshiki" else _box(spec)
    if family != "futoshiki" and box is None:
        return failed(CERTIFIED, completion, "malformed box")
    if not _int_latin(grid, size, box):
        return failed(CERTIFIED, completion, "grid breaks latin or box constraints")
    clues = _clue_list(item)
    if clues is None:
        return failed(CERTIFIED, completion, "malformed clues")
    reason = _givens_and_rest(grid, clues, family)
    if reason:
        return failed(CERTIFIED, completion, reason)
    if family == "killer":
        reason = _cages(grid, clues)
        if reason:
            return failed(CERTIFIED, completion, reason)
    return passed(CERTIFIED, completion, f"{family} constraints hold")


def sudoku_reward(completions: list[Any], **kwargs: Any) -> RewardBatch:
    """TRL GRPO reward for cram-sudoku text. Receipts are on ``.receipts``."""
    score_sudoku.verifier_name = SUDOKU  # type: ignore[attr-defined]
    return batch_reward(completions, score_sudoku, **kwargs)


def certified_puzzle_reward(completions: list[Any], **kwargs: Any) -> RewardBatch:
    """TRL GRPO reward for certified-puzzles items. Receipts are on ``.receipts``."""
    score_certified_puzzle.verifier_name = CERTIFIED  # type: ignore[attr-defined]
    return batch_reward(completions, score_certified_puzzle, **kwargs)
