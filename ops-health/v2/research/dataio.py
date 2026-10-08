"""Row loading for training, calibration and selection code: refuses any sealed path.

The only code allowed to read bytes under v2/data/sealed/ is make_data.py (writing and
hash verification) and sealed_guard.py (hash verification; parsing only inside the single
FINAL_TEST_OPENING).  Everything else loads rows through this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from . import registry


class SealedPathError(PermissionError):
    """Raised when non-final code tries to touch a sealed split."""


def is_sealed_path(path: Path | str) -> bool:
    resolved = Path(path).resolve()
    if any(part.lower() == registry.SEALED_DIRNAME for part in resolved.parts):
        return True
    try:
        resolved.relative_to(registry.SEALED_DIR.resolve())
        return True
    except ValueError:
        return False


def refuse_sealed_path(path: Path | str) -> Path:
    if is_sealed_path(path):
        raise SealedPathError(
            f"refusing sealed path {path}: sealed splits are opened only by "
            "sealed_guard with purpose FINAL_TEST_OPENING"
        )
    return Path(path)


def parse_jsonl_bytes(data: bytes) -> list[dict]:
    text = data.decode("utf-8")
    if text and not text.endswith("\n"):
        raise ValueError("JSONL must end with a newline")
    return [json.loads(line) for line in text.split("\n")[:-1]] if text else []


def load_rows(path: Path | str) -> list[dict]:
    """Load a non-sealed JSONL file (legacy_v1 or an explicit fixture)."""
    return parse_jsonl_bytes(refuse_sealed_path(path).read_bytes())


def labels_of(rows: Iterable[dict]) -> list[int]:
    return [1 if row["label"]["operator_attention_required"] else 0 for row in rows]


def features_of(rows: Iterable[dict]) -> list[dict]:
    return [row["features"] for row in rows]
