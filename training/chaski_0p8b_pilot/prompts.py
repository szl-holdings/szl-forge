"""Choose Chaski prompts for teacher generation.

Held-out and eval files never become training prompts. A file with no
license basis is marked UNKNOWN and left out. Assistant turns are dropped:
the teacher writes the completion. This module reads local JSONL only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# First-party rows in this Apache-2.0 tree. There is no second dataset-license
# file. The repository license is the basis, and it is not an operator
# admission receipt. launch_contract.py still has to accept a receipt before
# any real wave.
ADMITTED_SOURCES: dict[str, dict[str, str]] = {
    "szl_dataset.jsonl": {
        "license": "Apache-2.0",
        "basis": (
            "First-party rows in this Apache-2.0 repository. "
            "chaski/train_chaski.py loads only this file. "
            "No separate dataset-license file was found."
        ),
    },
    "chaski_r2/train.jsonl": {
        "license": "Apache-2.0",
        "basis": (
            "First-party Chaski courier rows in this Apache-2.0 repository. "
            "chaski_r2/README.md names this file as the train turns. "
            "No separate dataset-license file was found."
        ),
    },
    "chaski_r4/train.jsonl": {
        "license": "Apache-2.0",
        "basis": (
            "First-party Chaski courier rows in this Apache-2.0 repository. "
            "chaski_r4/training_plan.json names this file as the curriculum. "
            "No separate dataset-license file was found."
        ),
    },
}

# Present in the tree, and not admitted. The reason is the classification.
UNKNOWN_REASONS: dict[str, str] = {
    "szl_dataset_5050_merged.jsonl": (
        "Merged file. The extra rows beyond szl_dataset.jsonl have no "
        "separate license record, so the file is UNKNOWN."
    ),
    "chaski_r4/train.pre-fix-backup.jsonl": (
        "Superseded backup of the r4 curriculum. The fix is not re-audited "
        "here, so the file is UNKNOWN."
    ),
}

SCAN_ROOTS = (
    "szl_dataset.jsonl",
    "szl_dataset_5050_merged.jsonl",
    "chaski",
    "chaski_r2",
    "chaski_r4",
    "chaski-r2",
    "chaski-5050",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path} line {line_no} is not JSON") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"{path} line {line_no} is not an object")
        rows.append(row)
    return rows


def _user_text(row: dict[str, Any]) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    parts = [
        message.get("content")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if not parts or any(not isinstance(part, str) or not part.strip() for part in parts):
        return None
    return "\n".join(parts)


def _prompt_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    kept: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if role == "assistant":
            continue
        if role not in ("system", "user") or not isinstance(content, str) or not content.strip():
            return None
        kept.append({"role": role, "content": content})
    if not any(message["role"] == "user" for message in kept):
        return None
    return kept


def _is_heldout_path(relative: str) -> bool:
    name = Path(relative).name
    return (
        relative in {
            "chaski/gate/json_drafts.n5.jsonl",
            "chaski/gate/adversarial_refusals.n6.jsonl",
        }
        or "/chaski/gate/" in f"/{relative}"
        or relative.startswith("chaski/gate/")
        or "heldout" in name.casefold()
        or name in {"json_drafts.n5.jsonl", "adversarial_refusals.n6.jsonl"}
    )


def _candidate_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for item in SCAN_ROOTS:
        path = root / item
        if path.is_file() and path.suffix == ".jsonl":
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(path.rglob("*.jsonl")))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved in seen:
            continue
        resolved.relative_to(root.resolve())
        seen.add(resolved)
        unique.append(path)
    return unique


def _heldout_users(root: Path, files: list[Path]) -> dict[str, str]:
    users: dict[str, str] = {}
    for path in files:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        if not _is_heldout_path(relative):
            continue
        for row in _load_jsonl(path):
            text = _user_text(row)
            if text is None:
                continue
            users.setdefault(text, relative)
    if len(users) < 11:
        raise RuntimeError("held-out gate files did not yield the 5+6 user prompts")
    return users


def select_prompts(root: Path | None = None) -> dict[str, Any]:
    """Inventory Chaski JSONL under ``root`` and return the admitted prompts."""

    repo = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    files = _candidate_files(repo)
    heldout_users = _heldout_users(repo, files)
    inventory: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    seen_users: set[str] = set()

    for path in files:
        relative = path.resolve().relative_to(repo.resolve()).as_posix()
        digest = _sha256(path)
        if _is_heldout_path(relative):
            inventory.append(
                {
                    "path": relative,
                    "sha256": digest,
                    "status": "HELD_OUT",
                    "license": None,
                    "reason": "Eval or held-out file. Excluded from teacher prompts and from training.",
                }
            )
            continue
        if relative in UNKNOWN_REASONS or relative not in ADMITTED_SOURCES:
            inventory.append(
                {
                    "path": relative,
                    "sha256": digest,
                    "status": "UNKNOWN",
                    "license": None,
                    "reason": UNKNOWN_REASONS.get(
                        relative,
                        "No license basis is recorded for this Chaski JSONL. Excluded.",
                    ),
                }
            )
            continue

        source = ADMITTED_SOURCES[relative]
        kept = 0
        dropped_heldout = 0
        dropped_duplicate = 0
        for index, row in enumerate(_load_jsonl(path)):
            user = _user_text(row)
            messages = _prompt_messages(row)
            if user is None or messages is None:
                continue
            if user in heldout_users:
                dropped_heldout += 1
                continue
            if user in seen_users:
                dropped_duplicate += 1
                continue
            seen_users.add(user)
            kept += 1
            prompts.append(
                {
                    "id": f"{relative}:{index}",
                    "source": relative,
                    "source_sha256": digest,
                    "license": source["license"],
                    "messages": messages,
                }
            )
        inventory.append(
            {
                "path": relative,
                "sha256": digest,
                "status": "ADMITTED",
                "license": source["license"],
                "basis": source["basis"],
                "prompts_kept": kept,
                "dropped_heldout": dropped_heldout,
                "dropped_duplicate": dropped_duplicate,
                "operator_admission": False,
            }
        )

    admitted_users = {
        "\n".join(
            message["content"] for message in prompt["messages"] if message["role"] == "user"
        )
        for prompt in prompts
    }
    leaked = sorted(admitted_users & set(heldout_users))
    if leaked:
        raise RuntimeError("admitted prompts overlap held-out user text")
    return {
        "schema": "szl.chaski-0p8b-pilot.prompts/v1",
        "admitted_prompts": len(prompts),
        "heldout_user_prompts": len(heldout_users),
        "inventory": inventory,
        "prompts": prompts,
        "operator_admission": False,
        "launch_authorized": False,
    }
