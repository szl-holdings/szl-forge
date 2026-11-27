"""Lean reward: 1 only if a candidate checks in pinned lutar-lean.

The candidate is Lean source. It is never passed to a shell. The only
process this module spawns is ``lake env lean`` inside a checkout whose
HEAD is the doctrine v11 commit, with a wall-clock timeout and a secret-free
environment. ``sorry``, ``admit``, a new ``axiom``, and ``native_decide``
score 0 before that process starts. Kernel axioms must be inside the 14
pinned names plus the Lean trust base ``propext``, ``Classical.choice``,
and ``Quot.sound``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from training.rlvr.receipt import (
    RewardBatch,
    VerifierOutcome,
    batch_reward,
    completion_text,
    failed,
    passed,
)

VERIFIER = "szl.rlvr.lean"
_BASELINE_PATH = Path(__file__).resolve().parent / "baseline" / "lutar_lean_v11.json"
_BASELINE = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
PINNED_COMMIT = _BASELINE["commit"]
BASELINE_AXIOMS = frozenset(_BASELINE["axiom_names"])
KERNEL_TRUST_BASE = frozenset(_BASELINE["kernel_trust_base"])
ALLOWED_AXIOMS = BASELINE_AXIOMS | KERNEL_TRUST_BASE
TRACKED_SORRIES = int(_BASELINE["tracked_sorries"])
AXIOM_COUNT = int(_BASELINE["axioms_unique"])

_MAX_SOURCE_BYTES = 200_000
_DEFAULT_TIMEOUT = 30.0
_MAX_TIMEOUT = 120.0
_BLOCKED_IDENTS = frozenset({
    "sorry",
    "admit",
    "axiom",
    "native_decide",
    "sorryAx",
    "run_cmd",
})
_BLOCKED_DOTTED = frozenset({"IO.Process"})
_IDENT = re.compile(r"[A-Za-z_#][A-Za-z0-9_'!?]*")
_DOTTED = re.compile(r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)+")
_DECL = re.compile(r"\b(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_']*)")
_AXIOM_LINE = re.compile(r"depends on axioms:\s*\[([^\]]*)\]")
_KEEP_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "ELAN_HOME", "USER", "TMPDIR")


class LeanSourceError(ValueError):
    """The candidate cannot be lexed unambiguously. Fail closed."""


def _skip_block_comment(src: str, index: int) -> int:
    depth = 1
    index += 2
    limit = len(src)
    while index < limit and depth:
        if src.startswith("/-", index):
            depth += 1
            index += 2
        elif src.startswith("-/", index):
            depth -= 1
            index += 2
        else:
            index += 1
    if depth:
        raise LeanSourceError("unterminated block comment")
    return index


def _skip_string(src: str, index: int) -> int:
    index += 1
    limit = len(src)
    while index < limit:
        char = src[index]
        if char == "\\":
            index += 2
            continue
        if char == '"':
            return index + 1
        if char == "\n":
            break
        index += 1
    raise LeanSourceError("unterminated string")


def strip_lean(src: str) -> str:
    """Code with comments removed and string text emptied. Fails closed."""
    out: list[str] = []
    index = 0
    limit = len(src)
    while index < limit:
        if src.startswith("--", index):
            while index < limit and src[index] != "\n":
                index += 1
            continue
        if src.startswith("/-", index):
            index = _skip_block_comment(src, index)
            out.append(" ")
            continue
        if src[index] == '"':
            index = _skip_string(src, index)
            out.append('""')
            continue
        out.append(src[index])
        index += 1
    return "".join(out)


def static_block_reason(src: str) -> str | None:
    """Return why this source must not be elaborated, or None."""
    if "\x00" in src:
        return "source contains NUL"
    if len(src.encode("utf-8")) > _MAX_SOURCE_BYTES:
        return "source exceeds size limit"
    try:
        code = strip_lean(src)
    except LeanSourceError as exc:
        return f"lean source rejected: {exc}"
    idents = set(_IDENT.findall(code))
    blocked = sorted(idents & _BLOCKED_IDENTS)
    if blocked:
        return "disallowed lean token: " + ", ".join(blocked)
    if "#eval" in idents:
        return "disallowed lean token: #eval"
    dotted = set(_DOTTED.findall(code))
    if dotted & _BLOCKED_DOTTED or any(name.startswith("IO.Process.") for name in dotted):
        return "disallowed lean token: IO.Process"
    if not _DECL.search(code):
        return "no theorem or lemma"
    return None


def declared_names(src: str) -> list[str]:
    return _DECL.findall(strip_lean(src))


def parse_kernel_axioms(text: str) -> set[str]:
    found: set[str] = set()
    for match in _AXIOM_LINE.finditer(text):
        inner = match.group(1).strip()
        if not inner:
            continue
        for part in inner.split(","):
            name = part.strip().strip("'\"`")
            if name:
                found.add(name)
    return found


def _baseline_extra() -> dict[str, Any]:
    return {
        "baseline": {
            "repository": _BASELINE["repository"],
            "commit": PINNED_COMMIT,
            "axioms": AXIOM_COUNT,
            "tracked_sorries": TRACKED_SORRIES,
        }
    }


def default_probe(checkout: Path | None) -> dict[str, Any]:
    if checkout is None or not Path(checkout).is_dir():
        return {"ok": False, "reason": "lutar-lean checkout absent"}
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if head.returncode != 0:
        return {"ok": False, "reason": "lutar-lean checkout is not a git worktree"}
    sha = head.stdout.strip()
    if sha != PINNED_COMMIT:
        return {
            "ok": False,
            "reason": f"checkout HEAD {sha} is not pinned {PINNED_COMMIT}",
        }
    status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if status.returncode != 0:
        return {"ok": False, "reason": "lutar-lean checkout status unreadable"}
    if status.stdout.strip():
        return {"ok": False, "reason": "lutar-lean checkout is dirty"}
    return {"ok": True, "head": sha}


def lean_argv(lake: str, lean_file: Path) -> list[str]:
    return [lake, "env", "lean", str(lean_file)]


def _sandbox_env() -> dict[str, str]:
    env = {key: os.environ[key] for key in _KEEP_ENV if key in os.environ}
    env.setdefault("LANG", "C.UTF-8")
    return env


def default_lean_check(source: str, checkout: Path, timeout: float) -> dict[str, Any]:
    lake = shutil.which("lake")
    if shutil.which("lean") is None or lake is None:
        return {"ok": False, "absent": True, "axioms": [], "detail": "lean/lake absent"}
    names = declared_names(source)
    elaborated = source
    if not source.endswith("\n"):
        elaborated += "\n"
    for name in names:
        elaborated += f"#print axioms {name}\n"
    path: Path | None = None
    try:
        handle = tempfile.NamedTemporaryFile(
            prefix="rlvr-lean-", suffix=".lean", delete=False
        )
        path = Path(handle.name)
        handle.write(elaborated.encode("utf-8"))
        handle.close()
        os.chmod(path, 0o600)
        proc = subprocess.Popen(
            lean_argv(lake, path),
            cwd=str(checkout),
            env=_sandbox_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return {"ok": False, "timed_out": True, "axioms": [], "detail": "lean timed out"}
    finally:
        if path is not None:
            path.unlink(missing_ok=True)
    text = stdout.decode("utf-8", "replace") + "\n" + stderr.decode("utf-8", "replace")
    axioms = sorted(parse_kernel_axioms(text))
    if proc.returncode != 0:
        return {"ok": False, "axioms": axioms, "detail": "lean rejected the candidate"}
    return {"ok": True, "axioms": axioms, "detail": text}


def _clamp_timeout(timeout: float) -> float:
    try:
        value = float(timeout)
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT
    if value < 1:
        return 1.0
    if value > _MAX_TIMEOUT:
        return _MAX_TIMEOUT
    return value


def score_lean(
    completion: Any,
    *,
    checkout: Path | None = None,
    probe=None,
    runner=None,
    timeout: float = _DEFAULT_TIMEOUT,
    **_ignored: Any,
) -> VerifierOutcome:
    """Score one Lean candidate. Reward is 1.0 only when every gate passes."""
    extra = _baseline_extra()
    try:
        source = completion_text(completion)
    except TypeError:
        return failed(VERIFIER, completion, "unreadable completion", **extra)
    reason = static_block_reason(source)
    if reason:
        return failed(VERIFIER, completion, reason, **extra)
    checker = probe or default_probe
    try:
        pinned = checker(None if checkout is None else Path(checkout))
    except Exception as exc:  # noqa: BLE001
        return failed(VERIFIER, completion, f"checkout probe failed: {type(exc).__name__}", **extra)
    if not pinned.get("ok"):
        return failed(VERIFIER, completion, str(pinned.get("reason") or "checkout rejected"), **extra)
    elapsed = _clamp_timeout(timeout)
    execute = runner or default_lean_check
    try:
        result = execute(source, Path(checkout) if checkout is not None else Path("."), elapsed)
    except Exception as exc:  # noqa: BLE001
        return failed(VERIFIER, completion, f"lean runner failed: {type(exc).__name__}", **extra)
    if result.get("absent"):
        return failed(VERIFIER, completion, "lean/lake absent", **extra)
    if result.get("timed_out"):
        return failed(VERIFIER, completion, "lean timed out", **extra)
    axioms = set(result.get("axioms") or [])
    detail = str(result.get("detail") or "")
    if "sorryAx" in axioms or "sorryAx" in detail:
        return failed(VERIFIER, completion, "kernel depends on sorryAx", **extra)
    outside = sorted(axioms - ALLOWED_AXIOMS)
    if outside:
        return failed(
            VERIFIER,
            completion,
            "axiom outside pinned baseline: " + ", ".join(outside),
            **extra,
        )
    if not result.get("ok"):
        return failed(VERIFIER, completion, str(result.get("detail") or "lean rejected the candidate"), **extra)
    new_axioms = sorted(axioms - KERNEL_TRUST_BASE)
    return passed(
        VERIFIER,
        completion,
        "lean accepted the candidate with no new sorry and no new axiom",
        kernel_axioms=sorted(axioms),
        baseline_axioms_used=new_axioms,
        **extra,
    )


def lean_reward(completions: list[Any], **kwargs: Any) -> RewardBatch:
    """TRL GRPO reward. Returns floats; receipts are on ``.receipts``."""
    score_lean.verifier_name = VERIFIER  # type: ignore[attr-defined]
    return batch_reward(completions, score_lean, **kwargs)
