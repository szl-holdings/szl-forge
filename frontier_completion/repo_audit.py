"""Read-only inventory of explicit local SZL clones. No pull, reset, push or merge."""
from __future__ import annotations
import subprocess
from pathlib import Path
from .core import EvidenceError, digest, observation_shell, receipt, require
from .registry import OWNERS


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                text=True, timeout=15, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as exc:
        raise EvidenceError("git_read_unavailable") from exc


def local_audit(workspace: Path, source: str) -> dict:
    rows = []
    for name in OWNERS:
        root = workspace / name
        if not root.is_dir() or root.is_symlink():
            rows.append({"repository": "szl-holdings/" + name, "state": "MISSING_OR_SYMLINK"})
            continue
        try:
            # Do not print remote URLs: an operator may have embedded credentials.
            remote = _git(root, "remote", "get-url", "origin")
            expected_https = "https://github.com/szl-holdings/" + name
            expected_ssh = "git@github.com:szl-holdings/" + name
            require(remote.rstrip("/").removesuffix(".git") in {expected_https, expected_ssh}, "origin_not_canonical")
            revision = digest(_git(root, "rev-parse", "HEAD"), 40)
            dirty = bool(_git(root, "status", "--porcelain=v1", "--untracked-files=normal"))
            rows.append({"repository": "szl-holdings/" + name, "state": "DIRTY" if dirty else "CLEAN_LOCAL_CHECKOUT",
                         "head": revision, "remoteHeadFetchedThisRun": False,
                         "role": OWNERS[name]})
        except EvidenceError as exc:
            rows.append({"repository": "szl-holdings/" + name, "state": "UNAVAILABLE", "reasonCode": str(exc)})
    return receipt({**observation_shell("szl.local-checkout-audit.v1", source), "repositories": rows,
                    "networkActions": False, "remoteCurrentnessEstablished": False})
