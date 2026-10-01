"""Fail-closed wrapper for the existing publication gate's four grep patterns.

Copyright 2026 SZL Holdings. SPDX-License-Identifier: Apache-2.0

A clear result is scoped pattern evidence, not absence of all possible secrets.
The recursive grep scope (including binary/hidden files, excluding .git folders)
and its symlink behavior are retained. No allowlist or suppression is added.
Raw matches and filenames never reach the Actions log. Missing tooling, scanner
errors and timeout cannot be accepted as a clean scan. This is not a publisher.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# [ ] matches exactly the same literal spaces without embedding a private-key
# marker in the scanner's own source. All four original detection rules remain.
PATTERN = (
    r"AIza[0-9A-Za-z_-]{35}|ghp_[A-Za-z0-9]{36,}|sk-[A-Za-z0-9]{20,}|"
    r"-----BEGIN[ ]PRIVATE[ ]KEY-----"
)
TIMEOUT_SECONDS = 300


def scan(root: Path) -> tuple[int, dict[str, Any]]:
    """Only grep status 1 means no match; errors are a separate blocking state."""
    report: dict[str, Any] = {
        "schema": "szl.publication-pattern-scan/v1",
        "scope": "RECURSIVE_GREP_FOUR_PATTERNS_EXCLUDING_DOT_GIT_DIRECTORIES",
        "status": "UNAVAILABLE",
        "reason": "SCAN_NOT_COMPLETED",
        "grepExitCode": None,
        "publicationAuthorized": False,
    }
    try:
        if root.is_symlink() or not root.is_dir():
            report["reason"] = "INVALID_SCAN_ROOT"
            return 2, report
        # Do not search an owner-controlled PATH or forward publishing tokens to
        # the scanner. GNU grep is already present on the native Ubuntu runner.
        executable = shutil.which("grep", path=os.defpath)
        if executable is None:
            report["reason"] = "SCANNER_UNAVAILABLE"
            return 2, report
        result = subprocess.run(
            [executable, "-rEl", "--exclude-dir=.git", "--", PATTERN, "."],
            cwd=root, env={"PATH": os.defpath, "LC_ALL": "C"},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=TIMEOUT_SECONDS, check=False,
        )
    except subprocess.TimeoutExpired:
        report["reason"] = "SCAN_TIMEOUT"
        return 2, report
    except OSError:
        report["reason"] = "SCANNER_IO_ERROR"
        return 2, report
    report["grepExitCode"] = result.returncode
    if result.returncode == 0:
        report.update(status="FINDING", reason="PATTERN_MATCH_CONTENT_REDACTED")
        return 1, report
    if result.returncode == 1:
        report.update(status="CLEAR", reason="NO_MATCH_WITHIN_DECLARED_SCOPE")
        return 0, report
    report["reason"] = "SCANNER_NON_SUCCESS"
    return 2, report


def main() -> int:
    """Scan this checkout only. No target paths or patterns are caller inputs."""
    if len(sys.argv) != 1:
        print('{"status":"UNAVAILABLE","reason":"ARGUMENTS_NOT_ACCEPTED"}')
        return 2
    code, report = scan(ROOT)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
