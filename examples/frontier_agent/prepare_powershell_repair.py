#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare a bounded repair mission for the historical PowerShell audit gap."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(repository, output):
    repository = repository.resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    for path in ("census/file_audit.py", "tests/test_file_audit.py"):
        if not (repository / path).is_file():
            raise ValueError("Repository must contain the szl-org-health audit sources")
    check = Path(__file__).resolve().with_name("check_powershell_audit.py")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / "evidence.json"
    evidence.write_text(json.dumps({
        "schema": "szl.repair-observation/v1", "source_revision": revision,
        "observation": "PowerShell ps1, psm1, and psd1 scripts need source inspection.",
        "acceptance": "Recognize extensions case-insensitively; preserve text-size, binary, documentation, and Git-object integrity checks.",
        "scope": "Software classification only; no production or model-quality claim.",
    }, indent=2) + "\n", encoding="utf-8")
    mission = {
        "schema": "szl.frontier-agent-mission/v1", "mode": "build",
        "repository": str(repository), "source_revision": revision, "timeout_seconds": 600,
        "objective": "Propose the smallest PowerShell source-inspection repair and focused regression tests. Return only the structured edit proposal; do not invoke tools or claim tests ran. Preserve all existing integrity checks. Tampering fixtures asserting a digest failure must have equal byte lengths so they exercise digest verification rather than the preceding size check.",
        "allowed_paths": ["census/file_audit.py", "tests/test_file_audit_powershell.py"],
        "source_paths": ["census/file_audit.py", "tests/test_file_audit.py"],
        "checks": [[sys.executable, "-I", str(check)],
                   [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_file_audit.py", "-q"],
                   [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_file_audit_powershell.py", "-q"]],
        "baseline_exit_codes": [1, 0, 0],
        "check_files": [{"path": str(check), "sha256": sha256(check)}],
        "evidence": [{"path": str(evidence), "sha256": sha256(evidence)}],
    }
    path = output / "mission.json"
    path.write_text(json.dumps(mission, indent=2) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(prepare(args.repository, args.output_dir))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(type(exc).__name__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
