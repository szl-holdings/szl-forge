"""Seal an exact-source HF core-stack evaluation receipt in CI."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from tools.hf_core_stack_guard import DENIED_AUTHORITY, EXPECTED


def git_revision(ref: str) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", ref], text=True, timeout=10).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    head = git_revision("HEAD")
    if head is None:
        raise SystemExit("cannot identify checked-out source revision")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("sourceRepository") != "szl-holdings/szl-forge":
        raise SystemExit("report repository mismatch")
    if report.get("sourceRevision") != head:
        raise SystemExit("report source revision mismatch")
    if report.get("productionDisposition") != "HOLD":
        raise SystemExit("evaluation must remain on HOLD")

    gate = report.get("existingCoreStackGate")
    if not isinstance(gate, dict) or gate.get("disposition") != "HOLD":
        raise SystemExit("core-stack gate evidence is missing or promoted")

    receipt: dict[str, Any] = {
        "schema": "szl.forge.hf-core-stack-receipt.v1",
        "repository": "szl-holdings/szl-forge",
        "sourceRevision": head,
        "rollbackTarget": git_revision("HEAD^") ,
        "observedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "workflow": {
            "runId": os.environ.get("GITHUB_RUN_ID"),
            "event": os.environ.get("GITHUB_EVENT_NAME"),
            "ref": os.environ.get("GITHUB_REF"),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "exactSources": EXPECTED,
        "runtimeStatus": report.get("runtimeStatus"),
        "installationStatus": report.get("installationStatus"),
        "measuredChecks": report.get("checks", {}),
        "coreStackGate": gate,
        "reportSha256": sha256_file(args.report),
        "authority": dict(DENIED_AUTHORITY),
        "productionDisposition": "HOLD",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema": receipt["schema"], "sourceRevision": head, "runtimeStatus": receipt["runtimeStatus"], "productionDisposition": "HOLD"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


