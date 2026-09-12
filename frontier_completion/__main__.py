"""Portable CLI. Default `plan` and local audits are offline; `scan` needs --live.

This module never schedules a model, consumes private data, pushes Git, changes a
Space, or sends notifications. A scan emits only material candidate notices to
stdout; incomplete acquisition is recorded in its output file and exit status.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import sys
from pathlib import Path
from .core import EvidenceError, load_json, observation_shell, receipt, write_json
from .registry import CANDIDATES, OWNERS, RELEASES
from .repo_audit import local_audit
from .scan import PublicClient, observe_release, scan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "scan", "releases", "repo-audit"), nargs="?", default="plan")
    parser.add_argument("--source", help="exact handoff/runner Git SHA; never default to a historical SHA")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args(argv)
    if args.action == "plan":
        print(json.dumps({"candidates": [dataclasses.asdict(c) for c in CANDIDATES],
                          "releases": RELEASES, "owners": OWNERS,
                          "execution": "NONE", "productionDisposition": "HOLD"}, indent=2))
        return 0
    if not args.source or not args.output:
        parser.error("non-plan actions require --source and --output")
    if args.action in {"scan", "releases"} and not args.live:
        parser.error("public network reads require explicit --live")
    if args.previous and args.previous.resolve() == args.output.resolve():
        parser.error("preserve the previous snapshot; output must be a new path")
    try:
        if args.action == "repo-audit":
            if args.workspace is None:
                parser.error("repo-audit requires --workspace")
            report = local_audit(args.workspace, args.source)
            write_json(args.output, report)
            return 0 if all(row["state"] == "CLEAN_LOCAL_CHECKOUT" for row in report["repositories"]) else 2
        client = PublicClient()
        if args.action == "scan":
            prior = load_json(args.previous) if args.previous else None
            report = scan(CANDIDATES, args.source, client, prior)
            write_json(args.output, report)
            for item in report["notifications"]:
                print(json.dumps(item))
            return 0 if report["completeMaterialityCoverage"] else 2
        results, failures = [], []
        for repo, tag, source in RELEASES:
            try:
                results.append(observe_release(repo, tag, source, client))
            except EvidenceError as exc:
                failures.append({"repository": repo, "tag": tag, "reasonCode": str(exc)})
        report = receipt({**observation_shell("szl.handoff.release-source-audit.v1", args.source),
                          "releases": results, "failures": failures,
                          "state": "COMPLETE" if not failures else "PARTIAL"})
        write_json(args.output, report)
        return 0 if not failures else 2
    except (EvidenceError, OSError, ValueError, OverflowError) as exc:
        print("Handoff operation failed closed: " + type(exc).__name__, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
