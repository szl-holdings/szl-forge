#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline, fail-closed check of portfolio/flagship_training_map.v1.json.

The map says which repo and verifier each model-line flagship should train and
evaluate against. This check keeps it honest: no authority, no production or
trained claims, registration flags that agree with model_portfolio.json, and
szl-forge paths that exist in this checkout. It reads local files only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = ROOT / "portfolio" / "flagship_training_map.v1.json"
PORTFOLIO = ROOT / "portfolio" / "model_portfolio.json"
SCHEMA = "szl.forge.flagship-training-map/v1"
STATES = {"EXISTING", "MISSING", "BROKEN", "LOCAL_PATCH"}
KINDS = {"sft_data", "dpo_pairs", "rlvr_reward", "rlvr_adapter", "verifier", "eval",
         "eval_data", "config", "jobspec", "license"}
FORBIDDEN = re.compile(r"\b(PRODUCTION[_ -]?READY|FULLY[_ -]?TRAINED|PROMOTED|CERTIFIED)\b", re.I)
SHA = re.compile(r"[0-9a-f]{40}\Z")


def check(map_path: Path = DEFAULT_MAP, portfolio_path: Path = PORTFOLIO) -> list[str]:
    errors: list[str] = []
    data = json.loads(map_path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if data.get("authority") != "NONE":
        errors.append("top-level authority must be NONE")
    pins = data.get("source_pins") or {}
    for repo, sha in pins.items():
        if not isinstance(sha, str) or not SHA.match(sha):
            errors.append(f"pin for {repo} is not a 40-hex commit")
    registered = {a.get("repo_id") for a in json.loads(portfolio_path.read_text(encoding="utf-8"))["artifacts"]}
    seen = set()
    for f in data.get("flagships") or []:
        line = f.get("line")
        if line in seen:
            errors.append(f"duplicate line {line}")
        seen.add(line)
        if f.get("training_authority") != "NONE":
            errors.append(f"{line}: training_authority must be NONE")
        blob = json.dumps(f, ensure_ascii=False)
        if FORBIDDEN.search(blob.replace("unpromoted", "")):
            errors.append(f"{line}: contains a production/trained/promoted claim")
        expect = f.get("hub_repo") in registered
        if bool(f.get("forge_portfolio_registered")) != expect:
            errors.append(f"{line}: forge_portfolio_registered={f.get('forge_portfolio_registered')} but portfolio says {expect}")
        if not f.get("signals"):
            errors.append(f"{line}: no signals recorded")
        for sig in f.get("signals") or []:
            if sig.get("state") not in STATES:
                errors.append(f"{line}: bad state {sig.get('state')}")
            if sig.get("kind") not in KINDS:
                errors.append(f"{line}: bad kind {sig.get('kind')}")
            repo = sig.get("repo", "")
            if not repo.startswith("(") and repo not in pins:
                errors.append(f"{line}: repo {repo} has no source pin")
            if repo == "szl-forge" and sig.get("state") in {"EXISTING", "LOCAL_PATCH", "BROKEN"}:
                if not list(ROOT.glob(sig["path"])):
                    errors.append(f"{line}: szl-forge path missing: {sig['path']}")
    if not seen:
        errors.append("no flagships")
    return errors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", type=Path, default=DEFAULT_MAP)
    args = ap.parse_args(argv)
    errors = check(args.map)
    data = json.loads(args.map.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for f in data["flagships"]:
        for sig in f["signals"]:
            counts[sig["state"]] = counts.get(sig["state"], 0) + 1
    print(json.dumps({"ok": not errors, "errors": errors, "flagships": len(data["flagships"]),
                      "signal_states": counts, "authority": "NONE"}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
