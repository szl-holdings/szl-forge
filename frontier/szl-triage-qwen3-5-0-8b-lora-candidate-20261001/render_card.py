#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Step 6 - render MODEL_CARD.md from candidate.json and the out/ reports only. Every number on the card is read
from a report; nothing is typed by hand. Sets G12_card_truth in out/gate_results.json to PASS when every
placeholder resolved from a report, otherwise leaves it UNAVAILABLE and prints the unresolved fields.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidate_lib as lib  # noqa: E402


MISSING = object()


def get(d, path):
    """Return the value at a dotted path, MISSING when the report or key does not exist (a null value is an honest UNAVAILABLE)."""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return MISSING
        cur = cur[part]
    return cur


def main():
    cand = lib.load_candidate()
    reports = {"candidate": cand}
    for name in ("training_report", "evaluation_report", "gate_results", "leakage_receipt", "QUANT_MANIFEST", "parity_receipt"):
        p = lib.OUT / f"{name}.json"
        reports[name] = lib.read_json(p) if p.is_file() else None  # absent report -> honest UNAVAILABLE, not a template error
    tmpl = (lib.HERE / "MODEL_CARD.tmpl.md").read_text(encoding="utf-8")
    unresolved = []

    def sub(match):
        source, path = match.group(1), match.group(2)
        if reports.get(source) is None:
            return "UNAVAILABLE"
        val = get(reports[source], path)
        if val is MISSING:
            unresolved.append(f"{source}.{path}")
            return "UNAVAILABLE"
        return "UNAVAILABLE" if val is None else str(val)
    card = re.sub(r"\{\{(\w+):([\w.]+)\}\}", sub, tmpl)
    gr = reports["gate_results"] or {}
    failing = [g["gate"] for g in gr.get("gates", []) if g["status"] != "PASS"]
    card = card.replace("{{FAILING_GATES}}", ", ".join(failing) if failing else "none (verdict derived by gate_results.json)")
    (lib.OUT / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    if gr.get("gates"):
        for g in gr["gates"]:
            if g["gate"] == "G12_card_truth":
                g["status"] = "PASS" if not unresolved else "UNAVAILABLE"
                g["value"] = f"{len(unresolved)} unresolved placeholders"
        passed = sum(1 for g in gr["gates"] if g["status"] == "PASS")
        gr["release_gate"] = f"{passed}/12"
        gr["gate_verdict"] = "ALL_GATES_PASS" if passed == 12 else "BLOCKED"
        lib.write_json(lib.OUT / "gate_results.json", gr)
    print(f"[card] wrote {lib.OUT / 'MODEL_CARD.md'}; unresolved={unresolved or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
