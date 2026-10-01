#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Step 2 - bind, verify and leakage-gate the curriculum. Writes out/curriculum_report.json and out/leakage_receipt.json.

Exit codes: 0 PASS, 3 curriculum UNBOUND, 4 leakage FAIL, 5 malformed rows above tolerance.
A prior study lost its evaluation to a held-out set that was not independent of training data; this gate runs
before every training attempt and is never tuned after results are seen (thresholds live in frozen gates.json).
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidate_lib as lib  # noqa: E402


def main():
    cand = lib.load_candidate()
    gates = lib.load_gates(cand)
    rows, bad, excluded = lib.load_splits(cand)
    counts = {k: len(v) for k, v in rows.items()}
    total = sum(counts.values())
    tolerance = float(gates["leakage"].get("max_malformed_fraction", 0.0))
    malformed_frac = len(bad) / max(1, total + len(bad))
    abstain = {k: sum(1 for r in v if lib.expected_abstain(r["expected"])) for k, v in rows.items()}
    families = {k: len({r["family_key"] for r in v if r["family_key"]}) for k, v in rows.items()}
    report = {"kind": "szl.frontier-curriculum-report/v1", "generated_utc": lib.now_utc(),
              "candidate_id": cand["candidate_id"], "binding": cand["training_data"],
              "rows": counts, "excluded_rows": excluded, "malformed_rows": len(bad), "malformed_examples": bad[:20],
              "abstain_rows": abstain, "distinct_family_keys": families,
              "schema_kinds": dict(Counter(("chat" if r["messages"] else "other") for v in rows.values() for r in v))}
    if rows["train"] == [] or (rows["dev"] == [] and rows["test"] == []):
        report["verdict"] = "FAIL_SPLITS"
        report["reason"] = "need a non-empty train split and at least one held split (dev or test)"
        lib.write_json(lib.OUT / "curriculum_report.json", lib.stringify_floats(report))
        print("[curriculum] FAIL_SPLITS", counts)
        return 5
    if malformed_frac > tolerance:
        report["verdict"] = "FAIL_MALFORMED"
        report["reason"] = f"{len(bad)} malformed rows ({malformed_frac:.3f}) exceed tolerance {tolerance}"
        lib.write_json(lib.OUT / "curriculum_report.json", lib.stringify_floats(report))
        print("[curriculum]", report["reason"])
        return 5
    held = rows["dev"] + rows["test"] + rows["adversarial"]
    leak = lib.leakage_report(rows["train"], held, gates["leakage"])
    leak["candidate_id"] = cand["candidate_id"]
    leak["curriculum_digests"] = {k: v.get("sha256") for k, v in (cand["training_data"].get("files") or {}).items()}
    lib.write_json(lib.OUT / "leakage_receipt.json", lib.stringify_floats(leak))
    report["leakage_verdict"] = leak["verdict"]
    report["verdict"] = "PASS" if leak["verdict"] == "PASS" else "FAIL_LEAKAGE"
    lib.write_json(lib.OUT / "curriculum_report.json", lib.stringify_floats(report))
    print(f"[curriculum] rows {counts} excluded {excluded} malformed {len(bad)} | leakage {leak['verdict']}")
    for name, c in leak["checks"].items():
        print(f"[leakage] {c.get('status') or ('PASS' if c['pass'] else 'FAIL')} {name} = {c['value']}")
    return 0 if report["verdict"] == "PASS" else 4


if __name__ == "__main__":
    sys.exit(main())
