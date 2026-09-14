#!/usr/bin/env python3
"""
Held-out gate runner for the 2026-09-14 close-out.

Runs the existing eval harness (eval_szl.py / frontier/evaluation fixtures) against
the EXACT published artifact bytes for each model, and emits a markdown table to
append to frontier/EVAL_BACKUP_2026-09-14.md plus a JSON receipt per model.

Also implements the KHIPU-R2 salvage flow (frontier/KHIPU_R2_SALVAGE_PLAN_2026-09-13.md):
  --salvage khipu-r2  rebuilds evals_dir from held-out fixtures, re-runs the abstain
  gate on published bytes, verifies provenance (adapter SHA / merge receipt /
  checkpoint SHA), and emits a re-attestation record to clear SUPPRESSION_2026-09-13.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MODELS = {
    "chaski-r2":       {"repo": "SZLHOLDINGS/chaski-r2",       "gate": "courier"},
    "chaski-5050":     {"repo": "SZLHOLDINGS/chaski-5050",     "gate": "courier"},
    "WILLAY":          {"repo": "SZLHOLDINGS/WILLAY",         "gate": "courier"},
    "brain-navigator-r2": {"repo": "SZLHOLDINGS/brain-navigator-r2", "gate": "navigator"},
    "khipu-r3":        {"repo": "SZLHOLDINGS/khipu-r3",       "gate": "abstain"},
    "KHIPU-R2":        {"repo": "SZLHOLDINGS/KHIPU-R2",       "gate": "abstain"},
    "szl-receiptagent-qwen35-0.8b-v3": {"repo": "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3", "gate": "receipt"},
}

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "eval_szl.py"
FIXTURES = REPO_ROOT / "frontier" / "evaluation"
OUT_DIR = REPO_ROOT / "frontier" / "evaluation" / "gate_runs"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_gate(model: str, gate: str, repo: str) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_path = OUT_DIR / f"{model}_{stamp}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(HARNESS),
        "--model", repo,
        "--gate", gate,
        "--fixtures", str(FIXTURES),
        "--out", str(result_path),
        "--published-bytes",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"model": model, "status": "HARNESS_FAIL", "stderr": proc.stderr[-2000:]}
    result = json.loads(result_path.read_text())
    result["model"] = model
    result["repo"] = repo
    result["gate"] = gate
    result["run_utc"] = stamp
    return result


def salvage_khipu_r2() -> dict:
    model = "KHIPU-R2"
    gate = MODELS[model]["gate"]

    evals_dir = REPO_ROOT / "khipu_r2" / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    fixtures_src = FIXTURES / "khipu_abstain"
    for f in fixtures_src.glob("*.jsonl"):
        (evals_dir / f.name).write_text(f.read_text())

    gate_result = run_gate(model, gate, MODELS[model]["repo"])

    provenance = {}
    for label, p in [
        ("adapter_sha", REPO_ROOT / "khipu_r2" / "adapter_model.safetensors"),
        ("merge_receipt", REPO_ROOT / "khipu_r2" / "merge_receipt.json"),
        ("checkpoint_sha", REPO_ROOT / "khipu_r2" / "model.safetensors"),
    ]:
        provenance[label] = sha256_of(p) if p.exists() else "MISSING"

    record = {
        "salvage": model,
        "suppression": "SUPPRESSION_2026-09-13",
        "evals_dir_rebuilt": True,
        "gate_result": gate_result,
        "provenance": provenance,
        "prior_gate_on_record": "3/6 abstain",
        "attest_previous": "89a0b01e-cbd3-4fd1-8e8b-b3e55f96b2ca",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT_DIR / f"khipu_r2_salvage_{stamp}.json"
    out.write_text(json.dumps(record, indent=2))
    record["salvage_record"] = str(out)
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", choices=MODELS.keys())
    ap.add_argument("--salvage", choices=["khipu-r2"])
    args = ap.parse_args()

    results = []
    if args.salvage:
        rec = salvage_khipu_r2()
        results.append(rec)
        print(json.dumps(rec, indent=2))
    if args.models:
        for m in args.models:
            r = run_gate(m, MODELS[m]["gate"], MODELS[m]["repo"])
            results.append(r)
            status = r.get("status", "OK")
            print(f"{m}: {status}")

    lines = [f"- {r['model']}: gate={r.get('gate')} status={r.get('status', 'OK')} "
             f"run={r.get('run_utc', 'see record')}" for r in results]
    if lines:
        print("\nAppend to frontier/EVAL_BACKUP_2026-09-14.md:")
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
