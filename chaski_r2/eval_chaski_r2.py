#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""CHASKI-R2 eval. Reuse parent named-N gates. Do not stamp MEASURED unrun.

Held-out files (eval-only, never ingested):
  chaski/gate/json_drafts.n5.jsonl
  chaski/gate/adversarial_refusals.n6.jsonl

publication_eligible false until MEASURED generate.
quality is UNAVAILABLE until the generate gate actually runs.
No ROADMAP parking. No Hub PUT. Separate SKU — not SZLHOLDINGS/chaski.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CANONICAL_BASE = "Qwen/Qwen3.5-0.8B"
HUB = "SZLHOLDINGS/chaski-r2"
FORBIDDEN_HUB = "SZLHOLDINGS/chaski"
FORBIDDEN_5050 = "SZLHOLDINGS/chaski-5050"
JSON_DRAFT_GATE = ROOT / "chaski" / "gate" / "json_drafts.n5.jsonl"
ADVERSARIAL_GATE = ROOT / "chaski" / "gate" / "adversarial_refusals.n6.jsonl"
ADAPTER_DIR = HERE / "chaski-r2-adapter"
PARENT_GATE_ARTIFACT = "SZLHOLDINGS/chaski"


def load_named_n_gate(path: Path, expected_kind: str) -> dict[str, Any]:
    raw_lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not raw_lines:
        raise SystemExit(f"[chaski-r2-eval] empty gate file: {path}")
    header = json.loads(raw_lines[0])
    rows = [json.loads(line) for line in raw_lines[1:]]
    n = header.get("n")
    if not isinstance(n, int) or n < 1:
        raise SystemExit(f"[chaski-r2-eval] {path.name} does not name a positive n")
    if header.get("kind") != expected_kind:
        raise SystemExit(
            f"[chaski-r2-eval] {path.name} kind {header.get('kind')!r} "
            f"!= {expected_kind!r}"
        )
    if header.get("artifact") != PARENT_GATE_ARTIFACT:
        raise SystemExit(
            f"[chaski-r2-eval] {path.name} must remain the parent named-N file "
            f"({PARENT_GATE_ARTIFACT})"
        )
    if header.get("base_model") != CANONICAL_BASE:
        raise SystemExit(
            f"[chaski-r2-eval] {path.name} base_model drifted from {CANONICAL_BASE}"
        )
    if len(rows) != n:
        raise SystemExit(
            f"[chaski-r2-eval] {path.name} names n={n} but has {len(rows)} rows"
        )
    if f".n{n}." not in path.name:
        raise SystemExit(f"[chaski-r2-eval] {path.name} does not carry n{n} in the name")
    if header.get("publication_eligible") is not False:
        raise SystemExit(
            f"[chaski-r2-eval] {path.name} publication_eligible must be false until run"
        )
    if header.get("gate_ran") is not False:
        raise SystemExit(f"[chaski-r2-eval] {path.name} gate_ran must be false until run")
    for row in rows:
        if row.get("n") != n:
            raise SystemExit(f"[chaski-r2-eval] {path.name} row n drifted from header")
        if "messages" not in row:
            raise SystemExit(f"[chaski-r2-eval] {path.name} row missing messages")
    return {"path": path, "header": header, "n": n, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Attempt generate. Without a local adapter this stays UNAVAILABLE.",
    )
    args = parser.parse_args()
    drafts = load_named_n_gate(JSON_DRAFT_GATE, "chaski-json-draft-gate")
    refusals = load_named_n_gate(
        ADVERSARIAL_GATE, "chaski-adversarial-refusal-gate"
    )
    local = ADAPTER_DIR.is_dir() and any(ADAPTER_DIR.glob("*.safetensors"))
    if args.run and not local:
        print("[chaski-r2-eval] --run requested but local adapter UNAVAILABLE")
        print("[chaski-r2-eval] not fabricating k/n; not stamping MEASURED")
    gate_ran = False
    report = {
        "kind": "szl-chaski-r2-eval-report",
        "artifact": HUB,
        "sku": "CHASKI-R2",
        "separate_sku": True,
        "does_not_overwrite": FORBIDDEN_HUB,
        "forbidden_5050": FORBIDDEN_5050,
        "canonical_base": CANONICAL_BASE,
        "base_model": CANONICAL_BASE,
        "json_draft_gate": "chaski/gate/json_drafts.n5.jsonl",
        "json_draft_n": drafts["n"],
        "adversarial_refusal_gate": "chaski/gate/adversarial_refusals.n6.jsonl",
        "adversarial_refusal_n": refusals["n"],
        "evals": "none-this-run",
        "gate_ran": gate_ran,
        "quality": "UNAVAILABLE",
        "jobs": "UNAVAILABLE",
        "train_loss_is_not_eval": True,
        "adapter": "LOCAL" if local else "UNAVAILABLE",
        "weights": "LOCAL" if local else "UNAVAILABLE",
        "publication_eligible": False,
        "hub_put": False,
        "khipu_lab_pin": False,
        "claim_boundary": (
            "Reused parent named-N files. The generate gate has not run. "
            "Do not stamp MEASURED from an unrun gate. Do not claim 5/5 or 6/6. "
            "quality=UNAVAILABLE. publication_eligible false until MEASURED "
            "generate. Separate SKU — not an overwrite of SZLHOLDINGS/chaski."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    path = HERE / "eval_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[chaski-r2-eval] evals=none-this-run quality=UNAVAILABLE")
    print(
        f"[chaski-r2-eval] json_draft_n={drafts['n']} "
        f"adversarial_refusal_n={refusals['n']} gate_ran=false"
    )
    print("[chaski-r2-eval] publication_eligible=false until MEASURED generate")
    print("[chaski-r2-eval] not stamping MEASURED from an unrun gate")
    print(f"[chaski-r2-eval] artifact={HUB} canonical_base={CANONICAL_BASE}")
    print(f"[chaski-r2-eval] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
