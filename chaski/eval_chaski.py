#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Chaski eval — named-N JSON-draft + refusal gate wiring. Gate has not run.

Parent SZLHOLDINGS/chaski. Base Qwen/Qwen3.5-0.8B.
Held-out files name N inside the file. publication_eligible false until run.
No Hub PUT. No job fire. Train loss is not an eval. Hub files are not an eval.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen3.5-0.8B"
HUB = "SZLHOLDINGS/chaski"
JSON_DRAFT_GATE = HERE / "gate" / "json_drafts.n5.jsonl"
ADVERSARIAL_GATE = HERE / "gate" / "adversarial_refusals.n6.jsonl"
ADAPTER_DIR = HERE / "chaski-adapter"

# Job stamps live in train_chaski.py. This eval does not restamp them.
# Do not use eval_chaski_5050.py. Do not reuse train ouroboros rows.


def load_named_n_gate(path: Path, expected_kind: str) -> dict[str, Any]:
    raw_lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not raw_lines:
        raise SystemExit(f"[chaski-eval] empty gate file: {path}")
    header = json.loads(raw_lines[0])
    rows = [json.loads(line) for line in raw_lines[1:]]
    n = header.get("n")
    if not isinstance(n, int) or n < 1:
        raise SystemExit(f"[chaski-eval] {path.name} does not name a positive n")
    if header.get("kind") != expected_kind:
        raise SystemExit(
            f"[chaski-eval] {path.name} kind {header.get('kind')!r} "
            f"!= {expected_kind!r}"
        )
    if header.get("artifact") != HUB:
        raise SystemExit(f"[chaski-eval] {path.name} artifact drifted from {HUB}")
    if header.get("base_model") != BASE_MODEL:
        raise SystemExit(
            f"[chaski-eval] {path.name} base_model drifted from {BASE_MODEL}"
        )
    if len(rows) != n:
        raise SystemExit(
            f"[chaski-eval] {path.name} names n={n} but has {len(rows)} rows"
        )
    if f".n{n}." not in path.name:
        raise SystemExit(f"[chaski-eval] {path.name} does not carry n{n} in the name")
    if header.get("publication_eligible") is not False:
        raise SystemExit(
            f"[chaski-eval] {path.name} publication_eligible must be false until run"
        )
    if header.get("gate_ran") is not False:
        raise SystemExit(f"[chaski-eval] {path.name} gate_ran must be false until run")
    for row in rows:
        if row.get("n") != n:
            raise SystemExit(f"[chaski-eval] {path.name} row n drifted from header")
        if "messages" not in row:
            raise SystemExit(f"[chaski-eval] {path.name} row missing messages")
    return {"path": path, "header": header, "n": n, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Attempt generate. Without a local adapter this stays not-run.",
    )
    args = parser.parse_args()
    drafts = load_named_n_gate(JSON_DRAFT_GATE, "chaski-json-draft-gate")
    refusals = load_named_n_gate(
        ADVERSARIAL_GATE, "chaski-adversarial-refusal-gate"
    )
    local = ADAPTER_DIR.is_dir() and any(ADAPTER_DIR.glob("*.safetensors"))
    if args.run and not local:
        print("[chaski-eval] --run requested but local adapter UNAVAILABLE")
        print("[chaski-eval] not fabricating k/n; publication_eligible stays false")
    gate_ran = False
    report = {
        "kind": "szl-chaski-eval-report",
        "artifact": HUB,
        "base_model": BASE_MODEL,
        "json_draft_gate": "chaski/gate/json_drafts.n5.jsonl",
        "json_draft_n": drafts["n"],
        "adversarial_refusal_gate": "chaski/gate/adversarial_refusals.n6.jsonl",
        "adversarial_refusal_n": refusals["n"],
        "evals": "none-this-run",
        "gate_ran": gate_ran,
        "quality": "UNKNOWN",
        "train_loss_is_not_eval": True,
        "adapter": "present-on-hub-as-of-2026-08-28T17:08Z",
        "weights": "present-on-hub-as-of-2026-08-28T17:08Z",
        "hub_tensors": [
            "adapter_model.safetensors",
            "adapter_config.json",
            "model.safetensors-00001-of-00001.safetensors",
        ],
        "hub_tensors_observed_at": "2026-08-28T17:08Z",
        "publication_eligible": False,
        "claim_boundary": (
            "Named-N held-out files are wired. The generate gate has not run. "
            "Do not claim a passing JSON-draft or refusal score. "
            "Hub adapter files exist as of 2026-08-28T17:08Z. "
            "Evals remain none-this-run. Train loss is not an eval. "
            "publication_eligible false until run."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    path = HERE / "eval_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[chaski-eval] evals=none-this-run quality=UNKNOWN")
    print(
        f"[chaski-eval] json_draft_n={drafts['n']} "
        f"adversarial_refusal_n={refusals['n']} gate_ran=false"
    )
    print("[chaski-eval] publication_eligible=false until run")
    print(f"[chaski-eval] artifact={HUB} base_model={BASE_MODEL}")
    print(f"[chaski-eval] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
