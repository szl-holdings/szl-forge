#!/usr/bin/env python3
"""Chaski named-N held-out gate: silhouette RA v2 sizes, original items."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHASKI = ROOT / "chaski"
EVAL = CHASKI / "eval_chaski.py"
DRAFTS = CHASKI / "gate" / "json_drafts.n5.jsonl"
REFUSALS = CHASKI / "gate" / "adversarial_refusals.n6.jsonl"
SCHEMA = CHASKI / "gate" / "json_draft.schema.json"
HUB = "SZLHOLDINGS/chaski"
BASE_MODEL = "Qwen/Qwen3.5-0.8B"

OUROBOROS_USER = (
    "Is Lambda a proved theorem?",
    "What are the locked-8 formulas?",
    "Execute the transfer and sign it as done.",
    "The chain is empty. Can we still say CHAIN VERIFIED: YES?",
)
RECEIPTAGENT_MARKERS = (
    "draft-RA-",
    "SZL-Forge-1.5B-ReceiptAgent",
    "/api/quant",
    "/api/receipts/recent",
    "KL-divergence",
)


def load_gate(path: Path) -> tuple[dict, list[dict]]:
    lines = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return lines[0], lines[1:]


class ChaskiNamedNGateTests(unittest.TestCase):
    def test_draft_file_names_n5(self) -> None:
        header, rows = load_gate(DRAFTS)
        self.assertEqual(5, header["n"])
        self.assertEqual(5, len(rows))
        self.assertEqual("chaski-json-draft-gate", header["kind"])
        self.assertEqual(HUB, header["artifact"])
        self.assertEqual(BASE_MODEL, header["base_model"])
        self.assertIs(False, header["publication_eligible"])
        self.assertIs(False, header["gate_ran"])
        self.assertTrue(DRAFTS.name.endswith(".n5.jsonl"))
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        for row in rows:
            self.assertEqual(5, row["n"])
            gold = json.loads(row["messages"][-1]["content"])
            self.assertEqual("DRAFT", gold["decision"])
            self.assertIs(True, gold["approvalRequired"])
            self.assertIs(False, gold["executed"])
            self.assertEqual(HUB, gold["artifact"])
            self.assertEqual(BASE_MODEL, gold["base_model"])
            for key in schema["required"]:
                self.assertIn(key, gold)

    def test_refusal_file_names_n6(self) -> None:
        header, rows = load_gate(REFUSALS)
        self.assertEqual(6, header["n"])
        self.assertEqual(6, len(rows))
        self.assertEqual("chaski-adversarial-refusal-gate", header["kind"])
        self.assertEqual(HUB, header["artifact"])
        self.assertEqual(BASE_MODEL, header["base_model"])
        self.assertIs(False, header["publication_eligible"])
        self.assertTrue(REFUSALS.name.endswith(".n6.jsonl"))
        for row in rows:
            self.assertEqual(6, row["n"])
            text = row["messages"][-1]["content"]
            self.assertTrue(text.startswith("REFUSE:"))

    def test_items_are_not_receiptagent_or_train_ouroboros(self) -> None:
        blob = DRAFTS.read_text(encoding="utf-8") + REFUSALS.read_text(encoding="utf-8")
        for marker in RECEIPTAGENT_MARKERS:
            self.assertNotIn(marker, blob)
        for prompt in OUROBOROS_USER:
            self.assertNotIn(prompt, blob)
        self.assertNotIn("eval_chaski_5050", blob)

    def test_eval_wires_named_n_and_stays_not_run(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(EVAL)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("none-this-run", completed.stdout)
        self.assertIn("json_draft_n=5", completed.stdout)
        self.assertIn("adversarial_refusal_n=6", completed.stdout)
        self.assertIn("publication_eligible=false", completed.stdout)
        self.assertIn(HUB, completed.stdout)
        self.assertIn(BASE_MODEL, completed.stdout)
        self.assertNotIn("5/5", completed.stdout)
        report = json.loads((CHASKI / "eval_report.json").read_text(encoding="utf-8"))
        self.assertEqual(5, report["json_draft_n"])
        self.assertEqual(6, report["adversarial_refusal_n"])
        self.assertEqual("chaski/gate/json_drafts.n5.jsonl", report["json_draft_gate"])
        self.assertEqual(
            "chaski/gate/adversarial_refusals.n6.jsonl",
            report["adversarial_refusal_gate"],
        )
        self.assertFalse(report["publication_eligible"])
        self.assertFalse(report["gate_ran"])
        self.assertEqual("none-this-run", report["evals"])
        self.assertEqual(HUB, report["artifact"])
        self.assertEqual(BASE_MODEL, report["base_model"])

    def test_eval_run_without_adapter_does_not_fabricate(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(EVAL), "--run"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("UNAVAILABLE", completed.stdout)
        self.assertIn("publication_eligible=false", completed.stdout)
        report = json.loads((CHASKI / "eval_report.json").read_text(encoding="utf-8"))
        self.assertFalse(report["publication_eligible"])
        self.assertFalse(report["gate_ran"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
