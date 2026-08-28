#!/usr/bin/env python3
"""Guards for the local RTX 5050 Chaski bf16 LoRA recut (PR 59 kit)."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "chaski" / "train_chaski_bf16_5050.py"
CARD = ROOT / "chaski" / "HF_MODEL_CARD_5050.md"
README = ROOT / "chaski" / "README_5050.md"
EVAL = ROOT / "chaski" / "eval_chaski_5050.py"
SERVE = ROOT / "chaski" / "serve_chaski_5050.py"

sys.path.insert(0, str(ROOT / "chaski"))
import train_chaski_bf16_5050 as recut  # noqa: E402


class Chaski5050GuardTests(unittest.TestCase):
    def test_script_refuses_load_in_4bit_true(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--load-in-4bit"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(0, completed.returncode)
        blob = completed.stderr + completed.stdout
        self.assertIn("refuse", blob.lower())
        self.assertIn("load_in_4bit=True", blob)
        self.assertIn("QLoRA", blob)

    def test_helper_refuses_load_in_4bit_true(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            recut.assert_bf16_loader(True, True)
        self.assertIn("load_in_4bit=True", str(ctx.exception))
        recut.assert_bf16_loader(False, True)

    def test_script_refuses_push_to_chaski(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--push",
                "--hub-model-id",
                "SZLHOLDINGS/chaski",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(0, completed.returncode)
        blob = completed.stderr + completed.stdout
        self.assertIn("refuse", blob.lower())
        self.assertIn("SZLHOLDINGS/chaski", blob)
        self.assertIn("chaski-5050", blob)

    def test_helper_refuses_push_to_chaski(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            recut.assert_push_repo("SZLHOLDINGS/chaski")
        self.assertIn("never overwrite SZLHOLDINGS/chaski", str(ctx.exception))
        recut.assert_push_repo("SZLHOLDINGS/chaski-5050")

    def test_alpha_is_16_not_live_chaski_16_32(self) -> None:
        self.assertEqual(16, recut.LORA_R)
        self.assertEqual(16, recut.LORA_ALPHA)
        self.assertNotEqual(32, recut.LORA_ALPHA)
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("LORA_ALPHA = 16", text)
        self.assertNotIn("LORA_ALPHA = 32", text)

    def test_no_live_job_stamp_or_a11oy_dot_com(self) -> None:
        for path in (SCRIPT, CARD, README, EVAL, SERVE):
            blob = path.read_text(encoding="utf-8")
            self.assertNotIn("6a91bf10", blob)
            self.assertNotIn("a11oy.com", blob.lower())

    def test_house_card_atelier_lock(self) -> None:
        card = CARD.read_text(encoding="utf-8")
        self.assertIn("SZLHOLDINGS/chaski-5050", card)
        self.assertIn("Qwen/Qwen3.5-0.8B", card)
        self.assertIn("alpha=16", card)
        self.assertIn("REPORTED owner-metal", card)
        self.assertIn("Not MEASURED", card)
        self.assertIn("none-this-run", card)
        self.assertIn("publication_eligible: false", card)
        self.assertIn("QLoRA", card)
        self.assertIn("a-11-oy.com", card)
        self.assertIn("live", card.lower())
        self.assertIn("A11OY-MINI", card)
        self.assertIn("Khipu lab", card)
        self.assertIn("No Hub PUT", card)

    def test_github_hub_receipt_stamp(self) -> None:
        comments = (
            "# Hub SZLHOLDINGS/chaski-5050 commit c907ebe6e1fa900021be7b6fec19b38ec45be574",
            "# adapter_model.safetensors present",
            "# training_receipt.json: train_loss MEASURED 2.228136855544466, train_runtime 883.2224s, 3 epochs, 41 rows, seed 11, r=16 alpha=16, QLoRA false, job local-5050",
            "# dataset_sha256 ddc5594bfb1c78449ba40a263f5ac41d21c896c3c7ed7346341c7c080611a243",
            "# evals none-this-run. publication_eligible false. weights AVAILABLE",
            "# train_loss MEASURED is a train metric, not an eval. Training label REPORTED owner-metal until a signed receipt exists. Not 5/5.",
            "# SKU is NOT MEASURED (evals none-this-run, publication_eligible false). Do not stamp the model as MEASURED.",
        )
        text = SCRIPT.read_text(encoding="utf-8")
        for line in comments:
            self.assertIn(line + "\n", text)
        self.assertIn("LORA_R = 16", text)
        self.assertIn("LORA_ALPHA = 16", text)
        self.assertNotIn("LORA_ALPHA = 32", text)
        for path in (SCRIPT, CARD, README):
            blob = path.read_text(encoding="utf-8")
            self.assertIn("c907ebe6e1fa900021be7b6fec19b38ec45be574", blob)
            self.assertIn("2.228136855544466", blob)
            self.assertIn("adapter_model.safetensors", blob)
            self.assertIn(
                "ddc5594bfb1c78449ba40a263f5ac41d21c896c3c7ed7346341c7c080611a243",
                blob,
            )
            self.assertIn("AVAILABLE", blob)
            self.assertIn("local-5050", blob)
            self.assertIn("REPORTED owner-metal", blob)
            self.assertIn("none-this-run", blob)
            self.assertIn("883.2224", blob)
            self.assertIn("NOT MEASURED", blob)
            self.assertIn("train metric, not an eval", blob)
            self.assertRegex(blob, r"Do not stamp (the|this) model as MEASURED")
            self.assertNotIn("6a91bf10", blob)
            self.assertNotIn("a11oy.com", blob.lower())
            self.assertNotRegex(blob, r"(?i)sku is MEASURED")
        card = CARD.read_text(encoding="utf-8")
        self.assertIn("Not MEASURED", card)
        self.assertIn("Not 5/5", card)
        self.assertIn("A11OY-MINI", card)
        self.assertIn("live", card.lower())
        self.assertIn("Do not stamp this model as MEASURED", card)
        self.assertEqual(16, recut.LORA_R)
        self.assertEqual(16, recut.LORA_ALPHA)
        self.assertEqual(
            "c907ebe6e1fa900021be7b6fec19b38ec45be574", recut.HUB_COMMIT
        )
        self.assertEqual("AVAILABLE", recut.HUB_WEIGHTS)
        self.assertEqual("local-5050", recut.HUB_JOB)
        self.assertEqual("NOT MEASURED", recut.SKU_STATUS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
