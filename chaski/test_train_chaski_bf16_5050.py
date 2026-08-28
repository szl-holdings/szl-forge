#!/usr/bin/env python3
"""Guards for the local RTX 5050 Chaski bf16 LoRA recut."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "chaski" / "train_chaski_bf16_5050.py"
SCHEMA = ROOT / "chaski" / "training_receipt.bf16_5050.schema.json"
CARD = ROOT / "chaski" / "HF_MODEL_CARD_5050.md"

sys.path.insert(0, str(ROOT / "chaski"))
import train_chaski_bf16_5050 as recut  # noqa: E402


class Chaski5050GuardTests(unittest.TestCase):
    def test_script_refuses_missing_dataset(self) -> None:
        missing = Path("/no/such/szl_dataset.jsonl")
        self.assertFalse(missing.exists())
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--dataset-file", str(missing)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(0, completed.returncode)
        blob = completed.stderr + completed.stdout
        self.assertIn("refuse", blob.lower())
        self.assertIn("missing dataset", blob.lower())

    def test_helper_refuses_missing_dataset(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            recut.load_doctrine_rows(Path("/no/such/szl_dataset.jsonl"))
        self.assertIn("missing dataset", str(ctx.exception).lower())

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

    def test_loads_admitted_jsonl_only(self) -> None:
        rows, digest = recut.load_doctrine_rows(ROOT / "szl_dataset.jsonl")
        self.assertEqual(41, len(rows))
        self.assertEqual(recut.ADMITTED_SHA256, digest)
        self.assertEqual("messages", next(iter(rows[0])))

    def test_refuses_estate_json_and_unaudited_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            estate = Path(tmp) / "SZL_ESTATE_MANAGED.json"
            estate.write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit) as ctx:
                recut.load_doctrine_rows(estate)
            self.assertIn("SZL_ESTATE_MANAGED.json", str(ctx.exception))
            fake = Path(tmp) / "szl_dataset.jsonl"
            fake.write_text(
                '{"messages": [{"role": "user", "content": "x"}, '
                '{"role": "assistant", "content": "y"}]}\n',
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit) as ctx:
                recut.load_doctrine_rows(fake)
            self.assertIn("unaudited", str(ctx.exception).lower())

    def test_source_pins_bf16_not_qlora(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("unsloth", text)
        self.assertIn("unsloth_zoo", text)
        self.assertIn("trackio", text)
        self.assertIn("load_in_4bit=LOAD_IN_4BIT", text)
        self.assertIn("load_in_16bit=LOAD_IN_16BIT", text)
        self.assertIn("LOAD_IN_4BIT = False", text)
        self.assertIn("LOAD_IN_16BIT = True", text)
        self.assertNotIn("load_in_4bit=True,", text)
        self.assertNotIn("load_in_4bit = True", text)
        self.assertIn("processing_class=tokenizer", text)
        self.assertIn('"max_length": MAX_SEQ_LEN', text)
        self.assertNotIn("OUROBOROS", text)

    def test_receipt_schema_and_sample(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        receipt = recut.build_receipt(
            train_loss=1.23,
            dataset_sha256=recut.ADMITTED_SHA256,
            training_rows=41,
        )
        self.assertEqual("none-this-run", receipt["evals"])
        self.assertIs(False, receipt["publication_eligible"])
        self.assertEqual(recut.TARGET_HARDWARE, receipt["hardware"])
        self.assertEqual("SZLHOLDINGS/szl-1-doctrine-sft", receipt["dataset"])
        self.assertEqual(recut.ADMITTED_SHA256, receipt["dataset_sha256"])
        self.assertFalse(receipt["load_in_4bit"])
        self.assertTrue(receipt["load_in_16bit"])
        try:
            import jsonschema
        except ImportError:
            for key in schema["required"]:
                self.assertIn(key, receipt)
            return
        jsonschema.validate(instance=receipt, schema=schema)

    def test_house_card_fashion(self) -> None:
        card = recut.house_model_card()
        recut.assert_house_card(card)
        self.assertEqual(card, CARD.read_text(encoding="utf-8"))
        self.assertIn("adapters", card.lower())
        self.assertIn("none-this-run", card.lower())
        self.assertIn("Not MEASURED", card)
        self.assertIn("cutting", recut._yaml_tags(card))
        self.assertNotIn("roadmap", recut._yaml_tags(card))
        self.assertNotIn("a11oy.com", card.lower())
        self.assertIn("a-11-oy.com", card)
        self.assertIn("Conjecture 1", card)
        self.assertNotIn("unsloth/unsloth", card.lower())

    def test_plan_invocation_does_not_train(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("QLoRA banned", completed.stdout)
        self.assertIn("none-this-run", completed.stdout)
        self.assertIn("SZLHOLDINGS/chaski-5050", completed.stdout)
        self.assertNotIn("5/5", completed.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
