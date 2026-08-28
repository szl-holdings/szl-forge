#!/usr/bin/env python3
"""KHIPU-R2 kit contract: separate SKU, refuse signed 1.5B, honest 2/6."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "khipu_r2"
TRAIN = KIT / "train_khipu_r2.py"
EVAL = KIT / "eval_khipu_r2.py"
SERVE = KIT / "serve_khipu_r2.py"
LAUNCH = KIT / "jobs" / "launch_khipu_r2_job.py"
README = KIT / "README.md"
FORBIDDEN = "SZLHOLDINGS/SZL-Khipu-1.5B"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def run(script: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


class KhipuR2KitTests(unittest.TestCase):
    def test_trainer_lives_in_separate_sku_not_khipu_dir(self) -> None:
        self.assertTrue(TRAIN.is_file())
        self.assertFalse((ROOT / "khipu" / "train_khipu_r2.py").exists())
        self.assertFalse((KIT / "train_khipu_abstain.py").exists())

    def test_status_receipt_is_roadmap_not_publication(self) -> None:
        completed = run(TRAIN, check=True)
        self.assertIn("ROADMAP", completed.stdout)
        self.assertIn("jobs=UNKNOWN", completed.stdout)
        self.assertIn("publication_eligible=false", completed.stdout)
        self.assertIn("2/6", completed.stdout)
        self.assertIn(BASE_MODEL.split("/")[-1], completed.stdout)
        receipt = json.loads(
            (KIT / "training_receipt.status.json").read_text(encoding="utf-8")
        )
        self.assertEqual(BASE_MODEL, receipt["base_model"])
        self.assertEqual(FORBIDDEN, receipt["does_not_overwrite"])
        self.assertEqual("UNKNOWN", receipt["jobs"])
        self.assertFalse(receipt["publication_eligible"])
        self.assertFalse(receipt["hub_put"])
        self.assertFalse(receipt["push_to_hub"])
        self.assertEqual("2/6", receipt["signed_original_abstain"])
        self.assertEqual(2, receipt["signed_original_abstain_correct"])
        self.assertEqual(6, receipt["signed_original_abstain_total"])
        self.assertEqual(11, receipt["seed"])
        self.assertEqual(4, receipt["ABSTAIN_OVERSAMPLE"])
        self.assertIn("v11", receipt["doctrine"])
        self.assertEqual("Conjecture 1", receipt["lambda"])

    def test_refuses_signed_khipu_hub(self) -> None:
        completed = run(TRAIN, "--hub", FORBIDDEN)
        self.assertNotEqual(0, completed.returncode)
        blob = completed.stderr + completed.stdout
        self.assertIn("refusing", blob.lower())
        self.assertIn(FORBIDDEN, blob)

    def test_eval_is_honest_about_two_of_six(self) -> None:
        completed = run(EVAL, check=True)
        self.assertIn("2/6", completed.stdout)
        self.assertIn("publication_eligible=false", completed.stdout)
        report = json.loads((KIT / "eval_report.json").read_text(encoding="utf-8"))
        self.assertEqual(BASE_MODEL, report["base_model"])
        self.assertEqual("2/6", report["signed_original_abstain"])
        self.assertEqual(2, report["signed_original_abstain_correct"])
        self.assertEqual(6, report["signed_original_abstain_total"])
        self.assertEqual("not-this-run", report["evals"])
        self.assertEqual("UNKNOWN", report["jobs"])
        self.assertFalse(report["publication_eligible"])
        self.assertEqual(FORBIDDEN, report["does_not_overwrite"])
        self.assertNotIn("3/6", json.dumps(report))

    def test_serve_check_discloses_base_model(self) -> None:
        completed = run(SERVE, "--check", check=True)
        payload = json.loads(completed.stdout)
        self.assertEqual(BASE_MODEL, payload["base_model"])
        self.assertEqual("UNAVAILABLE", payload["status"])
        self.assertEqual("UNKNOWN", payload["jobs"])
        self.assertFalse(payload["serve_pin"])
        self.assertFalse(payload["publication_eligible"])
        self.assertEqual(FORBIDDEN, payload["does_not_overwrite"])

    def test_jobs_launcher_refuses_to_fire(self) -> None:
        dry = run(LAUNCH, check=True)
        self.assertIn("UNKNOWN", dry.stdout)
        fired = run(LAUNCH, "--run-job")
        self.assertEqual(2, fired.returncode)
        self.assertIn("refusing", fired.stderr.lower())

    def test_no_hub_put_in_kit_source(self) -> None:
        forbidden_tokens = (
            "upload_folder",
            "upload_file",
            "HfApi(",
            "push_to_hub=True",
            "run_uv_job",
        )
        skip_names = {"test_khipu_r2.py"}
        for path in KIT.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".md"}:
                continue
            if path.name in skip_names or path.name == ".gitignore":
                continue
            text = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                self.assertNotIn(token, text, f"{path} contains {token}")

    def test_readme_is_roadmap_separate_sku(self) -> None:
        text = README.read_text(encoding="utf-8")
        self.assertIn("ROADMAP", text)
        self.assertIn("Separate SKU", text)
        self.assertIn("2/6", text)
        self.assertIn("publication_eligible", text)
        self.assertIn(FORBIDDEN, text)
        self.assertIn(BASE_MODEL, text)
        self.assertIn("UNKNOWN", text)
        self.assertIn("v11", text)

    def test_kit_does_not_import_forbidden_lanes(self) -> None:
        banned = (
            "KaLM",
            "MTEB",
            "Qantu",
            "Waman",
            "admitted_pairs",
            "admitted_triples",
            "5050",
            "A11OY-MINI",
            "A11OY_MINI",
        )
        skip_names = {"test_khipu_r2.py"}
        for path in KIT.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".md"}:
                continue
            if path.name in skip_names:
                continue
            text = path.read_text(encoding="utf-8")
            for token in banned:
                self.assertNotIn(token, text, f"{path} contains {token}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
