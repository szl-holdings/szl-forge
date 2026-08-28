#!/usr/bin/env python3
"""KHIPU-R2 kit contract: separate SKU, live Hub 3/6, refuse signed 1.5B."""

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
HUB_JOB_ID = "6a91bf11984507d9db4ea104"
STALE_PROFILE = "SZL-Khipu-1.5B-BrainNavigator"


def run(script: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def kit_text_files():
    skip_names = {"test_khipu_r2.py"}
    for path in KIT.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".md"}:
            continue
        if path.name in skip_names or path.name == ".gitignore":
            continue
        yield path, path.read_text(encoding="utf-8")


class KhipuR2KitTests(unittest.TestCase):
    def test_trainer_lives_in_separate_sku_not_khipu_dir(self) -> None:
        self.assertTrue(TRAIN.is_file())
        self.assertFalse((ROOT / "khipu" / "train_khipu_r2.py").exists())
        self.assertFalse((KIT / "train_khipu_abstain.py").exists())

    def test_status_receipt_stamps_live_hub_not_empty(self) -> None:
        completed = run(TRAIN, check=True)
        self.assertIn("COMPLETED", completed.stdout)
        self.assertIn("AVAILABLE", completed.stdout)
        self.assertIn("147.8MB", completed.stdout)
        self.assertIn("3/6", completed.stdout)
        self.assertIn("not a pass", completed.stdout)
        self.assertIn("jobs=UNKNOWN", completed.stdout)
        self.assertIn("evals=not-this-run", completed.stdout)
        self.assertIn("publication_eligible=false", completed.stdout)
        self.assertIn(BASE_MODEL.split("/")[-1], completed.stdout)
        self.assertNotIn("card=ROADMAP", completed.stdout)
        receipt = json.loads(
            (KIT / "training_receipt.status.json").read_text(encoding="utf-8")
        )
        self.assertEqual(BASE_MODEL, receipt["base_model"])
        self.assertEqual(FORBIDDEN, receipt["does_not_overwrite"])
        self.assertEqual(HUB_JOB_ID, receipt["hub_job_id"])
        self.assertEqual("COMPLETED", receipt["hub_job_status"])
        self.assertEqual("AVAILABLE", receipt["hub_adapter"])
        self.assertEqual("147.8MB", receipt["hub_adapter_size"])
        self.assertEqual("3/6", receipt["hub_abstain"])
        self.assertEqual(3, receipt["hub_abstain_correct"])
        self.assertEqual(6, receipt["hub_abstain_total"])
        self.assertEqual("MEASURED", receipt["hub_abstain_label"])
        self.assertFalse(receipt["hub_abstain_pass"])
        self.assertEqual("UNKNOWN", receipt["jobs"])
        self.assertEqual("this-kit", receipt["jobs_scope"])
        self.assertEqual("not-this-run", receipt["evals"])
        self.assertFalse(receipt["publication_eligible"])
        self.assertFalse(receipt["hub_put"])
        self.assertFalse(receipt["push_to_hub"])
        self.assertEqual(11, receipt["seed"])
        self.assertEqual(4, receipt["ABSTAIN_OVERSAMPLE"])
        self.assertIn("v11", receipt["doctrine"])
        self.assertEqual("Conjecture 1", receipt["lambda"])
        self.assertEqual("signed Khipu GGUF", receipt["lab"])
        self.assertNotIn("capabilityProfile", receipt)
        self.assertNotIn("signed_original_abstain", receipt)

    def test_refuses_signed_khipu_hub(self) -> None:
        completed = run(TRAIN, "--hub", FORBIDDEN)
        self.assertNotEqual(0, completed.returncode)
        blob = completed.stderr + completed.stdout
        self.assertIn("refusing", blob.lower())
        self.assertIn(FORBIDDEN, blob)

    def test_eval_stamps_hub_three_of_six_and_not_this_run(self) -> None:
        completed = run(EVAL, check=True)
        self.assertIn("3/6", completed.stdout)
        self.assertIn("not-this-run", completed.stdout)
        self.assertIn("publication_eligible=false", completed.stdout)
        report = json.loads((KIT / "eval_report.json").read_text(encoding="utf-8"))
        self.assertEqual(BASE_MODEL, report["base_model"])
        self.assertEqual("3/6", report["hub_abstain"])
        self.assertEqual(3, report["hub_abstain_correct"])
        self.assertEqual(6, report["hub_abstain_total"])
        self.assertEqual("MEASURED", report["hub_abstain_label"])
        self.assertFalse(report["hub_abstain_pass"])
        self.assertEqual("not-this-run", report["evals"])
        self.assertEqual("UNKNOWN", report["jobs"])
        self.assertEqual("this-kit", report["jobs_scope"])
        self.assertFalse(report["publication_eligible"])
        self.assertEqual(FORBIDDEN, report["does_not_overwrite"])
        self.assertEqual("COMPLETED", report["hub_job_status"])
        self.assertEqual("AVAILABLE", report["hub_adapter"])
        self.assertNotIn("signed_original_abstain", report)
        self.assertNotIn("capabilityProfile", report)

    def test_serve_check_discloses_base_model_and_hub_adapter(self) -> None:
        completed = run(SERVE, "--check", check=True)
        payload = json.loads(completed.stdout)
        self.assertEqual(BASE_MODEL, payload["base_model"])
        self.assertEqual("UNAVAILABLE", payload["status"])
        self.assertEqual("AVAILABLE", payload["hub_adapter"])
        self.assertEqual("147.8MB", payload["hub_adapter_size"])
        self.assertEqual("COMPLETED", payload["hub_job_status"])
        self.assertEqual("UNKNOWN", payload["jobs"])
        self.assertFalse(payload["serve_pin"])
        self.assertFalse(payload["inference_lab_pin"])
        self.assertEqual("signed Khipu GGUF", payload["lab"])
        self.assertFalse(payload["publication_eligible"])
        self.assertEqual(FORBIDDEN, payload["does_not_overwrite"])

    def test_jobs_launcher_refuses_to_fire(self) -> None:
        dry = run(LAUNCH, check=True)
        self.assertIn("UNKNOWN", dry.stdout)
        self.assertIn("COMPLETED", dry.stdout)
        fired = run(LAUNCH, "--run-job")
        self.assertEqual(2, fired.returncode)
        self.assertIn("refusing", fired.stderr.lower())
        self.assertIn("COMPLETED", fired.stderr)

    def test_no_hub_put_in_kit_source(self) -> None:
        forbidden_tokens = (
            "upload_folder",
            "upload_file",
            "HfApi(",
            "push_to_hub=True",
            "run_uv_job",
        )
        for path, text in kit_text_files():
            for token in forbidden_tokens:
                self.assertNotIn(token, text, f"{path} contains {token}")

    def test_readme_leads_with_live_hub_not_roadmap_copy(self) -> None:
        text = README.read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        first = next(line for line in lines if not line.startswith("#"))
        self.assertNotIn("ROADMAP", first)
        self.assertIn("not empty", first.lower())
        self.assertIn("COMPLETED", text)
        self.assertIn("147.8MB", text)
        self.assertIn("AVAILABLE", text)
        self.assertIn("3/6", text)
        self.assertIn("UNKNOWN", text)
        self.assertIn("not-this-run", text)
        self.assertIn("publication_eligible", text)
        self.assertIn(FORBIDDEN, text)
        self.assertIn(BASE_MODEL, text)
        self.assertIn("v11", text)
        self.assertIn("signed Khipu GGUF", text)
        self.assertNotIn(STALE_PROFILE, text)

    def test_stale_capability_profile_is_absent(self) -> None:
        for path, text in kit_text_files():
            self.assertNotIn(STALE_PROFILE, text, f"{path} still uses stale profile")
            self.assertNotIn("capabilityProfile", text, f"{path} still uses capabilityProfile")

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
        for path, text in kit_text_files():
            for token in banned:
                self.assertNotIn(token, text, f"{path} contains {token}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
