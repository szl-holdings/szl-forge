#!/usr/bin/env python3
"""Fall 2026 forge alignment: Chaski job stamps, SKIP lanes, no Hub recut."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHASKI_TRAIN = ROOT / "chaski" / "train_chaski.py"

EXACT_COMMENTS = (
    "# 6a91b8ba FAILED CastError",
    "# 6a91b990 FAILED pyyaml 30s",
    "# 6a91ba00 COMPLETED receipt-only, weights UNAVAILABLE",
    "# 6a91bb7c RUNNING upload_folder — live",
)


class Fall2026AlignmentTests(unittest.TestCase):
    def test_exact_chaski_job_stamp_comments(self) -> None:
        text = CHASKI_TRAIN.read_text(encoding="utf-8")
        for line in EXACT_COMMENTS:
            self.assertIn(line + "\n", text)
        lines = text.splitlines()
        idx = [lines.index(line) for line in EXACT_COMMENTS]
        self.assertEqual(idx, list(range(idx[0], idx[0] + 4)))

    def test_attempt3_is_completed_not_running(self) -> None:
        skip_parts = {".git", "agent-forge", "__pycache__", "reports"}
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            if any(part in skip_parts for part in path.parts):
                continue
            if path.suffix not in {".py", ".md", ".json", ".html", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), 1):
                if "6a91ba00" in line:
                    stamped_running = (
                        "RUNNING" in line and "not RUNNING" not in line
                    )
                    self.assertFalse(
                        stamped_running,
                        f"{path}:{lineno} attempt-3 stamp is not COMPLETED",
                    )

    def test_only_attempt4_is_running_among_chaski_jobs(self) -> None:
        text = CHASKI_TRAIN.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "6a91b8ba" in line or "6a91b990" in line:
                self.assertNotIn("RUNNING", line)
            if "6a91bb7c" in line and line.strip().startswith("#"):
                self.assertIn("RUNNING", line)

    def test_chaski_base_is_qwen35_08b_not_qwen25(self) -> None:
        text = CHASKI_TRAIN.read_text(encoding="utf-8")
        self.assertIn('CANONICAL_BASE = "Qwen/Qwen3.5-0.8B"', text)
        for path in (ROOT / "chaski").rglob("*"):
            if not path.is_file():
                continue
            blob = path.read_text(encoding="utf-8", errors="replace")
            self.assertNotIn("Qwen2.5", blob)
            self.assertNotIn("Qwen3.8-max", blob)
        card = ROOT / "chaski" / "HF_MODEL_CARD.md"
        self.assertTrue(card.is_file())
        self.assertIn("base_model: Qwen/Qwen3.5-0.8B", card.read_text(encoding="utf-8"))

    def test_no_qantu_or_waman_trainers(self) -> None:
        self.assertFalse(list((ROOT / "qantu").glob("train_*.py")))
        self.assertFalse(list((ROOT / "waman").glob("train_*.py")))
        self.assertFalse((ROOT / "khipu" / "train_khipu_r2.py").exists())

    def test_chaski_status_and_estate_refuse(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(CHASKI_TRAIN)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("CUTTING", completed.stdout)
        self.assertIn("UNAVAILABLE", completed.stdout)
        self.assertIn("6a91ba00984507d9db4ea07f COMPLETED", completed.stdout)
        self.assertIn("6a91bb7c984507d9db4ea0a4 RUNNING", completed.stdout)
        receipt = json.loads(
            (ROOT / "chaski" / "training_receipt.status.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual("Qwen/Qwen3.5-0.8B", receipt["base_model"])
        self.assertEqual(1.782708187121898, receipt["training_loss"])
        self.assertEqual("UNAVAILABLE", receipt["weights"])
        self.assertFalse(receipt["publication_eligible"])
        jobs = {item["id"]: item["status"] for item in receipt["jobs"]}
        self.assertEqual("FAILED", jobs["6a91b8ba984507d9db4ea071"])
        self.assertEqual("FAILED", jobs["6a91b990984507d9db4ea077"])
        self.assertEqual("COMPLETED", jobs["6a91ba00984507d9db4ea07f"])
        self.assertEqual("RUNNING", jobs["6a91bb7c984507d9db4ea0a4"])
        with tempfile.TemporaryDirectory() as tmp:
            estate = Path(tmp) / "SZL_ESTATE_MANAGED.json"
            estate.write_text("{}", encoding="utf-8")
            refused = subprocess.run(
                [sys.executable, str(CHASKI_TRAIN), "--dataset-file", str(estate)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(0, refused.returncode)
            self.assertIn("SZL_ESTATE_MANAGED.json", refused.stderr + refused.stdout)

    def test_skip_and_scaffold_scripts_run(self) -> None:
        scripts = [
            ROOT / "qantu" / "skip_receipt.py",
            ROOT / "waman" / "skip_receipt.py",
            ROOT / "chakana" / "train_chakana.py",
            ROOT / "tinku" / "train_tinku.py",
            ROOT / "chaski" / "eval_chaski.py",
            ROOT / "chakana" / "eval_chakana.py",
            ROOT / "tinku" / "eval_tinku.py",
            ROOT / "chaski" / "serve_chaski.py",
            ROOT / "khipu" / "serve_khipu.py",
            ROOT / "receiptagent" / "serve_receiptagent.py",
            ROOT / "frontier" / "qwen35-receiptagent-v2" / "serve_candidate.py",
        ]
        for script in scripts:
            extra = ["--check"] if "serve" in script.name else []
            completed = subprocess.run(
                [sys.executable, str(script), *extra],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, completed.returncode, script)

    def test_jobs_launchers_refuse_to_fire(self) -> None:
        launchers = [
            ROOT / "chaski" / "jobs" / "launch_chaski_job.py",
            ROOT / "chakana" / "jobs" / "launch_chakana_job.py",
            ROOT / "tinku" / "jobs" / "launch_tinku_job.py",
        ]
        for launcher in launchers:
            dry = subprocess.run(
                [sys.executable, str(launcher)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, dry.returncode)
            fired = subprocess.run(
                [sys.executable, str(launcher), "--run-job"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(2, fired.returncode)
            self.assertIn("refusing", fired.stderr.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
