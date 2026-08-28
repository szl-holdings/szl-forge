#!/usr/bin/env python3
"""Chakana kit contract: Qwen3-Embedding lock, UNKNOWN jobs/eval, unsigned stubs."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "chakana"
sys.path.insert(0, str(KIT))
TRAIN = KIT / "train_chakana.py"
EVAL = KIT / "eval_chakana.py"
SERVE = KIT / "serve_chakana.py"
LAUNCH = KIT / "jobs" / "launch_chakana_job.py"
SIGN = KIT / "sign_receipt.py"
README = KIT / "README.md"
CARD = KIT / "HF_MODEL_CARD.md"
TRAIN_STUB = KIT / "training_receipt.stub.json"
EVAL_STUB = KIT / "eval_receipt.stub.json"
BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
ALT_BASE = "BAAI/bge-m3"
HUB = "SZLHOLDINGS/chakana"
PASTED_MTEB = ("64.34", "70.58", "72.32")


def run(script: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


class ChakanaKitTests(unittest.TestCase):
    def test_status_receipt_is_roadmap_unknown(self) -> None:
        completed = run(TRAIN, check=True)
        self.assertIn("ROADMAP", completed.stdout)
        self.assertIn("jobs=UNKNOWN", completed.stdout)
        self.assertIn("publication_eligible=false", completed.stdout)
        self.assertIn(BASE_MODEL.split("/")[-1], completed.stdout)
        receipt = json.loads(
            (KIT / "training_receipt.status.json").read_text(encoding="utf-8")
        )
        self.assertEqual(BASE_MODEL, receipt["base_model"])
        self.assertEqual(ALT_BASE, receipt["alt_base_model"])
        self.assertEqual(HUB, receipt["artifact"])
        self.assertEqual("UNKNOWN", receipt["jobs"])
        self.assertEqual("UNKNOWN", receipt["evals"])
        self.assertEqual("UNKNOWN", receipt["ndcg10"])
        self.assertEqual("UNAVAILABLE", receipt["weights"])
        self.assertFalse(receipt["publication_eligible"])
        self.assertFalse(receipt["hub_put"])
        self.assertFalse(receipt["push_to_hub"])
        self.assertFalse(receipt["mteb_pasted"])
        self.assertEqual(11, receipt["seed"])
        self.assertEqual([1024, 512, 256], receipt["matryoshka_dims"])
        self.assertIn("v11", receipt["doctrine"])
        self.assertEqual("Conjecture 1", receipt["lambda"])
        self.assertEqual("NINA (FORGE-class)", receipt["lane"])
        self.assertEqual("Stephen Lutar", receipt["owner"])
        self.assertTrue(receipt["not_a11oy_chakana_wiring"])
        self.assertFalse(receipt["signed"])

    def test_refuses_kalm_and_embeddinggemma(self) -> None:
        for banned in (
            "HIT-TMG/KaLM-embedding-mini",
            "google/embeddinggemma-300m",
            "Qwen/Qwen3-0.6B",
        ):
            completed = run(TRAIN, "--base-model", banned)
            self.assertNotEqual(0, completed.returncode, banned)
            blob = completed.stderr + completed.stdout
            self.assertIn("refuse", blob.lower())

    def test_admits_bge_m3(self) -> None:
        completed = run(TRAIN, "--base-model", ALT_BASE, check=True)
        self.assertIn(ALT_BASE.split("/")[-1], completed.stdout)
        receipt = json.loads(
            (KIT / "training_receipt.status.json").read_text(encoding="utf-8")
        )
        self.assertEqual(ALT_BASE, receipt["base_model"])

    def test_refuses_mteb_beir_and_non_pairs_lakes(self) -> None:
        cases = (
            ("mteb/cqadupstack-reddit",),
            ("BeIR/msmarco",),
            ("SZLHOLDINGS/szl-lake",),
            ("SZLHOLDINGS/rag-corpus-v1",),
        )
        for (dataset_id,) in cases:
            completed = run(TRAIN, "--dataset", dataset_id)
            self.assertNotEqual(0, completed.returncode, dataset_id)
            blob = (completed.stderr + completed.stdout).lower()
            self.assertIn("refuse", blob)

    def test_refuses_estate_json_as_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            estate = Path(tmp) / "SZL_ESTATE_MANAGED.json"
            estate.write_text("{}", encoding="utf-8")
            completed = run(TRAIN, "--dataset-file", str(estate), "--train")
            self.assertNotEqual(0, completed.returncode)
            self.assertIn("SZL_ESTATE_MANAGED.json", completed.stderr + completed.stdout)

    def test_train_skip_without_pairs_is_honest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty_pairs.jsonl"
            empty.write_text("", encoding="utf-8")
            completed = run(TRAIN, "--train", "--dataset-file", str(empty), check=True)
            self.assertIn("SKIP-NO-ADMITTED-PAIRS", completed.stdout)
            receipt = json.loads((KIT / "training_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual("UNKNOWN", receipt["evals"])
            self.assertEqual("UNKNOWN", receipt["ndcg10"])
            self.assertFalse(receipt["publication_eligible"])
            self.assertEqual("UNAVAILABLE", receipt["weights"])

    def test_eval_stays_unknown_without_encoder(self) -> None:
        completed = run(EVAL, check=True)
        self.assertIn("UNKNOWN", completed.stdout)
        self.assertIn("publication_eligible=false", completed.stdout)
        report = json.loads((KIT / "eval_report.json").read_text(encoding="utf-8"))
        self.assertEqual(BASE_MODEL, report["base_model"])
        self.assertEqual("UNKNOWN", report["evals"])
        self.assertIsNone(report["ndcg10"])
        self.assertEqual("UNKNOWN", report["jobs"])
        self.assertFalse(report["publication_eligible"])
        self.assertFalse(report["mteb_pasted"])
        ran = run(EVAL, "--run", check=True)
        self.assertIn("UNKNOWN", ran.stdout)
        ran_report = json.loads((KIT / "eval_report.json").read_text(encoding="utf-8"))
        self.assertEqual("UNKNOWN", ran_report["evals"])
        self.assertIsNone(ran_report["ndcg10"])

    def test_ndcg_math_is_deterministic_without_claiming_a_score(self) -> None:
        import eval_chakana as evaluate

        perfect = evaluate.ndcg_at_k([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], k=10)
        self.assertEqual(1.0, perfect)
        zero = evaluate.ndcg_at_k([0.0, 0.0, 0.0], [1.0], k=10)
        self.assertEqual(0.0, zero)

    def test_serve_check_discloses_base_model(self) -> None:
        completed = run(SERVE, "--check", check=True)
        payload = json.loads(completed.stdout)
        self.assertEqual(BASE_MODEL, payload["base_model"])
        self.assertEqual("UNAVAILABLE", payload["status"])
        self.assertEqual("UNKNOWN", payload["jobs"])
        self.assertFalse(payload["serve_pin"])
        self.assertFalse(payload["publication_eligible"])
        self.assertEqual([256, 512, 1024], payload["matryoshka_dims"])
        self.assertTrue(payload["not_a11oy_chakana_wiring"])

    def test_jobs_launcher_refuses_to_fire(self) -> None:
        dry = run(LAUNCH, check=True)
        self.assertIn("UNKNOWN", dry.stdout)
        self.assertIn("HF_TOKEN", dry.stdout)
        fired = run(LAUNCH, "--run-job")
        self.assertEqual(2, fired.returncode)
        self.assertIn("refusing", fired.stderr.lower())
        payload = json.loads(run(LAUNCH, "--json", check=True).stdout)
        self.assertEqual("UNKNOWN", payload["jobs"])
        self.assertFalse(payload["submitted"])
        self.assertIn("--secrets", payload["command"])
        self.assertIn("HF_TOKEN", payload["command"])
        self.assertNotIn("job_id", payload)

    def test_unsigned_stubs_have_no_signature_and_unknown_eval(self) -> None:
        completed = run(SIGN, "stub", "--check", check=True)
        self.assertIn("STUB-UNSIGNED", completed.stdout)
        for path in (TRAIN_STUB, EVAL_STUB):
            wrapper = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(wrapper["signed"])
            self.assertIsNone(wrapper["signatureBase64"])
            self.assertEqual("UNKNOWN", wrapper["jobs"])
            self.assertEqual("UNKNOWN", wrapper["payload"]["ndcg10"])
            self.assertEqual("UNKNOWN", wrapper["payload"]["evals"])
            self.assertFalse(wrapper["payload"]["publication_eligible"])
            self.assertEqual(BASE_MODEL, wrapper["payload"]["baseModel"])
            self.assertNotIn("6a91", json.dumps(wrapper))

    def test_sign_without_key_refuses_to_invent_a_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = KIT / "training_receipt.stub.json"
            out = Path(tmp) / "signed.json"
            import os

            env = os.environ.copy()
            env["A11OY_OWNER_KEY_PEM"] = str(Path(tmp) / "missing.pem")
            env_run = subprocess.run(
                [sys.executable, str(SIGN), "sign", str(payload), str(out)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertNotEqual(0, env_run.returncode)
            self.assertIn("refusing", (env_run.stderr + env_run.stdout).lower())
            self.assertFalse(out.exists())

    def test_no_hub_put_in_kit_source(self) -> None:
        forbidden_tokens = (
            "upload_folder",
            "upload_file",
            "HfApi(",
            "push_to_hub=True",
            "run_uv_job",
        )
        skip_names = {"test_chakana.py"}
        for path in KIT.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".md", ".json"}:
                continue
            if path.name in skip_names or path.name == ".gitignore":
                continue
            text = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                self.assertNotIn(token, text, f"{path} contains {token}")

    def test_card_yaml_and_docs_lock(self) -> None:
        card = CARD.read_text(encoding="utf-8")
        readme = README.read_text(encoding="utf-8")
        self.assertTrue(card.startswith("---\n"))
        self.assertIn("license: apache-2.0", card)
        self.assertIn("pipeline_tag: feature-extraction", card)
        self.assertIn("library_name: sentence-transformers", card)
        self.assertIn("base_model: Qwen/Qwen3-Embedding-0.6B", card)
        self.assertIn("doctrine: v11-LOCKED", card)
        self.assertIn("jobs: UNKNOWN", card)
        for blob in (card, readme):
            self.assertIn("NINA", blob)
            self.assertIn("Stephen Lutar", blob)
            self.assertIn("a11oy CHAKANA wiring", blob)
            self.assertIn("UNKNOWN", blob)
            self.assertIn("publication_eligible", blob)
            self.assertIn(BASE_MODEL, blob)
            self.assertIn("Conjecture 1", blob)
            for number in PASTED_MTEB:
                self.assertNotIn(number, blob)

    def test_kit_does_not_invent_eval_scores_anywhere(self) -> None:
        skip_names = {
            "test_chakana.py",
            "training_receipt.status.json",
            "training_receipt.json",
            "eval_report.json",
        }
        for path in KIT.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".md", ".json"}:
                continue
            if path.name in skip_names:
                continue
            text = path.read_text(encoding="utf-8")
            for number in PASTED_MTEB:
                self.assertNotIn(number, text, f"{path} pasted {number}")

    def test_does_not_touch_sibling_lanes(self) -> None:
        self.assertTrue(TRAIN.is_file())
        self.assertFalse((ROOT / "khipu" / "train_chakana.py").exists())
        self.assertFalse((ROOT / "chaski" / "train_chakana.py").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
