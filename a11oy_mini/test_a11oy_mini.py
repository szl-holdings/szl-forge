#!/usr/bin/env python3
"""A11OY-MINI kit contracts: live Chaski parent, no Hub PUT, no 5050, no lab pin."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
README = HERE / "README.md"

sys.path.insert(0, str(HERE))
import convert_a11oy_mini_gguf as convert  # noqa: E402
import eval_a11oy_mini as evaluate  # noqa: E402


class A11oyMiniKitTests(unittest.TestCase):
    def setUp(self) -> None:
        for leftover in HERE.glob("*.gguf"):
            leftover.unlink()

    def test_parent_is_live_chaski_not_5050(self) -> None:
        self.assertEqual("SZLHOLDINGS/chaski", convert.PARENT)
        self.assertEqual("SZLHOLDINGS/chaski-5050", convert.FORBIDDEN_PARENT)
        self.assertEqual("SZLHOLDINGS/A11OY-MINI", convert.SKU)
        self.assertEqual("Qwen/Qwen3.5-0.8B", convert.CANONICAL_BASE)
        self.assertEqual(11, convert.SEED)
        self.assertIn("v11", convert.DOCTRINE)
        self.assertEqual(
            "model.safetensors-00001-of-00001.safetensors",
            convert.MERGED_SHARD_NAME,
        )

    def test_status_receipt_is_roadmap_without_gguf(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(HERE / "convert_a11oy_mini_gguf.py")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("ROADMAP", completed.stdout)
        self.assertIn("none-this-run", completed.stdout)
        self.assertIn("publication_eligible=false", completed.stdout)
        self.assertIn("hub_put=false", completed.stdout)
        self.assertIn("SZLHOLDINGS/chaski", completed.stdout)
        self.assertIn("chaski-5050", completed.stdout)
        self.assertNotIn("tok/s", completed.stdout.lower())
        receipt = json.loads((HERE / "conversion_receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(convert.PARENT, receipt["parent"])
        self.assertEqual(convert.FORBIDDEN_PARENT, receipt["forbidden_parent"])
        self.assertFalse(receipt["publication_eligible"])
        self.assertEqual("none-this-run", receipt["evals"])
        self.assertEqual("ROADMAP", receipt["quality"])
        self.assertFalse(receipt["hub_put"])
        self.assertFalse(receipt["gguf_exists"])
        self.assertFalse(receipt["bytes_measured"])
        self.assertFalse(receipt["khipu_lab_pin"])
        self.assertFalse(receipt["tok_s_claim"])
        self.assertFalse(receipt["third_llm"])
        self.assertFalse(receipt["new_train"])
        self.assertFalse(receipt["base_model_relation_quantized"])
        self.assertEqual(
            ["llama.cpp convert_hf_to_gguf.py --outtype f16", "llama-quantize Q4_K_M"],
            receipt["convert_path"],
        )

    def test_eval_without_gguf_is_none_this_run(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(HERE / "eval_a11oy_mini.py")],
            cwd=HERE,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("none-this-run", completed.stdout)
        self.assertIn("ROADMAP", completed.stdout)
        self.assertIn("UNAVAILABLE", completed.stdout)
        report = json.loads((HERE / "eval_report.json").read_text(encoding="utf-8"))
        self.assertEqual("none-this-run", report["evals"])
        self.assertEqual("none-this-run", report["parent_evals"])
        self.assertEqual("ROADMAP", report["quality"])
        self.assertFalse(report["publication_eligible"])
        self.assertEqual("UNAVAILABLE", report["gguf"])
        self.assertFalse(report["bytes_measured"])
        self.assertFalse(report["khipu_lab_pin"])
        self.assertIn("Do not claim 5/5", report["claim_boundary"])
        self.assertNotIn("evals\": \"5/5", json.dumps(report))

    def test_serve_check_pins_a11oy_mini_not_khipu_or_5050(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(HERE / "serve_a11oy_mini.py"), "--check"],
            cwd=HERE,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(convert.SKU, payload["serve_pin"])
        self.assertEqual(convert.PARENT, payload["parent"])
        self.assertFalse(payload["live_chaski_overwrite"])
        self.assertFalse(payload["parent_5050"])
        self.assertFalse(payload["khipu_lab_pin"])
        self.assertFalse(payload["inference_lab_pin"])
        self.assertFalse(payload["tok_s_claim"])
        self.assertEqual("UNAVAILABLE", payload["status"])
        self.assertEqual("ROADMAP", payload["quality"])

    def test_refuse_hub_put_and_live_overwrite_and_5050(self) -> None:
        with self.assertRaises(convert.ConvertError) as put:
            convert.refuse_hub_put("--upload")
        self.assertIn("Hub PUT", str(put.exception))
        with self.assertRaises(convert.ConvertError):
            convert.refuse_live_chaski_overwrite("SZLHOLDINGS/chaski")
        with self.assertRaises(convert.ConvertError):
            convert.refuse_5050_parent("SZLHOLDINGS/chaski-5050")
        with self.assertRaises(convert.ConvertError):
            convert.main(["--upload"])
        with self.assertRaises(convert.ConvertError):
            convert.main(["--hub-id", "SZLHOLDINGS/chaski"])
        with self.assertRaises(convert.ConvertError):
            convert.main(["--convert", "--merged", "/tmp/chaski-5050-merge"])

    def test_refuse_safetensors_ollama(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "safetensors-merge"
            folder.mkdir()
            (folder / "model.safetensors").write_bytes(b"not-a-model")
            with self.assertRaises(convert.ConvertError) as caught:
                convert.refuse_safetensors_ollama(folder)
            self.assertIn("BANNED", str(caught.exception))
            with self.assertRaises(convert.ConvertError):
                convert.quantize_q4_k_m(folder / "model.safetensors", folder / "out.gguf")
        convert.refuse_safetensors_ollama(HERE / convert.F16_NAME)

    def test_eval_hashes_local_gguf_as_bytes_measured_not_eval(self) -> None:
        fake = HERE / convert.Q4_NAME
        self.addCleanup(lambda: fake.exists() and fake.unlink())
        fake.write_bytes(b"GGUF-fake-not-a-model")
        report = evaluate.report_payload()
        self.assertEqual("none-this-run", report["evals"])
        self.assertEqual("ROADMAP", report["quality"])
        self.assertEqual("LOCAL", report["gguf"])
        self.assertTrue(report["bytes_measured"])
        self.assertEqual("MEASURED", report["bytes"]["label"])
        self.assertEqual(21, report["bytes"]["bytes"])
        self.assertFalse(report["publication_eligible"])

    def test_readme_roadmap_parent_and_locks(self) -> None:
        text = README.read_text(encoding="utf-8")
        self.assertIn("ROADMAP until a `.gguf` file exists on `SZLHOLDINGS/A11OY-MINI`", text)
        self.assertIn("SZLHOLDINGS/chaski", text)
        self.assertIn("Qwen/Qwen3.5-0.8B", text)
        self.assertIn("not a new train", text.lower())
        self.assertIn("publication_eligible: false", text)
        self.assertIn("none-this-run", text)
        self.assertIn("convert_hf_to_gguf.py", text)
        self.assertIn("Q4_K_M", text)
        self.assertIn("MEASURED 2026-07-12", text)
        self.assertIn("publication-eligible (INTI)", text)
        self.assertIn("Scripts only this week", text)
        self.assertIn("No GGUF bytes", text)
        self.assertIn("FORGE does not push after this PR exists", text)
        self.assertNotIn("FORGE pushes after the PR exists", text)
        self.assertNotIn("FORGE pushes after this PR exists", text)
        self.assertNotIn("a11oy.com", text)
        front = text.split("---", 2)[1]
        self.assertIsNone(re.search(r"^base_model_relation\s*:", front, flags=re.M))
        self.assertIn("base_model_relation: quantized", text)
        self.assertIn("Not a `base_model_relation: quantized`", text)
        self.assertNotIn("tok/s", text)
        self.assertIn("khipu_lab_pin: false", text)
        self.assertIn("chaski-5050", text)
        self.assertNotIn("https://a-11-oy.com", text)
        convert_src = (HERE / "convert_a11oy_mini_gguf.py").read_text(encoding="utf-8")
        self.assertNotIn("FORGE pushes after the PR exists", convert_src)
        self.assertNotIn("FORGE pushes after this PR exists", convert_src)
        self.assertIn("FORGE does not push after this PR exists", convert_src)
        self.assertIn("publication-eligible (INTI)", convert_src)
        self.assertIn("Scripts only this week", convert_src)

    def test_kit_does_not_touch_foreign_trees(self) -> None:
        for path in (
            ROOT / "chaski" / "train_chaski.py",
            ROOT / "chaski" / "train_chaski_bf16_5050.py",
            ROOT / "spaces" / "szl-model-inference-lab" / "app.py",
            ROOT / "khipu" / "serve_khipu.py",
        ):
            self.assertTrue(path.is_file())
        self.assertFalse(list((ROOT / "qantu").glob("train_*.py")))
        self.assertFalse(list((ROOT / "waman").glob("train_*.py")))

    def test_convert_f16_invokes_llama_cpp_python_converter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            merged = Path(tmp) / "chaski-merged"
            merged.mkdir()
            (merged / convert.MERGED_SHARD_NAME).write_bytes(b"x")
            (merged / "config.json").write_text("{}", encoding="utf-8")
            converter = Path(tmp) / "convert_hf_to_gguf.py"
            converter.write_text("# stub\n", encoding="utf-8")
            outfile = Path(tmp) / convert.F16_NAME

            def fake_run(cmd, cwd=None):  # noqa: ANN001
                self.assertEqual(str(converter), cmd[1])
                self.assertEqual(str(merged), cmd[2])
                self.assertIn("--outtype", cmd)
                self.assertIn("f16", cmd)
                outfile.write_bytes(b"GGUF")
                return mock.Mock(returncode=0)

            with mock.patch("convert_a11oy_mini_gguf.subprocess.run", fake_run):
                written = convert.convert_f16(merged, outfile, converter)
            self.assertTrue(written.is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
