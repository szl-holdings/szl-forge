"""Offline source and numerical-oracle tests for synthetic materials curriculum.

No GPU, no HF, no scientific data scraping and no official benchmark answers.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "training" / "materials_frontier_seed.py"
spec = importlib.util.spec_from_file_location("materials_frontier_seed", FILE)
assert spec and spec.loader
material = importlib.util.module_from_spec(spec)
spec.loader.exec_module(material)


class SourceAndChallengeBounds(unittest.TestCase):
    def test_five_challenges_and_official_urls_are_source_only(self):
        self.assertEqual(len(material.CHALLENGES), 5)
        self.assertEqual(set(material.DOMAIN_ORDER), set(material.CHALLENGES))
        for key, item in material.CHALLENGES.items():
            self.assertIn("discoveredmaterials.com/", item["url"], key)
            self.assertIn("https://", item["url"], key)
            self.assertTrue(item["missing"])
        self.assertEqual(len(material.BENCHMARK_CAPABILITIES), 7)
        self.assertEqual(sum(map(len, material.BENCHMARK_CAPABILITIES.values())), 10)

    def test_train_dev_diagnostic_population_and_domain_balance(self):
        expected = {"train": 240, "dev": 60, "diagnostic": 60}
        for split, total in expected.items():
            with self.subTest(split=split):
                items = material.rows(split)
                self.assertEqual(len(items), total)
                self.assertEqual(
                    {item["metadata"]["schema"] for item in items}, {material.SCHEMA}
                )
                tally = Counter(
                    (item["metadata"]["challenge"], item["metadata"]["task"])
                    for item in items
                )
                self.assertEqual(len(tally), 15)
                self.assertEqual(set(tally.values()), {total // 15})
                self.assertTrue(
                    all(
                        item["metadata"]["training_authority"] == "NONE"
                        and item["metadata"]["review_state"] == "CANDIDATE_ONLY"
                        and not item["metadata"]["independent_holdout"]
                        and not item["metadata"]["official_benchmark_item"]
                        for item in items
                    )
                )

    def test_source_inputs_and_text_are_not_benchmark_answer_memorization(self):
        dataset = material.rows("train")
        full = "\n".join(
            message["content"] for row in dataset for message in row["messages"]
        )
        for forbidden in (
            "GPT-6.2",
            "Claude Fable 5.2",
            "78.3%",
            "FrontierCode 2.0",
            "official secret test",
            "Grand Challenges solved",
        ):
            self.assertNotIn(forbidden, full)
        self.assertTrue(
            all(
                "Synthetic exercise" in item["messages"][1]["content"]
                for item in dataset
            )
        )
        self.assertTrue(
            all(
                not row["metadata"]["scientific_prediction_verified"] for row in dataset
            )
        )

    def test_parameter_and_input_disjointness_not_independent_science(self):
        ids = []
        prompts = []
        for split in material.SPLITS:
            for row in material.rows(split):
                ids.append(row["metadata"]["case_id"])
                prompts.append(
                    tuple((m["role"], m["content"]) for m in row["messages"][:-1])
                )
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(prompts), len(set(prompts)))


class PhysicalMathTests(unittest.TestCase):
    def test_d_wave_gap_antisymmetry_for_all_cases(self):
        cos = {"0": 1, "pi/2": 0, "pi": -1}
        for case in (1, 3, 7, 15, 101, 202):
            q, a, p = material._physics("cuprate", "calculate", case)
            kx, ky = p["kx"], p["ky"]
            expected = Fraction(p["delta0_mev"] * (cos[kx] - cos[ky]), 2)
            self.assertIn(f"Delta={material.fmt(expected)} meV", a)
            q2, a2, p2 = material._physics("cuprate", "invariant", case)
            self.assertIn(f"Delta(ky,kx)={material.fmt(-expected)} meV", a2)
            self.assertTrue(q2 and q)

    def test_positive_deal_grove_root_and_monotonicity(self):
        for case in (1, 3, 16, 101, 104, 204):
            q, a, p = material._physics("interface_electrical", "calculate", case)
            x = p["x_nm"]
            self.assertEqual(x * x + p["A_nm"] * x, p["B_nm2_per_min"] * p["time_min"])
            self.assertIn(f"x={x} nm", a)
            _, monotone, _ = material._physics(
                "interface_electrical", "invariant", case
            )
            self.assertIn("t2>t1", monotone)
            self.assertIn("nm^2/min", q)

    def test_hess_cycle_and_signed_reaction_delta(self):
        for case in (1, 16, 102, 203):
            q, a, p = material._physics("quantum_chemistry", "calculate", case)
            expected = p["EC_kj_mol"] + p["ED_kj_mol"] - p["EA_kj_mol"] - p["EB_kj_mol"]
            self.assertIn(f"DeltaE={expected} kJ/mol", a)
            _, cycle, _ = material._physics("quantum_chemistry", "invariant", case)
            self.assertIn(f"DeltaE_AC={p['EC_kj_mol'] - p['EA_kj_mol']}", cycle)
            self.assertIn("invented arithmetic", a)
            self.assertIn("kJ/mol", q)

    def test_driven_material_energy_conservation(self):
        for case in (2, 16, 101, 204):
            q, a, p = material._physics("driven_materials", "calculate", case)
            final = (
                p["E0_mev_cell"] + p["pump_work_mev_cell"] - p["dissipation_mev_cell"]
            )
            self.assertIn(f"Efinal={final} meV/unit-cell", a)
            _, inv, _ = material._physics("driven_materials", "invariant", case)
            self.assertIn("<= E0=", inv)
            self.assertIn("toy pulse-energy ledger", q.lower())

    def test_thermal_conductance_resistance_and_units(self):
        for case in (3, 6, 101, 104, 204):
            q, a, p = material._physics("heat_interfaces", "calculate", case)
            self.assertIn(
                f"DeltaT={material.fmt(Fraction(p['q_mw_m2'], p['G_mw_m2_k']))} K", a
            )
            _, inv, _ = material._physics("heat_interfaces", "invariant", case)
            self.assertIn("RK2/RK1=1/2", inv)
            self.assertIn("MW/(m^2 K)", q)

    def test_refusal_in_each_lane_and_no_fake_research_answer(self):
        for challenge in material.DOMAIN_ORDER:
            for case in (1, 101, 202):
                prompt, answer, _ = material._physics(challenge, "refusal", case)
                self.assertTrue(answer.startswith("HOLD."), challenge)
                self.assertIn("?", prompt)
                self.assertNotIn("solved", answer.lower()[:10])

    def test_out_of_bounds_and_unknown_oracle_refuse(self):
        for args in (
            ("foo", "calculate", 1),
            ("cuprate", "bad", 1),
            ("cuprate", "calculate", 251),
            ("cuprate", "calculate", True),
        ):
            with self.subTest(args=args):
                with self.assertRaises(material.CurriculumHold):
                    material._physics(*args)

    def test_fraction_formatting_exact(self):
        self.assertEqual(material.fmt(Fraction(1, 3)), "1/3")
        self.assertEqual(material.fmt(Fraction(-8, 4)), "-2")
        self.assertEqual(material.fmt(0), "0")


class StagingAndValidationTests(unittest.TestCase):
    def test_candidate_stage_reuses_canonical_forge_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "candidate"
            manifest = material.stage(target)
            self.assertEqual(manifest["schema"], material.MANIFEST_SCHEMA)
            self.assertEqual(manifest["training_authority"], "NONE")
            self.assertFalse(manifest["training_launch"])
            self.assertFalse(manifest["model_published"])
            self.assertFalse(manifest["contest_submission"])
            self.assertFalse(manifest["independent_holdout"])
            self.assertFalse(manifest["external_papers_or_benchmark_items_copied"])
            self.assertEqual(
                manifest["holdout_statement"],
                "DEV_AND_DIAGNOSTIC_REUSE_SAME_GENERATOR_NOT_INDEPENDENT",
            )
            self.assertEqual(manifest["sft_validation"]["records"], 240)
            self.assertEqual(manifest["sft_validation"]["cross_dataset_overlaps"], 0)
            self.assertEqual(
                [r["records"] for r in manifest["sft_validation"]["references"]],
                [60, 60],
            )
            for split in material.SPLITS:
                data = (target / (split + ".jsonl")).read_bytes()
                self.assertEqual(
                    hashlib.sha256(data).hexdigest(), manifest["files"][split]["sha256"]
                )
                self.assertEqual(len(data), manifest["files"][split]["bytes"])
                self.assertEqual(
                    len(data.splitlines()), manifest["files"][split]["rows"]
                )
            reloaded = json.loads((target / "manifest.json").read_bytes())
            self.assertEqual(reloaded, manifest)

    def test_existing_output_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "candidate"
            target.mkdir()
            marker = target / "keep.txt"
            marker.write_text("DO NOT TOUCH")
            with self.assertRaisesRegex(
                material.CurriculumHold, "NEW_OUTSIDE_REPO_DIRECTORY_REQUIRED"
            ):
                material.stage(target)
            self.assertEqual(marker.read_text(), "DO NOT TOUCH")

    def test_malformed_split_and_parent_refused(self):
        with self.assertRaises(material.CurriculumHold):
            material.rows("official_hidden_test")
        with self.assertRaises(material.CurriculumHold):
            material.stage(Path("relative_path"))
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing" / "candidate"
            with self.assertRaises(material.CurriculumHold):
                material.stage(missing)

    def test_forge_detects_single_tampered_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "candidate"
            material.stage(target)
            train = (target / "train.jsonl").read_bytes()
            dev = (target / "dev.jsonl").read_bytes().splitlines()
            # Replace one dev reference with exact existing training example:
            # as an independent validation failure, not a real test alteration.
            dev[0] = train.splitlines()[0]
            broken = target / "mutated_reference.jsonl"
            broken.write_bytes(b"\n".join(dev) + b"\n")
            result = material._validator()(
                str(target / "train.jsonl"),
                min_examples=32,
                disjoint_from=[str(broken)],
            )
            self.assertEqual(result["status"], "INVALID")
            self.assertGreaterEqual(result["cross_dataset_overlaps"], 1)

    def test_source_can_be_imported_without_ml_or_provider_libs(self):
        code = FILE.read_text(encoding="utf-8")
        for forbidden in (
            "import torch",
            "import unsloth",
            "import huggingface_hub",
            "HfApi(",
            "salad_api",
            "requests.get(",
            "urllib.request",
            "openai.OpenAI",
        ):
            self.assertNotIn(forbidden, code)
        self.assertTrue(code.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
