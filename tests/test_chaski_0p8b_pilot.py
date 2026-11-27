"""Chaski 0.8B pilot: prompt admission, unlaunchable manifests, CPU dry-run.

The dry-run trains tiny random models when the train-image stack is installed
and skips only when that import is absent. Source gates always run. Nothing
here launches a job or writes to the Hub.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "training" / "chaski_0p8b_pilot"
ADMITTED = {
    "szl_dataset.jsonl",
    "chaski_r2/train.jsonl",
    "chaski_r4/train.jsonl",
}


def _pilot(name: str):
    return importlib.import_module(f"training.chaski_0p8b_pilot.{name}")


class PromptAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = _pilot("prompts").select_prompts(ROOT)

    def test_heldout_hashes_match_the_r4_manifest(self):
        constants = _pilot("constants")
        for relative, expected in constants.HOLDOUT_SHA256.items():
            digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(digest, expected)

    def test_admitted_prompts_exclude_heldout_and_unknown(self):
        report = self.report
        by_path = {row["path"]: row for row in report["inventory"]}
        self.assertEqual(
            {path for path, row in by_path.items() if row["status"] == "ADMITTED"},
            ADMITTED,
        )
        self.assertEqual(report["admitted_prompts"], 91)
        self.assertEqual(report["heldout_user_prompts"], 11)
        self.assertFalse(report["launch_authorized"])
        self.assertFalse(report["operator_admission"])
        for row in report["inventory"]:
            if row["status"] == "ADMITTED":
                self.assertEqual(row["license"], "Apache-2.0")
                self.assertEqual(row["dropped_heldout"], 0)
                self.assertFalse(row["operator_admission"])
            else:
                self.assertIn(row["status"], {"HELD_OUT", "UNKNOWN"})
                self.assertIsNone(row["license"])
        self.assertEqual(by_path["chaski/gate/json_drafts.n5.jsonl"]["status"], "HELD_OUT")
        self.assertEqual(
            by_path["chaski_r4/train.pre-fix-backup.jsonl"]["status"], "UNKNOWN"
        )
        self.assertEqual(by_path["szl_dataset_5050_merged.jsonl"]["status"], "UNKNOWN")
        users = []
        for prompt in report["prompts"]:
            self.assertIn(prompt["source"], ADMITTED)
            roles = [message["role"] for message in prompt["messages"]]
            self.assertNotIn("assistant", roles)
            self.assertIn("user", roles)
            users.append(
                "\n".join(
                    message["content"]
                    for message in prompt["messages"]
                    if message["role"] == "user"
                )
            )
        self.assertEqual(len(users), len(set(users)))
        self.assertEqual(len(users), 91)

    def test_forbidden_output_repos(self):
        refuse = _pilot("constants").refuse_forbidden_repo
        refuse("SZLHOLDINGS/chaski-0p8b-pilot")
        refuse("SZLHOLDINGS/chaski-0p8b-pilot-sft")
        for repo in ("SZLHOLDINGS/chaski", "szlholdings/chaski-r4", "SZLHOLDINGS/Chaski"):
            with self.assertRaises(RuntimeError):
                refuse(repo)


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = _pilot("manifest").build_all(ROOT)

    def test_committed_manifests_match_the_builder(self):
        directory = PILOT / "manifests"
        expected = {
            "phase1-teacher.json": self.docs["phase1"],
            "phase2-student.json": self.docs["phase2"],
            "pool-queue.json": self.docs["queue"],
        }
        for name, document in expected.items():
            on_disk = json.loads((directory / name).read_text(encoding="utf-8"))
            self.assertEqual(on_disk, json.loads(json.dumps(document, sort_keys=True)))

    def test_pool_shape_is_refused_and_inside_the_budget(self):
        manifest = _pilot("manifest")
        constants = _pilot("constants")
        queue = self.docs["queue"]
        self.assertEqual(queue["schema"], "szl.gpu-pool.queue/v1")
        self.assertFalse(queue["launch_authorized"])
        self.assertEqual(queue["gpu_h_hard_total"], 6)
        self.assertLessEqual(queue["gpu_h_hard_total"], constants.GPU_H_BUDGET)
        self.assertEqual(queue["caps"], {"per_run_usd": 100, "per_day_usd": 500, "per_month_usd": 5000})
        self.assertEqual(queue["allowed_providers"], ["salad", "massed"])
        self.assertEqual(
            [run["run_id"] for run in queue["runs"]],
            ["chaski-0p8b-pilot-teacher", "chaski-0p8b-pilot-student"],
        )
        for run in queue["runs"]:
            blockers = manifest.pool_blockers(run)
            self.assertIn("RUN_JOBSPEC_MISSING", blockers)
            self.assertIn("RUN_BINDING_MISSING", blockers)
            self.assertNotIn("PROVIDER_NOT_ALLOWED", blockers)
            self.assertNotIn("CAPS_MISMATCH", blockers)

    def test_launch_contract_rejects_both_waves(self):
        contract = _pilot_launch()
        for key in ("phase1", "phase2"):
            decision = contract.evaluate(
                self.docs[key]["wave"],
                config_dir=PILOT,
            )
            self.assertFalse(decision["ok"])
            self.assertFalse(decision["launch_executed"])
            self.assertFalse(decision["training_started"])
            self.assertFalse(decision["hub_write"])
            self.assertFalse(decision["teacher_invoked"])
            self.assertIn("LAUNCH_PIN_NOT_IMMUTABLE", decision["codes"])


class SourceRefusalTests(unittest.TestCase):
    def test_entry_points_do_not_load_or_upload(self):
        for name in ("teacher.py", "student.py", "dry_run.py", "prompts.py"):
            tree = ast.parse((PILOT / name).read_text(encoding="utf-8"))
            calls = [
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            ]
            attrs = [
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ]
            self.assertNotIn("from_pretrained", attrs)
            self.assertNotIn("push_to_hub", attrs)
            self.assertNotIn("upload_folder", attrs)
            self.assertNotIn("HfApi", calls)

    def test_generation_refuses_an_empty_heldout_and_a_live_repo(self):
        teacher = _pilot("teacher")
        base = {"id": teacher.TEACHER_ID, "revision": teacher.TEACHER_REVISION}
        prompt = {
            "id": "x",
            "source": "dry-run",
            "license": "Apache-2.0",
            "messages": [{"role": "user", "content": "user question"}],
        }
        kwargs = {
            "compat": None,
            "base": base,
            "guardrails": {},
            "reasoning": {},
            "max_new_tokens": 1,
            "out_dir": PILOT,
            "dataset_repo": "SZLHOLDINGS/chaski-0p8b-pilot-sft",
        }
        with self.assertRaises(RuntimeError) as empty:
            teacher.generate_completions(
                None, None, [prompt], heldout_users=set(), **kwargs
            )
        self.assertIn("empty", str(empty.exception))
        with self.assertRaises(RuntimeError) as live:
            teacher.generate_completions(
                None,
                None,
                [prompt],
                heldout_users={"other"},
                **{**kwargs, "dataset_repo": "SZLHOLDINGS/chaski"},
            )
        self.assertIn("chaski", str(live.exception).casefold())

    def test_execute_is_refused(self):
        for module in ("teacher", "student"):
            with self.assertRaises(SystemExit) as caught:
                _pilot(module).main(["--execute"])
            self.assertIn("not authorized", str(caught.exception))


class CpuDryRunTests(unittest.TestCase):
    def test_tiny_teacher_and_student(self):
        try:
            for name in ("torch", "transformers", "trl", "peft", "datasets"):
                importlib.import_module(name)
        except ImportError:
            self.skipTest("pinned trainer stack is not installed")
        dry = _pilot("dry_run")
        teacher = dry.run_teacher_dry_run()
        student = dry.run_student_dry_run()
        self.assertEqual(teacher["rows"], 2)
        self.assertFalse(teacher["hub_write"])
        self.assertFalse(teacher["gpu_job_launched"])
        self.assertIn("held-out", teacher["heldout_refusal"])
        self.assertTrue(teacher["notice_present"])
        self.assertFalse(teacher["enable_thinking"])
        self.assertEqual(student["global_steps"], 2)
        self.assertTrue(math.isfinite(student["training_loss"]))
        self.assertEqual(student["architecture"], "Qwen3_5ForConditionalGeneration")
        self.assertEqual(student["output_repo"], "SZLHOLDINGS/chaski-0p8b-pilot")
        self.assertIn("q_proj", student["lora_targets"])
        self.assertIn("in_proj_qkv", student["lora_targets"])
        self.assertNotIn("lm_head", student["lora_targets"])
        self.assertNotIn("qkv", student["lora_targets"])
        self.assertNotIn("proj", student["lora_targets"])
        self.assertTrue(student["notice_present"])
        self.assertFalse(student["hub_write"])
        self.assertFalse(student["gpu_job_launched"])
        self.assertIn("chaski", student["forbidden_repo_refusal"].casefold())


def _pilot_launch():
    path = ROOT / "training" / "launch_contract.py"
    spec = importlib.util.spec_from_file_location("szl_launch_contract", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"missing {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
