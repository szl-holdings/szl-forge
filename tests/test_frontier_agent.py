# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

FILE = Path(__file__).resolve().parents[1] / "tools" / "szl_frontier_agent.py"
spec = importlib.util.spec_from_file_location("frontier_agent", FILE)
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evidence = self.root / "evidence.json"
        self.evidence.write_text('{"observed":true}', encoding="utf-8")
        self.mission = {"schema": agent.SCHEMA, "mode": "research", "objective": "Find a testable improvement",
                        "repository": str(self.root), "source_revision": "a" * 40, "timeout_seconds": 30,
                        "evidence": [{"path": str(self.evidence), "sha256": agent.digest(self.evidence.read_bytes())}],
                        "checks": []}
        self.path = self.root / "mission.json"
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.mission), encoding="utf-8")

    def test_changed_evidence_is_rejected_before_execution(self):
        self.evidence.write_text('{"observed":false}', encoding="utf-8")
        with self.assertRaisesRegex(agent.MissionError, "Evidence changed"):
            agent.bound_evidence(self.mission)

    def test_build_requires_independent_operator_checks(self):
        self.mission["mode"] = "build"
        self.save()
        with self.assertRaisesRegex(agent.MissionError, "independent checks"):
            agent.load_mission(self.path)

    def test_command_uses_sandbox_and_stdin_without_permission_bypass(self):
        command = agent.codex_command("codex.exe", self.root, "build")
        self.assertIn("read-only", command)
        self.assertIn('approval_policy="never"', command)
        self.assertIn("--ignore-user-config", command)
        self.assertEqual(command[-1], "-")
        self.assertFalse(any("bypass" in part for part in command))
        self.assertIn("read-only", agent.codex_command("codex.exe", self.root, "research"))

    def test_shell_text_and_unbounded_deadline_are_rejected(self):
        for update in ({"checks": ["python -m unittest"]}, {"timeout_seconds": 999999}):
            with self.subTest(update=update):
                previous = self.mission.copy()
                self.mission.update(update)
                self.save()
                with self.assertRaises(agent.MissionError):
                    agent.load_mission(self.path)
                self.mission = previous

    def fake_git(self, repository, *args):
        if args[0] == "status":
            return ""
        if args[0] == "rev-parse":
            return "a" * 40
        if args[0] == "remote":
            return "https://github.com/szl-holdings/example.git"
        if "worktree" in args:
            Path(args[-2]).mkdir()
        return ""

    def test_prepare_never_launches_model_or_creates_worktree(self):
        with patch.object(agent, "git", side_effect=self.fake_git) as git, \
             patch.object(agent.shutil, "which", return_value="codex.exe"), \
             patch.object(agent, "bounded_run") as run:
            receipt = agent.run_mission(self.path, self.root / "prepared")
        self.assertEqual(receipt["state"], "PREPARED")
        self.assertFalse(receipt["model_execution_observed"])
        self.assertFalse(receipt["production_authorized"])
        self.assertFalse(any("worktree" in call.args for call in git.call_args_list))
        run.assert_not_called()

    def test_exit_zero_without_model_completion_is_incomplete(self):
        def fake_run(argv, cwd, folder, label, seconds, incoming):
            (folder / "model.jsonl").write_text('{"type":"thread.started"}\n', encoding="utf-8")
            return {"exit_code": 0, "stopped": None}
        with patch.object(agent, "git", side_effect=self.fake_git), \
             patch.object(agent.shutil, "which", return_value="codex.exe"), \
             patch.object(agent.subprocess, "check_output", return_value=b""), \
             patch.object(agent, "bounded_run", side_effect=fake_run):
            receipt = agent.run_mission(self.path, self.root / "no-completion", True)
        self.assertEqual(receipt["state"], "INCOMPLETE")
        self.assertFalse(receipt["model_execution_observed"])

    def test_model_report_does_not_authorize_release_or_training(self):
        def fake_run(argv, cwd, folder, label, seconds, incoming):
            events = [{"type": "item.completed", "item": {"type": "agent_message", "text": "Proposed experiment"}},
                      {"type": "turn.completed"}]
            (folder / "model.jsonl").write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
            return {"exit_code": 0, "stopped": None}
        with patch.object(agent, "git", side_effect=self.fake_git), \
             patch.object(agent.shutil, "which", return_value="codex.exe"), \
             patch.object(agent.subprocess, "check_output", return_value=b""), \
             patch.object(agent, "bounded_run", side_effect=fake_run):
            receipt = agent.run_mission(self.path, self.root / "research", True)
        self.assertEqual(receipt["state"], "MODEL_RESPONSE_OBSERVED")
        self.assertFalse(receipt["tool_execution_observed"])
        self.assertFalse(receipt["production_authorized"])
        self.assertFalse(receipt["training_admitted"])

    def test_source_bundle_uses_exact_git_revision(self):
        self.mission["source_paths"] = ["python/szl_frontier/engine.py"]
        with patch.object(agent.subprocess, "check_output", return_value=b"source\n") as read:
            source = agent.source_snapshot(self.root, self.mission)
        self.assertEqual(read.call_args.args[0][-1], "a" * 40 + ":python/szl_frontier/engine.py")
        self.assertEqual(source[0]["sha256"], agent.digest(b"source\n"))
        self.assertIn('"id": "S1"', agent.prompt_for(self.mission, [], source))

    def test_source_path_traversal_is_rejected(self):
        self.mission["source_paths"] = ["../private"]
        self.save()
        with self.assertRaisesRegex(agent.MissionError, "relative repository"):
            agent.load_mission(self.path)

    def test_actual_process_output_and_failure_are_retained(self):
        result = agent.bounded_run([sys.executable, "-c", "print('measured'); raise SystemExit(3)"],
                                   self.root, self.root, "process", 10)
        self.assertEqual(result["exit_code"], 3)
        self.assertIsNone(result["stopped"])
        self.assertIn("measured", (self.root / "process.jsonl").read_text())

    def test_actual_process_timeout_is_incomplete(self):
        result = agent.bounded_run([sys.executable, "-c", "import time; time.sleep(30)"],
                                   self.root, self.root, "timeout", 0.3)
        self.assertEqual(result["stopped"], "TIMEOUT")
        self.assertNotEqual(result["exit_code"], 0)

    def test_fast_process_output_overflow_is_not_success(self):
        with patch.object(agent, "MAX_LOG", 10):
            result = agent.bounded_run([sys.executable, "-c", "print('x' * 100)"],
                                       self.root, self.root, "overflow", 10)
        self.assertEqual(result["stopped"], "OUTPUT_LIMIT")

    def test_windows_path_aliases_are_rejected(self):
        for path in ("C:/private", "file:stream", ".GIT/config", "a/../b", "a./b", "a /b"):
            self.assertFalse(agent.relative_file(path), path)


class BuildGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init")
        self.git("config", "core.autocrlf", "false")
        self.git("remote", "add", "origin", "https://github.com/szl-holdings/test-fixture.git")
        (self.repo / "value.py").write_bytes(b"VALUE = 0\n")
        (self.repo / "guard.py").write_bytes(b"protected\n")
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "-m", "fixture")
        self.check = self.root / "check.py"
        self.check.write_text("from pathlib import Path\nraise SystemExit(0 if Path('value.py').read_text() == 'VALUE = 1\\n' else 1)\n", encoding="utf-8")
        evidence = self.root / "evidence.json"
        evidence.write_text('{"kind":"synthetic-unit-fixture"}', encoding="utf-8")
        self.mission = {
            "schema": agent.SCHEMA, "mode": "build", "objective": "Repair fixture",
            "repository": str(self.repo), "source_revision": self.git("rev-parse", "HEAD").strip(),
            "timeout_seconds": 30, "allowed_paths": ["value.py"], "baseline_exit_codes": [1],
            "checks": [[sys.executable, "-I", str(self.check)]],
            "check_files": [{"path": str(self.check), "sha256": agent.digest(self.check.read_bytes())}],
            "evidence": [{"path": str(evidence), "sha256": agent.digest(evidence.read_bytes())}],
        }

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True,
                                       stderr=subprocess.DEVNULL)

    def execute(self, mutate):
        path = self.root / "mission.json"
        path.write_text(json.dumps(self.mission), encoding="utf-8")
        real_run = agent.bounded_run

        def model_or_check(argv, cwd, folder, label, seconds, incoming=""):
            if label != "model":
                return real_run(argv, cwd, folder, label, seconds, incoming)
            proposal = mutate(cwd)
            events = [{"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(proposal)}},
                      {"type": "turn.completed"}]
            (folder / "model.jsonl").write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
            return {"exit_code": 0, "stopped": None}

        with patch.object(agent.shutil, "which", return_value="codex.exe"), \
             patch.object(agent, "bounded_run", side_effect=model_or_check):
            return agent.run_mission(path, self.root / "run", True)

    def proposal(self, after="VALUE = 1\n", path="value.py"):
        return {"summary": "Unit fixture", "edits": [{"path": path, "before": "VALUE = 0\n", "after": after}]}

    def test_real_baseline_and_checks_are_required_for_verified_repair(self):
        result = self.execute(lambda cwd: self.proposal())
        self.assertEqual(result["state"], "REPAIR_VERIFIED")
        self.assertEqual(result["baseline_checks"][0]["exit_code"], 1)
        self.assertEqual(result["checks"][0]["exit_code"], 0)
        self.assertEqual(len(result["patch_sha256"]), 64)
        self.assertFalse(result["production_authorized"])
        self.assertFalse(result["training_admitted"])

    def test_no_op_is_not_a_verified_build(self):
        result = self.execute(lambda cwd: self.proposal("VALUE = 0\n"))
        self.assertEqual(result["state"], "INCOMPLETE")
        self.assertEqual(result["error"], "No-op edit")

    def test_out_of_scope_change_blocks_checks(self):
        result = self.execute(lambda cwd: self.proposal(path="guard.py"))
        self.assertEqual(result["state"], "INCOMPLETE")
        self.assertEqual(result["error"], "Change outside assigned files")
        self.assertEqual(result["checks"], [])

    def test_changed_independent_check_is_rejected(self):
        def mutate(cwd):
            self.check.write_text("raise SystemExit(0)\n")
            return self.proposal()
        result = self.execute(mutate)
        self.assertEqual(result["state"], "INCOMPLETE")
        self.assertEqual(result["error"], "Independent check file changed")

    def test_failed_candidate_check_is_not_verified(self):
        result = self.execute(lambda cwd: self.proposal("VALUE = 2\n"))
        self.assertEqual(result["state"], "INCOMPLETE")
        self.assertEqual(result["checks"][0]["exit_code"], 1)

    def test_unexpected_baseline_blocks_model_execution(self):
        self.mission["baseline_exit_codes"] = [0, 1]
        self.mission["checks"] *= 2
        called = []
        result = self.execute(lambda cwd: called.append(True))
        self.assertEqual(result["state"], "INCOMPLETE")
        self.assertFalse(called)
        self.assertFalse(result["model_execution_observed"])

    def test_direct_model_write_is_rejected_before_application(self):
        def mutate(cwd):
            (cwd / "value.py").write_bytes(b"VALUE = 1\n")
            return self.proposal()
        result = self.execute(mutate)
        self.assertEqual(result["state"], "INCOMPLETE")
        self.assertEqual(result["error"], "Model modified source before proposal validation")

    def test_invalid_later_edit_does_not_apply_earlier_edit(self):
        self.mission["allowed_paths"].append("guard.py")
        proposal = self.proposal()
        proposal["edits"].append({"path": "guard.py", "before": "not present", "after": "bad"})
        result = self.execute(lambda cwd: proposal)
        self.assertEqual(result["state"], "INCOMPLETE")
        self.assertEqual((Path(result["workspace"]) / "value.py").read_bytes(), b"VALUE = 0\n")

    def test_new_file_proposal_is_retained_with_content_digest(self):
        self.mission["allowed_paths"].append("tests/new.py")
        proposal = self.proposal()
        proposal["edits"].append({"path": "tests/new.py", "before": "", "after": "# regression\n"})
        result = self.execute(lambda cwd: proposal)
        self.assertEqual(result["state"], "REPAIR_VERIFIED")
        self.assertEqual({row["path"] for row in result["changed_files"]}, {"value.py", "tests/new.py"})

    def test_duplicate_or_ambiguous_edits_are_rejected(self):
        for proposal in (
            {"summary": "fixture", "edits": [self.proposal()["edits"][0]] * 2},
            {"summary": "fixture", "edits": [{"path": "value.py", "before": "absent", "after": "bad"}]},
        ):
            with self.subTest(proposal=proposal):
                with self.assertRaises(agent.MissionError):
                    agent.apply_proposal(self.repo, proposal, ["value.py"])
        self.assertEqual((self.repo / "value.py").read_bytes(), b"VALUE = 0\n")


if __name__ == "__main__":
    unittest.main()
