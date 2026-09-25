# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path
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
        self.assertIn("workspace-write", command)
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
        if args[0] == "worktree":
            Path(args[3]).mkdir()
        return ""

    def test_prepare_never_launches_model_or_creates_worktree(self):
        with patch.object(agent, "git", side_effect=self.fake_git) as git, \
             patch.object(agent.shutil, "which", return_value="codex.exe"), \
             patch.object(agent, "bounded_run") as run:
            receipt = agent.run_mission(self.path, self.root / "prepared")
        self.assertEqual(receipt["state"], "PREPARED")
        self.assertFalse(receipt["model_execution_observed"])
        self.assertFalse(receipt["production_authorized"])
        self.assertFalse(any(call.args[1] == "worktree" for call in git.call_args_list))
        run.assert_not_called()

    def test_exit_zero_without_model_completion_is_incomplete(self):
        def fake_run(argv, cwd, folder, label, seconds, incoming):
            (folder / "model.jsonl").write_text('{"type":"thread.started"}\n', encoding="utf-8")
            return {"exit_code": 0, "stopped": None}
        with patch.object(agent, "git", side_effect=self.fake_git), \
             patch.object(agent.shutil, "which", return_value="codex.exe"), \
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


if __name__ == "__main__":
    unittest.main()
