"""Validate advertised static reads without importing or starting the legacy app."""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StaticAgentContractTests(unittest.TestCase):
    def test_advertised_get_paths_are_packaged_static_files(self):
        contract = (ROOT / "agents.md").read_text(encoding="utf-8")
        paths = re.findall(r"`GET (/[^`\s]*)`", contract)
        self.assertTrue(paths, "Agent contract must advertise static read paths")
        for path in paths:
            with self.subTest(path=path):
                relative = "index.html" if path == "/" else path.lstrip("/")
                target = (ROOT / relative).resolve()
                self.assertTrue(target.is_relative_to(ROOT.resolve()))
                self.assertTrue(target.is_file(), f"Unpackaged GET path: {path}")

    def test_contract_covers_the_entrypoint_and_every_initial_evidence_fetch(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertRegex(readme, r"(?m)^sdk: static$")
        entry = re.search(r"(?m)^app_file: (\S+)$", readme)
        self.assertIsNotNone(entry)
        index = (ROOT / entry.group(1)).read_text(encoding="utf-8")
        match = re.search(r"const FILES\s*=\s*(\[[^;]+\]);", index)
        self.assertIsNotNone(match, "Review agent reads when the fetch manifest changes")
        files = json.loads(match.group(1))
        advertised = set(re.findall(r"`GET (/[^`\s]*)`",
                                    (ROOT / "agents.md").read_text(encoding="utf-8")))
        self.assertIn("/" + entry.group(1), advertised)
        self.assertTrue({"/" + name for name in files}.issubset(advertised))
        for name in files:
            with self.subTest(name=name):
                json.loads((ROOT / name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
