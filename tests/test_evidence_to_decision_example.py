"""Collects the examples/evidence_to_decision suites under the repository's tests/ gate.

Runs the example's own unittest suites and its acceptance command in a subprocess so that
its module names never enter this test session.  Synthetic, offline; no model or network.
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "evidence_to_decision"


def _run(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, *args], cwd=EXAMPLE, capture_output=True, text=True,
                          env=env, timeout=300)


@unittest.skipUnless(EXAMPLE.is_dir(), "examples/evidence_to_decision not present")
class EvidenceToDecisionExample(unittest.TestCase):
    def test_example_unit_suites(self):
        r = _run("-m", "unittest", "discover", "-s", ".", "-p", "test_*.py", "-t", ".")
        self.assertEqual(r.returncode, 0, r.stderr[-4000:])

    def test_example_acceptance(self):
        r = _run("acceptance.py")
        self.assertEqual(r.returncode, 0, r.stdout[-4000:])
