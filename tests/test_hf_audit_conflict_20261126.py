"""Replay of szl-forge Hugging Face evidence audit run 45903315277.

These fixtures are the immutable-revision snapshots retained in artifact
15156821597. They are documentation/evidence consistency checks, not live
model evaluation or a promotion verdict.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "audit_hf_estate.py"
SPEC = importlib.util.spec_from_file_location("audit_hf_estate", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

FIX = Path(__file__).resolve().parent / "fixtures" / "hf-evidence-audit-20261126"
INVENTORY = ROOT / "publishing" / "evidence-audit-inventory.json"

REPOS = (
    "SZLHOLDINGS/brain-navigator-r2",
    "SZLHOLDINGS/chaski-r2",
    "SZLHOLDINGS/khipu-r3",
)
CONFLICT_CODE = "CANONICAL_STATE_NOT_DISCLOSED"
HOLD_CODES = {"REQUIRED_DISCLOSURE_MISSING", "REQUIRED_BOUNDARY_MISSING"}


def inventory_entry(repo: str) -> dict:
    data = json.loads(INVENTORY.read_text(encoding="utf-8"))
    return next(item for item in data["repositories"] if item["repo"] == repo)


def snapshot(name: str) -> dict:
    return json.loads((FIX / f"{name}.snapshot.json").read_text(encoding="utf-8"))


def with_readme(snap: dict, text: str) -> dict:
    out = deepcopy(snap)
    out["contents"]["README.md"] = text
    out["file_sha256"]["README.md"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return out


def codes(result: dict) -> set[str]:
    return {item["code"] for item in result["findings"]}


def conflict_findings(result: dict) -> list[dict]:
    return [item for item in result["findings"] if item["status"] == "CONFLICT"]


class CanonicalGateDisclosureTests(unittest.TestCase):
    def test_audited_readmes_conflict_because_gate_fail_is_not_visible(self):
        for repo in REPOS:
            name = repo.split("/", 1)[1]
            with self.subTest(repo=repo):
                result = audit.classify_repository(inventory_entry(repo), snapshot(name))
                self.assertEqual(result["status"], "CONFLICT")
                self.assertIn(CONFLICT_CODE, codes(result))
                self.assertTrue(conflict_findings(result))
                self.assertTrue(
                    all(item["code"] == CONFLICT_CODE for item in conflict_findings(result))
                )
                self.assertIn(
                    "The canonical post-publication gate FAIL must be visible.",
                    {item["message"] for item in conflict_findings(result)},
                )

    def test_patched_readmes_clear_conflict_and_required_card_holds(self):
        for repo in REPOS:
            name = repo.split("/", 1)[1]
            with self.subTest(repo=repo):
                text = (FIX / f"{name}.patched.README.md").read_text(encoding="utf-8")
                result = audit.classify_repository(
                    inventory_entry(repo), with_readme(snapshot(name), text)
                )
                found = codes(result)
                self.assertNotIn(CONFLICT_CODE, found)
                self.assertFalse(HOLD_CODES & found)
                self.assertFalse(conflict_findings(result))
                self.assertEqual(result["status"], "PASS")
                self.assertEqual(result["promotion_verdict"], "NOT_ASSESSED")

    def test_live_brain_navigator_harvest_card_still_conflicts(self):
        """Hub main moved after the audit (a545dce, 2026-11-27T00:11:40Z)."""
        text = (FIX / "brain-navigator-r2.live-20261126T1911ET.README.md").read_text(
            encoding="utf-8"
        )
        result = audit.classify_repository(
            inventory_entry("SZLHOLDINGS/brain-navigator-r2"),
            with_readme(snapshot("brain-navigator-r2"), text),
        )
        self.assertEqual(result["status"], "CONFLICT")
        self.assertIn(CONFLICT_CODE, codes(result))
        self.assertIn("REQUIRED_DISCLOSURE_MISSING", codes(result))
        self.assertIn("REQUIRED_BOUNDARY_MISSING", codes(result))
        self.assertNotIn("license:", text)
        self.assertNotIn("apache", text.lower())


if __name__ == "__main__":
    unittest.main()
