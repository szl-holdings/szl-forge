from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if os.environ.get("OAC_TEST_INSTALLED_PACKAGE") != "1" and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from owned_agent_control.controller import self_test  # noqa: E402


@unittest.skipUnless(os.name == "nt", "requires real Windows Job Object enforcement")
class WindowsEnforcementIntegrationTests(unittest.TestCase):
    def test_real_parent_child_tree_isolated_and_restart_denied(self) -> None:
        result = self_test()
        self.assertIs(result["ok"], True)
        self.assertEqual(result["operation_status"], "VERIFIED_LOCAL_WINDOWS_ENFORCEMENT")
        self.assertIs(result["checks"]["restart_blocked"], True)
        self.assertEqual(result["checks"]["restart_denial"], "TARGET_ISOLATED")
        self.assertEqual(result["checks"]["replay_denial"], "REPLAY_DENIED")
        self.assertIs(result["checks"]["parent_running_after"], False)
        self.assertIs(result["checks"]["child_running_after"], False)


if __name__ == "__main__":
    unittest.main()
