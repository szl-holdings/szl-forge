"""Execute the shipped browser script against real Python scoring fixtures.

Node is a required test tool, not a service/runtime dependency. These VM/DOM
contracts do not claim browser layout, clinical performance or deployment proof.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "oac_health_ui_fixture_app", ROOT / "app.py"
)
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)


class BrowserScriptTests(unittest.TestCase):
    def test_shipped_script_real_fixtures_and_malformed_contracts(self):
        node = shutil.which("node")
        self.assertIsNotNone(
            node, "Node 22 must be installed for required UI contracts"
        )
        application = app.Application(ROOT, "b" * 40)
        self.assertTrue(application.ready)
        self.assertTrue(application.v2_ready)
        healthy = {
            "listener_running": True,
            "tls_enabled": True,
            "peer_allowlist_configured": True,
            "queue_utilization": 0.08,
            "consecutive_failures": 0,
            "seconds_since_last_success": 20,
            "ledger_integrity_ok": True,
            "configuration_valid": True,
        }
        degraded = {
            "listener_running": False,
            "tls_enabled": False,
            "peer_allowlist_configured": False,
            "queue_utilization": 0.92,
            "consecutive_failures": 12,
            "seconds_since_last_success": 21600,
            "ledger_integrity_ok": False,
            "configuration_valid": False,
        }
        ambiguous = {
            "listener_running": True,
            "tls_enabled": True,
            "peer_allowlist_configured": True,
            "queue_utilization": 0.1,
            "consecutive_failures": 5,
            "seconds_since_last_success": 1,
            "ledger_integrity_ok": True,
            "configuration_valid": True,
        }
        fixtures = {
            "identity": application.identity(),
            "healthy": application.score(app.canonical({"features": healthy})),
            "degraded": application.score(app.canonical({"features": degraded})),
            "v2_identity": application.v2_identity(),
            "v2_healthy": application.score_v2(app.canonical({"features": healthy})),
            "v2_degraded": application.score_v2(app.canonical({"features": degraded})),
            "v2_ambiguous": application.score_v2(app.canonical({"features": ambiguous})),
            "healthy_features": healthy,
            "degraded_features": degraded,
        }
        result = subprocess.run(
            [node, str(ROOT / "tests" / "ui_contract.cjs"), str(ROOT / "index.html")],
            input=json.dumps(fixtures, allow_nan=False),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["complete"])
        self.assertEqual(report["actual_backend_fixtures"], 2)
        self.assertGreaterEqual(report["malformed_contracts_rejected"], 81)
        self.assertGreaterEqual(report["interaction_checks"], 15)
        self.assertEqual(report["actual_v2_backend_fixtures"], 3)
        self.assertGreaterEqual(report["v2_malformed_contracts_rejected"], 8)
        self.assertGreaterEqual(report["v2_interaction_checks"], 20)
        self.assertEqual(report["scope"], "LOCAL_SCRIPT_CONTRACT_ONLY")
        print("OAC UI contract: " + result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
