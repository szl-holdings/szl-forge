from __future__ import annotations

import json
from pathlib import Path
import socket
import tempfile
import unittest

import owned_agent_clinical_control as oac
from oac_live_transport_bridge import MLLP_END, MLLP_START
from oac_stack_integration import ClinicalKernel
from test_live_shadow_clinical import live_shadow_message


MODULE_PATH = Path(oac.__file__).resolve()


class ClinicalGatewayEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="oac-gateway-e2e-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / "state"
        self.paths = oac.state_paths(self.state)
        oac.clinical_initialize(self.paths)

        assay_map = self.root / "assay-map.json"
        assay_map.write_text(
            oac.pretty_json(
                {
                    "SITE-FLU": {
                        "display": "Site configured influenza assay",
                        "local_system": "urn:szl:site:assay",
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        oac.clinical_add_source(
            self.paths,
            "liat-live-shadow",
            "LIVE_SHADOW",
            oac.CLINICAL_LIVE_SHADOW_PROFILE_ID,
            "LIAT-DEID",
            "SITE-DEID",
            assay_map,
        )

        self.binding = self.root / "binding.json"
        self.binding.write_text(
            oac.pretty_json(
                {
                    "deidentified": True,
                    "order_reference": "ServiceRequest/DEID-ORDER0001",
                    "patient_reference": "Patient/DEID-SUBJECT01",
                    "recipient_id": "SHADOW-OFFLINE-RECIPIENT",
                    "source_order_token": "DEID-ORDER0001",
                    "source_subject_token": "DEID-SUBJECT01",
                    "specimen_reference": "Specimen/DEID-SPECIMEN1",
                    "supersedes_result_id": None,
                    "synthetic": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.kernel = ClinicalKernel(
            state_dir=self.state,
            script=MODULE_PATH,
            data_root=self.root,
        )

    def _stop_transport(self) -> None:
        try:
            self.kernel.transport_stop(
                transport_id="e2e-listener",
                state_dir=self.state,
            )
        except Exception:
            pass

    def test_deidentified_mllp_listener_reaches_real_control_engine(self) -> None:
        started = self.kernel.transport_start(
            transport_id="e2e-listener",
            kind="mllp-listener",
            source_id="liat-live-shadow",
            binding_path=self.binding,
            state_dir=self.state,
            config={
                "allowed_peer_ips": ["127.0.0.1"],
                "bind_host": "127.0.0.1",
                "bind_port": 0,
                "required_hl7_version": "2.5",
                "required_message_type": "ORU^R30^ORU_R30",
                "single_message_per_connection": True,
            },
        )
        self.addCleanup(self._stop_transport)
        port = int(started["status"]["listener_port"])
        self.assertGreater(port, 0)

        with socket.create_connection(("127.0.0.1", port), timeout=5.0) as connection:
            connection.settimeout(10.0)
            connection.sendall(MLLP_START + live_shadow_message() + MLLP_END)
            response = connection.recv(65536)

        self.assertTrue(response.startswith(MLLP_START))
        self.assertTrue(response.endswith(MLLP_END))
        self.assertIn(b"MSA|AA|TECH-MSG-001", response)

        with oac.clinical_connect(self.paths) as connection:
            row = connection.execute(
                "SELECT result_id FROM clinical_results"
            ).fetchone()
            self.assertIsNotNone(row)
            state = oac.clinical_current_state(connection, row["result_id"])
        self.assertEqual(state, "PENDING_REVIEW")

        dataset_rows = self.kernel.dataset_tail(state_dir=self.state, limit=10)
        self.assertEqual(len(dataset_rows), 1)
        serialized = json.dumps(dataset_rows, sort_keys=True)
        self.assertNotIn("PERSON^NAME", serialized)
        self.assertNotIn("Not detected", serialized)
        self.assertEqual(
            dataset_rows[0]["score_semantics"],
            "deterministic_operational_evidence_not_clinical_confidence",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
