from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import owned_agent_clinical_control as oac

def live_shadow_message(*, include_patient_name: bool = False) -> bytes:
    msh = ["MSH"] + [""] * 17
    msh[1] = "^~\\&"
    msh[2] = "LIAT-DEID"
    msh[3] = "SITE-DEID"
    msh[6] = "20260904090000-0400"
    msh[8] = "ORU^R30^ORU_R30"
    msh[9] = "TECH-MSG-001"
    msh[10] = "P"
    msh[11] = "2.5"
    msh[17] = "UNICODE UTF-8"

    pid = ["PID"] + [""] * 5
    pid[3] = "DEID-SUBJECT01"
    if include_patient_name:
        pid[5] = "PERSON^NAME"

    orc = ["ORC", "NW"]
    obr = ["OBR"] + [""] * 25
    obr[1] = "1"
    obr[2] = "DEID-ORDER0001"
    obr[3] = "DEID-REPORT001"
    obr[4] = "SITE-FLU^Site configured influenza assay^urn:szl:site:assay"
    obr[7] = "20260904085900-0400"
    obr[11] = "O"
    obr[22] = "20260904090000-0400"
    obr[25] = "F"

    nte = [
        "NTE",
        "1",
        "L",
        "Run=RUN001;Device=LIAT001;Version=3.5.0;Tube=TUBE001;TubeExp=2099-01-01",
    ]
    obx = ["OBX"] + [""] * 11
    obx[1] = "1"
    obx[2] = "ST"
    obx[3] = "FLU-A-B^Influenza A/B result^urn:szl:site:observation"
    obx[5] = "Not detected"
    obx[11] = "F"
    return ("\r".join("|".join(segment) for segment in (msh, pid, orc, obr, nte, obx)) + "\r").encode()


class LiveShadowClinicalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="oac-live-shadow-test-")
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.paths = oac.state_paths(root / "state")
        oac.clinical_initialize(self.paths)

        self.assay_map = root / "assay-map.json"
        self.assay_map.write_text(
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
            self.assay_map,
        )

        self.key = oac.Ed25519PrivateKey.generate()
        oac.clinical_add_reviewer_raw(
            self.paths,
            "shadow_reviewer",
            self.key.public_key(),
            actor="live-shadow-test",
        )
        oac.clinical_seal_reviewer_trust(self.paths)

        self.binding = root / "binding.json"
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

    def _message_path(self, *, include_patient_name: bool = False) -> Path:
        path = Path(self.temp.name) / "result.hl7"
        path.write_bytes(live_shadow_message(include_patient_name=include_patient_name))
        return path

    def test_live_shadow_review_and_not_delivered_fhir_candidate(self) -> None:
        ingested = oac.clinical_ingest(
            self.paths,
            "liat-live-shadow",
            self._message_path(),
            self.binding,
        )
        self.assertEqual(ingested["state"], "PENDING_REVIEW")
        self.assertEqual(ingested["failed_gates"], [])

        review = oac.build_clinical_review(
            self.paths,
            ingested["result_id"],
            "shadow_reviewer",
            120,
        )
        signed = oac.sign_clinical_review(self.paths, review, self.key)
        applied = oac.apply_clinical_review(self.paths, signed)
        self.assertEqual(applied["state"], "AUTHORIZED_FOR_EXPORT")

        out = Path(self.temp.name) / "live-shadow-fhir.json"
        exported = oac.clinical_export_fhir(self.paths, ingested["result_id"], out)
        self.assertFalse(exported["clinical_delivery"])
        self.assertEqual(exported["source_mode"], "LIVE_SHADOW")
        self.assertEqual(
            exported["truth_boundary"],
            "deidentified_live_shadow_artifact_not_clinical_delivery",
        )
        bundle = json.loads(out.read_text(encoding="utf-8"))
        tags = {tag["code"] for tag in bundle["meta"]["tag"]}
        self.assertEqual(tags, {"deidentified-live-shadow", "not-delivered"})

        manifest = json.loads(
            Path(exported["authorization_manifest_out"]).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["source_mode"], "LIVE_SHADOW")
        self.assertFalse(manifest["clinical_delivery"])

    def test_patient_name_is_rejected_before_persistence(self) -> None:
        with self.assertRaisesRegex(oac.ControlError, "UNEXPECTED_POPULATED_FIELD"):
            oac.clinical_ingest(
                self.paths,
                "liat-live-shadow",
                self._message_path(include_patient_name=True),
                self.binding,
            )
        with oac.clinical_connect(self.paths) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM clinical_messages").fetchone()[0], 0)

    def test_unbounded_live_and_real_phi_flags_are_denied(self) -> None:
        value = json.loads(self.binding.read_text(encoding="utf-8"))
        value["deidentified"] = False
        self.binding.write_text(oac.pretty_json(value) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(oac.ControlError, "deidentified=true"):
            oac.clinical_ingest(
                self.paths,
                "liat-live-shadow",
                self._message_path(),
                self.binding,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
