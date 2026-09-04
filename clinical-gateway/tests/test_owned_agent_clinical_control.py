import ast
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

import owned_agent_clinical_control as oac

SOURCE = Path(oac.__file__).resolve()


class ClinicalHarness:
    def __init__(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="oac-adversarial-", ignore_cleanup_errors=True)
        self.paths = oac.state_paths(Path(self.temp.name).resolve())
        oac.clinical_initialize(self.paths)
        self.assay_map_path = self.paths.root / "assay-map.json"
        oac.atomic_write(
            self.assay_map_path,
            (
                oac.pretty_json(
                    {
                        "SYNTH-FLU": {
                            "display": "Synthetic influenza assay",
                            "local_system": "urn:synthetic:assay",
                        }
                    }
                )
                + "\n"
            ).encode(),
        )
        oac.clinical_add_source(
            self.paths,
            "synthetic-liat",
            "MOCK",
            oac.CLINICAL_PROFILE_ID,
            "LIAT-SIM",
            "LAB-SIM",
            self.assay_map_path,
        )
        self.key = oac.Ed25519PrivateKey.generate()
        self.public_key_path = self.paths.root / "synthetic-reviewer-public.pem"
        oac.atomic_write(
            self.public_key_path,
            self.key.public_key().public_bytes(
                encoding=oac.serialization.Encoding.PEM,
                format=oac.serialization.PublicFormat.SubjectPublicKeyInfo,
            ),
        )
        oac.clinical_add_reviewer_raw(
            self.paths, "synthetic_reviewer", self.key.public_key(), actor="adversarial-test"
        )
        oac.clinical_seal_reviewer_trust(self.paths)
        self.counter = 0

    def close(self) -> None:
        self.temp.cleanup()

    def files(
        self,
        *,
        assay="SYNTH-FLU",
        status="F",
        value="Detected",
        sender="LIAT-SIM",
        subject=None,
        order=None,
        binding_subject=None,
        binding_order=None,
        mode="MOCK",
    ):
        self.counter += 1
        suffix = f"{self.counter:03d}"
        subject = subject or f"SYNTH-SUBJECT-{suffix}"
        order = order or f"SYNTH-SOURCE-ORDER-{suffix}"
        raw = oac.synthetic_hl7_message(
            f"SYNTH-MSG-{suffix}",
            order,
            f"SYNTH-REPORT-{suffix}",
            subject,
            assay_code=assay,
            report_status=status,
            qualitative_value=value,
        )
        if sender != "LIAT-SIM":
            raw = raw.replace(b"LIAT-SIM", sender.encode(), 1)
        hl7 = self.paths.root / f"message-{suffix}.hl7"
        binding_file = self.paths.root / f"binding-{suffix}.json"
        oac.atomic_write(hl7, raw)
        binding = oac.synthetic_binding(
            binding_subject or subject,
            binding_order or order,
            suffix,
        )
        if mode == "SHADOW":
            binding["synthetic"] = False
            binding["deidentified"] = True
        oac.atomic_write(binding_file, (oac.pretty_json(binding) + "\n").encode())
        return hl7, binding_file

    def ingest(self, **kwargs):
        hl7, binding = self.files(**kwargs)
        source_id = "shadow-liat" if kwargs.get("mode") == "SHADOW" else "synthetic-liat"
        return oac.clinical_ingest(self.paths, source_id, hl7, binding), hl7, binding

    def authorize(self, result_id):
        request = oac.build_clinical_review(
            self.paths, result_id, "synthetic_reviewer", oac.MAX_TTL_SECONDS
        )
        signed = oac.sign_clinical_review(self.paths, request, self.key)
        applied = oac.apply_clinical_review(self.paths, signed)
        return request, signed, applied


class ClinicalAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.h = ClinicalHarness()

    def tearDown(self):
        self.h.close()

    def assert_no_clinical_messages(self):
        with oac.clinical_connect(self.h.paths) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM clinical_messages").fetchone()[0],
                0,
            )

    def ingest_exact_correction(self, predecessor_result_id):
        hl7, binding_path = self.h.files(
            subject="SYNTH-SUBJECT-001", order="SYNTH-SOURCE-ORDER-001"
        )
        oac.atomic_write(
            hl7,
            hl7.read_bytes().replace(
                f"SYNTH-REPORT-{self.h.counter:03d}".encode(), b"SYNTH-REPORT-001", 1
            ),
        )
        binding = json.loads(binding_path.read_text())
        binding.update(
            {
                "order_reference": "ServiceRequest/SYNTH-ORDER-001",
                "patient_reference": "Patient/SYNTH-PATIENT-001",
                "specimen_reference": "Specimen/SYNTH-SPECIMEN-001",
                "supersedes_result_id": predecessor_result_id,
            }
        )
        oac.atomic_write(binding_path, (oac.pretty_json(binding) + "\n").encode())
        return oac.clinical_ingest(
            self.h.paths, "synthetic-liat", hl7, binding_path
        )

    def test_01_capabilities_hard_disable_device_and_network(self):
        caps = oac.clinical_capabilities()
        self.assertFalse(caps["device_control"])
        self.assertFalse(caps["direct_device_transport"])
        self.assertFalse(caps["network_client"])
        self.assertFalse(caps["live_listener"])
        self.assertEqual(caps["device_commands"], [])
        self.assertEqual(caps["clinical_modes"], ["LIVE_SHADOW", "MOCK"])
        self.assertFalse(caps["real_phi_authorized"])
        self.assertFalse(caps["site_validated"])
        self.assertEqual(caps["transport_bridge"], "SEPARATE_PROCESS_REQUIRED")

    def test_02_valid_ingest_requires_review(self):
        result, _, _ = self.h.ingest()
        self.assertEqual(result["state"], "PENDING_REVIEW")
        self.assertEqual(result["failed_gates"], [])

    def test_03_exact_retry_is_idempotent(self):
        first, hl7, binding = self.h.ingest()
        second = oac.clinical_ingest(self.h.paths, "synthetic-liat", hl7, binding)
        self.assertTrue(second["idempotent"])
        self.assertEqual(second["result_id"], first["result_id"])

    def test_04_same_control_id_different_bytes_is_denied(self):
        _, hl7, binding = self.h.ingest(value="Not detected")
        raw = hl7.read_bytes().replace(b"Not detected", b"Indeterminate", 1)
        conflict = self.h.paths.root / "conflict.hl7"
        oac.atomic_write(conflict, raw)
        with self.assertRaisesRegex(oac.ControlError, "different raw bytes"):
            oac.clinical_ingest(self.h.paths, "synthetic-liat", conflict, binding)

    def test_05_unknown_assay_is_quarantined(self):
        result, _, _ = self.h.ingest(assay="SYNTH-UNMAPPED")
        self.assertEqual(result["state"], "QUARANTINED")
        self.assertIn("ASSAY_MAP_MATCH", result["failed_gates"])

    def test_06_sender_mismatch_is_rejected_before_storage(self):
        hl7, binding = self.h.files(sender="SYNTH-BAD")
        with self.assertRaisesRegex(oac.ControlError, "SENDER_PROFILE_MISMATCH"):
            oac.clinical_ingest(self.h.paths, "synthetic-liat", hl7, binding)
        self.assert_no_clinical_messages()

    def test_07_preliminary_result_is_not_export_eligible(self):
        result, _, _ = self.h.ingest(status="P")
        self.assertEqual(result["state"], "QUARANTINED")
        self.assertIn("FINAL_STATUS_REQUIRED_FOR_EXPORT", result["failed_gates"])

    def test_08_binding_mismatch_is_quarantined(self):
        result, _, _ = self.h.ingest(binding_subject="SYNTH-SUBJECT-999")
        self.assertEqual(result["state"], "QUARANTINED")
        self.assertIn("SUBJECT_BINDING_SELF_CONSISTENT", result["failed_gates"])

    def test_09_mock_rejects_non_synthetic_binding(self):
        hl7, binding_path = self.h.files()
        binding = json.loads(binding_path.read_text())
        binding["synthetic"] = False
        oac.atomic_write(binding_path, (oac.pretty_json(binding) + "\n").encode())
        with self.assertRaisesRegex(oac.ControlError, "synthetic=true"):
            oac.clinical_ingest(self.h.paths, "synthetic-liat", hl7, binding_path)

    def test_10_export_requires_signed_reviewer_key_authorization(self):
        result, _, _ = self.h.ingest()
        with self.assertRaisesRegex(oac.ControlError, "signed reviewer-key authorization"):
            oac.clinical_export_fhir(
                self.h.paths, result["result_id"], self.h.paths.root / "premature.json"
            )

    def test_11_invalid_review_signature_is_denied(self):
        result, _, _ = self.h.ingest()
        request = oac.build_clinical_review(
            self.h.paths, result["result_id"], "synthetic_reviewer", oac.MAX_TTL_SECONDS
        )
        signed = oac.sign_clinical_review(self.h.paths, request, self.h.key)
        signature = bytearray(oac.b64url_decode(signed["authorization"]["signature"]))
        signature[0] ^= 1
        signed["authorization"]["signature"] = oac.b64url_encode(bytes(signature))
        with self.assertRaisesRegex(oac.ControlError, "signature is invalid"):
            oac.apply_clinical_review(self.h.paths, signed)

    def test_12_stale_recipient_binding_is_denied_even_when_resigned(self):
        result, _, _ = self.h.ingest()
        request = oac.build_clinical_review(
            self.h.paths, result["result_id"], "synthetic_reviewer", oac.MAX_TTL_SECONDS
        )
        request["recipient_id"] = "SYNTHETIC-DIFFERENT-RECIPIENT"
        signed = oac.sign_clinical_review(self.h.paths, request, self.h.key)
        with self.assertRaisesRegex(oac.ControlError, "recipient_id is stale"):
            oac.apply_clinical_review(self.h.paths, signed)

    def test_13_review_replay_is_denied(self):
        result, _, _ = self.h.ingest()
        _, signed, _ = self.h.authorize(result["result_id"])
        with self.assertRaises(oac.ControlError):
            oac.apply_clinical_review(self.h.paths, signed)

    def test_14_offline_fhir_bundle_is_tagged_not_delivered(self):
        result, _, _ = self.h.ingest()
        self.h.authorize(result["result_id"])
        output = self.h.paths.root / "bundle.json"
        receipt = oac.clinical_export_fhir(self.h.paths, result["result_id"], output)
        bundle = json.loads(output.read_text())
        self.assertFalse(receipt["clinical_delivery"])
        self.assertEqual(bundle["resourceType"], "Bundle")
        self.assertEqual(bundle["meta"]["tag"][1]["code"], "not-delivered")
        self.assertTrue(any(e["resource"]["resourceType"] == "Provenance" for e in bundle["entry"]))
        full_urls = {entry["fullUrl"] for entry in bundle["entry"]}
        report = next(
            entry["resource"]
            for entry in bundle["entry"]
            if entry["resource"]["resourceType"] == "DiagnosticReport"
        )
        provenance = next(
            entry["resource"]
            for entry in bundle["entry"]
            if entry["resource"]["resourceType"] == "Provenance"
        )
        self.assertTrue(all(item["reference"] in full_urls for item in report["result"]))
        self.assertTrue(all(item["reference"] in full_urls for item in provenance["target"]))

    def test_15_append_only_result_trigger_blocks_mutation(self):
        result, _, _ = self.h.ingest()
        with oac.clinical_connect(self.h.paths) as connection:
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute(
                    "UPDATE clinical_results SET source_status='F' WHERE result_id=?",
                    (result["result_id"],),
                )

    def test_16_blank_ct_is_preserved_as_blank_not_zero(self):
        result, _, _ = self.h.ingest(value="Not detected")
        with oac.clinical_connect(self.h.paths) as connection:
            row = connection.execute(
                "SELECT raw_value_lexeme FROM clinical_observations WHERE result_id=? AND ordinal=3",
                (result["result_id"],),
            ).fetchone()
        self.assertEqual(row["raw_value_lexeme"], "")

    def test_17_special_invalid_value_is_preserved(self):
        result, _, _ = self.h.ingest(value="Invalid")
        with oac.clinical_connect(self.h.paths) as connection:
            row = connection.execute(
                "SELECT raw_value_lexeme FROM clinical_observations WHERE result_id=? AND ordinal=2",
                (result["result_id"],),
            ).fetchone()
        self.assertEqual(row["raw_value_lexeme"], "Invalid")

    def test_18_unbounded_live_mode_is_not_implemented(self):
        with self.assertRaisesRegex(oac.ControlError, "LIVE_SHADOW"):
            oac.clinical_add_source(
                self.h.paths,
                "live-liat",
                "LIVE",
                oac.CLINICAL_PROFILE_ID,
                "LIAT-SIM",
                "LAB-SIM",
                self.h.assay_map_path,
            )

    def test_19_status_is_redacted(self):
        result, _, _ = self.h.ingest(subject="SYNTH-SUBJECT-999")
        status = oac.clinical_status(self.h.paths, result["result_id"])
        serialized = json.dumps(status)
        self.assertNotIn("SYNTH-SUBJECT-999", serialized)
        self.assertFalse(status["raw_result_content_in_status"])

    def test_20_invalid_mllp_frame_is_rejected(self):
        raw = oac.synthetic_hl7_message(
            "SYNTH-MSG-900", "SYNTH-SOURCE-ORDER-900", "SYNTH-REPORT-900", "SYNTH-SUBJECT-900"
        )
        with self.assertRaisesRegex(oac.ControlError, "FS CR"):
            oac.parse_roche_liat_hl7(raw[:-1])

    def test_21_ledger_revalidates(self):
        self.h.ingest()
        with oac.clinical_connect(self.h.paths) as connection:
            result = oac.clinical_verify_ledger(connection)
        self.assertEqual(result["integrity"], "VERIFIED_LOCAL_CLINICAL_HASH_CHAIN")

    def test_22_source_has_no_direct_network_or_device_cli(self):
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(SOURCE))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertTrue(imported_roots.isdisjoint({"socket", "serial"}))
        help_text = oac.build_parser().format_help()
        self.assertNotIn("--device-host", help_text)
        self.assertNotIn("--device-port", help_text)
        self.assertNotIn("auto-approve", help_text.lower())

    def test_23_fixture_uses_documented_illustrative_message_shape(self):
        raw = oac.synthetic_hl7_message(
            "SYNTH-MSG-901",
            "SYNTH-SOURCE-ORDER-901",
            "SYNTH-REPORT-901",
            "SYNTH-SUBJECT-901",
        )
        parsed = oac.parse_roche_liat_hl7(raw)
        self.assertEqual(parsed["hl7_version"], "2.5")
        self.assertEqual(parsed["processing_id"], "P")
        self.assertEqual(parsed["order_control"], "NW")
        self.assertEqual(parsed["specimen_action_code"], "O")

    def test_24_missing_orc_is_rejected(self):
        raw = oac.synthetic_hl7_message(
            "SYNTH-MSG-902",
            "SYNTH-SOURCE-ORDER-902",
            "SYNTH-REPORT-902",
            "SYNTH-SUBJECT-902",
        )
        with self.assertRaisesRegex(oac.ControlError, "MSH, PID, ORC, OBR, NTE"):
            oac.parse_roche_liat_hl7(raw.replace(b"\rORC|NW", b"", 1))

    def test_25_non_synthetic_note_is_denied_before_storage(self):
        hl7, binding = self.h.files()
        raw = hl7.read_bytes().replace(
            b"SYNTHETIC DATA - NOT FOR CLINICAL USE",
            b"ALICE SMITH DOB 19700101 REAL-MRN",
            1,
        )
        oac.atomic_write(hl7, raw)
        with self.assertRaisesRegex(oac.ControlError, "OBSERVATION_NOTE_PROFILE_MISMATCH"):
            oac.clinical_ingest(self.h.paths, "synthetic-liat", hl7, binding)
        with oac.clinical_connect(self.h.paths) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM clinical_messages").fetchone()[0], 0)

    def test_26_cross_subject_supersession_is_quarantined(self):
        original, _, _ = self.h.ingest()
        self.h.authorize(original["result_id"])
        oac.clinical_export_fhir(
            self.h.paths, original["result_id"], self.h.paths.root / "original.json"
        )
        hl7, binding_path = self.h.files()
        binding = json.loads(binding_path.read_text())
        binding["supersedes_result_id"] = original["result_id"]
        oac.atomic_write(binding_path, (oac.pretty_json(binding) + "\n").encode())
        corrected = oac.clinical_ingest(self.h.paths, "synthetic-liat", hl7, binding_path)
        self.assertEqual(corrected["state"], "QUARANTINED")
        self.assertIn("SUPERSESSION_LINEAGE_VALID", corrected["failed_gates"])

    def test_27_blank_ct_fhir_uses_data_absent_reason_not_empty_string(self):
        result, _, _ = self.h.ingest(value="Not detected")
        self.h.authorize(result["result_id"])
        output = self.h.paths.root / "blank-ct.json"
        oac.clinical_export_fhir(self.h.paths, result["result_id"], output)
        bundle = json.loads(output.read_text())
        observations = [
            entry["resource"]
            for entry in bundle["entry"]
            if entry["resource"]["resourceType"] == "Observation"
        ]
        ct = next(item for item in observations if item["id"].endswith("-3"))
        self.assertNotIn("valueString", ct)
        self.assertEqual(ct["dataAbsentReason"]["coding"][0]["code"], "unknown")

    def test_28_clinical_connection_context_actually_closes_handle(self):
        connection = oac.clinical_connect(self.h.paths)
        with connection:
            connection.execute("SELECT 1").fetchone()
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    def test_29_post_review_observation_insert_breaks_ledger(self):
        result, _, _ = self.h.ingest()
        self.h.authorize(result["result_id"])
        with oac.clinical_connect(self.h.paths) as connection:
            connection.execute(
                "INSERT INTO clinical_observations VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    result["result_id"], 99, "ST", "SYNTH-MALICIOUS",
                    "Synthetic malicious", "urn:synthetic:observation", "Detected",
                    "", "", "F", "",
                ),
            )
        with oac.clinical_connect(self.h.paths) as connection:
            with self.assertRaises(oac.IntegrityFailure):
                oac.clinical_verify_ledger(connection)

    def test_30_idempotent_retry_rejects_changed_binding(self):
        _, hl7, binding_path = self.h.ingest()
        binding = json.loads(binding_path.read_text())
        binding["order_reference"] = "ServiceRequest/SYNTH-ORDER-999"
        changed = self.h.paths.root / "changed-binding.json"
        oac.atomic_write(changed, (oac.pretty_json(binding) + "\n").encode())
        with self.assertRaisesRegex(oac.ControlError, "raw bytes or binding"):
            oac.clinical_ingest(self.h.paths, "synthetic-liat", hl7, changed)

    def test_31_signed_candidate_hash_equals_export_hash(self):
        result, _, _ = self.h.ingest()
        request, _, _ = self.h.authorize(result["result_id"])
        output = self.h.paths.root / "exact-candidate.json"
        receipt = oac.clinical_export_fhir(self.h.paths, result["result_id"], output)
        self.assertEqual(request["candidate_artifact_sha256"], receipt["artifact_sha256"])
        self.assertEqual(
            oac.hashlib.sha256(output.read_bytes()).hexdigest(),
            receipt["artifact_sha256"],
        )

    def test_32_output_collision_is_not_overwritten_and_is_recoverable(self):
        result, _, _ = self.h.ingest()
        self.h.authorize(result["result_id"])
        collision = self.h.paths.root / "collision.json"
        sentinel = b"DO NOT OVERWRITE\n"
        oac.atomic_write(collision, sentinel)
        with self.assertRaisesRegex(oac.ControlError, "refusing overwrite"):
            oac.clinical_export_fhir(self.h.paths, result["result_id"], collision)
        self.assertEqual(collision.read_bytes(), sentinel)
        status = oac.clinical_status(self.h.paths, result["result_id"])
        self.assertEqual(status["state"], "ARTIFACT_CREATED")
        recovered = oac.clinical_export_fhir(
            self.h.paths, result["result_id"], self.h.paths.root / "recovered.json"
        )
        self.assertTrue(recovered["idempotent"])

    def test_33_reviewer_insert_after_seal_is_blocked_by_database(self):
        with oac.clinical_connect(self.h.paths) as connection:
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute(
                    "INSERT INTO clinical_reviewers VALUES (?,?,?,?,?)",
                    ("synthetic_attacker", "AAAA", "00" * 32, 1, oac.format_time(oac.utc_now())),
                )

    def test_34_obx_preliminary_status_is_quarantined(self):
        hl7, binding = self.h.files()
        payload = hl7.read_bytes()[1:-2].decode()
        segments = payload.split("\r")
        for index, segment in enumerate(segments):
            if segment.startswith("OBX|"):
                fields = segment.split("|")
                fields[11] = "P"
                segments[index] = "|".join(fields)
                break
        oac.atomic_write(hl7, b"\x0b" + "\r".join(segments).encode() + b"\x1c\r")
        with self.assertRaisesRegex(oac.ControlError, "OBX_STATUS_PROFILE_MISMATCH"):
            oac.clinical_ingest(self.h.paths, "synthetic-liat", hl7, binding)
        self.assert_no_clinical_messages()

    def test_35_invalid_fhir_reference_is_denied(self):
        hl7, binding_path = self.h.files()
        binding = json.loads(binding_path.read_text())
        binding["patient_reference"] = "Patient/SYNTH BAD"
        oac.atomic_write(binding_path, (oac.pretty_json(binding) + "\n").encode())
        with self.assertRaisesRegex(oac.ControlError, "valid FHIR id"):
            oac.clinical_ingest(self.h.paths, "synthetic-liat", hl7, binding_path)

    def test_36_impossible_hl7_datetime_is_not_emitted(self):
        self.assertIsNone(oac.hl7_time_to_fhir("20261340050000-0400"))
        self.assertIsNone(oac.hl7_time_to_fhir("20260813050000+1460"))

    def test_37_noop_trigger_replacement_breaks_schema_fingerprint(self):
        with oac.clinical_connect(self.h.paths) as connection:
            connection.execute("DROP TRIGGER clinical_results_no_update")
            connection.execute(
                """
                CREATE TRIGGER clinical_results_no_update
                BEFORE UPDATE ON clinical_results WHEN 0
                BEGIN SELECT RAISE(ABORT,'clinical_results is append-only'); END
                """
            )
        with oac.clinical_connect(self.h.paths) as connection:
            with self.assertRaisesRegex(oac.IntegrityFailure, "fingerprint"):
                oac.clinical_verify_ledger(connection)

    def test_38_export_row_cannot_borrow_review_from_another_result(self):
        reviewed, _, _ = self.h.ingest()
        _, signed, _ = self.h.authorize(reviewed["result_id"])
        other, _, _ = self.h.ingest()
        artifact = "{}"
        manifest = "{}"
        with oac.clinical_connect(self.h.paths) as connection:
            connection.execute(
                "INSERT INTO clinical_exports VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    "SYNTH-EXPORT-INJECTED",
                    other["result_id"],
                    signed["review_id"],
                    "FHIR_R4_JSON_OFFLINE_COLLECTION",
                    artifact,
                    oac.hashlib.sha256(artifact.encode()).hexdigest(),
                    manifest,
                    oac.hashlib.sha256(manifest.encode()).hexdigest(),
                    oac.format_time(oac.utc_now()),
                ),
            )
        with oac.clinical_connect(self.h.paths) as connection:
            with self.assertRaises(oac.IntegrityFailure):
                oac.clinical_verify_ledger(connection)

    def test_39_clinical_output_cannot_target_state_database(self):
        with self.assertRaisesRegex(oac.ControlError, "may not target"):
            oac.clinical_write_output(
                self.h.paths,
                oac.clinical_database_path(self.h.paths),
                b"unsafe",
                "test output",
            )

    def test_40_obx_display_canary_is_rejected_before_storage(self):
        hl7, binding = self.h.files()
        oac.atomic_write(
            hl7,
            hl7.read_bytes().replace(
                b"Synthetic influenza A^urn:synthetic:observation",
                b"SYNTH ALICE SMITH DOB 19700101^urn:synthetic:observation",
                1,
            ),
        )
        with self.assertRaisesRegex(oac.ControlError, "OBX_IDENTIFIER_PROFILE_MISMATCH") as caught:
            oac.clinical_ingest(self.h.paths, "synthetic-liat", hl7, binding)
        self.assertNotIn("ALICE", str(caught.exception))
        self.assert_no_clinical_messages()

    def test_41_obx_unit_canary_is_rejected_before_storage(self):
        hl7, binding = self.h.files()
        raw = hl7.read_bytes().replace(
            b"|0|||", b"|ALICE SMITH DOB 19700101|||", 1
        )
        oac.atomic_write(hl7, raw)
        with self.assertRaisesRegex(oac.ControlError, "OBX_UNITS_PROFILE_MISMATCH") as caught:
            oac.clinical_ingest(self.h.paths, "synthetic-liat", hl7, binding)
        self.assertNotIn("ALICE", str(caught.exception))
        self.assert_no_clinical_messages()

    def test_42_obx_trailing_identifier_component_is_rejected(self):
        hl7, binding = self.h.files()
        oac.atomic_write(
            hl7,
            hl7.read_bytes().replace(
                b"urn:synthetic:observation||0|0|||||F",
                b"urn:synthetic:observation^EXTRA||0|0|||||F",
                1,
            ),
        )
        with self.assertRaisesRegex(oac.ControlError, "OBX_IDENTIFIER_PROFILE_MISMATCH"):
            oac.clinical_ingest(self.h.paths, "synthetic-liat", hl7, binding)
        self.assert_no_clinical_messages()

    def test_43_obx_order_is_exact(self):
        hl7, binding = self.h.files()
        payload = hl7.read_bytes()[1:-2].decode()
        segments = payload.split("\r")
        segments[5], segments[7] = segments[7], segments[5]
        oac.atomic_write(hl7, b"\x0b" + "\r".join(segments).encode() + b"\x1c\r")
        with self.assertRaisesRegex(oac.ControlError, "OBX_SET_ID_PROFILE_MISMATCH"):
            oac.clinical_ingest(self.h.paths, "synthetic-liat", hl7, binding)
        self.assert_no_clinical_messages()

    def test_44_external_assay_map_must_equal_built_in_profile(self):
        with tempfile.TemporaryDirectory(prefix="oac-unpinned-map-") as directory:
            path = Path(directory) / "map.json"
            path.write_text(
                oac.pretty_json(
                    {
                        "SYNTH-FLU": {
                            "display": "Synthetic Jane Doe assay",
                            "local_system": "urn:synthetic:assay",
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(oac.ControlError, "built-in synthetic assay map"):
                oac.load_assay_map(path)

    def test_45_invalid_correction_does_not_reserve_predecessor(self):
        original, _, _ = self.h.ingest()
        self.h.authorize(original["result_id"])
        oac.clinical_export_fhir(
            self.h.paths, original["result_id"], self.h.paths.root / "original-for-lineage.json"
        )

        bad_hl7, bad_binding_path = self.h.files()
        bad_binding = json.loads(bad_binding_path.read_text())
        bad_binding["supersedes_result_id"] = original["result_id"]
        oac.atomic_write(bad_binding_path, (oac.pretty_json(bad_binding) + "\n").encode())
        invalid = oac.clinical_ingest(
            self.h.paths, "synthetic-liat", bad_hl7, bad_binding_path
        )
        self.assertEqual(invalid["state"], "QUARANTINED")

        valid_hl7, valid_binding_path = self.h.files(
            subject="SYNTH-SUBJECT-001", order="SYNTH-SOURCE-ORDER-001"
        )
        valid_raw = valid_hl7.read_bytes().replace(
            f"SYNTH-REPORT-{self.h.counter:03d}".encode(), b"SYNTH-REPORT-001", 1
        )
        oac.atomic_write(valid_hl7, valid_raw)
        valid_binding = json.loads(valid_binding_path.read_text())
        valid_binding["patient_reference"] = "Patient/SYNTH-PATIENT-001"
        valid_binding["order_reference"] = "ServiceRequest/SYNTH-ORDER-001"
        valid_binding["specimen_reference"] = "Specimen/SYNTH-SPECIMEN-001"
        valid_binding["supersedes_result_id"] = original["result_id"]
        oac.atomic_write(valid_binding_path, (oac.pretty_json(valid_binding) + "\n").encode())
        valid = oac.clinical_ingest(
            self.h.paths, "synthetic-liat", valid_hl7, valid_binding_path
        )
        self.assertEqual(valid["state"], "PENDING_REVIEW")
        with oac.clinical_connect(self.h.paths) as connection:
            self.assertIsNone(connection.execute(
                "SELECT successor_result_id FROM clinical_supersession_claims "
                "WHERE predecessor_result_id=?",
                (original["result_id"],),
            ).fetchone())
        self.h.authorize(valid["result_id"])
        with oac.clinical_connect(self.h.paths) as connection:
            claim = connection.execute(
                "SELECT successor_result_id FROM clinical_supersession_claims "
                "WHERE predecessor_result_id=?",
                (original["result_id"],),
            ).fetchone()
            self.assertEqual(claim["successor_result_id"], valid["result_id"])

    def test_46_unreviewed_correction_does_not_reserve_predecessor(self):
        original, _, _ = self.h.ingest()
        self.h.authorize(original["result_id"])
        oac.clinical_export_fhir(
            self.h.paths, original["result_id"], self.h.paths.root / "original-46.json"
        )

        corrections = []
        for _ in range(2):
            hl7, binding_path = self.h.files(
                subject="SYNTH-SUBJECT-001", order="SYNTH-SOURCE-ORDER-001"
            )
            oac.atomic_write(
                hl7,
                hl7.read_bytes().replace(
                    f"SYNTH-REPORT-{self.h.counter:03d}".encode(), b"SYNTH-REPORT-001", 1
                ),
            )
            binding = json.loads(binding_path.read_text())
            binding.update(
                {
                    "order_reference": "ServiceRequest/SYNTH-ORDER-001",
                    "patient_reference": "Patient/SYNTH-PATIENT-001",
                    "specimen_reference": "Specimen/SYNTH-SPECIMEN-001",
                    "supersedes_result_id": original["result_id"],
                }
            )
            oac.atomic_write(binding_path, (oac.pretty_json(binding) + "\n").encode())
            corrections.append(
                oac.clinical_ingest(self.h.paths, "synthetic-liat", hl7, binding_path)
            )
        self.assertTrue(all(item["state"] == "PENDING_REVIEW" for item in corrections))
        self.h.authorize(corrections[1]["result_id"])
        request = oac.build_clinical_review(
            self.h.paths,
            corrections[0]["result_id"],
            "synthetic_reviewer",
            oac.MAX_TTL_SECONDS,
        )
        signed = oac.sign_clinical_review(self.h.paths, request, self.h.key)
        with self.assertRaisesRegex(oac.ControlError, "another correction"):
            oac.apply_clinical_review(self.h.paths, signed)

    def test_47_output_publish_race_never_overwrites_competitor(self):
        target = self.h.paths.root / "race-output.json"
        real_link = oac.os.link

        def racing_link(source, destination, *args, **kwargs):
            Path(destination).write_bytes(b"SENTINEL\n")
            return real_link(source, destination, *args, **kwargs)

        oac.os.link = racing_link
        try:
            with self.assertRaisesRegex(oac.ControlError, "refusing overwrite"):
                oac.safe_new_or_identical_output(target, b"REQUESTED\n", "race test")
        finally:
            oac.os.link = real_link
        self.assertEqual(target.read_bytes(), b"SENTINEL\n")

    def test_48_apply_requires_sealed_reviewer_trust(self):
        with tempfile.TemporaryDirectory(prefix="oac-unsealed-") as original_dir, tempfile.TemporaryDirectory(
            prefix="oac-sealed-clone-"
        ) as clone_dir:
            original = oac.state_paths(Path(original_dir).resolve())
            clone = oac.state_paths(Path(clone_dir).resolve())
            oac.clinical_initialize(original)
            assay_map = original.root / "assay-map.json"
            oac.atomic_write(
                assay_map,
                (oac.pretty_json(oac.CLINICAL_SYNTHETIC_ASSAY_MAP) + "\n").encode(),
            )
            oac.clinical_add_source(
                original,
                oac.CLINICAL_SYNTHETIC_SOURCE_ID,
                "MOCK",
                oac.CLINICAL_PROFILE_ID,
                oac.CLINICAL_SYNTHETIC_SENDER_APPLICATION,
                oac.CLINICAL_SYNTHETIC_SENDER_FACILITY,
                assay_map,
            )
            key = oac.Ed25519PrivateKey.generate()
            oac.clinical_add_reviewer_raw(
                original, "synthetic_reviewer", key.public_key(), actor="test"
            )
            message = original.root / "message.hl7"
            binding = original.root / "binding.json"
            oac.atomic_write(
                message,
                oac.synthetic_hl7_message(
                    "SYNTH-MSG-001",
                    "SYNTH-SOURCE-ORDER-001",
                    "SYNTH-REPORT-001",
                    "SYNTH-SUBJECT-001",
                ),
            )
            oac.atomic_write(
                binding,
                (
                    oac.pretty_json(
                        oac.synthetic_binding(
                            "SYNTH-SUBJECT-001", "SYNTH-SOURCE-ORDER-001", "001"
                        )
                    )
                    + "\n"
                ).encode(),
            )
            result = oac.clinical_ingest(
                original, oac.CLINICAL_SYNTHETIC_SOURCE_ID, message, binding
            )
            shutil.copy2(oac.clinical_database_path(original), oac.clinical_database_path(clone))
            oac.clinical_seal_reviewer_trust(clone)
            request = oac.build_clinical_review(
                clone, result["result_id"], "synthetic_reviewer", oac.MAX_TTL_SECONDS
            )
            signed = oac.sign_clinical_review(clone, request, key)
            with self.assertRaisesRegex(oac.ControlError, "must be sealed"):
                oac.apply_clinical_review(original, signed)

    def test_49_ledger_rejects_count_matched_semantic_audit_mismatch(self):
        config = "{}"
        config_sha = oac.hashlib.sha256(config.encode()).hexdigest()
        with oac.clinical_connect(self.h.paths) as connection:
            connection.execute(
                "INSERT INTO clinical_sources VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    "synthetic-injected",
                    "MOCK",
                    oac.CLINICAL_PROFILE_ID,
                    "LIAT-SIM",
                    "LAB-SIM",
                    oac.canonical_json(oac.CLINICAL_SYNTHETIC_ASSAY_MAP).decode(),
                    config,
                    config_sha,
                    1,
                    oac.format_time(oac.utc_now()),
                ),
            )
            oac.clinical_append_audit(
                connection,
                "CLINICAL_SOURCE_ADDED",
                None,
                "local-operator",
                {
                    "config_sha256": config_sha,
                    "mode": "MOCK",
                    "profile_id": oac.CLINICAL_PROFILE_ID,
                    "source_id": "different-source-id",
                },
            )
        with oac.clinical_connect(self.h.paths) as connection:
            with self.assertRaisesRegex(oac.IntegrityFailure, "one-to-one semantically bound"):
                oac.clinical_verify_ledger(connection)

    def test_50_superseded_original_artifact_remains_recoverable(self):
        original, _, _ = self.h.ingest()
        self.h.authorize(original["result_id"])
        first = self.h.paths.root / "original-50.json"
        original_receipt = oac.clinical_export_fhir(
            self.h.paths, original["result_id"], first
        )
        hl7, binding_path = self.h.files(
            subject="SYNTH-SUBJECT-001", order="SYNTH-SOURCE-ORDER-001"
        )
        oac.atomic_write(
            hl7,
            hl7.read_bytes().replace(
                f"SYNTH-REPORT-{self.h.counter:03d}".encode(), b"SYNTH-REPORT-001", 1
            ),
        )
        binding = json.loads(binding_path.read_text())
        binding.update(
            {
                "order_reference": "ServiceRequest/SYNTH-ORDER-001",
                "patient_reference": "Patient/SYNTH-PATIENT-001",
                "specimen_reference": "Specimen/SYNTH-SPECIMEN-001",
                "supersedes_result_id": original["result_id"],
            }
        )
        oac.atomic_write(binding_path, (oac.pretty_json(binding) + "\n").encode())
        correction = oac.clinical_ingest(
            self.h.paths, "synthetic-liat", hl7, binding_path
        )
        self.h.authorize(correction["result_id"])
        oac.clinical_export_fhir(
            self.h.paths, correction["result_id"], self.h.paths.root / "correction-50.json"
        )
        recovered = self.h.paths.root / "original-recovered-50.json"
        receipt = oac.clinical_export_fhir(
            self.h.paths, original["result_id"], recovered
        )
        self.assertTrue(receipt["idempotent"])
        self.assertEqual(receipt["state"], "SUPERSEDED")
        self.assertEqual(receipt["artifact_sha256"], original_receipt["artifact_sha256"])
        self.assertEqual(recovered.read_bytes(), first.read_bytes())

    def test_51_state_database_sidecars_are_forbidden_outputs(self):
        databases = (self.h.paths.database, oac.clinical_database_path(self.h.paths))
        for database in databases:
            for suffix in ("-journal", "-shm", "-wal"):
                with self.subTest(database=database.name, suffix=suffix):
                    with self.assertRaisesRegex(oac.ControlError, "may not target"):
                        oac.clinical_write_output(
                            self.h.paths, Path(str(database) + suffix), b"unsafe", "sidecar"
                        )

    def test_52_fhir_provenance_does_not_claim_clinical_verifier(self):
        result, _, _ = self.h.ingest()
        self.h.authorize(result["result_id"])
        output = self.h.paths.root / "provenance-52.json"
        oac.clinical_export_fhir(self.h.paths, result["result_id"], output)
        bundle = json.loads(output.read_text())
        provenance = next(
            entry["resource"]
            for entry in bundle["entry"]
            if entry["resource"]["resourceType"] == "Provenance"
        )
        self.assertNotIn("Verifier", json.dumps(provenance))
        self.assertIn("not clinical verification", provenance["activity"]["text"])
        self.assertEqual(
            provenance["agent"][0]["who"]["identifier"]["value"],
            oac.CLINICAL_TRANSFORM_VERSION,
        )

    def test_53_repeated_trust_seal_is_idempotent_and_audited_once(self):
        repeated = oac.clinical_seal_reviewer_trust(self.h.paths)
        self.assertTrue(repeated["idempotent"])
        with oac.clinical_connect(self.h.paths) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM clinical_audit_events "
                "WHERE event_type='CLINICAL_REVIEW_TRUST_SEALED'"
            ).fetchone()[0]
            verified = oac.clinical_verify_ledger(connection)
        self.assertEqual(count, 1)
        self.assertEqual(verified["integrity"], "VERIFIED_LOCAL_CLINICAL_HASH_CHAIN")

    def test_54_every_clinical_connection_enforces_durability_and_completion(self):
        for _ in range(2):
            with oac.clinical_connect(self.h.paths) as connection:
                self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
                self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
                self.assertGreaterEqual(connection.execute("PRAGMA synchronous").fetchone()[0], 2)
                self.assertEqual(
                    connection.execute(
                        "SELECT value FROM clinical_metadata "
                        "WHERE key='initialization_complete'"
                    ).fetchone()[0],
                    "1",
                )

    def test_55_failed_clinical_initialization_is_unpublished_and_retryable(self):
        with tempfile.TemporaryDirectory(prefix="oac-clinical-init-failure-") as directory:
            paths = oac.state_paths(Path(directory).resolve())
            real_append = oac.clinical_append_audit

            def fail_initial_audit(*args, **kwargs):
                if args[1] == "CLINICAL_STATE_INITIALIZED":
                    raise RuntimeError("injected initialization failure")
                return real_append(*args, **kwargs)

            oac.clinical_append_audit = fail_initial_audit
            try:
                with self.assertRaisesRegex(RuntimeError, "injected initialization failure"):
                    oac.clinical_initialize(paths)
            finally:
                oac.clinical_append_audit = real_append
            self.assertFalse(oac.clinical_database_path(paths).exists())
            self.assertEqual(list(paths.root.glob(".clinical.sqlite3.*.initialize*")), [])
            oac.clinical_initialize(paths)
            with oac.clinical_connect(paths) as connection:
                self.assertEqual(
                    oac.clinical_verify_ledger(connection)["integrity"],
                    "VERIFIED_LOCAL_CLINICAL_HASH_CHAIN",
                )

    def test_56_partial_clinical_database_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory(prefix="oac-clinical-partial-") as directory:
            paths = oac.state_paths(Path(directory).resolve())
            paths.root.mkdir(parents=True, exist_ok=True)
            sqlite3.connect(oac.clinical_database_path(paths)).close()
            with self.assertRaisesRegex(oac.ControlError, "pre-existing clinical database") as caught:
                oac.clinical_initialize(paths)
            self.assertEqual(caught.exception.code, "CLINICAL_STATE_INCOMPLETE")
            with self.assertRaises(oac.ControlError) as connect_error:
                oac.clinical_connect(paths)
            self.assertEqual(connect_error.exception.code, "CLINICAL_STATE_INCOMPLETE")
            self.assertTrue(oac.clinical_database_path(paths).exists())

    def test_57_versions_status_and_correction_fhir_bind_exact_predecessor(self):
        original, _, _ = self.h.ingest()
        self.h.authorize(original["result_id"])
        original_path = self.h.paths.root / "v1.json"
        original_receipt = oac.clinical_export_fhir(
            self.h.paths, original["result_id"], original_path
        )
        before = oac.clinical_status(self.h.paths, original["result_id"])
        self.assertEqual(before["result_version"], 1)
        self.assertIsNone(before["successor_result_id"])

        correction = self.ingest_exact_correction(original["result_id"])
        pending = oac.clinical_status(self.h.paths, correction["result_id"])
        self.assertEqual(pending["result_version"], 2)
        self.assertEqual(pending["supersedes_result_id"], original["result_id"])
        self.assertFalse(pending["successor_authorized"])
        self.h.authorize(correction["result_id"])
        selected = oac.clinical_status(self.h.paths, original["result_id"])
        self.assertTrue(selected["successor_authorized"])
        self.assertEqual(selected["successor_result_id"], correction["result_id"])

        correction_path = self.h.paths.root / "v2.json"
        oac.clinical_export_fhir(self.h.paths, correction["result_id"], correction_path)
        bundle = json.loads(correction_path.read_text())
        report = next(
            entry["resource"]
            for entry in bundle["entry"]
            if entry["resource"]["resourceType"] == "DiagnosticReport"
        )
        observations = [
            entry["resource"]
            for entry in bundle["entry"]
            if entry["resource"]["resourceType"] == "Observation"
        ]
        provenance = next(
            entry["resource"]
            for entry in bundle["entry"]
            if entry["resource"]["resourceType"] == "Provenance"
        )
        revision = next(entity for entity in provenance["entity"] if entity["role"] == "revision")
        self.assertEqual(report["status"], "corrected")
        self.assertTrue(all(item["status"] == "final" for item in observations))
        self.assertIn(
            "source-hl7-status-F",
            {tag["code"] for tag in report["meta"]["tag"]},
        )
        self.assertIn(
            {"system": "urn:owned-agent-control:result-version", "value": "2"},
            report["identifier"],
        )
        self.assertEqual(
            revision["what"]["identifier"]["value"],
            original_receipt["artifact_sha256"],
        )
        self.assertTrue(revision["what"]["reference"].endswith(original["result_id"]))
        self.assertEqual(
            oac.clinical_status(self.h.paths, original["result_id"])["state"],
            "SUPERSEDED",
        )

    def test_58_second_order_correction_references_immediate_predecessor(self):
        original, _, _ = self.h.ingest()
        self.h.authorize(original["result_id"])
        oac.clinical_export_fhir(
            self.h.paths, original["result_id"], self.h.paths.root / "chain-v1.json"
        )
        second = self.ingest_exact_correction(original["result_id"])
        self.h.authorize(second["result_id"])
        second_receipt = oac.clinical_export_fhir(
            self.h.paths, second["result_id"], self.h.paths.root / "chain-v2.json"
        )
        third = self.ingest_exact_correction(second["result_id"])
        self.assertEqual(oac.clinical_status(self.h.paths, third["result_id"])["result_version"], 3)
        self.h.authorize(third["result_id"])
        third_path = self.h.paths.root / "chain-v3.json"
        oac.clinical_export_fhir(self.h.paths, third["result_id"], third_path)
        bundle = json.loads(third_path.read_text())
        provenance = next(
            entry["resource"]
            for entry in bundle["entry"]
            if entry["resource"]["resourceType"] == "Provenance"
        )
        revision = next(entity for entity in provenance["entity"] if entity["role"] == "revision")
        self.assertTrue(revision["what"]["reference"].endswith(second["result_id"]))
        self.assertEqual(revision["what"]["identifier"]["value"], second_receipt["artifact_sha256"])
        self.assertEqual(
            oac.clinical_status(self.h.paths, second["result_id"])["successor_result_id"],
            third["result_id"],
        )

    def test_59_unknown_correction_predecessor_is_controlled_and_not_stored(self):
        hl7, binding_path = self.h.files()
        binding = json.loads(binding_path.read_text())
        binding["supersedes_result_id"] = str(oac.uuid.uuid4())
        oac.atomic_write(binding_path, (oac.pretty_json(binding) + "\n").encode())
        with self.assertRaises(oac.ControlError) as caught:
            oac.clinical_ingest(self.h.paths, "synthetic-liat", hl7, binding_path)
        self.assertEqual(caught.exception.code, "SUPERSESSION_PREDECESSOR_NOT_FOUND")
        self.assert_no_clinical_messages()

    def test_60_detached_authorization_manifest_verifies_with_trust_anchor(self):
        result, _, _ = self.h.ingest()
        request, _, _ = self.h.authorize(result["result_id"])
        artifact = self.h.paths.root / "authorized-60.json"
        receipt = oac.clinical_export_fhir(self.h.paths, result["result_id"], artifact)
        manifest_path = Path(receipt["authorization_manifest_out"])
        verified = oac.verify_clinical_export_authorization(
            artifact, manifest_path, self.h.public_key_path
        )
        self.assertTrue(verified["artifact_exactly_bound"])
        self.assertTrue(verified["signature_cryptographically_valid"])
        self.assertTrue(verified["trusted_reviewer_key_matched"])
        self.assertFalse(verified["ledger_correlation_verified"])
        self.assertEqual(receipt["artifact_sha256"], request["candidate_artifact_sha256"])
        self.assertEqual(
            receipt["authorization_manifest_sha256"],
            oac.hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        )
        self.assertNotIn("PRIVATE KEY", manifest_path.read_text())

    def test_61_export_verifier_rejects_artifact_signature_and_key_tampering(self):
        result, _, _ = self.h.ingest()
        self.h.authorize(result["result_id"])
        artifact = self.h.paths.root / "authorized-61.json"
        receipt = oac.clinical_export_fhir(self.h.paths, result["result_id"], artifact)
        manifest_path = Path(receipt["authorization_manifest_out"])

        changed_artifact = self.h.paths.root / "changed-61.json"
        oac.atomic_write(changed_artifact, artifact.read_bytes() + b" ")
        with self.assertRaises(oac.ControlError):
            oac.verify_clinical_export_authorization(
                changed_artifact, manifest_path, self.h.public_key_path
            )

        manifest = json.loads(manifest_path.read_text())
        signature = manifest["authorization"]["envelope"]["authorization"]["signature"]
        manifest["authorization"]["envelope"]["authorization"]["signature"] = (
            ("A" if signature[0] != "A" else "B") + signature[1:]
        )
        changed_manifest = self.h.paths.root / "changed-61.authorization.json"
        oac.atomic_write(changed_manifest, (oac.pretty_json(manifest) + "\n").encode())
        with self.assertRaises(oac.ControlError):
            oac.verify_clinical_export_authorization(
                artifact, changed_manifest, self.h.public_key_path
            )

        attacker = oac.Ed25519PrivateKey.generate()
        attacker_path = self.h.paths.root / "attacker-public.pem"
        oac.atomic_write(
            attacker_path,
            attacker.public_key().public_bytes(
                encoding=oac.serialization.Encoding.PEM,
                format=oac.serialization.PublicFormat.SubjectPublicKeyInfo,
            ),
        )
        with self.assertRaisesRegex(oac.ControlError, "trust anchor"):
            oac.verify_clinical_export_authorization(
                artifact, manifest_path, attacker_path
            )

    def test_62_idempotent_export_recovers_identical_artifact_and_manifest(self):
        result, _, _ = self.h.ingest()
        self.h.authorize(result["result_id"])
        first = self.h.paths.root / "idempotent-first.json"
        second = self.h.paths.root / "idempotent-second.json"
        first_receipt = oac.clinical_export_fhir(self.h.paths, result["result_id"], first)
        second_receipt = oac.clinical_export_fhir(self.h.paths, result["result_id"], second)
        self.assertTrue(second_receipt["idempotent"])
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(
            Path(first_receipt["authorization_manifest_out"]).read_bytes(),
            Path(second_receipt["authorization_manifest_out"]).read_bytes(),
        )
        self.assertEqual(
            first_receipt["authorization_manifest_sha256"],
            second_receipt["authorization_manifest_sha256"],
        )

    @unittest.skipUnless(os.name == "nt", "controller uses Windows Job Objects")
    def test_63_controller_connections_enforce_durability_and_completion(self):
        with tempfile.TemporaryDirectory(prefix="oac-controller-v2-") as directory:
            paths = oac.state_paths(Path(directory).resolve())
            oac.initialize_state(paths)
            with oac.connect(paths) as connection:
                self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
                self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
                self.assertGreaterEqual(connection.execute("PRAGMA synchronous").fetchone()[0], 2)
                self.assertEqual(
                    connection.execute(
                        "SELECT value FROM metadata WHERE key='initialization_complete'"
                    ).fetchone()[0],
                    "1",
                )
                self.assertEqual(oac.verify_audit(connection)["integrity"], "VERIFIED_LOCAL_HASH_CHAIN")

    @unittest.skipUnless(os.name == "nt", "controller uses Windows Job Objects")
    def test_64_failed_controller_initialization_is_unpublished_and_retryable(self):
        with tempfile.TemporaryDirectory(prefix="oac-controller-init-failure-") as directory:
            paths = oac.state_paths(Path(directory).resolve())
            real_append = oac.append_audit

            def fail_initial_audit(*args, **kwargs):
                if args[1] == "STATE_INITIALIZED":
                    raise RuntimeError("injected controller initialization failure")
                return real_append(*args, **kwargs)

            oac.append_audit = fail_initial_audit
            try:
                with self.assertRaisesRegex(RuntimeError, "injected controller initialization"):
                    oac.initialize_state(paths)
            finally:
                oac.append_audit = real_append
            self.assertFalse(paths.database.exists())
            self.assertEqual(list(paths.root.glob(".control.sqlite3.*.initialize*")), [])
            oac.initialize_state(paths)
            with oac.connect(paths) as connection:
                self.assertEqual(oac.verify_audit(connection)["integrity"], "VERIFIED_LOCAL_HASH_CHAIN")

    @unittest.skipUnless(os.name == "nt", "controller uses Windows Job Objects")
    def test_65_partial_controller_database_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory(prefix="oac-controller-partial-") as directory:
            paths = oac.state_paths(Path(directory).resolve())
            paths.root.mkdir(parents=True, exist_ok=True)
            sqlite3.connect(paths.database).close()
            with self.assertRaises(oac.ControlError) as caught:
                oac.initialize_state(paths)
            self.assertEqual(caught.exception.code, "STATE_INCOMPLETE")
            with self.assertRaises(oac.ControlError) as connect_error:
                oac.connect(paths)
            self.assertEqual(connect_error.exception.code, "STATE_INCOMPLETE")
            self.assertTrue(paths.database.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
