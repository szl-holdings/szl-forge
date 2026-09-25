"""Synthetic regression fixtures. They are not live model evaluation evidence."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "audit_hf_estate.py"
SPEC = importlib.util.spec_from_file_location("audit_hf_estate", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class AuditClassificationTests(unittest.TestCase):
    def fixture(self, body="Measured research candidate. Not promotable. No autonomy. No deployment.", evidence=None, extra=None):
        entry = {"repo": "example/research", "type": "model", "class": "measured_research_candidate",
                 "evidence_files": ["eval_report.json"],
                 "required_boundaries": ["publication_eligible_false", "no_autonomy", "no_deployment"]}
        if extra:
            entry.update(extra)
        evidence = evidence if evidence is not None else {"label": "MEASURED", "publication_eligible": False}
        contents = {"README.md": "---\nlanguage: en\n---\n" + body,
                    "eval_report.json": json.dumps(evidence)}
        snap = {"repo": entry["repo"], "type": entry["type"], "revision": "a" * 40,
                "requested_revision": "main",
                "files": [*contents, "adapter_model.safetensors"], "contents": contents,
                "file_sha256": {p: hashlib.sha256(c.encode()).hexdigest() for p, c in contents.items()},
                "artifact_sha256": {"adapter_model.safetensors": "b" * 64}, "errors": []}
        return entry, snap

    def classify(self, *args, **kwargs):
        entry, snap = self.fixture(*args, **kwargs)
        return audit.classify_repository(entry, snap)

    def codes(self, result):
        return {f["code"] for f in result["findings"]}

    def test_scoped_consistency_pass_never_promotes(self):
        result = self.classify()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["promotion_verdict"], "NOT_ASSESSED")
        self.assertEqual(result["permission_status"], "NOT_PROBED_READ_ONLY")

    def test_archive_selection_requires_exact_structured_metadata(self):
        expected = [{"config_name": "default", "data_files": [
            {"split": "train", "path": "runs/**/summary.json"}]}]
        cases = [
            ("configs:\n- config_name: default\n  data_files:\n  - split: train\n    path: runs/**/summary.json\n", True),
            ("language: en\n", False),
            ("configs:\n- config_name: default\n  data_files:\n  - split: train\n    path: runs/**/*.json\n", False),
            ("configs:\n- config_name: default\n  data_files:\n  - split: train\n    path: runs/**/summary.json\n  - split: train\n    path: runs/**/receipt.json\n", False),
        ]
        for metadata, matches in cases:
            with self.subTest(metadata=metadata):
                entry, snap = self.fixture(extra={"checks": {"frontmatter_equals": [
                    {"field": "configs", "value": expected}]}})
                snap["contents"]["README.md"] = ("---\n" + metadata + "---\n"
                    "Summary index uses runs/**/summary.json. Not promotable. No autonomy. No deployment.")
                snap["file_sha256"]["README.md"] = hashlib.sha256(
                    snap["contents"]["README.md"].encode()).hexdigest()
                result = audit.classify_repository(entry, snap)
                self.assertEqual("FRONTMATTER_VALUE_MISMATCH" not in self.codes(result), matches)

    def test_present_no_eval_claim_conflicts_with_measured(self):
        result = self.classify("evals: none-this-run\nNot promotable. No autonomy. No deployment.")
        self.assertIn("MEASURED_RECORD_CONTRADICTS_NO_EVAL", self.codes(result))

    def test_historical_none_this_run_disclosure_is_not_current_denial(self):
        result = self.classify("MEASURED gate records exist.\nOlder training receipts retain NONE_THIS_RUN.\n"
                               "Exact public-byte artifact binding is incomplete.\nNot promotable. No autonomy. No deployment.")
        self.assertEqual(result["status"], "PASS", result)

    def test_historical_heading_skips_obsolete_claim(self):
        result = self.classify("## Current evidence\nMEASURED named-N only.\nNot promotable. No autonomy. No deployment.\n"
                               "## Historical receipt\nevals: none-this-run\n")
        self.assertNotIn("MEASURED_RECORD_CONTRADICTS_NO_EVAL", self.codes(result))

    def test_historical_only_boundaries_do_not_satisfy_current_requirements(self):
        for heading in ("Historical", "Deprecated"):
            with self.subTest(heading=heading):
                result = self.classify(
                    "## Current\nPromotion, autonomy and deployment are permitted.\n"
                    f"## {heading}\nNot promotable. No autonomy. No deployment.\n",
                    extra={"required_boundaries": ["publication_eligible_false", "no_autonomy", "no_deployment"],
                           "checks": {"required_patterns": [{"pattern": "No deployment"}]}})
                self.assertIn("REQUIRED_DISCLOSURE_MISSING", self.codes(result))
                self.assertNotEqual(result["status"], "PASS")
                missing = [finding for finding in result["findings"] if finding["code"] == "REQUIRED_BOUNDARY_MISSING"]
                self.assertEqual(len(missing), 3)

    def test_explicit_document_scope_can_require_a_history_label_only(self):
        body = "Not promotable. No autonomy. No deployment.\n## Historical examples\nOld instructions."
        result = self.classify(body, extra={"checks": {"required_patterns": [
            {"pattern": "^## Historical examples", "scope": "document"}]}})
        self.assertEqual(result["status"], "PASS")
        invalid = self.classify(body, extra={"checks": {"required_patterns": [
            {"pattern": "No autonomy", "scope": "unknown"}]}})
        self.assertIn("UNKNOWN_PATTERN_SCOPE", self.codes(invalid))

    def test_training_loss_is_not_measured_eval(self):
        result = self.classify("evals: none-this-run\nNot promotable. No autonomy. No deployment.",
                               {"finalTrainLoss": 0.0537, "status": "TRAINED", "evals": "none-this-run"})
        self.assertEqual(result["measured_records"], [])
        self.assertNotIn("MEASURED_RECORD_CONTRADICTS_NO_EVAL", self.codes(result))

    def test_unavailable_required_evidence_holds(self):
        entry, snap = self.fixture()
        snap["files"].remove("eval_report.json")
        del snap["contents"]["eval_report.json"]
        self.assertEqual(audit.classify_repository(entry, snap)["status"], "HOLD")

    def test_fetch_failure_holds_even_if_readme_has_pass_keywords(self):
        entry, snap = self.fixture("PASS QUALIFIED SAFE. Not promotable. No autonomy. No deployment.")
        snap["errors"].append({"code": "FETCH_ERROR", "message": "HTTP 503"})
        self.assertEqual(audit.classify_repository(entry, snap)["status"], "HOLD")

    def test_mutated_snapshot_content_holds(self):
        entry, snap = self.fixture()
        snap["contents"]["eval_report.json"] = '{"label":"QUALIFIED"}'
        result = audit.classify_repository(entry, snap)
        self.assertIn("SOURCE_HASH_MISMATCH", self.codes(result))
        self.assertNotEqual(result["status"], "PASS")

    def test_branch_name_is_not_immutable_revision(self):
        entry, snap = self.fixture()
        snap["revision"] = "main"
        self.assertIn("IMMUTABLE_REVISION_MISSING", self.codes(audit.classify_repository(entry, snap)))

    def test_yaml_invalid_conflicts(self):
        entry, snap = self.fixture()
        snap["contents"]["README.md"] = "---\nlanguage: [en\n---\nNo autonomy. No deployment. Not promotable."
        snap["file_sha256"]["README.md"] = hashlib.sha256(snap["contents"]["README.md"].encode()).hexdigest()
        self.assertIn("YAML_FRONTMATTER_INVALID", self.codes(audit.classify_repository(entry, snap)))

    def test_local_link_to_absent_receipt_conflicts(self):
        result = self.classify("[record](missing.json)\nNot promotable. No autonomy. No deployment.")
        self.assertIn("BROKEN_LOCAL_LINK", self.codes(result))

    def test_valid_directory_link_is_allowed(self):
        entry, snap = self.fixture("[receipts](runs/)\nNot promotable. No autonomy. No deployment.")
        snap["files"].append("runs/run.json")
        self.assertNotIn("BROKEN_LOCAL_LINK", self.codes(audit.classify_repository(entry, snap)))

    def test_traversal_link_conflicts(self):
        result = self.classify("[record](%2e%2e/secrets.json)\nNot promotable. No autonomy. No deployment.")
        self.assertIn("UNSAFE_LOCAL_LINK", self.codes(result))

    def test_missing_load_target_conflicts_but_historical_code_does_not(self):
        current = self.classify('```python\njoblib.load("model.joblib")\n```\nNot promotable. No autonomy. No deployment.')
        historical = self.classify('Not promotable. No autonomy. No deployment.\n## Historical example\n```python\njoblib.load("model.joblib")\n```')
        self.assertIn("LOAD_TARGET_ABSENT", self.codes(current))
        self.assertNotIn("LOAD_TARGET_ABSENT", self.codes(historical))

    def test_binding_is_missing_without_exact_hash(self):
        result = self.classify(extra={"checks": {"artifact_binding": {"artifact_paths": ["adapter_model.safetensors"]}}})
        self.assertEqual(result["status"], "MISSING_BINDING")

    def test_exact_artifact_hash_satisfies_configured_binding(self):
        result = self.classify(evidence={"label": "MEASURED", "artifact": {"file": "adapter_model.safetensors", "sha256": "b" * 64}},
                               extra={"checks": {"artifact_binding": {"artifact_paths": ["adapter_model.safetensors"], "bindings": [{
                                   "artifact_path": "adapter_model.safetensors", "evidence_file": "eval_report.json",
                                   "artifact_field": "artifact.file", "sha256_field": "artifact.sha256"}]}}})
        self.assertEqual(result["status"], "PASS")

    def test_unrelated_hash_does_not_satisfy_artifact_binding(self):
        result = self.classify(evidence={"label": "MEASURED", "sourceSha256": "c" * 64},
                               extra={"checks": {"artifact_binding": {"artifact_paths": ["adapter_model.safetensors"]}}})
        self.assertEqual(result["status"], "MISSING_BINDING")

    def test_unsupported_heldout_fraction_requires_receipt(self):
        result = self.classify("Held-out generate: MEASURED grounding 4/5, abstain 0/6.\nNot promotable. No autonomy. No deployment.",
                               extra={"evidence_files": []})
        self.assertIn("EVALUATION_RECEIPT_MISSING", self.codes(result))

    def test_explicit_unsupported_historical_fraction_is_not_eval_claim(self):
        result = self.classify("Historical unsupported held-out generate claim: 4/5.\nNot promotable. No autonomy. No deployment.",
                               extra={"evidence_files": []})
        self.assertNotIn("EVALUATION_RECEIPT_MISSING", self.codes(result))

    def test_numeric_dev_claim_catches_12_of_12_vs_11_of_12(self):
        result = self.classify("DEV n=12 MEASURED 12/12.\nNot promotable. No autonomy. No deployment.",
                               evidence={"correct": 11, "n": 12}, extra={"checks": {"numeric_claims": [{
                                   "pattern": r"DEV.*?(\d+)/(\d+)", "evidence_file": "eval_report.json",
                                   "numerator_field": "correct", "denominator_field": "n"}]}})
        self.assertIn("NUMERIC_CLAIM_CONTRADICTION", self.codes(result))

    def test_numeric_nested_matching_values_pass(self):
        result = self.classify("DEV result 11/12.\nNot promotable. No autonomy. No deployment.",
                               evidence={"result": {"correct": 11, "n": 12}}, extra={"checks": {"numeric_claims": [{
                                   "pattern": r"DEV.*?(\d+)/(\d+)", "evidence_file": "eval_report.json",
                                   "numerator_field": "result.correct", "denominator_field": "result.n"}]}})
        self.assertEqual(result["status"], "PASS")

    def test_canonical_fail_must_be_visible(self):
        result = self.classify(evidence={"decision": "FAIL"}, extra={"checks": {"source_disclosures": [{
            "evidence_file": "eval_report.json", "field": "decision", "value": "FAIL",
            "readme_pattern": r"held.out.{0,30}FAIL"}]}})
        self.assertIn("CANONICAL_STATE_NOT_DISCLOSED", self.codes(result))

    def test_missing_external_canonical_evidence_holds(self):
        result = self.classify(extra={"external_evidence": [{"path": "canonical/publication.json",
                               "url": "https://raw.githubusercontent.com/example/repo/" + "a" * 40 + "/publication.json",
                               "revision": "a" * 40}]})
        self.assertIn("EXTERNAL_EVIDENCE_UNVERIFIED", self.codes(result))

    def test_publication_contradiction_uses_recorded_boolean(self):
        result = self.classify("NOT YET PUBLISHED.\nNot promotable. No autonomy. No deployment.",
                               evidence={"published": True}, extra={"checks": {"publication_record": {"path": "eval_report.json"}}})
        self.assertIn("PUBLICATION_RECORD_CONTRADICTION", self.codes(result))

    def test_hash_expectation_mismatch_conflicts(self):
        result = self.classify(extra={"evidence_files": [{"path": "eval_report.json", "sha256": "0" * 64}]})
        self.assertIn("EVIDENCE_HASH_MISMATCH", self.codes(result))

    def test_derived_binding_negation_does_not_become_an_overclaim(self):
        result = self.classify("GGUF files are not covered by signed source bindings.\nNot promotable. No autonomy. No deployment.",
                               extra={"checks": {"derived_artifact_binding": {"unsupported_claim_pattern": r"GGUF.*covered by.*signed"}}})
        self.assertNotIn("DERIVED_ARTIFACT_BINDING_OVERCLAIM", self.codes(result))

    def test_no_class_is_hold(self):
        result = self.classify(extra={"class": "unclassified"})
        self.assertEqual(result["status"], "HOLD")

    def test_hashed_readme_without_captured_listing_is_not_pass(self):
        entry, snap = self.fixture(extra={"evidence_files": []})
        snap["files"] = []
        result = audit.classify_repository(entry, snap)
        self.assertIn("README_NOT_IN_LISTING", self.codes(result))
        self.assertEqual(result["status"], "HOLD")

    def test_retained_source_absent_from_listing_holds(self):
        entry, snap = self.fixture()
        snap["contents"]["extra.json"] = "{}"
        snap["file_sha256"]["extra.json"] = hashlib.sha256(b"{}").hexdigest()
        self.assertIn("SOURCE_NOT_IN_LISTING", self.codes(audit.classify_repository(entry, snap)))

    def test_hub_relative_routes_resolve_to_listing_with_anchor(self):
        entry, snap = self.fixture("[repro](./tree/main/repro#files) [receipt](./blob/main/eval_report.json#data)\nNot promotable. No autonomy. No deployment.")
        snap["files"].append("repro/run.py")
        result = audit.classify_repository(entry, snap)
        self.assertNotIn("BROKEN_LOCAL_LINK", self.codes(result))

    def test_contradictory_kernel_source_holds_even_with_disclosure(self):
        result = self.classify("Source conflict is disclosed: no surrogate is published. Not promotable. No autonomy. No deployment.",
                               evidence={"model": {"trained_weights_present": False},
                                         "claims": {"trained_model": "SURROGATE_PRESENT_MEASURED_FIDELITY"}})
        self.assertIn("SOURCE_EVIDENCE_CONFLICT", self.codes(result))
        self.assertEqual(result["status"], "HOLD")

    def test_arbitrary_matching_hash_does_not_bind_another_artifact(self):
        result = self.classify(evidence={"label": "MEASURED", "sourceSha256": "b" * 64,
                                         "evaluatedArtifact": "other.gguf", "artifactBinding": False},
                               extra={"checks": {"artifact_binding": {"artifact_paths": ["adapter_model.safetensors"]}}})
        self.assertEqual(result["status"], "MISSING_BINDING")

    def test_explicit_binding_false_rejects_even_matching_structured_fields(self):
        result = self.classify(evidence={"artifactBinding": False, "artifact": {"file": "adapter_model.safetensors", "sha256": "b" * 64}},
                               extra={"checks": {"artifact_binding": {"bindings": [{
                                   "artifact_path": "adapter_model.safetensors", "evidence_file": "eval_report.json",
                                   "artifact_field": "artifact.file", "sha256_field": "artifact.sha256"}]}}})
        self.assertEqual(result["status"], "MISSING_BINDING")

    def test_later_historical_sentence_does_not_hide_current_numeric_claim(self):
        result = self.classify("Current DEV 99/12. Historical result 11/12.\nNot promotable. No autonomy. No deployment.",
                               evidence={"correct": 11, "n": 12}, extra={"checks": {"numeric_claims": [{
                                   "pattern": r"DEV.*?(\d+)/(\d+)", "evidence_file": "eval_report.json",
                                   "numerator_field": "correct", "denominator_field": "n"}]}})
        self.assertIn("NUMERIC_CLAIM_CONTRADICTION", self.codes(result))

    def test_offline_branch_identity_must_match(self):
        entry, snap = self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            audit.write_snapshot(entry, snap, Path(temp))
            entry["revision"] = "release-candidate"
            with self.assertRaises(ValueError):
                audit.load_snapshot(entry, Path(temp))

    def test_admission_draft_current_unpublished_status_conflicts(self):
        result = self.classify("**Status: QUALIFIED, NOT YET PUBLISHED** — this card is the admission draft;\nthe repository goes live only through the canonical publisher.\nNot promotable. No autonomy. No deployment.",
                               evidence={"publication": {"hub_write": True}}, extra={"checks": {"publication_record": {
                                   "path": "eval_report.json", "field": "publication.hub_write"}}})
        self.assertIn("PUBLICATION_RECORD_CONTRADICTION", self.codes(result))

    def test_wrapped_stale_admission_banner_quote_does_not_conflict(self):
        result = self.classify('The canonical publication record records an owner-authorized Hub write\nafter lane admission. The earlier\n"not yet published" admission banner was stale.\nNot promotable. No autonomy. No deployment.',
                               evidence={"publication": {"hub_write": True}}, extra={"checks": {"publication_record": {
                                   "path": "eval_report.json", "field": "publication.hub_write"}}})
        self.assertNotIn("PUBLICATION_RECORD_CONTRADICTION", self.codes(result))

    def test_training_step_none_this_run_is_not_current_eval_denial(self):
        result = self.classify("The training receipt retains evals=none-this-run for its training step. That historical field does not erase the later generate record.\nNot promotable. No autonomy. No deployment.")
        self.assertNotIn("MEASURED_RECORD_CONTRADICTS_NO_EVAL", self.codes(result))

    def test_separate_current_eval_denial_is_not_hidden_by_training_scope(self):
        result = self.classify("No evaluation exists. The training receipt retains evals=none-this-run for its training step.\nNot promotable. No autonomy. No deployment.")
        self.assertIn("MEASURED_RECORD_CONTRADICTS_NO_EVAL", self.codes(result))

    def test_training_receipt_measured_label_does_not_create_evaluation(self):
        entry, snap = self.fixture("evals: none-this-run\nNot promotable. No autonomy. No deployment.",
                                   evidence={"label": "MEASURED", "train_loss": 0.0537})
        entry["evidence_files"] = ["training_receipt.json"]
        snap["files"].remove("eval_report.json")
        snap["files"].append("training_receipt.json")
        snap["contents"]["training_receipt.json"] = snap["contents"].pop("eval_report.json")
        snap["file_sha256"]["training_receipt.json"] = snap["file_sha256"].pop("eval_report.json")
        result = audit.classify_repository(entry, snap)
        self.assertEqual(result["measured_records"], [])
        self.assertNotIn("MEASURED_RECORD_CONTRADICTS_NO_EVAL", self.codes(result))

    def test_archive_summary_schema_is_readable_evaluation_evidence(self):
        entry, snap = self.fixture("Evaluation successful calls 3/4.\nNot promotable. No autonomy. No deployment.",
                                   evidence={"candidate_metrics": {"successful_call_rate": .75}, "baseline_metrics": {}, "comparison": {}, "decision": "HOLD"})
        entry["evidence_files"] = ["runs/example/summary.json"]
        snap["files"].remove("eval_report.json")
        snap["files"].append("runs/example/summary.json")
        snap["contents"]["runs/example/summary.json"] = snap["contents"].pop("eval_report.json")
        snap["file_sha256"]["runs/example/summary.json"] = snap["file_sha256"].pop("eval_report.json")
        result = audit.classify_repository(entry, snap)
        self.assertNotIn("EVALUATION_RECEIPT_MISSING", self.codes(result))

    def test_offline_cli_writes_both_reports_and_returns_failure_when_requested(self):
        entry, snap = self.fixture("evals: none-this-run\nNot promotable. No autonomy. No deployment.")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audit.write_snapshot(entry, snap, root / "snapshots")
            inventory = root / "inventory.json"
            inventory.write_text(json.dumps([entry]), encoding="utf-8")
            code = audit.main(["--inventory", str(inventory), "--output", str(root / "report.md"),
                               "--snapshot-dir", str(root / "snapshots"), "--offline", "--fail-on-findings"])
            self.assertEqual(code, 1)
            payload = json.loads((root / "report.json").read_text())
            self.assertTrue(payload["read_only"])
            self.assertEqual(payload["mode"], "offline_snapshot")
            self.assertIn("never a promotion", (root / "report.md").read_text())

    def test_path_rejection_covers_windows_and_encoded_paths(self):
        for path in ["../token", "C:/token", "C:\\token", "/etc/token", "a//b", "a/./b", "\\\\server\\share"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                audit.safe_relative(path)

    def test_inventory_rejects_mutable_external_evidence(self):
        entry, _ = self.fixture(extra={"external_evidence": [{"path": "canonical/evidence.json",
                                  "url": "https://raw.githubusercontent.com/example/repo/main/evidence.json", "revision": "main"}]})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "inventory.json"
            path.write_text(json.dumps([entry]))
            with self.assertRaises(ValueError):
                audit.read_inventory(path)


if __name__ == "__main__":
    unittest.main()
