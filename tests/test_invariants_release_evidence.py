"""Synthetic receipt consistency tests; no fixture constitutes publication proof."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import stat
import tempfile
import unittest
import warnings
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "invariants_release_evidence_under_test",
    Path(__file__).resolve().parents[1] / "tools" / "inspect_invariants_release_evidence.py",
)
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)
SOURCE, PUBLISHER = "a" * 40, "b" * 40


def fixture():
    """Mirror the current v1 report contract, explicitly with invented fixture IDs."""
    auth = {
        "schema": "szl.invariants-release-authorization/v1",
        "status": "AUTHORIZED_PROTECTED_MAIN", "authorized_at": "2026-09-13T00:14:22+00:00",
        "publisher": {"repository": evidence.PUBLISHER_REPO, "revision": PUBLISHER,
                      "protected_main": PUBLISHER, "branch_protection_observed": True},
        "source": {
            "repository": evidence.SOURCE_REPO, "revision": SOURCE, "protected_main": SOURCE,
            "branch_protection_observed": True, "signature_verified": True, "signature_reason": "valid",
            "checks": [{"name": "verify canonical kernel", "app_id": 15368,
                        "required_app_id": 15368, "run_id": 123, "status": "completed",
                        "conclusion": "success"}],
            "required_check_enforcement": {"basis": ["classic_branch_protection"],
                "required_checks_observed": [{"context": "verify canonical kernel", "app_id": 15368}],
                "required_contexts_observed": ["verify canonical kernel"]},
        },
    }
    rows = [{"path": p, "bytes": 12, "sha256": "c" * 64} for p in sorted(evidence.SOURCE_FILES)]
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["path"].encode() + b"\0" + row["sha256"].encode() + b"\n")
    prior, after = {"main": "1" * 40, "v1": "2" * 40}, {"main": "1" * 40, "v1": "3" * 40}
    report = {
        "schema": "szl.invariants-publication-report/v1", "mode": "PUBLISH",
        "status": "PUBLISHED_AND_EXACT_READBACK_VERIFIED", "repo_id": evidence.HUB_REPO,
        "source_repository": evidence.SOURCE_REPO, "source_revision": SOURCE,
        "authorization": copy.deepcopy(auth), "artifact_tree_sha256": digest.hexdigest(),
        "declared_files": rows,
        "publisher": {"repository": evidence.PUBLISHER_REPO, "revision": PUBLISHER,
            "workflow_path": evidence.WORKFLOW,
            "workflow_ref": f"{evidence.PUBLISHER_REPO}/{evidence.WORKFLOW}@refs/heads/main",
            "workflow_url": f"https://github.com/{evidence.PUBLISHER_REPO}/blob/{PUBLISHER}/{evidence.WORKFLOW}",
            "run_id": "42", "run_attempt": "1",
            "run_url": f"https://github.com/{evidence.PUBLISHER_REPO}/actions/runs/42"},
        "observed_before": {"model": {"revision": "4" * 40},
                            "kernel": {"revision": "2" * 40, "branch": "v1", "branches": prior.copy()}},
        "targets": {
            "model": {"status": "EXACT_READBACK_VERIFIED", "revision_before": "4" * 40,
                      "revision_after": "5" * 40, "publication_interface": "huggingface_hub.create_commit"},
            "kernel": {"status": "V1_EXACT_READBACK_VERIFIED", "revision_before": "2" * 40,
                "revision_after": "3" * 40, "branch": "v1", "branches_before": prior.copy(),
                "branches_after": after.copy(), "verified_branches": after.copy(),
                "parents_revalidated_before_upload": prior.copy(),
                "staged_files": sorted(evidence.STAGED_FILES), "publication_interface": "kernel-builder",
                "publication_interface_version": "0.17.0-dev0",
                "publication_interface_source_revision": "633246310320d85def0c67d62c7912fd444a842f"},
        },
    }
    return auth, report


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "evidence.zip"
        self.auth, self.report = fixture()

    def archive(self, entries=None, compression=zipfile.ZIP_DEFLATED):
        if entries is None:
            entries = [(evidence.AUTH_FILE, json.dumps(self.auth)),
                       (evidence.PUBLICATION_FILE, json.dumps(self.report))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.path, "w", compression=compression) as z:
                for name, data in entries:
                    z.writestr(name, data)
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def check(self, digest=None, **kwargs):
        values = dict(expected_digest=digest or self.archive(), source=SOURCE, publisher=PUBLISHER,
                      run_id="42", run_attempt="1")
        values.update(kwargs)
        with mock.patch("socket.socket.connect", side_effect=AssertionError("NETWORK_FORBIDDEN")):
            verdict = evidence.inspect_archive(self.path, **values)
        self.assertFalse(verdict["publication_verified"])
        self.assertFalse(verdict["independent_hub_readback"])
        self.assertFalse(verdict["signature_independently_verified"])
        self.assertEqual(verdict["repository_mutation"], "NOT_ATTEMPTED")
        return verdict

    def rejects(self, reason=None, **kwargs):
        verdict = self.check(**kwargs)
        self.assertFalse(verdict["evidence_consistent"])
        self.assertEqual(verdict["status"], "REJECTED")
        if reason:
            self.assertEqual(verdict["reason"], reason)
        return verdict

    def test_consistent_fixture_is_review_required_not_publication_proof(self):
        verdict = self.check()
        self.assertTrue(verdict["evidence_consistent"])
        self.assertEqual(verdict["status"], "EVIDENCE_CONSISTENT_REVIEW_REQUIRED")

    def test_authorization_only_archive_is_not_a_release(self):
        digest = self.archive([(evidence.AUTH_FILE, json.dumps(self.auth))])
        self.rejects("MISSING_PUBLICATION_REPORT", digest=digest)

    def test_missing_authorization_is_rejected(self):
        digest = self.archive([(evidence.PUBLICATION_FILE, json.dumps(self.report))])
        self.rejects("MISSING_AUTHORIZATION_REPORT", digest=digest)

    def test_digest_mismatch_and_malformed_expected_digest(self):
        self.archive()
        self.rejects("ARCHIVE_DIGEST_MISMATCH", digest="0" * 64)
        self.rejects("INVALID_EXPECTED_ARCHIVE_DIGEST", digest="latest")

    def test_wrong_expected_source_publisher_run_and_attempt(self):
        for key, value in (("source", "d" * 40), ("publisher", "d" * 40),
                           ("run_id", "43"), ("run_attempt", "2"), ("run_id", "01")):
            with self.subTest(key=key):
                self.rejects(**{key: value})

    def test_no_dry_run_or_in_progress_receipt_is_accepted(self):
        for field, value in (("mode", "DRY_RUN"), ("status", "VERIFIED_DRY_RUN"),
                             ("status", "PUBLICATION_IN_PROGRESS"), ("status", True)):
            with self.subTest(value=value):
                self.auth, self.report = fixture()
                self.report[field] = value
                self.rejects("PUBLICATION_NOT_SUCCESSFUL")

    def test_partial_target_readback_fails_closed(self):
        for target in ("model", "kernel"):
            with self.subTest(target=target):
                self.auth, self.report = fixture()
                self.report["targets"][target]["status"] = "READBACK_PENDING"
                self.rejects("TARGET_READBACK_INCOMPLETE")

    def test_moving_or_malformed_target_revision_is_rejected(self):
        for target in ("model", "kernel"):
            for revision in ("main", "v1", "f" * 39, "f" * 40 + "\n", None):
                with self.subTest(target=target, revision=revision):
                    self.auth, self.report = fixture()
                    self.report["targets"][target]["revision_after"] = revision
                    self.rejects("TARGET_REVISION_NOT_IMMUTABLE")

    def test_kernel_default_head_drift_is_rejected(self):
        self.report["targets"]["kernel"]["branches_after"]["main"] = "6" * 40
        self.rejects("KERNEL_READBACK_MISMATCH")

    def test_kernel_parent_drift_is_rejected(self):
        self.report["targets"]["kernel"]["parents_revalidated_before_upload"]["v1"] = "6" * 40
        self.rejects("KERNEL_PARENT_MISMATCH")

    def test_incomplete_kernel_staging_is_rejected(self):
        self.report["targets"]["kernel"]["staged_files"].pop()
        self.rejects("KERNEL_STAGED_FILE_SET_MISMATCH")

    def test_missing_or_extra_targets_are_rejected(self):
        del self.report["targets"]["model"]
        self.rejects("TARGET_SET_MISMATCH")
        self.auth, self.report = fixture()
        self.report["targets"]["space"] = {}
        self.rejects("TARGET_SET_MISMATCH")

    def test_authorization_mismatch_including_boolean_integer_alias(self):
        self.report["authorization"]["source"]["signature_verified"] = 1
        self.rejects("AUTHORIZATION_REPORT_MISMATCH")

    def test_boolean_protection_and_signature_claims_are_strict(self):
        for field in ("signature_verified", "branch_protection_observed"):
            with self.subTest(field=field):
                self.auth, self.report = fixture()
                self.auth["source"][field] = 1
                self.report["authorization"] = copy.deepcopy(self.auth)
                self.rejects()

    def test_failed_check_and_forged_app_binding_are_rejected(self):
        for key, value, reason in (("conclusion", "failure", "CHECK_NOT_SUCCESSFUL"),
                                   ("app_id", 7, "CHECK_APP_MISMATCH"),
                                   ("run_id", True, "CHECK_NOT_SUCCESSFUL")):
            with self.subTest(key=key):
                self.auth, self.report = fixture()
                self.auth["source"]["checks"][0][key] = value
                self.report["authorization"] = copy.deepcopy(self.auth)
                self.rejects(reason)

    def test_missing_and_duplicate_check_bindings_are_rejected(self):
        policy = self.auth["source"]["required_check_enforcement"]
        policy["required_checks_observed"].append(copy.deepcopy(policy["required_checks_observed"][0]))
        self.rejects("DUPLICATE_CHECK_BINDING")
        self.auth, self.report = fixture()
        self.auth["source"]["checks"] = []
        self.rejects("REQUIRED_CHECK_EVIDENCE_MISSING")

    def test_effective_rules_and_explicit_any_app_binding_are_supported(self):
        self.auth["source"]["required_check_enforcement"]["basis"] = ["effective_branch_rules"]
        self.auth["source"]["required_check_enforcement"]["required_checks_observed"][0]["app_id"] = None
        self.auth["source"]["checks"][0]["required_app_id"] = None
        self.report["authorization"] = copy.deepcopy(self.auth)
        self.assertTrue(self.check()["evidence_consistent"])

    def test_absent_app_binding_is_not_an_explicit_wildcard(self):
        del self.auth["source"]["required_check_enforcement"]["required_checks_observed"][0]["app_id"]
        self.rejects("INVALID_CHECK_POLICY")

    def test_hash_tree_and_source_variants_are_checked(self):
        self.report["artifact_tree_sha256"] = "0" * 64
        self.rejects("ARTIFACT_TREE_MISMATCH")
        self.auth, self.report = fixture()
        self.report["declared_files"][0]["sha256"] = "0" * 64
        self.rejects("SOURCE_VARIANT_MISMATCH")

    def test_duplicate_paths_and_boolean_byte_counts_are_rejected(self):
        self.report["declared_files"][1] = copy.deepcopy(self.report["declared_files"][0])
        self.rejects("DECLARED_FILE_SET_MISMATCH")
        self.auth, self.report = fixture()
        self.report["declared_files"][0]["bytes"] = True
        self.rejects("INVALID_DECLARED_FILE")

    def test_duplicate_json_keys_and_nonfinite_numbers_are_rejected(self):
        for data, reason in (('{"status":1,"status":2}', "DUPLICATE_JSON_KEY"),
                             ('{"status":NaN}', "NONFINITE_JSON_VALUE"),
                             ('{"status":1e999}', "NONFINITE_JSON_VALUE"),
                             (b'\xff', "INVALID_JSON"), ('[]', "JSON_NOT_OBJECT")):
            with self.subTest(reason=reason):
                digest = self.archive([(evidence.AUTH_FILE, json.dumps(self.auth)),
                                       (evidence.PUBLICATION_FILE, data)])
                self.rejects(reason, digest=digest)

    def test_nested_or_traversal_archive_names_are_rejected_without_extraction(self):
        for name in ("../" + evidence.PUBLICATION_FILE, "/" + evidence.PUBLICATION_FILE,
                     "reports/" + evidence.PUBLICATION_FILE, "C:\\" + evidence.PUBLICATION_FILE):
            with self.subTest(name=name):
                digest = self.archive([(evidence.AUTH_FILE, json.dumps(self.auth)), (name, "{}")])
                self.rejects("UNEXPECTED_ARCHIVE_MEMBERS", digest=digest)
        self.assertEqual([p.name for p in Path(self.temp.name).iterdir()], ["evidence.zip"])

    def test_duplicate_zip_names_are_rejected(self):
        digest = self.archive([(evidence.AUTH_FILE, "{}"), (evidence.AUTH_FILE, "{}")])
        self.rejects("DUPLICATE_ARCHIVE_MEMBER", digest=digest)

    def test_symlink_member_is_rejected(self):
        info = zipfile.ZipInfo(evidence.PUBLICATION_FILE)
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        digest = self.archive([(evidence.AUTH_FILE, json.dumps(self.auth)), (info, "target")])
        self.rejects("NONREGULAR_ARCHIVE_MEMBER", digest=digest)

    def test_member_and_archive_size_limits(self):
        digest = self.archive([(evidence.AUTH_FILE, json.dumps(self.auth)),
                               (evidence.PUBLICATION_FILE, "x" * (evidence.MEMBER_LIMIT + 1))])
        self.rejects("MEMBER_SIZE_INVALID", digest=digest)
        self.path.write_bytes(b"x" * (evidence.ARCHIVE_LIMIT + 1))
        self.rejects("ARCHIVE_TOO_LARGE", digest="0" * 64)

    def test_corrupt_archive_is_rejected(self):
        self.path.write_bytes(b"not a zip")
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.rejects("INVALID_ARCHIVE", digest=digest)

    def test_cli_exit_status_and_sanitized_diagnostics(self):
        sentinel = "PRIVATE_CREDENTIAL_SENTINEL"
        self.report["publisher"]["run_id"] = sentinel
        digest = self.archive()
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = evidence.main(["--archive", str(self.path), "--archive-sha256", digest,
                                  "--source-revision", SOURCE, "--publisher-revision", PUBLISHER,
                                  "--run-id", "42", "--run-attempt", "1"])
        self.assertEqual(code, 1)
        self.assertNotIn(sentinel, stream.getvalue())
        self.assertFalse(json.loads(stream.getvalue())["publication_verified"])


if __name__ == "__main__":
    unittest.main()
