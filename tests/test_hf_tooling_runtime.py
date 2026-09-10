"""Offline SZL adapter/controller regressions, not upstream runtime evidence.

Copyright 2026 SZL Holdings. SPDX-License-Identifier: Apache-2.0
The dedicated workflow installs upstream packages and runs the actual probes.
"""
from __future__ import annotations

import copy
import io
import json
import unittest
from unittest.mock import MagicMock, patch

from inference import hf_tooling as adapters
from inference.hf_frontier import sha256
from tools import evaluate_hf_tooling as runner

SOURCE = "a" * 40


def install_document(package):
    expected = adapters.RELEASES[package]
    return {"url": "https://github.com/" + expected["repository"] + ".git",
            "vcs_info": {"vcs": "git", "commit_id": expected["revision"]}}


def observed_install(package):
    return {"package": package, **adapters.RELEASES[package]}


def fixture_report():
    with patch.object(runner, "require_install", side_effect=observed_install), \
            patch.dict(runner.PROBES, {"hub": (lambda: {"fixture": True},) * 3}):
        return runner.evaluate("hub", SOURCE)


def rehash(report):
    report["reportSha256"] = sha256({k: v for k, v in report.items() if k != "reportSha256"})


class ToolingAdapterTests(unittest.TestCase):
    def test_all_primary_releases_are_exact_git_pins(self):
        for package, release in adapters.RELEASES.items():
            with self.subTest(package=package):
                self.assertEqual(adapters.validate_install(package, release["version"],
                                 install_document(package)), observed_install(package))

    def test_wrong_installed_version_is_rejected(self):
        with self.assertRaises(adapters.ToolingError):
            adapters.validate_install("trl", "0.24.0", install_document("trl"))

    def test_wrong_repository_is_rejected(self):
        value = install_document("tau-ai")
        value["url"] = "https://github.com/untrusted/tau.git"
        with self.assertRaises(adapters.ToolingError):
            adapters.validate_install("tau-ai", "0.4.2", value)

    def test_wrong_revision_and_mutable_refs_are_rejected(self):
        for value in ("main", "v1.31.0", "b" * 40, None, 5):
            document = install_document("huggingface-hub")
            document["vcs_info"]["commit_id"] = value
            with self.subTest(value=value), self.assertRaises(adapters.ToolingError):
                adapters.validate_install("huggingface-hub", "1.31.0", document)

    def test_pypi_only_version_cannot_claim_git_source(self):
        for doc in ({}, {"vcs_info": []}, {"archive_info": {"hash": "sha256:test"}}):
            with self.assertRaises(adapters.ToolingError):
                adapters.validate_install("tau-ai", "0.4.2", doc)

    def test_distribution_provenance_is_used(self):
        distribution = MagicMock(version="0.4.2")
        distribution.read_text.return_value = json.dumps(install_document("tau-ai"))
        with patch.object(adapters.metadata, "distribution", return_value=distribution):
            self.assertEqual(adapters.require_install("tau-ai"), observed_install("tau-ai"))

    def test_missing_dependency_does_not_become_pass(self):
        with patch.object(adapters.metadata, "distribution", side_effect=adapters.metadata.PackageNotFoundError):
            with self.assertRaises(adapters.ToolingError):
                adapters.require_install("tau-ai")

    def test_requirements_use_tau_ai_and_never_moving_main(self):
        for lane in adapters.LANES:
            text = adapters.requirements(lane)
            self.assertNotIn("@main", text)
            self.assertNotIn("--extra-index", text)
            for package in adapters.LANES[lane]:
                self.assertIn("@" + adapters.RELEASES[package]["revision"], text)
        self.assertTrue(adapters.requirements("tau").startswith("tau-ai @"))

    def test_unknown_lane_is_rejected(self):
        with self.assertRaises(adapters.ToolingError):
            adapters.requirements("arbitrary-pip-command")

    def test_labels_bind_run_source_and_vertical(self):
        labels = adapters.job_labels("run-42", SOURCE, "lyte")
        self.assertEqual(labels, {"szl-run": "run-42", "szl-source": SOURCE,
                                 "szl-vertical": "lyte", "szl-purpose": "evaluation"})

    def test_labels_refuse_free_text_and_unsafe_inputs(self):
        for value in ("", "private secret words", "../path", "a\nheader", "x" * 64, 42):
            with self.subTest(value=value), self.assertRaises(adapters.ToolingError):
                adapters.job_labels(value, SOURCE)

    def test_config_bounds_fail_before_optional_imports(self):
        for value in (True, 0, -1, 1_048_577, 1.5, "4096"):
            with patch.object(adapters, "require_install", side_effect=AssertionError("import")):
                with self.subTest(value=value), self.assertRaises(adapters.ToolingError):
                    adapters.sft_config("temporary", max_length=value)

    def test_invalid_device_type_fails_before_optional_imports(self):
        with self.assertRaises(adapters.ToolingError):
            adapters.sft_config("temporary", cpu="true")

    def test_config_calls_real_api_contract_only_after_pin_gate(self):
        factory = MagicMock()
        module = MagicMock(SFTConfig=factory)
        with patch.object(adapters, "require_install", side_effect=observed_install), \
                patch.dict("sys.modules", {"trl": module}):
            adapters.sft_config("test-output", max_length=4096)
        self.assertEqual(factory.call_args.kwargs["loss_type"], "chunked_nll")
        self.assertIs(factory.call_args.kwargs["push_to_hub"], False)
        self.assertIs(factory.call_args.kwargs["trust_remote_code"], False)
        self.assertIs(factory.call_args.kwargs["packing"], False)

    def test_session_references_require_hashes(self):
        for digest in ("main", "a" * 63, True):
            with self.assertRaises(adapters.ToolingError):
                adapters.tau_evidence_entries(run_id="test", source_revision=SOURCE, receipt_sha256=digest)

    def test_session_timestamp_finite_and_nonnegative(self):
        for timestamp in (True, float("nan"), float("inf"), -1, "now"):
            with self.assertRaises(adapters.ToolingError):
                adapters.tau_evidence_entries(run_id="test", source_revision=SOURCE,
                                             receipt_sha256="b" * 64, timestamp=timestamp)

    def test_catalog_intersection_does_not_elevate_new_models(self):
        self.assertEqual(adapters.authorized_catalog(["a", "b", "a"], ["a", "c"]), ["a"])
        self.assertEqual(adapters.authorized_catalog(["new/model"], []), [])

    def test_catalog_rejects_bad_shapes(self):
        for value in (None, "model", [0], [""], ["a"] * 257):
            with self.assertRaises(adapters.ToolingError):
                adapters.authorized_catalog(value, [])


class ToolingControllerTests(unittest.TestCase):
    def test_fixture_receipt_passes_integrity_and_policy(self):
        report = fixture_report()
        runner.verify_report(report, SOURCE)
        self.assertEqual(report["runtimeStatus"], "SMOKE_PASS")
        self.assertEqual(report["productionDisposition"], "HOLD")

    def test_source_revision_must_match_expected_outer_binding(self):
        with self.assertRaises(adapters.ToolingError):
            runner.verify_report(fixture_report(), "b" * 40)

    def test_tampered_receipt_rejected(self):
        report = fixture_report()
        report["checks"]["sandbox_label_validation"]["status"] = "FAIL"
        with self.assertRaises(adapters.ToolingError):
            runner.verify_report(report, SOURCE)

    def test_rehashed_authority_escalation_still_rejected(self):
        for key in adapters.AUTHORITY:
            report = fixture_report()
            report["authority"][key] = True
            rehash(report)
            with self.subTest(key=key), self.assertRaises(adapters.ToolingError):
                runner.verify_report(report, SOURCE)

    def test_rehashed_extra_fields_rejected(self):
        report = fixture_report()
        report["productionApproved"] = True
        rehash(report)
        with self.assertRaises(adapters.ToolingError):
            runner.verify_report(report, SOURCE)

    def test_rehashed_missing_check_rejected(self):
        report = fixture_report()
        report["checks"].pop("filesystem_path_boundary")
        rehash(report)
        with self.assertRaises(adapters.ToolingError):
            runner.verify_report(report, SOURCE)

    def test_rehashed_false_pass_rejected(self):
        report = fixture_report()
        report["checks"]["filesystem_path_boundary"]["status"] = "FAIL"
        rehash(report)
        with self.assertRaises(adapters.ToolingError):
            runner.verify_report(report, SOURCE)

    def test_unmeasured_capabilities_cannot_be_rehashed_into_pass(self):
        report = fixture_report()
        report["remainingEvaluation"]["billable_sandbox_job_roundtrip"] = "PASS"
        rehash(report)
        with self.assertRaises(adapters.ToolingError):
            runner.verify_report(report, SOURCE)

    def test_installation_failure_emits_unavailable_and_no_probe(self):
        probe = MagicMock()
        with patch.object(runner, "require_install", side_effect=adapters.ToolingError("missing")), \
                patch.dict(runner.PROBES, {"hub": (probe,) * 3}):
            report = runner.evaluate("hub", SOURCE)
        probe.assert_not_called()
        self.assertEqual(report["runtimeStatus"], "UNAVAILABLE")
        runner.verify_report(report, SOURCE)

    def test_runtime_failure_is_recorded_without_false_green(self):
        broken = MagicMock(side_effect=ValueError("fixture failure"))
        with patch.object(runner, "require_install", side_effect=observed_install), \
                patch.dict(runner.PROBES, {"hub": (broken, lambda: {}, lambda: {})}):
            report = runner.evaluate("hub", SOURCE)
        self.assertEqual(report["runtimeStatus"], "SMOKE_FAIL")
        runner.verify_report(report, SOURCE)

    def test_live_probe_requires_correct_lane(self):
        with self.assertRaises(adapters.ToolingError):
            runner.evaluate("tau", SOURCE, live_hub=True)

    def test_live_probe_failure_survives_other_passes(self):
        with patch.object(runner, "require_install", side_effect=observed_install), \
                patch.dict(runner.PROBES, {"hub": (lambda: {},) * 3}), \
                patch.object(runner, "hub_public_snapshot", side_effect=OSError("network down")):
            report = runner.evaluate("hub", SOURCE, live_hub=True)
        self.assertEqual(report["runtimeStatus"], "SMOKE_FAIL")
        self.assertIs(report["liveHubRequested"], True)
        runner.verify_report(report, SOURCE)

    def test_live_check_cannot_be_silently_dropped(self):
        report = fixture_report()
        report["liveHubRequested"] = True
        rehash(report)
        with self.assertRaises(adapters.ToolingError):
            runner.verify_report(report, SOURCE)

    def test_network_guard_blocks_and_restores_socket_factory(self):
        previous = runner.socket.create_connection
        with runner.offline_network():
            with self.assertRaises(adapters.ToolingError):
                runner.socket.create_connection(("example.com", 443))
        self.assertIs(runner.socket.create_connection, previous)

    def test_plan_and_requirements_are_offline(self):
        with patch.object(runner, "require_install", side_effect=AssertionError("import")), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(runner.main(["plan"]), 0)
            self.assertEqual(json.loads(output.getvalue())["authority"], adapters.AUTHORITY)
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(runner.main(["requirements", "--lane", "tau"]), 0)
            self.assertIn("tau-ai @ git+", output.getvalue())

    def test_source_file_digests_are_bound_into_receipt(self):
        report = fixture_report()
        self.assertEqual(set(report["sourceFilesSha256"]),
                         {"inference/hf_tooling.py", "tools/evaluate_hf_tooling.py"})
        self.assertTrue(all(len(digest) == 64 for digest in report["sourceFilesSha256"].values()))

    def test_verifier_does_not_mutate_input(self):
        report = fixture_report()
        before = copy.deepcopy(report)
        runner.verify_report(report, SOURCE)
        self.assertEqual(report, before)


if __name__ == "__main__":
    unittest.main()
