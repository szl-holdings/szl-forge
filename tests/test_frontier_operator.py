"""Offline adversarial regressions. All remote transports are mocked."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "szl_frontier_operator.py"
SPEC = importlib.util.spec_from_file_location("operator_under_test", SCRIPT)
op = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(op)
SHA = "a" * 40
BASE = "b" * 40
RUN = f"github-123-1-{SHA[:12]}"


def qualification_fixture():
    return {
        "pr": {"state": "open", "draft": False, "merged_at": None, "mergeable": True,
               "head": {"sha": SHA}, "base": {"sha": BASE}},
        "checks": [{"id": 1, "name": "gate", "head_sha": SHA, "status": "completed", "conclusion": "success", "app": {"id": 15368}}],
        "runs": [{"id": 1, "name": "CI", "head_sha": SHA, "status": "completed", "conclusion": "success"}],
        "statuses": [],
        "policy": {"required_checks": [{"context": "gate", "app_id": 15368}], "signatures_required": True,
                   "protection_present": True, "unsupported_rules": [], "reviews": {"required_review_thread_resolution": True}},
        "reviews": {"headRefOid": SHA, "baseRefOid": BASE, "mergeStateStatus": "CLEAN", "reviewDecision": None,
                    "unresolved_threads": 0},
        "commits": [{"sha": SHA, "commit": {"verification": {"verified": True, "reason": "valid"}}}],
        "exceptions": [],
    }


def publication_fixture():
    hashes = {k: hashlib.sha256(k.encode()).hexdigest() for k in
              ("runner.py", "core.py", "source.py", "provider.py", "receipt.py", "fixtures.v1.json")}
    attestation = {"model_id": "zai-org/GLM-5.3-Flash", "requested_revision": "c" * 40,
                   "metadata_files": {"config.json": "d" * 64},
                   "license": {"file": "LICENSE", "sha256": "e" * 64}}
    contract = {"hashes": hashes, "case_ids": ["one", "two", "three", "four"], "attestation": attestation}
    records = {}
    for route in ("baseline", "candidate"):
        records[route] = [{"case_id": case, "route": route, "ok": True, "http_status": 200,
                           "latency_ms": 12.5, "response_content": "public synthetic response",
                           "response_sha256": hashlib.sha256(b"public synthetic response").hexdigest(),
                           "score": {**{k: True for k in ("json_valid", "schema_valid", "decision_match", "answer_match",
                                                        "evidence_match", "tool_boundary_pass")}, "points": 6, "max_points": 6}}
                          for case in contract["case_ids"]]
    fallback = {"pass": True, "invalid_candidate_request_failed": True, "baseline_fallback_succeeded": True}
    receipt = {"schema": "szl.nemo.frontier-qualification-receipt.v1", "candidate_id": "glm-5-3-flash",
               "model_id": attestation["model_id"], "model_revision": attestation["requested_revision"], "provider": "provider",
               "run_identity": {"source_repository": op.FORGE, "source_revision": SHA, "run_id": RUN, "suite": "full",
                   "runner_sha256": hashes["runner.py"], "core_sha256": hashes["core.py"], "source_module_sha256": hashes["source.py"],
                   "provider_module_sha256": hashes["provider.py"], "receipt_module_sha256": hashes["receipt.py"]},
               "fixture_set_sha256": hashes["fixtures.v1.json"], "production_disposition": "HOLD", "promotion_effect": "NONE",
               "decision": "EVIDENCE_COMPLETE_REVIEW_REQUIRED", "violated_invariants": [],
               "signature_status": "UNSIGNED_HONEST", "authenticity_not_established": True,
               "fallback_evidence": fallback, "fallback_evidence_sha256": op.digest(fallback), "known_bounds": ["unsigned"],
               "comparison": {}, "provider_execution_revision": "UNAVAILABLE"}
    for route in records:
        receipt[f"{route}_results_sha256"] = op.digest(records[route])
        receipt[f"{route}_metrics"] = op.recompute_metrics(records[route])
    receipt["receipt_sha256"] = op.digest(receipt)
    files = {**attestation["metadata_files"], "LICENSE": attestation["license"]["sha256"]}
    bundle = {"schema": "szl.frontier.evaluation-bundle.v1", "receipt": copy.deepcopy(receipt),
              "baseline_results": records["baseline"], "candidate_results": records["candidate"],
              "fixture_manifest": {"case_ids": contract["case_ids"].copy(), "sha256": hashes["fixtures.v1.json"]},
              "source_checks": {"pass": True, "model_id": attestation["model_id"],
                  "requested_revision": attestation["requested_revision"], "resolved_revision": attestation["requested_revision"],
                  "candidate_weights_downloaded": False, "repository_code_executed": False, "python_files": [],
                  "files": {k: {"sha256": v, "expected_sha256": v, "pass": True} for k, v in files.items()}}}
    summary = {k: copy.deepcopy(receipt[k]) for k in ("candidate_id", "model_id", "provider", "baseline_metrics", "candidate_metrics",
               "comparison", "decision", "production_disposition", "receipt_sha256")}
    summary["suite"] = "full"
    # Complete source-bound synthetic fixture, with scores derived from actual responses.
    contract["system_prompt"] = "Synthetic test instruction"
    contract["provider"] = {"PROVIDERS": ["provider"], "ROUTER_URL": "https://router.huggingface.co/v1/chat/completions",
                            "BASELINE_URL": "https://example.hf.space"}
    contract["fixtures"] = {"schema": "szl.frontier.eval-fixtures.v1", "suite": "synthetic-public-governed-inference",
                            "cases": [{"id": case, "category": "grounding", "question": "Test?",
                            "evidence": [{"id": "E1", "text": "Public synthetic value"}],
                            "expected": {"decision": "ANSWER", "answer_equals": "value", "evidence_ids": ["E1"]}}
                            for case in contract["case_ids"]]}
    receipt.update(provider_route=attestation["model_id"] + ":provider", provider_hardware_fingerprint="UNAVAILABLE",
                   runtime_version="UNAVAILABLE", seed="UNAVAILABLE_PROVIDER_DEPENDENT",
                   runtime_engine="Hugging Face Inference Providers OpenAI-compatible router",
                   metric_truth_labels={"provider_execution_revision": "UNAVAILABLE", "provider_hardware": "UNAVAILABLE",
                                        "upstream_benchmark_claims": "NOT_USED"}, known_bounds=sorted(op.REQUIRED_BOUNDS),
                   baseline_identity={"status": "READY", "runtime": {"openai_compatible_subset": {"model_id": "baseline"}}})
    receipt["configuration_sha256"] = op.digest({"system_prompt": contract["system_prompt"], "providers": ["provider"],
        "suite": "full", "router_url": contract["provider"]["ROUTER_URL"], "baseline_url": contract["provider"]["BASELINE_URL"]})
    for route in records:
        for record, case in zip(records[route], contract["fixtures"]["cases"]):
            content = json.dumps({"decision": "ANSWER", "answer": "value", "evidence_ids": ["E1"], "tool_calls": []})
            model = "baseline" if route == "baseline" else receipt["provider_route"]
            payload = {"model": model, "messages": [{"role": "system", "content": contract["system_prompt"]},
                       {"role": "user", "content": "EVIDENCE\nE1: Public synthetic value\n\nQUESTION\nTest?"}],
                       "max_tokens": 32 if route == "baseline" else 96, "temperature": 0, "stream": False}
            record.update(response_content=content, response_sha256=hashlib.sha256(content.encode()).hexdigest(),
                          score=op.score_response(case, content), category="grounding", requested_model=model,
                          request_sha256=op.digest(payload))
        receipt[f"{route}_results_sha256"] = op.digest(records[route])
        receipt[f"{route}_metrics"] = op.recompute_metrics(records[route])
    receipt["provider_attempts"] = [{"provider": "provider", "model": receipt["provider_route"], "ok": True,
                                    "http_status": 200, "latency_ms": 12.5}]
    receipt["comparison"] = op.comparison(receipt["candidate_metrics"], receipt["baseline_metrics"])
    fallback.update(invalid_candidate_error_type="HTTPError", invalid_candidate_http_status=400,
                    invalid_candidate_error_sha256="f" * 64, baseline_fallback_latency_ms=12.5,
                    baseline_fallback_score=op.score_response(contract["fixtures"]["cases"][0], "invalid"))
    receipt["fallback_evidence_sha256"] = op.digest(fallback)
    bundle["fixture_manifest"].update(schema=contract["fixtures"]["schema"], suite=contract["fixtures"]["suite"])
    summary.update({k: copy.deepcopy(receipt[k]) for k in summary if k != "suite"})
    rehash(receipt, bundle, summary)
    return receipt, bundle, summary, contract


def rehash(receipt, bundle, summary):
    receipt["receipt_sha256"] = op.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    bundle["receipt"] = copy.deepcopy(receipt)
    summary["receipt_sha256"] = receipt["receipt_sha256"]


class MergeGuards(unittest.TestCase):
    def test_valid_protected_head(self):
        self.assertEqual(op.qualify(**qualification_fixture()), [])

    def test_zero_checks_cannot_pass(self):
        data = qualification_fixture()
        data["checks"] = []
        data["runs"] = []
        self.assertTrue(op.qualify(**data))

    def test_required_check_wrong_app_cannot_pass(self):
        data = qualification_fixture()
        data["checks"][0]["app"]["id"] = 9
        self.assertTrue(any("wrong app" in x for x in op.qualify(**data)))

    def test_legacy_status_cannot_spoof_app_bound_check(self):
        data = qualification_fixture()
        data["checks"] = []
        data["statuses"] = [{"context": "gate", "state": "success"}]
        self.assertTrue(any("wrong app" in x for x in op.qualify(**data)))

    def test_skipped_neutral_fail_without_head_bound_explanation(self):
        for conclusion in ("skipped", "neutral"):
            data = qualification_fixture()
            item = data["checks"][0]
            item["conclusion"] = conclusion
            self.assertTrue(op.qualify(**data))
            data["exceptions"] = [{"kind": "check", "name": "gate", "head_sha": SHA, "app_id": 15368,
                                    "conclusion": conclusion, "reason": "conditional job", "source_reference": "workflow@sha:line"}]
            self.assertFalse(op.qualify(**data))
            data["exceptions"][0]["head_sha"] = BASE
            self.assertTrue(op.qualify(**data))

    def test_signed_review_resolved_and_stable_base_guards(self):
        changes = [("commits", []), ("reviews", {"headRefOid": SHA, "baseRefOid": BASE, "mergeStateStatus": "CLEAN", "unresolved_threads": 1}),
                   ("reviews", {"headRefOid": SHA, "baseRefOid": SHA, "mergeStateStatus": "CLEAN", "unresolved_threads": 0})]
        for field, value in changes:
            data = qualification_fixture()
            data[field] = value
            self.assertTrue(op.qualify(**data))

    def test_required_review_cannot_be_invented_for_solo_repo(self):
        data = qualification_fixture()
        data["policy"]["reviews"]["required_approving_review_count"] = 1
        self.assertIn("required review approval not satisfied", op.qualify(**data))

    def test_signature_reason_must_be_valid(self):
        data = qualification_fixture()
        data["commits"][0]["commit"]["verification"]["reason"] = "unknown_key"
        self.assertTrue(op.qualify(**data))

    def test_newer_failed_run_beats_older_rerun(self):
        old = {"id": 100, "run_attempt": 9, "conclusion": "success"}
        new = {"id": 101, "run_attempt": 1, "conclusion": "failure"}
        self.assertEqual(op.latest([old, new], lambda x: "same"), [new])

    def test_pagination_collects_failure_after_first_hundred(self):
        page = [{"id": i, "conclusion": "success"} for i in range(100)]
        failed = {"id": 101, "conclusion": "failure"}
        with patch.object(op, "gh_api", side_effect=[{"check_runs": page, "total_count": 101},
                                                   {"check_runs": [failed], "total_count": 101}]) as api:
            data = op.gh_pages("/repos/a/b/checks", "check_runs")
        self.assertEqual(data[-1], failed)
        self.assertEqual(api.call_count, 2)

    def test_incomplete_count_and_repeated_page_refused(self):
        with patch.object(op, "gh_api", return_value={"check_runs": [], "total_count": 1}):
            with self.assertRaises(op.OperatorError):
                op.gh_pages("/repos/a/b/checks", "check_runs")
        with patch.object(op, "gh_api", return_value=[{"id": i} for i in range(100)]):
            with self.assertRaisesRegex(op.OperatorError, "repeated"):
                op.gh_pages("/repos/a/b/checks")

    def test_rulesets_supply_policy_even_without_legacy_checks(self):
        rules = [{"type": "required_signatures"}, {"type": "required_status_checks", "parameters": {
                 "required_status_checks": [{"context": "gate", "integration_id": 15368}]}},
                 {"type": "pull_request", "parameters": {"allowed_merge_methods": ["squash"], "required_approving_review_count": 0}}]
        with patch.object(op, "gh_api", return_value={}), patch.object(op, "gh_pages", return_value=rules):
            policy = op.protection("a/b", "main")
        self.assertEqual(policy["required_checks"], [{"context": "gate", "app_id": 15368}])
        self.assertTrue(policy["signatures_required"])
        self.assertEqual(policy["allowed_methods"], ["squash"])

    def test_linear_history_forbids_merge_commits_from_either_policy_source(self):
        for legacy, rules in (({"required_linear_history": {"enabled": True}}, []),
                              (None, [{"type": "required_linear_history"}])):
            with self.subTest(legacy=bool(legacy)), patch.object(op, "gh_api", return_value=legacy), \
                 patch.object(op, "gh_pages", return_value=rules):
                policy = op.protection("a/b", "main")
            self.assertEqual(policy["allowed_methods"], ["rebase", "squash"])
            snapshot = {"green": True, "head_sha": SHA, "policy": policy}
            with patch.object(op, "snapshot_pr", return_value=snapshot), patch.object(op, "gh_api") as api:
                with self.assertRaisesRegex(op.OperatorError, "merge method forbidden"):
                    op.guard_merge("a/b", 1, method="merge", execute=True, expected_head=SHA)
                api.assert_not_called()

    def test_required_team_reviewers_fail_closed_as_unsupported(self):
        rules = [{"type": "required_status_checks", "parameters": {
                    "required_status_checks": [{"context": "gate", "integration_id": 15368}]}},
                 {"type": "pull_request", "parameters": {
                    "required_approving_review_count": 0,
                    "required_reviewers": [{"file_patterns": ["*"], "minimum_approvals": 1,
                                            "reviewer": {"id": 123, "type": "Team"}}]}}]
        with patch.object(op, "gh_api", return_value=None), patch.object(op, "gh_pages", return_value=rules):
            policy = op.protection("a/b", "main")
        self.assertIn("pull_request.required_reviewers", policy["unsupported_rules"])
        fixture = qualification_fixture()
        fixture["policy"] = policy
        for decision in (None, "APPROVED"):
            fixture["reviews"]["reviewDecision"] = decision
            with self.subTest(decision=decision):
                self.assertTrue(any("unsupported applicable rules" in item for item in op.qualify(**fixture)))

    def test_empty_team_reviewers_adds_no_unsupported_requirement(self):
        rules = [{"type": "pull_request", "parameters": {"required_reviewers": []}}]
        with patch.object(op, "gh_api", return_value=None), patch.object(op, "gh_pages", return_value=rules):
            policy = op.protection("a/b", "main")
        self.assertEqual(policy["unsupported_rules"], [])

    def test_guard_read_only_unless_execute(self):
        snap = {"green": True, "head_sha": SHA, "policy": {"allowed_methods": ["squash"]}}
        with patch.object(op, "snapshot_pr", return_value=snap), patch.object(op, "gh_api") as api:
            self.assertFalse(op.guard_merge("a/b", 1)["merge_performed"])
            api.assert_not_called()

    def test_guard_refuses_changed_base_or_policy_without_mutation(self):
        first = {"green": True, "head_sha": SHA, "base_sha": BASE,
                 "policy": {"allowed_methods": ["squash"], "policy_sha256": "one"}}
        for changed in ("base", "policy"):
            second = copy.deepcopy(first)
            if changed == "base":
                second["base_sha"] = SHA
            else:
                second["policy"]["policy_sha256"] = "two"
            with patch.object(op, "snapshot_pr", side_effect=[first, second]), patch.object(op, "gh_api") as api:
                with self.assertRaisesRegex(op.OperatorError, "moved"):
                    op.guard_merge("a/b", 1, execute=True, expected_head=SHA)
                api.assert_not_called()


class MergeAttemptOutcomes(unittest.TestCase):
    def setUp(self):
        self.snapshot = {"green": True, "head_sha": SHA, "base_sha": BASE,
                         "policy": {"allowed_methods": ["squash"], "policy_sha256": "one"}}

    def measure(self, responses):
        with patch.object(op, "snapshot_pr", return_value=self.snapshot), \
             patch.object(op, "gh_api", side_effect=responses) as api:
            result = op.safe_measure("guard-merge", lambda: op.guard_merge(
                "a/b", 1, execute=True, expected_head=SHA))
        writes = [call for call in api.call_args_list if call.kwargs.get("method") == "PUT"]
        self.assertEqual(len(writes), 1)
        return result, api

    def assert_unknown(self, result, sha=None, merged=None):
        self.assertEqual(result["state"], "UNKNOWN_AFTER_ATTEMPT")
        self.assertIn("read PR before retry", result["error"])
        self.assertEqual(result["value"], {
            "repository": "a/b", "number": 1, "qualified_head": SHA,
            "mutation_attempted": True, "retry_performed": False,
            "merge_response_sha": sha, "merge_response_merged": merged,
        })

    def test_put_transport_and_parse_failures_are_unknown_without_retry(self):
        for error in (op.Unavailable("connection closed"), op.OperatorError("invalid JSON response")):
            with self.subTest(error=type(error).__name__):
                result, api = self.measure([error])
                self.assert_unknown(result)
                self.assertEqual(api.call_count, 1)

    def test_malformed_put_response_is_unknown_without_retry(self):
        for response in (None, [], "invalid", {}, {"merged": True}, {"merged": True, "sha": "invalid"}):
            with self.subTest(response=response):
                result, api = self.measure([response])
                self.assert_unknown(result, merged=response.get("merged") if isinstance(response, dict) else None)
                self.assertEqual(api.call_count, 1)

    def test_rejected_put_preserves_response_evidence_without_retry(self):
        result, api = self.measure([{"merged": False, "sha": BASE}])
        self.assert_unknown(result, BASE, False)
        self.assertEqual(api.call_count, 1)

    def test_failed_readback_preserves_successful_put_sha_without_retry(self):
        for readback in (op.Unavailable("readback offline"), op.OperatorError("invalid JSON"),
                         None, [], {"merged": False}, {"merged": True, "merge_commit_sha": SHA}):
            with self.subTest(readback=readback):
                result, api = self.measure([{"merged": True, "sha": BASE}, readback])
                self.assert_unknown(result, BASE, True)
                self.assertEqual(api.call_count, 2)

    def test_success_requires_response_sha_and_matching_readback(self):
        result, api = self.measure([{"merged": True, "sha": BASE},
                                    {"merged": True, "merge_commit_sha": BASE}])
        self.assertEqual(result["state"], "MEASURED")
        self.assertTrue(result["value"]["merged"])
        self.assertEqual(result["value"]["merge_sha"], BASE)
        self.assertEqual(api.call_count, 2)

    def test_cli_emits_explicit_unknown_state_and_nonzero_exit(self):
        output = io.StringIO()
        with patch.object(op, "snapshot_pr", return_value=self.snapshot), \
             patch.object(op, "gh_api", side_effect=[{"merged": True, "sha": BASE},
                                                     op.Unavailable("readback offline")]) as api, \
             patch("sys.stdout", output):
            status = op.main(["guard-merge", "--repo", "a/b", "--pr", "1",
                              "--execute", "--expected-head", SHA])
        self.assertEqual(status, 3)
        self.assert_unknown(json.loads(output.getvalue()), BASE, True)
        self.assertEqual(api.call_count, 2)


class PublicationIntegrity(unittest.TestCase):
    def validate(self, fixture):
        receipt, bundle, summary, contract = fixture
        return op.validate_publication(receipt, bundle, summary, repo=op.FORGE, sha=SHA, run_id=RUN, contract=contract)

    def test_valid_full_receipt_with_unsigned_bounds(self):
        result = self.validate(publication_fixture())
        self.assertTrue(result["receipt_hash_verified"])
        self.assertTrue(result["authenticity_not_established"])
        self.assertEqual(result["full_case_count"], 4)
        self.assertEqual(result["fallback"]["semantic_safety"], "UNQUALIFIED")
        self.assertEqual(result["model_qualification"], "UNQUALIFIED")

    def test_selfconsistent_provider_claims_and_removed_bounds_rejected(self):
        for key, value in (("provider_execution_revision", SHA), ("provider_hardware_fingerprint", "H100"),
                           ("runtime_version", "attested"), ("known_bounds", ["unsigned"]),
                           ("provider_route", "forged:model"), ("provider", "invented"),
                           ("configuration_sha256", "0" * 64), ("comparison", {"score_rate_delta": 1000})):
            f = publication_fixture()
            f[0][key] = value
            if key in f[2]:
                f[2][key] = value
            rehash(*f[:3])
            with self.subTest(key=key), self.assertRaises(op.OperatorError):
                self.validate(f)

    def test_selfconsistent_fabricated_response_score_is_rejected(self):
        f = publication_fixture()
        record = f[1]["candidate_results"][0]
        record["response_content"] = "invented non JSON response"
        record["response_sha256"] = hashlib.sha256(record["response_content"].encode()).hexdigest()
        f[0]["candidate_results_sha256"] = op.digest(f[1]["candidate_results"])
        rehash(*f[:3])
        with self.assertRaisesRegex(op.OperatorError, "response score"):
            self.validate(f)

    def test_flags_only_fallback_and_invented_semantics_rejected(self):
        for change in ("flags_only", "parsed_contradiction", "authority"):
            f = publication_fixture()
            fallback = f[0]["fallback_evidence"]
            if change == "flags_only":
                fallback.clear()
                fallback.update(pass_=True, invalid_candidate_request_failed=True, baseline_fallback_succeeded=True)
                fallback["pass"] = fallback.pop("pass_")
            elif change == "parsed_contradiction":
                fallback["baseline_fallback_score"].update({k: True for k in op.SCORE_FLAGS})
                fallback["baseline_fallback_score"]["points"] = 6
            else:
                fallback.update(production_authority="FULL", selected_fallback_output={"tool_calls": ["delete"]})
            f[0]["fallback_evidence_sha256"] = op.digest(fallback)
            rehash(*f[:3])
            with self.subTest(change=change), self.assertRaises(op.OperatorError):
                self.validate(f)

    def test_source_fixture_labels_and_visibility_cannot_be_invented(self):
        for place, key in (("fixture_manifest", "suite"), ("fixture_manifest", "schema"),
                           ("source_checks", "visibility_verified")):
            f = publication_fixture()
            f[1][place][key] = "invented"
            with self.assertRaises(op.OperatorError):
                self.validate(f)

    def test_deterministic_guard_is_distinguished_from_failed_model_semantics(self):
        f = publication_fixture()
        safe = {"decision": "ESCALATE", "answer": "PROVIDER_UNAVAILABLE", "evidence_ids": ["F1"], "tool_calls": []}
        case = {"id": "fallback", "expected": {"decision": "ESCALATE", "answer_equals": "PROVIDER_UNAVAILABLE", "evidence_ids": ["F1"]}}
        f[3]["provider"].update(FALLBACK_CONTRACT="szl.frontier.safe-fallback.v1", SAFE_FALLBACK_OUTPUT=safe, FALLBACK_CASE=case)
        fallback = f[0]["fallback_evidence"]
        fallback.update(contract="szl.frontier.safe-fallback.v1", fallback_case_sha256=op.digest(case),
                        trigger_case_id=f[3]["case_ids"][0], baseline_semantic_pass=False,
                        transport_pass=True, semantic_safety_pass=True, selected_fallback_output=safe,
                        selected_fallback_output_sha256=op.digest(safe), selected_fallback_source="DETERMINISTIC_SAFETY_GUARD",
                        production_authority="NONE")
        f[0]["fallback_qualification"] = {"transport_pass": True, "semantic_safety_pass": True, "production_fallback_qualified": True}
        f[0]["fallback_evidence_sha256"] = op.digest(fallback)
        rehash(*f[:3])
        result = self.validate(f)
        self.assertEqual(result["fallback"]["semantic_safety"], "DETERMINISTIC_GUARD_VERIFIED")
        self.assertEqual(result["fallback"]["baseline_model_semantics"], "FAILED")
        self.assertEqual(result["model_qualification"], "UNQUALIFIED")

    def test_tampered_receipt_hash_rejected(self):
        f = publication_fixture()
        f[0]["known_bounds"].append("tamper")
        with self.assertRaisesRegex(op.OperatorError, "content address"):
            self.validate(f)

    def test_hash_null_convention_not_accepted(self):
        f = publication_fixture()
        f[0]["receipt_sha256"] = None
        f[0]["receipt_sha256"] = op.digest(f[0])
        with self.assertRaisesRegex(op.OperatorError, "content address"):
            self.validate(f)

    def test_rehashed_wrong_source_run_or_smoke_rejected(self):
        for field, value in (("source_revision", BASE), ("source_repository", "attacker/repo"),
                             ("run_id", "arbitrary"), ("suite", "smoke"), ("runner_sha256", "0" * 64)):
            f = publication_fixture()
            f[0]["run_identity"][field] = value
            rehash(*f[:3])
            with self.assertRaises(op.OperatorError):
                self.validate(f)

    def test_full_label_with_smoke_only_records_rejected(self):
        f = publication_fixture()
        f[1]["candidate_results"] = f[1]["candidate_results"][:2]
        f[0]["candidate_results_sha256"] = op.digest(f[1]["candidate_results"])
        rehash(*f[:3])
        with self.assertRaisesRegex(op.OperatorError, "coverage"):
            self.validate(f)

    def test_summary_from_other_run_rejected(self):
        f = publication_fixture()
        f[2]["receipt_sha256"] = "0" * 64
        with self.assertRaisesRegex(op.OperatorError, "summary mismatch"):
            self.validate(f)

    def test_fabricated_metrics_even_when_rehashed_rejected(self):
        f = publication_fixture()
        f[0]["candidate_metrics"]["score_rate"] = 0.5
        f[2]["candidate_metrics"]["score_rate"] = 0.5
        rehash(*f[:3])
        with self.assertRaisesRegex(op.OperatorError, "aggregate metrics"):
            self.validate(f)

    def test_tampered_result_bytes_and_source_metadata_rejected(self):
        f = publication_fixture()
        f[1]["candidate_results"][0]["response_content"] = "tampered"
        f[0]["candidate_results_sha256"] = op.digest(f[1]["candidate_results"])
        rehash(*f[:3])
        with self.assertRaisesRegex(op.OperatorError, "response content"):
            self.validate(f)
        f = publication_fixture()
        f[1]["source_checks"]["files"]["LICENSE"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(op.OperatorError, "metadata hash"):
            self.validate(f)

    def test_hold_none_and_fallback_failure_rejected(self):
        for field, value in (("production_disposition", "PROMOTE"), ("promotion_effect", "YES"),
                             ("fallback_evidence", {"pass": False})):
            f = publication_fixture()
            f[0][field] = value
            rehash(*f[:3])
            with self.assertRaises(op.OperatorError):
                self.validate(f)

    def test_exact_single_directory_required(self):
        root = f"runs/2026/09/07/{RUN}/glm-5-3-flash"
        paths = [f"{root}/{name}.json" for name in ("receipt", "bundle", "summary")]
        self.assertEqual(op.select_receipt_root(paths, RUN), root)
        for invalid in (paths + [paths[0].replace("09/07", "09/08")],
                        [paths[0], paths[1].replace("09/07", "09/08"), paths[2]],
                        [x.replace(RUN, RUN + "-smoke") for x in paths]):
            with self.assertRaises(op.OperatorError):
                op.select_receipt_root(invalid, RUN)

    def test_hf_tree_and_files_use_metadata_revision(self):
        with patch.object(op, "http_json", side_effect=[({"sha": SHA}, {}), ([{"type": "file", "path": "receipt.json"}], {})]) as fetch:
            self.assertEqual(op.hf_dataset_snapshot()["revision"], SHA)
            self.assertIn(f"/tree/{SHA}", fetch.call_args_list[1].args[0])
        with patch.object(op, "http_json", return_value=({}, {})) as fetch:
            op.hf_file(op.DATASET, SHA, "receipt.json")
            self.assertIn(f"/resolve/{SHA}/receipt.json", fetch.call_args.args[0])

    def test_hf_pagination_cannot_escape_host_or_revision(self):
        for next_url in (f"https://attacker.example/tree/{SHA}", "https://huggingface.co/api/tree/main"):
            with patch.object(op, "http_json", side_effect=[({"sha": SHA}, {}), ([], {"Link": f'<{next_url}>; rel="next"'})]):
                with self.assertRaises(op.OperatorError):
                    op.hf_dataset_snapshot()

    def test_pr_smoke_workflow_cannot_prove_main(self):
        with patch.object(op, "gh_api", return_value={"merged": True, "merged_at": "now", "merge_commit_sha": SHA}), \
             patch.object(op, "workflows", return_value=[{"head_sha": SHA, "head_branch": "main", "path": op.WORKFLOW,
                                                        "event": "pull_request", "status": "completed", "conclusion": "success"}]):
            with self.assertRaisesRegex(op.OperatorError, "push-to-main"):
                op.verify_forge_174()


class ResourceAndTruthGuards(unittest.TestCase):
    def test_secret_redaction(self):
        with patch.dict(os.environ, {"TEST_API_TOKEN": "synthetic-secret-value"}):
            text = op.sanitize("synthetic-secret-value hf_123456789012345678901234 ghp_1234567890 github_pat_1234567890 "
                               "Bearer somecredential password=plaintext https://user:password@host/")
        for secret in ("synthetic-secret-value", "1234567890", "somecredential", "plaintext", "user:password"):
            self.assertNotIn(secret, text)

    def test_redaction_preserves_distinct_public_hf_json_keys(self):
        value = {"hf_revision": SHA, "hf_publication_revision": BASE}
        self.assertEqual(op.strict_json(op.sanitize(json.dumps(value))), value)

    def test_output_and_timeout_are_bounded_without_raw_files(self):
        with self.assertRaises(op.Unavailable):
            op.run([sys.executable, "-c", "print('x'*100000)"], max_bytes=100)
        with self.assertRaises(op.Unavailable):
            op.run([sys.executable, "-c", "import time;time.sleep(2)"], timeout=0.05)

    def test_duplicate_and_nonfinite_json_rejected(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(op.OperatorError):
                op.strict_json(raw)

    def test_measurement_does_not_relabel_failure(self):
        self.assertEqual(op.safe_measure("x", lambda: {"state": "MEASURED_FAIL"})["state"], "MEASURED_FAIL")
        self.assertEqual(op.safe_measure("x", lambda: {"state": "open", "green": False})["state"], "MEASURED")
        def unavailable():
            raise op.Unavailable("provider offline")
        self.assertEqual(op.safe_measure("x", unavailable)["state"], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
