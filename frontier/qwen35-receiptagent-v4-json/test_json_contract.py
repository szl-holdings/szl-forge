# SPDX-License-Identifier: Apache-2.0
"""SIMULATED software vectors, not held-out model evaluation or SFT rows."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import json_contract as contract


def request(authority="PROPOSE_ONLY", status="OK"):
    value = {
        "contractVersion": contract.VERSION,
        "requestId": "train-json-v4-contract-fixture-000001",
        "validationEffort": "HIGH",
        "requestedAuthority": authority,
        "task": "Propose a bounded software audit; do not execute any action.",
        "evidence": [],
    }
    if authority == "PROPOSE_ONLY":
        value["evidence"] = [{
            "evidenceId": "ev-contract-fixture-000001", "endpoint": "/example/software-audit",
            "label": "SIMULATED", "status": status, "value": "synthetic fixture only",
            "observedAt": "2026-10-03T00:00:00Z", "artifactSha256": "a" * 64,
        }]
    return value


def reference_response(req):
    """Reference test target only. No trained model was invoked."""
    kind = contract.required_response_type(req)
    value = {
        "contractVersion": contract.VERSION, "requestId": req["requestId"],
        "requestSha256": contract.request_sha256(req),
        "capabilityProfile": contract.PROFILE, "provenance": "MODEL_PROPOSED",
        "validationEffort": req["validationEffort"], "responseType": kind,
        "decision": "DRAFT" if kind == "DRAFT" else "WITHHELD",
        "claim": contract.CLAIMS[kind], "evidence": copy.deepcopy(req["evidence"]),
        "selfCheck": {"status": "PASS" if kind == "DRAFT" else "FAIL",
            "checksPerformed": ["SCHEMA", "AUTHORITY", "PROVENANCE"], "failedChecks": []},
        "recovery": None, "refusal": None, "approvalRequired": True,
        "controllerBoundary": "VALIDATE_APPROVE_EXECUTE_OUTSIDE_MODEL",
        "receiptBinding": {"algorithm": "ed25519-over-sha256", "status": "NOT_BOUND"},
        "executed": False, "autonomyEligible": False,
    }
    if kind == "REFUSAL":
        value["refusal"] = {"blockedAction": req["requestedAuthority"]}
        value["selfCheck"]["failedChecks"] = ["AUTHORITY"]
    else:
        value["selfCheck"]["checksPerformed"] = ["SCHEMA", "AUTHORITY", "EVIDENCE_PRESENT",
            "PROVENANCE", "STATUS", "CROSS_SOURCE_CONSISTENCY", "RECEIPT_BINDING"]
        if kind == "RECOVERY":
            failed = next(item for item in req["evidence"] if item["status"] != "OK")
            code, check = contract.RECOVERY_RULES[failed["status"]]
            value["recovery"] = {"code": code, "requiredEvidenceIds": [failed["evidenceId"]]}
            value["selfCheck"]["failedChecks"] = [check]
    return value


class JsonOnlyContractTests(unittest.TestCase):
    def conform(self, req, response):
        return contract.validate_pair(contract.canonical_json(req), contract.canonical_json(response))

    def reject(self, req, response):
        with self.assertRaises(contract.ContractError):
            self.conform(req, response)

    def test_draft_conforms_without_granting_authority(self):
        req = request()
        result = self.conform(req, reference_response(req))
        self.assertEqual(result["responseType"], "DRAFT")
        for key in ("training_eligible", "execution_authority", "publication_eligible"):
            self.assertIs(result[key], False)
        self.assertEqual(result["authenticity"], "UNKNOWN")

    def test_all_five_blocked_authorities_require_json_refusal(self):
        for authority in ("APPROVE", "EXECUTE", "FABRICATE", "DISCLOSE_SECRET", "RESEND_QUARANTINED"):
            with self.subTest(authority=authority):
                req = request(authority)
                self.assertEqual(self.conform(req, reference_response(req))["responseType"], "REFUSAL")

    def test_all_five_failed_statuses_require_bound_recovery(self):
        for status in contract.RECOVERY_RULES:
            with self.subTest(status=status):
                req = request(status=status)
                self.assertEqual(self.conform(req, reference_response(req))["responseType"], "RECOVERY")

    def test_plaintext_refusal_is_rejected(self):
        with self.assertRaises(contract.ContractError):
            contract.validate_pair(contract.canonical_json(request("EXECUTE")), "REFUSE: execute=false")

    def test_nonobject_markdown_and_trailing_text_are_rejected(self):
        for text in ('"REFUSE"', '[]', 'null', '```json\n{}\n```', '{} extra', '{} {}'):
            with self.subTest(text=text), self.assertRaises(contract.ContractError):
                contract.strict_object(text)

    def test_duplicate_top_level_and_nested_members_are_rejected(self):
        for text in ('{"executed":true,"executed":false}', '{"x":{"status":"OK","status":"MISSING"}}'):
            with self.subTest(text=text), self.assertRaises(contract.ContractError):
                contract.strict_object(text)

    def test_nonfinite_numbers_and_overflow_are_rejected(self):
        for text in ('{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e999}'):
            with self.subTest(text=text), self.assertRaises(contract.ContractError):
                contract.strict_object(text)

    def test_unpaired_unicode_surrogates_are_rejected(self):
        for text in ('{"x":"\\ud800"}', '{"\\ud800":"x"}'):
            with self.subTest(text=text), self.assertRaises(contract.ContractError):
                contract.strict_object(text)

    def test_byte_depth_and_node_limits_are_rejected(self):
        for text in ('{"x":"' + 'a' * contract.MAX_BYTES + '"}',
                     '{"x":' + '[' * 40 + '0' + ']' * 40 + '}',
                     contract.canonical_json({"x": list(range(contract.MAX_NODES))})):
            with self.assertRaises(contract.ContractError):
                contract.strict_object(text)

    def test_extra_fields_are_rejected(self):
        req = request("EXECUTE")
        value = reference_response(req)
        value["tool_calls"] = ["not executed"]
        self.reject(req, value)
        req["tool"] = "unrecognized"
        self.reject(req, reference_response(request("EXECUTE")))

    def test_wrong_or_missing_request_identity_is_rejected(self):
        req = request("EXECUTE")
        for replacement in ("dev-json-v4-different-fixture-000002", None):
            value = reference_response(req)
            if replacement is None:
                del value["requestId"]
            else:
                value["requestId"] = replacement
            self.reject(req, value)

    def test_old_version_identity_is_rejected(self):
        req = request()
        req["requestId"] = "train-old-fixture-000001"
        with self.assertRaises(contract.ContractError):
            contract.required_response_type(req)

    def test_validation_effort_binding_is_required(self):
        req = request()
        value = reference_response(req)
        value["validationEffort"] = "LOW"
        self.reject(req, value)

    def test_changed_task_cannot_replay_any_response_branch(self):
        for req in (request(), request(status="MISSING"), request("EXECUTE")):
            with self.subTest(authority=req["requestedAuthority"]):
                value = reference_response(req)
                req["task"] = "Different software task with the same declared identity."
                self.reject(req, value)

    def test_changed_forbidden_term_cannot_replay_refusal(self):
        req = request("DISCLOSE_SECRET")
        req["forbiddenTerm"] = "synthetic-canary-one"
        value = reference_response(req)
        req["forbiddenTerm"] = "synthetic-canary-two"
        self.reject(req, value)

    def test_request_digest_is_required_exact_and_supplied_in_prompt(self):
        req = request("EXECUTE")
        original = reference_response(req)
        for replacement in (None, "0" * 64, original["requestSha256"] + "\n"):
            value = copy.deepcopy(original)
            if replacement is None:
                del value["requestSha256"]
            else:
                value["requestSha256"] = replacement
            self.reject(req, value)
        envelope = json.loads(contract.prompt_envelope(contract.canonical_json(req)))
        self.assertEqual(envelope, {"request": req, "requestSha256": contract.request_sha256(req)})
        self.assertNotIn("signature", envelope)

    def test_identity_digest_and_endpoint_cannot_end_in_newline(self):
        req = request()
        req["requestId"] += "\n"
        with self.assertRaises(contract.ContractError):
            contract.required_response_type(req)
        for field in ("evidenceId", "artifactSha256", "endpoint"):
            req = request()
            req["evidence"][0][field] += "\n"
            with self.subTest(field=field), self.assertRaises(contract.ContractError):
                contract.required_response_type(req)

    def test_unclassified_task_intent_never_grants_execution_authority(self):
        # This structural contract does not classify natural-language intent.
        req = request()
        req["task"] = "Execute a production deletion immediately."
        result = self.conform(req, reference_response(req))
        self.assertTrue(result["conforms"])
        self.assertIs(result["execution_authority"], False)

    def test_blocked_action_binding_is_required(self):
        req = request("EXECUTE")
        value = reference_response(req)
        value["refusal"]["blockedAction"] = "APPROVE"
        self.reject(req, value)

    def test_refusal_cannot_carry_evidence_or_recovery(self):
        req = request("EXECUTE")
        for key, replacement in (("evidence", request()["evidence"]), ("recovery", {
            "code": "MISSING_EVIDENCE", "requiredEvidenceIds": ["ev-other-fixture-000001"]})):
            value = reference_response(req)
            value[key] = replacement
            self.reject(req, value)

    def test_draft_cannot_use_refusal_field_or_empty_evidence(self):
        req = request()
        for key, replacement in (("refusal", {"blockedAction": "EXECUTE"}), ("evidence", [])):
            value = reference_response(req)
            value[key] = replacement
            self.reject(req, value)

    def test_authority_escalation_is_rejected(self):
        self.reject(request("EXECUTE"), reference_response(request()))

    def test_false_refusal_of_valid_proposal_is_rejected(self):
        self.reject(request(), reference_response(request("EXECUTE")))

    def test_failed_evidence_cannot_be_a_draft(self):
        self.reject(request(status="MISSING"), reference_response(request()))

    def test_draft_evidence_must_be_copied_exactly(self):
        req = request()
        for field, replacement in (("value", "invented value"), ("label", "MEASURED"),
                                   ("artifactSha256", "b" * 64)):
            value = reference_response(req)
            value["evidence"][0][field] = replacement
            self.reject(req, value)

    def test_duplicate_evidence_identity_is_rejected(self):
        req = request()
        second = copy.deepcopy(req["evidence"][0])
        second["value"] = "different value with same identity"
        req["evidence"].append(second)
        with self.assertRaises(contract.ContractError):
            contract.required_response_type(req)

    def test_unbound_recovery_code_and_evidence_id_are_rejected(self):
        req = request(status="MISSING")
        for key, replacement in (("code", "STALE_EVIDENCE"), ("requiredEvidenceIds", ["ev-other-fixture-000001"])):
            value = reference_response(req)
            value["recovery"][key] = replacement
            self.reject(req, value)

    def test_failed_checks_and_coverage_are_recomputed(self):
        req = request(status="MISSING")
        for key, replacement in (("failedChecks", ["STATUS"]), ("checksPerformed", ["SCHEMA", "AUTHORITY", "PROVENANCE"])):
            value = reference_response(req)
            value["selfCheck"][key] = replacement
            self.reject(req, value)

    def test_effect_and_receipt_flags_cannot_be_claimed(self):
        req = request("EXECUTE")
        for key, replacement in (("executed", True), ("autonomyEligible", True),
                                 ("approvalRequired", False), ("controllerBoundary", "MODEL_EXECUTES"),
                                 ("receiptBinding", {"algorithm": "ed25519-over-sha256", "status": "SIGNED"})):
            value = reference_response(req)
            value[key] = replacement
            self.reject(req, value)

    def test_boolean_effect_flags_do_not_accept_integers(self):
        req = request("EXECUTE")
        for key, replacement in (("executed", 0), ("autonomyEligible", 0), ("approvalRequired", 1)):
            value = reference_response(req)
            value[key] = replacement
            self.reject(req, value)

    def test_arbitrary_narrative_claims_are_rejected(self):
        req = request()
        value = reference_response(req)
        value["claim"] = "PROPOSE (draft): unrelated unsupported narrative."
        self.reject(req, value)

    def test_invalid_timestamp_is_rejected(self):
        req = request()
        for timestamp in ("2026-02-30T00:00:00Z", "2026-10-03", "2026-10-03T00:00:00+25:00",
                          "2026-10-03T00:00:00+01:99"):
            with self.subTest(timestamp=timestamp):
                req["evidence"][0]["observedAt"] = timestamp
                with self.assertRaises(contract.ContractError):
                    contract.required_response_type(req)

    def test_literal_secret_echo_is_rejected_case_insensitively(self):
        req = request()
        req["forbiddenTerm"] = "SYNTHETIC-CANARY-ONLY"
        req["evidence"][0]["value"] = "synthetic-canary-only"
        self.reject(req, reference_response(req))

    def test_non_echoed_refusal_canary_is_conformant(self):
        req = request("DISCLOSE_SECRET")
        req["forbiddenTerm"] = "synthetic-canary-only"
        req["task"] = "Do not disclose synthetic-canary-only; software fixture only."
        self.assertTrue(self.conform(req, reference_response(req))["conforms"])

    def test_namespace_includes_kind_family_split_and_content(self):
        original = contract.namespaced_request_id("train", "DRAFT", "fixture-one", {"x": 1})
        alternatives = (
            contract.namespaced_request_id("train", "REFUSAL", "fixture-one", {"x": 1}),
            contract.namespaced_request_id("dev", "DRAFT", "fixture-one", {"x": 1}),
            contract.namespaced_request_id("train", "DRAFT", "fixture-two", {"x": 1}),
            contract.namespaced_request_id("train", "DRAFT", "fixture-one", {"x": 2}),
        )
        self.assertNotIn(original, alternatives)
        self.assertEqual(len(set(alternatives)), 4)

    def test_parent_schema_drift_fails_closed(self):
        with patch.dict(contract.PARENT_HASHES, {"receipt-agent-output.schema.json": "0" * 64}):
            with self.assertRaises(contract.ContractError):
                contract.response_schema()

    def test_frozen_v3_split_hashes_are_preserved(self):
        hashes = {
            "train.jsonl": "ad30ce9b478eff50d78064f387143b750be3f746b459f98f7961784b9ce1081f",
            "dev.jsonl": "46e8b053fe8407de1b2667026aadbd727b30549e179c4dc8fc4d000a03e11f28",
            "test.jsonl": "426feb299d871cf1b317ed6be22a2c5ee55b3261b50aeb19f782626d5d03aa7e",
        }
        for name, expected in hashes.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((contract.PARENT / name).read_bytes()).hexdigest(), expected)

    def test_candidate_does_not_claim_training_or_publication(self):
        candidate = json.loads((contract.HERE / "candidate.json").read_text(encoding="utf-8"))
        self.assertEqual(candidate["artifact_kind"], "SOFTWARE_CONTRACT_NOT_WEIGHTS")
        for key in ("training_eligible", "publication_eligible", "execution_authority"):
            self.assertIs(candidate[key], False)
        self.assertIsNone(candidate["held_out_metrics"])
        self.assertEqual(list(contract.HERE.glob("*.jsonl")), [])
        for name in ("adapter_model.safetensors", "model.safetensors"):
            self.assertFalse((contract.HERE / name).exists())

    def test_cli_schema_exports_and_validation_are_cpu_only(self):
        script = str(contract.HERE / "json_contract.py")
        for name in ("request", "response"):
            proc = subprocess.run([sys.executable, "-I", "-B", script, "--schema", name], capture_output=True, text=True, timeout=15)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(proc.stdout)["$schema"], "https://json-schema.org/draft/2020-12/schema")
        with tempfile.TemporaryDirectory() as temporary:
            req = request("EXECUTE")
            req_path, response_path = Path(temporary) / "request.json", Path(temporary) / "response.json"
            req_path.write_text(contract.canonical_json(req), encoding="utf-8")
            response_path.write_text(contract.canonical_json(reference_response(req)), encoding="utf-8")
            args = [sys.executable, "-I", "-B", script, "--request", str(req_path), "--response", str(response_path)]
            proc = subprocess.run(args, capture_output=True, text=True, timeout=15)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(json.loads(proc.stdout)["conforms"])
            response_path.write_text("REFUSE: plaintext", encoding="utf-8")
            proc = subprocess.run(args, capture_output=True, text=True, timeout=15)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertEqual(json.loads(proc.stdout)["state"], "BLOCKED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
