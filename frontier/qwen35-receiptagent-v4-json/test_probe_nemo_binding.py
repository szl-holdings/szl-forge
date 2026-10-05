# SPDX-License-Identifier: Apache-2.0
"""SIMULATED probe-control tests; no kernel, curriculum, or model execution."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_nemo_binding as probe


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("SIMULATED-no-source-on-disk")
        self.closed = []
        self.entered = []
        self.harness = b"SIMULATED harness source; not the real probe bytes"
        self.rules = ["R1_no_fabrication_label", "R2_honest_unknown"]
        self.kernel = SimpleNamespace(
            input_hash=Mock(return_value="sha256:" + "a" * 64),
            verify_chain=Mock(side_effect=[True, False]),
            provenance={"schema": "SIMULATED source binding", "authenticated": False},
        )
        self.gate = SimpleNamespace(
            check_pair=Mock(side_effect=[
                (self.rules, ["SIMULATED numeric denial"]),
                (self.rules, ["SIMULATED numeric denial"]),
                (self.rules, ["SIMULATED numeric denial"]),
                ([], []),
                ([], []),
                ([], []),
            ]),
            Decision=lambda **fields: fields,
            BLOCK="BLOCK", ALLOW="ALLOW", RULE_VERSION="SIMULATED",
            receipt_chain=Mock(side_effect=lambda decisions: [
                {"decision": dict(decision), "receipt_sha256": "sha256:" + str(i) * 64}
                for i, decision in enumerate(decisions)
            ]),
        )

    @contextmanager
    def simulated_kernel(self, root, reader):
        self.entered.append((root, reader))
        try:
            yield self.kernel
        finally:
            self.closed.append(True)

    @contextmanager
    def fixtures(self):
        with patch.object(probe.binding, "load_kernel", side_effect=self.simulated_kernel) as load, \
                patch.object(probe.admission, "_load_gate", return_value=self.gate) as gate, \
                patch.object(probe.admission, "_read_once", return_value=self.harness) as read:
            yield load, gate, read

    def test_success_retains_closed_authority_and_binds_probe_evidence(self):
        with self.fixtures() as (load, gate, read):
            result = probe.probe(self.root)
            load.assert_called_once_with(self.root, read)
            gate.assert_called_once_with("nemo_doctrine_gate.py", self.kernel)
            read.assert_called_once_with(Path(probe.__file__), probe.binding.MAX_SOURCE_BYTES)
        self.assertEqual(self.closed, [True])
        self.assertIs(result["software_checks_passed"], True)
        self.assertEqual(result["fixture_evidence_class"], "SIMULATED")
        self.assertEqual(result["cases"], 4)
        self.assertEqual(result["expected_block_cases"], 3)
        self.assertEqual(result["expected_allow_cases"], 1)
        self.assertEqual(result["fixture_sha256"], hashlib.sha256(
            probe.admission.contract.canonical_json({
                "required_cases": probe.CASES,
                "exploratory_cases": probe.EXPLORATORY_CASES,
            }).encode("utf-8")
        ).hexdigest())
        self.assertEqual(result["harness_source_sha256"], hashlib.sha256(self.harness).hexdigest())
        self.assertEqual(result["nemo_source_binding"], self.kernel.provenance)
        for key in ("training_eligible", "publication_eligible", "execution_authority"):
            self.assertIs(result[key], False)
        for key in ("model_training", "model_evaluation"):
            self.assertEqual(result[key], "NOT_RUN")
        self.assertEqual(result["signature"], "UNAVAILABLE")
        self.assertEqual(result["key_trust"], "REPO_DECLARED")
        self.assertEqual(result["energy_joules"], "UNAVAILABLE")
        self.assertEqual([item["decision"]["decision"] for item in result["receipt_chain"]],
                         ["BLOCK", "BLOCK", "BLOCK", "ALLOW"])

    def test_initial_unsupported_lexical_cases_remain_visible_without_pass_credit(self):
        with self.fixtures():
            result = probe.probe(self.root)
        self.assertEqual(probe.EXPLORATORY_CASES, ("accuracy_0.91", "0.91精度"))
        observations = result["exploratory_observations"]
        self.assertEqual([item["answer"] for item in observations], list(probe.EXPLORATORY_CASES))
        self.assertEqual([item["decision"] for item in observations], ["ALLOW", "ALLOW"])
        self.assertTrue(all("no passing-test credit" in item["scope"] for item in observations))
        self.assertEqual(result["cases"], 4)
        self.assertEqual(len(result["receipt_chain"]), 4)
        self.assertEqual(result["semantic_claim_detection"], "UNAVAILABLE")

    def test_main_success_output_is_ascii_with_no_authority(self):
        output = io.StringIO()
        with self.fixtures(), redirect_stdout(output):
            exit_code = probe.main(["--nemo-source-root", str(self.root)])
        self.assertEqual(exit_code, 0)
        self.assertTrue(output.getvalue().isascii())
        report = json.loads(output.getvalue())
        self.assertEqual(report["exploratory_observations"][1]["answer"], "0.91精度")
        for key in ("training_eligible", "publication_eligible", "execution_authority"):
            self.assertIs(report[key], False)

    def test_tamper_probe_uses_a_copy_and_verifies_changed_input_hash(self):
        with self.fixtures():
            result = probe.probe(self.root)
        original, tampered = [call.args[0] for call in self.kernel.verify_chain.call_args_list]
        self.assertIsNot(original, tampered)
        self.assertEqual(original[0]["decision"]["input_hash"], "sha256:" + "a" * 64)
        self.assertEqual(tampered[0]["decision"]["input_hash"], "sha256:" + "f" * 64)
        self.assertEqual(result["receipt_chain"], original)

    def test_wrong_numeric_rule_result_fails_and_closes_context(self):
        self.gate.check_pair.side_effect = [([], [])]
        with self.fixtures():
            with self.assertRaises(probe.admission.AdmissionError):
                probe.probe(self.root)
        self.assertEqual(self.closed, [True])
        self.gate.receipt_chain.assert_not_called()
        self.kernel.verify_chain.assert_not_called()

    def test_unexpected_rule_or_false_block_of_honest_case_is_not_credited(self):
        for results in ([(["R5_trust_ceiling"], ["SIMULATED wrong rule"])],
                        [(self.rules, [])] * 4):
            with self.subTest(results=results):
                self.gate.check_pair.side_effect = results
                with self.fixtures():
                    with self.assertRaises(probe.admission.AdmissionError):
                        probe.probe(self.root)

    def test_initial_chain_verifier_requires_literal_true(self):
        for invalid in (False, None, 1, "true"):
            with self.subTest(verdict=invalid):
                self.setUp()
                self.kernel.verify_chain.side_effect = [invalid]
                with self.fixtures():
                    with self.assertRaises(probe.admission.AdmissionError):
                        probe.probe(self.root)
                self.assertEqual(self.closed, [True])
                self.assertEqual(self.kernel.verify_chain.call_count, 1)

    def test_tampered_chain_verifier_requires_literal_false(self):
        for invalid in (True, None, 0, "false"):
            with self.subTest(verdict=invalid):
                self.setUp()
                self.kernel.verify_chain.side_effect = [True, invalid]
                with self.fixtures():
                    with self.assertRaises(probe.admission.AdmissionError):
                        probe.probe(self.root)
                self.assertEqual(self.closed, [True])

    def test_main_reports_blocked_on_missing_or_drifted_kernel(self):
        for error in (probe.binding.MissingSourceError("SIMULATED missing"),
                      probe.binding.BindingError("SIMULATED drift")):
            with self.subTest(error=type(error).__name__):
                output = io.StringIO()
                with patch.object(probe.binding, "load_kernel", side_effect=error), \
                        patch.object(probe.admission, "_load_gate") as gate, \
                        redirect_stdout(output):
                    exit_code = probe.main(["--nemo-source-root", str(self.root)])
                self.assertEqual(exit_code, 1)
                report = json.loads(output.getvalue())
                self.assertEqual(report["state"], "BLOCKED")
                self.assertIs(report["software_checks_passed"], False)
                for key in ("training_eligible", "publication_eligible", "execution_authority"):
                    self.assertIs(report[key], False)
                gate.assert_not_called()

    def test_main_reports_blocked_on_rule_or_receipt_failure(self):
        self.kernel.verify_chain.side_effect = [True, True]
        output = io.StringIO()
        with self.fixtures(), redirect_stdout(output):
            exit_code = probe.main(["--nemo-source-root", str(self.root)])
        self.assertEqual(exit_code, 1)
        self.assertEqual(self.closed, [True])
        report = json.loads(output.getvalue())
        self.assertEqual(report["state"], "BLOCKED")
        self.assertIs(report["software_checks_passed"], False)
        self.assertNotIn("receipt_chain", report)


if __name__ == "__main__":
    unittest.main()
