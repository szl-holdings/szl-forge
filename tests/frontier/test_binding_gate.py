"""SIMULATED binding claims through the callback API; no model execution."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from frontier.harness.heldout_gate import load_probes, run_gate
from tests.frontier.test_binding_admission import claims


class BindingGateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "software-probes.jsonl"
        self.path.write_text(json.dumps({"id": "fixture", "kind": "refusal",
                                         "prompt": "Software fixture question"}) + "\n",
                             encoding="utf-8")
        _, self.probe_sha = load_probes(str(self.path))
        self.declared, self.observed = claims()
        self.declared["probe_set_sha256"] = self.probe_sha
        self.observed["probe_set_sha256"] = self.probe_sha

    def run_claims(self, generate, **kwargs):
        return run_gate(artifact="SZLHOLDINGS/fixture-candidate", probes_path=str(self.path),
                        baseline={"refusal": 0}, generate=generate, **kwargs)

    def assert_no_authority(self, receipt):
        for key in ("candidate_evaluated", "qualification_gate_ran", "publication_eligible",
                    "promotion_eligible"):
            self.assertIs(receipt[key], False)
        self.assertEqual(receipt["promotion_effect"], "NONE")
        self.assertEqual(receipt["authority"], "NONE")

    def test_legacy_callback_without_binding_stays_unavailable(self):
        receipt = self.run_claims(lambda _: "REFUSE")
        self.assertEqual(receipt["gate"], "PASS")
        self.assertEqual(receipt["binding_admission"]["status"], "UNAVAILABLE")
        self.assert_no_authority(receipt)

    def test_matching_claims_never_establish_verified_candidate_evaluation(self):
        receipt = self.run_claims(lambda _: "REFUSE", declared_binding=self.declared,
                                  observed_binding=self.observed, mock=True)
        self.assertEqual(receipt["gate"], "PASS")
        self.assertEqual(receipt["evals"], "SIMULATED")
        binding = receipt["binding_admission"]
        self.assertEqual(binding["label"], "DECLARED")
        self.assertEqual(binding["status"], "DECLARED_BINDING_MATCH")
        self.assertIs(binding["artifact_bytes_verified"], False)
        self.assertIs(binding["loader_verified"], False)
        self.assertEqual(binding["observation_independence"], "UNKNOWN")
        self.assertEqual(binding["identity"], self.declared)
        self.assertEqual(len(binding["claims_sha256"]), 64)
        self.assert_no_authority(receipt)

    def test_every_identity_mismatch_rejects_before_generator(self):
        for key in self.declared:
            with self.subTest(key=key):
                changed = dict(self.observed)
                changed[key] = "mismatch"
                generate = mock.Mock(side_effect=AssertionError("must not generate"))
                receipt = self.run_claims(generate, declared_binding=self.declared,
                                          observed_binding=changed)
                self.assertEqual(receipt["gate"], "INVALID")
                self.assertIs(receipt["generator_invoked"], False)
                self.assertEqual(receipt["binding_admission"]["label"], "BLOCKED")
                self.assertNotIn("rows", receipt)
                self.assert_no_authority(receipt)
                generate.assert_not_called()

    def test_one_sided_claim_is_not_silently_ignored(self):
        generate = mock.Mock(side_effect=AssertionError("must not generate"))
        for inputs in ({"declared_binding": self.declared},
                       {"observed_binding": self.observed}):
            with self.subTest(inputs=list(inputs)):
                receipt = self.run_claims(generate, **inputs)
                self.assertEqual(receipt["gate"], "INVALID")
                self.assert_no_authority(receipt)
        generate.assert_not_called()

    def test_wrong_probe_binding_rejects_even_when_claims_match_each_other(self):
        self.declared["probe_set_sha256"] = "0" * 64
        self.observed["probe_set_sha256"] = "0" * 64
        generate = mock.Mock(side_effect=AssertionError("must not generate"))
        receipt = self.run_claims(generate, declared_binding=self.declared,
                                  observed_binding=self.observed)
        self.assertEqual(receipt["gate"], "INVALID")
        generate.assert_not_called()

    def test_probe_hash_rejection_precedes_any_binding_or_generation(self):
        generate = mock.Mock(side_effect=AssertionError("must not generate"))
        receipt = self.run_claims(generate, declared_probe_sha256="0" * 64,
                                  declared_binding=False, observed_binding=False)
        self.assertEqual(receipt["gate"], "INVALID")
        self.assertIn("probe set hash mismatch", receipt["reason"])
        self.assert_no_authority(receipt)
        generate.assert_not_called()

    def test_environment_cannot_spoof_binding_admission(self):
        receipt = self.run_claims(lambda _: "REFUSE", env={
            "binding_admission": {"label": "MEASURED", "loader_verified": True}})
        self.assertEqual(receipt["binding_admission"]["label"], "UNAVAILABLE")
        self.assertIs(receipt["binding_admission"]["loader_verified"], False)
        self.assert_no_authority(receipt)

    def test_generator_mutating_inputs_cannot_rewrite_binding_snapshot(self):
        original_revision = self.declared["artifact_revision"]

        def mutate(_):
            self.declared["artifact_revision"] = "changed"
            self.observed["adapter_admission"]["fully_applied"] = False
            return "REFUSE"

        receipt = self.run_claims(mutate, declared_binding=self.declared,
                                  observed_binding=self.observed)
        self.assertEqual(receipt["binding_admission"]["identity"]["artifact_revision"],
                         original_revision)
        self.assert_no_authority(receipt)


if __name__ == "__main__":
    unittest.main()
