"""SIMULATED claim fixtures; no model, provider readback or independent witness."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest

from frontier.harness.binding_admission import (
    BINDING_SCHEMA, MAX_STRUCTURAL_COUNT, check_binding,
)


def claims():
    declared = {
        "schema": BINDING_SCHEMA,
        "artifact_repo_id": "SZLHOLDINGS/fixture-candidate",
        "artifact_revision": "a" * 40,
        "artifact_file": "adapter/adapter_model.safetensors",
        "artifact_sha256": "b" * 64,
        "base_repo_id": "Qwen/fixture-base",
        "base_revision": "c" * 40,
        "source_repository": "szl-holdings/szl-forge",
        "source_revision": "d" * 40,
        "probe_set_sha256": "e" * 64,
        "expected_adapter": "default",
        "loader_class": "FixtureModelForConditionalGeneration",
    }
    observed = copy.deepcopy(declared)
    observed["adapter_admission"] = {
        "checkpoint_tensors": 4,
        "applied": 4,
        "unapplied": 0,
        "unapplied_sample": [],
        "checkpoint_layout": "language_model",
        "fully_applied": True,
        "expected_adapter": "default",
        "adapter_activation": {
            "enabled": True,
            "active_adapters": ["default"],
            "num_adapter_layers": 2,
            "available_adapters": ["default"],
            "merged_adapters": [],
        },
        "coverage_basis": "parameter_names_and_model_status",
        "numerical_application": "UNKNOWN",
    }
    return declared, observed


class BindingAdmissionTests(unittest.TestCase):
    def check(self, declared, observed, **kwargs):
        return check_binding(
            artifact=kwargs.get("artifact", "SZLHOLDINGS/fixture-candidate"),
            probe_sha256=kwargs.get("probe_sha256", "e" * 64),
            declared=declared, observed=observed,
        )

    def invalid(self, declared, observed, **kwargs):
        with self.assertRaisesRegex(ValueError, r"^BINDING_INVALID: "):
            self.check(declared, observed, **kwargs)

    def test_absent_claims_are_unavailable_not_an_admission(self):
        self.assertEqual(self.check(None, None), {
            "status": "UNAVAILABLE", "label": "UNAVAILABLE",
            "identity": None, "claims_sha256": None,
            "artifact_bytes_verified": False, "loader_verified": False,
            "observation_independence": "UNKNOWN",
        })

    def test_matching_claims_do_not_verify_bytes_loader_or_independence(self):
        declared, observed = claims()
        digest = hashlib.sha256(json.dumps({"declared": declared, "observed": observed},
                                           sort_keys=True, separators=(",", ":"),
                                           ensure_ascii=False).encode("utf-8")).hexdigest()
        self.assertEqual(self.check(declared, observed), {
            "status": "DECLARED_BINDING_MATCH", "label": "DECLARED",
            "identity": declared, "claims_sha256": digest,
            "artifact_bytes_verified": False, "loader_verified": False,
            "observation_independence": "UNKNOWN",
        })

    def test_one_sided_claims_are_invalid(self):
        declared, observed = claims()
        self.invalid(declared, None)
        self.invalid(None, observed)

    def test_nonobject_claims_are_invalid(self):
        declared, observed = claims()
        for value in (False, 0, "", [], ()):
            with self.subTest(value=value):
                self.invalid(value, observed)
                self.invalid(declared, value)

    def test_every_identity_field_is_required_on_both_sides(self):
        declared, observed = claims()
        for key in declared:
            for side in (0, 1):
                values = [copy.deepcopy(declared), copy.deepcopy(observed)]
                del values[side][key]
                with self.subTest(key=key, side=side):
                    self.invalid(*values)

    def test_unexpected_claim_fields_are_invalid(self):
        for side in (0, 1):
            declared, observed = claims()
            (declared if side == 0 else observed)["publication_eligible"] = True
            self.invalid(declared, observed)

    def test_immutable_pins_reject_mutable_nonhex_uppercase_and_wrong_types(self):
        for key, size in (("artifact_revision", 40), ("base_revision", 40),
                          ("source_revision", 40), ("artifact_sha256", 64),
                          ("probe_set_sha256", 64)):
            for value in (None, False, 1, "main", "g" * size, "A" * size,
                          "a" * (size - 1), "a" * (size + 1), "a" * size + "\n"):
                declared, observed = claims()
                declared[key] = observed[key] = value
                with self.subTest(key=key, value=value):
                    self.invalid(declared, observed)

    def test_immutable_identity_mismatch_is_invalid(self):
        for key, value in (("artifact_repo_id", "SZLHOLDINGS/other-candidate"),
                           ("artifact_revision", "f" * 40),
                           ("artifact_file", "other.safetensors"),
                           ("artifact_sha256", "f" * 64),
                           ("base_repo_id", "Qwen/other-base"),
                           ("base_revision", "f" * 40),
                           ("source_revision", "f" * 40),
                           ("probe_set_sha256", "f" * 64),
                           ("expected_adapter", "other"),
                           ("loader_class", "OtherModel")):
            declared, observed = claims()
            observed[key] = value
            with self.subTest(key=key):
                self.invalid(declared, observed)

    def test_wrong_artifact_or_computed_probe_identity_is_invalid(self):
        declared, observed = claims()
        for value in ("SZLHOLDINGS/other-candidate", "main", False, None):
            self.invalid(declared, observed, artifact=value)
        for value in ("f" * 64, "main", False, None):
            self.invalid(declared, observed, probe_sha256=value)

    def test_repository_ids_are_bounded_canonical_identifiers(self):
        for key in ("artifact_repo_id", "base_repo_id"):
            for value in ("owner", "/repo", "owner/", "../repo", "owner/a/b",
                          "https://example.invalid/model", "owner/re po", "owner/repo\0",
                          "owner/a..b", "owner/a--b", "owner/" + "x" * 97, False):
                declared, observed = claims()
                declared[key] = observed[key] = value
                with self.subTest(key=key, value=value):
                    self.invalid(declared, observed)

    def test_canonical_source_repository_cannot_be_replaced(self):
        for value in (None, "owner/source", "szl-holdings/SZL-Forge",
                      "szl-holdings/szl-forge\n"):
            declared, observed = claims()
            declared["source_repository"] = observed["source_repository"] = value
            self.invalid(declared, observed)

    def test_artifact_path_rejects_traversal_absolute_backslash_and_nonweights(self):
        for value in (None, "", "../model.safetensors", "/model.safetensors",
                      "a/../model.safetensors", "a//model.safetensors",
                      "a/./model.safetensors", "C:/model.safetensors",
                      "a\\model.safetensors", "model.bin", "model.safetensors\0",
                      "a/model.safetensors/", "x" * 1025):
            declared, observed = claims()
            declared["artifact_file"] = observed["artifact_file"] = value
            self.invalid(declared, observed)

    def test_adapter_namespace_and_loader_class_are_bounded_identifiers(self):
        for key, values in (("expected_adapter", (None, False, "", "a.b", "a\0", "a b", "a" * 129)),
                            ("loader_class", (None, False, "", "a.b", "0Model", "A\n", "A" * 129))):
            for value in values:
                declared, observed = claims()
                declared[key] = observed[key] = value
                with self.subTest(key=key, value=value):
                    self.invalid(declared, observed)

    def test_admission_object_and_every_guard_field_are_required(self):
        declared, observed = claims()
        for value in (None, False, [], "PASS"):
            changed = copy.deepcopy(observed)
            changed["adapter_admission"] = value
            self.invalid(declared, changed)
        for key in observed["adapter_admission"]:
            changed = copy.deepcopy(observed)
            del changed["adapter_admission"][key]
            self.invalid(declared, changed)

    def test_guard_fields_cannot_claim_authority_or_numerical_verification(self):
        for key, value in (("publication_eligible", True), ("fully_applied", 1),
                           ("coverage_basis", "trust_me"),
                           ("numerical_application", "MEASURED"),
                           ("checkpoint_layout", "unknown"),
                           ("unapplied_sample", ["missing.weight"]),
                           ("expected_adapter", "other")):
            declared, observed = claims()
            observed["adapter_admission"][key] = value
            with self.subTest(key=key):
                self.invalid(declared, observed)

    def test_checkpoint_counts_require_positive_exact_integers_and_complete_coverage(self):
        for key, values in (("checkpoint_tensors", (False, True, 0, -1, 4.0, "4", 5)),
                            ("applied", (False, True, 0, -1, 4.0, "4", 3)),
                            ("unapplied", (False, True, -1, 0.0, "0", 1))):
            for value in values:
                declared, observed = claims()
                observed["adapter_admission"][key] = value
                with self.subTest(key=key, value=value):
                    self.invalid(declared, observed)

    def test_structural_counts_and_available_names_are_bounded(self):
        for key in ("checkpoint_tensors", "applied"):
            declared, observed = claims()
            observed["adapter_admission"][key] = MAX_STRUCTURAL_COUNT + 1
            self.invalid(declared, observed)
        declared, observed = claims()
        activation = observed["adapter_admission"]["adapter_activation"]
        activation["num_adapter_layers"] = MAX_STRUCTURAL_COUNT + 1
        self.invalid(declared, observed)
        activation["num_adapter_layers"] = 2
        activation["available_adapters"] = ["default"] + [f"other_{i}" for i in range(128)]
        self.invalid(declared, observed)

    def test_activation_object_and_every_status_field_are_required(self):
        declared, observed = claims()
        for value in (None, False, [], "PASS"):
            changed = copy.deepcopy(observed)
            changed["adapter_admission"]["adapter_activation"] = value
            self.invalid(declared, changed)
        for key in observed["adapter_admission"]["adapter_activation"]:
            changed = copy.deepcopy(observed)
            del changed["adapter_admission"]["adapter_activation"][key]
            self.invalid(declared, changed)

    def test_enabled_and_layer_counts_fail_closed_without_type_coercion(self):
        for key, values in (("enabled", (False, None, 1, "True", "irregular")),
                            ("num_adapter_layers", (False, True, 0, -1, 2.0, "2"))):
            for value in values:
                declared, observed = claims()
                observed["adapter_admission"]["adapter_activation"][key] = value
                self.invalid(declared, observed)

    def test_only_expected_active_unmerged_namespace_is_admitted(self):
        for key, values in (("active_adapters", (None, "default", [], ["other"],
                                                  ["default", "other"], ["default", "default"])),
                            ("merged_adapters", (None, "", ["default"], False)),
                            ("available_adapters", (None, "default", [], ["other"],
                                                     ["default", "default"], ["default", False],
                                                     ["default", "bad.name"]))):
            for value in values:
                declared, observed = claims()
                observed["adapter_admission"]["adapter_activation"][key] = value
                with self.subTest(key=key, value=value):
                    self.invalid(declared, observed)

    def test_unused_available_adapter_does_not_claim_multiple_active_adapters(self):
        declared, observed = claims()
        observed["adapter_admission"]["adapter_activation"]["available_adapters"] += ["other"]
        self.assertEqual(self.check(declared, observed)["status"], "DECLARED_BINDING_MATCH")

    def test_custom_expected_namespace_matches_complete_guard_claim(self):
        declared, observed = claims()
        declared["expected_adapter"] = observed["expected_adapter"] = "candidate_v2"
        guard = observed["adapter_admission"]
        guard["expected_adapter"] = "candidate_v2"
        guard["adapter_activation"]["active_adapters"] = ["candidate_v2"]
        guard["adapter_activation"]["available_adapters"] = ["candidate_v2"]
        self.assertEqual(self.check(declared, observed)["label"], "DECLARED")

    def test_claims_are_not_mutated_and_result_does_not_alias_inputs(self):
        declared, observed = claims()
        before = copy.deepcopy((declared, observed))
        result = self.check(declared, observed)
        self.assertEqual((declared, observed), before)
        declared["artifact_revision"] = "f" * 40
        observed["adapter_admission"]["fully_applied"] = False
        self.assertEqual(result["status"], "DECLARED_BINDING_MATCH")
        self.assertEqual(result["identity"]["artifact_revision"], "a" * 40)
        self.assertNotEqual(result["identity"], declared)

    def test_returned_identity_does_not_mutate_caller_claim(self):
        declared, observed = claims()
        result = self.check(declared, observed)
        result["identity"]["artifact_revision"] = "f" * 40
        self.assertEqual(declared["artifact_revision"], "a" * 40)
        self.assertEqual(observed["artifact_revision"], "a" * 40)

    def test_claim_hash_captures_guard_and_immutable_identity(self):
        declared, observed = claims()
        original = self.check(declared, observed)["claims_sha256"]
        observed["adapter_admission"]["adapter_activation"]["num_adapter_layers"] = 3
        changed_guard = self.check(declared, observed)["claims_sha256"]
        self.assertNotEqual(original, changed_guard)
        declared["artifact_revision"] = observed["artifact_revision"] = "f" * 40
        self.assertNotEqual(changed_guard, self.check(declared, observed)["claims_sha256"])

    def test_claim_hash_is_independent_of_object_key_insertion_order(self):
        declared, observed = claims()
        reordered_declared = dict(reversed(list(declared.items())))
        reordered_observed = dict(reversed(list(observed.items())))
        self.assertEqual(self.check(declared, observed)["claims_sha256"],
                         self.check(reordered_declared, reordered_observed)["claims_sha256"])

    def test_errors_do_not_echo_untrusted_claim_contents(self):
        declared, observed = claims()
        secret = "fixture-private-string-do-not-echo"
        observed["artifact_revision"] = secret
        with self.assertRaises(ValueError) as caught:
            self.check(declared, observed)
        self.assertNotIn(secret, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
