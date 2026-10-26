"""SIMULATED, in-memory loader descriptor checks; never framework qualification."""
from __future__ import annotations

import builtins
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/estate_payload/loader_profile.py"
module = types.ModuleType("_simulated_loader_profile")
module.__file__ = str(SOURCE)
with patch.dict(sys.modules, {module.__name__: module}):
    exec(compile(SOURCE.read_bytes(), str(SOURCE), "exec"), module.__dict__)
profile = module

CANDIDATE = "a" * 64
RECIPE = "b" * 64
PIN = "0123456789abcdef" * 2 + "01234567"


def descriptor():
    return {
        "schema": "szl.loader-profile/v1", "purpose": "evaluate",
        "candidate_sha256": CANDIDATE, "recipe_sha256": RECIPE,
        "base": {"repo_id": "SIMULATED/base", "revision": PIN, "config_sha256": "c" * 64},
        "tokenizer": {"repo_id": "SIMULATED/tokenizer", "revision": "d" * 40,
                      "asset_manifest_sha256": "e" * 64},
        "runtime_identity_sha256": "f" * 64,
        "loader": {"backend": "transformers", "factory": "AutoModelForCausalLM"},
        "expected_model": {"class_name": "simulated.models.TextModel", "model_type": "simulated_text",
                           "view": "full_model"},
        "numerics": {"quantization": "none", "quant_type": "none", "compute_dtype": "float32",
                     "storage_dtype": "float32", "double_quant": False},
        "placement": {"device": "cpu", "offload": "none"},
    }


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def observe(raw, **overrides):
    expected = {
        "expected_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "expected_candidate_sha256": CANDIDATE, "expected_recipe_sha256": RECIPE,
        "expected_purpose": "evaluate",
    }
    expected.update(overrides)
    return profile.observe_loader_profile(raw, **expected)


def change(value, keys, replacement):
    node = value
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = replacement


class LoaderProfileContractTests(unittest.TestCase):
    def test_positive_result_is_only_comparison_not_authority(self):
        raw = encode(descriptor())
        result = observe(raw)
        self.assertEqual(result.profile_sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(result.candidate_sha256, CANDIDATE)
        self.assertEqual(result.recipe_sha256, RECIPE)
        self.assertEqual(result.purpose, "evaluate")
        self.assertEqual(result.base.repo_id, "SIMULATED/base")
        self.assertEqual(result.tokenizer.repo_id, "SIMULATED/tokenizer")
        self.assertEqual(result.loader.factory, "AutoModelForCausalLM")
        self.assertEqual(result.expected_model.class_name, "simulated.models.TextModel")
        self.assertEqual(result.numerics.compute_dtype, "float32")
        self.assertEqual(result.placement.device, "cpu")
        self.assertEqual(result.evidence_class, "MODELED")
        self.assertEqual(result.state, "BLOCKED")
        self.assertEqual(result.signing_trust, "UNKNOWN")
        self.assertIs(result.execution_authority, False)
        self.assertIs(result.training_admitted, False)
        self.assertIs(result.publication_eligible, False)
        with self.assertRaisesRegex(TypeError, "NOT_AUTHORIZATION"):
            bool(result)

    def test_observation_and_nested_fields_are_immutable_and_flags_not_inputs(self):
        result = observe(encode(descriptor()))
        for target, name, value in ((result, "purpose", "train"), (result.base, "revision", "0" * 40),
                                    (result.loader, "factory", "Other"),
                                    (result.numerics, "double_quant", True)):
            with self.subTest(field=name), self.assertRaises((dataclasses.FrozenInstanceError, AttributeError)):
                setattr(target, name, value)
        for name in ("execution_authority", "training_admitted", "publication_eligible"):
            with self.subTest(flag=name):
                with self.assertRaises((TypeError, ValueError)):
                    dataclasses.replace(result, **{name: True})
                with self.assertRaises(AttributeError):
                    object.__setattr__(result, name, True)
                self.assertIs(getattr(result, name), False)

    def test_external_pins_and_purpose_are_required_and_exact(self):
        raw = encode(descriptor())
        for key in ("expected_raw_sha256", "expected_candidate_sha256", "expected_recipe_sha256"):
            for invalid in (None, True, 0, b"a" * 64, "", "g" * 64, "A" * 64, "a" * 63, "a" * 64 + "\n"):
                with self.subTest(key=key, value=invalid), self.assertRaises(profile.LoaderProfileError):
                    observe(raw, **{key: invalid})
            with self.subTest(mismatched_pin=key), self.assertRaises(profile.LoaderProfileError):
                observe(raw, **{key: "0" * 64})
        for invalid in (None, True, "", "train", "export", "Evaluate", "evaluate "):
            with self.subTest(purpose=invalid), self.assertRaises(profile.LoaderProfileError):
                observe(raw, expected_purpose=invalid)
        with self.assertRaises(TypeError):
            profile.observe_loader_profile(raw)

    def test_raw_digest_binds_bytes_without_normalization_or_canonicalization(self):
        value = descriptor()
        raw = encode(value)
        spaced = json.dumps(value, indent=2).encode()
        self.assertNotEqual(raw, spaced)
        with self.assertRaises(profile.LoaderProfileError):
            observe(spaced, expected_raw_sha256=hashlib.sha256(raw).hexdigest())
        self.assertEqual(observe(spaced).profile_sha256, hashlib.sha256(spaced).hexdigest())

    def test_size_and_exact_bytes_type_are_checked_before_hash_or_parse(self):
        for invalid in (None, "{}", bytearray(b"{}"), memoryview(b"{}"), b"", b" " * (profile.MAX_PROFILE_BYTES + 1)):
            with self.subTest(type=type(invalid).__name__), \
                    patch.object(profile.hashlib, "sha256", side_effect=AssertionError("hash reached")), \
                    patch.object(profile.json, "loads", side_effect=AssertionError("parse reached")):
                with self.assertRaises(profile.LoaderProfileError):
                    profile.observe_loader_profile(invalid, expected_raw_sha256="0" * 64,
                        expected_candidate_sha256=CANDIDATE, expected_recipe_sha256=RECIPE,
                        expected_purpose="evaluate")

    def test_digest_mismatch_rejects_before_parsing(self):
        with patch.object(profile.json, "loads", side_effect=AssertionError("parse reached")):
            with self.assertRaises(profile.LoaderProfileError):
                observe(b"not JSON", expected_raw_sha256="0" * 64)

    def test_invalid_utf8_json_nonfinite_numbers_and_duplicate_keys_reject(self):
        raws = [b"\xff", b"\xef\xbb\xbf{}", b"{", b"null", b"[]", b'"x"',
                b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}', b'{"x":1}', b'{"x":1.0}',
                b'{"schema":"a","schema":"b"}', b'{"x":{"a":false,"a":true}}',
                rb'{"x":"\ud800"}', b"[" * 2000 + b"]" * 2000]
        for raw in raws:
            with self.subTest(raw=raw[:50]), self.assertRaises(profile.LoaderProfileError):
                observe(raw)

    def test_every_object_has_closed_required_fields(self):
        for path in ((), ("base",), ("tokenizer",), ("loader",), ("expected_model",),
                     ("numerics",), ("placement",)):
            original = descriptor()
            node = original
            for key in path:
                node = node[key]
            for missing in tuple(node):
                value = copy.deepcopy(original)
                target = value
                for key in path:
                    target = target[key]
                del target[missing]
                with self.subTest(path=path, missing=missing), self.assertRaises(profile.LoaderProfileError):
                    observe(encode(value))
            node["execution_authority"] = True
            with self.subTest(path=path, unknown=True), self.assertRaises(profile.LoaderProfileError):
                observe(encode(original))

    def test_all_fields_reject_wrong_types_and_bad_identifiers(self):
        values = [(("schema",), "other"), (("purpose",), ["evaluate"]),
                  (("candidate_sha256",), False), (("recipe_sha256",), "g" * 64),
                  (("runtime_identity_sha256",), None), (("base",), []),
                  (("tokenizer",), "same_as_base"), (("loader",), []),
                  (("expected_model",), []), (("numerics",), []), (("placement",), []),
                  (("base", "repo_id"), "https://example.invalid/model"),
                  (("base", "repo_id"), "../model"), (("tokenizer", "repo_id"), "model"),
                  (("base", "revision"), "main"), (("tokenizer", "revision"), "a" * 39),
                  (("base", "config_sha256"), "a" * 63),
                  (("tokenizer", "asset_manifest_sha256"), True),
                  (("expected_model", "class_name"), "Model"),
                  (("expected_model", "class_name"), "simulated.Model()"),
                  (("expected_model", "class_name"), "simulated." + "x" * 300),
                  (("expected_model", "model_type"), ""),
                  (("expected_model", "view"), "infer_from_name"),
                  (("numerics", "double_quant"), 1), (("placement", "device"), True)]
        for path, replacement in values:
            value = descriptor()
            change(value, path, replacement)
            with self.subTest(path=path, value=replacement), self.assertRaises(profile.LoaderProfileError):
                observe(encode(value))

    def test_supported_loader_pairs_are_explicit_and_no_fallback_list_is_allowed(self):
        pairs = (("transformers", "AutoModelForCausalLM"), ("transformers", "AutoModelForImageTextToText"),
                 ("transformers", "AutoModelForVision2Seq"), ("unsloth", "FastLanguageModel"))
        for backend, factory in pairs:
            value = descriptor()
            value["loader"] = {"backend": backend, "factory": factory}
            self.assertEqual(observe(encode(value)).loader.factory, factory)
        for backend, factory in (("transformers", "FastLanguageModel"), ("unsloth", "AutoModelForCausalLM"),
                                 ("arbitrary.module", "factory"), (["transformers", "unsloth"], "AutoModelForCausalLM")):
            value = descriptor()
            value["loader"] = {"backend": backend, "factory": factory}
            with self.subTest(backend=backend), self.assertRaises(profile.LoaderProfileError):
                observe(encode(value))

    def test_quantization_and_placement_are_explicit_consistent_and_bounded(self):
        value = descriptor()
        value["numerics"] = {"quantization": "bitsandbytes_4bit", "quant_type": "nf4",
                             "compute_dtype": "bfloat16", "storage_dtype": "uint8", "double_quant": True}
        value["placement"] = {"device": "cuda", "offload": "none"}
        result = observe(encode(value))
        self.assertTrue(result.numerics.double_quant)
        self.assertEqual(result.numerics.storage_dtype, "uint8")
        for path, replacement in ((("numerics", "quantization"), "auto"),
                                  (("numerics", "quant_type"), "none"),
                                  (("numerics", "storage_dtype"), "bfloat16"),
                                  (("numerics", "compute_dtype"), "auto"),
                                  (("numerics", "double_quant"), "true"),
                                  (("placement", "device"), "auto"),
                                  (("placement", "offload"), "cpu"),
                                  (("placement", "offload"), "disk")):
            altered = copy.deepcopy(value)
            change(altered, path, replacement)
            with self.subTest(path=path, value=replacement), self.assertRaises(profile.LoaderProfileError):
                observe(encode(altered))
        for field, replacement in (("quant_type", "nf4"), ("storage_dtype", "uint8"), ("double_quant", True)):
            altered = descriptor()
            altered["numerics"][field] = replacement
            with self.subTest(nonquantized_field=field), self.assertRaises(profile.LoaderProfileError):
                observe(encode(altered))

    def test_base_tokenizer_and_purpose_are_not_inferred_from_names(self):
        value = descriptor()
        value["base"]["repo_id"] = "misleading/Unsloth-Qwen-GGUF"
        value["tokenizer"]["repo_id"] = value["base"]["repo_id"]
        value["tokenizer"]["revision"] = value["base"]["revision"]
        before = copy.deepcopy(value)
        for purpose in ("train", "evaluate", "export"):
            value["purpose"] = purpose
            result = observe(encode(value), expected_purpose=purpose)
            self.assertEqual(result.loader.backend, "transformers")
            self.assertEqual(result.base.repo_id, result.tokenizer.repo_id)
            self.assertEqual(result.base.revision, result.tokenizer.revision)
            self.assertEqual(result.purpose, purpose)
            self.assertIs(result.training_admitted, False)
        value["purpose"] = before["purpose"]
        self.assertEqual(value, before)

    def test_validation_performs_no_io_dynamic_imports_or_mutations(self):
        raw = encode(descriptor())
        original_import = builtins.__import__
        def denied_import(name, *args, **kwargs):
            raise AssertionError("validation imported " + name)
        with patch("builtins.open", side_effect=AssertionError("file access")), \
                patch.object(builtins, "__import__", side_effect=denied_import):
            result = observe(raw)
        self.assertEqual(result.profile_sha256, hashlib.sha256(raw).hexdigest())
        self.assertIs(builtins.__import__, original_import)
        self.assertIs(result.publication_eligible, False)


if __name__ == "__main__":
    unittest.main()
