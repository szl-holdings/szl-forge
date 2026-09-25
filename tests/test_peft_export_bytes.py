"""Deterministic adversarial byte fixtures; no model, provider, or GPU calls."""
from __future__ import annotations

import copy
import json
import math
import struct
import unittest
from unittest.mock import patch

from tools.evaluate_peft_export import (
    MAX_BYTES, MAX_HEADER, MAX_TENSORS, UPSTREAM_REVISION,
    digest, fixture, inspect_adapter, main,
)

A = "base.model.layers.0.q_proj.lora_A.weight"
B = "base.model.layers.0.q_proj.lora_B.weight"


class SerializedLoraTests(unittest.TestCase):
    def setUp(self):
        self.tensors = {A: ([2, 3], [0.25] * 6), B: ([4, 2], [0.5] * 8)}
        self.args = fixture(self.tensors)

    def report(self, tensors=None, **kwargs):
        return inspect_adapter(*fixture(self.tensors if tensors is None else tensors, **kwargs))

    def mutate(self, callback):
        raw, config, binding = self.args
        count = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + count])
        payload = raw[8 + count:]
        header, payload = callback(header, payload)
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        result = struct.pack("<Q", len(encoded)) + encoded + payload
        binding = copy.deepcopy(binding)
        binding["artifactSha256"] = digest(result)
        return inspect_adapter(result, config, binding)

    def test_positive_is_not_authority(self):
        report = self.report()
        self.assertEqual((report["state"], report["moduleCount"]), ("PASS", 1))
        self.assertEqual(report["upstreamRevision"], UPSTREAM_REVISION)
        self.assertEqual(report["bindingAssurance"], "CALLER_DECLARED_NOT_AUTHENTICATED")
        for key in ("publicationAuthorized", "productionAuthorized", "automaticPromotion"):
            self.assertIs(report[key], False)
        self.assertEqual(report["disposition"], "EVALUATION_HOLD")
        for key in ("upstreamWarningExecution", "distributedGather", "exactBaseModelReload", "providerReadback"):
            self.assertEqual(report[key], "NOT_RUN")

    def test_supported_float_payloads(self):
        for dtype in ("F16", "BF16", "F32", "F64"):
            with self.subTest(dtype=dtype):
                self.assertEqual(self.report(dtype=dtype)["state"], "PASS")

    def test_nonfinite_actual_payloads(self):
        for dtype in ("F16", "BF16", "F32", "F64"):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(dtype=dtype, value=value):
                    tensors = dict(self.tensors)
                    tensors[A] = ([2, 3], [value] + [1.0] * 5)
                    self.assertEqual(self.report(tensors, dtype=dtype)["reason"], "NONFINITE_TENSOR_PAYLOAD")

    def test_flat_and_empty_shards_are_rejected(self):
        for name in (A, B):
            for shape, values in (([6], [1.0] * 6), ([0, 2], []), ([], [1.0])):
                with self.subTest(name=name, shape=shape):
                    tensors = dict(self.tensors)
                    tensors[name] = (shape, values)
                    self.assertEqual(self.report(tensors)["reason"], "UNGATHERED_AB_SHAPE")

    def test_explicit_state_dict_never_bypasses_byte_check(self):
        tensors = dict(self.tensors)
        tensors[A] = ([6], [1.0] * 6)
        self.assertEqual(self.report(tensors, provenance="EXPLICIT")["reason"], "UNGATHERED_AB_SHAPE")
        self.assertEqual(self.report(provenance="EXPLICIT")["distributedGather"], "NOT_RUN")

    def test_auxiliary_1d_is_not_an_ab_false_positive(self):
        for auxiliary, updates in (("lora_magnitude_vector", {"use_dora": True}),
                                   ("lora_E", {"peft_type": "ADALORA"})):
            with self.subTest(auxiliary=auxiliary):
                tensors = dict(self.tensors)
                name = "base.model.linear." + auxiliary
                tensors[name] = ([4], [1.0] * 4)
                report = self.report(tensors, config_updates=updates)
                self.assertEqual(report["state"], "UNSUPPORTED")
                self.assertEqual(report["auxiliaryKeysOutsideABShapeHeuristic"], [name])

    def test_no_ab_and_missing_pair(self):
        self.assertEqual(self.report({"something.weight": ([2, 3], [0.0] * 6)})["reason"], "NO_LORA_AB_TENSORS")
        self.assertEqual(self.report({A: self.tensors[A]})["reason"], "INCOMPLETE_AB_PAIR")

    def test_pair_rank_and_dtype(self):
        tensors = dict(self.tensors)
        tensors[B] = ([4, 3], [0.0] * 12)
        self.assertEqual(self.report(tensors)["reason"], "AB_RANK_MISMATCH")
        # Same width, different dtype: finite values remain finite, but pair must fail.
        raw, config, binding = fixture(self.tensors, dtype="F16")
        self.args = raw, config, binding
        result = self.mutate(lambda h, p: ({**h, B: {**h[B], "dtype": "BF16"}}, p))
        self.assertEqual(result["reason"], "AB_DTYPE_MISMATCH")

    def test_config_rank_boolean_is_not_integer(self):
        self.assertEqual(self.report(config_updates={"r": True})["reason"], "INVALID_CONFIG_RANK")

    def test_convolution_is_unqualified_not_malformed_shard(self):
        tensors = {A: ([2, 3, 1], [1.0] * 6), B: ([4, 2], [1.0] * 8)}
        self.assertEqual(self.report(tensors)["state"], "UNSUPPORTED")

    def test_exact_inventory_and_shapes_required(self):
        raw, config, binding = self.args
        for shapes, reason in (({}, "EXPECTED_INVENTORY_REQUIRED"),
                               ({A: [2, 3]}, "TENSOR_INVENTORY_MISMATCH"),
                               ({A: [2, 4], B: [4, 2]}, "EXPECTED_SHAPE_MISMATCH"),
                               ({A: [True, 3], B: [4, 2]}, "EXPECTED_SHAPE_INVALID")):
            with self.subTest(reason=reason):
                self.assertEqual(inspect_adapter(raw, config, {**binding, "expectedTensorShapes": shapes})["reason"], reason)

    def test_exact_source_base_and_content_binding(self):
        raw, config, binding = self.args
        for key, value, reason in (
            ("peftRevision", "b" * 40, "PEFT_REVISION_MISMATCH"),
            ("baseModelRevision", "main", "BASE_REVISION_NOT_EXACT"),
            ("baseModelRevision", "b" * 40, "CONFIG_BASE_BINDING_MISMATCH"),
            ("baseModelId", "other/model", "CONFIG_BASE_BINDING_MISMATCH"),
            ("artifactSha256", "0" * 64, "ARTIFACT_DIGEST_MISMATCH"),
            ("configSha256", "0" * 64, "CONFIG_DIGEST_MISMATCH"),
            ("stateDictProvenance", "GATHERED_CERTIFIED", "PROVENANCE_CLASS"),
        ):
            with self.subTest(key=key, reason=reason):
                self.assertEqual(inspect_adapter(raw, config, {**binding, key: value})["reason"], reason)
        self.assertEqual(inspect_adapter(raw + b"x", config, binding)["reason"], "ARTIFACT_DIGEST_MISMATCH")

    def test_offsets_no_gaps_overlap_or_trailing_bytes(self):
        for start, end, reason in ((1, 25, "PAYLOAD_GAP_OR_OVERLAP"),
                                   (0, 20, "TENSOR_BYTE_LENGTH"),
                                   (-1, 23, "OFFSET_BOUNDS"),
                                   (0, 9999, "OFFSET_BOUNDS"),
                                   (False, 24, "INVALID_OFFSETS")):
            with self.subTest(start=start, end=end):
                result = self.mutate(lambda h, p: ({**h, A: {**h[A], "data_offsets": [start, end]}}, p))
                self.assertEqual(result["reason"], reason)
        self.assertEqual(self.mutate(lambda h, p: (h, p + b"tail"))["reason"], "UNACCOUNTED_PAYLOAD")
        self.assertEqual(self.mutate(lambda h, p: (h, p[:-1]))["reason"], "OFFSET_BOUNDS")
        self.assertEqual(self.mutate(lambda h, p: ({**h, B: {**h[B], "data_offsets": [0, 32]}}, p))["reason"],
                         "PAYLOAD_GAP_OR_OVERLAP")

    def test_invalid_layout_is_rejected_before_value_scan(self):
        with patch("tools.evaluate_peft_export.finite_values", side_effect=AssertionError("early scan")) as scan:
            result = self.mutate(lambda h, p: ({**h, B: {**h[B], "data_offsets": [0, 32]}}, p))
        self.assertEqual(result["reason"], "PAYLOAD_GAP_OR_OVERLAP")
        scan.assert_not_called()

    def test_malformed_descriptor_shape_metadata_and_dtype(self):
        changes = [({**{}, "dtype": "I32"}, "UNSUPPORTED_DTYPE"),
                   ({"shape": [True, 3]}, "INVALID_SHAPE"),
                   ({"shape": [-1, 3]}, "INVALID_SHAPE"),
                   ({"shape": [MAX_BYTES + 1, 2]}, "INVALID_SHAPE"),
                   ({"shape": [1, 1, 1, 1, 1]}, "INVALID_SHAPE"),
                   ({"extra": "no"}, "TENSOR_DESCRIPTOR")]
        for delta, reason in changes:
            with self.subTest(reason=reason, delta=delta):
                self.assertEqual(self.mutate(lambda h, p: ({**h, A: {**h[A], **delta}}, p))["reason"], reason)
        self.assertEqual(self.mutate(lambda h, p: ({**h, "__metadata__": {"x": 1}}, p))["reason"], "INVALID_METADATA")

    def test_duplicate_and_nonfinite_json_fail_closed(self):
        raw, config, binding = self.args
        for bad, reason in ((b'{"x":1,"x":2}', "DUPLICATE_JSON_KEY"),
                            (b'{"x":NaN}', "NONFINITE_JSON_NUMBER"),
                            (b'{"x":1e9999}', "NONFINITE_JSON_NUMBER"),
                            (b'[]', "JSON_OBJECT_REQUIRED"), (b'\xff', "INVALID_JSON")):
            with self.subTest(bad=bad):
                bound = {**binding, "configSha256": digest(bad)}
                self.assertEqual(inspect_adapter(raw, bad, bound)["reason"], reason)
        duplicate = b'{"duplicate":{},"duplicate":{}}'
        bad = struct.pack("<Q", len(duplicate)) + duplicate
        self.assertEqual(inspect_adapter(bad, config, {**binding, "artifactSha256": digest(bad)})["reason"],
                         "DUPLICATE_JSON_KEY")

    def test_empty_and_oversized_format_limits(self):
        raw, config, binding = self.args
        for bad, reason in ((b"", "ARTIFACT_SIZE_OR_TYPE"),
                            (b"x" * (MAX_BYTES + 1), "ARTIFACT_SIZE_OR_TYPE"),
                            (struct.pack("<Q", MAX_HEADER + 1) + b"{}", "HEADER_SIZE"),
                            (struct.pack("<Q", 100) + b"{}", "HEADER_SIZE"),
                            (struct.pack("<Q", 2) + b"{}", "TENSOR_COUNT")):
            with self.subTest(reason=reason, length=len(bad)):
                self.assertEqual(inspect_adapter(bad, config, {**binding, "artifactSha256": digest(bad)})["reason"], reason)
        tensors = {f"weight_{i}": ([1], [1.0]) for i in range(MAX_TENSORS + 1)}
        self.assertEqual(self.report(tensors)["reason"], "TENSOR_COUNT")

    def test_name_substring_cannot_hide_as_a_lora_pair(self):
        tensors = dict(self.tensors)
        tensors["base.model.linear.lora_A_fake.weight"] = ([1], [1.0])
        self.assertEqual(self.report(tensors)["state"], "UNSUPPORTED")

    def test_order_and_repeat_are_deterministic(self):
        self.assertEqual(self.report(), self.report(dict(reversed(list(self.tensors.items())))))
        self.assertEqual(self.report(), self.report())

    def test_cli_has_no_filesystem_or_network_effect(self):
        with patch("builtins.open", side_effect=AssertionError("unexpected open")), \
                patch("socket.socket", side_effect=AssertionError("unexpected socket")), \
                patch("builtins.print") as output:
            self.assertEqual(main(), 0)
        self.assertEqual(json.loads(output.call_args.args[0])["disposition"], "EVALUATION_HOLD")


if __name__ == "__main__":
    unittest.main()
