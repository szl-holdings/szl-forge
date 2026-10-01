# SPDX-License-Identifier: Apache-2.0
"""Small independently assembled GGUF fixtures; no models, CUDA or APIs."""
import contextlib
import io
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import gguf_forensics as m


def text(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def value(kind, data):
    if kind in m.PRIMITIVES:
        return struct.pack("<" + m.PRIMITIVES[kind], data)
    if kind == 8:
        return text(data)
    if kind == 9:
        element_kind, items = data
        return struct.pack("<IQ", element_kind, len(items)) + b"".join(value(element_kind, x) for x in items)
    return b""


def defaults():
    return [("general.architecture", 8, "qwen2"),
            ("tokenizer.ggml.model", 8, "gpt2"),
            ("tokenizer.ggml.tokens", 9, (8, ["a", "b", "@"])),
            ("tokenizer.chat_template", 8, "template-fixture-not-executed"),
            ("qwen2.context_length", 4, 1024)]


def tensors():
    return [{"name": "blk.0.weight", "shape": [2, 2], "type": 1,
             "data": struct.pack("<4e", 1, 2, 3, 4)},
            {"name": "output_norm.weight", "shape": [2], "type": 0,
             "data": struct.pack("<2f", 1, 2)}]


def fixture(metadata=None, ts=None, *, version=3, alignment=32, reverse_data=False, padding=0):
    metadata = defaults() if metadata is None else metadata
    ts = tensors() if ts is None else ts
    offsets, body = {}, bytearray()
    for index in (reversed(range(len(ts))) if reverse_data else range(len(ts))):
        body.extend(bytes([padding]) * ((-len(body)) % alignment))
        offsets[index] = len(body)
        body.extend(ts[index]["data"])
    header = bytearray(b"GGUF" + struct.pack("<IQQ", version, len(ts), len(metadata)))
    for key, kind, data in metadata:
        header.extend(text(key) + struct.pack("<I", kind) + value(kind, data))
    for index, tensor in enumerate(ts):
        header.extend(text(tensor["name"]) + struct.pack("<I", len(tensor["shape"])))
        header.extend(b"".join(struct.pack("<Q", x) for x in tensor["shape"]))
        header.extend(struct.pack("<IQ", tensor["type"], tensor.get("offset", offsets[index])))
    header.extend(bytes([padding]) * ((-len(header)) % alignment))
    return bytes(header + body)


def inspect(raw):
    return m.fingerprint_stream(io.BytesIO(raw), len(raw))


class ForensicTests(unittest.TestCase):
    def test_full_file_and_tensor_hashes(self):
        raw = fixture()
        result = inspect(raw)
        self.assertEqual(result["container_sha256"], m.digest(raw))
        actual = {t["name_sha256"]: t for t in result["tensors"]}
        for item in tensors():
            self.assertEqual(actual[m.digest(item["name"].encode())]["data_sha256"], m.digest(item["data"]))
        self.assertEqual(result["tensor_bytes"], 16)
        self.assertFalse(result["tensor_values_or_health_checked"])

    def test_container_change_does_not_imply_tensor_change(self):
        a = inspect(fixture())
        b = inspect(fixture(metadata=defaults() + [("general.name", 8, "different wrapper")]))
        result = m.compare(a, b)
        self.assertFalse(result["same_container_bytes"])
        self.assertTrue(result["same_tensor_set_bytes"])
        self.assertEqual(result["identical_tensors"], 2)
        self.assertFalse(result["root_cause_proven"])

    def test_storage_order_and_padding_do_not_change_tensor_fingerprint(self):
        a, b = inspect(fixture()), inspect(fixture(ts=list(reversed(tensors())), reverse_data=True, padding=7))
        self.assertTrue(m.compare(a, b)["same_tensor_set_bytes"])
        self.assertNotEqual(a["container_sha256"], b["container_sha256"])

    def test_changed_tensor_is_identified_without_values(self):
        ts = tensors(); ts[0]["data"] = struct.pack("<4e", 1, 2, 3, 5)
        result = m.compare(inspect(fixture()), inspect(fixture(ts=ts)))
        self.assertEqual(result["different_shared_tensors"], 1)
        self.assertFalse(result["same_tensor_set_bytes"])

    def test_shape_changes_identity_even_for_equal_payload(self):
        ts = tensors(); ts[0]["shape"] = [4]
        self.assertFalse(m.compare(inspect(fixture()), inspect(fixture(ts=ts)))["same_tensor_set_bytes"])

    def test_added_and_removed_tensor_counts(self):
        result = m.compare(inspect(fixture()), inspect(fixture(ts=tensors()[:1])))
        self.assertEqual(result["left_only_tensors"], 1)
        self.assertEqual(result["right_only_tensors"], 0)

    def test_template_separate_from_tokenizer(self):
        md = [(k, t, "changed" if k == "tokenizer.chat_template" else v) for k, t, v in defaults()]
        result = m.compare(inspect(fixture()), inspect(fixture(metadata=md)))
        self.assertTrue(result["same_tensor_set_bytes"])
        self.assertTrue(result["same_serialized_tokenizer_metadata"])
        self.assertFalse(result["same_serialized_chat_template_metadata"])

    def test_metadata_order_ignored(self):
        a, b = inspect(fixture()), inspect(fixture(metadata=list(reversed(defaults()))))
        self.assertEqual(a["metadata_fingerprints"], b["metadata_fingerprints"])
        self.assertTrue(m.compare(a, b)["same_tensor_set_bytes"])

    def test_absent_tokenizer_is_unknown_not_equal(self):
        raw = fixture(metadata=[defaults()[0]])
        result = m.compare(inspect(raw), inspect(raw))
        self.assertIsNone(result["same_serialized_tokenizer_metadata"])
        self.assertIsNone(result["same_serialized_chat_template_metadata"])

    def test_metadata_and_tensor_names_not_disclosed(self):
        md = defaults() + [("private.text", 8, "NEVER_EXPORT_PRIVATE_TEXT")]
        report = json.dumps(inspect(fixture(metadata=md)))
        self.assertNotIn("NEVER_EXPORT", report)
        self.assertNotIn("blk.0.weight", report)
        self.assertNotIn("template-fixture", report)

    def test_alignment_override(self):
        result = inspect(fixture(metadata=defaults() + [("general.alignment", 4, 64)], alignment=64))
        self.assertEqual(result["tensor_count"], 2)

    def test_all_scalar_metadata_types_and_nested_arrays(self):
        md = defaults() + [("test.t" + str(k), k, 1) for k in m.PRIMITIVES]
        md.append(("test.nested", 9, (9, [(8, ["a", "b"])])))
        self.assertEqual(inspect(fixture(metadata=md))["tensor_count"], 2)

    def test_v2_and_v3_supported(self):
        self.assertEqual(inspect(fixture(version=2))["gguf_version"], 2)
        self.assertEqual(inspect(fixture(version=3))["gguf_version"], 3)

    def test_big_endian_and_unknown_version_refused(self):
        for version in (1, 4, 0x03000000):
            with self.subTest(version=version), self.assertRaisesRegex(m.ForensicsError, "VERSION_OR_ENDIAN"):
                inspect(fixture(version=version))

    def test_magic_rejected(self):
        with self.assertRaisesRegex(m.ForensicsError, "BAD_MAGIC"):
            inspect(b"FAIL" + fixture()[4:])

    def test_all_truncation_points_refused(self):
        raw = fixture()
        for size in range(len(raw)):
            with self.subTest(size=size), self.assertRaises(m.ForensicsError):
                inspect(raw[:size])

    def test_unsupported_quantization_never_guessed(self):
        ts = tensors(); ts[0]["type"] = 12
        with self.assertRaisesRegex(m.ForensicsError, "UNSUPPORTED_TENSOR_TYPE"):
            inspect(fixture(ts=ts))

    def test_overlapping_and_unaligned_tensor_ranges_refused(self):
        for offset, expected in ((0, "OVERLAPPING"), (1, "ALIGNMENT"), (2**63, "RANGE")):
            ts = tensors(); ts[1]["offset"] = offset
            with self.subTest(offset=offset), self.assertRaisesRegex(m.ForensicsError, expected):
                inspect(fixture(ts=ts))

    def test_duplicate_tensor_and_metadata_names(self):
        ts = tensors(); ts[1]["name"] = ts[0]["name"]
        for raw in (fixture(ts=ts), fixture(metadata=defaults() + [defaults()[0]])):
            with self.assertRaisesRegex(m.ForensicsError, "DUPLICATE"):
                inspect(raw)

    def test_invalid_dimensions_refused(self):
        for shape in ([], [0], [2**32], [1]*5):
            ts = tensors(); ts[0]["shape"] = shape
            with self.subTest(shape=shape), self.assertRaisesRegex(m.ForensicsError, "DIMENSION"):
                inspect(fixture(ts=ts))

    def test_bad_alignment_and_boolean(self):
        for md in ([("general.alignment", 4, 7)], [("test.flag", 7, 2)]):
            with self.assertRaises(m.ForensicsError):
                inspect(fixture(metadata=defaults() + md))

    def test_metadata_nan_refused(self):
        with self.assertRaisesRegex(m.ForensicsError, "NONFINITE_METADATA"):
            inspect(fixture(metadata=defaults() + [("test.nan", 6, math.nan)]))

    def test_tensor_nan_is_not_mistaken_for_health_check(self):
        ts = tensors(); ts[0]["data"] = struct.pack("<4e", math.nan, math.inf, 0, -1)
        result = inspect(fixture(ts=ts))
        self.assertFalse(result["tensor_values_or_health_checked"])

    def test_architecture_and_split_scope(self):
        for md, code in (([("general.architecture", 8, "llama")], "OUTSIDE_QWEN2"),
                         ([("test.value", 4, 1)], "ARCHITECTURE_MISSING"),
                         (defaults()+[("split.no", 2, 0)], "SHARDED")):
            with self.subTest(code=code), self.assertRaisesRegex(m.ForensicsError, code):
                inspect(fixture(metadata=md))

    def test_invalid_utf8(self):
        raw = fixture().replace(b"qwen2", b"\xffwen2", 1)
        with self.assertRaisesRegex(m.ForensicsError, "UTF8"):
            inspect(raw)

    def test_header_and_count_budgets(self):
        with patch.object(m, "MAX_HEADER", 64), self.assertRaisesRegex(m.ForensicsError, "HEADER_LIMIT"):
            inspect(fixture())
        with patch.object(m, "MAX_TENSORS", 1), self.assertRaisesRegex(m.ForensicsError, "TENSOR_COUNT"):
            inspect(fixture())
        with patch.object(m, "MAX_METADATA", 1), self.assertRaisesRegex(m.ForensicsError, "METADATA_COUNT"):
            inspect(fixture())

    def test_string_array_and_node_budgets(self):
        with patch.object(m, "MAX_ARRAY_ITEMS", 1), self.assertRaisesRegex(m.ForensicsError, "ARRAY_LIMIT"):
            inspect(fixture())
        with patch.object(m, "MAX_VALUE_NODES", 1), self.assertRaisesRegex(m.ForensicsError, "VALUE_BUDGET"):
            inspect(fixture())
        ts = tensors(); ts[0]["name"] = "x" * 65
        with self.assertRaisesRegex(m.ForensicsError, "STRING_LIMIT"):
            inspect(fixture(ts=ts))

    def test_deadline_and_short_read(self):
        with patch.object(m.time, "monotonic", side_effect=[0, 9999]), self.assertRaisesRegex(m.ForensicsError, "DEADLINE"):
            inspect(fixture())
        class Short(io.BytesIO):
            def read(self, n): return super().read(max(0, n-1))
        with self.assertRaisesRegex(m.ForensicsError, "SHORT_READ"):
            m.fingerprint_stream(Short(fixture()), len(fixture()))

    def test_chunk_bounds_and_full_hash_with_trailer(self):
        raw = fixture() + b"Z" * (m.CHUNK + 17)
        class Bounded(io.BytesIO):
            def read(self, n):
                if n > m.CHUNK: raise AssertionError("oversized read")
                return super().read(n)
        result = m.fingerprint_stream(Bounded(raw), len(raw))
        self.assertEqual(result["container_sha256"], m.digest(raw))

    def test_file_identity_verification_is_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weights"; raw = fixture(); path.write_bytes(raw)
            result = m.inspect_file(path, m.digest(raw))
            self.assertTrue(result["matches_historical_from_digest"])
            self.assertEqual(path.read_bytes(), raw)
            with self.assertRaisesRegex(m.ForensicsError, "BLOB_DIGEST_MISMATCH"):
                m.inspect_file(path, "0"*64)
            self.assertEqual(path.read_bytes(), raw)

    def test_file_change_during_read_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weights"; raw = fixture(); path.write_bytes(raw)
            original = m.fingerprint_stream
            def changing(stream, size, seconds):
                result = original(stream, size, seconds)
                # Simulate an external writer, never part of the inspector.
                with path.open("ab") as other: other.write(b"X")
                return result
            with patch.object(m, "fingerprint_stream", side_effect=changing), self.assertRaisesRegex(m.ForensicsError, "CHANGED"):
                m.inspect_file(path, m.digest(raw))

    def test_link_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); target = root / "target"; target.write_bytes(fixture()); link = root / "link"
            try: link.symlink_to(target)
            except OSError: self.skipTest("Host does not permit symlink creation")
            with self.assertRaisesRegex(m.ForensicsError, "LINKED_INPUT"):
                m.inspect_file(link, m.digest(fixture()))

    def test_plan_does_not_read_blobs(self):
        with patch.object(m, "inspect_file") as read, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(m.main([]), 0)
        read.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["state"], "PLAN_ONLY_NO_BLOB_READ")

    def test_wrong_host_refused_before_blobs(self):
        with patch.dict(os.environ, {"COMPUTERNAME": "OMEN"}), patch.object(m, "inspect_file") as read:
            with self.assertRaisesRegex(m.ForensicsError, "BETTERWITHAGE"):
                m.main(["--inspect-blobs"])
        read.assert_not_called()

    def test_no_execution_or_network_dependencies(self):
        import ast
        tree = ast.parse(Path(m.__file__).read_text())
        imports = {node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        imports |= {alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertFalse(imports & {"subprocess", "socket", "urllib", "requests", "torch", "numpy", "pickle", "huggingface_hub"})
        self.assertEqual(set(m.BLOBS), {"szl1:latest", "szl-sovereign-qwen:latest"})


if __name__ == "__main__":
    unittest.main()
