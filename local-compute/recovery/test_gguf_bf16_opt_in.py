# SPDX-License-Identifier: Apache-2.0
"""Opt-in BF16 layout regressions (forensic v1.3.0). Synthetic bytes only.

The fixtures below are assembled here, independently of the other recovery test
modules and of the parser. Every BF16 payload is a hand-written hex literal whose
decoded values are cross-checked by two independent formulas before any parser
call: bf16 -> f32 is a 16-bit left shift, and an exactly representable f32
truncates to its upper two little-endian bytes. None of these bytes is, or was
derived from, an owner model file.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import gguf_forensics as m

# Hand-written little-endian BF16 elements and the f32 values they encode.
#   0x3F80 = 1.0, 0xC000 = -2.0, 0x3F00 = 0.5, 0x4040 = 3.0, 0x3FC0 = 1.5,
#   0x4049 = 3.140625 (BF16 is the upper half of the IEEE-754 binary32 pattern).
BF16_BYTES = bytes.fromhex("803f" "00c0" "003f" "4040" "c03f" "4940")
BF16_VALUES = (1.0, -2.0, 0.5, 3.0, 1.5, 3.140625)
F32_BYTES = bytes.fromhex("0000803f" "000000c0")      # 1.0, -2.0 as binary32
F16_BYTES = bytes.fromhex("003c" "00c0" "0038" "0042")  # 1.0, -2.0, 0.5, 3.0 as binary16

# GGML type IDs and widths from ggml-org/ggml@7840aaba (include/ggml.h enum
# ggml_type; src/ggml.c type_traits: blck_size 1, type_size sizeof(ggml_bf16_t)).
F32, F16, BF16 = 0, 1, 30

# Canonical digests produced by the UNMODIFIED parent v1.2.1 source
# (main c9a03e46, gguf_forensics.py blob 40dbbc84) for the fixtures below.
# They pin the default F32/F16 result and the default type-30 rejection record.
PARENT_DEFAULT_F32_F16_RESULT = "b4569bbdf373923c8ec9e9bead3716a1355c2e827a91e6f165046112da55eb44"
PARENT_DEFAULT_BF16_REJECTION = "259c6fca6691f4152c9562dc5beab9cc5f3f9b78be75121f6deb172c93e7bac8"


def gguf_string(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def build(tensors, *, alignment=32, tail=b""):
    """Assemble a GGUF v3 file. tensors: (name, shape, ggml_type, offset, payload).

    Offsets are given explicitly so layouts are fixed by the test, not computed
    from the parser's width table. Returns (raw bytes, type-field offsets).
    """
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1))
    header += gguf_string("general.architecture") + struct.pack("<I", 8) + gguf_string("qwen2")
    type_fields = []
    for name, shape, kind, offset, _ in tensors:
        header += gguf_string(name) + struct.pack("<I", len(shape))
        header += b"".join(struct.pack("<Q", dim) for dim in shape)
        type_fields.append(len(header))
        header += struct.pack("<IQ", kind, offset)
    header += b"\x00" * ((-len(header)) % alignment)
    body = bytearray()
    for _, _, _, offset, payload in sorted(tensors, key=lambda item: item[3]):
        body += b"\x00" * (offset - len(body))
        body += payload
    return bytes(header + body) + tail, type_fields


def bf16_only():
    return build([("blk.0.attn_q.weight", [2, 3], BF16, 0, BF16_BYTES)])


def mixed():
    # F32 [2] = 8 bytes, F16 [4] = 8 bytes, BF16 [2, 3] = 12 bytes ending at EOF.
    return build([("output_norm.weight", [2], F32, 0, F32_BYTES),
                  ("blk.0.ffn_up.weight", [4], F16, 32, F16_BYTES),
                  ("blk.0.attn_q.weight", [2, 3], BF16, 64, BF16_BYTES)])


def f32_f16_only():
    return build([("output_norm.weight", [2], F32, 0, F32_BYTES),
                  ("blk.0.ffn_up.weight", [4], F16, 32, F16_BYTES)])


def parse(raw, **kw):
    return m.fingerprint_stream(io.BytesIO(raw), len(raw), **kw)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def by_name(result):
    return {t["name_sha256"]: t for t in result["tensors"]}


def name_key(name):
    return sha(name.encode())


class FixtureIndependenceTests(unittest.TestCase):
    """The fixture itself is checked before it is used to judge the parser."""

    def test_bf16_bytes_decode_by_sixteen_bit_left_shift(self):
        halves = struct.unpack("<6H", BF16_BYTES)
        decoded = tuple(struct.unpack("<f", struct.pack("<I", h << 16))[0] for h in halves)
        self.assertEqual(decoded, BF16_VALUES)

    def test_bf16_bytes_equal_truncated_binary32_upper_half(self):
        for index, value in enumerate(BF16_VALUES):
            binary32 = struct.pack("<f", value)
            with self.subTest(value=value):
                # Exactly representable: the dropped low half is zero, so no rounding.
                self.assertEqual(binary32[:2], b"\x00\x00")
                self.assertEqual(binary32[2:], BF16_BYTES[2 * index:2 * index + 2])

    def test_bf16_bytes_are_not_f16_encodings_of_the_same_values(self):
        as_f16 = struct.unpack("<6e", BF16_BYTES)
        self.assertNotEqual(as_f16, BF16_VALUES)
        self.assertEqual(struct.unpack("<4e", F16_BYTES), BF16_VALUES[:4])
        self.assertEqual(struct.unpack("<2f", F32_BYTES), BF16_VALUES[:2])


class DefaultBehaviourUnchangedTests(unittest.TestCase):
    def test_default_allowlist_is_still_f32_f16(self):
        self.assertEqual(m.TENSOR_TYPES, {0: ("F32", 4), 1: ("F16", 2)})
        self.assertNotIn(BF16, m.TENSOR_TYPES)
        self.assertEqual(m.OPT_IN_BF16, {30: ("BF16", 2)})

    def test_default_f32_f16_result_matches_parent_bytes(self):
        raw, _ = f32_f16_only()
        result = parse(raw)
        self.assertEqual(m.digest(m.canonical(result)), PARENT_DEFAULT_F32_F16_RESULT)
        self.assertNotIn("opt_in_tensor_types", result)

    def test_default_rejects_bf16_exactly_as_parent(self):
        raw, fields = bf16_only()
        with self.assertRaises(m.UnsupportedTensorType) as raised:
            parse(raw)
        error = raised.exception
        self.assertEqual(str(error), "UNSUPPORTED_TENSOR_TYPE:ggml_type_id=30")
        self.assertEqual(error.observation["type_field_byte_offset"], fields[0])
        self.assertEqual(error.observation["supported_ggml_type_ids"], [0, 1])
        self.assertEqual(m.digest(m.canonical(m.rejected_model(error))), PARENT_DEFAULT_BF16_REJECTION)

    def test_default_rejects_mixed_file_at_bf16_descriptor_without_payload_read(self):
        raw, fields = mixed()

        class Bounded(io.BytesIO):
            def read(self, n=-1):
                if self.tell() + n > fields[2] + 4:
                    raise AssertionError("read past rejected type field")
                return super().read(n)

        stream = Bounded(raw)
        with self.assertRaises(m.UnsupportedTensorType) as raised:
            m.fingerprint_stream(stream, len(raw))
        self.assertEqual(raised.exception.observation["tensor_index_zero_based"], 2)
        self.assertEqual(raised.exception.observation["ggml_type_id"], 30)
        self.assertEqual(stream.tell(), fields[2] + 4)

    def test_default_unknown_types_still_reject(self):
        for kind in (2, 12, 24, 28, 29, 31, 42, 0xffffffff):
            raw, _ = build([("t", [2], kind, 0, b"\x00" * 8)])
            with self.subTest(kind=kind), self.assertRaises(m.UnsupportedTensorType) as raised:
                parse(raw)
            self.assertEqual(raised.exception.observation["supported_ggml_type_ids"], [0, 1])


class OptInBf16Tests(unittest.TestCase):
    def test_opt_in_admits_bf16_and_hashes_exact_payload(self):
        raw, _ = bf16_only()
        result = parse(raw, allow_bf16=True)
        (tensor,) = result["tensors"]
        self.assertEqual(tensor["type"], "BF16")
        self.assertEqual(tensor["shape"], [2, 3])
        self.assertEqual(tensor["bytes"], len(BF16_BYTES))
        self.assertEqual(tensor["data_sha256"], sha(BF16_BYTES))
        self.assertEqual(result["container_sha256"], sha(raw))
        self.assertEqual(result["bytes"], len(raw))
        self.assertEqual(result["opt_in_tensor_types"], ["BF16"])
        self.assertFalse(result["tensor_values_or_health_checked"])

    def test_opt_in_mixed_format_layout(self):
        raw, _ = mixed()
        result = parse(raw, allow_bf16=True)
        tensors = by_name(result)
        expected = {"output_norm.weight": ("F32", F32_BYTES),
                    "blk.0.ffn_up.weight": ("F16", F16_BYTES),
                    "blk.0.attn_q.weight": ("BF16", BF16_BYTES)}
        self.assertEqual(set(tensors), {name_key(n) for n in expected})
        for name, (label, payload) in expected.items():
            with self.subTest(name=name):
                tensor = tensors[name_key(name)]
                self.assertEqual(tensor["type"], label)
                self.assertEqual(tensor["bytes"], len(payload))
                self.assertEqual(tensor["data_sha256"], sha(payload))
        self.assertEqual(result["tensor_bytes"], 8 + 8 + 12)
        self.assertEqual(result["container_sha256"], sha(raw))

    def test_bf16_width_is_two_bytes(self):
        # 16 BF16 elements fill exactly one 32-byte aligned slot before the next tensor.
        sixteen = BF16_BYTES * 2 + BF16_BYTES[:8]
        self.assertEqual(len(sixteen), 32)
        raw, _ = build([("a", [16], BF16, 0, sixteen), ("b", [2], F32, 32, F32_BYTES)])
        self.assertEqual(parse(raw, allow_bf16=True)["tensor_bytes"], 40)
        # A 17-element BF16 extent (34 bytes) must collide with the tensor at 32.
        raw, _ = build([("a", [17], BF16, 0, sixteen), ("b", [2], F32, 32, F32_BYTES)])
        with self.assertRaisesRegex(m.ForensicsError, "OVERLAPPING_TENSORS"):
            parse(raw, allow_bf16=True)
        # A BF16 extent ending one byte past EOF is truncation, not a guess.
        raw, _ = bf16_only()
        with self.assertRaisesRegex(m.ForensicsError, "TRUNCATED_TENSOR"):
            m.fingerprint_stream(io.BytesIO(raw[:-1]), len(raw) - 1, allow_bf16=True)

    def test_trailing_bytes_participate_in_container_not_tensor(self):
        raw, _ = build([("blk.0.attn_q.weight", [2, 3], BF16, 0, BF16_BYTES)], tail=b"Z" * 5)
        result = parse(raw, allow_bf16=True)
        self.assertEqual(result["container_sha256"], sha(raw))
        self.assertEqual(result["tensors"][0]["data_sha256"], sha(BF16_BYTES))

    def test_every_truncation_point_is_refused(self):
        raw, _ = mixed()
        for size in range(len(raw)):
            with self.subTest(size=size), self.assertRaises(m.ForensicsError):
                m.fingerprint_stream(io.BytesIO(raw[:size]), size, allow_bf16=True)

    def test_opt_in_still_rejects_unknown_types(self):
        for kind in (2, 12, 24, 28, 29, 31, 42, 0xffffffff):
            raw, fields = build([("ok", [2, 3], BF16, 0, BF16_BYTES),
                                 ("t", [2], kind, 32, b"\x00" * 8)])
            with self.subTest(kind=kind), self.assertRaises(m.UnsupportedTensorType) as raised:
                parse(raw, allow_bf16=True)
            observation = raised.exception.observation
            self.assertEqual(observation["ggml_type_id"], kind)
            self.assertEqual(observation["tensor_index_zero_based"], 1)
            self.assertEqual(observation["type_field_byte_offset"], fields[1])
            self.assertEqual(observation["supported_ggml_type_ids"], [0, 1, 30])
            self.assertEqual(observation["scope"], "PARTIAL_HEADER_NOT_FULL_FILE_AUTHENTICATED")
            self.assertEqual(observation["historical_blob_sha256_verification"], "NOT_COMPLETED")

    def test_opt_in_flag_must_be_a_real_boolean(self):
        raw, _ = bf16_only()
        for flag in (1, 0, "true", "", None, 30, [True]):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(m.ForensicsError, "OPT_IN_FLAG_BOUND"):
                    parse(raw, allow_bf16=flag)
                # Refused before any filesystem access: the path does not exist.
                with self.assertRaisesRegex(m.ForensicsError, "OPT_IN_FLAG_BOUND"):
                    m.inspect_file(Path("does-not-exist.gguf"), "0" * 64, allow_bf16=flag)

    def test_opt_in_does_not_change_f32_f16_fingerprints(self):
        raw, _ = f32_f16_only()
        default, opted = parse(raw), parse(raw, allow_bf16=True)
        for key in ("container_sha256", "tensor_set_sha256", "tensors", "tensor_bytes",
                    "metadata_fingerprints", "tensor_count"):
            self.assertEqual(default[key], opted[key])
        changed = {k for k in default.keys() | opted.keys() if default.get(k) != opted.get(k)}
        self.assertEqual(changed, {"opt_in_tensor_types", "boundary"})
        self.assertIn("BF16", opted["boundary"])

    def test_tensor_type_is_bound_into_identity(self):
        # Same name, shape and payload bytes; only the declared type differs.
        as_f16, _ = build([("w", [2], F16, 0, BF16_BYTES[:4])])
        as_bf16, _ = build([("w", [2], BF16, 0, BF16_BYTES[:4])])
        result = m.compare(parse(as_f16, allow_bf16=True), parse(as_bf16, allow_bf16=True))
        self.assertFalse(result["same_tensor_set_bytes"])
        self.assertEqual(result["different_shared_tensors"], 1)
        self.assertFalse(result["root_cause_proven"])
        self.assertFalse(result["model_qualified"])


class OptInFileAndCliTests(unittest.TestCase):
    def test_opt_in_keeps_full_file_authentication_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.gguf"
            raw, _ = mixed()
            path.write_bytes(raw)
            with self.assertRaisesRegex(m.ForensicsError, "BLOB_DIGEST_MISMATCH"):
                m.inspect_file(path, "0" * 64, allow_bf16=True)
            result = m.inspect_file(path, sha(raw), allow_bf16=True)
            self.assertTrue(result["matches_historical_from_digest"])
            # Default inspection of the same file is a rejection, never authentication.
            with self.assertRaises(m.UnsupportedTensorType) as raised:
                m.inspect_file(path, sha(raw))
            record = m.rejected_model(raised.exception)
            self.assertNotIn("matches_historical_from_digest", record)
            self.assertEqual(record["tensor_type_observation"]["historical_blob_sha256_verification"],
                             "NOT_COMPLETED")
            self.assertEqual(path.read_bytes(), raw)

    def run_main(self, argv):
        calls = []

        def fake_inspect(path, expected, *args, **kwargs):
            calls.append(kwargs)
            raise m.UnsupportedTensorType(30, 0, 100)

        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            (home / ".ollama/models/blobs").mkdir(parents=True)
            with patch.object(m.os, "name", "nt"), \
                    patch.dict(os.environ, {"COMPUTERNAME": "BETTERWITHAGE"}), \
                    patch.object(m, "Path") as paths, \
                    patch.object(m, "require_local_windows_path"), \
                    patch.object(m, "inspect_file", side_effect=fake_inspect), \
                    patch.object(m, "compare") as compare, \
                    contextlib.redirect_stdout(io.StringIO()):
                paths.side_effect = type(home)
                paths.home.return_value = home
                code = m.main(argv)
            compare.assert_not_called()
            directory = next(home.glob("szl-gguf-forensics-*"))
            report = json.loads((directory / "gguf-forensics.json").read_text())
            first = json.loads((directory / "checkpoint-000.json").read_text())
        return code, calls, report, first

    def test_cli_default_does_not_opt_in(self):
        code, calls, report, first = self.run_main(["--inspect-blobs"])
        self.assertEqual(code, 1)
        # The default read is the parent's exact call: no opt-in keyword at all.
        self.assertEqual(calls, [{}, {}])
        self.assertEqual(report["opt_in_tensor_types"], [])
        self.assertEqual(first["report"]["opt_in_tensor_types"], [])
        self.assertEqual(report["version"], m.VERSION)

    def test_cli_flag_threads_explicit_opt_in_and_records_it(self):
        code, calls, report, first = self.run_main(["--inspect-blobs", "--allow-bf16"])
        self.assertEqual(code, 1)
        self.assertEqual(calls, [{"allow_bf16": True}, {"allow_bf16": True}])
        self.assertEqual(report["opt_in_tensor_types"], ["BF16"])
        self.assertEqual(first["report"]["opt_in_tensor_types"], ["BF16"])
        self.assertEqual(report["state"], "INCOMPLETE")
        self.assertIsNone(report["comparison"])
        for flag in ("inference", "training", "model_weights_changed",
                     "paired_attempts_touched", "publication_eligible"):
            self.assertIs(report[flag], False)

    def test_plan_only_reads_nothing_and_default_plan_is_unchanged(self):
        for argv, extra in (([], {}), (["--allow-bf16"], {"opt_in_tensor_types": ["BF16"]})):
            with self.subTest(argv=argv), patch.object(m, "inspect_file") as read, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(m.main(argv), 0)
            read.assert_not_called()
            self.assertEqual(json.loads(output.getvalue()),
                             {"state": "PLAN_ONLY_NO_BLOB_READ", "targets": list(m.BLOBS),
                              "inference": False, "training": False, **extra})

    def test_flag_does_not_bypass_host_gate(self):
        with patch.dict(os.environ, {"COMPUTERNAME": "OMEN"}), patch.object(m, "inspect_file") as read:
            with self.assertRaisesRegex(m.ForensicsError, "BETTERWITHAGE"):
                m.main(["--inspect-blobs", "--allow-bf16"])
        read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
