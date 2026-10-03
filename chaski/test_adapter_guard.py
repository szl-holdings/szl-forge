"""Pure-Python tests for chaski/adapter_guard.py (no torch required)."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

CHASKI = Path(__file__).resolve().parent
sys.path.insert(0, str(CHASKI))
import adapter_guard as guard

MM = "base_model.model.model.language_model.layers.{i}.mlp.down_proj.lora_{ab}.weight"
TXT = "base_model.model.model.layers.{i}.mlp.down_proj.lora_{ab}.weight"


def write_safetensors_header(adapter_dir: Path, keys: list[str]) -> None:
    header = {
        k: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * (i + 1)]}
        for i, k in enumerate(keys)
    }
    header["__metadata__"] = {"format": "pt"}
    body = json.dumps(header).encode("utf-8")
    (adapter_dir / guard.ADAPTER_WEIGHTS).write_bytes(
        struct.pack("<Q", len(body)) + body + b"\0" * (4 * len(keys))
    )


class CheckpointHeaderTests(unittest.TestCase):
    """Malformed headers fail closed before an unbounded read or key admission."""

    def parse(self, header: bytes | dict, payload: bytes = b"\0" * 4) -> list[str]:
        body = json.dumps(header).encode("utf-8") if isinstance(header, dict) else header
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / guard.ADAPTER_WEIGHTS).write_bytes(
                struct.pack("<Q", len(body)) + body + payload
            )
            return guard.checkpoint_keys(Path(tmp))

    @staticmethod
    def tensor(**changes):
        return {"dtype": "F32", "shape": [1], "data_offsets": [0, 4], **changes}

    def test_short_prefix_fails_with_normalized_error(self) -> None:
        for size in range(8):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / guard.ADAPTER_WEIGHTS).write_bytes(b"\0" * size)
                with self.assertRaisesRegex(guard.AdapterNotApplied, "INVALID_ADAPTER_CHECKPOINT:"):
                    guard.checkpoint_keys(Path(tmp))

    def test_declared_length_is_bounded_before_header_read(self) -> None:
        for length in (0, 1, guard.MAX_ADAPTER_HEADER_BYTES + 1, (1 << 64) - 1, 100):
            with self.subTest(length=length), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / guard.ADAPTER_WEIGHTS
                path.write_bytes(struct.pack("<Q", length) + b"{}")
                with path.open("rb") as handle, patch.object(Path, "open") as opened:
                    reader = opened.return_value.__enter__.return_value
                    reader.fileno.return_value = handle.fileno()
                    reader.read.side_effect = handle.read
                    with self.assertRaises(guard.InvalidAdapterCheckpoint):
                        guard.checkpoint_keys(Path(tmp))
                    self.assertEqual(reader.read.call_args_list, [call(8)])

    def test_valid_read_budget_excludes_tensor_payload(self) -> None:
        body = json.dumps({"t": self.tensor()}).encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / guard.ADAPTER_WEIGHTS
            path.write_bytes(struct.pack("<Q", len(body)) + body + b"\0" * 4)
            with path.open("rb") as handle, patch.object(Path, "open") as opened:
                reader = opened.return_value.__enter__.return_value
                reader.fileno.return_value = handle.fileno()
                reader.read.side_effect = handle.read
                self.assertEqual(guard.checkpoint_keys(Path(tmp)), ["t"])
                self.assertEqual(reader.read.call_args_list, [call(8), call(len(body))])

    def test_short_header_read_after_size_check_fails(self) -> None:
        body = json.dumps({"t": self.tensor()}).encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / guard.ADAPTER_WEIGHTS
            path.write_bytes(struct.pack("<Q", len(body)) + body + b"\0" * 4)
            with path.open("rb") as handle, patch.object(Path, "open") as opened:
                reader = opened.return_value.__enter__.return_value
                reader.fileno.return_value = handle.fileno()
                reader.read.side_effect = [struct.pack("<Q", len(body)), body[:-1]]
                with self.assertRaises(guard.InvalidAdapterCheckpoint):
                    guard.checkpoint_keys(Path(tmp))

    def test_scalar_empty_tensor_padding_and_empty_checkpoint(self) -> None:
        self.assertEqual(self.parse({"t": self.tensor(shape=[])}), ["t"])
        self.assertEqual(self.parse({"t": self.tensor(shape=[0, 9], data_offsets=[0, 0])}, b""), ["t"])
        self.assertEqual(self.parse(b'{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}   '), ["t"])
        self.assertEqual(self.parse({}, b""), [])

    def test_header_order_does_not_determine_payload_order(self) -> None:
        header = {"z": self.tensor(data_offsets=[4, 8]), "a": self.tensor()}
        self.assertEqual(self.parse(header, b"\0" * 8), ["a", "z"])

    def test_invalid_encoding_root_json_and_depth_fail_closed(self) -> None:
        headers = [b"\xff\xff", b"[]", b"null", b' {"t":{}}', b'{"t":',
                   b'{"t":NaN}', b'{"t":Infinity}', b'{"t":-Infinity}',
                   b'{"t":' + b"[" * 1100 + b"]" * 1100 + b"}"]
        for body in headers:
            with self.subTest(body=body[:20]), self.assertRaises(guard.InvalidAdapterCheckpoint):
                self.parse(body)

    def test_duplicate_keys_at_each_level_are_rejected(self) -> None:
        headers = [b'{"t":{},"t":{}}',
                   b'{"t":{"dtype":"F32","dtype":"F32","shape":[1],"data_offsets":[0,4]}}',
                   b'{"__metadata__":{"format":"pt","format":"pt"}}']
        for body in headers:
            with self.subTest(body=body), self.assertRaises(guard.InvalidAdapterCheckpoint):
                self.parse(body)

    def test_bad_metadata_and_tensor_names_fail_closed(self) -> None:
        for metadata in (None, [], {"format": 1}, {"format": "\ud800"}):
            with self.subTest(metadata=metadata), self.assertRaises(guard.InvalidAdapterCheckpoint):
                self.parse({"__metadata__": metadata, "t": self.tensor()})
        for name in ("", "\ud800"):
            with self.subTest(name=name), self.assertRaises(guard.InvalidAdapterCheckpoint):
                self.parse({name: self.tensor()})

    def test_invalid_descriptors_fail_closed(self) -> None:
        descriptors = [None, [], {}, self.tensor(extra=True),
                       {"dtype": "F32", "shape": [1]}, self.tensor(dtype="UNKNOWN"),
                       self.tensor(dtype=[])]
        for descriptor in descriptors:
            with self.subTest(descriptor=descriptor), self.assertRaises(guard.InvalidAdapterCheckpoint):
                self.parse({"t": descriptor})

    def test_invalid_shapes_and_offsets_fail_closed(self) -> None:
        for shape in (None, "1", [True], [-1], [1.0], [10**1000, 10**1000]):
            with self.subTest(shape=shape), self.assertRaises(guard.InvalidAdapterCheckpoint):
                self.parse({"t": self.tensor(shape=shape)})
        for offsets in (None, [0], [0, 4, 4], [False, 4], [-1, 3], [4, 0], [0, 5], [0.0, 4]):
            with self.subTest(offsets=offsets), self.assertRaises(guard.InvalidAdapterCheckpoint):
                self.parse({"t": self.tensor(data_offsets=offsets)})

    def test_shape_byte_size_holes_overlaps_and_tail_fail_closed(self) -> None:
        cases = [({"t": self.tensor(shape=[2])}, 4),
                 ({"t": self.tensor(data_offsets=[1, 5])}, 5),
                 ({"a": self.tensor(), "b": self.tensor()}, 4),
                 ({"t": self.tensor()}, 5), ({}, 1)]
        for header, size in cases:
            with self.subTest(header=header), self.assertRaises(guard.InvalidAdapterCheckpoint):
                self.parse(header, b"\0" * size)

    def test_empty_tensor_cannot_hide_unrepresentable_shape_or_prior_overflow(self) -> None:
        for shape in ([0, 2**64], [2**64, 0], [0, 10**1000],
                      [10**1000, 0], [2**63, 3, 0]):
            with self.subTest(shape=shape), self.assertRaises(guard.InvalidAdapterCheckpoint):
                self.parse({"t": self.tensor(shape=shape, data_offsets=[0, 0])}, b"")
        self.assertEqual(
            self.parse({"t": self.tensor(shape=[0, 2**63, 3], data_offsets=[0, 0])}, b""),
            ["t"],
        )

    def test_safetensors_v080_dtype_widths_and_packed_alignment(self) -> None:
        for dtype, bits in guard.DTYPE_BITS.items():
            count = 4 if bits == 6 else 2 if bits == 4 else 1
            size = count * bits // 8
            with self.subTest(dtype=dtype):
                header = {"t": self.tensor(dtype=dtype, shape=[count], data_offsets=[0, size])}
                self.assertEqual(self.parse(header, b"\0" * size), ["t"])
        for dtype in ("F4", "F6_E2M3", "F6_E3M2"):
            with self.subTest(dtype=dtype), self.assertRaises(guard.InvalidAdapterCheckpoint):
                self.parse({"t": self.tensor(dtype=dtype)})

    def test_malformed_checkpoint_does_not_query_loaded_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / guard.ADAPTER_WEIGHTS).write_bytes(b"bad")
            model = Mock()
            with self.assertRaises(guard.AdapterNotApplied):
                guard.assert_adapter_applied(model, Path(tmp))
            model.named_parameters.assert_not_called()


class _FakeParam:
    pass


class _FakePeftModel:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def named_parameters(self):
        return [(n, _FakeParam()) for n in self._names]


class AdapterGuardTests(unittest.TestCase):
    def test_header_parse_returns_sorted_tensor_names_without_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            keys = [MM.format(i=i, ab=ab) for i in (1, 0) for ab in ("B", "A")]
            write_safetensors_header(Path(tmp), keys)
            self.assertEqual(guard.checkpoint_keys(Path(tmp)), sorted(keys))

    def test_live_keys_strip_the_adapter_name_segment(self) -> None:
        model = _FakePeftModel(
            [
                "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.default.weight",
                "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.default.weight",
                "base_model.model.model.language_model.layers.0.mlp.down_proj.base_layer.weight",
            ]
        )
        self.assertEqual(
            guard.live_lora_keys(model),
            {MM.format(i=0, ab="A"), MM.format(i=0, ab="B")},
        )

    def test_full_coverage_passes_and_reports_layout(self) -> None:
        keys = [MM.format(i=i, ab=ab) for i in range(3) for ab in ("A", "B")]
        report = guard.coverage_report(keys, set(keys))
        self.assertTrue(report["fully_applied"])
        self.assertEqual((report["applied"], report["unapplied"]), (6, 0))
        self.assertEqual(report["checkpoint_layout"], "language_model")

    def test_layout_mismatch_is_zero_coverage_and_raises(self) -> None:
        ckpt = [MM.format(i=i, ab=ab) for i in range(3) for ab in ("A", "B")]
        live = {TXT.format(i=i, ab=ab) for i in range(3) for ab in ("A", "B")}
        report = guard.coverage_report(ckpt, live)
        self.assertFalse(report["fully_applied"])
        self.assertEqual(report["applied"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            write_safetensors_header(Path(tmp), ckpt)
            model = _FakePeftModel([k.replace(".weight", ".default.weight") for k in sorted(live)])
            with self.assertRaises(guard.AdapterNotApplied) as ctx:
                guard.assert_adapter_applied(model, Path(tmp))
            self.assertIn("ADAPTER_NOT_APPLIED: 6/6", str(ctx.exception))

    def test_partial_coverage_also_fails_closed(self) -> None:
        ckpt = [MM.format(i=i, ab=ab) for i in range(2) for ab in ("A", "B")]
        report = guard.coverage_report(ckpt, set(ckpt[:-1]))
        self.assertFalse(report["fully_applied"])
        self.assertEqual(report["unapplied_sample"], [ckpt[-1]])

    def test_empty_checkpoint_never_counts_as_applied(self) -> None:
        self.assertFalse(guard.coverage_report([], set())["fully_applied"])


class R4CandidateTests(unittest.TestCase):
    """The optional --chaski-r4-adapter candidate is additive and never records an owner path."""

    def test_r4_adapter_path_is_publicized_repo_relative(self) -> None:
        import bakeoff_named_n as bakeoff

        self.assertEqual(
            bakeoff.publicize_runtime(r"C:\Users\owner\szl-forge\chaski_r4\chaski-r4-adapter"),
            "chaski_r4/chaski-r4-adapter",
        )

    def test_r4_candidate_is_appended_only_when_requested(self) -> None:
        import argparse

        import bakeoff_named_n as bakeoff

        base = argparse.Namespace(base_model=None, chaski_5050_adapter=None, chaski_r2_adapter=None, chaski_r4_adapter=None)
        ids = [spec["id"] for spec in bakeoff.candidate_specs(base)]
        self.assertEqual(ids, ["base-qwen35-0.8b", "chaski-5050", "chaski-r2"])
        with tempfile.TemporaryDirectory() as tmp:
            r4 = Path(tmp) / "chaski-r4-adapter"
            r4.mkdir()
            (r4 / "adapter_config.json").write_text("{}", encoding="utf-8")
            (r4 / "adapter_model.safetensors").write_bytes(b"")
            withr4 = argparse.Namespace(base_model=None, chaski_5050_adapter=None, chaski_r2_adapter=None, chaski_r4_adapter=str(r4))
            specs = bakeoff.candidate_specs(withr4)
            self.assertEqual([s["id"] for s in specs][-1], "chaski-r4-local")
            self.assertTrue(specs[-1]["local_only"])
            self.assertTrue(specs[-1]["hub_id_declared_only"])
            self.assertEqual(bakeoff.publicize_runtime(specs[-1]["adapter"]), "chaski_r4/chaski-r4-adapter")


if __name__ == "__main__":
    unittest.main()
