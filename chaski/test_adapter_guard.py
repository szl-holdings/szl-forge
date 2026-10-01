"""Pure-Python tests for chaski/adapter_guard.py (no torch required)."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

CHASKI = Path(__file__).resolve().parent
sys.path.insert(0, str(CHASKI))
import adapter_guard as guard

MM = "base_model.model.model.language_model.layers.{i}.mlp.down_proj.lora_{ab}.weight"
TXT = "base_model.model.model.layers.{i}.mlp.down_proj.lora_{ab}.weight"


def write_safetensors_header(adapter_dir: Path, keys: list[str]) -> None:
    header = {k: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for k in keys}
    header["__metadata__"] = {"format": "pt"}
    body = json.dumps(header).encode("utf-8")
    (adapter_dir / guard.ADAPTER_WEIGHTS).write_bytes(struct.pack("<Q", len(body)) + body + b"\0" * 4)


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


if __name__ == "__main__":
    unittest.main()


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
