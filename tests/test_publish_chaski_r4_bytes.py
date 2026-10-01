"""Offline contracts for the chaski-r4 bytes publisher: digests and fail-closed preflight."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("publish_chaski_r4_bytes", ROOT / "tools" / "publish_chaski_r4_bytes.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_dir_digest_matches_bakeoff_named_n_algorithm(tmp_path: Path) -> None:
    (tmp_path / "b.safetensors").write_bytes(b"BBB")
    (tmp_path / "a.safetensors").write_bytes(b"AAA")
    (tmp_path / "ignored.json").write_bytes(b"{}")
    expected = hashlib.sha256()
    for name, payload in (("a.safetensors", b"AAA"), ("b.safetensors", b"BBB")):
        expected.update(name.encode("utf-8"))
        expected.update(b"\0")
        expected.update(payload)
    assert MODULE.dir_digest(tmp_path) == expected.hexdigest()
    sys.path.insert(0, str(ROOT / "chaski"))
    import bakeoff_named_n  # noqa: E402

    assert bakeoff_named_n.sha256_adapter(tmp_path) == MODULE.dir_digest(tmp_path)


def test_dir_digest_refuses_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no safetensors"):
        MODULE.dir_digest(tmp_path)


def test_expected_identities_are_the_receipted_ones() -> None:
    assert MODULE.EXPECT["adapter_model.safetensors"] == "f1a2cdc313795775966280bc8648367005700327dd28010da8d2f88d2a5e2a02"
    assert MODULE.EXPECT_DIR_DIGEST == "e1abc37a5c41a82b0fc2cd98ccd6edbb2a08fceb8ca9e883bcfb76b861c221cf"
    assert MODULE.EXPECT["evidence/canonical_rerun_20261001_140615.receipt.json"] == "5ae3de970014726f190dfe8e4d5b35e667fd737b1be29e27205c25d143bdf403"
    assert MODULE.REPO == "SZLHOLDINGS/chaski-r4"


def test_preflight_stops_before_any_hub_call_on_wrong_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"not the receipted bytes")
    (adapter / "adapter_config.json").write_bytes(b"{}")
    staging = tmp_path / "chaski_r4"
    (staging / "evidence").mkdir(parents=True)
    (staging / "card").mkdir()
    (staging / "training_receipt.json").write_bytes(b"{}")
    (staging / "evidence" / "canonical_rerun_20261001_140615.receipt.json").write_bytes(b"{}")
    (staging / "QUALIFICATION_STATE.md").write_bytes(b"# x")
    (staging / "card" / "README.md").write_bytes(b"---\n")
    license_path = tmp_path / "LICENSE"
    license_path.write_text("Apache License Version 2.0", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["x", str(adapter), str(staging), str(license_path), str(tmp_path / "receipt.json")])
    with pytest.raises(SystemExit, match="PREFLIGHT: adapter_model.safetensors sha256"):
        MODULE.main()
    assert not (tmp_path / "receipt.json").exists()
