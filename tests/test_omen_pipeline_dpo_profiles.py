"""DPO training profiles of tools/szl_omen_pipeline.py are data, receipted, and override-honest.

Loads the module without importing torch: the profile resolver is pure Python, and the test asserts
the collapse-resistant C3 profile is the default while C2 (the configuration behind the 2026-10-01
collapsed bundle) stays reproducible on request.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("szl_omen_pipeline_profiles_test", ROOT / "tools" / "szl_omen_pipeline.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_c3_is_the_default_and_differs_from_the_collapsed_c2_on_every_axis():
    assert MODULE.DEFAULT_DPO_PROFILE == "C3"
    c2, c3 = MODULE.DPO_PROFILES["C2"], MODULE.DPO_PROFILES["C3"]
    assert c2["rpo_alpha"] is None and c3["rpo_alpha"] == 1.0
    assert c3["learning_rate"] < c2["learning_rate"]
    assert c3["beta"] > c2["beta"]
    assert c3["epochs"] < c2["epochs"]
    assert "20261001-a989dd523998" in c2["note"]


def test_resolve_returns_profile_values_with_no_overrides():
    settings = MODULE.resolve_dpo_profile("C3")
    assert settings["profile"] == "C3"
    assert settings["overrides"] == {}
    assert (settings["learning_rate"], settings["beta"], settings["rpo_alpha"], settings["epochs"]) == (1e-5, 0.3, 1.0, 1)


def test_overrides_are_recorded_not_hidden():
    settings = MODULE.resolve_dpo_profile("C3", epochs=2, beta=0.3, learning_rate=2e-5)
    assert settings["epochs"] == 2 and settings["learning_rate"] == 2e-5
    assert settings["overrides"] == {
        "epochs": {"profile": 1, "override": 2},
        "learning_rate": {"profile": 1e-5, "override": 2e-5},
    }
    assert "beta" not in settings["overrides"]  # equal to the profile value is not an override


def test_c2_reproduces_the_historical_configuration():
    settings = MODULE.resolve_dpo_profile("C2")
    assert (settings["learning_rate"], settings["beta"], settings["rpo_alpha"], settings["epochs"]) == (5e-5, 0.1, None, 2)


def test_unknown_profile_fails_closed():
    with pytest.raises(ValueError):
        MODULE.resolve_dpo_profile("C9")


def test_cli_exposes_profile_and_override_flags():
    source = (ROOT / "tools" / "szl_omen_pipeline.py").read_text(encoding="utf-8")
    for flag in ("--profile", "--lr", "--beta", "--rpo-alpha"):
        assert flag in source
    assert 'cfg_kwargs["rpo_alpha"]' in source
