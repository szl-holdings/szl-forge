"""Dependency and source-publication admission regressions from hosted integration."""
from pathlib import Path
import tomllib
import pytest
from packaging.requirements import Requirement
from szl_model_lab.blueprints import publish_context


@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
def test_admitted_main_events_are_explicit(event):
    publish_context({"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "szl-holdings/szl-forge",
                     "GITHUB_REF": "refs/heads/main", "GITHUB_SHA": "a" * 40,
                     "GITHUB_EVENT_NAME": event}, "a" * 40)


def test_safetensors_runtime_dependency_declared():
    config = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    names = {Requirement(item).name for item in config["project"]["dependencies"]}
    assert "numpy" in names and "safetensors" in names
