"""Source assertions complement hosted CI; they do not establish hosted success."""
from pathlib import Path
import re
import yaml


def workflow():
    path = Path(__file__).resolve().parents[2] / ".github/workflows/model-lab-contract.yml"
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)


def test_workflow_prs_never_get_publication_credentials():
    data = workflow()
    assert data["permissions"] == {"contents": "read"}
    assert data["on"]["workflow_dispatch"]["inputs"]["publish_blueprints"]["default"] == "false"
    job = data["jobs"]["publish-blueprints"]
    assert job["needs"] == "test"
    assert "workflow_dispatch" in job["if"] and "refs/heads/main" in job["if"]
    assert "inputs.publish_blueprints" in job["if"]
    assert "secrets." not in str(data["jobs"]["test"])


def test_actions_are_pinned_and_main_rechecked():
    data = workflow()
    for job in data["jobs"].values():
        for step in job["steps"]:
            if "uses" in step:
                assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"])
    assert "branches/main" in str(data["jobs"]["publish-blueprints"])
    assert "parent_commit" in (Path(__file__).resolve().parents[1] / "src/szl_model_lab/blueprints.py").read_text()
