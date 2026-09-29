"""Source assertions complement hosted CI; they do not establish hosted success."""
from pathlib import Path
import re
import yaml

from szl_model_lab.catalog import HUB_STATE_NOT_PUBLISHED, catalog

REPO = Path(__file__).resolve().parents[2]


def workflow():
    path = REPO / ".github/workflows/model-lab-contract.yml"
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)


def test_workflow_prs_never_get_publication_credentials():
    data = workflow()
    assert data["permissions"] == {"contents": "read"}
    assert "secrets." not in str(data["jobs"])
    assert "HF_TOKEN" not in str(data["jobs"]) and "acquire_hf_publisher_token" not in str(data["jobs"])


def test_workflow_holds_no_hub_write_path():
    data = workflow()
    assert set(data["jobs"]) == {"test"}
    assert "inputs" not in (data["on"]["workflow_dispatch"] or {})
    assert "--publish" not in str(data["jobs"])
    assert "id-token" not in str(data)


def test_actions_are_pinned():
    data = workflow()
    for job in data["jobs"].values():
        for step in job["steps"]:
            if "uses" in step:
                assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"])
    assert "parent_commit" in (Path(__file__).resolve().parents[1] / "src/szl_model_lab/blueprints.py").read_text()


def test_unpublished_catalog_ids_have_no_committed_writer():
    rows = catalog()
    assert rows and all(row["hub_state"] == HUB_STATE_NOT_PUBLISHED for row in rows)
    proposed = {row["proposed_hf_id"] for row in rows if row["proposed_hf_id"]}
    assert {"SZLHOLDINGS/A11OY-Router", "SZLHOLDINGS/A11OY-Invariant"} <= proposed
    for path in sorted((REPO / ".github/workflows").glob("*.y*ml")):
        text = path.read_text(encoding="utf-8")
        for repo_id in proposed:
            assert not re.search(re.escape(repo_id) + r"(?![A-Za-z0-9_.-])", text), (path.name, repo_id)
