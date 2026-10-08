"""Hub write locks for szl-forge-owned publishers (HF upgrade plan D3).

Each publisher job listed in LOCKED_WRITERS writes an asset szl-forge keeps
under plan P9 (the szl-forge-lab and szl-model-inference-lab Spaces, the chaski
family, the model source bindings). It holds the canonical lock
``hf-write/<type>/SZLHOLDINGS/<id>``, or the org lock ``hf-write/org/SZLHOLDINGS``
when one job writes several assets. Locks are never keyed by event name or ref
and never cancel an in-flight write. Writers of assets moving to other source
repos (khipu, kernel twins) are not listed; they are retired, not relocked.
These are source assertions; they do not prove a hosted run.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
ORG_LOCK = "hf-write/org/SZLHOLDINGS"

# (workflow file, job id) -> expected lock group; matrix jobs resolve per leg.
LOCKED_WRITERS = {
    ("foundation-runtime.yml", "publish"): "hf-write/space/SZLHOLDINGS/szl-foundation-confirmation",
    ("publish-forge-lab.yml", "deploy"): "hf-write/space/SZLHOLDINGS/szl-forge-lab",
    ("publish-model-inference-lab.yml", "deploy"): "hf-write/space/SZLHOLDINGS/szl-model-inference-lab",
    ("publish-model-inference-lab.yml", "publish-bindings"): ORG_LOCK,
    ("publish-chaski-card.yml", "publish"): "hf-write/model/${{ matrix.repo_id }}",
    ("publish-oac-hub.yml", "publish-oac-v2-model"): "hf-write/model/SZLHOLDINGS/oac-ops-health-v2",
}


def load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(("workflow", "job"), sorted(LOCKED_WRITERS))
def test_writer_job_holds_canonical_hub_lock(workflow: str, job: str) -> None:
    concurrency = load(workflow)["jobs"][job]["concurrency"]
    assert concurrency == {"group": LOCKED_WRITERS[(workflow, job)], "cancel-in-progress": False}
    assert "event_name" not in concurrency["group"] and "github.ref" not in concurrency["group"]


def test_matrix_lock_resolves_to_each_exact_model() -> None:
    data = load("publish-chaski-card.yml")
    legs = data["jobs"]["publish"]["strategy"]["matrix"]["include"]
    groups = {LOCKED_WRITERS[("publish-chaski-card.yml", "publish")].replace("${{ matrix.repo_id }}", leg["repo_id"])
              for leg in legs}
    assert len(groups) == len(legs)
    assert all(re.fullmatch(r"hf-write/model/SZLHOLDINGS/[A-Za-z0-9._-]+", group) for group in groups)


def test_lock_targets_match_the_publish_commands() -> None:
    forge = (WORKFLOWS / "publish-forge-lab.yml").read_text(encoding="utf-8")
    assert "--repo-id SZLHOLDINGS/szl-forge-lab" in forge
    lab = load("publish-model-inference-lab.yml")
    deploy = str(lab["jobs"]["deploy"])
    assert "--repo-id SZLHOLDINGS/szl-model-inference-lab" in deploy
    bindings = json.loads((ROOT / "publishing" / "model-source-bindings.json").read_text(encoding="utf-8"))
    # A multi-asset job must take the org lock rather than one of its targets' locks.
    assert len({artifact["repo_id"] for artifact in bindings["artifacts"]}) > 1
