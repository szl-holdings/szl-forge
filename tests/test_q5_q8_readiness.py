# SPDX-License-Identifier: Apache-2.0
"""Q5–Q8 readiness records stay inside the approval and do not launch."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

from training.q5_q8_dry_run import run_lane
from training.q5_q8_lanes import (
    APPROVED_GPU_HOURS,
    COMBINED_GPU_HOURS,
    DENIED_TARGETS,
    LaneError,
    load_evidence,
    load_lane,
    load_lanes,
    readiness_rows,
    validate_lane,
)
from training.transformers5_image import TRAIN_IMAGE, nemotron_trainer_present

ROOT = Path(__file__).resolve().parents[1]
LANES = ROOT / "frontier" / "q5-q8-readiness"
DIGEST = "8eb919654e0da406d41a2e307e7705cce8f6b79ad0be9556d74d4c4640e63235"


def test_records_load_and_stay_closed() -> None:
    lanes = load_lanes()
    assert set(lanes) == {"Q4", "Q5", "Q6", "Q7", "Q8"}
    assert lanes["Q4"]["status"] == "NOT_APPROVED"
    assert lanes["Q5"]["status"] == "READY_NOT_LAUNCHED"
    assert lanes["Q6"]["status"] == "BLOCKED"
    assert lanes["Q7"]["status"] == "BLOCKED"
    assert lanes["Q8"]["status"] == "BLOCKED"
    for lane in lanes.values():
        assert lane["launch_authorized"] is False
        assert lane["hub_write"] is False
        assert lane["image_push"] is False
        assert lane["provider_call"] is False
    assert lanes["Q5"]["gpu_hours"]["estimate"] <= APPROVED_GPU_HOURS["Q5"]
    assert lanes["Q6"]["gpu_hours"]["estimate"] <= APPROVED_GPU_HOURS["Q6"]
    combined = lanes["Q7"]["gpu_hours"]["estimate"] + lanes["Q8"]["gpu_hours"]["estimate"]
    assert combined <= COMBINED_GPU_HOURS
    for lane_id in ("Q5", "Q6", "Q7", "Q8"):
        lane = lanes[lane_id]
        assert lane["train_image"] == TRAIN_IMAGE
        assert lane["runtime"]["rejected_trainer"] == "trl-0.23.1-unsloth"
        assert lane["runtime"]["duplicates_nemotron_trainer"] is False
        assert lane["spend_caps"]["per_run_usd"] == "100"
        assert lane["spend_caps"]["per_day_usd"] == "500"
        assert lane["spend_caps"]["per_month_usd"] == "5000"
        assert lane["spend_caps"]["allowed_providers"] == ["salad", "massed"]
        assert lane["target_repo"].casefold() not in DENIED_TARGETS
        assert lane["target_existence"] == "UNKNOWN"
        assert lane["base"]["license"] == "apache-2.0"
        assert lane["heldout"]["independence"] == "UNKNOWN"


def test_q5_heldout_bytes_match_the_manifest() -> None:
    lane = load_lane("Q5")
    assert lane["heldout"]["eval_only"] is True
    for row in lane["datasets"]:
        digest = hashlib.sha256((ROOT / row["source"]).read_bytes()).hexdigest()
        assert digest == row["sha256"]
        if row["eval_only"]:
            assert row["training_eligible"] is False
    heldout = next(row for row in lane["datasets"] if row["role"] == "heldout")
    assert heldout["source"] == lane["heldout"]["source"]
    assert heldout["rows"] == 72


def test_blocked_lanes_keep_missing_evidence_unknown() -> None:
    q6 = load_lane("Q6")
    assert q6["heldout"]["status"] == "ABSENT"
    assert q6["replaced_base"]["license_name"] == "qwen-research"
    assert q6["replaced_base"]["repo"] != q6["base"]["repo"]
    assert "ABSENT" in q6["blocker"]
    q7 = load_lane("Q7")
    assert q7["datasets"] == []
    assert q7["heldout"]["status"] == "UNKNOWN"
    assert q7["dry_run"]["kind"] == "contrastive"
    assert "verticals-sft-v1" in q7["refused_corpora"][0]["hub_repo"]
    q8 = load_lane("Q8")
    assert q8["heldout"]["status"] == "UNKNOWN"
    assert q8["blocker"].startswith("dataset")
    evidence = load_evidence()
    granite = evidence["by_id"]["granite-3b-code-128k"]
    assert granite["license"] == "apache-2.0"
    assert granite["max_position_embeddings"] == 128000
    incumbent = evidence["by_id"]["qwen25-coder-3b-instruct"]
    assert incumbent["license_name"] == "qwen-research"


def test_cpu_dry_run_per_lane_uses_a_tiny_random_config() -> None:
    before = set(sys.modules)
    results = [run_lane(lane_id) for lane_id in ("Q5", "Q6", "Q7", "Q8")]
    kinds = {row["lane"]: row["kind"] for row in results}
    assert kinds == {"Q5": "causal", "Q6": "causal", "Q7": "contrastive", "Q8": "causal"}
    losses = [tuple(row["losses"]) for row in results]
    assert len(set(losses)) == 4
    for row in results:
        assert row["steps"] == 2
        assert row["device"] == "cpu"
        assert row["loss_finite"] is True
        assert row["parameter_moved"] is True
        assert row["launch_executed"] is False
        assert row["hub_write"] is False
        assert row["weights_are_random_stand_in"] is True
        assert all(math_ok(value) for value in row["losses"])
    added = set(sys.modules) - before
    assert not added.intersection({"torch", "transformers", "trl", "peft", "unsloth"})
    with pytest.raises(LaneError, match="Q4_NOT_APPROVED"):
        run_lane("Q4")


def math_ok(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def test_image_pin_is_not_duplicated_and_nemotron_trainer_is_not_copied() -> None:
    trainer = ROOT / "training" / "nemotron_wave.py"
    assert trainer.is_file()
    assert nemotron_trainer_present(ROOT) is True
    copies = [
        path.name
        for path in (ROOT / "training").glob("nemotron_wave*.py")
        if path.name != "nemotron_wave.py"
    ]
    assert copies == []
    offenders = []
    for path in (ROOT / "training").glob("*.py"):
        if DIGEST in path.read_text(encoding="utf-8") and path.name != "transformers5_image.py":
            offenders.append(path.name)
    assert offenders == []
    source = (ROOT / "training" / "q5_q8_lanes.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert "torch" not in imported
    assert "nemotron_wave" not in source


def test_gate_rejects_overwrite_extra_provider_and_a_raised_budget() -> None:
    evidence = load_evidence()
    original = load_lane("Q5")
    overwritten = copy.deepcopy(original)
    overwritten["target_repo"] = "SZLHOLDINGS/chaski-r4"
    with pytest.raises(LaneError, match="TARGET_OVERWRITE"):
        validate_lane(overwritten, evidence)
    chaski = copy.deepcopy(original)
    chaski["target_repo"] = "SZLHOLDINGS/chaski"
    with pytest.raises(LaneError, match="TARGET_OVERWRITE"):
        validate_lane(chaski, evidence)
    provider = copy.deepcopy(original)
    provider["spend_caps"]["allowed_providers"] = ["salad", "massed", "other"]
    with pytest.raises(LaneError, match="PROVIDER"):
        validate_lane(provider, evidence)
    budget = copy.deepcopy(original)
    budget["gpu_hours"]["estimate"] = 17
    with pytest.raises(LaneError, match="GPU_HOURS_OVER_APPROVAL"):
        validate_lane(budget, evidence)
    launch = copy.deepcopy(original)
    launch["launch_authorized"] = True
    with pytest.raises(LaneError, match="LAUNCH_AUTHORIZED"):
        validate_lane(launch, evidence)
    ready = copy.deepcopy(load_lane("Q6"))
    ready["status"] = "READY_NOT_LAUNCHED"
    ready["blocker"] = None
    with pytest.raises(LaneError, match="READY_WITHOUT_HELDOUT"):
        validate_lane(ready, evidence)
    over = copy.deepcopy(load_lane("Q7"))
    over["gpu_hours"]["estimate"] = 13
    with pytest.raises(LaneError, match="GPU_HOURS_OVER_APPROVAL"):
        validate_lane(over, evidence)


def test_one_lane_load_still_enforces_the_combined_budget(tmp_path: Path) -> None:
    shutil.copytree(LANES, tmp_path, dirs_exist_ok=True)
    for name, estimate in (("q7-embedder.manifest.json", 8), ("q8-brain-navigator.manifest.json", 8)):
        path = tmp_path / name
        document = json.loads(path.read_text(encoding="utf-8"))
        document["gpu_hours"]["estimate"] = estimate
        path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(LaneError, match="GPU_HOURS_COMBINED"):
        load_lane("Q7", tmp_path)
    with pytest.raises(LaneError, match="GPU_HOURS_COMBINED"):
        run_lane("Q8", tmp_path)


def test_recipes_match_the_manifests() -> None:
    yaml = pytest.importorskip("yaml")
    lanes = load_lanes()
    names = {
        "Q5": "q5-receiptagent.recipe.yaml",
        "Q6": "q6-a11oy-code.recipe.yaml",
        "Q7": "q7-embedder.recipe.yaml",
        "Q8": "q8-brain-navigator.recipe.yaml",
    }
    for lane_id, name in names.items():
        recipe = yaml.safe_load((LANES / name).read_text(encoding="utf-8"))
        lane = lanes[lane_id]
        assert recipe["lane"] == lane_id
        assert recipe["status"] == lane["status"]
        assert recipe["launch_authorized"] is False
        assert recipe["base"]["repo"] == lane["base"]["repo"]
        assert recipe["base"]["revision"] == lane["base"]["revision"]
        assert recipe["publish"]["target_repo"] == lane["target_repo"]
        assert recipe["gpu_hours"]["estimate"] == lane["gpu_hours"]["estimate"]
        assert recipe["spend_caps"]["allowed_providers"] == ["salad", "massed"]


def test_readiness_rows_do_not_authorize_launch() -> None:
    rows = readiness_rows()
    assert [row["id"] for row in rows] == ["Q5", "Q6", "Q7", "Q8"]
    assert rows[0]["status"] == "READY_NOT_LAUNCHED"
    assert rows[0]["base"].startswith("Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17")
    assert all(row["launch_authorized"] is False for row in rows)
    assert all(row["blocker"] for row in rows)
