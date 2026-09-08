import json
import re
from pathlib import Path

WAVE = Path("frontier/PINNED_MODEL_WAVE_2026-09-07.json")
BAKEOFF = Path("frontier/model_bakeoff_manifest.json")

EXPECTED_REVISIONS = {
    "zai-org/GLM-5.3-Flash": "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
    "zai-org/GLM-5.3": "aca966e4e02791568aa6a4ced368624b3d897f42",
    "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp": (
        "6821d6ad3681a4b137b066b76094fa82ebd0a380"
    ),
    "nvidia/Qwen3.8-Flash-Next-NVFP4": (
        "fc694b54fb0174e0913e6adf86691ef85a4ead47"
    ),
    "Qwen/Qwen3.8-Flash-Next": "de4b8e4d43b917e7706784d8bb445c9af86a3540",
}


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _candidates():
    return {
        item["upstream_model_id"]: item
        for item in _load(WAVE)["candidates"]
    }


def test_wave_is_registry_bound_and_non_promotional():
    data = _load(WAVE)
    source = data["source_registry"]
    assert data["schema_version"] == "szl.forge.frontier-model-wave.v1"
    assert data["status"] == "PINNED_UNQUALIFIED"
    assert data["promotion_authority"] == "NONE"
    assert data["production_disposition"] == "HOLD"
    assert source["repository"] == "szl-holdings/szl-frontier"
    assert source["commit"] == "dcc128140f3873d0ca396b9a83cf0eb5a0102b87"
    assert re.fullmatch(r"[0-9a-f]{40}", source["commit"])
    assert source["default_effect"] == "hold"
    assert source["production_promotion_count"] == 0


def test_exact_source_revisions_are_immutable_and_complete():
    candidates = _candidates()
    assert set(candidates) == set(EXPECTED_REVISIONS)
    for model_id, revision in EXPECTED_REVISIONS.items():
        candidate = candidates[model_id]
        assert candidate["upstream_revision"] == revision
        assert re.fullmatch(r"[0-9a-f]{40}", revision)
        assert candidate["source_verification_pass"] is True


def test_qualification_order_covers_each_candidate_once():
    data = _load(WAVE)
    order = data["qualification_order"]
    assert order == [
        "zai-org/GLM-5.3-Flash",
        "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
        "nvidia/Qwen3.8-Flash-Next-NVFP4",
        "zai-org/GLM-5.3",
        "Qwen/Qwen3.8-Flash-Next",
    ]
    assert set(order) == set(_candidates())


def test_every_candidate_stays_on_hold_without_results():
    for candidate in _candidates().values():
        assert candidate["production_disposition"] == "HOLD"
        assert candidate["results"] is None
        assert candidate["required_metrics"]
        for gate in {
            "license_compatibility_pass",
            "runtime_compatibility_pass",
            "safety_pass",
            "cost_pass",
            "fallback_pass",
        }:
            assert candidate[gate] is None


def test_review_required_licenses_cannot_enter_primary_execution():
    for candidate in _candidates().values():
        if candidate["declared_license"] == "other":
            assert candidate["evaluation_state"] == "LICENSE_REVIEW_REQUIRED"
            assert candidate["role"] in {
                "REFERENCE_ONLY",
                "QUANTIZED_BENCHMARK",
                "BASE_REFERENCE_ONLY",
            }


def test_nvfp4_derivative_is_bound_to_exact_base_model():
    candidate = _candidates()["nvidia/Qwen3.8-Flash-Next-NVFP4"]
    assert candidate["base_model_id"] == "Qwen/Qwen3.8-Flash-Next"
    assert (
        candidate["base_model_revision"]
        == EXPECTED_REVISIONS["Qwen/Qwen3.8-Flash-Next"]
    )
    assert candidate["base_model_declared_license"] == "other"


def test_target_lanes_exist_or_are_explicitly_reference_only():
    lane_ids = {lane["id"] for lane in _load(BAKEOFF)["lanes"]}
    lane_ids.add("frontier-reference-only")
    for candidate in _candidates().values():
        assert set(candidate["target_lanes"]) <= lane_ids


def test_execution_defaults_fail_closed_and_keep_a11oy_authority():
    policy = _load(WAVE)["execution_policy"]
    assert policy["automatic_weight_download"] is False
    assert policy["automatic_remote_code_execution"] is False
    assert policy["network_access_default"] == "DENY"
    assert policy["model_output_authority"] == "PROPOSAL_ONLY"
    assert policy["consequential_action_admission_layer"] == "A11oy"
    assert policy["baseline_fallback"] == "REQUIRED"


def test_shared_gates_cover_license_code_eval_fallback_and_receipts():
    gates = " ".join(_load(WAVE)["shared_qualification_gates"])
    for requirement in {
        "license compatibility",
        "trust_remote_code",
        "baseline fallback",
        "upstream benchmark",
        "A11oy",
        "reproducible receipt",
    }:
        assert requirement in gates
