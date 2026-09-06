import json
from pathlib import Path

MANIFEST = Path("frontier/model_bakeoff_manifest.json")


def _load():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_is_non_promotional_by_default():
    data = _load()
    assert data["status"] == "EXPERIMENTAL"
    assert "measured workload improvement" in data["promotion_rule"]


def test_core_frontier_lanes_are_present():
    ids = {lane["id"] for lane in _load()["lanes"]}
    assert {
        "khipu-governed-navigation",
        "receipt-agent",
        "multivector-retrieval",
        "quantized-sovereign-inference",
    } <= ids


def test_experiments_are_revision_license_and_receipt_bound():
    fields = set(_load()["required_experiment_fields"])
    assert {
        "upstream_model_id",
        "upstream_revision",
        "license",
        "license_compatibility_pass",
        "dataset_revision",
        "tokenizer_revision",
        "remote_code_review_sha256",
        "model_card_claims_snapshot_sha256",
        "training_config_sha256",
        "hardware_fingerprint",
        "software_fingerprint",
        "seed",
        "baseline_results_sha256",
        "candidate_results_sha256",
        "governance_pass",
        "reproducibility_pass",
        "receipt_sha256",
    } <= fields


def test_every_lane_has_baseline_candidates_and_metrics():
    for lane in _load()["lanes"]:
        assert lane["baseline"]
        assert lane["candidate_families"]
        assert lane["metrics"]


def test_k2_horizon_is_routed_into_governed_and_quantized_lanes():
    lanes = {lane["id"]: lane for lane in _load()["lanes"]}
    assert "IFM K2 Horizon dense/MoVA" in lanes["khipu-governed-navigation"]["candidate_families"]
    assert "IFM K2 Horizon dense/MoVA" in lanes["receipt-agent"]["candidate_families"]
    assert "K2 Horizon GGUF/FP8" in lanes["quantized-sovereign-inference"]["candidate_families"]
