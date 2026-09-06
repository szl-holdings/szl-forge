import json
from pathlib import Path

INTAKE = Path("frontier/K2_HORIZON_INTAKE_2026-09-06.json")


def _load():
    return json.loads(INTAKE.read_text(encoding="utf-8"))


def test_intake_is_fail_closed_and_non_promotional():
    data = _load()
    assert data["status"] == "REPORTED_UPSTREAM"
    assert data["promotion_authority"] == "NONE"
    assert data["release"]["queue_state"] == "QUEUED_UNQUALIFIED"


def test_exact_k2_evaluation_order_is_bound():
    release = _load()["release"]
    assert release["priority"] == "P0"
    assert release["evaluation_order"] == [
        "IFM/K2-Horizon-7B",
        "IFM/K2-Horizon-MoVA-36B-A4B",
        "IFM/K2-Horizon-375B-A23B",
    ]


def test_every_model_and_dataset_is_unpinned_until_qualification():
    release = _load()["release"]
    for asset in [*release["models"], *release["datasets"]]:
        assert asset["upstream_revision"] == "UNPINNED"


def test_family_openness_is_variant_specific():
    models = {item["model_id"]: item for item in _load()["release"]["models"]}
    assert (
        models["IFM/K2-Horizon-7B"]["training_assets_state"]
        == "REPORTED_PUBLIC_NOT_INDEPENDENTLY_ENUMERATED"
    )
    for model_id in [
        "IFM/K2-Horizon-MoVA-36B-A4B",
        "IFM/K2-Horizon-375B-A23B",
    ]:
        assert models[model_id]["training_assets_state"] == "ANNOUNCED_NOT_OBSERVED"
        assert (
            models[model_id]["intermediate_checkpoints_state"]
            == "ANNOUNCED_NOT_OBSERVED"
        )


def test_remote_code_and_authority_boundaries_are_explicit():
    release = _load()["release"]
    assert all(model["requires_remote_code_review"] for model in release["models"])
    gates = " ".join(release["qualification_gates"])
    assert "trust_remote_code" in gates
    assert "Nemo witnesses" in gates
    assert "A11oy remains the sole consequential-action admission layer" in gates


def test_only_primary_hugging_face_sources_are_bound():
    sources = _load()["release"]["primary_sources"]
    assert sources
    assert all(
        item["url"].startswith("https://huggingface.co/") for item in sources
    )


def test_no_asset_is_claimed_as_szl_measured():
    release = _load()["release"]
    serialized = json.dumps(release, sort_keys=True)
    assert '"MEASURED"' not in serialized
