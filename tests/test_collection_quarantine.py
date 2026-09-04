import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "publishing" / "collection-quarantine.json"

REQUIRED = {
    "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3",
    "SZLHOLDINGS/SZL-Khipu-1.5B-abstain",
    "SZLHOLDINGS/chaski",
    "SZLHOLDINGS/chaski-5050",
    "SZLHOLDINGS/A11OY-MINI",
}


def test_quarantine_policy_denies_flagship() -> None:
    data = json.loads(POLICY.read_text(encoding="utf-8"))
    assert data["schema"] == "szl.collection-quarantine/v1"
    assert data["hub_write"] == "DENIED_IN_THIS_CHANGE"
    flagship = set(data["flagship_collection_slugs"])
    seen = {item["hub_id"] for item in data["items"]}
    assert REQUIRED <= seen
    for item in data["items"]:
        assert item["promotion"] == "HOLD"
        denied = set(item["denied_collections"])
        assert flagship <= denied
        assert not set(item["allowed_collections"]) & denied
