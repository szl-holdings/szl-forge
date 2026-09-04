import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_flagship_collection_is_empty() -> None:
    data = json.loads((ROOT / "publishing" / "collection-rebuild.json").read_text(encoding="utf-8"))
    assert data["hub_write"] == "DENIED_IN_THIS_CHANGE"
    assert data["collections"]["flagship"] == []
    assert "SZLHOLDINGS/chaski" in data["collections"]["research"]
    assert "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3" in data["collections"]["research"]
