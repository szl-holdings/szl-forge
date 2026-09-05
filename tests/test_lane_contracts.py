import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def test_receiptagent_tournament_has_no_invented_winner() -> None:
    data = _load("frontier/receiptagent_tournament.json")
    assert data["schema"] == "szl.receiptagent-tournament/v1"
    assert data["winner"] is None
    assert data["promotion"] == "HOLD"
    assert data["status"] == "NOT_RUN"
    v3 = next(e for e in data["entrants"] if e["hub_id"].endswith("v3"))
    assert v3["eligible"] is False
    assert "sample_level_predictions" in data["required_logs"]


def test_chaski_family_stays_research_only() -> None:
    data = _load("frontier/chaski_research_only.json")
    assert data["promotion"] == "HOLD"
    assert data["status"] == "RESEARCH_ONLY"
    assert data["gate"]["beats_disclosed_baseline"] is False
    ids = {row["hub_id"] for row in data["family"]}
    assert ids == {
        "SZLHOLDINGS/chaski",
        "SZLHOLDINGS/chaski-r2",
        "SZLHOLDINGS/chaski-5050",
        "SZLHOLDINGS/A11OY-MINI",
    }
