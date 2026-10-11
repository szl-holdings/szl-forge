"""Preserve the old C2 receipt and distinguish its score from current grading."""

import hashlib
import json
from pathlib import Path

from frontier.harness.heldout_gate import grade_abstain


ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "frontier/khipu-c2-20260915/lane_l2_hidden_c2.json"
SCHEMA = ROOT / "khipu/khipu.schema.json"


def test_historical_c2_pass_does_not_satisfy_current_khipu_schema():
    receipt_bytes = RECEIPT.read_bytes()
    assert hashlib.sha256(receipt_bytes).hexdigest() == (
        "0664e2ed8d512723dcb520bb64dc14603ea7f41862b8f6ec7b8b2c06ff62f03c"
    )
    receipt = json.loads(receipt_bytes)
    rows = receipt["rows"]
    assert receipt["gate"] == "PASS"
    assert receipt["publication_eligible"] is True  # historical raw field only
    assert len(rows) == 12 and all(row["pass"] is True for row in rows)

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert "groundedOnly" in schema["required"]
    assert schema["properties"]["candidates"]["minItems"] == 1
    for row in rows:
        plan = json.loads(row["raw"])
        assert "groundedOnly" not in plan
        assert plan["candidates"] == []
        assert grade_abstain(row["raw"])[0] is False
