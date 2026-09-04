import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "assert_collection_membership.py"


def _run(payload: dict) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )


def test_flagship_refuses_quarantined_chaski() -> None:
    proc = _run({"collection": "flagship", "members": ["SZLHOLDINGS/chaski"]})
    assert proc.returncode == 1
    assert "REFUSED" in proc.stderr


def test_research_allows_quarantined_chaski() -> None:
    proc = _run({"collection": "research", "members": ["SZLHOLDINGS/chaski"]})
    assert proc.returncode == 0
    assert "OK" in proc.stdout
