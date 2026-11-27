import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cftm", ROOT / "tools" / "check_flagship_training_map.py")
cftm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cftm)


def test_committed_map_is_clean():
    assert cftm.check() == []


def _mutated(tmp_path, fn):
    data = json.loads(cftm.DEFAULT_MAP.read_text(encoding="utf-8"))
    fn(data)
    p = tmp_path / "map.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return cftm.check(p)


def test_authority_cannot_be_granted(tmp_path):
    assert _mutated(tmp_path, lambda d: d["flagships"][0].update(training_authority="GRANTED"))


def test_production_claim_rejected(tmp_path):
    assert _mutated(tmp_path, lambda d: d["flagships"][0].update(status_note="production ready"))


def test_registration_must_match_portfolio(tmp_path):
    def flip(d):
        d["flagships"][0]["forge_portfolio_registered"] = not d["flagships"][0]["forge_portfolio_registered"]
    assert _mutated(tmp_path, flip)


def test_unknown_state_rejected(tmp_path):
    assert _mutated(tmp_path, lambda d: d["flagships"][0]["signals"][0].update(state="PASS"))


def test_missing_forge_path_rejected(tmp_path):
    def bad(d):
        d["flagships"][0]["signals"].append({"kind": "eval", "repo": "szl-forge", "path": "no/such/file.py", "state": "EXISTING"})
    assert _mutated(tmp_path, bad)
