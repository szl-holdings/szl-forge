import importlib.util
import json
import sys
from pathlib import Path

import pytest

si = pytest.importorskip("szl_invariants")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("invariant_reward", ROOT / "training" / "rewards" / "invariant_reward.py")
ir = importlib.util.module_from_spec(spec)
sys.modules["invariant_reward"] = ir
spec.loader.exec_module(ir)


def chain(tamper=False):
    return si._build_demo_chain(tamper)


def expected_for(rows):
    return [{"id": r["id"], "model": r["model"]} for r in rows]


def test_clean_bound_export_scores_full():
    rows = chain()
    r = ir.InvariantReward()
    assert r([json.dumps(rows)], expected_rows=[expected_for(rows)]) == [1.0]


def test_fenced_conversation_completion():
    rows = chain()
    comp = [[{"role": "assistant", "content": "Here:\n```json\n" + json.dumps(rows) + "\n```"}]]
    assert ir.InvariantReward()(comp, expected_rows=[expected_for(rows)]) == [1.0]


def test_tampered_chain_scores_zero():
    good, bad = chain(), chain(True)
    out = ir.InvariantReward().score_one(json.dumps(bad), expected_for(good))
    assert out["reward"] == 0.0 and out["reason"] in {"VIOLATED", "TASK_BINDING_FAILED"}
    out2 = ir.InvariantReward().score_one(json.dumps(bad), expected_for(bad))
    assert out2 == {**out2, "reward": 0.0, "reason": "VIOLATED"}


def test_trivial_row_cannot_game_without_task_binding():
    bare = [{"id": 1, "ok": True, "demo": False, "mode": "live", "servedProvider": "x", "model": "m",
             "servedNode": "n", "loopSteps": 1, "receiptId": None, "keyId": None,
             "receiptJson": None, "signature": None}]
    rows = chain()
    assert ir.InvariantReward()([json.dumps(bare)], expected_rows=[expected_for(rows)]) == [0.0]


@pytest.mark.parametrize("text", ["", "[]", "not json", "{\"a\": 1}", "[1, 2]", "[NaN]",
                                  "```json\n[]\n```\n```json\n[]\n```"])
def test_unparseable_scores_zero(text):
    assert ir.InvariantReward().score_one(text, [{"id": 1}])["reward"] == 0.0


def test_no_data_is_not_coerced():
    r = ir.InvariantReward(required=("ledger-failure-shape",))
    rows = chain()
    assert r([json.dumps(rows)], expected_rows=[expected_for(rows)]) == [0.0]


def test_batch_wiring_errors_raise():
    r = ir.InvariantReward()
    with pytest.raises(ValueError):
        r(["[]"], expected_rows=None)
    with pytest.raises(ValueError):
        r(["[]", "[]"], expected_rows=[[{"id": 1}]])
    with pytest.raises(ValueError):
        r([[{"role": "user", "content": "[]"}]], expected_rows=[[{"id": 1}]])


def test_required_must_be_known_ids():
    with pytest.raises(ValueError):
        ir.InvariantReward(required=("made-up",))
    with pytest.raises(ValueError):
        ir.InvariantReward(required=())
