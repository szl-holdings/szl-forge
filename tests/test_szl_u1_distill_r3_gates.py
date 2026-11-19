"""Focused tests for the U1 distill-r3 doctrine gates.

Mirrors the estate test style of tests/test_szl_math_edge_admission_cp.py:
strict contract checks, HOLD code checks, owner-gate honesty checks.
Pure stdlib, deterministic, no network. The gates under test NEVER fire a GPU
job; the best verdict without an explicit owner order is
ARMED_FIRE_PENDING_OWNER_GPU_ORDER.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.join(HERE, "..", "tools", "szl_u1_distill_r3_gates.py")
spec = importlib.util.spec_from_file_location("szl_u1_distill_r3_gates", os.path.abspath(MOD))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def seed_row(seed, admitted, true_violations, lambda_hat):
    return {"seed": seed, "admitted": admitted,
            "true_violations": true_violations, "lambda_hat": lambda_hat}


def clean_seeds(n=3):
    return [seed_row(f"s{i}", 2600 + i, 0, 0.02 + 0.01 * (i % 2)) for i in range(n)]


def test_schema_constant():
    assert m.SCHEMA == "szl.collab/distill-r3-gates-v1"


def test_variance_gate_promote():
    g = m.variance_gate(clean_seeds(), alpha=0.05)
    assert g["verdict"] == "PROMOTE"
    assert g["realized_violations"] == 0
    assert g["production_authorized"] is False


def test_variance_gate_floor_annotation_not_blame():
    g = m.variance_gate([seed_row("s0", 2600, 0, 0.02)])
    assert g["verdict"] == "PROMOTE"  # floor hit is an annotation, not a HOLD
    assert g["lambda_at_floor_annotation"] == ["s0"]


def test_variance_gate_hold_on_violations():
    rows = clean_seeds() + [seed_row("bad", 2600, 1, 0.02)]
    g = m.variance_gate(rows)
    assert g["verdict"] == "HOLD"
    assert g["reason"] == "realized_violations"
    assert g["seeds"] == ["bad"]


def test_variance_gate_hold_on_infeasible_lambda():
    g = m.variance_gate([seed_row("sX", 2600, 0, None)])
    assert g["verdict"] == "HOLD"
    assert g["reason"] == "infeasible_lambda"


def test_variance_gate_hold_malformed_or_empty():
    assert m.variance_gate([])["verdict"] == "HOLD"
    assert m.variance_gate(["nonsense"])["verdict"] == "HOLD"
    assert m.variance_gate([{"seed": "s", "admitted": 1}])["verdict"] == "HOLD"


def test_mixed_policy_record():
    leader = [{"seed": "s0", "admitted": 2616, "true_violations": 0},
              {"seed": "s1", "admitted": 2648, "true_violations": 0}]
    canary = [{"seed": "s0", "admitted": 11102, "true_violations": 0},
              {"seed": "s1", "admitted": 11215, "true_violations": 0}]
    rec = m.mixed_policy(leader, canary)
    assert rec["leader"]["policy"] == "scrc_t_exact_transfer"
    assert rec["canary"]["policy"] == "fixed_band_v1"
    assert rec["leader_zero_violations"] is True
    assert rec["leader"]["total_admitted"] == 2616 + 2648
    assert rec["canary"]["total_admitted"] == 11102 + 11215
    assert rec["production_authorized"] is False
    dishonest = m.mixed_policy(leader, [{"seed": "s", "admitted": 10, "true_violations": 3}])
    commotion = m.mixed_policy([{"seed": "s", "admitted": 10, "true_violations": 2}], canary)
    assert dishonest["canary"]["true_violations"] == 3
    assert commotion["leader_zero_violations"] is False


def test_session_gate_owner_gated():
    a = m.session_gate(True, True, "PROMOTE", owner_gpu_order=False)
    assert a["verdict"] == "ARMED_FIRE_PENDING_OWNER_GPU_ORDER"
    assert a["gpu_release"] == "PENDING_OWNER_ORDER"
    b = m.session_gate(True, True, "PROMOTE", owner_gpu_order=True)
    assert b["verdict"] == "FIRE_AUTHORIZED"  # verdict only; module has no fire path
    assert b["production_authorized"] is False
    assert "no fire path exists" in b["note"]


def test_session_gate_requires_both_cycles_and_promote():
    assert m.session_gate(False, True, "PROMOTE")["verdict"] == "STAGED"
    assert m.session_gate(True, False, "PROMOTE")["verdict"] == "STAGED"
    assert m.session_gate(True, True, "HOLD")["verdict"] == "STAGED"
    assert m.session_gate(True, True, None)["verdict"] == "STAGED"


def test_seal_roundtrip_and_tamper():
    rec = m.seal({"label": "evidence", "rows": 19})
    assert m.sealed_ok(rec) is True
    bad = dict(rec)
    bad["rows"] = 20
    assert m.sealed_ok(bad) is False
    assert m.seal({"x": 1})["production_authorized"] is False
