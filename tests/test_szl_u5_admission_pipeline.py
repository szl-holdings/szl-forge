"""Focused tests for the U5 admission-gated proving pipeline.

Mirrors the estate test style of tests/test_szl_math_edge_admission_cp.py:
strict contract checks, HOLD code checks, fail-closed ordering checks.
All values are deterministic; no network, no randomness, no secrets.
The pipeline under test composes the REAL merged estate modules
(szl_math_edge_admission_cp, szl_math_kotte_feasibility); nothing is faked.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.join(HERE, "..", "tools", "szl_u5_admission_pipeline.py")
spec = importlib.util.spec_from_file_location("szl_u5_admission_pipeline", os.path.abspath(MOD))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

ALPHA = 0.05
GRID = [0.02, 0.05, 0.1, 0.5]


def cal(uppers, flags, grid=GRID):
    return {"certified_uppers": uppers, "true_flags": flags, "grid": grid}


def clean_cal(n=40):
    return cal([0.02] * n, [False] * n)


def lean_ok(hygiene="PASS_CLEAN"):
    return {"lean_verify": "PASS", "hygiene": hygiene}


def spec_ok(problem="P_TEST_1", calibration=None, lean=None):
    return {"problem": problem, "alpha": ALPHA,
            "calibration": calibration or clean_cal(), "lean": lean or lean_ok()}


def verdicts_get():
    return m.LEAN_VERIFY_VERDICTS, m.HYGIENE_VERDICTS


def test_schema_constant():
    assert m.SCHEMA == "szl.edge-math-evidence/u5-pipeline-v1"


def test_certified_spend_ready():
    r = m.admit(spec_ok())
    assert r["verdict"] == "CERTIFIED_SPEND_READY"
    assert r["tier"] == "certified_spend"
    assert r["production_authorized"] is False
    assert r["gpu_release"] == "PENDING_OWNER_ORDER"
    assert r["lambda_hat"] == 0.02
    assert r["mu_hat"] == 0.0
    assert m.receipt_ok(r) is True
    gate_names = [g["gate"] for g in r["gate_trace"]]
    assert gate_names == ["cp_admission", "kotte_cb", "lean_verify", "lean_hygiene"]


def test_sketch_tier_is_never_certified():
    r = m.admit(spec_ok(lean=lean_ok("FAIL_SKETCH")))
    assert r["verdict"] == "SKETCH_TIER_READY_NOT_CERTIFIED"
    assert r["tier"] == "sketch"
    assert "certified" not in r["verdict"].split("_")[0]
    assert r["production_authorized"] is False
    assert m.receipt_ok(r) is True


def test_hold_no_feasible_lambda():
    r = m.admit(spec_ok(calibration=cal([0.02] * 20, [True] * 20)))
    assert r["verdict"] == "HOLD_ADMISSION"
    assert r["gate_trace"][-1]["outcome"] == "HOLD_NO_FEASIBLE_LAMBDA"
    assert len(r["gate_trace"]) == 1  # fail-closed: nothing downstream ran
    assert m.receipt_ok(r) is True


def test_hold_no_admitted_cases():
    r = m.admit(spec_ok(calibration=cal([0.9] * 20, [False] * 20, grid=[0.5])))
    assert r["verdict"] == "HOLD_KOTTE"
    assert r["gate_trace"][-1]["outcome"] == "HOLD_NO_ADMITTED_CASES"
    assert len(r["gate_trace"]) == 2  # lean gates never ran
    assert m.receipt_ok(r) is True


def test_kotte_refuse_stops_pipeline():
    # 3 admitted true violations among 4 admitted -> mu_hat 0.75 >> alpha,
    # while the CP aggregate formula stays feasible ((3+1)/101 <= 0.05).
    uppers = [0.02, 0.02, 0.02, 0.02] + [0.9] * 96
    flags = [True, True, True, False] + [False] * 96
    r = m.admit(spec_ok(calibration=cal(uppers, flags)))
    assert r["verdict"] == "REFUSED_KOTTE"
    assert abs(r["mu_hat"] - 0.75) < 1e-12
    gate = r["gate_trace"][1]
    assert gate["outcome"] == "REFUSE"
    assert abs(gate["abstention_floor"] - (0.75 - 0.05) / 0.95) < 1e-9
    assert len(r["gate_trace"]) == 2  # CAREFUL: lean gates never run after REFUSE
    assert r["production_authorized"] is False
    assert m.receipt_ok(r) is True


def test_stop_lean_verify_fail():
    r = m.admit(spec_ok(lean={"lean_verify": "FAIL", "hygiene": "PASS_CLEAN"}))
    assert r["verdict"] == "STOP_LEAN_VERIFY_FAIL"
    assert len(r["gate_trace"]) == 3  # hygiene gate never evaluated
    assert m.receipt_ok(r) is True


def test_stop_hygiene():
    r = m.admit(spec_ok(lean=lean_ok("FAIL_SKETCH_AND_COMPILE")))
    assert r["verdict"] == "STOP_HYGIENE"
    assert m.receipt_ok(r) is True


def test_contract_holds():
    for bad in (
        {},
        {"problem": "x"},
        {"problem": "x", "alpha": ALPHA, "calibration": clean_cal()},  # no lean
        {"problem": "x", "alpha": 0.05, "calibration": {}, "lean": lean_ok()},
        {"problem": "x", "alpha": 0.06, "calibration": cal([0.01], [False]), "lean": lean_ok()},  # int alpha
        {"problem": "x", "alpha": ALPHA, "calibration": cal([0.02], [1]), "lean": lean_ok()},  # non-bool flag
        {"problem": "", "alpha": ALPHA, "calibration": clean_cal(), "lean": lean_ok()},
    ):
        r = m.admit(bad)
        assert r["verdict"].startswith("HOLD_") or r["verdict"].startswith("STOP_") or "HOLD" in r["verdict"]
        assert r["production_authorized"] is False
        assert m.receipt_ok(r) is True


def test_receipt_tamper_detected():
    r = m.admit(spec_ok())
    bad = dict(r)
    bad["verdict"] = "CERTIFIED_SPEND_READY_TAMPERED"
    assert m.receipt_ok(bad) is False
    bad2 = dict(r)
    del bad2["sha256"]
    assert m.receipt_ok(bad2) is False


def test_module_pinnings_shape():
    rows = m.module_pinnings()
    keys = {row["module"] for row in rows}
    assert keys == {"cp_module", "kotte_module"}
    cp_row = next(row for row in rows if row["module"] == "cp_module")
    assert cp_row["cycle_receipt_pin"] is not None and len(cp_row["sha256"]) == 64
    ko_row = next(row for row in rows if row["module"] == "kotte_module")
    assert ko_row["cycle_receipt_pin"] is None
    r = m.admit(spec_ok())
    assert any(row is not None for row in r["module_pinnings"])


def test_ledger_chain_roundtrip():
    ledger = []
    e1 = m.ledger_append(ledger, m.admit(spec_ok("P_LEDGER_1")), None)
    assert e1["prev"] == "0" * 64
    e2 = m.ledger_append(ledger, m.admit(spec_ok("P_LEDGER_2", lean=lean_ok("FAIL_SKETCH"))), e1)
    e3 = m.ledger_append(ledger, m.admit(spec_ok("P_LEDGER_3")), e2)
    assert m.ledger_ok(ledger) is True
    # tampering any inner receipt breaks the chain
    ledger[1]["receipt"]["verdict"] = "TAMPERED"
    assert m.ledger_ok(ledger) is False


def test_ledger_rejects_unverifiable_receipt():
    ledger = []
    try:
        m.ledger_append(ledger, {"verdict": "NOT_A_RECEIPT"})
        raised = False
    except ValueError:
        raised = True
    assert raised is True
    assert ledger == []


def test_fail_closed_everywhere():
    # every reachable outcome keeps the GPU owner-gated
    outs = [
        m.admit(spec_ok()),
        m.admit(spec_ok(lean=lean_ok("FAIL_SKETCH"))),
        m.admit(spec_ok(calibration=cal([0.02] * 20, [True] * 20))),
        m.admit(spec_ok(lean={"lean_verify": "FAIL", "hygiene": "FAIL"})),
    ]
    for r in outs:
        assert r["production_authorized"] is False
        assert r["gpu_release"] == "PENDING_OWNER_ORDER"
        assert r["fail_closed"] is True
        assert m.receipt_ok(r) is True
